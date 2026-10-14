"""Batch FaceAge inference: biological age for every image in a directory.

Reads a folder of preprocessed face crops, runs the FaceAge model on each, optionally matches the
patient's chronological age so the estimation error can be computed, and writes one row per image
to Excel.

Must run in the separate Python 3.7 environment (TensorFlow 2.6.2, Keras 2.6.0,
protobuf==3.20.3). The released FaceAge model uses Keras Lambda layers serialised under Python
3.6, which Python 3.8 and later cannot deserialise — see README section 2.

Model. This script loads `faceage_model.h5` from release v1 of the original repository
(https://github.com/AIM-Harvard/FaceAge/releases/tag/v1) — an Inception-ResNet v1 network that
takes a 160 x 160 x 3 RGB input and outputs a single continuous age estimate. It is applied as
distributed, with no retraining or fine-tuning. The SHA-256 of the file used for this study is
recorded in the README section 5.

Preprocessing at inference, matching the original implementation: each cropped face is resized
to 160 x 160 and standardised by its own mean and standard deviation across the three RGB
channels. Paths are set in the configuration block below."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import cv2
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 禁用 eager execution（FaceAge 要求）
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("=" * 80)
print("FaceAge 生物学年龄预测工具")
print("=" * 80)
print(f"Python: {sys.version.split()[0]}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")
print()


# ============================================================================
# 配置参数
# ============================================================================
IMAGE_DIR = os.environ.get("FACEAGE_CROPS_DIR", "data/face_crops_faceage")  # 图像目录
OUTPUT_FILE = os.environ.get("FACEAGE_PREDICTIONS_FILE", "data/faceage_predictions.xlsx")  # 输出文件
MATCH_PATIENT_INFO = True  # 是否尝试匹配患者信息


# ============================================================================
# 1. 加载 FaceAge 模型
# ============================================================================
print("正在加载 FaceAge 模型...")
model = keras.models.load_model(os.environ.get('FACEAGE_MODEL_H5', 'FaceAge/models/faceage_model.h5'))
print("✓ 模型加载成功")
print()


# ============================================================================
# 2. 加载患者信息（可选）
# ============================================================================
patient_info = {}

if MATCH_PATIENT_INFO:
    print("正在加载患者信息...")
    try:
        # 尝试从 merged_data.h5 加载
        import h5py
        with h5py.File(os.environ.get('FACEAGE_MERGED_H5', 'data/merged_data.h5'), 'r') as f:
            # 读取图像文件名
            raw_filenames = f['metadata/image_filenames'][:]
            image_filenames = [
                fn.decode('utf-8') if isinstance(fn, bytes) else fn
                for fn in raw_filenames
            ]

            # 读取年龄数据
            numeric_data = f['numeric_data'][:]
            numeric_columns = [
                col.decode('utf-8') if isinstance(col, bytes) else col
                for col in f['numeric_columns'][:]
            ]

            # 找到年龄列
            age_idx = None
            for i, col in enumerate(numeric_columns):
                if '年龄' in col:
                    age_idx = i
                    break

            if age_idx is None:
                # 启发式查找：年龄通常在 10-110 之间
                for i, col in enumerate(numeric_columns):
                    col_data = numeric_data[:, i]
                    if 10 <= col_data.min() and col_data.max() <= 110:
                        age_idx = i
                        break

            if age_idx is not None:
                ages = numeric_data[:, age_idx]

                # 构建文件名 -> 年龄的映射
                for filename, age in zip(image_filenames, ages):
                    patient_info[filename] = {
                        'actual_age': float(age)
                    }

                print(f"✓ 成功加载 {len(patient_info)} 个患者的信息")
            else:
                print("⚠ 未找到年龄列，将只输出预测年龄")
                MATCH_PATIENT_INFO = False

    except Exception as e:
        print(f"⚠ 无法加载患者信息: {str(e)}")
        print("  将只输出预测年龄")
        MATCH_PATIENT_INFO = False

print()


# ============================================================================
# 3. 获取所有图像文件
# ============================================================================
print(f"正在扫描图像目录: {IMAGE_DIR}")

image_dir = Path(IMAGE_DIR)
if not image_dir.exists():
    print(f"✗ 错误：目录不存在: {IMAGE_DIR}")
    sys.exit(1)

# 支持的图像格式
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}

# 获取所有图像文件（去重）
all_files = set()
for ext in image_extensions:
    all_files.update(image_dir.glob(f'*{ext}'))

image_files = sorted(list(all_files))

if not image_files:
    print(f"✗ 错误：在 {IMAGE_DIR} 中未找到图像文件")
    sys.exit(1)

print(f"✓ 找到 {len(image_files)} 个图像文件")
print()


# ============================================================================
# 4. 批量预测
# ============================================================================
print("开始批量预测...")
print("=" * 80)

results = []
failed_files = []

for i, image_path in enumerate(tqdm(image_files, desc="预测进度")):
    filename = image_path.name

    try:
        # 读取图像（支持中文路径）
        img = cv2.imdecode(
            np.fromfile(str(image_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if img is None:
            failed_files.append((filename, "无法读取图像"))
            continue

        # 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理（FaceAge 标准流程）
        from PIL import Image
        img_pil = Image.fromarray(img_rgb).convert('RGB')
        img_resized = np.asarray(img_pil.resize((160, 160)))

        # 归一化
        mean, std = img_resized.mean(), img_resized.std()
        if std == 0:
            std = 1  # 避免除零
        img_norm = (img_resized - mean) / std
        img_input = img_norm.reshape(1, 160, 160, 3)

        # 预测
        age_pred = float(np.squeeze(model.predict(img_input, verbose=0)))

        # 构建结果记录
        result = {
            '文件名': filename,
            '预测年龄': round(age_pred, 2)
        }

        # 尝试匹配患者信息
        if MATCH_PATIENT_INFO and filename in patient_info:
            actual_age = patient_info[filename]['actual_age']
            result['实际年龄'] = round(actual_age, 1)
            result['年龄差异'] = round(age_pred - actual_age, 2)
            result['绝对误差'] = round(abs(age_pred - actual_age), 2)
        else:
            result['实际年龄'] = None
            result['年龄差异'] = None
            result['绝对误差'] = None

        results.append(result)

        # 定期清理内存
        if (i + 1) % 100 == 0:
            gc.collect()

    except Exception as e:
        failed_files.append((filename, str(e)))
        continue

print("=" * 80)
print(f"预测完成！")
print(f"  成功: {len(results)} 个")
print(f"  失败: {len(failed_files)} 个")
print()


# ============================================================================
# 5. 计算统计指标
# ============================================================================
if MATCH_PATIENT_INFO:
    # 过滤出有实际年龄的记录
    matched_results = [r for r in results if r['实际年龄'] is not None]

    if matched_results:
        actual_ages = np.array([r['实际年龄'] for r in matched_results])
        predicted_ages = np.array([r['预测年龄'] for r in matched_results])

        from sklearn.metrics import mean_absolute_error, r2_score
        from scipy import stats

        mae = mean_absolute_error(actual_ages, predicted_ages)
        r2 = r2_score(actual_ages, predicted_ages)
        pearson_r, _ = stats.pearsonr(actual_ages, predicted_ages)

        age_diff = predicted_ages - actual_ages

        print("=" * 80)
        print("预测质量评估")
        print("=" * 80)
        print(f"匹配样本数: {len(matched_results)} / {len(results)}")
        print(f"MAE (平均绝对误差): {mae:.2f} 岁")
        print(f"R² (决定系数): {r2:.4f}")
        print(f"Pearson 相关系数: {pearson_r:.4f}")
        print(f"年龄差异: {age_diff.mean():.2f} ± {age_diff.std():.2f} 岁")
        print()

        # 误差分布
        within_10 = np.sum(np.abs(age_diff) <= 10)
        print(f"误差分布:")
        print(f"  ≤10 岁: {within_10} ({within_10/len(matched_results)*100:.1f}%)")
        print(f"  >10 岁: {len(matched_results) - within_10} ({(len(matched_results) - within_10)/len(matched_results)*100:.1f}%)")
        print("=" * 80)
        print()


# ============================================================================
# 6. 保存结果
# ============================================================================
print("正在保存结果...")

# 创建 DataFrame
df = pd.DataFrame(results)

# 添加预测时间
df['预测时间'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 调整列顺序
if MATCH_PATIENT_INFO:
    columns = ['文件名', '实际年龄', '预测年龄', '年龄差异', '绝对误差', '预测时间']
else:
    columns = ['文件名', '预测年龄', '预测时间']

df = df[columns]

# 按绝对误差排序（如果有）
if '绝对误差' in df.columns:
    df = df.sort_values('绝对误差', ascending=False)

# 创建输出目录
output_path = Path(OUTPUT_FILE)
output_path.parent.mkdir(parents=True, exist_ok=True)

# 保存为 Excel
df.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')

print(f"✓ 结果已保存: {OUTPUT_FILE}")
print(f"  共 {len(df)} 条记录")
print()

# 如果有失败的文件，保存失败日志
if failed_files:
    failed_log = output_path.parent / 'faceage_prediction_failures.txt'
    with open(failed_log, 'w', encoding='utf-8') as f:
        f.write(f"预测失败的文件 ({len(failed_files)} 个)\n")
        f.write("=" * 80 + "\n\n")
        for filename, error in failed_files:
            f.write(f"{filename}\n  错误: {error}\n\n")

    print(f"⚠ 失败日志已保存: {failed_log}")
    print()

print("=" * 80)
print("完成！")
print("=" * 80)
