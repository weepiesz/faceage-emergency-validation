"""Face preprocessing for FaceAge: MTCNN detection, bounding-box crop, 160x160 resize.

This reproduces the preprocessing of the original FaceAge model, so that images from this
cohort are comparable with the ones the model was developed and validated on:

  1. locate the face with MTCNN (Zhang et al.), taking the bounding box it returns
  2. crop that box and resize to 160 x 160 px, the FaceAge model's input size

No geometric realignment of the face is applied. The original FaceAge pipeline uses the
detector's bounding box alone: MTCNN does localise five facial landmarks internally while it
refines the box, but the reference implementation does not use them to transform the image.

The default backend is the `mtcnn` package (v0.1.1), the implementation shipped in the
FaceAge environment and the one used for this study. facenet-pytorch is available as an
opt-in alternative via FACEAGE_MTCNN_BACKEND; note that its crop output *is* landmark-aligned,
so it does not reproduce the original pipeline.

This is the first step of the pipeline and it runs on the raw photographs, which are not
distributed — point it at your own image directory (paths are set in main() below)."""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# 人脸检测后端。默认 mtcnn==0.1.1，即 FaceAge 环境实际使用的实现（按边界框裁剪，不做几何对齐）；
# 设为 facenet_pytorch 可启用 facenet-pytorch 的 MTCNN（其裁剪输出按关键点对齐，与论文流程不同）。
BACKEND = os.environ.get('FACEAGE_MTCNN_BACKEND', 'mtcnn').strip().lower()
if BACKEND not in ('mtcnn', 'facenet_pytorch'):
    raise SystemExit(
        f"FACEAGE_MTCNN_BACKEND 只能是 'mtcnn' 或 'facenet_pytorch'，收到 {BACKEND!r}")

USE_FACENET = BACKEND == 'facenet_pytorch'

if USE_FACENET:
    try:
        from facenet_pytorch import MTCNN
        print("✓ 使用 facenet-pytorch MTCNN（裁剪输出含关键点对齐，仅作可选替代）")
    except ImportError:
        raise SystemExit(
            "✗ FACEAGE_MTCNN_BACKEND=facenet_pytorch，但未安装 facenet-pytorch\n"
            "  pip install facenet-pytorch")
else:
    try:
        from mtcnn import MTCNN as MTCNN_Standard
        print("✓ 使用 mtcnn 库 (mtcnn==0.1.1)：与 FaceAge 环境一致，按边界框裁剪，不做几何对齐")
    except ImportError:
        raise SystemExit("✗ 未安装 mtcnn\n  pip install mtcnn==0.1.1")


class FaceAgePreprocessor:
    """
    FaceAge风格的面部预处理器

    与FaceAge论文保持一致的预处理流程：
    - 人脸检测：MTCNN，取检测框
    - 裁剪并缩放到 160x160（FaceAge 模型输入尺寸）
    - 不做基于关键点的几何对齐（FaceAge 原始流程同样不做）
    """

    def __init__(self, image_size=160, margin=0, device='cuda'):
        """
        初始化预处理器

        Args:
            image_size (int): 输出图像尺寸（FaceAge使用160）
            margin (int): 人脸边界扩展（0表示紧贴边界）
            device (str): 计算设备 'cuda' 或 'cpu'（仅 facenet-pytorch 后端使用）
        """
        self.image_size = image_size
        self.margin = margin

        if USE_FACENET:
            # facenet-pytorch MTCNN
            import torch
            device = device if torch.cuda.is_available() else 'cpu'

            self.detector = MTCNN(
                image_size=image_size,
                margin=margin,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],  # MTCNN三阶段阈值
                factor=0.709,
                post_process=True,  # 自动归一化到[-1, 1]
                device=device,
                keep_all=False,  # 只保留置信度最高的人脸
                select_largest=True  # 选择最大的人脸
            )
            print(f"✓ MTCNN 初始化完成 (设备: {device})")
        else:
            # 标准MTCNN库
            self.detector = MTCNN_Standard()
            print("✓ 标准 MTCNN 初始化完成")

    def detect_and_crop_face(self, image_path, min_confidence=0.9):
        """
        检测并裁剪单张图像的人脸

        Args:
            image_path (str): 输入图像路径
            min_confidence (float): 最小置信度阈值

        Returns:
            dict 或 None:
                - 'face': 裁剪并缩放到 image_size 的人脸图像 (RGB, [0,255])
                - 'confidence': 检测置信度
                - 'bbox': 人脸边界框 [x, y, w, h]（facenet-pytorch 后端为 None）
                - 'landmarks': 面部关键点（facenet-pytorch 后端为 None；仅记录，未用于图像变换）
        """
        try:
            # 读取图像（使用 np.fromfile 支持中文路径）
            if isinstance(image_path, str):
                img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    return None
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                # 已经是numpy数组
                img = image_path
                if img.shape[2] == 4:  # RGBA
                    img = img[:, :, :3]

            if USE_FACENET:
                # facenet-pytorch 方式：其裁剪输出本身经过关键点对齐，故与 FaceAge 原始流程不同
                face_tensor, prob = self.detector(img, return_prob=True)

                if face_tensor is None or prob is None:
                    return None

                if prob < min_confidence:
                    return None

                # 转换为numpy数组 [0, 255]
                if face_tensor.is_cuda:
                    face_tensor = face_tensor.cpu()

                # facenet-pytorch 输出是 [-1, 1]，需要转换到 [0, 255]
                face_np = face_tensor.permute(1, 2, 0).numpy()
                face_np = ((face_np + 1) / 2 * 255).astype(np.uint8)

                return {
                    'face': face_np,
                    'confidence': float(prob),
                    'bbox': None,  # facenet-pytorch不直接返回bbox
                    'landmarks': None
                }

            else:
                # 标准MTCNN库方式
                result = self.detector.detect_faces(img)

                if not result or len(result) == 0:
                    return None

                # 选择置信度最高的人脸
                face = max(result, key=lambda x: x['confidence'])

                if face['confidence'] < min_confidence:
                    return None

                bbox = face['box']
                landmarks = face['keypoints']  # 仅记录；FaceAge 原始流程未用其做几何对齐

                # 手动裁剪和缩放
                x, y, w, h = bbox
                x, y = max(0, x), max(0, y)
                face_crop = img[y:y+h, x:x+w]

                # 缩放到160x160
                face_resized = cv2.resize(face_crop, (self.image_size, self.image_size))

                return {
                    'face': face_resized,
                    'confidence': face['confidence'],
                    'bbox': bbox,
                    'landmarks': landmarks
                }

        except Exception as e:
            print(f"处理图像时出错: {str(e)}")
            return None

    def process_directory(self, input_dir, output_dir, min_confidence=0.9):
        """
        批量处理目录中的所有图像

        Args:
            input_dir (str): 输入图像目录
            output_dir (str): 输出目录
            min_confidence (float): 最小置信度阈值

        Returns:
            dict: 处理统计信息
        """
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]

        if not image_files:
            print(f"在 {input_dir} 中未找到图像文件")
            return {'success': 0, 'failed': 0, 'skipped': 0}

        print(f"\n找到 {len(image_files)} 个图像文件")
        print(f"输出目录: {output_dir}")
        print(f"最小置信度: {min_confidence}")
        print("=" * 80)

        stats = {'success': 0, 'failed': 0, 'skipped': 0}

        for filename in tqdm(image_files, desc="处理图像"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # 跳过已存在的文件
            if os.path.exists(output_path):
                stats['skipped'] += 1
                continue

            # 检测并裁剪人脸
            result = self.detect_and_crop_face(input_path, min_confidence)

            if result is not None:
                # 保存裁剪后的人脸（使用 imencode+tofile 支持中文路径）
                face_bgr = cv2.cvtColor(result['face'], cv2.COLOR_RGB2BGR)
                ext = os.path.splitext(filename)[1].lower() or '.jpg'
                success, buf = cv2.imencode(ext, face_bgr)
                if success:
                    buf.tofile(output_path)
                stats['success'] += 1

                # 输出详细信息（每100个输出一次）
                if stats['success'] % 100 == 0:
                    tqdm.write(f"[{stats['success']}] {filename}: 置信度 {result['confidence']:.3f}")
            else:
                stats['failed'] += 1
                tqdm.write(f"✗ {filename}: 未检测到人脸或置信度过低")

        # 打印统计信息
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"总计: {len(image_files)} 个图像")
        print(f"成功: {stats['success']} 个")
        print(f"跳过: {stats['skipped']} 个 (已存在)")
        print(f"失败: {stats['failed']} 个")
        if (len(image_files) - stats['skipped']) > 0:
            success_rate = stats['success'] / (len(image_files) - stats['skipped']) * 100
            print(f"成功率: {success_rate:.1f}%")
        print("=" * 80)

        return stats


def main():
    """主函数：示例用法"""

    # 配置参数
    input_directory = os.environ.get("FACEAGE_IMAGE_DIR", "data/face_data_complete")  # 原始完整帧目录
    output_directory = os.environ.get("FACEAGE_CROPS_DIR", "data/face_crops_faceage")  # FaceAge风格处理后的目录

    # 创建预处理器
    # 注意：仅 facenet-pytorch 后端会用到 GPU；默认 mtcnn 后端不需要
    preprocessor = FaceAgePreprocessor(
        image_size=160,     # FaceAge标准尺寸
        margin=0,           # 紧贴边界（仅 facenet-pytorch 后端生效）
        device='cuda'       # 'cuda' 或 'cpu'
    )

    # 批量处理
    stats = preprocessor.process_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        min_confidence=0.9  # 置信度阈值
    )

    print("\n处理完成！")
    print(f"成功处理的图像保存在: {output_directory}")
    print("\n这些图像现在可以直接用于FaceAge模型预测")


if __name__ == "__main__":
    main()
