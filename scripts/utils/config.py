"""Path and parameter configuration.

Every path is derived from the repository root, so nothing is hard-coded. The input table is
located in this order:

  1. the FACEAGE_MERGED_XLSX environment variable, if it is set
  2. <repo>/data/merged_data.xlsx
  3. <repo>/../Clinical_dataset/merged_data.xlsx     original project layout, back-compatibility

Also holds the cohort-wide constants: age groups, occlusion grade labels, the random seed, the
plotting palettes and figure sizes, and the feature lists used by the machine-learning scripts."""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# 原始数据
RAW_DATA_DIR = DATA_DIR / "raw"
COMPLETE_SAMPLE_FILE = RAW_DATA_DIR / "complete_sample_data.xlsx"
FACEAGE_PREDICTIONS_FILE = RAW_DATA_DIR / "faceage_predictions.xlsx"
HIGH_OCCLUSION_FILE = RAW_DATA_DIR / "high_occlusion.xlsx"

# 主分析输入：Clinical_dataset/merged_data.xlsx（表 1-19、图 2-7 的统一数据源）
# 查找顺序：
#   1) 环境变量 FACEAGE_MERGED_XLSX（显式指定文件路径）；
#   2) <repo>/data/merged_data.xlsx（发布包的推荐位置）；
#   3) <repo>/../Clinical_dataset/merged_data.xlsx（原始项目目录结构，向后兼容）。
_env_xlsx = os.environ.get("FACEAGE_MERGED_XLSX")

DATA_DIR.mkdir(parents=True, exist_ok=True)

if _env_xlsx:
    MERGED_DATA_XLSX = Path(_env_xlsx)
elif (DATA_DIR / "merged_data.xlsx").exists():
    MERGED_DATA_XLSX = DATA_DIR / "merged_data.xlsx"
else:
    MERGED_DATA_XLSX = PROJECT_ROOT.parent / "Clinical_dataset" / "merged_data.xlsx"

CLINICAL_DATASET_DIR = MERGED_DATA_XLSX.parent
MERGED_DATA_H5 = CLINICAL_DATASET_DIR / "merged_data.h5"

# ============================================================================
# 数据处理参数
# ============================================================================

# 年龄分组（4组，新版）
AGE_GROUPS_4 = {
    '青年': (0, 45),
    '中年': (45, 65),
    '老年': (65, 85),
    '极高龄': (85, 200)
}

# 年龄分组（3组，旧版，仅用于对比）
AGE_GROUPS_3 = {
    '青年': (0, 45),
    '中年': (45, 65),
    '老年': (65, 200)
}

# 遮挡等级
OCCLUSION_LEVELS = {
    0: 'Negligible occlusion',
    1: 'Mild occlusion',
    2: 'Moderate occlusion',
    3: 'Severe occlusion'
}

# APACHE 评分分组
APACHE_GROUPS = {
    '低危': (0, 11),
    '中危': (11, 21),
    '高危': (21, 100)
}

# ============================================================================
# 多模型年龄估计配置
# ============================================================================

# 可用模型列表
AGE_MODELS = {
    'FaceAge': 'biological_ages',           # HDF5 metadata key
    'MiVOLO': 'biological_ages_mivolo',
    'DEX': 'biological_ages_dex',
    'DeepFace': 'biological_ages_deepface',
}

# xlsx 列名映射（merged_data.xlsx 中各模型生物学年龄的列名）
AGE_MODEL_XLSX_COLS = {
    'FaceAge': 'biological_age',
    'MiVOLO': 'biological_age_mivolo',
    'DEX': 'biological_age_dex',
    'DeepFace': 'biological_age_deepface',
}

# 模型显示名称
AGE_MODEL_LABELS = {
    'FaceAge': 'FaceAge',
    'MiVOLO': 'MiVOLO',
    'DEX': 'DEX',
    'DeepFace': 'DeepFace',
}

# ============================================================================
# 可视化参数
# ============================================================================

# English publication font
FONT_CHINESE = ['Arial', 'DejaVu Sans', 'Helvetica']

# 颜色方案
COLORS_OCCLUSION = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']  # 绿、橙、橙红、红
COLORS_AGE_GROUPS = ['#3498db', '#9b59b6', '#e67e22', '#e74c3c']  # 蓝、紫、橙、红
COLORS_OUTCOME = {
    'survived': '#2ecc71',  # 绿色
    'died': '#e74c3c'  # 红色
}
# 多模型颜色方案
COLORS_MODEL = {
    'FaceAge': '#3498db',     # 蓝色
    'MiVOLO': '#e74c3c',      # 红色
    'DEX': '#2ecc71',         # 绿色
    'DeepFace': '#9b59b6',    # 紫色
}

# 图表尺寸
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_GRID_2X2 = (14, 12)
FIGSIZE_ROC = (8, 8)

# 图表DPI
DPI_DEFAULT = 300

# ============================================================================
# 统计检验参数
# ============================================================================

# 显著性水平
ALPHA = 0.05

# 交叉验证折数
CV_FOLDS = 5

# 随机种子
RANDOM_SEED = 42

# ============================================================================
# 快速评估指标（方向2使用）
# ============================================================================

# ============================================================================
# 炎症标志物（快速实验室指标，允许缺失）
# ============================================================================

INFLAMMATORY_MARKERS = [
    '白细胞介素-6',
    'PCT降钙素原',
    'C反应蛋白',
    'WBC',
    '中性粒细胞',
]

QUICK_ASSESSMENT_FEATURES = [
    '性别',
    '体温',
    '心率',
    '呼吸频率',
    '收缩压',
    '舒张压',
    '血氧饱和度',
    'GCS评分',
    '血糖'
]
