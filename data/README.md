# Input data

Place the de-identified analysis table here as:

```
data/merged_data.xlsx
```

This file is **not distributed** with the repository — it contains patient-level clinical data and
derived facial-age estimates from identifiable photographs. See the Data availability section of
the manuscript for the request procedure.

Alternatively, keep the file anywhere and point the scripts at it:

```bash
export FACEAGE_MERGED_XLSX=/path/to/merged_data.xlsx
```

`scripts/utils/config.py` resolves the input in this order:

1. `$FACEAGE_MERGED_XLSX`, if set;
2. `data/merged_data.xlsx`;
3. `../Clinical_dataset/merged_data.xlsx` (the originating project layout).

---

## Expected schema

One row per patient. Column names are Chinese, as produced by the clinical data pipeline.

### Required by `load_from_merged_xlsx()`

| Column | Meaning | Type |
|---|---|---|
| `编号` | Record identifier | int / str |
| `年龄` | Chronological age (years) | float |
| `biological_age` | FaceAge-predicted biological age (years) | float |
| `7天内死亡` | 7-day mortality outcome (0 = survived, 1 = died) | int |
| `ApacheⅡ评分` | APACHE II score | float |
| `occlusion_level` | Facial occlusion grade (0–3; see below) | int |

The loader renames these internally to `实际年龄`, `生物学年龄`, `7天死亡`, `APACHE评分` and
`遮挡等级`, and derives:

```
Age_Gap = 生物学年龄 − 实际年龄
```

Rows with a missing `biological_age` or an `occlusion_level` outside the requested range are
dropped before analysis.

### Required by the ML pipeline

The eight bedside indicators (feature set A):

`性别`, `体温`, `心率`, `呼吸频率`, `收缩压`, `舒张压`, `血氧饱和度`, `血糖`

`scripts/ml_modeling/ml_final.py` may also use `GCS评分`, `白细胞介素-6`, `PCT降钙素原`,
`C反应蛋白`, `WBC` and `中性粒细胞` for the exploratory analyses.

### Occlusion grades

| Code | Grade |
|---|---|
| 0 | none / negligible |
| 1 | mild |
| 2 | moderate |
| 3 | severe |

The analysis set keeps `occlusion_level ≤ 2` (grade 3, n = 15, is excluded), giving N = 1,470.

### Recovering the pre-imputation missingness (optional)

The eight bedside columns in the distributed table are already filled with full-cohort medians, so
the original blanks are gone. The fold-internal multiple-imputation pipeline — `ml_final.py` and
`10_figures_mice.py` — can recover them from the per-patient source workbooks if you have them,
placed as:

```
data/raw_groups/*.xlsx
```

Each workbook must carry a `编号` column, and its file name without the extension must match the
`excel_file` column of `merged_data.xlsx`. Point elsewhere with:

```bash
export FACEAGE_RAW_GROUPS_DIR=/path/to/raw_groups
```

If the directory is absent, those scripts still run but there is nothing for MICE to impute, and
the `MICE` rows then reproduce the fold-internal median reference instead. See README §7.

### Optional columns

The originating table also carries `诊断`, `PH`, `氧分压`, `乳酸Lac`, `group_name`, `patient_id`,
`image_filename`, `excel_file`, and alternative biological-age columns (`biological_age_mivolo`,
`biological_age_dex`, `biological_age_deepface`) that are **not** used in the manuscript, apart
from `excel_file` (see above) and `性别`, which is one of the eight bedside indicators.
