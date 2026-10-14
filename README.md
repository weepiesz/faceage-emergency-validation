# Analysis code — External validation of FaceAge in critically ill emergency patients

This repository contains the analysis code accompanying the manuscript:

> **External Validation of FaceAge in Critically Ill Emergency Patients: Prediction Performance,
> Occlusion Impact, and Prognostic Boundary of Age Gap**

It reproduces the age-prediction performance analysis, the facial-occlusion impact analysis, the
prognostic evaluation of Age Gap, and the machine-learning incremental-value analysis reported in
the manuscript (Tables 1–19, Figures 1–7).

---

## 1. Data availability

**Patient-level data are not included in this repository.** The dataset contains identifiable facial
photographs and clinical records of ICU patients and cannot be shared publicly. De-identified data
may be available from the corresponding author on reasonable request, subject to institutional
review board approval.

All analysis scripts read a single de-identified table:

```
data/merged_data.xlsx
```

See [`data/README.md`](data/README.md) for the expected schema. If you hold the file elsewhere,
point to it explicitly instead of copying it:

```bash
export FACEAGE_MERGED_XLSX=/path/to/merged_data.xlsx
```

Every script in this repository reads `merged_data.xlsx`. **One optional additional input** is
described in §7: the fold-internal multiple-imputation pipeline can also read the per-patient
source workbooks, if you have them, to recover the values that were missing before imputation.
Without them the pipeline still runs, but it reproduces a different imputation scheme — see §7.

---

## 2. Environment

### Main analysis environment

Python ≥ 3.9 with the packages in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

### FaceAge inference environment (separate, mandatory)

FaceAge inference **cannot** run in the main environment. The released FaceAge model uses Keras
Lambda layers serialised under Python 3.6, which Python 3.8+ cannot deserialise.

- Python 3.7, TensorFlow 2.6.2, Keras 2.6.0, `protobuf==3.20.3`

Use the scripts in [`preprocessing/`](preprocessing/) there. See §5.

---

## 3. Repository layout

```
.
├── data/                       # input data (not distributed) — see data/README.md
├── results/                    # all script outputs land here
├── preprocessing/
│   ├── faceage_mtcnn_preprocess.py      # MTCNN detection + bounding-box crop, 160x160
│   └── predict_faceage_from_images.py   # FaceAge biological-age inference  [faceage-py37 env]
└── scripts/
    ├── utils/                  # shared helpers: paths, data loading, statistics, plotting
    ├── annotation/             # manual occlusion grading: grading package, ingest, reliability
    ├── ml_modeling/            # machine-learning pipeline (feature sets A–I)
    ├── stage1*.py              # FaceAge age-prediction performance
    ├── stage2*.py              # occlusion impact
    ├── stage3_roc_analysis.py  # Age Gap prognostic value
    ├── age_stratified_bias.py  # age-stratified Age Gap bias
    ├── age_stratified_error_apache.py
    ├── age_gap_bias_correction.py
    ├── fig6_bias_corrected_roc.py
    ├── figure2_agreement_composite.py
    └── generate_table1.py
```

All paths are resolved relative to the repository root through
[`scripts/utils/config.py`](scripts/utils/config.py) — no absolute paths are hard-coded anywhere in
this repository.

---

## 4. Study cohort (as implemented)

| Step | n |
|---|---|
| Patients photographed | 1,942 |
| Excluded (unusable photograph or incomplete record) | 457 |
| Initial dataset (all graded for occlusion) | 1,485 |
| Excluded — severe occlusion (grade 3) | 15 |
| **Analysis set** | **1,470** (150 deaths, 10.2%) |

Occlusion grades (`occlusion_level`): 0 = none/negligible, 1 = mild, 2 = moderate, 3 = severe.
The analysis set uses `occlusion_level ≤ 2`, which is the default of
`load_from_merged_xlsx(max_occlusion_level=2)`.

Feature sets used by the ML pipeline:

| Set | Contents |
|---|---|
| A | 8 bedside indicators (sex, temperature, heart rate, respiratory rate, systolic BP, diastolic BP, oxygen saturation, glucose) |
| B | A + chronological age |
| C | A + FaceAge |
| D | A + chronological age + Age Gap |
| E | A + Age Gap |
| F | APACHE II |
| G | F + FaceAge |
| H | F + Age Gap |
| I | F + FaceAge + Age Gap |

---

## 5. Pipeline

### Step 0 — Face extraction (FaceAge-compatible)

```bash
python preprocessing/faceage_mtcnn_preprocess.py     # -> 160x160 MTCNN-cropped faces
```

Reads `FACEAGE_IMAGE_DIR` (default `data/face_data_complete`) and writes `FACEAGE_CROPS_DIR`
(default `data/face_crops_faceage`). The raw photographs are not distributed.

The pipeline is the one described in the manuscript: MTCNN localises the face, its bounding
box is cropped and resized to 160 x 160 px, and the cropped image is later standardised by its
own mean and standard deviation at inference time. **No landmark-based geometric realignment is
applied** — the original FaceAge pipeline does not apply one either.

Detection uses the `mtcnn` package (v0.1.1), the implementation shipped in the FaceAge
environment. facenet-pytorch is available as an opt-in alternative:

```bash
FACEAGE_MTCNN_BACKEND=facenet_pytorch python preprocessing/faceage_mtcnn_preprocess.py
```

Note that facenet-pytorch *does* return a landmark-aligned crop, so it will not reproduce the
manuscript's pipeline. The default was left on `mtcnn` for that reason.

### Step 1 — FaceAge inference  ⚠️ `faceage-py37` environment

```bash
conda activate faceage-py37
python preprocessing/predict_faceage_from_images.py
```

Produces the biological-age predictions that become the `biological_age` column of
`merged_data.xlsx`.

The model weights are the pre-trained FaceAge model distributed in release **v1** of the original
repository:

```
https://github.com/AIM-Harvard/FaceAge/releases/download/v1/faceage_model.h5
```

It is an Inception-ResNet v1 network taking a 160 x 160 x 3 RGB input, applied as distributed,
with no retraining or fine-tuning. The file used for this study was 92,003,240 bytes
(87.74 MiB) with

```
SHA-256  dd43eb66616558a70f86d55d405c2e5be1e49569654b7eb852c206393fd7a85d
```

Download it to `FaceAge/models/faceage_model.h5`, or point `FACEAGE_MODEL_H5` at it.

### Step 2 — Occlusion grading and reliability

```bash
python scripts/annotation/build_annotation_package.py   # build the de-identified grading package
python scripts/annotation/ingest_assessor2.py           # ingest the second assessor's grades
python scripts/annotation/analyze_inter_rater.py        # Cohen's kappa + agreement
python scripts/annotation/sensitivity_assessor2.py      # sensitivity analysis on assessor 2
```

### Step 3 — FaceAge performance and agreement

```bash
python scripts/stage1_performance.py           # MAE, RMSE, bias, limits of agreement  -> Table 3
python scripts/stage1b_bland_altman.py         # Bland-Altman agreement               -> Figure 2
python scripts/stage1c_age_band_alignment.py   # agreement in the >=60-yr applicability population
python scripts/figure2_agreement_composite.py  # Figure 2 composite panel
python scripts/generate_table1.py              # baseline characteristics             -> Table 2
```

### Step 4 — Occlusion impact, Age Gap, stratified bias

```bash
python scripts/stage2_occlusion_analysis.py     # occlusion impact on error   -> Tables 4, 6
python scripts/stage2_age_adjusted_error.py     # age-adjusted occlusion error
python scripts/stage3_roc_analysis.py           # ROC-AUC for 7-day mortality -> Table 5, Figure 3
python scripts/age_stratified_bias.py           # age-stratified Age Gap      -> Tables 8-10, Figure 4
python scripts/age_stratified_error_apache.py   # error vs APACHE / inflammatory markers -> Tables 11, 12
```

### Step 5 — Bias correction of Age Gap

```bash
python scripts/age_gap_bias_correction.py       # functional form x cross-fitting -> Tables 15, 16
python scripts/fig6_bias_corrected_roc.py       # ROC before/after correction    -> Figure 7
```

### Step 6 — Machine-learning pipeline

The authoritative pipeline is **`scripts/ml_modeling/ml_final.py`**. It implements 5 × 10 repeated
stratified cross-validation (seed 42), fold-internal multiple imputation (`m = 5`, chained
equations with random forest), six metrics at a fixed 90%-specificity operating point, and
calibration intercept/slope with 2,000 stratified bootstrap resamples. It writes
`results/ml_modeling/ml_final.{txt,log,npz}`.

```bash
cd scripts/ml_modeling
python ml_final.py               # main pipeline -> Table 13, Table 14, Figure 5, Figure 6
python final_table13.py          # imputation-scheme comparison (superseded as the Table 13 source)
python extract_metric_tables.py  # extract the reported metric tables from ml_final.log
python bootstrap_ci.py           # paired-bootstrap 95% CIs for the key contrasts -> Table 19
python bootstrap_ci_three_scheme.py  # same contrasts, three-scheme version (superseded by bootstrap_ci.py)
python 07_apache_incremental.py  # APACHE II benchmark sets F-I -> Table 14
python 06_leakfree_bias_corrected_ml.py  # bias correction fitted within training folds -> Tables 17, 18
python 10_figures_mice.py        # -> Figure 5, Figure 6  (fold-internal MICE)
python 02_cumulative_occlusion_stratified.py
python 01_data_exploration.py    # missing-value analysis reported in Methods
```

---

## 6. Figure and table provenance

| Manuscript item | Producing script |
|---|---|
| Figure 1 — STROBE flow diagram | `scripts/utils/plot_strobe_flowchart.py` |
| Table 1 — occlusion grading criteria | criteria defined in the manuscript (no script) |
| Table 2 — baseline characteristics | `scripts/generate_table1.py` |
| Table 3 — FaceAge performance by occlusion level | `scripts/stage1_performance.py` |
| Results text — agreement in the ≥60-year applicability population | `scripts/stage1c_age_band_alignment.py` |
| Methods — missing-value analysis of the eight bedside indicators | `scripts/ml_modeling/01_data_exploration.py` |
| Figure 2 — agreement with chronological age | `scripts/figure2_agreement_composite.py`, `scripts/stage1b_bland_altman.py` |
| Table 4 — pairwise Age Gap comparisons across occlusion levels | `scripts/stage2_occlusion_analysis.py` |
| Table 5 — ROC-AUC by predictor | `scripts/stage3_roc_analysis.py` |
| Figure 3 — ROC curves (age, FaceAge, Age Gap, APACHE II) | `scripts/stage3_roc_analysis.py` |
| Table 6 — Age Gap AUC across cumulative occlusion strata | `scripts/stage3_roc_analysis.py` |
| Table 7 — predictors by survival status | `scripts/stage3_roc_analysis.py`, `scripts/age_stratified_error_apache.py` |
| Tables 8–10, Figure 4 — Age Gap by age group | `scripts/age_stratified_bias.py` |
| Tables 11–12 — inflammatory markers, Age Gap correlations | `scripts/age_stratified_error_apache.py` |
| Table 13 — nine feature sets, RF and LR | `scripts/ml_modeling/ml_final.py` (metric tables via `extract_metric_tables.py`) |
| Table 14 — pairwise AUC comparisons | `scripts/ml_modeling/ml_final.py`, `scripts/ml_modeling/07_apache_incremental.py` |
| Figure 5 — ROC for feature sets A–E | `scripts/ml_modeling/10_figures_mice.py` |
| Figure 6 — calibration and decision-curve analysis | `scripts/ml_modeling/10_figures_mice.py` |
| Table 15 — Age Gap before/after bias correction | `scripts/age_gap_bias_correction.py` |
| Table 16 — functional-form sensitivity | `scripts/age_gap_bias_correction.py` |
| Figure 7 — ROC before/after bias correction | `scripts/fig6_bias_corrected_roc.py` |
| Tables 17–18 — ML before/after bias correction | `scripts/ml_modeling/06_leakfree_bias_corrected_ml.py` |
| Table 19 — achieved precision | `scripts/ml_modeling/bootstrap_ci.py` (from `ml_final.py`; `bootstrap_ci_three_scheme.py` is the three-scheme version) |
| Additional file 4, Table 3 — imputation sensitivity | S0 / S1 from `scripts/ml_modeling/final_table13.py`; S2 (the reported scheme) from `scripts/ml_modeling/ml_final.py` (`MICE`, raw variables, m = 5) |

---

## 7. Notes on reproducibility

- **Random seed**: 42, set in `scripts/utils/config.py`. Cross-validation uses
  `StratifiedKFold(shuffle=True, random_state=42 + repeat)`.
- **Bootstrap CIs** use `RandomState(42)` with 2,000 resamples (percentile method).
- **Imputation**: the reported ML results use multiple imputation fitted **within training folds
  only**. Results from an earlier pipeline that imputed on the full cohort before splitting are
  superseded and are not included here.
- **Recovering the original blanks (required by the MICE scheme)**: `merged_data.xlsx` carries
  full-cohort medians in the eight bedside columns, so the pre-imputation missingness is no longer
  in the file. `ml_final.py` and `10_figures_mice.py` therefore look for the per-patient source
  workbooks in `data/raw_groups/` (override with the `FACEAGE_RAW_GROUPS_DIR` environment
  variable), matching them through the `excel_file` column of `merged_data.xlsx`. **If that
  directory is absent, both scripts still run, but MICE has nothing to impute and the `MICE` rows
  then equal the fold-internal median reference rather than the multiple-imputation scheme
  reported in the manuscript.** The console prints a warning when this happens; `final_table13.py`
  exits with an error instead, because the scheme comparison is its entire purpose. To reproduce
  the reported MICE numbers, supply the source workbooks. (Alternatively, publish
  `merged_data.xlsx` with the original blanks left as blanks instead of median-filled, and the
  workbooks become unnecessary.)
- **Bias correction of Age Gap** is a quadratic regression of Age Gap on chronological age, fitted
  **within training folds only**; AUC was insensitive to functional form (linear 0.476, quadratic
  0.488, cubic 0.495) and to cross-fitting (Table 16).
- **Run order**: `ml_final.py` must be run before `10_figures_mice.py`. The latter re-runs the same
  5 × 10 cross-validation and aborts if its fold-mean AUCs do not match `results/ml_modeling/ml_final.txt`
  to four decimal places — an intentional guard against a silently divergent pipeline.
- **Runtime**: `ml_final.py` and `10_figures_mice.py` each take roughly 8–12 minutes on a desktop
  CPU. `10_figures_mice.py --nrep 1` is a ~2-minute smoke test.

---

## 8. Citation and licence

If you use this code, please cite the manuscript above. A licence will be added on publication;
until then, all rights are reserved by the authors.
