[README.md](https://github.com/user-attachments/files/22120149/README.md)
# BSc Thesis – Cloud Prediction and Solar Irradiance Forecasting with Deep Learning

This repository contains the code, processed data, and experiments carried out for my **Bachelor Thesis**.
The goal is to **segment clouds in all-sky (fisheye) images** and **predict short-term global horizontal irradiance (GHI)** (nowcasting and forecasting up to +60 min horizons).

---

## Requirements

* macOS with Apple Silicon (M1/M2) or Linux.
* Python 3.10+.
* TensorFlow 2.13 (with Metal support on Mac).
* Libraries listed in `requirements.txt`.

Quick installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```
tfg_project/
├── checkpoints_future_gru/     # FUTURE weights (GRU)
├── checkpoints_future_lstm/    # FUTURE weights (LSTM)
├── checkpoints_now8h/          # NOW weights (8h, GHI + SEG)
├── checkpoints_now8h_seg/      # NOW weights (8h, SEG only)
├── data/                       # Processed data and pair CSVs
├── figuras_future/             # Qualitative figures (FUTURE suite)
├── reports_future/             # FUTURE reports and tables
├── reports_now/                # NOW reports and tables
├── src/                        # Source code (training, eval, viz)
└── requirements.txt            # Dependencies
```

---

## Training

### NOW

* `src/def_train_now_m1.py` → joint training (SEG+GHI).
* `src/def_train_now_seg_m1.py` → segmentation-only training.

Example:

```bash
caffeinate -d -i python src/def_train_now_m1.py
```

```bash
caffeinate -d -i python src/def_train_now_seg_m1.py
```

### FUTURE

Training in 3 phases (GRU or LSTM):

**SEG phase**

```bash
caffeinate -d -i python src/def_train_future_segphase_m1.py
```

**GHI phase**

```bash
caffeinate -d -i python src/def_train_future_ghiphase_m1.py
```

**Balanced phase**

```bash
caffeinate -d -i python src/def_train_future_balanced_m1.py
```

---

## Evaluation

### NOW

Evaluate one or multiple NOW models on test set:

```bash
caffeinate -d -i python src/eval_now_model_m1.py \
  --csv data/processed/pairs_test.csv \
  --dir checkpoints_now8h,checkpoints_now8h_seg
```

Reports in `reports_now/<tag>/` with metrics and plots.
Global tables in `reports_now/_tables/`.

### FUTURE

Evaluate a FUTURE model:

```bash
# List available models
python src/eval_future_models_m1.py --list

# Example: best GRU balanced compromise
caffeinate -d -i python src/eval_future_models_m1.py --model gru_bal \
  --csv data/processed/pairs_test.csv
```

Reports in `reports_future/<tag>/`.

#### FUTURE (full suite)

Evaluate all 6 FUTURE models and compile global tables:

```bash
caffeinate -d -i python src/eval_future_suite_m1.py \
  --csv data/processed/pairs_test.csv
```

* Tables in `reports_future/_tables/summary_<timestamp>.{csv,md}`.
* Horizon-based GHI metrics in `ghi_per_horizon_<timestamp>.{csv,md}`.

#### FUTURE (comparative summary)

Compile a final table and plots from all summaries:

```bash
python src/summarize_future_reports_m1.py
```

* Tables in `reports_future/_summary/summary_table.{csv,md}`.
* Comparative figures:

  * `mae_rmse.png` → GHI MAE/RMSE per model.
  * `miou_pixelacc.png` → mIoU and PixelAcc per model.
  * `recall_por_clase.png` → Class-wise Recall (SKY, THICK, THIN, SUN).
  * `ghi_mae_por_horizonte.png` → MAE evolution by forecast horizon.

---

## Visualization

### NOW – 2-model comparison

```bash
caffeinate -d -i python src/quick_viz_now_m1_dual.py
```

Output in `reports_now/_figures/`.

### FUTURE – Qualitative suite

```bash
caffeinate -d -i python src/quick_viz_future_suite_m1.py \
  --csv data/processed/pairs_test.csv
```

Output in `reports_future/_figures/<timestamp>/` with 6 figures (gru/lstm × seg/ghi/bal).

---

## Results

* **NOW**: segmentation metrics (PixelAcc, class-wise Recall, mIoU) and training curves.
* **FUTURE**: GHI prediction metrics (MAE, RMSE, NMAE%) + parallel segmentation metrics.

CSV/Markdown tables in `reports_*` for reproducible analysis.
Comparative plots (GHI curves, segmentation overlays) in `figuras_future/` and `reports_now/_figures/`.
Global summaries generated with `summarize_future_reports_m1.py`.
