# -*- coding: utf-8 -*-
# eval_future_suite_m1.py
# Launches the evaluation of the 6 FUTURE models and compiles tables (CSV and Markdown),
# showing progress bars with tqdm.
#
# Usage:
#   # Run evals + create tables
#   python src/eval_future_suite_m1.py --csv data/processed/pairs_test.csv
#
#   # Only compile tables from previous reports (does not re-evaluate)
#   python src/eval_future_suite_m1.py --csv data/processed/pairs_test.csv --skip-eval
#
#   # Select a subset of labels
#   python src/eval_future_suite_m1.py --csv data/processed/pairs_test.csv --labels gru_seg,gru_ghi,gru_bal
#
# Output:
#   reports_future/_tables/summary_<timestamp>.{csv,md}
#   reports_future/_tables/ghi_per_horizon_<timestamp>.{csv,md}

import os, sys, json, argparse, datetime as dt, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ── tqdm (progress) ───────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kw):
        # Fallback dummy: no bars, same minimal API
        return iterable if iterable is not None else range(0)

ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
EVAL_SCRIPT = SRC / "eval_future_models_m1.py"

# Labels to evaluate (the main 6)
LABELS = ["gru_seg","gru_ghi","gru_bal","lstm_seg","lstm_ghi","lstm_bal"]

def run_eval(label: str, csv_path: Path) -> Path:
    """Runs the evaluation script for a label and returns the path to summary.json."""
    out_dir = ROOT / "reports_future" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"

    # Note: the called script already shows its own internal progress bar
    cmd = [sys.executable, str(EVAL_SCRIPT), "--model", label, "--csv", str(csv_path)]
    subprocess.run(cmd, check=True)
    if not summary_json.exists():
        raise FileNotFoundError(f"No se generó {summary_json}. Revisa la evaluación de {label}.")
    return summary_json

def load_metrics(summary_json: Path) -> dict:
    with open(summary_json, "r") as f:
        s = json.load(f)
    ghi_over = s["metrics"]["ghi"].get("overall", {})
    seg      = s["metrics"]["seg"]
    rnn      = s.get("rnn_type","?")
    ckpt     = s.get("ckpt_prefix","?")
    per_h    = s["metrics"]["ghi"].get("per_horizon", None)
    return dict(
        label   = summary_json.parent.name,
        rnn     = rnn.upper(),
        ckpt    = ckpt,
        ghi_mae = ghi_over.get("mae", np.nan),
        ghi_rmse= ghi_over.get("rmse", np.nan),
        ghi_mbe = ghi_over.get("mbe", np.nan),
        ghi_nmae_pct = ghi_over.get("nmae_pct", np.nan),
        pixacc  = seg.get("pixel_acc", np.nan),
        rec_sky = seg.get("rec_sky", np.nan),
        rec_thick = seg.get("rec_thick", np.nan),
        rec_thin  = seg.get("rec_thin", np.nan),
        rec_sun   = seg.get("rec_sun", np.nan),
        miou    = seg.get("miou", np.nan),
        per_h   = per_h
    )

def to_markdown_table(df: pd.DataFrame) -> str:
    # Simple Markdown without external dependencies
    cols = df.columns
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"]*len(cols)) + " |")
    for _, row in df.iterrows():
        vals = [row[c] if not isinstance(row[c], float) else (f"{row[c]:.3f}" if not np.isnan(row[c]) else "") for c in cols]
        lines.append("| " + " | ".join(map(str, vals)) + " |")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Test CSV (pairs_test.csv).")
    ap.add_argument("--skip-eval", action="store_true", help="Do not re-evaluate; only read existing summaries.")
    ap.add_argument("--labels", type=str, default=",".join(LABELS),
                    help="Comma-separated list of labels to evaluate (default is all 6).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    assert csv_path.exists(), f"CSV does not exist: {csv_path}"

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    out_tables = ROOT / "reports_future" / "_tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    summaries = []

    # ── Progress: model evaluation ─────────────────────────────────────────────
    if not args.skip_eval:
        with tqdm(total=len(labels), desc="Evaluando modelos (suite)", unit="modelo", ncols=100) as pbar:
            for lab in labels:
                pbar.set_postfix_str(lab, refresh=True)
                summ_path = run_eval(lab, csv_path)
                summaries.append(load_metrics(summ_path))
                pbar.update(1)
    else:
        # Read existing summaries with a bar
        for lab in tqdm(labels, desc="Leyendo summaries existentes", unit="modelo", ncols=100):
            summ_path = ROOT / "reports_future" / lab / "summary.json"
            if not summ_path.exists():
                print(f"[warn] No existe {summ_path}; saltando {lab}.")
                continue
            summaries.append(load_metrics(summ_path))

    if not summaries:
        print("No hay summaries para compilar tablas."); return

    # ── Progress: build main table ────────────────────────────────────────────
    rows = []
    for s in tqdm(summaries, desc="Compilando tabla global", unit="modelo", ncols=100):
        rows.append(dict(
            Modelo = s["label"],
            RNN    = s["rnn"],
            MAE_Wm2   = s["ghi_mae"],
            RMSE_Wm2  = s["ghi_rmse"],
            NMAE_pct  = s["ghi_nmae_pct"],
            MBE_Wm2   = s["ghi_mbe"],
            PixelAcc  = s["pixacc"],
            Recall_SKY   = s["rec_sky"],
            Recall_THICK = s["rec_thick"],
            Recall_THIN  = s["rec_thin"],
            Recall_SUN   = s["rec_sun"],
            mIoU         = s["miou"],
            Checkpoint   = s["ckpt"]
        ))
    df = pd.DataFrame(rows)

    # Formatting for Markdown (CSV stays raw)
    df_fmt = df.copy()
    for col in ["MAE_Wm2","RMSE_Wm2","MBE_Wm2","NMAE_pct"]:
        if col in df_fmt: df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    for col in ["PixelAcc","Recall_SKY","Recall_THICK","Recall_THIN","Recall_SUN","mIoU"]:
        if col in df_fmt: df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")

    csv_out = out_tables / f"summary_{ts}.csv"
    md_out  = out_tables / f"summary_{ts}.md"
    df.to_csv(csv_out, index=False)
    with open(md_out, "w") as f:
        f.write("# Resumen de modelos FUTURE (test)\n\n")
        f.write(to_markdown_table(df_fmt[[
            "Modelo","RNN","MAE_Wm2","RMSE_Wm2","NMAE_pct","MBE_Wm2",
            "PixelAcc","Recall_SKY","Recall_THICK","Recall_THIN","Recall_SUN","mIoU"
        ]]))
        f.write("\n")

    print(f"[ok] Tabla principal guardada:\n- {csv_out}\n- {md_out}")

    # ── Progress: per-horizon table ───────────────────────────────────────────
    horizons = []
    for s in tqdm(summaries, desc="Compilando tabla GHI por horizonte", unit="modelo", ncols=100):
        ph = s["per_h"]
        if not ph: continue
        rec = dict(Modelo=s["label"], RNN=s["rnn"])
        for i, v in enumerate(ph.get("mae", [])):
            rec[f"MAE_t{i}"] = v
        for i, v in enumerate(ph.get("rmse", [])):
            rec[f"RMSE_t{i}"] = v
        horizons.append(rec)

    if horizons:
        dfh = pd.DataFrame(horizons).fillna(np.nan)
        csv_h = out_tables / f"ghi_per_horizon_{ts}.csv"
        md_h  = out_tables / f"ghi_per_horizon_{ts}.md"

        dfh.to_csv(csv_h, index=False)

        mae_cols = ["Modelo","RNN"] + [c for c in dfh.columns if c.startswith("MAE_t")]
        dfh_fmt = dfh[mae_cols].copy()
        for c in mae_cols[2:]:
            dfh_fmt[c] = dfh_fmt[c].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
        with open(md_h, "w") as f:
            f.write("# MAE GHI por horizonte (test)\n\n")
            f.write(to_markdown_table(dfh_fmt))
            f.write("\n")
        print(f"[ok] Tabla GHI por horizonte guardada:\n- {csv_h}\n- {md_h}")

    # Quick console view (without checkpoint for readability)
    try:
        from tabulate import tabulate
        print("\n" + tabulate(df_fmt.drop(columns=["Checkpoint"]), headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print("\n" + df_fmt.drop(columns=["Checkpoint"]).to_string(index=False))

if __name__ == "__main__":
    main()
