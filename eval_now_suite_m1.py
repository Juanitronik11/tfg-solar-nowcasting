# -*- coding: utf-8 -*-
# eval_now_suite_m1.py
# Launches the evaluation of the 2 NOW models and compiles tables + copies figures.
#
# Usage:
#   # Run evals + create tables and copy PNGs
#   python src/eval_now_suite_m1.py --csv data/processed/pairs_test.csv
#
#   # Only compile tables/copy figures from existing reports (no re-evaluation)
#   python src/eval_now_suite_m1.py --csv data/processed/pairs_test.csv --skip-eval
#
#   # Customize directories and labels
#   python src/eval_now_suite_m1.py --csv data/processed/pairs_test.csv \
#       --dirs checkpoints_now8h,checkpoints_now8h_seg \
#       --tags now8h,now8hSEG
#
# Output:
#   reports_now/now8h/[…png|summary.json|summary.md]
#   reports_now/now8hSEG/[…png|summary.json|summary.md]
#   reports_now/_figuras/<ts>/* (renamed copies for LaTeX)
#   reports_now/_tables/summary_<ts>.{csv,md}

import os, sys, json, argparse, datetime as dt, subprocess, shutil
from pathlib import Path
import pandas as pd
import numpy as np

# ── tqdm (progress) ────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kw):
        return iterable if iterable is not None else range(0)

ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
EVAL_SCRIPT = SRC / "eval_now_model_m1.py"  # Single-model evaluation script

# By default: two NOW variants
DEFAULT_DIRS = [ROOT/"checkpoints_now8h", ROOT/"checkpoints_now8h_seg"]
DEFAULT_TAGS = ["now8h", "now8hSEG"]

# ---------- Helpers ----------
def run_eval(dir_path: Path, csv_path: Path, tag: str) -> Path:
    """Runs the NOW evaluator for a directory and returns the path to summary.json."""
    out_dir = ROOT / "reports_now" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = out_dir / "summary.json"

    cmd = [sys.executable, str(EVAL_SCRIPT), "--csv", str(csv_path), "--dir", str(dir_path), "--tag", tag]
    subprocess.run(cmd, check=True)
    if not summary_json.exists():
        raise FileNotFoundError(f"No se generó {summary_json}. Revisa la evaluación de {tag}.")
    return summary_json

def load_metrics(summary_json: Path) -> dict:
    with open(summary_json, "r") as f:
        s = json.load(f)
    seg   = s["metrics"]["seg"]
    speed = s.get("speed", {})
    return dict(
        tag      = summary_json.parent.name,
        pixelacc = seg.get("pixel_acc", np.nan),
        rec_sky  = seg.get("rec_sky", np.nan),
        rec_thick= seg.get("rec_thick", np.nan),
        rec_thin = seg.get("rec_thin", np.nan),
        rec_sun  = seg.get("rec_sun", np.nan),
        miou     = seg.get("miou", np.nan),
        iou_sky  = seg.get("iou_per_class",{}).get("sky", np.nan),
        iou_thick= seg.get("iou_per_class",{}).get("thick", np.nan),
        iou_thin = seg.get("iou_per_class",{}).get("thin", np.nan),
        iou_sun  = seg.get("iou_per_class",{}).get("sun", np.nan),
        ms_per_img = speed.get("ms_per_img", np.nan),
        fps        = speed.get("fps", np.nan),
    )

def to_markdown_table(df: pd.DataFrame) -> str:
    cols = df.columns
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"]*len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if pd.isna(v): vals.append("")
                elif "ms" in c or "fps" in c: vals.append(f"{v:.1f}")
                elif "IoU" in c or "Recall" in c or "PixelAcc" in c or c=="mIoU":
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def copy_figures(tag: str, ts_dir: Path):
    """Copies key PNGs to _figuras/<ts>/ with prefix <tag>__*.png."""
    src_dir = ROOT / "reports_now" / tag
    if not src_dir.exists(): return []

    wanted = [
        "now_seg_recall.png",
        "now_seg_iou.png",
        "now_confusion.png",
        "now_train_curves.png",
    ]
    copied = []
    for w in wanted:
        src = src_dir / w
        if src.exists():
            dst = ts_dir / f"{tag}__{w}"
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Test CSV (pairs_test.csv).")
    ap.add_argument("--dirs", type=str, default=",".join(map(str, DEFAULT_DIRS)),
                    help="Comma-separated list of NOW checkpoint folders.")
    ap.add_argument("--tags", type=str, default=",".join(DEFAULT_TAGS),
                    help="Comma-separated list of labels for reports_now/ (aligned with --dirs).")
    ap.add_argument("--skip-eval", action="store_true", help="Do not re-evaluate; only read summaries and copy PNG.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    assert csv_path.exists(), f"No existe CSV: {csv_path}"

    dir_list = [Path(x.strip()) for x in args.dirs.split(",") if x.strip()]
    tag_list = [x.strip() for x in args.tags.split(",") if x.strip()]
    assert len(dir_list) == len(tag_list) >= 1, "--dirs y --tags deben tener el mismo número de elementos."

    # output folders (tables + figures)
    tables_dir = ROOT / "reports_now" / "_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    figs_dir = ROOT / "reports_now" / "_figuras" / ts
    figs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    # ── Progress: model evaluation ─────────────────────────────────────────────
    if not args.skip_eval:
        with tqdm(total=len(dir_list), desc="Evaluando modelos NOW", unit="modelo", ncols=100) as pbar:
            for d, tag in zip(dir_list, tag_list):
                pbar.set_postfix_str(tag, refresh=True)
                sjson = run_eval(d, csv_path, tag)
                summaries.append(load_metrics(sjson))
                pbar.update(1)
    else:
        for tag in tqdm(tag_list, desc="Leyendo summaries NOW", unit="modelo", ncols=100):
            sjson = ROOT / "reports_now" / tag / "summary.json"
            if not sjson.exists():
                print(f"[warn] No existe {sjson}; salta {tag}.")
                continue
            summaries.append(load_metrics(sjson))

    if not summaries:
        print("No hay summaries para compilar tablas."); return

    # ── Main table ─────────────────────────────────────────────────────────────
    rows = []
    for s in summaries:
        rows.append(dict(
            Modelo = s["tag"],
            PixelAcc = s["pixelacc"],
            Recall_SKY = s["rec_sky"],
            Recall_THICK = s["rec_thick"],
            Recall_THIN = s["rec_thin"],
            Recall_SUN = s["rec_sun"],
            mIoU = s["miou"],
            IoU_SKY = s["iou_sky"],
            IoU_THICK = s["iou_thick"],
            IoU_THIN = s["iou_thin"],
            IoU_SUN = s["iou_sun"],
            ms_per_img = s["ms_per_img"],
            FPS = s["fps"],
        ))
    df = pd.DataFrame(rows)

    # Save CSV and Markdown
    csv_out = tables_dir / f"summary_{ts}.csv"
    md_out  = tables_dir / f"summary_{ts}.md"
    df.to_csv(csv_out, index=False)

    # Markdown formatting
    df_md = df.copy()
    for c in ["PixelAcc","Recall_SKY","Recall_THICK","Recall_THIN","Recall_SUN","mIoU",
              "IoU_SKY","IoU_THICK","IoU_THIN","IoU_SUN"]:
        df_md[c] = df_md[c].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    for c in ["ms_per_img","FPS"]:
        df_md[c] = df_md[c].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "")
    with open(md_out, "w") as f:
        f.write("# Resumen modelos NOW (test)\n\n")
        f.write(to_markdown_table(df_md))

    print(f"[ok] Tablas guardadas:\n- {csv_out}\n- {md_out}")

    # ── Copy key figures to _figuras/<ts>/ ────────────────────────────────────
    copied_all = []
    for tag in tqdm(tag_list, desc="Copiando figuras NOW", unit="modelo", ncols=100):
        copied_all += copy_figures(tag, figs_dir)

    # Quick figures index
    idx = ROOT / "reports_now" / "_figuras" / f"index_{ts}.md"
    with open(idx, "w") as f:
        f.write("# Figuras NOW (para LaTeX)\n\n")
        for p in sorted(copied_all):
            f.write(f"- {p.name}\n")
    print(f"[ok] Figuras copiadas en: {figs_dir}\n[ok] Índice: {idx}")

    # Console view
    try:
        from tabulate import tabulate
        print("\n" + tabulate(df_md, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print("\n" + df_md.to_string(index=False))

if __name__ == "__main__":
    main()
