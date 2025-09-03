# -*- coding: utf-8 -*-# make_future_figs_m1.py
# Generates comparative PNGs and gathers per-model PNGs in ROOT/figuras_future.

import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports_future"
OUTDIR_DEFAULT = ROOT / "figuras_future"

DEFAULT_LABELS = ["gru_seg","gru_ghi","gru_bal","lstm_seg","lstm_ghi","lstm_bal"]

def load_summary(label: str):
    p = REPORT / label / "summary.json"
    if not p.exists(): return None
    with open(p, "r") as f: s = json.load(f)
    seg = s["metrics"]["seg"]; ghi = s["metrics"]["ghi"]
    over = ghi.get("overall", {})
    ph   = ghi.get("per_horizon", {})
    return dict(
        label=label,
        rnn=s.get("rnn_type","?").upper(),
        mae=over.get("mae", np.nan),
        rmse=over.get("rmse", np.nan),
        mbe=over.get("mbe", np.nan),
        nmae=over.get("nmae_pct", np.nan),
        pixacc=seg.get("pixel_acc", np.nan),
        miou=seg.get("miou", np.nan),
        rec_sky=seg.get("rec_sky", np.nan),
        rec_thick=seg.get("rec_thick", np.nan),
        rec_thin=seg.get("rec_thin", np.nan),
        rec_sun=seg.get("rec_sun", np.nan),
        mae_t=ph.get("mae", []),
        rmse_t=ph.get("rmse", []),
    )

def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"[fig] {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(OUTDIR_DEFAULT),
                    help="Carpeta de salida para PNGs (por defecto ROOT/figuras_future).")
    ap.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS),
                    help="Lista de etiquetas separadas por comas.")
    ap.add_argument("--gather-per-model", action="store_true",
                    help="Copia también los PNG por-modelo a la carpeta de salida.")
    args = ap.parse_args()

    out_dir = Path(args.out)

    # Labels to use
    asked = [x.strip() for x in args.labels.split(",") if x.strip()]
    labels = [lab for lab in asked if (REPORT/lab/"summary.json").exists()]
    if not labels:
        # fallback: all folders with summary.json
        labels = [d.name for d in REPORT.iterdir()
                  if d.is_dir() and not d.name.startswith("_") and (d/"summary.json").exists()]
        labels.sort()
    if not labels:
        print("No encontré summaries. Ejecuta primero eval_future_suite_m1.py."); return

    rows = [load_summary(l) for l in labels]
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)

    # 1) GHI — Global MAE per model
    order = df.sort_values("mae").index
    fig = plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(df)), df.loc[order,"mae"])
    plt.xticks(np.arange(len(df)), df.loc[order,"label"], rotation=15)
    plt.ylabel("MAE (W/m²)"); plt.title("GHI — MAE global (test)"); plt.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "ghi_mae_by_model.png")

    # 2) GHI — Global RMSE per model
    fig = plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(df)), df.loc[order,"rmse"])
    plt.xticks(np.arange(len(df)), df.loc[order,"label"], rotation=15)
    plt.ylabel("RMSE (W/m²)"); plt.title("GHI — RMSE global (test)"); plt.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "ghi_rmse_by_model.png")

    # 3) GHI — MAE per horizon (comparative lines)
    if any(len(x)>0 for x in df["mae_t"]):
        T = max(len(x) for x in df["mae_t"])
        ts = np.arange(T)
        fig = plt.figure(figsize=(10,4))
        for _, r in df.iterrows():
            if len(r["mae_t"]) == 0: continue
            plt.plot(ts, r["mae_t"], marker="o", label=r["label"])
        plt.xlabel("Horizonte t"); plt.ylabel("MAE (W/m²)")
        plt.title("GHI — MAE por horizonte (test)"); plt.grid(True, alpha=0.3); plt.legend()
        save_fig(fig, out_dir, "ghi_mae_per_horizon.png")

    # 4) Segmentation — PixelAcc per model
    fig = plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(df)), df.loc[order,"pixacc"])
    plt.xticks(np.arange(len(df)), df.loc[order,"label"], rotation=15)
    plt.ylim(0,1); plt.ylabel("PixelAcc"); plt.title("Segmentación — PixelAcc (test)")
    plt.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "seg_pixelacc_by_model.png")

    # 5) Segmentation — mIoU per model
    fig = plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(df)), df.loc[order,"miou"])
    plt.xticks(np.arange(len(df)), df.loc[order,"label"], rotation=15)
    plt.ylim(0,1); plt.ylabel("mIoU"); plt.title("Segmentación — mIoU (test)")
    plt.grid(axis="y", alpha=0.3)
    save_fig(fig, out_dir, "seg_miou_by_model.png")

    # 6) Segmentation — Recall per class (grouped)
    width = 0.2; x = np.arange(len(df))
    fig = plt.figure(figsize=(10,4))
    plt.bar(x-1.5*width, df["rec_sky"],   width, label="SKY")
    plt.bar(x-0.5*width, df["rec_thick"], width, label="THICK")
    plt.bar(x+0.5*width, df["rec_thin"],  width, label="THIN")
    plt.bar(x+1.5*width, df["rec_sun"],   width, label="SUN")
    plt.xticks(x, df["label"], rotation=15); plt.ylim(0,1); plt.ylabel("Recall")
    plt.title("Segmentación — Recall por clase (test)"); plt.grid(axis="y", alpha=0.3); plt.legend()
    save_fig(fig, out_dir, "seg_recall_by_class.png")

    # Copy training curves to the same folder
    copied = []
    for lab in labels:
        src = REPORT / lab / "train_curves.png"
        if src.exists():
            dst = out_dir / f"train_curves_{lab}.png"
            shutil.copy2(src, dst)
            copied.append(dst)
    print(f"[info] Curvas de entrenamiento copiadas: {len(copied)}")

    # (optional) also copy per-model PNGs (GHI per horizon, recall, IoU)
    if args.gather_per_model:
        for lab in labels:
            for name in ["ghi_horizonte.png","seg_recall.png","seg_iou.png"]:
                src = REPORT / lab / name
                if src.exists():
                    dst = out_dir / f"{lab}__{name}"
                    shutil.copy2(src, dst)
        print("[info] PNG por-modelo copiados a la carpeta de salida.")

    print(f"[ok] PNG listos en: {out_dir}")

if __name__ == "__main__":
    main()
