# -*- coding: utf-8 -*-
# summarize_future_reports_m1.py
# Reads reports_future/<label>/summary.json and generates tables + comparative plots.

import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports_future"

LABELS = [
    ("gru_seg",  "GRU-SEG"),
    ("gru_ghi",  "GRU-GHI"),
    ("gru_bal",  "GRU-BAL"),
    ("lstm_seg", "LSTM-SEG"),
    ("lstm_ghi", "LSTM-GHI"),
    ("lstm_bal", "LSTM-BAL"),
]

def load_one(label: str):
    p = REPORTS / label / "summary.json"
    if not p.exists():
        return None, p
    with open(p, "r") as f:
        d = json.load(f)
    return d, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(REPORTS / "_summary"),
                    help="Carpeta donde guardar la tabla y figuras.")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    per_horizon = {}  # label -> dict(mae=[...], rmse=[...])

    for lab, pretty in LABELS:
        d, path = load_one(lab)
        if d is None:
            missing.append((lab, path))
            continue
        ghi = d["metrics"]["ghi"].get("overall", {})
        seg = d["metrics"]["seg"]
        row = dict(
            Modelo=lab, RNN=d.get("rnn_type","").upper() or ("GRU" if "gru" in lab else "LSTM"),
            MAE_Wm2 = float(ghi.get("mae", float("nan"))),
            RMSE_Wm2= float(ghi.get("rmse", float("nan"))),
            NMAE_pct= float(ghi.get("nmae_pct", float("nan"))),
            MBE_Wm2 = float(ghi.get("mbe", float("nan"))),
            PixelAcc = float(seg.get("pixel_acc", float("nan"))),
            Recall_SKY = float(seg.get("rec_sky", float("nan"))),
            Recall_THICK = float(seg.get("rec_thick", float("nan"))),
            Recall_THIN = float(seg.get("rec_thin", float("nan"))),
            Recall_SUN = float(seg.get("rec_sun", float("nan"))),
            mIoU = float(seg.get("miou", float("nan"))),
        )
        rows.append(row)

        ph = d["metrics"]["ghi"].get("per_horizon", None)
        if ph and "mae" in ph:
            per_horizon[lab] = dict(mae=np.array(ph["mae"], dtype=float),
                                    rmse=np.array(ph["rmse"], dtype=float))

    if missing:
        print("\n[aviso] Faltan resúmenes para:")
        for lab, p in missing:
            print(f"  - {lab:9s}  → falta {p}")
        print("\nGenera los que faltan, por ejemplo:")
        print("  python src/eval_future_models_m1.py --model gru_bal --csv data/processed/pairs_test.csv")
        print("  ...y vuelve a ejecutar este script.\n")

    if not rows:
        print("No hay ningún summary.json disponible. Nada que hacer.")
        return

    # -------- Table --------
    df = pd.DataFrame(rows).set_index("Modelo").loc[[lab for lab,_ in LABELS if any(r['Modelo']==lab for r in rows)]]
    csv_path = outdir / "summary_table.csv"
    md_path  = outdir / "summary_table.md"
    df.to_csv(csv_path, float_format="%.3f")
    # Pretty Markdown
    df_md = df.copy()
    for c in ["MAE_Wm2","RMSE_Wm2","MBE_Wm2"]: df_md[c] = df_md[c].map(lambda v: f"{v:.1f}")
    df_md["NMAE_pct"] = df_md["NMAE_pct"].map(lambda v: f"{v:.1f}")
    for c in ["PixelAcc","Recall_SKY","Recall_THICK","Recall_THIN","Recall_SUN","mIoU"]:
        df_md[c] = df_md[c].map(lambda v: f"{v:.3f}")
    with open(md_path, "w") as f:
        f.write("# Resumen modelos FUTURE (test)\n\n")
        f.write(df_md.to_markdown())
        f.write("\n")
    print(f"[tabla] CSV → {csv_path}")
    print(f"[tabla] MD  → {md_path}")
    print("\n", df.round(3))

    # -------- Figures --------
    # 1) MAE & RMSE
    order = df.index.tolist()
    x = np.arange(len(order))
    w = 0.36
    plt.figure(figsize=(9,4), dpi=120)
    plt.bar(x - w/2, df.loc[order,"MAE_Wm2"],  width=w, label="MAE (W/m²)")
    plt.bar(x + w/2, df.loc[order,"RMSE_Wm2"], width=w, label="RMSE (W/m²)")
    plt.xticks(x, order, rotation=15); plt.grid(True, axis="y", alpha=0.25)
    plt.title("GHI — MAE/RMSE por modelo"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir/"mae_rmse.png"); plt.close()
    print(f"[fig] {outdir/'mae_rmse.png'}")

    # 2) mIoU & PixelAcc
    plt.figure(figsize=(9,4), dpi=120)
    plt.bar(x - w/2, df.loc[order,"mIoU"],     width=w, label="mIoU")
    plt.bar(x + w/2, df.loc[order,"PixelAcc"], width=w, label="PixelAcc")
    plt.ylim(0,1); plt.xticks(x, order, rotation=15); plt.grid(True, axis="y", alpha=0.25)
    plt.title("SEG — mIoU / PixelAcc por modelo"); plt.legend()
    plt.tight_layout(); plt.savefig(outdir/"miou_pixelacc.png"); plt.close()
    print(f"[fig] {outdir/'miou_pixelacc.png'}")

    # 3) Recall per class (4 subplots)
    fig, axs = plt.subplots(1, 4, figsize=(13,3.6), dpi=120, sharey=True)
    classes = [("Recall_SKY","SKY"), ("Recall_THICK","THICK"),
               ("Recall_THIN","THIN"), ("Recall_SUN","SUN")]
    for i,(col,title) in enumerate(classes):
        axs[i].bar(x, df.loc[order,col])
        axs[i].set_title(title); axs[i].set_xticks(x); axs[i].set_xticklabels(order, rotation=45, ha="right")
        axs[i].set_ylim(0,1); axs[i].grid(True, axis="y", alpha=0.25)
    fig.suptitle("SEG — Recall por clase")
    fig.tight_layout(); fig.savefig(outdir/"recall_por_clase.png"); plt.close(fig)
    print(f"[fig] {outdir/'recall_por_clase.png'}")

    # 4) MAE per horizon (if present in the summaries)
    if per_horizon:
        plt.figure(figsize=(9,4), dpi=120)
        for lab,_ in LABELS:
            if lab in per_horizon:
                y = per_horizon[lab]["mae"]
                plt.plot(np.arange(len(y)), y, marker="o", label=lab)
        plt.xlabel("Horizonte t"); plt.ylabel("MAE (W/m²)")
        plt.title("GHI — MAE por horizonte"); plt.grid(True, alpha=0.25); plt.legend(ncol=2)
        plt.tight_layout(); plt.savefig(outdir/"ghi_mae_por_horizonte.png"); plt.close()
        print(f"[fig] {outdir/'ghi_mae_por_horizonte.png'}")

if __name__ == "__main__":
    main()
