"""
# ── GRU ─────────────────────────────────────────────
python src/eval_future_models_m1.py --model gru_seg --csv data/processed/pairs_test.csv
python src/eval_future_models_m1.py --model gru_ghi --csv data/processed/pairs_test.csv
python src/eval_future_models_m1.py --model gru_bal --csv data/processed/pairs_test.csv

# ── LSTM ────────────────────────────────────────────
python src/eval_future_models_m1.py --model lstm_seg --csv data/processed/pairs_test.csv
python src/eval_future_models_m1.py --model lstm_ghi --csv data/processed/pairs_test.csv
python src/eval_future_models_m1.py --model lstm_bal --csv data/processed/pairs_test.csv

"""
# eval_future_models_m1.py
# Evaluates FUTURE (SEG/GHI/BALANCED) on test and generates report + plots.

import os, sys, math, json, argparse, datetime as dt
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ==== Project modules ====
ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
sys.path.append(str(SRC))
import dataset_loader_m1 as dl
import models_m1 as models

# ===== Config consistent with the training runs =====
IMG_SIZE   = (320, 320)
NUM_CLASSES= 5
TIMESTEPS  = 6
STRIDE     = 6
ALPHA      = 0.5
OUTPUT_STRIDE = 16
BATCH_EVAL = 4         # batch for evaluation
SEED       = 1337

def set_policy_float32():
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

# ---------- MODEL REGISTRY ----------
REGISTRY = {
    # GRU
    "gru_seg": dict(rnn="gru",  pattern="*segphase_best_segloss.index",
                    desc="GRU — mejor segmentación (val_upseg_loss mínima)"),
    "gru_ghi": dict(rnn="gru",  pattern="*ghiphase_best.index",
                    desc="GRU — mejor GHI (val_ghi_pred_mae mínima)"),
    "gru_bal": dict(rnn="gru",  pattern="*balanced_best_mix.index",
                    desc="GRU — mejor compromiso SEG+GHI (val_loss mínima en balanced)"),
    "gru_last":dict(rnn="gru",  pattern="*_last.index",
                    desc="GRU — último checkpoint de la fase más reciente"),
    # LSTM
    "lstm_seg":dict(rnn="lstm", pattern="*segphase_best_segloss.index",
                    desc="LSTM — mejor segmentación (val_upseg_loss mínima)"),
    "lstm_ghi":dict(rnn="lstm", pattern="*ghiphase_best.index",
                    desc="LSTM — mejor GHI (val_ghi_pred_mae mínima)"),
    "lstm_bal":dict(rnn="lstm", pattern="*balanced_best_mix.index",
                    desc="LSTM — mejor compromiso SEG+GHI (val_loss mínima en balanced)"),
    "lstm_last":dict(rnn="lstm", pattern="*_last.index",
                    desc="LSTM — último checkpoint de la fase más reciente"),
}

def list_registry():
    print("\nModelos disponibles (etiquetas → descripción / patrón de ckpt):\n")
    for k, v in REGISTRY.items():
        print(f"  - {k:10s} → {v['desc']}\n      patrón: {v['pattern']}")
    print("\nTambién puedes pasar --ckpt /ruta/prefijo para evaluar un checkpoint concreto.\n")

# ---------- CHECKPOINT SEARCH ----------
def resolve_ckpt_from_label(label: str) -> (str, str):
    assert label in REGISTRY, f"Etiqueta '{label}' no reconocida. Usa --list."
    rnn   = REGISTRY[label]["rnn"]
    pat   = REGISTRY[label]["pattern"]
    outdir= ROOT / f"checkpoints_future_{rnn}"
    fs = sorted(outdir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not fs:
        raise FileNotFoundError(f"No se encontró '{pat}' en {outdir}")
    prefix = str(fs[0])[:-len(".index")]
    return prefix, rnn

# ---------- DATA UTILITIES ----------
def ensure_targets_SI(ds: tf.data.Dataset) -> tf.data.Dataset:
    if getattr(dl, "GHI_MODE", "si") == "normalized":
        @tf.function
        def _to_si(img, y):
            y2 = dict(y)
            y2["ghi_pred"] = dl._denormalize_ghi(y2["ghi_pred"])
            return img, y2
        return ds.map(_to_si, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def estimate_future_windows(csv_path: Path, timesteps: int, stride: int) -> int:
    df = pd.read_csv(csv_path).copy()
    if "dt" in df.columns:
        df["__dt"] = pd.to_datetime(df["dt"], errors="coerce")
    else:
        df["__dt"] = pd.to_datetime(df["stamp"], errors="coerce", format="%Y%m%d%H%M%S")
    df = df.sort_values("__dt").reset_index(drop=True)
    total = 0
    for _, g in df.groupby(df["__dt"].dt.date, sort=True):
        M = len(g)
        if M >= 2*timesteps:
            total += max(0, 1 + (M - 2*timesteps) // max(1, stride))
    return max(1, total)

def default_csv():
    for c in [ROOT/"data/processed/pairs_test.csv", ROOT/"data/processed/pairs_val.csv"]:
        if c.exists(): return str(c)
    raise FileNotFoundError("No se encontró ni pairs_test.csv ni pairs_val.csv")

# ---------- MODEL ----------
def build_future_for_eval(rnn_type: str) -> tf.keras.Model:
    _, fut = models.build_models_shared(
        timesteps=TIMESTEPS,
        img_height=IMG_SIZE[0], img_width=IMG_SIZE[1], img_channels=3,
        num_classes=NUM_CLASSES,
        rnn_type=rnn_type,
        d=models.D_REPR,
        output_stride=OUTPUT_STRIDE,
        mbv2_alpha=ALPHA,
        weights=None,
        learning_rate=1e-4,
        clipnorm=1.0,
        build_future_model=True
    )
    return fut

# ---------- SEG METRICS ----------
def make_seg_metrics():
    return dict(
        pixel_acc = models.PixelAccWithMask(name="pixel_acc"),
        rec_sky   = models.RecallForClassMasked(models.SKY_IDX,   name="recall_sky"),
        rec_thick = models.RecallForClassMasked(models.THICK_IDX, name="recall_thick"),
        rec_thin  = models.RecallForClassMasked(models.THIN_IDX,  name="recall_thin"),
        rec_sun   = models.RecallForClassMasked(models.SUN_IDX,   name="recall_sun"),
    )

def iou_inter_union(y_true, y_pred_probs):
    M = models.valid_mask(y_true)
    pred_lbl = tf.argmax(y_pred_probs, axis=-1)
    pred_onehot = tf.one_hot(pred_lbl, depth=tf.shape(y_true)[-1], dtype=y_true.dtype)
    y_true_m = y_true * M
    y_pred_m = pred_onehot * M
    inter = tf.reduce_sum(y_true_m * y_pred_m, axis=[0,1,2,3])
    union = (tf.reduce_sum(y_true_m, axis=[0,1,2,3]) +
             tf.reduce_sum(y_pred_m, axis=[0,1,2,3]) - inter)
    return inter, union

# ---------- TRAINING CURVES ----------
def find_trainlog_for_ckpt(ckpt_prefix: str) -> Path:
    p = Path(ckpt_prefix); d = p.parent; name = p.name
    key = "balanced" if "balanced" in name else ("ghiphase" if "ghiphase" in name else "segphase")
    cands = sorted(d.glob(f"*{key}*_train_log.csv"), key=lambda q: q.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def plot_training_curves(train_csv: Path, out_png: Path):
    try:
        df = pd.read_csv(train_csv)
    except Exception as e:
        print(f"[warn] No pude leer {train_csv}: {e}"); return
    plt.figure(figsize=(8,4), dpi=120)
    plotted = False
    for col in ["val_loss","val_ghi_pred_mae","val_upseg_loss","ghi_pred_mae","upseg_loss","loss"]:
        if col in df.columns:
            plt.plot(df.index.values, df[col].values, label=col.replace("_"," "))
            plotted = True
    if not plotted:
        print("[warn] CSV sin columnas esperables para curvas."); return
    plt.xlabel("Época"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("Curvas de entrenamiento/validación")
    plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print(f"[fig] Guardado: {out_png}")

# ---------- EVALUATION ----------
def run_eval(ckpt_prefix: str, rnn_type: str, csv_path: Path, report_dir: Path):
    set_policy_float32()
    dl.IMG_SIZE = tuple(IMG_SIZE)
    np.random.seed(SEED); tf.random.set_seed(SEED)

    report_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (no shuffle/augment, no repeat)
    ds = dl.build_future_dataset(
        csv_path=str(csv_path),
        timesteps=TIMESTEPS, stride=STRIDE,
        batch_size=BATCH_EVAL, seed_number=SEED,
        shuffle=False, cache_path=None, buffer_size=512,
        reshuffle_each_iteration_value=False,
        num_classes=NUM_CLASSES, augment=False,
        drop_remainder=False
    )
    ds = ensure_targets_SI(ds)

    # Model + weights
    model = build_future_for_eval(rnn_type)
    model.load_weights(ckpt_prefix)
    print(f"[ckpt] Cargado: {ckpt_prefix}")

    # Streaming SEG metrics
    seg_metrics = make_seg_metrics()
    C = NUM_CLASSES
    i_inter = np.zeros((C,), dtype=np.float64)
    i_union = np.zeros((C,), dtype=np.float64)

    # GHI accumulators (totals and per horizon)
    T = TIMESTEPS
    ghi_sum_abs = np.zeros((T,), dtype=np.float64)
    ghi_sum_sq  = np.zeros((T,), dtype=np.float64)
    ghi_sum_err = np.zeros((T,), dtype=np.float64)
    ghi_sum_gt  = np.zeros((T,), dtype=np.float64)
    ghi_count   = np.zeros((T,), dtype=np.int64)

    # --- Progress ---
    n_windows  = estimate_future_windows(csv_path, TIMESTEPS, STRIDE)
    n_batches  = int(math.ceil(n_windows / float(BATCH_EVAL)))
    print(f"[data] Ventanas≈{n_windows}  |  BatchEval={BATCH_EVAL}  |  Batches≈{n_batches}")

    use_tqdm = False
    try:
        from tqdm import tqdm
        bar = tqdm(total=n_batches, desc="Evaluando test", ncols=80)
        use_tqdm = True
    except Exception:
        prog = tf.keras.utils.Progbar(n_batches, unit_name="batch")
        step_shown = 0

    total_examples = 0

    for step, (x, y) in enumerate(ds.take(n_batches), start=1):
        # Prediction
        outs = model(x, training=False)
        if not isinstance(outs, (list, tuple)): outs = [outs]
        name2pred = {n: t for n, t in zip(model.output_names, outs)}
        y_seg = y.get("upseg", None)
        y_ghi = y.get("ghi_pred", None)

        upseg = name2pred.get("upseg", None)
        ghi_p = name2pred.get("ghi_pred", None)
        if upseg is None or ghi_p is None:
            a, b = outs
            if len(a.shape) == 5 and int(a.shape[-1]) == NUM_CLASSES: upseg, ghi_p = a, b
            else: upseg, ghi_p = b, a

        # --- Segmentation ---
        if y_seg is not None:
            for m in seg_metrics.values():
                m.update_state(y_seg, upseg)
            inter, uni = iou_inter_union(y_seg, upseg)
            i_inter += inter.numpy().astype(np.float64)
            i_union += uni.numpy().astype(np.float64)

        # --- GHI ---
        if y_ghi is not None:
            y_true = tf.reshape(y_ghi, (-1, T))     # (B,T)
            y_pred = tf.reshape(ghi_p, (-1, T))     # (B,T)
            err    = tf.abs(y_pred - y_true).numpy()
            sqe    = tf.square(y_pred - y_true).numpy()
            e      = (y_pred - y_true).numpy()
            gt     = y_true.numpy()
            ghi_sum_abs += err.sum(axis=0)
            ghi_sum_sq  += sqe.sum(axis=0)
            ghi_sum_err += e.sum(axis=0)
            ghi_sum_gt  += np.abs(gt).sum(axis=0)
            ghi_count   += err.shape[0]
            total_examples += err.shape[0]

        # progress
        if use_tqdm:
            bar.update(1)
        else:
            step_shown = step
            prog.update(step_shown)

    if use_tqdm:
        bar.close()

    # ---- SEG summary ----
    seg = {}
    for name, m in seg_metrics.items():
        seg[name] = float(m.result().numpy())
    class_names = dict([
        (models.SKY_IDX,  "sky"),
        (models.THICK_IDX,"thick"),
        (models.THIN_IDX, "thin"),
        (models.SUN_IDX,  "sun"),
    ])
    seg["iou_per_class"] = {}
    for c_idx, cname in class_names.items():
        if i_union[c_idx] > 0:
            seg["iou_per_class"][cname] = float(i_inter[c_idx] / i_union[c_idx])
        else:
            seg["iou_per_class"][cname] = None
    valid_ious = [v for v in seg["iou_per_class"].values() if v is not None]
    seg["miou"] = float(np.mean(valid_ious)) if valid_ious else None

    # ---- GHI summary ----
    ghi = {}
    tot = float(np.sum(ghi_count))
    if tot > 0:
        mae_t  = ghi_sum_abs / ghi_count.clip(min=1)
        rmse_t = np.sqrt(ghi_sum_sq / ghi_count.clip(min=1))
        mbe_t  = ghi_sum_err / ghi_count.clip(min=1)
        nmae_t = 100.0 * (ghi_sum_abs / (ghi_sum_gt.clip(min=1e-6)))
        ghi["per_horizon"] = {
            "mae":  [float(x) for x in mae_t],
            "rmse": [float(x) for x in rmse_t],
            "mbe":  [float(x) for x in mbe_t],
            "nmae_pct": [float(x) for x in nmae_t],
        }
        ghi["overall"] = {
            "mae":  float(np.sum(ghi_sum_abs) / tot),
            "rmse": float(np.sqrt(np.sum(ghi_sum_sq) / tot)),
            "mbe":  float(np.sum(ghi_sum_err) / tot),
            "nmae_pct": float(100.0 * np.sum(ghi_sum_abs) / np.sum(ghi_sum_gt.clip(min=1e-6))),
        }
    else:
        ghi["overall"] = {}

    # ---- Figures ----
    report_dir.mkdir(parents=True, exist_ok=True)
    if tot > 0:
        ts = np.arange(TIMESTEPS)
        plt.figure(figsize=(7.5,3.6), dpi=120)
        plt.bar(ts - 0.2, ghi["per_horizon"]["mae"],  width=0.4, label="MAE (W/m²)")
        plt.bar(ts + 0.2, ghi["per_horizon"]["rmse"], width=0.4, label="RMSE (W/m²)")
        plt.xticks(ts); plt.xlabel("Horizonte t"); plt.title("GHI error por horizonte")
        plt.grid(True, axis="y", alpha=0.25); plt.legend()
        path = report_dir/"ghi_horizonte.png"
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[fig] Guardado: {path}")

    bar_names, recall_vals, iou_vals = [], [], []
    for cname in ["sky","thick","thin","sun"]:
        rkey = f"rec_{cname}" if cname != "sun" else "rec_sun"
        if rkey in seg:
            bar_names.append(cname.upper())
            recall_vals.append(seg[rkey])
            iou_vals.append(seg["iou_per_class"].get(cname, None))
    if bar_names:
        plt.figure(figsize=(6.5,3.2), dpi=120)
        plt.bar(range(len(bar_names)), recall_vals)
        plt.xticks(range(len(bar_names)), bar_names)
        plt.ylim(0,1); plt.grid(True, axis="y", alpha=0.25)
        plt.title("Recall por clase (segmentación)")
        path = report_dir/"seg_recall.png"
        plt.tight_layout(); plt.savefig(path); plt.close()
        print(f"[fig] Guardado: {path}")
        if any(v is not None for v in iou_vals):
            vals = [0.0 if v is None else v for v in iou_vals]
            plt.figure(figsize=(6.5,3.2), dpi=120)
            plt.bar(range(len(bar_names)), vals)
            plt.xticks(range(len(bar_names)), bar_names)
            plt.ylim(0,1); plt.grid(True, axis="y", alpha=0.25)
            plt.title("IoU por clase (segmentación)")
            path = report_dir/"seg_iou.png"
            plt.tight_layout(); plt.savefig(path); plt.close()
            print(f"[fig] Guardado: {path}")

    tl = find_trainlog_for_ckpt(ckpt_prefix)
    if tl is not None:
        plot_training_curves(tl, report_dir/"train_curves.png")

    # ---- Summary to disk ----
    summary = {
        "ckpt_prefix": ckpt_prefix,
        "rnn_type": rnn_type,
        "img_size": list(IMG_SIZE),
        "timesteps": TIMESTEPS,
        "stride": STRIDE,
        "batch_eval": BATCH_EVAL,
        "examples_eval": int(total_examples),
        "when": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "ghi": ghi,
            "seg": seg
        }
    }
    with open(report_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = []
    md.append(f"# Informe: {Path(ckpt_prefix).name}")
    md.append(f"- Arquitectura: **{rnn_type.upper()}**")
    if ghi.get("overall"):
        md.append(f"- **GHI** → MAE: **{ghi['overall']['mae']:.1f} W/m²**, "
                  f"RMSE: **{ghi['overall']['rmse']:.1f} W/m²**, "
                  f"MBE: **{ghi['overall']['mbe']:.1f} W/m²**, "
                  f"NMAE: **{ghi['overall']['nmae_pct']:.1f}%**")
    md.append(f"- **SEG** → PixelAcc(máscara): **{seg.get('pixel_acc',float('nan')):.3f}**  "
              f"| Recall SKY: **{seg.get('rec_sky',float('nan')):.3f}**, "
              f"THICK: **{seg.get('rec_thick',float('nan')):.3f}**, "
              f"THIN: **{seg.get('rec_thin',float('nan')):.3f}**, "
              f"SUN: **{seg.get('rec_sun',float('nan')):.3f}**  "
              f"| mIoU: **{(seg.get('miou') if seg.get('miou') is not None else float('nan')):.3f}**")
    md.append("")
    md.append("![GHI por horizonte](ghi_horizonte.png)")
    if (report_dir/"seg_recall.png").exists():
        md.append("![Recall por clase](seg_recall.png)")
    if (report_dir/"seg_iou.png").exists():
        md.append("![IoU por clase](seg_iou.png)")
    if (report_dir/"train_curves.png").exists():
        md.append("![Curvas de entrenamiento](train_curves.png)")
    with open(report_dir/"summary.md", "w") as f:
        f.write("\n".join(md))

    print(f"[ok] Reporte guardado en: {report_dir}")
    print(json.dumps(summary["metrics"]["ghi"].get("overall", {}), indent=2))
    print(json.dumps(summary["metrics"]["seg"], indent=2))

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="Muestra las etiquetas disponibles y sale.")
    parser.add_argument("--model", type=str, default=None, help="Etiqueta del REGISTRY (p.ej., gru_bal).")
    parser.add_argument("--ckpt",  type=str, default=None, help="Prefijo de checkpoint a cargar (sobre-escribe --model).")
    parser.add_argument("--csv",   type=str, default=None, help="CSV de test (por defecto pairs_test.csv).")
    parser.add_argument("--tag",   type=str, default=None, help="Nombre de carpeta dentro de reports_future/ (por defecto = etiqueta o nombre de ckpt).")
    args = parser.parse_args()

    if args.list:
        list_registry(); return

    if args.ckpt:
        ckpt_prefix = args.ckpt.rstrip(".index")
        rnn = "gru" if "/future_gru/" in ckpt_prefix or "GRU_" in ckpt_prefix.upper() else \
              ("lstm" if "/future_lstm/" in ckpt_prefix or "LSTM_" in ckpt_prefix.upper() else None)
        if rnn is None:
            raise SystemExit("No pude inferir RNN (gru/lstm) desde --ckpt. Pon --model en su lugar o ajusta la ruta.")
        label = args.tag or Path(ckpt_prefix).name
    else:
        assert args.model in REGISTRY, "Debes pasar --model válido o --ckpt."
        ckpt_prefix, rnn = resolve_ckpt_from_label(args.model)
        label = args.tag or args.model

    csv_path = Path(args.csv) if args.csv else Path(default_csv())
    out_dir  = ROOT / "reports_future" / label
    print(f"[eval] etiqueta='{label}'  rnn={rnn.upper()}  ckpt={ckpt_prefix}")
    print(f"[data] CSV={csv_path}")

    run_eval(ckpt_prefix, rnn, csv_path, out_dir)

if __name__ == "__main__":
    main()
