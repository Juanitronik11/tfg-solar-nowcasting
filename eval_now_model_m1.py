"""
caffeinate -d -i python src/eval_now_model_m1.py --csv data/processed/pairs_test.csv \
  --dir checkpoints_now8h,checkpoints_now8h_seg
"""


# src/eval_now_model_m1.py
# Evaluates one or multiple NOW models (segmentation) on test: metrics + figures + speed.
# - Supports --dir with a comma-separated list (e.g., checkpoints_now8h,checkpoints_now8h_seg)
# - Or --ckpt with one or several checkpoint prefixes (comma-separated)
# - Generates per-model reports in reports_now/<tag>/
# - If you evaluate >1, it compiles global tables and copies all PNGs to reports_now/_figuras/<timestamp>/
#
# Examples:
#   python src/eval_now_model_m1.py --csv data/processed/pairs_test.csv \
#       --dir checkpoints_now8h,checkpoints_now8h_seg
#
#   # Only one
#   python src/eval_now_model_m1.py --csv data/processed/pairs_test.csv --dir checkpoints_now8h
#   # Or by specific checkpoint
#   python src/eval_now_model_m1.py --csv data/processed/pairs_test.csv \
#       --ckpt checkpoints_now8h/now8h_..._best_segloss

import os, sys, math, json, argparse, datetime as dt, shutil
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# ==== Project modules ====
ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
sys.path.append(str(SRC))
import dataset_loader_m1 as dl
import models_m1 as models

# ===== Config (consistent with trainings) =====
IMG_SIZE    = (320, 320)
NUM_CLASSES = 5
TIMESTEPS   = 6
BATCH_EVAL  = 8
SEED        = 1337

def set_policy_float32():
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

def _count_csv_rows(csv_path: Path) -> int:
    try:
        with open(csv_path, "r") as f:
            n = sum(1 for _ in f)
        return max(0, n-1)
    except Exception:
        return 0

def ensure_targets_SI(ds: tf.data.Dataset) -> tf.data.Dataset:
    if getattr(dl, "GHI_MODE", "si") == "normalized":
        @tf.function
        def _to_si(img, y):
            y2 = dict(y)
            if "ghi_pred" in y2:
                y2["ghi_pred"] = dl._denormalize_ghi(y2["ghi_pred"])
            return img, y2
        return ds.map(_to_si, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def build_now_for_eval() -> tf.keras.Model:
    now, *_ = models.build_models_shared(
        timesteps=TIMESTEPS,
        img_height=IMG_SIZE[0], img_width=IMG_SIZE[1], img_channels=3,
        num_classes=NUM_CLASSES,
        rnn_type="gru",
        d=models.D_REPR,
        output_stride=16,
        mbv2_alpha=0.5,
        weights=None,
        learning_rate=1e-4,
        clipnorm=1.0,
        build_future_model=False
    )
    return now

def make_now_dataset(csv_path: Path, batch: int):
    if hasattr(dl, "build_now_dataset"):
        ds = dl.build_now_dataset(
            csv_path=str(csv_path),
            batch_size=batch, seed_number=SEED,
            shuffle=False, cache_path=None, buffer_size=512,
            reshuffle_each_iteration_value=False,
            num_classes=NUM_CLASSES, augment=False,
            drop_remainder=False
        )
        return ensure_targets_SI(ds)
    # Fallback: FUTURE with T=1
    ds = dl.build_future_dataset(
        csv_path=str(csv_path),
        timesteps=1, stride=1,
        batch_size=batch, seed_number=SEED,
        shuffle=False, cache_path=None, buffer_size=512,
        reshuffle_each_iteration_value=False,
        num_classes=NUM_CLASSES, augment=False,
        drop_remainder=False
    )
    return ensure_targets_SI(ds)

# --------- Seg metrics + IoU + confusion ----------
def make_seg_metrics():
    return dict(
        pixel_acc = models.PixelAccWithMask(name="pixel_acc"),
        rec_sky   = models.RecallForClassMasked(models.SKY_IDX,   name="rec_sky"),
        rec_thick = models.RecallForClassMasked(models.THICK_IDX, name="rec_thick"),
        rec_thin  = models.RecallForClassMasked(models.THIN_IDX,  name="rec_thin"),
        rec_sun   = models.RecallForClassMasked(models.SUN_IDX,   name="rec_sun"),
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

def confusion_accumulators(num_classes: int):
    return np.zeros((num_classes, num_classes), dtype=np.int64)

def update_confmat(conf, y_true, y_pred_probs):
    M = models.valid_mask(y_true)  # (B,T,H,W,1)
    y_true_lbl = tf.argmax(y_true, axis=-1)   # (B,T,H,W)
    y_pred_lbl = tf.argmax(y_pred_probs, axis=-1)
    valid = tf.cast(tf.reduce_any(M > 0, axis=-1), tf.bool)  # (B,T,H,W)
    yt = tf.boolean_mask(y_true_lbl, valid).numpy().ravel()
    yp = tf.boolean_mask(y_pred_lbl, valid).numpy().ravel()
    for t, p in zip(yt, yp):
        if 0 <= t < conf.shape[0] and 0 <= p < conf.shape[1]:
            conf[t, p] += 1

# --------- Training curves ----------
def find_trainlog_for_now(ckpt_prefix: str) -> Path:
    p = Path(ckpt_prefix); d = p.parent
    cands = sorted(list(d.glob("*now*train_log.csv")) + list(d.glob("*seg*_train_log.csv")),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def plot_training_curves(train_csv: Path, out_png: Path):
    try:
        df = pd.read_csv(train_csv)
    except Exception as e:
        print(f"[warn] No pude leer {train_csv}: {e}"); return
    plt.figure(figsize=(8,4), dpi=120)
    plotted = False
    for col in ["val_upseg_loss","upseg_loss","loss","val_loss"]:
        if col in df.columns:
            plt.plot(df.index.values, df[col].values, label=col.replace("_"," "))
            plotted = True
    if not plotted:
        print("[warn] CSV sin columnas esperables para curvas."); return
    plt.xlabel("Época"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("Curvas NOW (entrenamiento/validación)")
    plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print(f"[fig] Guardado: {out_png}")

# --------- Speed ----------
def measure_speed(model, ds, batches=30):
    import time
    it = iter(ds)
    # warmup
    for _ in range(3):
        try:
            x, _ = next(it)
        except StopIteration:
            return dict(ms_per_img=np.nan, fps=np.nan)
        _ = model(x, training=False)
    times, n_imgs = [], 0
    for _ in range(batches):
        try:
            x, _ = next(it)
        except StopIteration:
            break
        t0 = time.time()
        _ = model(x, training=False)
        dt = time.time() - t0
        times.append(dt)
        n_imgs += int(x.shape[0])
    if not times or n_imgs == 0:
        return dict(ms_per_img=np.nan, fps=np.nan)
    ms_per_img = 1000.0 * np.mean(times) / (n_imgs / len(times))
    fps = 1000.0 / ms_per_img
    return dict(ms_per_img=float(ms_per_img), fps=float(fps))

# --------- Checkpoint resolution ----------
def resolve_ckpt(dir_path: Path):
    patterns = ["*best_segloss.index", "*best.index", "*last.index"]
    for pat in patterns:
        fs = sorted(dir_path.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if fs:
            return str(fs[0])[:-len(".index")]
    raise FileNotFoundError(f"No se encontró checkpoint en {dir_path}")

# --------- Single-model evaluation ----------
def run_eval_now(ckpt_prefix: str, csv_path: Path, report_dir: Path):
    set_policy_float32()
    dl.IMG_SIZE = tuple(IMG_SIZE)
    np.random.seed(SEED); tf.random.set_seed(SEED)

    report_dir.mkdir(parents=True, exist_ok=True)

    ds = make_now_dataset(csv_path, BATCH_EVAL)
    print(f"[data] CSV={csv_path}")

    # Estimated number of batches for the progress bar
    n_rows = _count_csv_rows(csv_path)
    n_batches = int(math.ceil(max(1, n_rows) / float(BATCH_EVAL)))

    # Model + weights
    model = build_now_for_eval()
    model.load_weights(ckpt_prefix)
    print(f"[ckpt] Cargado: {ckpt_prefix}")

    seg_metrics = make_seg_metrics()
    C = NUM_CLASSES
    i_inter = np.zeros((C,), dtype=np.float64)
    i_union = np.zeros((C,), dtype=np.float64)
    confmat = confusion_accumulators(C)

    bar = tqdm(total=n_batches, desc="Evaluando NOW (test)", ncols=100) if tqdm else None

    for step, (x, y) in enumerate(ds.take(n_batches), start=1):
        outs = model(x, training=False)
        if isinstance(outs, (list, tuple)):
            cand = None
            for t in outs:
                if len(t.shape) in (4,5) and int(t.shape[-1]) == NUM_CLASSES:
                    cand = t; break
            upseg = cand if cand is not None else outs[0]
        else:
            upseg = outs
        if len(upseg.shape) == 4:                 # (B,H,W,C) -> (B,1,H,W,C)
            upseg = tf.expand_dims(upseg, axis=1)

        # Segmentation labels
        y_seg = None
        if isinstance(y, dict):
            y_seg = y.get("upseg", None)
            if y_seg is None and "seg" in y:
                y_seg = y["seg"]
        if y_seg is None:
            raise RuntimeError("No encontré etiquetas de segmentación en el dataset NOW.")
        if len(y_seg.shape) == 4:                 # (B,H,W,C) -> (B,1,H,W,C) to match
            y_seg = tf.expand_dims(y_seg, axis=1)

        # Metrics
        for m in seg_metrics.values():
            m.update_state(y_seg, upseg)
        inter, uni = iou_inter_union(y_seg, upseg)
        i_inter += inter.numpy().astype(np.float64)
        i_union += uni.numpy().astype(np.float64)
        update_confmat(confmat, y_seg, upseg)

        if bar: bar.update(1)

    if bar: bar.close()

    # Speed
    speed = measure_speed(model, make_now_dataset(csv_path, BATCH_EVAL), batches=30)

    # ---- SEG summary ----
    seg = {}
    for name, m in seg_metrics.items():
        seg[name] = float(m.result().numpy())

    cls_names = {models.SKY_IDX:"sky", models.THICK_IDX:"thick",
                 models.THIN_IDX:"thin", models.SUN_IDX:"sun"}
    seg["iou_per_class"] = {}
    for c_idx, cname in cls_names.items():
        seg["iou_per_class"][cname] = float(i_inter[c_idx]/i_union[c_idx]) if i_union[c_idx] > 0 else None
    vals = [v for v in seg["iou_per_class"].values() if v is not None]
    seg["miou"] = float(np.mean(vals)) if vals else None

    # ---- Figures ----
    names = ["SKY","THICK","THIN","SUN"]
    recalls = [seg.get("rec_sky",np.nan), seg.get("rec_thick",np.nan),
               seg.get("rec_thin",np.nan), seg.get("rec_sun",np.nan)]
    ious = [seg["iou_per_class"].get("sky"), seg["iou_per_class"].get("thick"),
            seg["iou_per_class"].get("thin"), seg["iou_per_class"].get("sun")]

    plt.figure(figsize=(6.5,3.2), dpi=120)
    plt.bar(range(len(names)), recalls)
    plt.xticks(range(len(names)), names)
    plt.ylim(0,1); plt.grid(True, axis="y", alpha=0.25)
    plt.title("NOW – Recall por clase")
    p = report_dir/"now_seg_recall.png"
    plt.tight_layout(); plt.savefig(p); plt.close()
    print(f"[fig] Guardado: {p}")

    if any(v is not None for v in ious):
        vals = [0 if v is None else v for v in ious]
        plt.figure(figsize=(6.5,3.2), dpi=120)
        plt.bar(range(len(names)), vals)
        plt.xticks(range(len(names)), names)
        plt.ylim(0,1); plt.grid(True, axis="y", alpha=0.25)
        plt.title("NOW – IoU por clase")
        p = report_dir/"now_seg_iou.png"
        plt.tight_layout(); plt.savefig(p); plt.close()
        print(f"[fig] Guardado: {p}")

    # Confusion matrix (optionally remove VOID)
    cm = confmat.copy()
    try:
        void_idx = int(models.VOID_IDX)
        if 0 <= void_idx < cm.shape[0]:
            cm = np.delete(cm, void_idx, axis=0)
            cm = np.delete(cm, void_idx, axis=1)
    except Exception:
        pass
    plt.figure(figsize=(5.2,4.6), dpi=120)
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("NOW – Matriz de confusión")
    plt.xlabel("Predicho"); plt.ylabel("Real")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    p = report_dir/"now_confusion.png"
    plt.tight.tight_layout(); plt.savefig(p); plt.close()
    print(f"[fig] Guardado: {p}")

    # Training curves (if present)
    tl = find_trainlog_for_now(ckpt_prefix)
    if tl is not None:
        plot_training_curves(tl, report_dir/"now_train_curves.png")

    # ---- Summary to disk ----
    summary = {
        "ckpt_prefix": ckpt_prefix,
        "img_size": list(IMG_SIZE),
        "timesteps_for_eval": TIMESTEPS,
        "batch_eval": BATCH_EVAL,
        "speed": speed,
        "when": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": { "seg": seg }
    }
    with open(report_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Short Markdown
    md = []
    md.append(f"# Informe NOW: {Path(ckpt_prefix).name}")
    md.append(f"- **PixelAcc(máscara)**: **{seg.get('pixel_acc',float('nan')):.3f}**")
    md.append(f"- **Recall** SKY: **{seg.get('rec_sky',float('nan')):.3f}**, "
              f"THICK: **{seg.get('rec_thick',float('nan')):.3f}**, "
              f"THIN: **{seg.get('rec_thin',float('nan')):.3f}**, "
              f"SUN: **{seg.get('rec_sun',float('nan')):.3f}**  "
              f"| **mIoU**: **{(seg.get('miou') if seg.get('miou') is not None else float('nan')):.3f}**")
    if not any(np.isnan([summary['speed'].get("ms_per_img", np.nan)])):
        md.append(f"- **Velocidad**: ~{summary['speed']['ms_per_img']:.1f} ms/imagen  (~{summary['speed']['fps']:.1f} FPS)")
    md.append("")
    md.append("![Recall por clase](now_seg_recall.png)")
    if (report_dir/"now_seg_iou.png").exists():
        md.append("![IoU por clase](now_seg_iou.png)")
    if (report_dir/"now_confusion.png").exists():
        md.append("![Matriz de confusión](now_confusion.png)")
    if (report_dir/"now_train_curves.png").exists():
        md.append("![Curvas NOW](now_train_curves.png)")
    with open(report_dir/"summary.md", "w") as f:
        f.write("\n".join(md))

    print(f"[ok] Reporte NOW guardado en: {report_dir}")
    return summary, report_dir

# --------- MAIN: one or multiple models ---------
def main():
    global BATCH_EVAL  # needed to reassign

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="Test CSV (by default looks for data/processed/pairs_test.csv).")
    ap.add_argument("--dir", type=str, default="checkpoints_now8h,checkpoints_now8h_seg",
                    help="Comma-separated checkpoint folders.")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Specific checkpoint prefixes (optional, comma-separated, overrides --dir).")
    ap.add_argument("--tags", type=str, default=None,
                    help="Labels for reports (comma-separated). By default inferred from dir or ckpt.")
    ap.add_argument("--batch-eval", type=int, default=BATCH_EVAL,
                    help="Batch for evaluation (default = %(default)s)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else (ROOT/"data/processed/pairs_test.csv")
    if not csv_path.exists():
        raise SystemExit(f"No existe CSV: {csv_path}")

    BATCH_EVAL = int(args.batch_eval)

    # Build list of (ckpt_prefix, tag)
    items = []
    if args.ckpt:
        ckpts = [c.strip().rstrip(".index") for c in args.ckpt.split(",") if c.strip()]
        tags  = [Path(c).parent.name for c in ckpts]
        if args.tags:
            t2 = [t.strip() for t in args.tags.split(",")]
            if len(t2) == len(ckpts): tags = t2
        items = list(zip(ckpts, tags))
    else:
        dirs = [Path(d.strip()) for d in args.dir.split(",") if d.strip()]
        tags = [d.name for d in dirs]
        if args.tags:
            t2 = [t.strip() for t in args.tags.split(",")]
            if len(t2) == len(dirs): tags = t2
        for d, tag in zip(dirs, tags):
            base = ROOT / d if not d.is_absolute() else d
            ckpt = resolve_ckpt(base)
            items.append((ckpt, tag))

    if not items:
        raise SystemExit("No hay modelos a evaluar (revisa --dir / --ckpt).")

    reports = []
    outer = tqdm(total=len(items), desc="Evaluando modelos NOW (suite)", ncols=100) if tqdm else None
    for ckpt_prefix, tag in items:
        out_dir = ROOT / "reports_now" / tag
        summary, rdir = run_eval_now(ckpt_prefix, csv_path, out_dir)
        reports.append((tag, summary, rdir))
        if outer: outer.update(1)
    if outer: outer.close()

    # If there are several, compile tables + copy all figs to _figuras/<ts>
    if len(reports) >= 2:
        tables_dir = ROOT / "reports_now" / "_tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

        rows = []
        for tag, summary, _ in reports:
            seg = summary["metrics"]["seg"]
            rows.append(dict(
                Modelo = tag,
                PixelAcc  = seg.get("pixel_acc", np.nan),
                Recall_SKY   = seg.get("rec_sky", np.nan),
                Recall_THICK = seg.get("rec_thick", np.nan),
                Recall_THIN  = seg.get("rec_thin", np.nan),
                Recall_SUN   = seg.get("rec_sun", np.nan),
                mIoU         = seg.get("miou", np.nan),
                ms_per_img   = summary["speed"].get("ms_per_img", np.nan),
                FPS          = summary["speed"].get("fps", np.nan),
                Checkpoint   = summary.get("ckpt_prefix","")
            ))
        df = pd.DataFrame(rows)

        # CSV + simple Markdown
        csv_out = tables_dir / f"summary_now_{ts}.csv"
        md_out  = tables_dir / f"summary_now_{ts}.md"
        df.to_csv(csv_out, index=False)

        def to_md(df_):
            cols = df_.columns
            lines = ["| " + " | ".join(cols) + " |",
                     "| " + " | ".join(["---"]*len(cols)) + " |"]
            for _, r in df_.iterrows():
                vals = []
                for c in cols:
                    v = r[c]
                    if isinstance(v, float) and not np.isnan(v):
                        if c in ("PixelAcc","Recall_SKY","Recall_THICK","Recall_THIN","Recall_SUN","mIoU"):
                            vals.append(f"{v:.3f}")
                        elif c in ("ms_per_img","FPS"):
                            vals.append(f"{v:.1f}")
                        else:
                            vals.append(f"{v:.3f}")
                    else:
                        vals.append("" if (isinstance(v,float) and np.isnan(v)) else str(v))
                lines.append("| " + " | ".join(vals) + " |")
            return "\n".join(lines)

        df_fmt = df.copy()
        with open(md_out, "w") as f:
            f.write("# Resumen modelos NOW (test)\n\n")
            f.write(to_md(df_fmt.drop(columns=["Checkpoint"])))
            f.write("\n")
        print(f"[ok] Tablas NOW:\n- {csv_out}\n- {md_out}")

        # Copy PNGs to _figuras/<ts>
        figs_dir = ROOT / "reports_now" / "_figuras" / ts
        figs_dir.mkdir(parents=True, exist_ok=True)
        for _, _, rdir in reports:
            for p in rdir.glob("*.png"):
                shutil.copy2(p, figs_dir / f"{rdir.name}__{p.name}")
        print(f"[ok] Figuras copiadas en: {figs_dir}")

if __name__ == "__main__":
    main()
