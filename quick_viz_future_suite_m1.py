"""
caffeinate -d -i python src/quick_viz_future_suite_m1.py
"""

# -*- coding: utf-8 -*-
# quick_viz_future_suite_m1.py
# Generates one figure per FUTURE model (6 in total) using the SAME random window:
#   - Top: GHI curve (GT vs Pred)
#   - Bottom: Image | GT(overlay) | Pred(overlay) per timestep
#
# Run:
#   caffeinate -d -i python src/quick_viz_future_suite_m1.py --csv data/processed/pairs_test.csv
#
# Output:
#   reports_future/_figuras/<timestamp>/
#     gru_seg_viz.png, gru_ghi_viz.png, gru_bal_viz.png, lstm_seg_viz.png, lstm_ghi_viz.png, lstm_bal_viz.png

import os, sys, argparse, datetime as dt
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
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

# ===== Config consistent with training =====
IMG_SIZE = (320, 320)
NUM_CLASSES = 5
TIMESTEPS = 6
STRIDE = 6
ALPHA = 0.5
OUTPUT_STRIDE = 16

LABELS = ["gru_seg","gru_ghi","gru_bal","lstm_seg","lstm_ghi","lstm_bal"]

REGISTRY = {
    # GRU
    "gru_seg": dict(rnn="gru",  patterns=[
        "*segphase_best_segloss.index", "*_best_segloss.index", "*_last.index"
    ]),
    "gru_ghi": dict(rnn="gru",  patterns=[
        "*ghiphase_best.index", "*_best.index", "*_last.index"
    ]),
    "gru_bal": dict(rnn="gru",  patterns=[
        "*balanced_best_mix.index", "*_best.index", "*_last.index"
    ]),
    # LSTM
    "lstm_seg":dict(rnn="lstm", patterns=[
        "*segphase_best_segloss.index", "*_best_segloss.index", "*_last.index"
    ]),
    "lstm_ghi":dict(rnn="lstm", patterns=[
        "*ghiphase_best.index", "*_best.index", "*_last.index"
    ]),
    "lstm_bal":dict(rnn="lstm", patterns=[
        "*balanced_best_mix.index", "*_best.index", "*_last.index"
    ]),
}

def set_policy_float32():
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

def make_palette():
    pal = np.zeros((NUM_CLASSES, 3), dtype=np.uint8)
    colors = {
        "void": (0, 0, 0),
        "sky":  (84, 84, 255),
        "thick":(149,149,149),
        "thin": (224,224,224),
        "sun":  (255,255,  0),
    }
    pal[getattr(models, "VOID_IDX", 0)]   = colors["void"]
    pal[getattr(models, "SKY_IDX", 1)]    = colors["sky"]
    pal[getattr(models, "THICK_IDX", 2)]  = colors["thick"]
    pal[getattr(models, "THIN_IDX", 3)]   = colors["thin"]
    pal[getattr(models, "SUN_IDX", 4)]    = colors["sun"]
    return pal
PALETTE = make_palette()

def to_uint8_img(arr):
    a = np.asarray(arr)
    if a.max() <= 1.5: a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)

def overlay(img_u8, rgb_u8, alpha=0.45):
    img = img_u8.astype(np.float32)/255.0
    rgb = rgb_u8.astype(np.float32)/255.0
    out = (1 - alpha) * img + alpha * rgb
    return np.clip(out*255.0, 0, 255).astype(np.uint8)

def onehot_to_rgb(mask_onehot):
    lbl = np.argmax(mask_onehot, axis=-1).astype(np.int32)
    return PALETTE[lbl]

def build_future(rnn_type: str):
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

def default_csv() -> str:
    for c in [ROOT/"data/processed/pairs_test.csv",
              ROOT/"data/processed/pairs_val.csv"]:
        if c.exists(): return str(c)
    raise FileNotFoundError("No se encontró ni pairs_test.csv ni pairs_val.csv")

def ensure_targets_SI(ds: tf.data.Dataset) -> tf.data.Dataset:
    if getattr(dl, "GHI_MODE", "si") == "normalized":
        @tf.function
        def _to_si(img, y):
            y2 = dict(y)
            y2["ghi_pred"] = dl._denormalize_ghi(y2["ghi_pred"])
            return img, y2
        return ds.map(_to_si, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

def resolve_ckpt_for_label(label: str) -> str:
    assert label in REGISTRY
    rnn = REGISTRY[label]["rnn"]
    patterns = REGISTRY[label]["patterns"]
    outdir = ROOT / f"checkpoints_future_{rnn}"
    for pat in patterns:
        fs = sorted(outdir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if fs:
            return str(fs[0])[:-len(".index")]
    raise FileNotFoundError(f"No checkpoint para {label} en {outdir}")

def sample_one_window(csv_path: Path):
    # A single random window (same for all models)
    dyn_seed = int.from_bytes(os.urandom(2), "big")
    ds = dl.build_future_dataset(
        csv_path=str(csv_path),
        timesteps=TIMESTEPS, stride=STRIDE,
        batch_size=1, seed_number=dyn_seed,
        shuffle=True,
        cache_path=None, buffer_size=64,
        reshuffle_each_iteration_value=True,
        num_classes=NUM_CLASSES, augment=False,
        drop_remainder=False
    )
    ds = ensure_targets_SI(ds).take(1).prefetch(1)
    (x, y) = next(iter(ds))
    return x, y

def predict_future(fut: tf.keras.Model, x):
    outs = fut.predict(x, verbose=0)
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    name2pred = {name: arr for name, arr in zip(fut.output_names, outs)}
    upseg = name2pred.get("upseg", None)
    ghi_p = name2pred.get("ghi_pred", None)
    if upseg is None or ghi_p is None:
        a, b = outs
        if a.ndim == 5 and a.shape[-1] == NUM_CLASSES: upseg, ghi_p = a, b
        else: upseg, ghi_p = b, a
    upseg = np.asarray(upseg)[0]              # (T,H,W,C)
    ghi_p = np.asarray(ghi_p)[0].reshape(-1)  # (T,)
    return upseg, ghi_p

def render_figure(label: str, x_np, y_mask, y_ghi, upseg, ghi_pred, out_png: Path):
    # Layout: top row GHI; below T rows × 3 columns (Image | GT overlay | Pred overlay)
    import matplotlib.gridspec as gridspec
    rows_grid = TIMESTEPS
    fig_h = 3.4 + rows_grid*2.7
    fig = plt.figure(figsize=(12.2, fig_h), dpi=120)
    gs = gridspec.GridSpec(nrows=rows_grid+2, ncols=3, height_ratios=[1.0, 0.2] + [2.7]*rows_grid)

    # --- Top: GHI curve
    ax_top = fig.add_subplot(gs[0,:])
    ts = np.arange(TIMESTEPS)
    if y_ghi is not None: ax_top.plot(ts, y_ghi, marker="o", label="GHI GT (W/m²)")
    ax_top.plot(ts, ghi_pred, marker="s", label="GHI Pred (W/m²)")
    ax_top.set_xticks(ts); ax_top.set_xlabel("Paso futuro t"); ax_top.set_ylabel("W/m²")
    ax_top.grid(True, alpha=0.3); ax_top.legend(loc="best")
    ax_top.set_title(f"FUTURE – {label}  |  Serie GHI")

    # thin separator bar
    ax_sep = fig.add_subplot(gs[1,:]); ax_sep.axis("off")

    # --- Grid per timestep
    col_titles = ["Imagen", "GT (overlay)", "Pred (overlay)"]
    for j, t in enumerate(col_titles):
        ax = fig.add_subplot(gs[2, j])
        ax.axis("off"); ax.set_title(t, fontsize=12, pad=6)

    for t in range(TIMESTEPS):
        img_u8 = to_uint8_img(x_np[t])
        pred_lbl = np.argmax(upseg[t], axis=-1).astype(np.int32)
        pred_rgb = PALETTE[pred_lbl]
        gt_rgb = onehot_to_rgb(y_mask[t]) if y_mask is not None else None

        r = 2 + t
        ax0 = fig.add_subplot(gs[r,0]); ax0.imshow(img_u8); ax0.axis("off")
        ax1 = fig.add_subplot(gs[r,1]); 
        ax1.imshow(img_u8); 
        if gt_rgb is not None: ax1.imshow(gt_rgb, alpha=0.45)
        ax1.axis("off")
        ax2 = fig.add_subplot(gs[r,2]); 
        ax2.imshow(img_u8); ax2.imshow(pred_rgb, alpha=0.45); ax2.axis("off")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="CSV de test (por defecto pairs_test.csv).")
    args = ap.parse_args()

    set_policy_float32()
    dl.IMG_SIZE = tuple(IMG_SIZE)

    csv_path = Path(args.csv) if args.csv else Path(default_csv())
    assert csv_path.exists(), f"CSV no encontrado: {csv_path}"

    # Single sample for all
    x, y = sample_one_window(csv_path)
    x_np  = x.numpy()[0]
    y_mask = y.get("upseg", None)
    y_ghi  = y.get("ghi_pred", None)
    y_mask = None if y_mask is None else y_mask.numpy()[0]
    y_ghi  = None if y_ghi  is None else y_ghi.numpy()[0].reshape(-1)

    out_dir = ROOT / "reports_future" / "_figuras" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    iterator = LABELS
    bar = tqdm(total=len(iterator), desc="Generando figuras FUTURE", ncols=100) if tqdm else None

    for label in iterator:
        rnn = REGISTRY[label]["rnn"]
        ckpt_prefix = resolve_ckpt_for_label(label)
        fut = build_future(rnn)
        fut.load_weights(ckpt_prefix)

        upseg, ghi_pred = predict_future(fut, x)
        out_png = out_dir / f"{label}_viz.png"
        render_figure(label, x_np, y_mask, y_ghi, upseg, ghi_pred, out_png)
        print(f"[ok] {label} → {out_png}")
        if bar: bar.update(1)

    if bar: bar.close()
    print(f"\nFiguras guardadas en: {out_dir}")

if __name__ == "__main__":
    main()
