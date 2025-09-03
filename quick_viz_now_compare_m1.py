"""
caffeinate -d -i python src/quick_viz_now_compare_m1.py \
  --csv data/processed/pairs_test.csv \
  --dir-a checkpoints_now8h --dir-b checkpoints_now8h_seg
"""
# -*- coding: utf-8 -*-
# quick_viz_now_m1_dual.py
# Visual comparison of 2 NOW models (segmentation only): Image, GT(overlay), Pred A, Pred B.
# Run:
#   caffeinate -d -i python src/quick_viz_now_m1_dual.py

import os, random, datetime as dt
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import dataset_loader_m1 as dl
import models_m1 as models

# ------------------ Config ------------------
IMG_SIZE = (320, 320)
ALPHA = 0.5
OUTPUT_STRIDE = 16
NUM_CLASSES = 5
BATCH_VIZ = 8
SEED = 1337

ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data" / "processed"
TEST_CSV = DATA / "pairs_test.csv"

# Models to compare (folders)
DIR_A = ROOT / "checkpoints_now8h"
DIR_B = ROOT / "checkpoints_now8h_seg"

# short names for headers
NAME_A = "now8h"
NAME_B = "now8h_seg"

# Output
OUT_DIR = ROOT / "reports_now" / "_figuras"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_PNG = OUT_DIR / f"quick_now_dual_{STAMP}.png"

# Palette
PALETTE = np.array([
    (0,0,0), (84,84,255), (149,149,149), (224,224,224), (255,255,0)
], dtype=np.uint8)

VOID_IDX, SKY_IDX, THICK_IDX, THIN_IDX, SUN_IDX = 0, 1, 2, 3, 4

def set_seeds(s: int):
    import random as _r
    os.environ["PYTHONHASHSEED"] = str(s)
    tf.random.set_seed(s); np.random.seed(s); _r.seed(s)

def set_m1_policy():
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

def onehot_to_rgb(mask_onehot: np.ndarray) -> np.ndarray:
    lbl = np.argmax(mask_onehot, axis=-1).astype(np.int32)
    return PALETTE[lbl]

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

def sample_batch(csv_path: Path, batch: int = BATCH_VIZ):
    ds = dl.build_now_dataset(
        csv_path=str(csv_path),
        batch_size=batch, shuffle=True, seed_number=SEED,
        cache_path=None, reshuffle_each_iteration_value=False,
        buffer_size=2048, num_classes=NUM_CLASSES,
        augment=False, drop_remainder=False
    )
    ds = ensure_targets_SI(ds)
    return next(iter(ds.take(1)))

def build_now():
    now, _ = models.build_models_shared(
        timesteps=6,
        img_height=IMG_SIZE[0], img_width=IMG_SIZE[1], img_channels=3,
        num_classes=NUM_CLASSES,
        rnn_type="gru", d=models.D_REPR,
        output_stride=OUTPUT_STRIDE, mbv2_alpha=ALPHA,
        weights=None, learning_rate=1e-4, clipnorm=1.0,
        build_future_model=False
    )
    return now

def resolve_ckpt(dir_path: Path) -> str:
    for pat in ["*best_segloss.index", "*best.index", "*last.index"]:
        fs = sorted(dir_path.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        if fs:
            return str(fs[0])[:-len(".index")]
    raise FileNotFoundError(f"No se encontró checkpoint en {dir_path}")

def predict_masks(model, x):
    outs = model(x, training=False)
    if isinstance(outs, (list, tuple)):
        # take the one with shape (..., C=NUM_CLASSES)
        for t in outs:
            if int(t.shape[-1]) == NUM_CLASSES:
                seg = t; break
        else:
            seg = outs[0]
    else:
        seg = outs
    # expected (B,H,W,C)
    if len(seg.shape) == 5:  # (B,T,H,W,C)
        seg = seg[:,0]
    return seg.numpy()

def main():
    assert TEST_CSV.exists(), f"No existe {TEST_CSV}"
    set_seeds(SEED); set_m1_policy()
    dl.IMG_SIZE = tuple(IMG_SIZE)

    # batch
    x, y = sample_batch(TEST_CSV, BATCH_VIZ)
    x_np = x.numpy()
    y_mask = y["upseg"].numpy()
    B = x_np.shape[0]

    # models
    ckpt_a = resolve_ckpt(DIR_A)
    ckpt_b = resolve_ckpt(DIR_B)
    mA = build_now(); mA.load_weights(ckpt_a)
    mB = build_now(); mB.load_weights(ckpt_b)
    print(f"[ckpt] A={ckpt_a}\n[ckpt] B={ckpt_b}")

    # preds
    pA = predict_masks(mA, x)
    pB = predict_masks(mB, x)

    # figure — titles ONLY in the FIRST ROW
    cols = 4
    fig, axs = plt.subplots(
        B, cols,
        figsize=(3.9*cols, 3.2*B),
        dpi=120,
        constrained_layout=True
    )
    if B == 1: axs = np.expand_dims(axs, 0)

    col_titles = ["Imagen", "GT (overlay)", f"Pred {NAME_A}", f"Pred {NAME_B}"]

    for i in range(B):
        img_u8 = (np.clip(x_np[i], 0, 1) * 255).astype(np.uint8)
        gt_rgb = onehot_to_rgb(y_mask[i])
        predA_rgb = PALETTE[np.argmax(pA[i], axis=-1).astype(np.int32)]
        predB_rgb = PALETTE[np.argmax(pB[i], axis=-1).astype(np.int32)]

        # col 0
        axs[i,0].imshow(img_u8); axs[i,0].axis("off")
        # col 1
        axs[i,1].imshow(img_u8); axs[i,1].imshow(gt_rgb, alpha=0.45)
        axs[i,1].axis("off")
        # col 2
        axs[i,2].imshow(img_u8); axs[i,2].imshow(predA_rgb, alpha=0.45)
        axs[i,2].axis("off")
        # col 3
        axs[i,3].imshow(img_u8); axs[i,3].imshow(predB_rgb, alpha=0.45)
        axs[i,3].axis("off")

    # Column headers only in the first row (with spacing)
    for j, t in enumerate(col_titles):
        axs[0, j].set_title(t, fontsize=12, pad=10)

    # A bit of top margin so they don't touch
    plt.subplots_adjust(top=0.93, wspace=0.04, hspace=0.08)
    fig.suptitle("NOW – Comparativa de segmentación", y=0.99, fontsize=14)

    plt.savefig(OUT_PNG)
    print(f"[viz] Guardado: {OUT_PNG}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
