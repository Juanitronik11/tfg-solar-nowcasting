# -*- coding: utf-8 -*-
# Run:  caffeinate -d -i python src/def_train_now_m1.py
# NOW (4 hours) — Lightweight metrics + GHI with more “breathing room”.

import os, time, json, math, datetime as dt
from pathlib import Path
from typing import Tuple, Optional

# M1: stable legacy optimizers
os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

# ==== PROJECT MODULES ====
import dataset_loader_m1 as dl
import models_m1 as models

# ===================== PARAMETERS (EDIT HERE) =====================
IMG_SIZE: Tuple[int, int] = (320, 320)
BATCH               = 16
EPOCHS_MAX          = 999
# MAX_MINUTES         = 480                 # ← 8 hours
MAX_MINUTES         = 480                 # ← 8 hours
STEPS_PER_EPOCH     = "auto"              # "auto" or an integer
TRAIN_AUTO_CAP      = 2000
VAL_STEPS           = "auto"              # "auto" or an integer
VAL_AUTO_CAP        = 400
VALIDATION_EVERY    = 1                   # validate every N epochs
ALPHA               = 0.5
OUTPUT_STRIDE       = 16
LR_TARGET           = 1e-4
LR_WARMUP_EPOCHS    = 3
CLIPNORM            = 1.0

# WEIGH GHI MORE AND LOWER SEG: give irradiance gradient some “air”
LAMBDA_SEG          = 2.5                # seg vs GHI
GHI_WEIGHT          = 1                # ↑ GHI weight in total loss

LAMBDA_DICE         = 1.0                 # Dice weight inside seg_total
AUGMENT_TRAIN       = True
USE_IMAGENET        = True
SEED                = 1337
STEPS_PER_EXEC      = 32                  # good throughput on M1
NUM_CLASSES         = 5
FREEZE_BACKBONE_EPOCHS = 1                # freeze MBV2 at the beginning

# ===== Balancing =====
# We avoid rejection_resample (slow in your setup) and use hard balance.
BALANCE_SOFT        = False               # <- disable soft balancing
HARD_BALANCE        = True                # <- enable strict per-batch oversampling
HARD_QUOTA_3P1      = (10, 5, 1, 0)       # SUN, THICK, THIN, REST (sum == BATCH)
# (Soft quotas for BATCH=16: 62.5% / 31.25% / 6.25% / 0%)
HARD_BALANCE_MODE   = "combo8"           # "3p1" | "combo8"
HARD_QUOTA_COMBO8   = (2,2,2,2,2,2,2,2)  # must sum to BATCH

# ================= PATHS =================
ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TRAIN_CSV = DATA / "pairs_train_with_cls.csv"
VAL_CSV   = DATA / "pairs_val_with_cls.csv"
OUTDIR    = ROOT / "checkpoints_now8h"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] ROOT={ROOT}")
print(f"[paths] DATA={DATA}")

# ---- Reproducibility / M1 ----
def set_seeds(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

def set_m1_policy() -> None:
    # On M1/Metal, float32 is more stable than mixed fp16
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

# ---- Utilities ----
def _count_csv_rows(csv_path: Path) -> int:
    try:
        with open(csv_path, "r") as f:
            n = sum(1 for _ in f)
        return max(n - 1, 0)  # subtract header
    except Exception:
        return 0

def infer_train_steps(train_csv: Path, batch: int, cap: int) -> int:
    if isinstance(STEPS_PER_EPOCH, int):
        return max(1, int(STEPS_PER_EPOCH))
    if isinstance(STEPS_PER_EPOCH, str) and STEPS_PER_EPOCH.lower() == "auto":
        n_rows = _count_csv_rows(train_csv)
        if n_rows > 0:
            steps = int(math.ceil(n_rows / float(batch)))
            steps = min(steps, cap)
            print(f"[train] STEPS_PER_EPOCH=auto → {steps} (cap={cap})")
            return max(1, steps)
        print(f"[train] STEPS_PER_EPOCH=auto but could not infer → fallback={cap}")
        return max(1, cap)
    return max(1, cap)

def infer_val_steps(val_csv: Path, batch: int, fallback: int = 300) -> int:
    if isinstance(VAL_STEPS, int):
        return max(1, int(VAL_STEPS))
    if isinstance(VAL_STEPS, str) and VAL_STEPS.lower() == "auto":
        n_rows = _count_csv_rows(val_csv)
        if n_rows > 0:
            steps = int(math.ceil(n_rows / float(batch)))
            steps = min(steps, VAL_AUTO_CAP)
            print(f"[val] VAL_STEPS=auto → {steps} (cap={VAL_AUTO_CAP})")
            return max(1, steps)
        print(f"[val] VAL_STEPS=auto but could not infer → fallback={fallback}")
        return max(1, fallback)
    return max(1, fallback)

def _jsonify(o):
    import numpy as np
    if hasattr(o, "numpy"): o = o.numpy()
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    raise TypeError(f"Type {type(o).__name__} not serializable")

# ---- If the loader were normalized, force targets to W/m² ----
def ensure_targets_in_SI(ds: tf.data.Dataset, name: str) -> tf.data.Dataset:
    if getattr(dl, "GHI_MODE", "si") == "normalized":
        print(f"[{name}] Converting GHI targets to W/m² via dl._denormalize_ghi…")
        @tf.function
        def _to_si(img, y):
            y2 = dict(y)
            y2["ghi_pred"] = dl._denormalize_ghi(y2["ghi_pred"])
            return img, y2
        return ds.map(_to_si, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        print(f"[{name}] GHI already in W/m² (dl.GHI_MODE='si').")
        return ds

# ---- Datasets (no on-disk cache) ----
def make_datasets(train_csv: Path, val_csv: Path, batch: int,
                  shuffle_train: bool = False, seed: int = 1337,
                  augment_train: bool = True):
    SHUFFLE_BUF_TRAIN = 512
    SHUFFLE_BUF_VAL   = 1024

    # TRAINING: hard balance by quotas to avoid massive rejections
    ds_train = dl.build_now_dataset(
        csv_path=str(train_csv),
        batch_size=batch,
        shuffle=shuffle_train,
        seed_number=seed,
        cache_path=None,
        reshuffle_each_iteration_value=True,
        buffer_size=SHUFFLE_BUF_TRAIN,
        num_classes=NUM_CLASSES,
        augment=augment_train,
        drop_remainder=True,
        balance=BALANCE_SOFT,
        hard_balance=HARD_BALANCE,
        hard_balance_mode=HARD_BALANCE_MODE,   
        hard_quota_3p1=HARD_QUOTA_3P1,
        hard_quota_combo8=HARD_QUOTA_COMBO8,      
    )

    # VALIDATION: unbalanced (natural distribution)
    ds_val = dl.build_now_dataset(
        csv_path=str(val_csv),
        batch_size=batch,
        shuffle=False,
        seed_number=seed,
        cache_path=None,
        reshuffle_each_iteration_value=False,
        buffer_size=SHUFFLE_BUF_VAL,
        num_classes=NUM_CLASSES,
        augment=False,
        drop_remainder=False,
        balance=False, hard_balance=False
    )

    ds_train = ensure_targets_in_SI(ds_train, "train")
    ds_val   = ensure_targets_in_SI(ds_val,   "val")

    opts = tf.data.Options(); opts.experimental_deterministic = False
    ds_train = ds_train.with_options(opts).repeat().prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.with_options(opts).repeat().prefetch(tf.data.AUTOTUNE)
    print(f"[balance] HARD_BALANCE={HARD_BALANCE} mode={HARD_BALANCE_MODE} "
      f"q3p1={HARD_QUOTA_3P1} q8={HARD_QUOTA_COMBO8}")
    return ds_train, ds_val

# ---- Handy callbacks (GHI “air”, time control) ----
class WarmupLR(tf.keras.callbacks.Callback):
    def __init__(self, lr_target: float, warmup_epochs: int):
        super().__init__()
        self.lr_target = float(lr_target)
        self.warmup_epochs = int(warmup_epochs)
    def _set_lr(self, value: float) -> None:
        opt = self.model.optimizer
        lr_var = getattr(opt, "lr", None) or getattr(opt, "learning_rate", None)
        tf.keras.backend.set_value(lr_var, value)
    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0: return
        if epoch < self.warmup_epochs:
            frac = 0.25 + 0.75 * (epoch + 1) / self.warmup_epochs
            self._set_lr(self.lr_target * frac)
            print(f"[warmup] epoch {epoch+1}/{self.warmup_epochs}: lr -> {self.lr_target * frac:.6g}")
        elif epoch == self.warmup_epochs:
            self._set_lr(self.lr_target)
            print(f"[warmup] set target lr: {self.lr_target:.6g}")

class TimeLimit(tf.keras.callbacks.Callback):
    """Stops training when MAX_MINUTES (wall clock) is exceeded."""
    def __init__(self, max_minutes: float):
        super().__init__()
        self.max_seconds = float(max_minutes) * 60.0
        self.t0 = None
    def on_train_begin(self, logs=None):
        self.t0 = time.time()
    def _should_stop(self) -> bool:
        return (time.time() - self.t0) >= self.max_seconds
    def on_batch_end(self, batch, logs=None):
        if self._should_stop():
            print(f"\n[TimeLimit] Limit {self.max_seconds/60:.1f} min reached. Stopping…")
            self.model.stop_training = True
    def on_test_batch_end(self, batch, logs=None):
        if self._should_stop():
            print(f"\n[TimeLimit] Limit {self.max_seconds/60:.1f} min reached (val). Stopping…")
            self.model.stop_training = True

class FreezeBackbone(tf.keras.callbacks.Callback):
    """Temporarily freezes the MBV2 submodel (all its sublayers)."""
    def __init__(self, freeze_epochs: int = 1):
        super().__init__()
        self.freeze_epochs = int(max(0, freeze_epochs))
        self.frozen = False
    def _set_trainable(self, flag: bool):
        n = 0
        for l in self.model.layers:
            if isinstance(l, tf.keras.Model) and l.name.startswith("mbv2_OS"):
                for sl in l.submodules:
                    if hasattr(sl, "trainable"):
                        sl.trainable = flag; n += 1
        print(f"[backbone] {'UNFREEZE' if flag else 'FREEZE'} {n} MBV2 sublayers")
    def on_train_begin(self, logs=None):
        if self.freeze_epochs > 0:
            self._set_trainable(False); self.frozen = True
    def on_epoch_begin(self, epoch, logs=None):
        if self.frozen and epoch >= self.freeze_epochs:
            self._set_trainable(True); self.frozen = False

class FreezeSegHead(tf.keras.callbacks.Callback):
    """GHI warm-up: freeze seg_head for the first epoch(s)."""
    def __init__(self, freeze_epochs: int = 1):
        super().__init__(); self.freeze_epochs=int(max(0, freeze_epochs)); self.frozen=False
    def _set_trainable(self, flag: bool):
        n=0
        for l in self.model.layers:
            if isinstance(l, tf.keras.Model) and l.name == "seg_head":
                for sl in l.submodules:
                    if hasattr(sl, "trainable"): sl.trainable = flag; n += 1
        print(f"[seg_head] {'UNFREEZE' if flag else 'FREEZE'} {n} sublayers")
    def on_train_begin(self, logs=None):
        if self.freeze_epochs>0: self._set_trainable(False); self.frozen=True
    def on_epoch_begin(self, epoch, logs=None):
        if self.frozen and epoch>=self.freeze_epochs:
            self._set_trainable(True); self.frozen=False

class GhiRange(tf.keras.callbacks.Callback):
    """Logs range of GHI predictions/targets to check it's “moving”."""
    def __init__(self, ds_val, n_batches=1):
        super().__init__(); self.ds_val = ds_val; self.n_batches = n_batches
    def on_epoch_end(self, epoch, logs=None):
        xs, ts = [], []
        val_it = iter(self.ds_val)
        for _ in range(self.n_batches):
            x1, y1 = next(val_it)
            xs.append(x1); ts.append(y1["ghi_pred"].numpy().reshape(-1))
        p = self.model.predict(tf.concat(xs, 0), verbose=0)[1].reshape(-1)
        t = np.concatenate(ts, 0)
        print(f"[GHI pred] e{epoch+1}: pred(mean±std)={p.mean():.1f}±{p.std():.1f}  tgt(mean±std)={t.mean():.1f}±{t.std():.1f}")

# ---- Extra losses to recompile (lightweight) ----
class BalancedCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name="balanced_ce", **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kw)
        self.from_logits = from_logits
    def call(self, y_true, y_pred):
        M = models.valid_mask(y_true)  # (circle ∧ non-VOID)
        eps = 1e-8
        p = tf.nn.softmax(y_pred, axis=-1) if self.from_logits else y_pred
        ce_map = - y_true * tf.math.log(tf.clip_by_value(p, eps, 1.0))
        axes = tf.range(1, tf.rank(ce_map)-1)               # sum over T?,H,W
        num = tf.reduce_sum(ce_map * M, axis=axes)          # (B,C)
        den = tf.reduce_sum(y_true * M, axis=axes) + eps    # (B,C)
        ce_c = num / den                                    # (B,C)
        present = tf.cast(den > eps, tf.float32)            # (B,C) → VOID will already be 0
        ce_img = tf.math.divide_no_nan(
            tf.reduce_sum(ce_c * present, axis=-1),
            tf.reduce_sum(present, axis=-1)
        )                                                   # [B]
        return ce_img


class HistKLDivergence(tf.keras.losses.Loss):
    def __init__(self, lam=0.05, name="hist_kl", **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kw)
        self.lam = tf.constant(lam, tf.float32)
    def call(self, y_true, y_pred):
        M = models.fixed_circle_mask_like(y_true)  # (B,[T,]H,W,1)
        eps = 1e-8
        axes = tf.range(1, tf.rank(y_true)-1)
        p_true = tf.reduce_sum(y_true * M, axis=axes) + eps
        p_pred = tf.reduce_sum(y_pred * M, axis=axes) + eps
        p_true = p_true / tf.reduce_sum(p_true, axis=-1, keepdims=True)
        p_pred = p_pred / tf.reduce_sum(p_pred, axis=-1, keepdims=True)
        kl = tf.reduce_sum(p_true * (tf.math.log(p_true) - tf.math.log(p_pred)), axis=-1)  # [B]
        return self.lam * kl

# ---- Model (recompiled WITHOUT mIoU) ----
def build_now_compiled(img_h: int, img_w: int, num_classes: int, alpha: float, output_stride: int,
                       lr_target: float, clipnorm: float, imagenet: bool,
                       lambda_seg: float, lambda_dice: float,
                       spe: int = STEPS_PER_EXEC) -> tf.keras.Model:
    weights = "imagenet" if imagenet else None
    now, _ = models.build_models_shared(
        timesteps=6,
        img_height=img_h, img_width=img_w, img_channels=3,
        num_classes=num_classes,
        rnn_type="gru",
        d=models.D_REPR,
        output_stride=output_stride,
        mbv2_alpha=alpha,
        weights=weights,
        learning_rate=lr_target,
        clipnorm=clipnorm,
        build_future_model=False
    )
    # Recompile without mIoU and with lightweight metrics
    seg_base  = BalancedCELoss(from_logits=False)
    dice_loss = models.TverskyLossMasked(alpha=0.3, beta=0.7, ignore=(models.VOID_IDX,))
    hist_reg  = HistKLDivergence(lam=0.05)

    def seg_total_mean(y_true, y_pred):
        base = seg_base(y_true, y_pred) + LAMBDA_DICE * dice_loss(y_true, y_pred)
        return tf.reduce_mean(base + hist_reg(y_true, y_pred))

    ghi_loss  = models.LogCoshLossSI(m=0.01, low_irr_thr=150.0, low_irr_weight=1.5)
    def ghi_mean(y_true, y_pred):
        return tf.reduce_mean(ghi_loss(y_true, y_pred))

    try:
        from tensorflow.keras.optimizers.legacy import Adam
        opt = Adam(learning_rate=lr_target, clipnorm=clipnorm)
        print("[opt] legacy.Adam")
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_target, clipnorm=clipnorm)
        print("[opt] Adam")

    now.compile(
        optimizer=opt,
        loss={"upseg": seg_total_mean, "ghi_pred": ghi_mean},
        loss_weights={"upseg": float(lambda_seg), "ghi_pred": float(GHI_WEIGHT)},
        metrics={
            "upseg": [
                models.PixelAccWithMask(name="pixel_acc"),
                models.RecallForClassMasked(models.SKY_IDX,   name="recall_sky"),
                models.RecallForClassMasked(models.THICK_IDX, name="recall_thick"),
                models.RecallForClassMasked(models.THIN_IDX,  name="recall_thin"),
                models.RecallForClassMasked(models.SUN_IDX,   name="recall_sun"),
            ],
            "ghi_pred": [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        },
        steps_per_execution=int(spe),
    )
    return now

def _init_ghi_bias_from_csv(model: tf.keras.Model, train_csv: Path) -> None:
    """Initialize 'ghi_dense' bias to the real train median."""
    try:
        med = float(np.median(pd.read_csv(train_csv)["ghi"].values))
        med = max(1.0, min(1499.0, med))
        def logit(p): return np.log(p/(1.0-p))
        b0 = logit(med/1500.0)
        ghi_head = model.get_layer("ghi_head")
        ghi_dense = ghi_head.get_layer("ghi_dense")
        W, b = ghi_dense.get_weights()
        ghi_dense.set_weights([W, np.array([b0], dtype=np.float32)])
        print(f"[init] ghi_dense.bias -> {b0:.3f}  (~{med:.0f} W/m²)")
    except Exception as e:
        print(f"[init] (warning) could not initialize GHI bias: {e}")

# ---- Base callbacks pack (without re-declaring above) ----
def make_callbacks(tag: str, outdir: Path, *, add_recall_ckpts: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_best       = outdir / f"{tag}_best"            # GHI MAE
    ckpt_best_seg   = outdir / f"{tag}_best_segloss"    # seg loss
    ckpt_last       = outdir / f"{tag}_last"            # always last
    ckpt_best_thin  = outdir / f"{tag}_best_recall_thin"
    ckpt_best_sun   = outdir / f"{tag}_best_recall_sun"
    log_dir = outdir / "logs" / (tag + "_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best),
            monitor="val_ghi_pred_mae", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_seg),
            monitor="val_upseg_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_last),
            save_best_only=False, save_weights_only=True, verbose=0,
        ),
        # optional: if you also want to prioritize THICK
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(outdir / f"{tag}_best_recall_thick"),
            monitor="val_upseg_recall_thick", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        WarmupLR(LR_TARGET, LR_WARMUP_EPOCHS),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_ghi_pred_mae", mode="min",
            factor=0.5, patience=5, cooldown=1, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_ghi_pred_mae", mode="min",
            patience=14, min_delta=0.0, restore_best_weights=False, verbose=1,
        ),
        TimeLimit(MAX_MINUTES),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(outdir / "backup_now")),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), write_graph=False),
        tf.keras.callbacks.CSVLogger(str(outdir / f"{tag}_train_log.csv"), append=True),
    ]
    if add_recall_ckpts:
        cbs += [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_best_thin),
                monitor="val_upseg_recall_thin", mode="max",
                save_best_only=True, save_weights_only=True, verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_best_sun),
                monitor="val_upseg_recall_sun", mode="max",
                save_best_only=True, save_weights_only=True, verbose=1,
            ),
        ]
    return cbs, ckpt_best, ckpt_best_seg, ckpt_last

# ============================== MAIN ==============================
def main():
    # Quick checks
    assert BATCH >= 1
    if HARD_BALANCE and HARD_BALANCE_MODE == "combo8":
        assert sum(HARD_QUOTA_COMBO8) == BATCH
    else:
        assert sum(HARD_QUOTA_3P1) == BATCH

    # Runtime config
    set_seeds(SEED); set_m1_policy()
    dl.IMG_SIZE = tuple(IMG_SIZE)

    # Dataset info
    n_train = _count_csv_rows(TRAIN_CSV)
    n_val   = _count_csv_rows(VAL_CSV)
    print(f"[data] rows: train={n_train}  val={n_val}")

    # Datasets
    ds_train, ds_val = make_datasets(
        TRAIN_CSV, VAL_CSV, batch=BATCH,
        shuffle_train=True, seed=SEED,
        augment_train=AUGMENT_TRAIN
    )
    steps_per_epoch = infer_train_steps(TRAIN_CSV, BATCH, TRAIN_AUTO_CAP)
    val_steps       = infer_val_steps(VAL_CSV, BATCH, fallback=300)

    # Model (compiled WITHOUT mIoU)
    now = build_now_compiled(
        img_h=dl.IMG_SIZE[0], img_w=dl.IMG_SIZE[1],
        num_classes=NUM_CLASSES,
        alpha=ALPHA, output_stride=OUTPUT_STRIDE,
        lr_target=LR_TARGET, clipnorm=CLIPNORM,
        imagenet=USE_IMAGENET, lambda_seg=LAMBDA_SEG, lambda_dice=LAMBDA_DICE,
        spe=STEPS_PER_EXEC
    )

    # Initialize GHI bias to the real train median
    _init_ghi_bias_from_csv(now, TRAIN_CSV)

    tag = f"now8h_{dl.IMG_SIZE[0]}x{dl.IMG_SIZE[1]}_a{ALPHA}_os{OUTPUT_STRIDE}_b{BATCH}_seed{SEED}_hb-{HARD_BALANCE_MODE}"
    callbacks, ck_best, ck_best_seg, ck_last = make_callbacks(tag=tag, outdir=OUTDIR, add_recall_ckpts=True)

    # Add extra callbacks for “GHI air”
    callbacks += [
        FreezeBackbone(FREEZE_BACKBONE_EPOCHS),
        FreezeSegHead(freeze_epochs=1),
        GhiRange(ds_val, n_batches=1),
    ]

    print(f"\n>>> Training NOW  IMG={dl.IMG_SIZE}, OS={OUTPUT_STRIDE}, "
          f"alpha={ALPHA}, batch={BATCH}, λ_seg={LAMBDA_SEG}, λ_dice={LAMBDA_DICE}, "
          f"GHI_W={GHI_WEIGHT}, LR*={LR_TARGET}, SPE={STEPS_PER_EXEC}, "
          f"STEPS/epoch={steps_per_epoch}, VAL_STEPS={val_steps}, "
          f"VAL_EVERY={VALIDATION_EVERY}, MAX_MIN={MAX_MINUTES}")
    now.summary(line_length=120)

    # Resume if previous weights exist (best, best_seg, last)
    def try_load(prefix: Path) -> bool:
        try:
            now.load_weights(str(prefix))
            print(f"[resume] Loaded weights: {prefix}")
            return True
        except Exception as e:
            print(f"[-] Could not load {prefix.name}: {e}")
            return False
    for ck in (ck_best, ck_best_seg, ck_last):
        if try_load(ck): break

    # Optional quick evaluation
    SANITY_EVAL_STEPS = max(10, STEPS_PER_EXEC)
    print(f"[check] Quick eval on val ({SANITY_EVAL_STEPS} steps)…")
    try:
        now.evaluate(ds_val, steps=SANITY_EVAL_STEPS, verbose=2)
    except Exception as e:
        print(f"[warn] sanity eval failed: {e}")

    # Training — TimeLimit stops by wall clock
    history = None
    try:
        history = now.fit(
            ds_train,
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_val,
            validation_steps=val_steps,
            validation_freq=VALIDATION_EVERY,
            epochs=EPOCHS_MAX,
            callbacks=callbacks,
            verbose=1
        )
    finally:
        # Always save “last”
        now.save_weights(str(ck_last))
        print(f"[+] Saved LAST: {ck_last}")

        # meta.json
        meta = {
            "img_size": list(dl.IMG_SIZE),
            "epochs_run": int(history.epoch[-1] + 1) if history else 0,
            "steps_per_epoch": steps_per_epoch,
            "val_steps": val_steps,
            "validation_freq": VALIDATION_EVERY,
            "batch": BATCH,
            "alpha": ALPHA,
            "output_stride": OUTPUT_STRIDE,
            "lambda_seg": LAMBDA_SEG,
            "lambda_dice": LAMBDA_DICE,
            "ghi_weight": GHI_WEIGHT,
            "lr_target": LR_TARGET,
            "lr_warmup_epochs": LR_WARMUP_EPOCHS,
            "clipnorm": CLIPNORM,
            "seed": SEED,
            "augment_train": AUGMENT_TRAIN,
            "use_imagenet": USE_IMAGENET,
            "num_classes": NUM_CLASSES,
            "steps_per_execution": STEPS_PER_EXEC,
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "outdir": str(OUTDIR),
            "tag": tag,
            "checkpoints": {
                "best_ghi": str(ck_best),
                "best_segloss": str(ck_best_seg),
                "last": str(ck_last),
            },
            "data_rows": {"train": n_train, "val": n_val},
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "max_minutes": MAX_MINUTES,
            "balance_soft": BALANCE_SOFT,
            "hard_balance": HARD_BALANCE,
            "hard_quota_3p1": HARD_QUOTA_3P1,
        }
        meta.update({
            "hard_balance_mode": HARD_BALANCE_MODE,
            "hard_quota_combo8": HARD_QUOTA_COMBO8,
        })
        with open(OUTDIR / f"{tag}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_jsonify)
        print(f"[+] Saved meta: {OUTDIR / f'{tag}_meta.json'}")

        if history is not None and hasattr(history, "history"):
            hist_path = OUTDIR / f"{tag}_history.json"
            safe_hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            with open(hist_path, "w") as f:
                json.dump(safe_hist, f, indent=2)
            print(f"[+] Saved history: {hist_path}")

    print("\nNOW training finished.")
    print(f"- Best (GHI):   {ck_best}")
    print(f"- Best (seg):   {ck_best_seg}")
    print(f"- Last:         {ck_last}")

if __name__ == "__main__":
    main()
