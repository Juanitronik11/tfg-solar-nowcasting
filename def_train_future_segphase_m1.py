# -*- coding: utf-8 -*-
# Run:  caffeinate -d -i python src/def_train_future_segphase_m1.py
# FUTURE — Phase 1 (segmentation-only, ~4h). Initializes from NOW if weights exist.

import os, time, json, math, datetime as dt
from pathlib import Path
from typing import Tuple

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
# Recurrent cell for FUTURE: "gru" or "lstm"
RNN_TYPE            = "lstm"

IMG_SIZE: Tuple[int, int] = (320, 320)
TIMESTEPS           = 6
STRIDE              = 6
BATCH               = 4
EPOCHS_MAX          = 999
MAX_MINUTES         = 240          # ← 4 hours
STEPS_PER_EPOCH     = "auto"
TRAIN_AUTO_CAP      = 2000
VAL_STEPS           = "auto"
VAL_AUTO_CAP        = 300
VALIDATION_EVERY    = 1
ALPHA               = 0.5
OUTPUT_STRIDE       = 16
LR_TARGET           = 1e-4
LR_WARMUP_EPOCHS    = 3
CLIPNORM            = 1.0

# Loss weights (SEG phase)
LAMBDA_SEG          = 1.0
GHI_WEIGHT          = 0.0          # ← CRITICAL: disable GHI in the loss
LAMBDA_DICE         = 1.0

AUGMENT_TRAIN       = True
USE_IMAGENET        = True
SEED                = 1337
STEPS_PER_EXEC      = 16
NUM_CLASSES         = 5

# Optional: freeze attention and GHI head in SEG phase
FREEZE_ATTN         = True
FREEZE_GHI_HEAD     = True

# Which NOW checkpoint folder to pull weights from if present:
NOW_DIR_NAME        = "checkpoints_now8h_seg"

# ================= PATHS =================
ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TRAIN_CSV = DATA / "pairs_train.csv"
VAL_CSV   = DATA / "pairs_val.csv"

_RNN = RNN_TYPE.lower().strip()
assert _RNN in ("gru", "lstm"), f"RNN_TYPE must be 'gru' or 'lstm', got: {RNN_TYPE}"
OUTDIR = ROOT / (f"checkpoints_future_{_RNN}")
OUTDIR.mkdir(parents=True, exist_ok=True)

NOW_CKPT_DIR = ROOT / NOW_DIR_NAME

print(f"[paths] ROOT={ROOT}")
print(f"[paths] DATA={DATA}")
print(f"[cfg]   FUTURE RNN={_RNN.upper()}  → OUTDIR={OUTDIR}")
print(f"[init]  NOW ckpt dir: {NOW_CKPT_DIR}")

# ---- Reproducibility / M1 ----
def set_seeds(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

def set_m1_policy() -> None:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

# ---- Utilities ----
def _jsonify(o):
    import numpy as np
    if hasattr(o, "numpy"): o = o.numpy()
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    raise TypeError(f"Type {type(o).__name__} not serializable")

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
    return total

def infer_train_steps_future(train_csv: Path, batch: int, cap: int,
                             timesteps: int, stride: int) -> int:
    if isinstance(STEPS_PER_EPOCH, int):
        return max(1, int(STEPS_PER_EPOCH))
    if isinstance(STEPS_PER_EPOCH, str) and STEPS_PER_EPOCH.lower() == "auto":
        n_windows = estimate_future_windows(train_csv, timesteps, stride)
        steps = int(math.ceil(n_windows / float(batch))) if n_windows > 0 else cap
        steps = min(max(1, steps), cap)
        print(f"[train] FUTURE windows≈{n_windows} → STEPS/epoch={steps} (cap={cap})")
        return steps
    return max(1, cap)

def infer_val_steps_future(val_csv: Path, batch: int, fallback: int,
                           timesteps: int, stride: int) -> int:
    if isinstance(VAL_STEPS, int):
        return max(1, int(VAL_STEPS))
    if isinstance(VAL_STEPS, str) and VAL_STEPS.lower() == "auto":
        n_windows = estimate_future_windows(val_csv, timesteps, stride)
        steps = int(math.ceil(n_windows / float(batch))) if n_windows > 0 else fallback
        steps = min(max(1, steps), VAL_AUTO_CAP)
        print(f"[val]   FUTURE windows≈{n_windows} → VAL_STEPS={steps} (cap={VAL_AUTO_CAP})")
        return steps
    return max(1, fallback)

# ---- Dataset builders ----
def make_datasets(train_csv: Path, val_csv: Path, batch: int,
                  timesteps: int, stride: int,
                  shuffle_train: bool = True, seed: int = 1337,
                  augment_train: bool = True):
    SHUF_T = 1024; SHUF_V = 512
    ds_train = dl.build_future_dataset(
        csv_path=str(train_csv),
        timesteps=timesteps, stride=stride,
        batch_size=batch, seed_number=seed,
        shuffle=shuffle_train,
        cache_path=None, buffer_size=SHUF_T,
        reshuffle_each_iteration_value=True,
        num_classes=NUM_CLASSES, augment=augment_train,
        drop_remainder=True
    )
    ds_val = dl.build_future_dataset(
        csv_path=str(val_csv),
        timesteps=timesteps, stride=stride,
        batch_size=batch, seed_number=seed,
        shuffle=False,
        cache_path=None, buffer_size=SHUF_V,
        reshuffle_each_iteration_value=False,
        num_classes=NUM_CLASSES, augment=False,
        drop_remainder=False
    )
    ds_train = ensure_targets_in_SI(ds_train, "train")
    ds_val   = ensure_targets_in_SI(ds_val,   "val")

    opts = tf.data.Options(); opts.experimental_deterministic = False
    ds_train = ds_train.with_options(opts).repeat().prefetch(tf.data.AUTOTUNE)
    ds_val   = ds_val.with_options(opts).repeat().prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_val

# ---- Helpers to (un)freeze by name ----
def set_trainable_by_name(model: tf.keras.Model, substrings, flag: bool) -> int:
    n = 0
    for m in model.submodules:
        if any(s in m.name for s in substrings):
            if hasattr(m, "trainable"):
                m.trainable = flag
                n += 1
    return n

# ---- Losses (compatible with temporal dimension) ----
class BalancedCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name="balanced_ce", **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kw)
        self.from_logits = from_logits
    def call(self, y_true, y_pred):
        M = models.valid_mask(y_true)  # (circle ∧ non-VOID)
        eps = 1e-8
        p = tf.nn.softmax(y_pred, axis=-1) if self.from_logits else y_pred
        ce_map = - y_true * tf.math.log(tf.clip_by_value(p, eps, 1.0))
        axes = tf.range(1, tf.rank(ce_map)-1)
        num = tf.reduce_sum(ce_map * M, axis=axes)
        den = tf.reduce_sum(y_true * M, axis=axes) + eps
        ce_c = num / den
        present = tf.cast(den > eps, tf.float32)
        present = tf.tensor_scatter_nd_update(present, [[0, models.VOID_IDX]], [0.0])
        ce_img = tf.math.divide_no_nan(tf.reduce_sum(ce_c * present, axis=-1),
                                       tf.reduce_sum(present, axis=-1))
        return ce_img

class HistKLDivergence(tf.keras.losses.Loss):
    def __init__(self, lam=0.05, name="hist_kl", **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kw)
        self.lam = tf.constant(lam, tf.float32)
    def call(self, y_true, y_pred):
        M = models.fixed_circle_mask_like(y_true)
        eps = 1e-8
        axes = tf.range(1, tf.rank(y_true)-1)
        p_true = tf.reduce_sum(y_true * M, axis=axes) + eps
        p_pred = tf.reduce_sum(y_pred * M, axis=axes) + eps
        p_true = p_true / tf.reduce_sum(p_true, axis=-1, keepdims=True)
        p_pred = p_pred / tf.reduce_sum(p_pred, axis=-1, keepdims=True)
        kl = tf.reduce_sum(p_true * (tf.math.log(p_true) - tf.math.log(p_pred)), axis=-1)
        return self.lam * kl

# ---- FUTURE model — compiled for SEG-ONLY ----
def build_future_compiled_seg(img_h: int, img_w: int, num_classes: int, timesteps: int,
                              alpha: float, output_stride: int, lr_target: float,
                              clipnorm: float, imagenet: bool,
                              lambda_seg: float, lambda_dice: float,
                              rnn_type: str,
                              freeze_attn: bool = True,
                              freeze_ghi_head: bool = True,
                              spe: int = STEPS_PER_EXEC) -> tf.keras.Model:
    weights = "imagenet" if imagenet else None
    _, fut = models.build_models_shared(
        timesteps=timesteps,
        img_height=img_h, img_width=img_w, img_channels=3,
        num_classes=num_classes,
        rnn_type=rnn_type,
        d=models.D_REPR,
        output_stride=output_stride,
        mbv2_alpha=alpha,
        weights=weights,
        learning_rate=lr_target,
        clipnorm=clipnorm,
        build_future_model=True
    )

    # Freeze attention / GHI head in this phase (optional)
    if freeze_attn:
        n = set_trainable_by_name(fut, ("attn", "attention"), False)
        print(f"[freeze] attention: {n} submodules -> trainable=False")
    if freeze_ghi_head:
        n = set_trainable_by_name(fut, ("ghi_pred_inner", "ghi_head"), False)
        print(f"[freeze] GHI head: {n} submodules -> trainable=False")

    seg_base  = BalancedCELoss(from_logits=False)
    dice_loss = models.TverskyLossMasked(alpha=0.3, beta=0.7, ignore=(models.VOID_IDX,))
    hist_reg  = HistKLDivergence(lam=0.05)

    def seg_total_mean(y_true, y_pred):
        base = seg_base(y_true, y_pred) + LAMBDA_DICE * dice_loss(y_true, y_pred)
        return tf.reduce_mean(base + hist_reg(y_true, y_pred))

    # GHI loss exists but with weight 0.0 it contributes no gradient
    ghi_loss  = models.LogCoshLossSI(m=0.01, low_irr_thr=150.0, low_irr_weight=1.5)
    def ghi_mean(y_true, y_pred):  # kept for compatibility
        return tf.reduce_mean(ghi_loss(y_true, y_pred))

    try:
        from tensorflow.keras.optimizers.legacy import Adam
        opt = Adam(learning_rate=lr_target, clipnorm=clipnorm)
        print("[opt] legacy.Adam")
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_target, clipnorm=clipnorm)
        print("[opt] Adam")

    losses, loss_w, metrics = {}, {}, {}
    for out_name in fut.output_names:
        if "ghi" in out_name:
            losses[out_name]  = ghi_mean
            loss_w[out_name]  = 0.0      # ← no gradient
            metrics[out_name] = []       # avoid unnecessary MAE
        else:
            losses[out_name]  = seg_total_mean
            loss_w[out_name]  = float(lambda_seg)
            metrics[out_name] = [
                models.PixelAccWithMask(name="pixel_acc"),
                models.RecallForClassMasked(models.SKY_IDX,   name="recall_sky"),
                models.RecallForClassMasked(models.THICK_IDX, name="recall_thick"),
                models.RecallForClassMasked(models.THIN_IDX,  name="recall_thin"),
                models.RecallForClassMasked(models.SUN_IDX,   name="recall_sun"),
            ]

    fut.compile(
        optimizer=opt,
        loss=losses,
        loss_weights=loss_w,
        metrics=metrics,
        steps_per_execution=int(spe),
    )
    return fut

# ---- Initial load ----
def try_load_now_into_future(future_model: tf.keras.Model, ckpt_dir: Path) -> bool:
    if not ckpt_dir.exists():
        print(f"[-] NOW ckpt dir not found: {ckpt_dir}")
        return False
    patterns = ["*_best_segloss.index", "*_best.index", "*_last.index"]
    for pat in patterns:
        files = sorted(ckpt_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for idx_file in files:
            prefix = str(idx_file)[:-len(".index")]
            try:
                future_model.load_weights(prefix)
                print(f"[init] Loaded NOW weights into FUTURE from: {idx_file.name}")
                return True
            except Exception as e:
                print(f"[-] Failed to load {idx_file.name}: {e}")
    print("[-] No NOW weights loaded (continuing from ImageNet/scratch).")
    return False

def try_resume_future_from_best_seg(future_model: tf.keras.Model, outdir: Path) -> bool:
    if not outdir.exists(): return False
    # priority: best_segloss -> best_recall_thin -> best -> last
    order = ["*_best_segloss", "*_best_recall_thin", "*_best", "*_last"]
    for pat in order:
        files = sorted(outdir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            try:
                future_model.load_weights(str(p))
                print(f"[resume] FUTURE resumed from: {p.name}")
                return True
            except Exception as e:
                print(f"[-] Could not load {p.name}: {e}")
    return False

# ---- Callbacks (SEG-focused) ----
def make_callbacks(tag: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_best_seg   = outdir / f"{tag}_best_segloss"
    ckpt_best_thin  = outdir / f"{tag}_best_recall_thin"
    ckpt_best_sky   = outdir / f"{tag}_best_recall_sky"
    ckpt_last       = outdir / f"{tag}_last"
    log_dir = outdir / "logs" / (tag + "_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_seg),
            monitor="val_upseg_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        # optional per-class monitors
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_thin),
            monitor="val_upseg_recall_thin", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_sky),
            monitor="val_upseg_recall_sky", mode="max",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_last),
            save_best_only=False, save_weights_only=True, verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        WarmupLR(LR_TARGET, LR_WARMUP_EPOCHS),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_upseg_loss", mode="min",
            factor=0.5, patience=5, cooldown=1, min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_upseg_loss", mode="min",
            patience=14, min_delta=0.0, restore_best_weights=False, verbose=1,
        ),
        TimeLimit(MAX_MINUTES),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(outdir / "backup_future_segphase")),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), write_graph=False),
        tf.keras.callbacks.CSVLogger(str(outdir / f"{tag}_train_log.csv"), append=True),
    ]
    return cbs, ckpt_best_seg, ckpt_last

# ============================== MAIN ==============================
def main():
    set_seeds(SEED); set_m1_policy()
    dl.IMG_SIZE = tuple(IMG_SIZE)

    # Datasets
    ds_train, ds_val = make_datasets(
        TRAIN_CSV, VAL_CSV, batch=BATCH,
        timesteps=TIMESTEPS, stride=STRIDE,
        shuffle_train=True, seed=SEED,
        augment_train=AUGMENT_TRAIN
    )
    steps_per_epoch = infer_train_steps_future(TRAIN_CSV, BATCH, TRAIN_AUTO_CAP, TIMESTEPS, STRIDE)
    val_steps       = infer_val_steps_future(VAL_CSV, BATCH, 200, TIMESTEPS, STRIDE)

    # FUTURE model (SEG Phase)
    fut = build_future_compiled_seg(
        img_h=dl.IMG_SIZE[0], img_w=dl.IMG_SIZE[1],
        num_classes=NUM_CLASSES, timesteps=TIMESTEPS,
        alpha=ALPHA, output_stride=OUTPUT_STRIDE,
        lr_target=LR_TARGET, clipnorm=CLIPNORM,
        imagenet=USE_IMAGENET,
        lambda_seg=LAMBDA_SEG, lambda_dice=LAMBDA_DICE,
        rnn_type=_RNN,
        freeze_attn=FREEZE_ATTN,
        freeze_ghi_head=FREEZE_GHI_HEAD,
        spe=STEPS_PER_EXEC
    )

    # Resume from FUTURE (best seg) or initialize from NOW
    resumed = try_resume_future_from_best_seg(fut, OUTDIR)
    if not resumed:
        try_load_now_into_future(fut, NOW_CKPT_DIR)

    tag = f"future{_RNN.upper()}SEG_T{TIMESTEPS}_s{STRIDE}_{dl.IMG_SIZE[0]}x{dl.IMG_SIZE[1]}_a{ALPHA}_os{OUTPUT_STRIDE}_b{BATCH}_seed{SEED}_segphase"
    callbacks, ck_best_seg, ck_last = make_callbacks(tag=tag, outdir=OUTDIR)

    print(f"\n>>> FUTURE-SEG({_RNN.upper()})  IMG={dl.IMG_SIZE}, T={TIMESTEPS}, stride={STRIDE}, "
          f"OS={OUTPUT_STRIDE}, alpha={ALPHA}, batch={BATCH}, λ_seg={LAMBDA_SEG}, "
          f"λ_dice={LAMBDA_DICE}, LR*={LR_TARGET}, SPE={STEPS_PER_EXEC}, "
          f"STEPS/epoch={steps_per_epoch}, VAL_STEPS={val_steps}, "
          f"VAL_EVERY={VALIDATION_EVERY}, MAX_MIN={MAX_MINUTES}  "
          f"[FREEZE_ATTN={FREEZE_ATTN}, FREEZE_GHI_HEAD={FREEZE_GHI_HEAD}]")
    fut.summary(line_length=120)

    # Mini sanity check
    SANITY_EVAL_STEPS = max(10, STEPS_PER_EXEC)
    print(f"[check] Quick val eval ({SANITY_EVAL_STEPS} steps)…")
    try:
        fut.evaluate(ds_val, steps=SANITY_EVAL_STEPS, verbose=2)
    except Exception as e:
        print(f"[warn] sanity eval failed: {e}")

    history = None
    try:
        history = fut.fit(
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
        fut.save_weights(str(ck_last))
        print(f"[+] Saved LAST: {ck_last}")

        meta = {
            "phase": "seg-only",
            "rnn_type": _RNN,
            "img_size": list(dl.IMG_SIZE),
            "timesteps": TIMESTEPS, "stride": STRIDE,
            "epochs_run": int(history.epoch[-1] + 1) if history else 0,
            "steps_per_epoch": steps_per_epoch,
            "val_steps": val_steps,
            "validation_freq": VALIDATION_EVERY,
            "batch": BATCH, "alpha": ALPHA, "output_stride": OUTPUT_STRIDE,
            "lambda_seg": LAMBDA_SEG, "lambda_dice": LAMBDA_DICE,
            "ghi_weight": GHI_WEIGHT,
            "lr_target": LR_TARGET, "lr_warmup_epochs": LR_WARMUP_EPOCHS,
            "clipnorm": CLIPNORM, "seed": SEED,
            "augment_train": AUGMENT_TRAIN, "use_imagenet": USE_IMAGENET,
            "steps_per_execution": STEPS_PER_EXEC,
            "train_csv": str(TRAIN_CSV), "val_csv": str(VAL_CSV),
            "outdir": str(OUTDIR), "tag": tag,
            "checkpoints": {"best_segloss": str(ck_best_seg), "last": str(ck_last)},
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "max_minutes": MAX_MINUTES,
            "now_ckpt_dir": str(NOW_CKPT_DIR),
            "freeze_attn": FREEZE_ATTN,
            "freeze_ghi_head": FREEZE_GHI_HEAD,
        }
        with open(OUTDIR / f"{tag}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_jsonify)
        if history is not None and hasattr(history, "history"):
            hist_path = OUTDIR / f"{tag}_history.json"
            safe_hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            with open(hist_path, "w") as f:
                json.dump(safe_hist, f, indent=2)
            print(f"[+] Saved history: {hist_path}")

    print("\nFUTURE (SEG phase) finished.")
    print(f"- Best (seg_loss): {ck_best_seg}")
    print(f"- Last:           {ck_last}")

# ---- Handy Callbacks ----
class WarmupLR(tf.keras.callbacks.Callback):
    def __init__(self, lr_target: float, warmup_epochs: int):
        super().__init__(); self.lr_target=float(lr_target); self.warmup_epochs=int(warmup_epochs)
    def _set_lr(self, value: float):
        opt = self.model.optimizer
        lr_var = getattr(opt, "lr", None) or getattr(opt, "learning_rate", None)
        tf.keras.backend.set_value(lr_var, float(value))
    def on_epoch_begin(self, epoch, logs=None):
        if self.warmup_epochs <= 0: return
        if epoch < self.warmup_epochs:
            frac = 0.25 + 0.75 * (epoch + 1) / self.warmup_epochs
            self._set_lr(self.lr_target * frac)
            print(f"[warmup] epoch {epoch+1}/{self.warmup_epochs}: lr -> {self.lr_target * frac:.6g}")
        elif epoch == self.warmup_epochs:
            self._set_lr(self.lr_target); print(f"[warmup] set target lr: {self.lr_target:.6g}")

class TimeLimit(tf.keras.callbacks.Callback):
    def __init__(self, max_minutes: float):
        super().__init__(); self.max_seconds=float(max_minutes)*60.0; self.t0=None
    def on_train_begin(self, logs=None): self.t0=time.time()
    def _should_stop(self) -> bool: return (time.time() - self.t0) >= self.max_seconds
    def on_batch_end(self, batch, logs=None):
        if self._should_stop():
            print(f"\n[TimeLimit] Limit {self.max_seconds/60:.1f} min reached. Stopping…")
            self.model.stop_training = True
    def on_test_batch_end(self, batch, logs=None):
        if self._should_stop():
            print(f"\n[TimeLimit] Limit {self.max_seconds/60:.1f} min reached (val). Stopping…")
            self.model.stop_training = True

if __name__ == "__main__":
    main()
