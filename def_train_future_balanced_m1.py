# -*- coding: utf-8 -*-
# Run:  caffeinate -d -i python src/def_train_future_balanced_m1.py
# FUTURE — Phase 3 (Balanced ~4h). Trade-off between SEG and GHI.
# Resume priority: ghiphase_best → segphase_best_segloss → best → best_segloss → last.

import os, time, json, math, datetime as dt
from pathlib import Path
from typing import Tuple

# M1: legacy optimizers
os.environ.setdefault("TF_USE_LEGACY_KERAS_OPTIMIZERS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Allow RNN_TYPE via environment variable (optional)
_ENV_RNN = os.environ.get("RNN_TYPE", "").strip().lower()

import numpy as np
import pandas as pd
import tensorflow as tf

# ==== PROJECT MODULES ====
import dataset_loader_m1 as dl
import models_m1 as models

# ===================== PARAMETERS (EDIT HERE) =====================
RNN_TYPE            = _ENV_RNN if _ENV_RNN in ("gru","lstm") else "gru"   # "gru" | "lstm"

IMG_SIZE: Tuple[int, int] = (320, 320)
TIMESTEPS           = 6
STRIDE              = 6
BATCH               = 4
EPOCHS_MAX          = 999
MAX_MINUTES         = 240          # ~4h
STEPS_PER_EPOCH     = "auto"
TRAIN_AUTO_CAP      = 2000
VAL_STEPS           = "auto"
VAL_AUTO_CAP        = 300
VALIDATION_EVERY    = 1
ALPHA               = 0.5
OUTPUT_STRIDE       = 16
LR_TARGET           = 5e-5         # gentler so as not to "break" what was learned
LR_WARMUP_EPOCHS    = 3
CLIPNORM            = 1.0

# Balanced weights
LAMBDA_SEG          = 1.0          # SEG still carries good weight
GHI_WEIGHT          = 0.04          # GHI slightly emphasized
LAMBDA_DICE         = 1.0

AUGMENT_TRAIN       = True
USE_IMAGENET        = True
SEED                = 1337
STEPS_PER_EXEC      = 16
NUM_CLASSES         = 5

# Paths
ROOT = Path(__file__).resolve().parents[1]
SRC  = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed"
TRAIN_CSV = DATA / "pairs_train.csv"
VAL_CSV   = DATA / "pairs_val.csv"

_RNN = RNN_TYPE.lower().strip()
assert _RNN in ("gru", "lstm")
OUTDIR = ROOT / (f"checkpoints_future_{_RNN}")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[paths] ROOT={ROOT}")
print(f"[paths] DATA={DATA}")
print(f"[cfg]   FUTURE RNN={_RNN.upper()}  → OUTDIR={OUTDIR}")

# ---- Runtime utils ----
def set_seeds(seed: int) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed); np.random.seed(seed); random.seed(seed)

def set_m1_policy() -> None:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

def _jsonify(o):
    import numpy as np
    if hasattr(o, "numpy"): o = o.numpy()
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.ndarray,)):  return o.tolist()
    raise TypeError(f"Type {type(o).__name__} not serializable")

def ensure_targets_in_SI(ds: tf.data.Dataset, name: str) -> tf.data.Dataset:
    if getattr(dl, "GHI_MODE", "si") == "normalized":
        print(f"[{name}] Convirtiendo objetivos GHI a W/m² vía dl._denormalize_ghi…")
        @tf.function
        def _to_si(img, y):
            y2 = dict(y)
            y2["ghi_pred"] = dl._denormalize_ghi(y2["ghi_pred"])
            return img, y2
        return ds.map(_to_si, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    else:
        print(f"[{name}] GHI ya en W/m² (dl.GHI_MODE='si').")
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

# ---- Datasets ----
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

# ---- Losses ----
class BalancedCELoss(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, name="balanced_ce", **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name=name, **kw)
        self.from_logits = from_logits
    def call(self, y_true, y_pred):
        M = models.valid_mask(y_true)
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

# ---- FUTURE model — balanced compile ----
def build_future_compiled_balanced(img_h: int, img_w: int, num_classes: int, timesteps: int,
                                   alpha: float, output_stride: int, lr_target: float,
                                   clipnorm: float, imagenet: bool,
                                   lambda_seg: float, lambda_dice: float,
                                   rnn_type: str,
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
        opt = Adam(learning_rate=lr_target, clipnorm=clipnorm); print("[opt] legacy.Adam")
    except Exception:
        opt = tf.keras.optimizers.Adam(learning_rate=lr_target, clipnorm=clipnorm); print("[opt] Adam")

    losses, loss_w, metrics = {}, {}, {}
    for out_name in fut.output_names:
        if "ghi" in out_name:
            losses[out_name]  = ghi_mean
            loss_w[out_name]  = float(GHI_WEIGHT)
            metrics[out_name] = [tf.keras.metrics.MeanAbsoluteError(name="mae")]
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

    fut.compile(optimizer=opt, loss=losses, loss_weights=loss_w, metrics=metrics,
                steps_per_execution=int(spe))
    return fut

def try_resume_future_with_priority(outdir: Path, model: tf.keras.Model) -> bool:
    """
    Priority when resuming this BALANCED phase:
      1) *_balanced_best_mix.index      ← if you are relaunching balanced
      2) *_ghiphase_best.index          ← good starting point for GHI
      3) *segphase_best_segloss.index   ← good starting point for SEG
      4) *_best.index                   ← best generic GHI
      5) *_best_segloss.index           ← best generic segmentation
      6) *_last.index                   ← last checkpoint
    """
    patterns = [
        "*balanced_best_mix.index",
        "*ghiphase_best.index",
        "*segphase_best_segloss.index",
        "*_best.index",
        "*_best_segloss.index",
        "*_last.index",
    ]
    for pat in patterns:
        files = sorted(outdir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            prefix = str(f)[:-len(".index")]
            try:
                model.load_weights(prefix)
                print(f"[init] Cargado checkpoint: {f.name}")
                return True
            except Exception as e:
                print(f"[-] Falló cargar {f.name}: {e}")
    print("[-] No se cargaron pesos previos en FUTURE.")
    return False

# ---- Callbacks (monitor mix, GHI and SEG) ----
def make_callbacks(tag: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_best_mix = outdir / f"{tag}_best_mix"       # monitor: val_loss (mixed objective)
    ckpt_best_ghi = outdir / f"{tag}_best"           # monitor: GHI
    ckpt_best_seg = outdir / f"{tag}_best_segloss"   # monitor: SEG
    ckpt_last     = outdir / f"{tag}_last"
    log_dir = outdir / "logs" / (tag + "_" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cbs = [
        # Best SEG+GHI compromise (the loss is already weighted)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_mix),
            monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1),
        # Specific monitors
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_ghi),
            monitor="val_ghi_pred_mae", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_best_seg),
            monitor="val_upseg_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_last),
            save_best_only=False, save_weights_only=True, verbose=0),

        # Schedulers / early-stop
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=5, cooldown=1, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=14, min_delta=0.0, restore_best_weights=False, verbose=1),

        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir), write_graph=False),
        tf.keras.callbacks.CSVLogger(str(outdir / f"{tag}_train_log.csv"), append=True),
    ]

    # Soft guards to avoid excessive degradation of a branch
    class SegGuard(tf.keras.callbacks.Callback):
        def __init__(self): super().__init__(); self.best=None
        def on_epoch_end(self, epoch, logs=None):
            if logs is None: return
            cur = logs.get("val_upseg_loss")
            if cur is None: return
            if self.best is None: self.best = cur
            elif cur > 1.15 * self.best:
                lr_var = getattr(self.model.optimizer, "lr", None) or getattr(self.model.optimizer, "learning_rate", None)
                new_lr = float(tf.keras.backend.get_value(lr_var)) * 0.5
                tf.keras.backend.set_value(lr_var, new_lr)
                print(f"[SegGuard] val_upseg_loss subió >15% (de {self.best:.3f} a {cur:.3f}). LR -> {new_lr:.2e}")

    class GHIGuard(tf.keras.callbacks.Callback):
        def __init__(self): super().__init__(); self.best=None
        def on_epoch_end(self, epoch, logs=None):
            if logs is None: return
            cur = logs.get("val_ghi_pred_mae")
            if cur is None: return
            if self.best is None: self.best = cur
            elif cur > 1.10 * self.best:
                lr_var = getattr(self.model.optimizer, "lr", None) or getattr(self.model.optimizer, "learning_rate", None)
                new_lr = float(tf.keras.backend.get_value(lr_var)) * 0.7
                tf.keras.backend.set_value(lr_var, new_lr)
                print(f"[GHIGuard] val_ghi_pred_mae empeoró >10% (de {self.best:.1f} a {cur:.1f}). LR -> {new_lr:.2e}")

    cbs.insert(0, GHIGuard())
    cbs.insert(0, SegGuard())

    return cbs, ckpt_best_mix, ckpt_best_ghi, ckpt_best_seg, ckpt_last

# ============================== MAIN ==============================
def main():
    set_seeds(SEED); set_m1_policy()
    dl.IMG_SIZE = tuple(IMG_SIZE)

    ds_train, ds_val = make_datasets(
        TRAIN_CSV, VAL_CSV, batch=BATCH,
        timesteps=TIMESTEPS, stride=STRIDE,
        shuffle_train=True, seed=SEED,
        augment_train=AUGMENT_TRAIN
    )
    steps_per_epoch = infer_train_steps_future(TRAIN_CSV, BATCH, TRAIN_AUTO_CAP, TIMESTEPS, STRIDE)
    val_steps       = infer_val_steps_future(VAL_CSV, BATCH, 200, TIMESTEPS, STRIDE)

    fut = build_future_compiled_balanced(
        img_h=dl.IMG_SIZE[0], img_w=dl.IMG_SIZE[1],
        num_classes=NUM_CLASSES, timesteps=TIMESTEPS,
        alpha=ALPHA, output_stride=OUTPUT_STRIDE,
        lr_target=LR_TARGET, clipnorm=CLIPNORM,
        imagenet=USE_IMAGENET,
        lambda_seg=LAMBDA_SEG, lambda_dice=LAMBDA_DICE,
        rnn_type=_RNN, spe=STEPS_PER_EXEC
    )

    # Load best available with priority for balanced/ghi/seg
    _ = try_resume_future_with_priority(OUTDIR, fut)

    tag = f"future{_RNN.upper()}_T{TIMESTEPS}_s{STRIDE}_{dl.IMG_SIZE[0]}x{dl.IMG_SIZE[1]}_a{ALPHA}_os{OUTPUT_STRIDE}_b{BATCH}_seed{SEED}_balanced"
    callbacks, ck_best_mix, ck_best_ghi, ck_best_seg, ck_last = make_callbacks(tag=tag, outdir=OUTDIR)

    print(f"\n>>> FUTURE-BAL({_RNN.upper()}) IMG={dl.IMG_SIZE}, T={TIMESTEPS}, stride={STRIDE}, "
          f"OS={OUTPUT_STRIDE}, α={ALPHA}, batch={BATCH}, λ_seg={LAMBDA_SEG}, λ_dice={LAMBDA_DICE}, "
          f"GHI_W={GHI_WEIGHT}, LR*={LR_TARGET}, SPE={STEPS_PER_EXEC}, "
          f"STEPS/epoch={steps_per_epoch}, VAL_STEPS={val_steps}, VAL_EVERY={VALIDATION_EVERY}, "
          f"MAX_MIN={MAX_MINUTES}")
    fut.summary(line_length=120)

    # Sanity eval
    SANITY_EVAL_STEPS = max(10, STEPS_PER_EXEC)
    print(f"[check] Eval rápida en val ({SANITY_EVAL_STEPS} pasos)…")
    try:
        fut.evaluate(ds_val, steps=SANITY_EVAL_STEPS, verbose=2)
    except Exception as e:
        print(f"[warn] sanity eval falló: {e}")

    history = None
    try:
        # Simple linear warmup
        def _warmup_fn(epoch, lr):
            frac = min(1.0, (epoch + 1) / max(1, LR_WARMUP_EPOCHS))
            return float(LR_TARGET * (0.25 + 0.75 * frac))

        class TimeLimit(tf.keras.callbacks.Callback):
            def __init__(self, max_minutes: float):
                super().__init__(); self.max_seconds=float(max_minutes)*60.0; self.t0=None
            def on_train_begin(self, logs=None): self.t0=time.time()
            def _should_stop(self) -> bool: return (time.time() - self.t0) >= self.max_seconds
            def on_batch_end(self, batch, logs=None):
                if self._should_stop():
                    print(f"\n[TimeLimit] Límite {self.max_seconds/60:.1f} min alcanzado. Parando…")
                    self.model.stop_training = True
            def on_test_batch_end(self, batch, logs=None):
                if self._should_stop():
                    print(f"\n[TimeLimit] Límite {self.max_seconds/60:.1f} min alcanzado (val). Parando…")
                    self.model.stop_training = True

        callbacks = [tf.keras.callbacks.LearningRateScheduler(_warmup_fn, verbose=0)] + callbacks + [TimeLimit(MAX_MINUTES)]

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
        print(f"[+] Guardado LAST: {ck_last}")

        meta = {
            "phase": "balanced-finetune",
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
            "checkpoints": {
                "best_mix": str(ck_best_mix),
                "best_ghi": str(ck_best_ghi),
                "best_segloss": str(ck_best_seg),
                "last": str(ck_last)
            },
            "time": dt.datetime.now().isoformat(timespec="seconds"),
            "max_minutes": MAX_MINUTES
        }
        with open(OUTDIR / f"{tag}_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=_jsonify)
        if history is not None and hasattr(history, "history"):
            hist_path = OUTDIR / f"{tag}_history.json"
            safe_hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
            with open(hist_path, "w") as f:
                json.dump(safe_hist, f, indent=2)
            print(f"[+] Guardado history: {hist_path}")

    print("\nFUTURE (fase BALANCED) finalizado.")
    print(f"- Mejor (MIX val_loss): {ck_best_mix}")
    print(f"- Mejor (GHI):         {ck_best_ghi}")
    print(f"- Mejor (SEG):         {ck_best_seg}")
    print(f"- Último:              {ck_last}")

if __name__ == "__main__":
    main()
