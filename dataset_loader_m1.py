# -*- coding: utf-8 -*-
"""
Dataset loader for sky nowcasting/forecasting (with BRBG+CDOC FUSION).

Exposes:
  - build_now_dataset(...)
  - build_future_dataset(...)
  - peek_batch(ds, k=1)

Compatibility:
  - IMG_SIZE = (320, 320)
  - _load_image(path) is used by infer_now_demo.py
  - _denormalize_ghi(x) to convert back to W/m² (identity if already SI)
  - Mask colors → one-hot (num_classes = 5)

Fusion rules (CDOC has priority):
  - SUN   = BRBG_sun
  - THIN  = CDOC_cloud_fina (224,224,224)
  - THICK = CDOC_cloud_gruesa (149,149,149)  ∪  (CDOC_void ∧ BRBG_cloud)
  - SKY   = CDOC_sky (84,84,255)              ∪  (CDOC_void ∧ BRBG_sky)
  - VOID  = ¬(SUN ∪ THIN ∪ THICK ∪ SKY)  (outside the circle)

Notes:
  - Class balancing and losses/metrics are decided ONLY inside the circle.
"""

from typing import Optional, Tuple, List
import tensorflow as tf
import pandas as pd
import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k):  # simple fallback without progress bar
        return x

# ---------------------------------------------------------------------------
# Basic configuration
# ---------------------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

# Default image size (height, width)
IMG_SIZE: Tuple[int, int] = (320, 320)

# Default shuffle buffer size (None = whole dataset)
DEFAULT_BUFFER_SIZE: Optional[int] = 10000

# RGB colors for classes (5 classes):
#   0: VOID, 1: SKY, 2: THICK (dense cloud), 3: THIN (thin cloud), 4: SUN
VOID_RGB  = (0, 0, 0)
SKY_RGB   = (84, 84, 255)
THICK_RGB = (149, 149, 149)
THIN_RGB  = (224, 224, 224)
SUN_RGB   = (255, 255, 0)

_CLASS_COLORS: List[Tuple[int, int, int]] = [
    VOID_RGB,   # 0: VOID
    SKY_RGB,    # 1: SKY
    THICK_RGB,  # 2: THICK
    THIN_RGB,   # 3: THIN
    SUN_RGB,    # 4: SUN
]

# Palette per mask source (BRBG/CDOC)
BRBG_SKY   = SKY_RGB
BRBG_CLOUD = THICK_RGB
BRBG_SUN   = SUN_RGB
CDOC_SKY    = SKY_RGB
CDOC_CLOUD1 = THICK_RGB
CDOC_CLOUD2 = THIN_RGB
CDOC_VOID   = VOID_RGB

# ---------------- Fixed circular mask ----------------
VALID_MASK_PATH = "data/processed/valid_mask_320.npy"
_VM_NP = np.load(VALID_MASK_PATH).astype("float32")
if _VM_NP.ndim == 2:
    _VM_NP = _VM_NP[..., None]
_VM_TF = tf.constant(_VM_NP)  # (H0,W0,1)

def _circle_mask_like(y):  # y: (..., H, W, C)
    """Resize the precomputed circular validity mask to match y."""
    H = tf.shape(y)[-3]; W = tf.shape(y)[-2]
    m = tf.image.resize(_VM_TF, [H, W], method="nearest")
    return tf.clip_by_value(m, 0., 1.)  # (..., H, W, 1)

# ---------------- Example-level class inside the circle ----------------
def _class_id_from_y(y):
    """
    Returns an example-level class ID decided ONLY inside the geometric circle:
      0=SUN, 1=THICK, 2=THIN, 3=REST (REST = SKY and/or VOID; THIN requires CDOC).
    """
    m = y["upseg"]                             # (...,H,W,5) [VOID, SKY, THICK, THIN, SUN]
    M = _circle_mask_like(m)                   # (...,H,W,1)
    has_sun   = tf.reduce_any((m[..., 4:5] * M) > 0.5)
    has_thick = tf.reduce_any((m[..., 2:3] * M) > 0.5)
    has_thin  = tf.reduce_any((m[..., 3:4] * M) > 0.5)
    return tf.where(has_sun, 0,
           tf.where(has_thick, 1,
           tf.where(has_thin,  2, 3)))

# ---------------- GHI units ----------------
GHI_MODE: str = "si"               # "si" | "normalized"
GHI_SCALE: float = 1000.0          # normalization scale (≈ kW/m²)
GHI_MAX_NORMALIZED: float = 1.2    # safety clip when using normalized mode

def _ghi_to_target(ghi: tf.Tensor) -> tf.Tensor:
    """Convert raw GHI to training target based on GHI_MODE."""
    ghi = tf.cast(ghi, tf.float32)
    if GHI_MODE == "normalized":
        ghi = tf.clip_by_value(ghi / GHI_SCALE, 0.0, GHI_MAX_NORMALIZED)
    return ghi

def _denormalize_ghi(x: tf.Tensor) -> tf.Tensor:
    """Convert model outputs back to W/m² if normalized; identity otherwise."""
    x = tf.cast(x, tf.float32)
    if GHI_MODE == "normalized":
        return x * GHI_SCALE
    return x

# ---------------------------------------------------------------------------
# Image/mask utilities
# ---------------------------------------------------------------------------

def _decode_rgb(img_bytes: tf.Tensor) -> tf.Tensor:
    """Decode bytes → RGB uint8 tensor with static shape (H,W,3)."""
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.cast(img, tf.uint8)

def _to_u8_resized(path: tf.Tensor) -> tf.Tensor:
    """Load image as uint8 and resize (nearest) to IMG_SIZE for palette matching."""
    b = tf.io.read_file(path)
    img = _decode_rgb(b)
    return tf.cast(tf.image.resize(img, IMG_SIZE, method="nearest"), tf.uint8)

def _load_image(path: tf.Tensor) -> tf.Tensor:
    """Load image as float32 in [0,1] and resize (bilinear) to IMG_SIZE."""
    b = tf.io.read_file(path)
    img = _decode_rgb(b)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return tf.cast(img, tf.float32) / 255.0

def _eq_rgb(img_u8: tf.Tensor, rgb: Tuple[int, int, int]) -> tf.Tensor:
    """Per-pixel equality test with an RGB color; returns (H,W) bool."""
    col = tf.constant(rgb, dtype=tf.uint8, shape=[1, 1, 3])
    return tf.reduce_all(tf.equal(img_u8, col), axis=-1)

def _onehot_from_palette(mask_u8: tf.Tensor, class_colors=_CLASS_COLORS) -> tf.Tensor:
    """Map color palette to one-hot channels (num_classes)."""
    channels = []
    for col in class_colors:
        channels.append(_eq_rgb(mask_u8, col))
    onehot = tf.stack(channels, axis=-1)
    return tf.cast(onehot, tf.float32)

def _load_mask(path: tf.Tensor, class_colors=_CLASS_COLORS) -> tf.Tensor:
    """Load palette mask and convert to one-hot with given class_colors."""
    m = _to_u8_resized(path)
    return _onehot_from_palette(m, class_colors)

# -------------------------- F U S I O N ------------------------------------

def _fuse_brbg_cdoc(brbg_path: tf.Tensor, cdoc_path: tf.Tensor) -> tf.Tensor:
    """
    BRBG+CDOC fusion → one-hot with 5 channels [VOID, SKY, THICK, THIN, SUN].
    Implements the rules described at the top (CDOC priority).
    """
    br = _to_u8_resized(brbg_path)
    cd = _to_u8_resized(cdoc_path)

    # CDOC channels
    c_void  = _eq_rgb(cd, CDOC_VOID)
    c_sky   = _eq_rgb(cd, CDOC_SKY)
    c_thick = _eq_rgb(cd, CDOC_CLOUD1)
    c_thin  = _eq_rgb(cd, CDOC_CLOUD2)

    # BRBG channels
    b_sun   = _eq_rgb(br, BRBG_SUN)
    b_sky   = _eq_rgb(br, BRBG_SKY)
    b_cloud = _eq_rgb(br, BRBG_CLOUD)

    # Fusion logic
    sun   = b_sun
    thin  = c_thin
    thick = tf.logical_or(c_thick, tf.logical_and(c_void, b_cloud))
    sky   = tf.logical_or(c_sky, tf.logical_and(c_void, b_sky))

    any_cls = tf.logical_or(sun, tf.logical_or(thin, tf.logical_or(thick, sky)))
    void = tf.logical_not(any_cls)

    onehot = tf.stack([void, sky, thick, thin, sun], axis=-1)
    return tf.cast(onehot, tf.float32)

# ---------------------------------------------------------------------------
# Window packers (for FUTURE)
# ---------------------------------------------------------------------------

def _pack_window(
    raw_paths: tf.Tensor,
    mask_paths: tf.Tensor,
    ghi_seq: tf.Tensor,
    class_colors=_CLASS_COLORS,
    augment: bool = False,
):
    """
    Pack a window of inputs/targets for standard (non-fused) masks.
    Returns:
      imgs: (T,H,W,3)
      targets: {'upseg': (T,H,W,C), 'ghi_pred': (T,1)}
    """
    n_classes = len(class_colors)

    imgs = tf.map_fn(
        _load_image, raw_paths,
        fn_output_signature=tf.TensorSpec((*IMG_SIZE, 3), tf.float32)
    )
    masks = tf.map_fn(
        lambda p: _load_mask(p, class_colors),
        mask_paths,
        fn_output_signature=tf.TensorSpec((*IMG_SIZE, n_classes), tf.float32)
    )

    if augment:
        imgs, masks = _apply_augment_seq(imgs, masks)

    ghi_tgt = tf.expand_dims(_ghi_to_target(ghi_seq), -1)
    return imgs, {"upseg": masks, "ghi_pred": ghi_tgt}

def _pack_window_fused(
    raw_paths: tf.Tensor,
    brbg_paths: tf.Tensor,
    cdoc_paths: tf.Tensor,
    ghi_seq: tf.Tensor,
    augment: bool = False,
):
    """
    Pack a window for fused BRBG+CDOC masks using the fusion rules.
    """
    imgs = tf.map_fn(
        _load_image, raw_paths,
        fn_output_signature=tf.TensorSpec((*IMG_SIZE, 3), tf.float32)
    )

    T = tf.shape(raw_paths)[0]

    def _fuse_t(t):
        return _fuse_brbg_cdoc(brbg_paths[t], cdoc_paths[t])

    masks = tf.map_fn(
        _fuse_t,
        elems=tf.range(T),
        fn_output_signature=tf.TensorSpec((*IMG_SIZE, len(_CLASS_COLORS)), tf.float32)
    )

    if augment:
        imgs, masks = _apply_augment_seq(imgs, masks)

    ghi_tgt = tf.expand_dims(_ghi_to_target(ghi_seq), -1)
    return imgs, {"upseg": masks, "ghi_pred": ghi_tgt}

# ---------------------------------------------------------------------------
# Light augmentations
# ---------------------------------------------------------------------------

def _apply_augment_img(img: tf.Tensor, mask: tf.Tensor):
    """Apply simple flip/rotations + slight brightness/contrast jitter to a single image/mask."""
    do_flip = tf.random.uniform([]) < 0.5
    img  = tf.cond(do_flip, lambda: tf.image.flip_left_right(img),  lambda: img)
    mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)

    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img  = tf.image.rot90(img,  k)
    mask = tf.image.rot90(mask, k)

    img = tf.image.random_brightness(img, max_delta=0.03)
    img = tf.image.random_contrast(img, lower=0.97, upper=1.03)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, mask

def _apply_augment_seq(imgs, masks):
    """Apply the same random flip/rotation/photometric jitter consistently across a sequence."""
    do_flip = tf.random.uniform([]) < 0.5
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    b = tf.random.uniform([], -0.02, 0.02)
    c = tf.random.uniform([], 0.98, 1.02)

    def _aug_one(i, m):
        if do_flip:
            i = tf.image.flip_left_right(i); m = tf.image.flip_left_right(m)
        i = tf.image.rot90(i, k); m = tf.image.rot90(m, k)
        i = tf.clip_by_value(tf.image.adjust_brightness(i, b), 0., 1.)
        i = tf.image.adjust_contrast(i, c)
        return i, m

    T = tf.shape(imgs)[0]
    i2, m2 = tf.map_fn(
        lambda t: _aug_one(imgs[t], masks[t]),
        elems=tf.range(T),
        fn_output_signature=(tf.float32, tf.float32),
    )
    return i2, m2

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_now_dataset(
    csv_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    seed_number: Optional[int] = None,
    cache_path: Optional[str] = None,  # caching disabled by default (enable by passing a path)
    reshuffle_each_iteration_value: bool = False,
    buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE,
    num_classes: int = len(_CLASS_COLORS),
    augment: bool = False,
    drop_remainder: bool = True,
    # Balancing
    balance: bool = False,                 # "soft" average balancing
    hard_balance: bool = False,            # "strict" per-batch quotas
    # target SUN, THICK, THIN, REST → must sum to 1.0 ("soft" mode)
    target_dist_3p1: tuple = (0.60, 0.35, 0.05, 0.0),  # approx real: (0.844, 0.139, 0.017, 0.0)
    # quotas SUN, THICK, THIN, REST → must sum to batch_size ("strict" mode)
    hard_quota_3p1: tuple = (5, 5, 5, 1),
    hard_balance_combo8: bool = False,               # NEW
    hard_balance_mode: str = "3p1",              # "3p1" | "combo8"
    hard_quota_combo8: tuple = (2,2,2,2,2,2,2,2) # default 2×8=16
) -> tf.data.Dataset:
    """
    Phase 1 (NOW): single image → {'upseg': mask, 'ghi_pred': ghi_target}
    """
    if seed_number is not None:
        tf.random.set_seed(seed_number)

    if balance and hard_balance:
        raise ValueError("Use ONLY one: balance=True (soft) or hard_balance=True (strict).")

    df = pd.read_csv(csv_path).copy()

    # Temporal order
    if "dt" in df.columns:
        df["__dt"] = pd.to_datetime(df["dt"], errors="coerce")
    else:
        df["__dt"] = pd.to_datetime(df["stamp"], errors="coerce", format="%Y%m%d%H%M%S")
    df = df.sort_values("__dt").drop(columns=["__dt"]).reset_index(drop=True)

    # Class count check
    n_classes = len(_CLASS_COLORS)
    if num_classes != n_classes:
        raise ValueError(f"num_classes={num_classes} but the palette has {n_classes} classes.")

    # Fusion available?
    use_fusion = {"mask_brbg_path", "mask_cdoc_path"}.issubset(df.columns)
    if use_fusion:
        # Filter out rows with empty/NaN paths before creating the dataset
        for c in ("mask_brbg_path", "mask_cdoc_path"):
            df[c] = df[c].astype(str)
        bad = (df["mask_brbg_path"].str.strip() == "") | df["mask_brbg_path"].str.lower().eq("nan") | \
              (df["mask_cdoc_path"].str.strip() == "") | df["mask_cdoc_path"].str.lower().eq("nan")
        if bad.any():
            print(f"Descartando {int(bad.sum())} filas sin BRBG/CDOC para fusión.")
            df = df[~bad].reset_index(drop=True)

    if shuffle and buffer_size is None:
        buffer_size = len(df)

    if use_fusion:
        ds = tf.data.Dataset.from_tensor_slices((
            df["raw_path"].values,
            df["mask_brbg_path"].values,
            df["mask_cdoc_path"].values,
            df["ghi"].astype("float32").values,
        ))
        def _map_now(r, b, c, g):
            img  = _load_image(r)
            mask = _fuse_brbg_cdoc(b, c)
            if augment:
                img, mask = _apply_augment_img(img, mask)
            ghi  = tf.expand_dims(_ghi_to_target(g), -1)
            return img, {"upseg": mask, "ghi_pred": ghi}
    else:
        mask_col = "mask_brbg_path" if "mask_brbg_path" in df.columns else "mask_path"
        ds = tf.data.Dataset.from_tensor_slices((
            df["raw_path"].values,
            df[mask_col].values,
            df["ghi"].astype("float32").values,
        ))
        def _map_now(r, m, g):
            img  = _load_image(r)
            mask = _load_mask(m)  # 5 channels; THIN will remain 0 if not available
            if augment:
                img, mask = _apply_augment_img(img, mask)
            ghi  = tf.expand_dims(_ghi_to_target(g), -1)
            return img, {"upseg": mask, "ghi_pred": ghi}

    ds = ds.map(_map_now, num_parallel_calls=AUTOTUNE, deterministic=False)

    # ======= Optional class balancing =======
    if balance and not hard_balance:
        target = tf.cast(tf.constant(target_dist_3p1), tf.float32)

        # ALWAYS use the global function (with circle)
        def _class_fn(x, y):
            return _class_id_from_y(y)

        if hasattr(tf.data.Dataset, "rejection_resample"):
            # Modern API: emits (class, (x,y))
            ds = ds.rejection_resample(class_func=_class_fn,
                                       target_dist=target,
                                       seed=seed_number)
        else:
            # Experimental fallback
            from tensorflow.data.experimental import rejection_resample
            ds = ds.map(lambda x, y: ((x, y), _class_fn(x, y)),
                        num_parallel_calls=AUTOTUNE, deterministic=False)
            ds = ds.apply(rejection_resample(
                class_func=lambda data, cls: cls,
                target_dist=target,
                seed=seed_number
            ))

        # Normalize back to (x, {'upseg':…, 'ghi_pred':…})
        ds = ds.map(
            lambda cls, data: (data[0], {"upseg": data[1]["upseg"], "ghi_pred": data[1]["ghi_pred"]}),
            num_parallel_calls=AUTOTUNE, deterministic=False
        )

    # --- Strict per-batch balancing with fixed quotas ---
    if hard_balance:
        if hard_balance_mode not in {"3p1","combo8"}:
            raise ValueError("hard_balance_mode must be '3p1' or 'combo8'.")

        if hard_balance_mode == "combo8":
            col_name = "now_combo8"; quotas = list(hard_quota_combo8); nclasses = 8
        else:
            col_name = "now_cls";    quotas = list(hard_quota_3p1);    nclasses = 4

        if col_name not in df.columns:
            raise ValueError(f"hard_balance={hard_balance_mode} but CSV lacks column '{col_name}'.")

        df[col_name] = df[col_name].astype(int)

        def _make_stream(df_sub, quota):
            """Creates an infinite, shuffled, batched stream for one class/combo."""
            if quota <= 0 or len(df_sub) == 0: return None
            if use_fusion:
                base = tf.data.Dataset.from_tensor_slices((
                    df_sub["raw_path"].values,
                    df_sub["mask_brbg_path"].values,
                    df_sub["mask_cdoc_path"].values,
                    df_sub["ghi"].astype("float32").values,
                ))
                def _loader(r, b, c, g):
                    img  = _load_image(r)
                    mask = _fuse_brbg_cdoc(b, c)
                    if augment: img, mask = _apply_augment_img(img, mask)
                    ghi  = tf.expand_dims(_ghi_to_target(g), -1)
                    return img, {"upseg": mask, "ghi_pred": ghi}
            else:
                mask_col = "mask_brbg_path" if "mask_brbg_path" in df.columns else "mask_path"
                base = tf.data.Dataset.from_tensor_slices((
                    df_sub["raw_path"].values,
                    df_sub[mask_col].values,
                    df_sub["ghi"].astype("float32").values,
                ))
                def _loader(r, m, g):
                    img  = _load_image(r)
                    mask = _load_mask(m)
                    if augment: img, mask = _apply_augment_img(img, mask)
                    ghi  = tf.expand_dims(_ghi_to_target(g), -1)
                    return img, {"upseg": mask, "ghi_pred": ghi}

            return (base.shuffle(4096)
                        .repeat()
                        .map(_loader, num_parallel_calls=AUTOTUNE, deterministic=False)
                        .batch(quota, drop_remainder=True))

        # Build streams per class/combo
        streams_by_class = []
        for k in range(nclasses):
            streams_by_class.append(_make_stream(df[df[col_name] == k], quotas[k]))

        present_mask = [s is not None for s in streams_by_class]
        avail_ids = [i for i, ok in enumerate(present_mask) if ok]
        if not avail_ids:
            raise ValueError("hard_balance: no available combination with quota > 0.")

        present_quota = sum(quotas[i] for i in avail_ids)
        missing_total = max(0, batch_size - present_quota)

        # Keep only existing streams
        streams = [streams_by_class[i] for i in avail_ids]

        # Backfill to complete batch size if needed
        if missing_total > 0:
            fallback_df = df[df[col_name].isin(avail_ids)]
            fb = _make_stream(fallback_df, missing_total)
            if fb is not None:
                streams.append(fb)
            else:
                print("[warn] backfill no disponible; el batch quedará más pequeño.")

        if len(streams) == 1:
            ds = streams[0]
        else:
            merged = tf.data.Dataset.zip(tuple(streams))
            def _merge(*bs):
                xs = [b[0] for b in bs]
                up = [b[1]["upseg"] for b in bs]
                gh = [b[1]["ghi_pred"] for b in bs]
                return (tf.concat(xs, 0),
                        {"upseg": tf.concat(up, 0),
                        "ghi_pred": tf.concat(gh, 0)})
            ds = merged.map(_merge, num_parallel_calls=AUTOTUNE, deterministic=False)

        # Shuffle within the final batch
        def _shuffle_in_batch(x, y):
            idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            y2 = {k: tf.gather(v, idx) for k, v in y.items()}
            return tf.gather(x, idx), y2
        ds = ds.map(_shuffle_in_batch, num_parallel_calls=AUTOTUNE)

        if cache_path: ds = ds.cache(cache_path)
        if shuffle:
            ds = ds.shuffle(buffer_size if buffer_size is not None else len(df),
                            seed=seed_number,
                            reshuffle_each_iteration=reshuffle_each_iteration_value)
        return ds.prefetch(AUTOTUNE)


    # ======= end balancing =======  (← continues after hard_balance)
    # --- Strict balancing by SUN/THICK/THIN combinations (8 buckets) ---
    if hard_balance_combo8:
        if hard_balance:
            raise ValueError("Use ONLY one: hard_balance_combo8=True or hard_balance=True, not both.")
        # Requires has_sun/has_thick/has_thin columns precomputed
        needed_cols = {"has_sun","has_thick","has_thin"}
        if not needed_cols.issubset(df.columns):
            raise ValueError(
                "hard_balance_combo8=True but CSV lacks has_sun/has_thick/has_thin.\n"
                "Generate *_with_cls.csv with compute_now_class_column_with_details()."
            )

        # Combos order: (S,T,N) = 000,001,010,011,100,101,110,111
        combos = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
        quotas = list(hard_quota_combo8)
        if sum(quotas) != batch_size:
            raise ValueError(f"hard_quota_combo8 must sum to batch_size={batch_size}, got {sum(quotas)}.")

        def _make_stream(df_sub, quota):
            if quota <= 0 or len(df_sub) == 0:
                return None
            if use_fusion:
                base = tf.data.Dataset.from_tensor_slices((
                    df_sub["raw_path"].values,
                    df_sub["mask_brbg_path"].values,
                    df_sub["mask_cdoc_path"].values,
                    df_sub["ghi"].astype("float32").values,
                ))
                def _loader(r, b, c, g):
                    img  = _load_image(r)
                    mask = _fuse_brbg_cdoc(b, c)
                    if augment:
                        img, mask = _apply_augment_img(img, mask)
                    ghi  = tf.expand_dims(_ghi_to_target(g), -1)
                    return img, {"upseg": mask, "ghi_pred": ghi}
            else:
                mask_col = "mask_brbg_path" if "mask_brbg_path" in df.columns else "mask_path"
                base = tf.data.Dataset.from_tensor_slices((
                    df_sub["raw_path"].values,
                    df_sub[mask_col].values,
                    df_sub["ghi"].astype("float32").values,
                ))
                def _loader(r, m, g):
                    img  = _load_image(r)
                    mask = _load_mask(m)
                    if augment:
                        img, mask = _apply_augment_img(img, mask)
                    ghi  = tf.expand_dims(_ghi_to_target(g), -1)
                    return img, {"upseg": mask, "ghi_pred": ghi}

            return (base.shuffle(4096)
                        .repeat()
                        .map(_loader, num_parallel_calls=AUTOTUNE, deterministic=False)
                        .batch(quota, drop_remainder=True))

        streams = []
        for (s,t,n), q in zip(combos, quotas):
            sub = df[(df.has_sun==s) & (df.has_thick==t) & (df.has_thin==n)]
            streams.append(_make_stream(sub, q))

        # Filter out empty streams and warn
        exist = [i for i,s in enumerate(streams) if s is not None]
        if not exist:
            raise ValueError("hard_balance_combo8: no available combination with quota > 0.")
        missing = [i for i in range(8) if i not in exist]
        if missing:
            print(f"[warn] combos sin datos/cuota>0: {missing} (se omiten)")

        streams = [streams[i] for i in exist]

        # Merge all streams pairwise (supports N streams)
        def _merge2(b0, b1):
            return (tf.concat([b0[0], b1[0]], axis=0),
                    {"upseg": tf.concat([b0[1]["upseg"], b1[1]["upseg"]], axis=0),
                     "ghi_pred": tf.concat([b0[1]["ghi_pred"], b1[1]["ghi_pred"]], axis=0)})

        ds = streams[0]
        for s in streams[1:]:
            ds = tf.data.Dataset.zip((ds, s)).map(_merge2, num_parallel_calls=AUTOTUNE)

        def _shuffle_in_batch(x, y):
            idx = tf.random.shuffle(tf.range(tf.shape(x)[0]))
            y2 = {k: tf.gather(v, idx) for k, v in y.items()}
            return tf.gather(x, idx), y2

        ds = ds.map(_shuffle_in_batch, num_parallel_calls=AUTOTUNE)
        if cache_path:
            ds = ds.cache(cache_path)
        if shuffle:
            ds = ds.shuffle(buffer_size if buffer_size is not None else len(df),
                            seed=seed_number, reshuffle_each_iteration=reshuffle_each_iteration_value)
        return ds.prefetch(AUTOTUNE)

    # Normalize structure (ensure y has the expected dict fields)
    ds = ds.map(
        lambda x, y: (x, {"upseg": y["upseg"], "ghi_pred": y["ghi_pred"]}),
        num_parallel_calls=AUTOTUNE, deterministic=False
    )

    # Cache ONLY if a path is provided
    if cache_path:
        ds = ds.cache(cache_path)

    # Standard pipeline: optional shuffle → batch → prefetch
    if shuffle:
        ds = ds.shuffle(
            buffer_size if buffer_size is not None else len(df),
            seed=seed_number,
            reshuffle_each_iteration=reshuffle_each_iteration_value
        )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
    return ds

def build_future_dataset(
        csv_path: str,
        timesteps: int = 6,
        stride: Optional[int] = None,
        batch_size: int = 4,
        seed_number: Optional[int] = None,
        shuffle: bool = True,
        cache_path: Optional[str] = None,  # caching disabled by default
        buffer_size: Optional[int] = DEFAULT_BUFFER_SIZE,
        reshuffle_each_iteration_value: bool = False,
        num_classes: int = len(_CLASS_COLORS),
        augment: bool = False,
        drop_remainder: bool = True,
    ) -> tf.data.Dataset:
        """
        Phase 2 (FUTURE): T inputs → T outputs (future masks and GHI).
        Does not cross natural days to avoid temporal leakage.
        """
        if seed_number is not None:
            tf.random.set_seed(seed_number)

        df = pd.read_csv(csv_path).copy()

        # Robust datetime
        if "dt" in df.columns:
            df["__dt"] = pd.to_datetime(df["dt"], errors="coerce")
        else:
            df["__dt"] = pd.to_datetime(df["stamp"], errors="coerce", format="%Y%m%d%H%M%S")
        df = df.sort_values("__dt").reset_index(drop=True)

        n_classes = len(_CLASS_COLORS)
        if num_classes != n_classes:
            raise ValueError(f"num_classes={num_classes} but the palette has {n_classes} classes.")

        if stride is None:
            stride = timesteps

        use_fusion = {"mask_brbg_path", "mask_cdoc_path"}.issubset(df.columns)
        if use_fusion:
            for c in ("mask_brbg_path", "mask_cdoc_path"):
                df[c] = df[c].astype(str)
            bad = (df["mask_brbg_path"].str.strip() == "") | df["mask_brbg_path"].str.lower().eq("nan") | \
                (df["mask_cdoc_path"].str.strip() == "") | df["mask_cdoc_path"].str.lower().eq("nan")
            if bad.any():
                print(f"Descartando {int(bad.sum())} filas sin BRBG/CDOC para fusión.")
                df = df[~bad].reset_index(drop=True)

        seq_raws, seq_ghi = [], []
        if use_fusion:
            seq_brbg, seq_cdoc = [], []
        else:
            seq_masks = []

        # Group by natural day
        for _, group in df.groupby(df["__dt"].dt.date, sort=True):
            group = group.sort_values("__dt")

            raw_g  = group["raw_path"].values
            ghi_g  = group["ghi"].astype("float32").values
            if use_fusion:
                brbg_g = group["mask_brbg_path"].values
                cdoc_g = group["mask_cdoc_path"].values
            else:
                mask_col = "mask_brbg_path" if "mask_brbg_path" in group.columns else "mask_path"
                mask_g = group[mask_col].values

            M = len(raw_g)
            if M < 2 * timesteps:
                continue

            # Windows: inputs [j : j+T), outputs [j+T : j+2T)
            for j in range(0, M - 2 * timesteps + 1, stride):
                seq_raws.append(raw_g[j : j + timesteps])
                if use_fusion:
                    seq_brbg.append(brbg_g[j + timesteps : j + 2 * timesteps])
                    seq_cdoc.append(cdoc_g[j + timesteps : j + 2 * timesteps])
                else:
                    seq_masks.append(mask_g[j + timesteps : j + 2 * timesteps])
                seq_ghi.append(ghi_g[j + timesteps : j + 2 * timesteps])

        if not seq_raws:
            raise ValueError("No hay suficientes secuencias (revisa timesteps/CSV).")

        total_blocks = len(seq_raws)
        if shuffle and buffer_size is None:
            buffer_size = total_blocks

        if use_fusion:
            ds = tf.data.Dataset.from_tensor_slices((
                np.array(seq_raws),
                np.array(seq_brbg),
                np.array(seq_cdoc),
                np.array(seq_ghi, dtype="float32"),
            ))
            mapper = lambda r, b, c, g: _pack_window_fused(r, b, c, g, augment=augment)
        else:
            ds = tf.data.Dataset.from_tensor_slices((
                np.array(seq_raws),
                np.array(seq_masks),
                np.array(seq_ghi, dtype="float32"),
            ))
            mapper = lambda r, m, g: _pack_window(r, m, g, _CLASS_COLORS, augment=augment)

        ds = ds.map(mapper, num_parallel_calls=AUTOTUNE, deterministic=False)

        if cache_path:
            ds = ds.cache(cache_path)

        if shuffle:
            ds = ds.shuffle(buffer_size, seed=seed_number,
                            reshuffle_each_iteration=reshuffle_each_iteration_value)

        ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
        return ds

# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def peek_batch(ds: tf.data.Dataset, k: int = 1):
    """Print shapes of a few batches for quick sanity checks."""
    for i, (x, y) in enumerate(ds.take(k)):
        upseg = y["upseg"]
        ghi   = y["ghi_pred"]
        print(f"[Batch {i}] x: {x.shape}  upseg: {upseg.shape}  ghi_pred: {ghi.shape}")

# ============================ FAST DIST ESTIMATOR ============================

def fast_estimate_now_dist(csv_path: str, sample_limit: int = 20000, downsample: int = 8, seed: int = 1337):
    """
    Very fast estimation of (SUN, THICK, THIN, REST) distribution:
      - Reads ONLY masks (BRBG/CDOC), no RAW images or tf.data graph.
      - BRBG+CDOC fusion in NumPy (same rule as _fuse_brbg_cdoc).
      - Considers ONLY pixels inside the circular mask (VALID_MASK_PATH).
      - downsample: skip pixels (e.g., 8 => use 1 of each 8 in H and W).
      - sample_limit: max number of CSV rows to sample.
    """
    import pandas as pd
    import numpy as np
    try:
        from PIL import Image
        _use_pil = True
    except Exception:
        _use_pil = False
        import tensorflow as tf

    df = pd.read_csv(csv_path).copy()

    # Robust temporal ordering
    if "dt" in df.columns:
        df["__dt"] = pd.to_datetime(df["dt"], errors="coerce")
    else:
        df["__dt"] = pd.to_datetime(df["stamp"], errors="coerce", format="%Y%m%d%H%M%S")
    df = df.sort_values("__dt").reset_index(drop=True)

    # Fusion availability
    has_fusion = {"mask_brbg_path", "mask_cdoc_path"}.issubset(df.columns)
    if has_fusion:
        for c in ("mask_brbg_path", "mask_cdoc_path"):
            df[c] = df[c].astype(str)
        bad = (df["mask_brbg_path"].str.strip() == "") | df["mask_brbg_path"].str.lower().eq("nan") | \
              (df["mask_cdoc_path"].str.strip() == "") | df["mask_cdoc_path"].str.lower().eq("nan")
        df = df[~bad].reset_index(drop=True)
    else:
        # Without CDOC we can't detect THIN; it will be ~0
        mask_col = "mask_brbg_path" if "mask_brbg_path" in df.columns else "mask_path"
        df[mask_col] = df[mask_col].astype(str)
        bad = (df[mask_col].str.strip() == "") | df[mask_col].str.lower().eq("nan")
        df = df[~bad].reset_index(drop=True)

    n = len(df)
    if n == 0:
        print("[fast_estimate] CSV sin filas válidas.")
        return [0, 0, 0, 1]

    # Uniform temporal sampling (more stable than pure shuffle)
    rng = np.random.default_rng(seed)
    if sample_limit and sample_limit < n:
        idxs = np.linspace(0, n - 1, num=sample_limit, dtype=int)
    else:
        idxs = np.arange(n, dtype=int)

    # Circular mask (downsampled)
    M0 = _VM_NP[..., 0] > 0.5  # (H0,W0)
    M  = M0[::downsample, ::downsample]  # bool

    counts = np.zeros(4, dtype=np.int64)  # SUN, THICK, THIN, REST

    def _read_u8_rgb(path: str) -> np.ndarray:
        """Read RGB uint8, resize to IMG_SIZE, then downsample for speed."""
        if _use_pil:
            with Image.open(path) as im:
                im = im.convert("RGB")
                if tuple(im.size)[::-1] != tuple(IMG_SIZE):
                    im = im.resize((IMG_SIZE[1], IMG_SIZE[0]), resample=Image.NEAREST)
                if downsample > 1:
                    im = im.resize((IMG_SIZE[1] // downsample, IMG_SIZE[0] // downsample), resample=Image.NEAREST)
                arr = np.array(im, dtype=np.uint8)
        else:
            b = tf.io.read_file(path)
            img = tf.image.decode_image(b, channels=3, expand_animations=False)
            img = tf.cast(img, tf.uint8)
            img = tf.image.resize(img, IMG_SIZE, method="nearest")
            if downsample > 1:
                img = tf.image.resize(img, (IMG_SIZE[0] // downsample, IMG_SIZE[1] // downsample), method="nearest")
            arr = img.numpy()
        return arr

    def _eq_rgb_np(arr: np.ndarray, rgb: tuple) -> np.ndarray:
        """Vectorized equality for palette matching."""
        return np.all(arr == np.array(rgb, dtype=np.uint8), axis=-1)

    for i in idxs:
        if has_fusion:
            br = _read_u8_rgb(df.at[i, "mask_brbg_path"])
            cd = _read_u8_rgb(df.at[i, "mask_cdoc_path"])

            c_void  = _eq_rgb_np(cd, CDOC_VOID)
            c_thin  = _eq_rgb_np(cd, CDOC_CLOUD2)
            c_thick = _eq_rgb_np(cd, CDOC_CLOUD1)
            c_sky   = _eq_rgb_np(cd, CDOC_SKY)

            b_sun   = _eq_rgb_np(br, BRBG_SUN)
            b_cloud = _eq_rgb_np(br, BRBG_CLOUD)
            b_sky   = _eq_rgb_np(br, BRBG_SKY)

            sun   = b_sun
            thin  = c_thin
            thick = np.logical_or(c_thick, np.logical_and(c_void, b_cloud))
            sky   = np.logical_or(c_sky,   np.logical_and(c_void, b_sky))
        else:
            m = _read_u8_rgb(df.at[i, mask_col])
            # Without CDOC: THIN ≈ 0
            sun   = _eq_rgb_np(m, BRBG_SUN)
            thick = _eq_rgb_np(m, BRBG_CLOUD)
            thin  = np.zeros_like(thick, dtype=bool)
            sky   = _eq_rgb_np(m, BRBG_SKY)

        # Restrict to circle and decide example-level class (presence > 0)
        inside = M
        has_sun   = bool(np.any(sun[inside]))
        has_thick = bool(np.any(thick[inside]))
        has_thin  = bool(np.any(thin[inside]))

        if has_sun:
            counts[0] += 1
        elif has_thick:
            counts[1] += 1
        elif has_thin:
            counts[2] += 1
        else:
            counts[3] += 1

    total = int(counts.sum())
    dist = (counts / max(total, 1)).tolist()
    print(f"[fast_estimate] counts={counts.tolist()}  dist={dist}  (sampled={len(idxs)}, down={downsample}x)")
    return dist

# python -c "import src.dataset_loader_m1 as dl; dl.fast_estimate_now_dist('data/processed/pairs_train.csv', sample_limit=20000, downsample=8)"

def compute_now_class_column_with_details(csv_in: str,
                                          csv_out: str = None,
                                          downsample: int = 8) -> str:
    """
    Label each row using BRBG+CDOC fusion and add columns:
      - now_cls: 0=SUN,1=THICK,2=THIN,3=REST (priority SUN>THICK>THIN>REST)
      - now_src: 'SUN:BRBG_sun' | 'THICK:CDOC_cloud1' | 'THICK:CDOC_void&BRBG_cloud' | 'THIN:CDOC_cloud2' | 'REST'
      - has_sun/has_thick/has_thin/has_sky (0/1, inside the circle)
    Uses PIL and subsampling for speed. Robust to any 'downsample' value.
    """
    import pandas as pd, numpy as np
    from PIL import Image

    try:
        from tqdm import tqdm
    except Exception:
        def tqdm(x, **kw): return x  # fallback without progress bar

    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    df = pd.read_csv(csv_in).copy()

    # Robust temporal ordering
    if "dt" in df.columns:
        df["__dt"] = pd.to_datetime(df["dt"], errors="coerce")
    else:
        df["__dt"] = pd.to_datetime(df["stamp"], errors="coerce", format="%Y%m%d%H%M%S")
    df = df.sort_values("__dt").drop(columns=["__dt"]).reset_index(drop=True)

    # Requires both masks
    if not {"mask_brbg_path", "mask_cdoc_path"}.issubset(df.columns):
        raise ValueError("Missing 'mask_brbg_path' and/or 'mask_cdoc_path' columns.")

    # Base circle mask (320x320 bool) from the module
    M0 = (_VM_NP[..., 0] > 0.5)

    # --- helpers ---
    def _read_u8(path: str, out_hw=None) -> np.ndarray:
        """Read RGB u8, force to IMG_SIZE, then optionally subsample to out_hw."""
        with Image.open(path) as im:
            im = im.convert("RGB")
            if tuple(im.size) != (IMG_SIZE[1], IMG_SIZE[0]):
                im = im.resize((IMG_SIZE[1], IMG_SIZE[0]), resample=Image.NEAREST)
            if out_hw is not None:
                Ht, Wt = out_hw
                im = im.resize((Wt, Ht), resample=Image.NEAREST)
            return np.asarray(im, dtype=np.uint8)

    def _eq(arr: np.ndarray, rgb: tuple) -> np.ndarray:
        return np.all(arr == np.array(rgb, np.uint8), axis=-1)

    def _resize_circle_like(H: int, W: int) -> np.ndarray:
        """Resize the circular mask to (H,W) using nearest (no aliasing)."""
        mp = Image.fromarray((M0.astype(np.uint8) * 255))
        mp = mp.resize((W, H), resample=Image.NEAREST)
        return (np.array(mp) > 127)

    # Subsampled output size (floor is OK; circle is resized accordingly)
    Hds = max(1, IMG_SIZE[0] // downsample)
    Wds = max(1, IMG_SIZE[1] // downsample)

    now_cls, now_src = [], []
    has_sun, has_thick, has_thin, has_sky = [], [], [], []
    now_combo8 = []  # <-- NEW

    for i in tqdm(range(len(df)), desc="[preindex] fusionando", unit="img"):
        br = _read_u8(df.at[i, "mask_brbg_path"], out_hw=(Hds, Wds))
        cd = _read_u8(df.at[i, "mask_cdoc_path"], out_hw=(Hds, Wds))
        H, W = br.shape[:2]
        if cd.shape[:2] != (H, W):
            cd = np.array(Image.fromarray(cd).resize((W, H), resample=Image.NEAREST))
        inside = _resize_circle_like(H, W)

        # CDOC
        c_thin  = _eq(cd, CDOC_CLOUD2)
        c_thick = _eq(cd, CDOC_CLOUD1)
        c_sky   = _eq(cd, CDOC_SKY)
        c_void  = _eq(cd, CDOC_VOID)
        # BRBG
        b_sun   = _eq(br, BRBG_SUN)
        b_cloud = _eq(br, BRBG_CLOUD)
        b_sky   = _eq(br, BRBG_SKY)

        sun    = b_sun
        thin   = c_thin
        thick1 = c_thick
        thick2 = np.logical_and(c_void, b_cloud)
        thick  = np.logical_or(thick1, thick2)
        sky    = np.logical_or(c_sky, np.logical_and(c_void, b_sky))

        _has_sun   = int(np.any(sun[inside]))
        _has_thick = int(np.any(thick[inside]))
        _has_thin  = int(np.any(thin[inside]))
        _has_sky   = int(np.any(sky[inside]))

        # combo8 id = S(4) + THICK(2) + THIN(1)
        now_combo8_id = (_has_sun << 2) | (_has_thick << 1) | _has_thin

        has_sun.append(_has_sun)
        has_thick.append(_has_thick)
        has_thin.append(_has_thin)
        has_sky.append(_has_sky)
        now_combo8.append(now_combo8_id)

        if _has_sun:
            now_cls.append(0); now_src.append("SUN:BRBG_sun")
        elif _has_thick:
            src = "THICK:CDOC_cloud1" if np.any(thick1[inside]) else "THICK:CDOC_void&BRBG_cloud"
            now_cls.append(1); now_src.append(src)
        elif _has_thin:
            now_cls.append(2); now_src.append("THIN:CDOC_cloud2")
        else:
            now_cls.append(3); now_src.append("REST")

    df["now_cls"]   = now_cls
    df["now_src"]   = now_src
    df["has_sun"]   = has_sun
    df["has_thick"] = has_thick
    df["has_thin"]  = has_thin
    df["has_sky"]   = has_sky
    df["now_combo8"] = now_combo8

    out = csv_out or csv_in.replace(".csv", "_with_cls.csv")
    df.to_csv(out, index=False)

    # Summary
    vc = pd.Series(now_cls).value_counts().reindex([0, 1, 2, 3], fill_value=0)
    tot = int(vc.sum())
    dist = [float(vc[k]) / tot if tot else 0.0 for k in [0, 1, 2, 3]]
    print(f"[preindex] guardado {out} | counts={vc.to_dict()} | dist={dist}")
    return out

# caffeinate -d -i python -c "from src.dataset_loader_m1 import compute_now_class_column_with_details as go; go('data/processed/pairs_train.csv','data/processed/pairs_train_with_cls.csv',downsample=12); go('data/processed/pairs_val.csv','data/processed/pairs_val_with_cls.csv',downsample=8)"

