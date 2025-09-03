# models_m1.py
# -----------------------------------------------------------------------------
# Efficient nowcasting/forecasting for Apple M1 (5 classes: VOID, SKY, THICK, THIN, SUN).
# - Losses/metrics count ONLY inside the geometric circle and ignore VOID.
# - Additionally, VOID is forced outside the circle in the segmentation output,
#   and attention/GHI only "see" the inside region (hard prior).
# - MobileNetV2 backbone + shared heads (segmentation/GHI) with interpretable attention.
# - Optimizer legacy.Adam with clipnorm (recommended on M1).
# -----------------------------------------------------------------------------

from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam  # recommended on M1

def _mbv2_preproc(x):
    # Equivalent to tf.keras.applications.mobilenet_v2.preprocess_input(255*x)
    # Your images come in [0,1] → convert them to [-1,1]
    return x * 2.0 - 1.0


# --- Fixed validity mask (derived from BRBG) ---
VALID_MASK_PATH = "data/processed/valid_mask_320.npy"   # path to your .npy mask
try:
    _VM_NP = np.load(VALID_MASK_PATH).astype("float32")   # (H,W) or (H,W,1)
except Exception as e:
    raise RuntimeError(f"No se pudo cargar la máscara fija: {VALID_MASK_PATH}") from e  # keep original message

if _VM_NP.ndim == 2:
    _VM_NP = _VM_NP[..., None]
_VM_TF = tf.constant(_VM_NP)  # constant 0/1 tensor

# ----------------------------
# Global parameters
# ----------------------------
IMG_H, IMG_W = 320, 320
NUM_CLASSES = 5  # VOID, SKY, THICK, THIN, SUN

# Class indices in the fused one-hot
VOID_IDX  = 0
SKY_IDX   = 1
THICK_IDX = 2
THIN_IDX  = 3
SUN_IDX   = 4
CLOUD_IDS = (THICK_IDX, THIN_IDX)  # for attention and GHI features

# Class weights (CE/Focal). VOID is unused (masked out); extra weight can be given to THIN/SUN.
# ALPHA_DEFAULT: List[float] = [0.0, 0.1, 0.4, 0.15, 1.90]
ALPHA_DEFAULT: List[float] = [0.0, 1, 1, 1, 1]

# Dimensionality of the compact representation
D_REPR = 8

# ------------------ Geometric circle mask (adjustable) ----------------
CIRCLE_CX_REL = 0.5
CIRCLE_CY_REL = 0.5
CIRCLE_R_REL  = 0.485  # tweak if needed


# =============================================================================
# Valid-pixel masks (circle ∧ non-VOID)
# =============================================================================
def _non_void_mask(y_true: tf.Tensor) -> tf.Tensor:
    """Returns (B,[T,]H,W,1) in {0,1}: 1 where y_true is NOT VOID."""
    void = y_true[..., VOID_IDX:VOID_IDX+1]
    return 1.0 - tf.cast(void, tf.float32)

def fixed_circle_mask_like(y_true: tf.Tensor) -> tf.Tensor:
    """
    Returns the fixed binary mask (B,[T,]H,W,1) loaded from VALID_MASK_PATH,
    resized if H×W differs from the model, and broadcast to batch/time dims.
    """
    Ht = tf.shape(y_true)[-3]
    Wt = tf.shape(y_true)[-2]

    mask = _VM_TF  # (H0,W0,1)
    need_resize = tf.logical_or(
        tf.not_equal(tf.shape(mask)[0], Ht),
        tf.not_equal(tf.shape(mask)[1], Wt)
    )

    def _resize():
        m = tf.image.resize(mask, size=[Ht, Wt], method="nearest")
        return tf.clip_by_value(m, 0.0, 1.0)

    mask_hw = tf.cond(need_resize, _resize, lambda: mask)  # (Ht,Wt,1)

    # Shape and broadcast to (B,[T,]H,W,1)
    rank = tf.rank(y_true)  # 4 or 5
    circ = tf.reshape(mask_hw, (1, 1, Ht, Wt, 1))
    if rank == 4:
        circ = tf.squeeze(circ, axis=1)
    reps = tf.concat([tf.shape(y_true)[:rank-3], [1, 1, 1]], axis=0)
    return tf.tile(circ, reps)

def _circle_const(h: int, w: int) -> tf.Tensor:
    """Mask (1,h,w,1) 0/1 as a constant tensor for graph use (broadcast-safe)."""
    m = _VM_TF
    if int(m.shape[0]) != int(h) or int(m.shape[1]) != int(w):
        m = tf.image.resize(m, [h, w], method="nearest")
    m = tf.clip_by_value(tf.cast(m, tf.float32), 0.0, 1.0)  # (h,w,1)
    return tf.reshape(m, (1, h, w, 1))  # add batch=1

def valid_mask(y_true: tf.Tensor) -> tf.Tensor:
    """Valid = (inside the circle) ∧ (y_true is not VOID)."""
    return fixed_circle_mask_like(y_true) * _non_void_mask(y_true)


# =============================================================================
# ConvGRU2D compatible (Metal/M1)
# =============================================================================
class ConvGRU2DCell(layers.Layer):
    def __init__(
        self, filters: int, kernel_size: Tuple[int, int] = (3, 3),
        padding: str = "same", activation: str = "tanh", recurrent_activation: str = "sigmoid",
        use_bias: bool = True, kernel_initializer: str = "glorot_uniform",
        recurrent_initializer: str = "orthogonal", bias_initializer: str = "zeros",
        name: Optional[str] = None, **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.x_z = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer, name="x_z")
        self.x_r = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer, name="x_r")
        self.x_h = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer, name="x_h")
        self.h_z = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                                 kernel_initializer=self.recurrent_initializer, name="h_z")
        self.h_r = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                                 kernel_initializer=self.recurrent_initializer, name="h_r")
        self.h_h = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=False,
                                 kernel_initializer=self.recurrent_initializer, name="h_h")

    @property
    def state_size(self):
        return tf.TensorShape([None, None, self.filters])

    @property
    def output_size(self):
        return tf.TensorShape([None, None, self.filters])

    def call(self, inputs, states):
        h_prev = states[0]
        z_t = self.recurrent_activation(self.x_z(inputs) + self.h_z(h_prev))
        r_t = self.recurrent_activation(self.x_r(inputs) + self.h_r(h_prev))
        h_tilde = self.activation(self.x_h(inputs) + self.h_h(r_t * h_prev))
        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t, [h_t]


class ConvGRU2DCompat(layers.Layer):
    def __init__(
        self, filters: int, kernel_size: Tuple[int, int] = (3, 3), padding: str = "same",
        return_sequences: bool = True, activation: str = "tanh", recurrent_activation: str = "sigmoid",
        name: Optional[str] = None, **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        cell = ConvGRU2DCell(filters=filters, kernel_size=kernel_size,
                             padding=padding, activation=activation,
                             recurrent_activation=recurrent_activation)
        self.rnn = layers.RNN(cell, return_sequences=return_sequences, return_state=False)

    def call(self, inputs, training=None, mask=None):
        rank = inputs.shape.rank
        shp = tf.shape(inputs)
        if rank == 5:   # (B,T,H,W,C)
            b = shp[0]; h = shp[2]; w = shp[3]
        elif rank == 4: # (B,H,W,C)
            b = shp[0]; h = shp[1]; w = shp[2]
        else:
            raise ValueError(f"ConvGRU2DCompat espera rank 5 o 4, recibido rank={rank}")
        h0 = tf.zeros([b, h, w, self.filters], dtype=inputs.dtype)
        return self.rnn(inputs, initial_state=[h0], training=training, mask=mask)


def _get_rnn_layer(rnn_type: str):
    rt = (rnn_type or "").lower()
    if rt in {"gru"}:
        return ConvGRU2DCompat
    if rt in {"lstm", "conv", "convlstm"}:
        return layers.ConvLSTM2D
    return ConvGRU2DCompat


# =============================================================================
#                                   L O S S E S
# =============================================================================
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.5, alpha: List[float] = ALPHA_DEFAULT,
                 from_logits=False, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.gamma = tf.constant(gamma, tf.float32)
        self.alpha = tf.constant(alpha, tf.float32)
        self.from_logits = from_logits
        self.ce_none = tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logits, reduction=tf.keras.losses.Reduction.NONE, label_smoothing=0.0
        )

    def call(self, y_true, y_pred):
        # Masking: circle ∧ non-VOID
        v = valid_mask(y_true)                          # (B,[T,]H,W,1)
        y_true_m = y_true * v                           # outside circle and VOID → 0

        ce_map = self.ce_none(y_true_m, y_pred)         # (B,[T,]H,W)
        probs = tf.nn.softmax(y_pred, axis=-1) if self.from_logits else y_pred
        p_t = tf.reduce_sum(y_true_m * probs, axis=-1)  # (B,[T,]H,W)
        alpha_t = tf.reduce_sum(self.alpha * y_true_m, axis=-1)

        focal_map = alpha_t * tf.pow(1.0 - p_t, self.gamma) * ce_map
        axes = tf.range(1, tf.rank(focal_map))          # reduce all but batch
        return tf.reduce_mean(focal_map, axis=axes)     # [B]


class CombinedSegLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha: List[float] = ALPHA_DEFAULT,
                 beta=0.5, from_logits=False, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.focal = FocalLoss(gamma, alpha, from_logits)
        self.ce_none = self.focal.ce_none
        self.alpha = self.focal.alpha
        self.beta = tf.constant(beta, tf.float32)

    def call(self, y_true, y_pred):
        v = valid_mask(y_true)
        y_true_m = y_true * v

        ce_map = self.ce_none(y_true_m, y_pred)
        alpha_t = tf.reduce_sum(self.alpha * y_true_m, axis=-1)
        ce = tf.reduce_mean(alpha_t * ce_map, axis=tf.range(1, tf.rank(ce_map)))  # [B]
        fl = self.focal(y_true, y_pred)  # masking already applied
        return fl + self.beta * ce


class TverskyLossMasked(tf.keras.losses.Loss):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, ignore=(VOID_IDX,), **kw):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kw)
        self.alpha=tf.constant(alpha,tf.float32); self.beta=tf.constant(beta,tf.float32)
        self.smooth=tf.constant(smooth,tf.float32); self.ignore=tuple(ignore)

    def call(self, y_true, y_pred):
        v = valid_mask(y_true)
        yt = tf.cast(y_true, tf.float32)*v
        yp = tf.cast(y_pred, tf.float32)*v
        axes = tf.range(1, tf.rank(yt)-1)
        TP = tf.reduce_sum(yt*yp, axis=axes)
        FP = tf.reduce_sum((1.0-yt)*yp, axis=axes)
        FN = tf.reduce_sum(yt*(1.0-yp), axis=axes)
        T  = (TP+self.smooth)/(TP+self.alpha*FP+self.beta*FN+self.smooth)  # (B,C)

        # mask out ignored classes (e.g., VOID, or SKY if desired)
        C = tf.shape(T)[-1]
        keep = tf.ones([C], tf.float32)
        for i in self.ignore:
            keep = tf.tensor_scatter_nd_update(keep, [[i]], [0.0])
        keep = tf.reshape(keep, [1, -1])
        T = T*keep
        denom = tf.reduce_sum(keep)
        return 1.0 - tf.math.divide_no_nan(tf.reduce_sum(T, axis=-1), denom)


# Retro alias
DiceLossIgnoreVoid = TverskyLossMasked


class LogCoshLossSI(tf.keras.losses.Loss):
    def __init__(self, m=0.01, low_irr_thr=150.0, low_irr_weight=1.5, **kwargs):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        self.m = tf.constant(m, tf.float32)
        self._log2 = tf.math.log(tf.constant(2.0, tf.float32))
        self.low_irr_thr = tf.constant(low_irr_thr, tf.float32)
        self.low_irr_weight = tf.constant(low_irr_weight, tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        diff = y_pred - y_true
        z = self.m * diff
        az = tf.abs(z)
        logcosh = az + tf.math.softplus(-2.0 * az) - self._log2
        loss = logcosh / tf.maximum(self.m, 1e-8)

        # extra weight at low irradiance
        if tf.rank(y_true) == 1:
            y_mean = y_true
        else:
            y_mean = tf.reduce_mean(y_true, axis=tf.range(1, tf.rank(y_true)))
        w = tf.where(y_mean < self.low_irr_thr, self.low_irr_weight, 1.0)

        axes = tf.range(1, tf.rank(loss))
        loss_b = tf.reduce_mean(loss, axis=axes)   # [B]
        return w * loss_b


# =============================================================================
#                               M E T R I C S
# =============================================================================
class PixelAccWithMask(tf.keras.metrics.Metric):
    """Accuracy only inside the circle and NON-VOID. Supports (B,[T,]H,W,C)."""
    def __init__(self, name="pixel_acc", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        v = tf.squeeze(valid_mask(y_true), axis=-1)   # (B,[T,]H,W)
        y_true_lbl = tf.argmax(y_true, axis=-1)       # (B,[T,]H,W)
        y_pred_lbl = tf.argmax(y_pred, axis=-1)

        yt = tf.boolean_mask(y_true_lbl, tf.cast(v, tf.bool))
        yp = tf.boolean_mask(y_pred_lbl, tf.cast(v, tf.bool))

        correct = tf.reduce_sum(tf.cast(tf.equal(yt, yp), tf.float32))
        total = tf.cast(tf.size(yt), tf.float32)
        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class RecallForClassMasked(tf.keras.metrics.Metric):
    """Recall for one specific class inside the circle, ignoring VOID."""
    def __init__(self, class_idx: int, name: str = "recall_cls", **kwargs):
        super().__init__(name=name, **kwargs)
        self.class_idx = int(class_idx)
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        v = tf.squeeze(valid_mask(y_true), axis=-1)  # (B,[T,]H,W)
        y_true_lbl = tf.argmax(y_true, axis=-1)
        y_pred_lbl = tf.argmax(y_pred, axis=-1)

        mask = tf.cast(v, tf.bool)
        yt = tf.boolean_mask(y_true_lbl, mask)
        yp = tf.boolean_mask(y_pred_lbl, mask)

        pos = tf.equal(yt, self.class_idx)
        tp = tf.reduce_sum(tf.cast(tf.logical_and(pos, tf.equal(yp, self.class_idx)), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(pos, tf.not_equal(yp, self.class_idx)), tf.float32))
        self.tp.assign_add(tp); self.fn.assign_add(fn)

    def result(self):
        return tf.math.divide_no_nan(self.tp, self.tp + self.fn)

    def reset_state(self):
        self.tp.assign(0.0); self.fn.assign(0.0)


class MeanIoUWithMask(tf.keras.metrics.Metric):
    """mIoU only inside the circle and NON-VOID. Supports (B,[T,]H,W,C)."""
    def __init__(self, num_classes: int, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.cm = self.add_weight(
            name="cm", shape=(num_classes, num_classes),
            initializer="zeros", dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        v = tf.squeeze(valid_mask(y_true), axis=-1)   # (B,[T,]H,W)
        y_true_lbl = tf.argmax(y_true, axis=-1)       # (B,[T,]H,W)
        y_pred_lbl = tf.argmax(y_pred, axis=-1)

        yt = tf.boolean_mask(y_true_lbl, tf.cast(v, tf.bool))
        yp = tf.boolean_mask(y_pred_lbl, tf.cast(v, tf.bool))

        n = tf.shape(yt)[0]
        def _add_cm():
            cm = tf.math.confusion_matrix(yt, yp, num_classes=self.num_classes, dtype=tf.float32)
            self.cm.assign_add(cm); return 0.0
        tf.cond(n > 0, _add_cm, lambda: 0.0)

    def result(self):
        sum_over_row = tf.reduce_sum(self.cm, axis=0)
        sum_over_col = tf.reduce_sum(self.cm, axis=1)
        diag = tf.linalg.diag_part(self.cm)
        denom = sum_over_row + sum_over_col - diag
        iou = tf.math.divide_no_nan(diag, denom)
        return tf.reduce_mean(iou[1:])  # ignore class 0 (VOID)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))


# =============================================================================
#                        B A C K B O N E   (MobileNetV2 OS=16/8)
# =============================================================================
def build_backbone_mbv2(img_h: int, img_w: int, img_c: int = 3,
                        alpha: float = 0.5, output_stride: int = 16,
                        weights: Optional[str] = "imagenet") -> Tuple[tf.keras.Model, int, int]:
    assert output_stride in (8, 16), "Solo OS=8/16 soportados"  # keep original message
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_h, img_w, img_c),
        include_top=False,
        alpha=alpha,
        weights=weights
    )
    if output_stride == 16:
        feat = base.get_layer("block_13_expand_relu").output  # 1/16
    else:
        feat = base.get_layer("block_6_expand_relu").output   # ~1/8
    model = Model(inputs=base.input, outputs=feat, name=f"mbv2_OS{output_stride}_a{alpha}")
    s_h, s_w = model.output_shape[1], model.output_shape[2]
    return model, s_h, s_w


# =============================================================================
#             C L O U D / S U N   A T T E N T I O N   (without VOID)
# =============================================================================
def _build_coord_grids_2d(h: int, w: int):
    xs = tf.linspace(0.0, 1.0, w)
    ys = tf.linspace(0.0, 1.0, h)
    X, Y = tf.meshgrid(xs, ys)
    X = tf.reshape(X, (1, h, w, 1)); Y = tf.reshape(Y, (1, h, w, 1))
    return tf.cast(X, tf.float32), tf.cast(Y, tf.float32)


def build_mixed_spatial_attention_from_logits(
    s_h: int, s_w: int, d: int, seg_logits_layer: layers.Layer,
    cloud_class_ids=CLOUD_IDS, sun_idx: int = SUN_IDX, k: int = 5,
    sigma_rel: float = 0.12
) -> tf.keras.Model:
    Xg, Yg = _build_coord_grids_2d(s_h, s_w)
    eps = 1e-6

    inp = Input((s_h, s_w, d), name="attn_now_input")
    logits = seg_logits_layer(inp)
    probs  = layers.Activation("softmax", name="attn_softmax_proxy")(logits)
    # Do not backpropagate gradients from attention to seg_logits
    probs  = layers.Lambda(tf.stop_gradient, name="attn_detach")(probs)

    # p_cloud = THICK + THIN; p_sun
    p_cloud = layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, cloud_class_ids, axis=-1), axis=-1, keepdims=True),
        name="attn_p_cloud")(probs)
    p_sun   = layers.Lambda(lambda t: t[..., sun_idx:sun_idx+1], name="attn_p_sun")(probs)

    # Spatial circle mask (broadcast-safe)
    M = _circle_const(s_h, s_w)  # (1, s_h, s_w, 1)

    # Zero-out probabilities outside the circle
    p_cloud = layers.Lambda(lambda t: t * M, name="attn_mask_cloud")(p_cloud)
    p_sun   = layers.Lambda(lambda t: t * M, name="attn_mask_sun")(p_sun)

    # Smoothing
    a_cloud = layers.Conv2D(1, k, padding="same", use_bias=False, name="attn_cloud_smooth")(p_cloud)
    a_sun   = layers.Conv2D(1, k, padding="same", use_bias=False, name="attn_sun_smooth")(p_sun)

    # Circumsolar Gaussian computed ONLY from masked psun
    def _circumsolar(ps):
        denom = tf.reduce_sum(ps, axis=[1, 2], keepdims=True) + eps
        x0 = tf.reduce_sum(ps * Xg, axis=[1, 2], keepdims=True) / denom
        y0 = tf.reduce_sum(ps * Yg, axis=[1, 2], keepdims=True) / denom
        sx = tf.constant(sigma_rel, tf.float32); sy = tf.constant(sigma_rel, tf.float32)
        G  = tf.exp(-0.5 * (((Xg - x0) / sx) ** 2 + ((Yg - y0) / sy) ** 2))
        return G

    G        = layers.Lambda(_circumsolar, name="attn_gaussian_cs")(p_sun)
    a_sun_cs = layers.Multiply(name="attn_sun_circum")([a_sun, G])

    mix = layers.Concatenate(name="attn_mix_concat")([a_cloud, a_sun_cs])
    mix = layers.Conv2D(1, 1, padding="same", use_bias=True, name="attn_mix_1x1")(mix)

    # --- Masked SPATIAL softmax with temperature + uniform mixing ---
    # Sum=1 inside the circle; zero outside. With mix>0 it never collapses into 1 pixel.
    def _masked_spatial_softmax(z, tau=1.5, mix=0.20):
        # z: (B, H, W, 1) and M: (1, H, W, 1) already defined above as the circle mask
        z = z / tau
        z = z - tf.reduce_max(z, axis=[1, 2, 3], keepdims=True)     # numerical stability
        e = tf.exp(z) * M                                           # zero outside the circle
        sum_e = tf.reduce_sum(e, axis=[1, 2, 3], keepdims=True) + 1e-9
        a = e / sum_e                                               # spatial softmax (sums to 1 inside)

        # Mix with a uniform distribution over the interior to avoid "turning off" regions
        n_valid = tf.reduce_sum(M, axis=[1, 2, 3], keepdims=True)   # (1,1,1,1)
        u = tf.math.divide_no_nan(M, n_valid)                       # uniform on valid pixels
        return (1.0 - mix) * a + mix * u                            # still sums to 1 inside
                                   

    A   = layers.Lambda(_masked_spatial_softmax, name="attn_masked_softmax")(mix)  # (B,H,W,1)
    out = layers.Multiply(name="attn_apply")([inp, A])
    return Model(inp, out, name="cloud_sun_spatial_attn")


def build_mixed_spatiotemporal_attention_td_from_logits(
    s_h: int, s_w: int, d: int, seg_logits_layer: layers.Layer,
    cloud_class_ids=CLOUD_IDS, sun_idx: int = SUN_IDX, k: int = 5,
    sigma_rel: float = 0.12
) -> tf.keras.Model:
    Xg = tf.reshape(_build_coord_grids_2d(s_h, s_w)[0], (1, 1, s_h, s_w, 1))
    Yg = tf.reshape(_build_coord_grids_2d(s_h, s_w)[1], (1, 1, s_h, s_w, 1))
    eps = 1e-6

    I_seq      = Input((None, s_h, s_w, d), name="attn_td_input")
    logits_seq = layers.TimeDistributed(seg_logits_layer, name="td_seg_logits")(I_seq)
    probs_seq  = layers.Activation("softmax", name="td_softmax_proxy")(logits_seq)
    # Do not backpropagate gradients from time-distributed attention to seg_logits
    probs_seq  = layers.Lambda(tf.stop_gradient, name="td_attn_detach")(probs_seq)

    p_cloud_t = layers.Lambda(
        lambda t: tf.reduce_sum(tf.gather(t, cloud_class_ids, axis=-1), axis=-1, keepdims=True),
        name="td_p_cloud")(probs_seq)
    p_sun_t   = layers.Lambda(lambda t: t[..., sun_idx:sun_idx+1], name="td_p_sun")(probs_seq)

    # Circle mask → expanded across time
    M   = _circle_const(s_h, s_w)  # (1, s_h, s_w, 1)
    M_t = layers.Lambda(lambda _: tf.reshape(M, (1, 1, s_h, s_w, 1)), name="td_mask_const")(I_seq)

    # Zero outside the circle before smoothing
    p_cloud_t = layers.Multiply(name="td_mask_cloud")([p_cloud_t, M_t])
    p_sun_t   = layers.Multiply(name="td_mask_sun")([p_sun_t,   M_t])

    a_cloud_t = layers.TimeDistributed(layers.Conv2D(1, k, padding="same", use_bias=False), name="td_cloud_smooth")(p_cloud_t)
    a_sun_t   = layers.TimeDistributed(layers.Conv2D(1, k, padding="same", use_bias=False), name="td_sun_smooth")(p_sun_t)

    def _td_circumsolar(ps):
        denom = tf.reduce_sum(ps, axis=[2, 3], keepdims=True) + eps
        x0 = tf.reduce_sum(ps * Xg, axis=[2, 3], keepdims=True) / denom
        y0 = tf.reduce_sum(ps * Yg, axis=[2, 3], keepdims=True) / denom
        sx = tf.constant(sigma_rel, tf.float32); sy = tf.constant(sigma_rel, tf.float32)
        G  = tf.exp(-0.5 * (((Xg - x0) / sx) ** 2 + ((Yg - y0) / sy) ** 2))
        return G

    G_t        = layers.Lambda(_td_circumsolar, name="td_gaussian_cs")(p_sun_t)
    a_sun_cs_t = layers.Multiply(name="td_sun_circum")([a_sun_t, G_t])

    mix_t = layers.Concatenate(name="td_mix_concat")([a_cloud_t, a_sun_cs_t])
    mix_t = layers.TimeDistributed(layers.Conv2D(1, 1, padding="same", use_bias=True), name="td_mix_1x1")(mix_t)

    # --- Masked TEMPORAL softmax (per-pixel) with temperature + uniform mixing ---
    def _masked_temporal_softmax(args, tau=1.5, mix=0.20):
        z, m_t = args  # z: (B,T,H,W,1), m_t: (1,1,H,W,1)
        z = z / tau
        z = z - tf.reduce_max(z, axis=1, keepdims=True)          # numerical stability across T
        e = tf.exp(z) * m_t                                      # zero outside the circle
        sum_e = tf.reduce_sum(e, axis=1, keepdims=True) + 1e-9
        a = tf.math.divide_no_nan(e, sum_e)                      # temporal softmax
        T = tf.cast(tf.shape(z)[1], tf.float32)
        u = tf.math.divide_no_nan(m_t, T)                        # uniform in time where m_t=1
        return (1.0 - mix) * a + mix * u

    attn_t     = layers.Lambda(_masked_temporal_softmax, name="td_masked_softmax_time")([mix_t, M_t])
    attn_t_rep = layers.Lambda(lambda t: tf.repeat(t, repeats=d, axis=-1), name="td_repeat_d")(attn_t)
    out        = layers.Multiply(name="td_apply")([I_seq, attn_t_rep])
    return Model(I_seq, out, name="cloud_sun_spatiotemporal_attn")




# =============================================================================
#                         S H A R E D   H E A D S
# =============================================================================
def build_shared_heads(img_h: int, img_w: int, s_h: int, s_w: int,
                       num_classes: int, d: int):
    seg_logits = layers.Conv2D(num_classes, 1, padding="same", name="seg_logits")

    # --- Segmentation ---
    inp_s = Input((s_h, s_w, d), name="seg_head_input")
    x = seg_logits(inp_s)
    x = layers.UpSampling2D((img_h // s_h, img_w // s_w),
                            interpolation="bilinear", name="seg_upsample")(x)
    seg_probs = layers.Activation("softmax", name="seg_softmax")(x)

    # Force VOID outside the circle (batch-safe)
    M_full = _circle_const(img_h, img_w)                         # (1,H,W,1)
    void_vec = tf.one_hot(VOID_IDX, num_classes, dtype=tf.float32)  # (C,)
    void_map = tf.reshape(void_vec, (1, 1, 1, num_classes))      # (1,1,1,C)

    def _force_void(y):
        m = M_full
        inside = y * m
        outside_void = (1.0 - m) * void_map
        return inside + outside_void

    seg_out = layers.Lambda(_force_void, name="seg_force_void")(seg_probs)
    seg_head = Model(inp_s, seg_out, name="seg_head")

    # --- GHI (W/m²) ---
    inp_g = Input((s_h, s_w, d), name="ghi_head_input")

    # Interior mask (broadcast over channels)
    M_small = _circle_const(s_h, s_w)  # (1,s_h,s_w,1)

    # Utility: masked GAP → sum(t)/sum(M)
    eps = 1e-6
    denom_inside = tf.reduce_sum(M_small) + eps  # scalar constant for (s_h,s_w)

    def _masked_gap(t):
        # t: (B,H,W,C) masked or not; we mask again here for safety
        return tf.reduce_sum(t * M_small, axis=[1, 2]) / denom_inside  # (B,C)

    # Mask features inside the circle
    feats_in = layers.Lambda(lambda z: z * M_small, name="ghi_mask_feats_inside")(inp_g)

    # Lightweight conv block over masked features
    y = layers.Conv2D(96, 3, padding="valid", activation="swish", name="ghi_c1")(feats_in)
    y = layers.BatchNormalization(name="ghi_bn1")(y)
    y = layers.Conv2D(96, 3, padding="valid", activation="swish", name="ghi_c2")(y)
    y = layers.BatchNormalization(name="ghi_bn2")(y)
    y = layers.GlobalAveragePooling2D(name="ghi_gap")(y)

    # Class proxy from seg_logits (shared)
    cls_logits = seg_logits(inp_g)
    cls_probs  = layers.Activation("softmax", name="ghi_softmax_proxy")(cls_logits)
    cls_probs  = layers.Lambda(tf.stop_gradient, name="ghi_proxy_detach")(cls_probs)
    cls_probs  = layers.Lambda(lambda z: z * M_small, name="ghi_mask_probs_inside")(cls_probs)

    # Global histogram (5) with mean ONLY inside the circle
    p_hist = layers.Lambda(_masked_gap, name="ghi_class_hist")(cls_probs)  # (B,5)

    # p_sun and p_cloud
    psun   = layers.Lambda(lambda z: z[..., SUN_IDX:SUN_IDX+1], name="ghi_p_sun")(cls_probs)
    pcloud = layers.Lambda(
        lambda z: tf.reduce_sum(tf.gather(z, CLOUD_IDS, axis=-1), axis=-1, keepdims=True),
        name="ghi_p_cloud")(cls_probs)

    # Circumsolar Gaussian over psun (already masked)
    Xg, Yg = _build_coord_grids_2d(s_h, s_w)

    def _gauss_cs(ps):
        denom = tf.reduce_sum(ps, axis=[1, 2], keepdims=True) + eps
        x0 = tf.reduce_sum(ps * Xg, axis=[1, 2], keepdims=True) / denom
        y0 = tf.reduce_sum(ps * Yg, axis=[1, 2], keepdims=True) / denom
        sx = tf.constant(0.12, tf.float32); sy = tf.constant(0.12, tf.float32)
        return tf.exp(-0.5 * (((Xg - x0) / sx) ** 2 + ((Yg - y0) / sy) ** 2))

    G = layers.Lambda(_gauss_cs, name="ghi_gaussian_cs")(psun)

    # Cloud near the sun (masked) and masked GAP areas
    cloud_near_sun_map = layers.Multiply(name="ghi_cloud_near_sun")([pcloud, G])         # (B,H,W,1)
    cloud_near_sun = layers.Lambda(_masked_gap, name="ghi_cloud_near_sun_gap")(cloud_near_sun_map)  # (B,1)
    sun_area       = layers.Lambda(_masked_gap, name="ghi_sun_area")(psun)                               # (B,1)

    # Dense head + safe output (0..1500) with gradient across full range
    z = layers.Concatenate(name="ghi_concat")([y, p_hist, sun_area, cloud_near_sun])
    z = layers.Dropout(0.25)(z)
    z = layers.Dense(64, activation="swish", name="ghi_fc1")(z)
    # Gentle initial bias (~400–500 W/m²). Adjust if your median differs.
    z = layers.Dense(1, name="ghi_dense",
                     bias_initializer=tf.keras.initializers.Constant(-1.0))(z)
    ghi_out = layers.Lambda(lambda t: 1500.0 * tf.nn.sigmoid(t), name="ghi_0_1500")(z)

    ghi_head = Model(inp_g, ghi_out, name="ghi_head")
    return seg_head, ghi_head, seg_logits



# =============================================================================
#                              N O W   M O D E L
# =============================================================================
def build_now_model(img_h: int, img_w: int, img_c: int,
                    backbone: tf.keras.Model,
                    seg_head: Model, ghi_head: Model,
                    attn_model: Model,
                    d: int = D_REPR) -> tf.keras.Model:
    inp = layers.Input((img_h, img_w, img_c), name="now_input")
    x_in = layers.Lambda(_mbv2_preproc, name="mbv2_preproc")(inp)  # PREPROCESS [-1,1]
    feats = backbone(x_in)
    I = layers.Conv2D(d, 1, name="proj_I_now")(feats)
    I = attn_model(I)
    seg = seg_head(I)
    ghi = ghi_head(I)
    seg = layers.Activation('linear', name='upseg')(seg)
    ghi = layers.Activation('linear', name='ghi_pred')(ghi)
    return models.Model(inp, [seg, ghi], name='NowNet_M1')


# =============================================================================
#                           F U T U R E  M O D E L
# =============================================================================
def build_future(backbone: tf.keras.Model, timesteps: int, s_h: int, s_w: int,
                 seg_head: Model, ghi_head: Model,
                 rnn_type: str = "gru", rnn_filters=(64, 64),
                 d: int = D_REPR,
                 seg_logits_layer: layers.Layer = None) -> tf.keras.Model:
    assert seg_logits_layer is not None, "Pasa seg_logits_layer para atención interpretable"  # keep original

    inp = layers.Input(
        shape=(timesteps, backbone.input_shape[1], backbone.input_shape[2], backbone.input_shape[3]),
        name="input_seq"
    )
    x_td = layers.TimeDistributed(layers.Lambda(_mbv2_preproc), name="preproc_td")(inp)  # PREPROCESS [-1,1]
    td_feats = layers.TimeDistributed(backbone, name="backbone_td")(x_td)
    td_I = layers.TimeDistributed(Conv2D(d, 1), name="proj_I_td")(td_feats)
    
    cloud_sun_attn_td = build_mixed_spatiotemporal_attention_td_from_logits(
        s_h, s_w, d, seg_logits_layer=seg_logits_layer,
        cloud_class_ids=(THICK_IDX,), sun_idx=SUN_IDX, k=5
    )
    td_I = cloud_sun_attn_td(td_I)

    """
    td_I = td_I
    """
    RNN = _get_rnn_layer(rnn_type)
    x = td_I
    for i, f in enumerate(rnn_filters, 1):
        x = RNN(f, (5, 5), padding="same", return_sequences=True, name=f"{rnn_type.lower()}_{i}")(x)
        x = layers.TimeDistributed(layers.BatchNormalization(), name=f"bn_{i}")(x)

    x_d = layers.TimeDistributed(Conv2D(d, 1), name="proj_post_rnn")(x)
    seg_seq = layers.TimeDistributed(seg_head, name="upseg")(x_d)
    ghi_seq = layers.TimeDistributed(ghi_head)(x_d)   # (B,T,1)
    ghi_seq = layers.Activation('linear', name='ghi_pred')(ghi_seq)
    
    return models.Model(inp, [seg_seq, ghi_seq], name=f"CloudAttn_{rnn_type.UPPER()}_Forecaster_M1")


# =============================================================================
#                   S H A R E D   C O N S T R U C T O R
# =============================================================================
def build_models_shared(timesteps: int = 6,
                        img_height: int = IMG_H,
                        img_width: int = IMG_W,
                        img_channels: int = 3,
                        num_classes: int = NUM_CLASSES,
                        gamma: float = 2.0,
                        alpha: List[float] = ALPHA_DEFAULT,
                        m: float = 0.01,
                        rnn_type: str = "gru",
                        d: int = D_REPR,
                        output_stride: int = 16,
                        mbv2_alpha: float = 0.5,
                        weights: Optional[str] = "imagenet",
                        learning_rate: float = 1e-3,
                        clipnorm: float = 1.0,
                        build_future_model: bool = True,
                        steps_per_execution: Optional[int] = None
                        ) -> Tuple[tf.keras.Model, Optional[tf.keras.Model]]:

    backbone, s_h, s_w = build_backbone_mbv2(
        img_height, img_width, img_channels,
        alpha=mbv2_alpha, output_stride=output_stride, weights=weights
    )

    seg_head, ghi_head, seg_logits = build_shared_heads(img_height, img_width, s_h, s_w, num_classes, d)

    """
    attn_now = build_mixed_spatial_attention_from_logits(
        s_h, s_w, d, seg_logits_layer=seg_logits,
        cloud_class_ids=CLOUD_IDS, sun_idx=SUN_IDX, k=5
    )
    """
    attn_now = tf.keras.Sequential([layers.Lambda(lambda x: x)], name="attn_identity")
   
    now = build_now_model(img_height, img_width, img_channels,
                          backbone=backbone,
                          seg_head=seg_head, ghi_head=ghi_head,
                          attn_model=attn_now, d=d)

    future = None
    if build_future_model:
        future = build_future(backbone, timesteps, s_h, s_w,
                              seg_head=seg_head, ghi_head=ghi_head,
                              rnn_type=rnn_type, d=d,
                              seg_logits_layer=seg_logits)

    # Losses and compile
    seg_loss  = CombinedSegLoss(gamma, alpha)
    dice_loss = TverskyLossMasked(alpha=0.3, beta=0.7, ignore=(VOID_IDX,))
    ghi_loss  = LogCoshLossSI(m=m, low_irr_thr=150.0, low_irr_weight=1.5)

    lambda_seg  = 1.0
    lambda_dice = 0.5

    opt = Adam(learning_rate=learning_rate, clipnorm=clipnorm)

    def seg_total_mean(y_true, y_pred):
        return tf.reduce_mean(seg_loss(y_true, y_pred) + lambda_dice * dice_loss(y_true, y_pred))

    def ghi_mean(y_true, y_pred):
        return tf.reduce_mean(ghi_loss(y_true, y_pred))

    compile_kwargs = dict(
        optimizer=opt,
        loss={"upseg": seg_total_mean, "ghi_pred": ghi_mean},
        loss_weights={"upseg": lambda_seg, "ghi_pred": 1.0},
        metrics={
            "upseg": [
            PixelAccWithMask(name="pixel_acc"),
            MeanIoUWithMask(num_classes=num_classes, name="mean_iou"),
            RecallForClassMasked(SUN_IDX, name="recall_sun"),
            RecallForClassMasked(THIN_IDX, name="recall_thin"),
            RecallForClassMasked(THICK_IDX, name="recall_thick"),
            ],
            "ghi_pred": [tf.keras.metrics.MeanAbsoluteError(name="mae")]
        }
    )
    if steps_per_execution is not None:
        compile_kwargs["steps_per_execution"] = int(steps_per_execution)

    now.compile(**compile_kwargs)
    if future is not None:
        future.compile(**compile_kwargs)

    return now, future


# =============================================================================
#                              Q U I C K   T E S T
# =============================================================================
if __name__ == "__main__":
    now, _ = build_models_shared(
        timesteps=6,
        mbv2_alpha=0.5,
        rnn_type="gru",
        weights="imagenet",
        learning_rate=1e-3,
        clipnorm=1.0,
        build_future_model=False,
        output_stride=16,
        steps_per_execution=16
    )
    print("\n>>> NOWNET (M1) - 5 clases (VOID,SKY,THICK,THIN,SUN) con máscara de círculo, VOID ignorado en pérdidas y forzado fuera en salida")

