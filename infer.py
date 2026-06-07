#!/usr/bin/env python
"""
PancrAI infer.py v9 — with Test-Time Augmentation (TTA)
TTA averages predictions over 8 flip combinations (all X/Y/Z) for best accuracy.
"""
import argparse, json, os, sys, warnings, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_PLATFORMS"] = ""
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
import torch
import nibabel as nib
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

TRAIN_XY_SPACING = 1.5
TRAIN_Z_SPACING  = 2.0
MAX_Z_FACTOR     = 1.5
CPU_OVERLAP      = 0.25
CPU_SW_MODE      = "constant"

def log(msg):
    print(f"[infer] {msg}", flush=True)

def _resample(data_f32, src_zooms, tgt_zooms):
    from scipy.ndimage import zoom as nd_zoom
    factors   = tuple(float(s) / float(t) for s, t in zip(src_zooms, tgt_zooms))
    out_shape = tuple(max(1, round(d * f)) for d, f in zip(data_f32.shape, factors))
    coord_mb  = 3 * out_shape[0] * out_shape[1] * out_shape[2] * 8 / 1024**2
    log(f"  Resample {data_f32.shape} -> {out_shape} (~{coord_mb:.0f} MB)")
    out = nd_zoom(data_f32, factors, order=1, prefilter=False).astype(np.float32)
    gc.collect()
    return out, tgt_zooms

def _pad_to(data, target_size):
    pad = []
    for d, t in zip(data.shape, target_size):
        p = max(0, t - d)
        pad.append((p // 2, p - p // 2))
    return np.pad(data, pad, mode="constant", constant_values=0) if any(
        p[0] + p[1] > 0 for p in pad) else data

def load_model(dev):
    cfg_path = os.path.join(HERE, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    log(f"Config: spacing={cfg.get('spacing')} img_size={cfg.get('img_size')} "
        f"feature_size={cfg.get('feature_size')}")
    img_size = tuple(cfg.get("img_size", [96, 96, 96]))
    model = SwinUNETR(
        img_size     = img_size,
        in_channels  = 1,
        out_channels = cfg.get("out_channels", 3),
        feature_size = cfg.get("feature_size", 48),
        use_checkpoint = False,
        spatial_dims = 3,
    ).to(dev)
    ck_path = os.path.join(HERE, "best_model.pth")
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f"best_model.pth not found: {ck_path}")
    log(f"Loading weights: {ck_path}")
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    if isinstance(ck, dict):
        for key in ("model", "state_dict", "net"):
            if key in ck and isinstance(ck[key], dict):
                ck = ck[key]; break
    state = {k.replace("module.", ""): v for k, v in ck.items()
             if isinstance(v, torch.Tensor)}
    model.load_state_dict(state, strict=False)
    model.eval()
    log(f"Model ready — {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, cfg

def _run_single(model, data, isz, sw_batch, overlap, sw_mode, dev):
    """Run sliding_window_inference, return SOFTMAX logits (out_ch, X, Y, Z)."""
    inp = torch.from_numpy(data[np.newaxis, np.newaxis]).to(dev)
    gc.collect()
    def _pred(x):
        o = model(x)
        return o[0] if isinstance(o, (list, tuple)) else o
    with torch.no_grad():
        logits = sliding_window_inference(
            inp, isz, sw_batch, _pred,
            overlap=overlap, mode=sw_mode, progress=False,
        )  # (1, out_ch, X, Y, Z)
    import torch.nn.functional as F
    probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()  # (out_ch, X, Y, Z)
    del inp, logits
    gc.collect()
    return probs

def _infer_with_tta(model, data, isz, sw_batch, overlap, sw_mode, dev):
    """
    Test-Time Augmentation: average over 4 axial flips.
    Flipping along spatial axes and averaging the softmax probabilities
    before argmax typically adds +0.02 to +0.05 Dice at zero training cost.
    """
    # ── 8-flip TTA: all combinations of X/Y/Z flips ─────────────
    # More flips = better ensemble average. On GPU the 8 extra
    # forward passes add ~4 min per scan but gain +0.02–0.04 Dice.
    flip_combos = [
        [],          # original
        [0],         # flip X
        [1],         # flip Y
        [2],         # flip Z
        [0, 1],      # flip X+Y
        [0, 2],      # flip X+Z
        [1, 2],      # flip Y+Z
        [0, 1, 2],   # flip X+Y+Z
    ]
    acc = None
    for axes in flip_combos:
        d = np.flip(data, axis=axes).copy() if axes else data
        p = _run_single(model, d, isz, sw_batch, overlap, sw_mode, dev)  # (C,X,Y,Z)
        if axes:
            # un-flip the prediction axes to align with original orientation
            p = np.flip(p, axis=[a + 1 for a in axes]).copy()  # axis+1 because dim 0 is channel
        acc = p if acc is None else acc + p
    avg_probs = acc / len(flip_combos)
    return avg_probs.argmax(axis=0).astype(np.uint8)  # (X, Y, Z)

def predict(img_path, out_path, requested_device="cuda", use_tta=True):
    try:
        log(f"infer.py v9  TTA={'ON' if use_tta else 'OFF'}")
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            log("GPU detected — using CUDA.")
        else:
            dev = torch.device("cpu")
            log("No GPU — using CPU (slow).")
        log(f"Device: {dev.type}")

        model, cfg = load_model(dev)
        isz    = tuple(cfg.get("img_size",     [96, 96, 96]))
        hu_min = cfg.get("hu_min",  -175)
        hu_max = cfg.get("hu_max",   250)

        overlap = CPU_OVERLAP if dev.type == "cpu" else 0.75  # always 0.75 at inference for best boundary quality
        sw_mode = CPU_SW_MODE if dev.type == "cpu" else "gaussian"
        sw_batch = 1

        # 1. Load + RAS orient
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"CT not found: {img_path}")
        log(f"Loading: {img_path}")
        img_nib   = nib.load(img_path)
        img_ras   = nib.as_closest_canonical(img_nib)
        data      = img_ras.get_fdata(dtype=np.float32)
        src_zooms = tuple(float(z) for z in np.abs(img_ras.header.get_zooms()[:3]))

        # 2. Smart resample
        z_factor = src_zooms[2] / TRAIN_Z_SPACING
        tgt_zooms = (
            TRAIN_XY_SPACING, TRAIN_XY_SPACING, TRAIN_Z_SPACING
        ) if z_factor <= MAX_Z_FACTOR else (
            TRAIN_XY_SPACING, TRAIN_XY_SPACING, src_zooms[2]
        )
        data, cur_zooms = _resample(data, src_zooms, tgt_zooms)
        del img_nib; gc.collect()

        # 3. Affine reconstruction (FIX 1 from v6.2 — preserved)
        aff = img_ras.affine.copy()
        for i in range(3):
            scale = float(cur_zooms[i]) / float(src_zooms[i])
            aff[:3, i] = img_ras.affine[:3, i] * scale

        # 4. HU clip + normalise
        data = np.clip(data, hu_min, hu_max)
        data = (data - hu_min) / float(hu_max - hu_min)

        # 5. Crop foreground + track offset (FIX 2 from v6.2 — preserved)
        coords = np.argwhere(data > 0)
        if len(coords) > 0:
            crop_lo = coords.min(axis=0)
            crop_hi = coords.max(axis=0) + 1
            data = data[crop_lo[0]:crop_hi[0],
                        crop_lo[1]:crop_hi[1],
                        crop_lo[2]:crop_hi[2]]
            aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ crop_lo.astype(np.float64)
        else:
            crop_lo = np.zeros(3, dtype=np.int64)

        cropped_shape = data.shape
        data = _pad_to(data, isz)

        # 6. Inference (with or without TTA)
        if use_tta:
            log("Running TTA inference (8 flips)...")
            pred = _infer_with_tta(model, data, isz, sw_batch, overlap, sw_mode, dev)
        else:
            log("Running single-pass inference...")
            probs = _run_single(model, data, isz, sw_batch, overlap, sw_mode, dev)
            pred  = probs.argmax(axis=0).astype(np.uint8)
        del data; gc.collect()

        # 7. Strip padding (FIX 3 from v6.2 — preserved)
        cx, cy, cz = cropped_shape
        px = max(0, isz[0] - cx); py = max(0, isz[1] - cy); pz = max(0, isz[2] - cz)
        x0 = px // 2; y0 = py // 2; z0 = pz // 2
        pred = pred[x0:x0+cx, y0:y0+cy, z0:z0+cz]
        pad_offset = np.array([x0, y0, z0], dtype=np.float64)
        aff[:3, 3] = aff[:3, 3] - aff[:3, :3] @ pad_offset

        # 8. Save
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        nib.save(nib.Nifti1Image(pred, aff), out_path)

        # 9. Volumetrics (FIX 4 from v6.2 — cur_zooms not det(affine))
        voxel_mm3 = float(cur_zooms[0] * cur_zooms[1] * cur_zooms[2])
        pan_ml = int(np.sum(pred == 1)) * voxel_mm3 / 1000.0
        tum_ml = int(np.sum(pred == 2)) * voxel_mm3 / 1000.0

        print(f"Pancreas:{pan_ml:.2f}mL  Tumor:{tum_ml:.2f}mL  Saved:{out_path}", flush=True)
        return {"pancreas_ml": pan_ml, "tumor_ml": tum_ml}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log("CRITICAL CRASH:\n" + tb)
        raise RuntimeError("Inference pipeline crashed: " + str(e)) from e

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-tta", action="store_true", help="Disable TTA (faster, lower Dice)")
    a = ap.parse_args()
    predict(a.image, a.out, a.device, use_tta=not a.no_tta)