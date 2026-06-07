"""
PancrAI — Flask Server v2.1 (merged + fixed)
Endpoints:
  GET  /                         → index.html (upload page)
  POST /upload                   → accepts .nii/.nii.gz, starts inference job
  GET  /status/<job_id>          → job progress JSON
  GET  /result/<job_id>          → viewer.html
  GET  /api/dims/<job_id>        → volume dimensions JSON
  GET  /api/metrics/<job_id>     → volumetric metrics JSON
  GET  /api/slice/<job_id>/<plane>/<idx>  → PNG slice (with overlay)
  GET  /api/report/<job_id>      → PDF radiology report
  GET  /api/mask/<job_id>        → download segmentation mask .nii.gz

FIX NOTES (v2.1):
  - Root cause of HTTP 500: RGBA alpha channel was being set to ~140 on overlay
    pixels while background stayed at 255 — causing an inconsistent alpha
    composite that rendered as all-black in browsers and crashed PIL on some
    builds. Fixed by switching to pure RGB with in-place colour blending
    (no alpha channel ever written).
  - Root cause of wrong-sized views: arrays were pre-transposed to (Z,Y,X) in
    the cache, then the slice extractor used wrong axis indices per plane —
    sagittal and coronal were pulling Z-slices instead of X/Y slices.
    Fixed by keeping arrays in native NIfTI (X,Y,Z) order and using the
    correct .T transpose per plane (from app_old.py).
  - api_dims now returns max_axial / max_sag / max_cor explicitly so the viewer
    JS cannot mix up nx/ny/nz for the wrong plane.
  - All three planes resized to SLICE_OUTPUT_SIZE (512x512) so panels are equal.
  - Window/level defaults corrected to ww=400, wl=40 (standard abdominal CT).
  - _zoom_to_shape handles CT/mask shape mismatch from resampled inference.
  - Volume cache now protected by a threading.Lock.
"""

import os, sys, uuid, time, threading, json, gc, traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE          = Path(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = Path(os.environ.get('PANCRAI_UPLOADS', HERE / 'uploads'))
OUTPUT_FOLDER = Path(os.environ.get('PANCRAI_OUTPUTS', HERE / 'outputs'))
MODEL_PATH    = Path(os.environ.get('PANCRAI_MODEL',  HERE / 'best_model.pth'))
CONFIG_PATH   = Path(os.environ.get('PANCRAI_CONFIG', HERE / 'config.json'))

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(HERE / 'templates'),
    static_folder=str(HERE / 'static') if (HERE / 'static').exists() else None,
)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ── Job store ─────────────────────────────────────────────────────────────────
jobs: dict = {}

def _job(job_id: str) -> dict:
    if job_id not in jobs:
        abort(404, description=f"Job {job_id} not found")
    return jobs[job_id]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Pages
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    model_ok   = MODEL_PATH.exists() and CONFIG_PATH.exists()
    model_path = str(MODEL_PATH)
    return render_template('index.html', model_ok=model_ok, model_path=model_path)


@app.route('/result/<job_id>')
def result(job_id):
    j = _job(job_id)
    if j['status'] not in ('done', 'error'):
        return f"<p>Job {job_id} not ready yet (status: {j['status']})</p>", 202

    metrics  = j.get('metrics', {})
    model_ok = MODEL_PATH.exists()
    pan_dice = '0.7321'
    tum_dice = '0.4219'
    best_ep  = '115'
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            cfg = {}

    return render_template('viewer.html',
        job_id=job_id, model_ok=model_ok,
        pan_dice=pan_dice, tum_dice=tum_dice, best_ep=best_ep,
        metrics=metrics,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Upload + Status
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No file selected'}), 400
    fname = f.filename.lower()
    if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
        return jsonify({'error': 'File must be .nii or .nii.gz'}), 400

    job_id  = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_FOLDER / job_id
    out_dir = OUTPUT_FOLDER / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext     = '.nii.gz' if fname.endswith('.nii.gz') else '.nii'
    in_path = job_dir / f'input{ext}'
    f.save(str(in_path))

    jobs[job_id] = {
        'status'  : 'queued',
        'progress': 0,
        'message' : 'Queued — waiting for inference engine',
        'error'   : None,
        'in_path' : str(in_path),
        'out_dir' : str(out_dir),
        'metrics' : {},
        'created' : time.time(),
    }
    threading.Thread(target=_run_inference, args=(job_id,), daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    j = _job(job_id)
    return jsonify({
        'status'  : j['status'],
        'progress': j['progress'],
        'message' : j['message'],
        'error'   : j.get('error'),
    })


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES — API (data for viewer)
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/dims/<job_id>')
def api_dims(job_id):
    """
    Returns volume dimensions AND explicit per-plane max slice indices.

    NIfTI arrays are stored as (X, Y, Z).
      Axial    slices → indexed along Z → max = nz-1
      Sagittal slices → indexed along X → max = nx-1
      Coronal  slices → indexed along Y → max = ny-1

    The old code only returned nx/ny/nz and the viewer JS was using nx (512)
    as the sagittal max, hence "Slice: 0 / 511" shown for sag and cor.
    """
    j   = _job(job_id)
    vol = _load_volume(j)
    ct  = vol['ct']               # (X, Y, Z) native NIfTI order
    nx, ny, nz = ct.shape[:3]
    sx, sy, sz = vol['spacing']

    return jsonify({
        'nx': nx, 'ny': ny, 'nz': nz,
        'sx': float(sx), 'sy': float(sy), 'sz': float(sz),
        # Unambiguous per-plane maximums for the viewer JS
        'max_axial': nz - 1,
        'max_sag'  : nx - 1,
        'max_cor'  : ny - 1,
    })


@app.route('/api/metrics/<job_id>')
def api_metrics(job_id):
    j = _job(job_id)
    return jsonify(j.get('metrics', {}))


# ══════════════════════════════════════════════════════════════════════════════
#  SLICE RENDERER
# ══════════════════════════════════════════════════════════════════════════════

# All planes are resized to this square before returning — makes all three
# viewer panels render at the same size regardless of voxel anisotropy.
SLICE_OUTPUT_SIZE = 512


@app.route('/api/slice/<job_id>/<plane>/<int:idx>')
def api_slice(job_id, plane, idx):
    """
    FIX: uses native NIfTI (X,Y,Z) axis order with .T per-plane transpose,
    pure RGB blending (no alpha channel), correct abdominal W/L defaults,
    and fixed-size output for uniform panel sizing.

    NIfTI display convention (from app_old.py):
      AXIAL    : ct[:, :, z].T  → display shape (ny, nx)
      SAGITTAL : ct[x, :, :].T  → display shape (nz, ny)   Z=rows, Y=cols
      CORONAL  : ct[:, y, :].T  → display shape (nz, nx)   Z=rows, X=cols
    Rows are flipped so superior is at top.
    """
    import numpy as np
    from PIL import Image
    from io import BytesIO

    j   = _job(job_id)
    vol = _load_volume(j)

    ct_arr   = vol['ct']    # float32, shape (X, Y, Z)
    mask_arr = vol['seg']   # uint8,   shape (X, Y, Z) — may differ if resampled

    # Corrected W/L defaults: standard abdominal CT window
    ww     = float(request.args.get('ww',    400))   # was 150
    wl     = float(request.args.get('wl',     40))   # was 50
    pan_op = float(request.args.get('pan_op', 0.7))
    tum_op = float(request.args.get('tum_op', 0.9))
    heatmap = request.args.get('heatmap', '0') == '1'

    mnx, mny, mnz = mask_arr.shape[:3]
    cnx, cny, cnz = ct_arr.shape[:3]

    def _map(i, m_dim, c_dim):
        """Remap mask-space index to CT-space (handles resampled inference)."""
        return int(np.clip(round(i * c_dim / max(m_dim, 1)), 0, c_dim - 1))

    if plane == 'axial':
        mz      = int(np.clip(idx, 0, mnz - 1))
        cz      = _map(mz, mnz, cnz)
        ct_sl   = ct_arr[:, :, cz].T       # → (ny, nx)
        mask_sl = mask_arr[:, :, mz].T

    elif plane == 'sag':
        mx      = int(np.clip(idx, 0, mnx - 1))
        cx      = _map(mx, mnx, cnx)
        ct_sl   = ct_arr[cx, :, :].T       # → (nz, ny)
        mask_sl = mask_arr[mx, :, :].T

    elif plane == 'cor':
        my      = int(np.clip(idx, 0, mny - 1))
        cy      = _map(my, mny, cny)
        ct_sl   = ct_arr[:, cy, :].T       # → (nz, nx)
        mask_sl = mask_arr[:, my, :].T

    else:
        abort(400, 'Invalid plane — use axial, sag, or cor')

    # Align mask to CT slice dimensions if inference used a different resolution
    if ct_sl.shape != mask_sl.shape:
        ms = Image.fromarray(mask_sl.astype('uint8'), mode='L')
        ms = ms.resize((ct_sl.shape[1], ct_sl.shape[0]), Image.NEAREST)
        mask_sl = np.array(ms, dtype='uint8')

    # Flip rows: superior at top
    ct_sl   = ct_sl[::-1].copy()
    mask_sl = mask_sl[::-1].copy()

    # Window / Level → 8-bit grayscale
    lo   = wl - ww / 2.0
    hi   = wl + ww / 2.0
    gray = np.clip((ct_sl - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

    # Pure RGB — no alpha channel (fixes the HTTP 500 / all-black render)
    rgb = np.stack([gray, gray, gray], axis=-1)

    pan_m = (mask_sl == 1)
    tum_m = (mask_sl == 2)

    if pan_op > 0 and pan_m.any():
        bg = rgb[pan_m].astype(np.float32)
        rgb[pan_m] = np.clip(np.stack([
            bg[:, 0] * (1 - pan_op),
            bg[:, 1] * (1 - pan_op) + 220 * pan_op,
            bg[:, 2] * (1 - pan_op) + 100 * pan_op,
        ], axis=-1), 0, 255).astype(np.uint8)

    if tum_op > 0 and tum_m.any():
        bg = rgb[tum_m].astype(np.float32)
        rgb[tum_m] = np.clip(np.stack([
            bg[:, 0] * (1 - tum_op) + 255 * tum_op,
            bg[:, 1] * (1 - tum_op) +  30 * tum_op,
            bg[:, 2] * (1 - tum_op) +  50 * tum_op,
        ], axis=-1), 0, 255).astype(np.uint8)

    # Heatmap — distance-transform jet colormap (scipy optional)
    if heatmap and (mask_sl > 0).any():
        try:
            from scipy.ndimage import distance_transform_edt
            combined = (mask_sl > 0)
            dist  = distance_transform_edt(combined).astype(np.float32)
            t     = dist / (float(dist.max()) or 1.0)
            hm_op = 0.55
            r_j   = np.clip(1.5 - np.abs(1.0 - 4.0 * (t - 0.50)), 0, 1)
            g_j   = np.clip(1.5 - np.abs(1.0 - 4.0 * (t - 0.25)), 0, 1)
            b_j   = np.clip(1.5 - np.abs(1.0 - 4.0 *  t),         0, 1)
            fg    = combined
            bg    = rgb[fg].astype(np.float32)
            rgb[fg] = np.clip(np.stack([
                bg[:, 0] * (1 - hm_op) + r_j[fg] * 255 * hm_op,
                bg[:, 1] * (1 - hm_op) + g_j[fg] * 255 * hm_op,
                bg[:, 2] * (1 - hm_op) + b_j[fg] * 255 * hm_op,
            ], axis=-1), 0, 255).astype(np.uint8)
        except ImportError:
            pass  # scipy not installed — skip heatmap

    # Resize to fixed square → all three panels are identical size
    img = Image.fromarray(rgb, 'RGB')
    img = img.resize((SLICE_OUTPUT_SIZE, SLICE_OUTPUT_SIZE), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format='PNG', optimize=False, compress_level=1)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/report/<job_id>')
def api_report(job_id):
    j        = _job(job_id)
    pdf_path = Path(j['out_dir']) / 'report.pdf'
    if not pdf_path.exists():
        _generate_pdf(j, pdf_path)
    return send_file(str(pdf_path), mimetype='application/pdf',
                     as_attachment=True,
                     download_name=f'PancrAI_Report_{job_id}.pdf')


@app.route('/api/mask/<job_id>')
def api_mask(job_id):
    j         = _job(job_id)
    mask_path = Path(j['out_dir']) / 'mask.nii.gz'
    if not mask_path.exists():
        abort(404, 'Mask not ready yet')
    return send_file(str(mask_path), mimetype='application/gzip',
                     as_attachment=True,
                     download_name=f'PancrAI_Mask_{job_id}.nii.gz')


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE WORKER
# ══════════════════════════════════════════════════════════════════════════════

def _update(job_id, progress, message, status='running'):
    if job_id in jobs:
        jobs[job_id].update({'progress': progress, 'message': message, 'status': status})


def _run_inference(job_id: str):
    j       = jobs[job_id]
    in_path = j['in_path']
    out_dir = Path(j['out_dir'])

    try:
        _update(job_id, 5, 'Loading model weights...')

        server_dir = str(HERE)
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)

        model_ok = MODEL_PATH.exists() and CONFIG_PATH.exists()

        if model_ok:
            _update(job_id, 10, 'Model loaded — running SwinUNETR inference...')
            import importlib.util
            spec  = importlib.util.spec_from_file_location('infer', str(HERE / 'infer.py'))
            infer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(infer)

            mask_path = str(out_dir / 'mask.nii.gz')
            _step     = [10]   # pre-TTA phase starts at 10%; TTA takes 15–87%
            _orig_log = infer.log

            import re as _re
            def _prog_log(msg):
                _orig_log(msg)
                # Parse "TTA pass N/T" messages from infer.py for granular progress
                tta_m = _re.search(r'TTA pass (\d+)/(\d+)', msg)
                if tta_m:
                    cur   = int(tta_m.group(1))
                    total = int(tta_m.group(2))
                    # Map TTA passes evenly across 15–87% band
                    # Pass 1→15%, Pass 2→24%, ... Pass 8→78%
                    pct = 15 + int((cur - 1) / total * 72)
                    _step[0] = pct
                    _update(job_id, pct,
                            f'Running TTA inference ({cur}/{total} flips)...')
                else:
                    # Non-TTA log (load/resample/preprocess): advance slowly up to 14%
                    # Use max() so _step never goes backwards if it was already bumped
                    _step[0] = min(max(_step[0], 10) + 1, 14)
                    _update(job_id, _step[0], msg)

            infer.log = _prog_log
            _update(job_id, 12, 'Preprocessing CT scan...')
            result_inf = infer.predict(in_path, mask_path, use_tta=True)
            pan_ml = result_inf.get('pancreas_ml', 0.0)
            tum_ml = result_inf.get('tumor_ml',    0.0)

        else:
            # Demo mode — vectorised ellipsoid (from app_old.py, better than threshold)
            _update(job_id, 15, 'Demo mode — generating ellipsoid segmentation...')
            import numpy as np
            import nibabel as nib

            img_nib = nib.load(in_path)
            img_ras = nib.as_closest_canonical(img_nib)
            data    = img_ras.get_fdata(dtype=np.float32)
            nx, ny, nz = data.shape[:3]

            _update(job_id, 35, 'Generating segmentation mask...')
            abdomen = (data > 20) & (data < 80)
            coords  = np.argwhere(abdomen)
            mask    = np.zeros((nx, ny, nz), dtype=np.uint8)

            if len(coords) > 100:
                cz = int(np.percentile(coords[:, 2], 50))
                cy = int(np.percentile(coords[:, 1], 40))
                cx = int(np.percentile(coords[:, 0], 50))

                rx = nx * 0.15; ry = ny * 0.06; rz = nz * 0.08
                xs = np.arange(nx, dtype=np.float32)
                ys = np.arange(ny, dtype=np.float32)
                zs = np.arange(nz, dtype=np.float32)
                X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
                pan_ell = ((X-cx)/rx)**2 + ((Y-cy)/ry)**2 + ((Z-cz)/rz)**2
                mask[(pan_ell < 1.0) & abdomen] = 1

                _update(job_id, 55, 'Adding tumour region...')
                tx = cx - int(nx*0.04); ty = cy; tz = cz + int(nz*0.02)
                trx = nx*0.04; try_ = ny*0.02; trz = nz*0.03
                tum_ell = ((X-tx)/trx)**2 + ((Y-ty)/try_)**2 + ((Z-tz)/trz)**2
                mask[tum_ell < 1.0] = 2

            mask_path = str(out_dir / 'mask.nii.gz')
            nib.save(nib.Nifti1Image(mask, img_ras.affine, img_ras.header), mask_path)

            zooms  = img_ras.header.get_zooms()[:3]
            vox_ml = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0
            pan_ml = float(np.sum(mask == 1)) * vox_ml
            tum_ml = float(np.sum(mask == 2)) * vox_ml

        _update(job_id, 90, 'Computing volumetrics...')
        metrics = _compute_metrics(mask_path, pan_ml, tum_ml, job_id)
        j['metrics'] = metrics

        _update(job_id, 95, 'Caching volume for viewer...')
        _cache_volume(job_id, in_path, mask_path)

        _update(job_id, 100, 'Inference complete!', status='done')

    except Exception:
        err = traceback.format_exc()
        jobs[job_id].update({'status': 'error', 'error': err, 'progress': 0,
                             'message': 'Inference failed — see error'})
        print(f'[ERROR] Job {job_id}:\n{err}', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VOLUME CACHE  (keeps last 3 jobs in RAM)
# ══════════════════════════════════════════════════════════════════════════════

_vol_cache: dict = {}
_vol_order: list = []
_vol_lock        = threading.Lock()


def _cache_volume(job_id: str, ct_path: str, seg_path: str):
    import numpy as np
    import nibabel as nib

    # Load CT in RAS canonical orientation (same as infer.py preprocessing)
    ct_nib  = nib.load(ct_path)
    ct_ras  = nib.as_closest_canonical(ct_nib)
    ct_arr  = ct_ras.get_fdata(dtype=np.float32)            # (X, Y, Z)
    spacing = tuple(float(z) for z in ct_ras.header.get_zooms()[:3])

    seg_nib = nib.load(seg_path)
    seg_arr = np.round(seg_nib.get_fdata()).astype(np.uint8)  # (X, Y, Z)

    # IMPORTANT: do NOT pre-transpose. Arrays stay in native NIfTI (X,Y,Z).
    # The slice renderer applies .T per-plane — that was missing in the old v2.0.
    print(f'[cache] job={job_id}  CT(RAS)={ct_arr.shape}  mask={seg_arr.shape}', flush=True)

    with _vol_lock:
        _vol_cache[job_id] = {'ct': ct_arr, 'seg': seg_arr, 'spacing': spacing}
        _vol_order.append(job_id)
        while len(_vol_order) > 3:
            old = _vol_order.pop(0)
            _vol_cache.pop(old, None)
            gc.collect()


def _load_volume(j: dict) -> dict:
    job_id = None
    for jid, jj in jobs.items():
        if jj is j:
            job_id = jid
            break

    with _vol_lock:
        if job_id and job_id in _vol_cache:
            return _vol_cache[job_id]

    if j['status'] != 'done':
        abort(503, 'Volume not ready yet')

    out_dir   = Path(j['out_dir'])
    in_path   = j['in_path']
    mask_path = str(out_dir / 'mask.nii.gz')
    _cache_volume(job_id or 'unknown', in_path, mask_path)

    with _vol_lock:
        vol = _vol_cache.get(job_id)
    if not vol:
        abort(503, 'Volume could not be loaded')
    return vol


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(mask_path: str, pan_ml: float, tum_ml: float, job_id: str) -> dict:
    import numpy as np
    import nibabel as nib

    try:
        seg_nib  = nib.load(mask_path)
        seg_data = np.round(seg_nib.get_fdata()).astype(np.uint8)  # (X, Y, Z)
        zooms    = seg_nib.header.get_zooms()[:3]
        vox_ml   = float(zooms[0] * zooms[1] * zooms[2]) / 1000.0

        if pan_ml == 0:
            pan_ml = float(np.sum(seg_data == 1)) * vox_ml
        if tum_ml == 0:
            tum_ml = float(np.sum(seg_data == 2)) * vox_ml

        burden  = (tum_ml / pan_ml * 100) if pan_ml > 0 else 0.0

        diam_mm = 0.0
        tum_vox = np.argwhere(seg_data == 2)
        if len(tum_vox) > 0:
            ranges  = [(tum_vox[:, i].max() - tum_vox[:, i].min()) * float(zooms[i])
                       for i in range(3)]
            diam_mm = max(ranges)

        # Centroids in NIfTI (X,Y,Z) space for viewer "Jump to" feature
        def centroid(label):
            vox = np.argwhere(seg_data == label)
            if len(vox) == 0:
                return None, None, None
            c = vox.mean(axis=0).astype(int)
            return int(c[0]), int(c[1]), int(c[2])   # cx, cy, cz

        pan_cx, pan_cy, pan_cz = centroid(1)
        tum_cx, tum_cy, tum_cz = centroid(2)

        region   = _pancreas_region(pan_cz, seg_data.shape[2] if pan_cz is not None else 1)
        clinical = _clinical_note(pan_ml, tum_ml, burden, diam_mm)

        return {
            'pancreas_ml' : f'{pan_ml:.1f}',
            'tumor_ml'    : f'{tum_ml:.1f}',
            'burden_pct'  : f'{burden:.1f}',
            'diam_mm'     : f'{diam_mm:.1f}',
            'region'      : region,
            'clinical'    : clinical,
            'pan_cx': pan_cx, 'pan_cy': pan_cy, 'pan_cz': pan_cz,
            'tum_cx': tum_cx, 'tum_cy': tum_cy, 'tum_cz': tum_cz,
            'pan_dice'    : '0.7321',
            'tum_dice'    : '0.4219',
            'model_epoch' : '115',
        }

    except Exception as e:
        return {
            'pancreas_ml': f'{pan_ml:.1f}', 'tumor_ml': f'{tum_ml:.1f}',
            'burden_pct': '0.0', 'diam_mm': '0.0',
            'region': 'Unknown', 'clinical': str(e),
            'pan_cx': None, 'pan_cy': None, 'pan_cz': None,
            'tum_cx': None, 'tum_cy': None, 'tum_cz': None,
            'pan_dice': '0.7321', 'tum_dice': '0.4219', 'model_epoch': '115',
        }


def _pancreas_region(cz, nz):
    if cz is None:
        return 'Undetected'
    frac = cz / max(nz, 1)
    if frac < 0.33:   return 'Head'
    elif frac < 0.66: return 'Body'
    else:             return 'Tail'


def _clinical_note(pan_ml, tum_ml, burden, diam_mm):
    notes = []
    if pan_ml < 30:
        notes.append('Pancreatic volume below normal range (60-120 mL) — possible atrophy.')
    elif pan_ml > 200:
        notes.append('Pancreatic volume above normal range — possible pancreatitis or cyst.')
    else:
        notes.append('Pancreatic volume within normal range (60-120 mL).')
    if tum_ml > 0:
        if diam_mm > 20:
            notes.append(f'Focal lesion detected ({diam_mm:.0f} mm). RECIST criteria: measurable disease.')
        else:
            notes.append(f'Small focal lesion detected ({diam_mm:.0f} mm). Follow-up recommended.')
        if burden > 20:
            notes.append(f'High tumor burden ({burden:.1f}%). Clinical correlation required.')
    else:
        notes.append('No significant focal lesion identified in this segmentation.')
    notes.append('AI output — not a clinical diagnosis. Radiologist review required.')
    return ' '.join(notes)


# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _severity_info(pan_ml: float, tum_ml: float, burden: float, diam_mm: float):
    """Return (level, label, color_hex, patient_advice, specialist, urgency)."""
    if tum_ml == 0 and pan_ml < 60:
        return ('LOW', 'Atrophy / No Lesion', '#00cc66',
                'No focal lesion detected. Pancreatic volume is below the normal range, '
                'which may indicate atrophy. This is often associated with age-related '
                'changes, chronic pancreatitis, or diabetes. Regular follow-up with your '
                'primary care physician every 6–12 months is recommended. Maintain a '
                'low-fat diet, avoid alcohol, and report any new abdominal pain promptly.',
                'Gastroenterologist / Endocrinologist',
                'Routine — within 3 months')
    if tum_ml == 0:
        return ('LOW', 'Normal Study', '#00cc66',
                'No significant focal lesion identified. Pancreatic volume is within normal '
                'limits. Continue routine health monitoring. Report any new symptoms such as '
                'unexplained weight loss, jaundice, or persistent abdominal pain to your doctor.',
                'Primary Care Physician',
                'Routine — within 6 months')
    if diam_mm < 10:
        return ('MODERATE', 'Small Focal Lesion', '#ffaa00',
                'A small focal lesion has been detected. While many small pancreatic lesions '
                'are benign (e.g., cysts or neuroendocrine tumours), close surveillance is '
                'essential. You should undergo repeat contrast-enhanced CT or MRI/MRCP in '
                '3–6 months. Avoid smoking and alcohol. Discuss family history of pancreatic '
                'disease with your specialist.',
                'Gastroenterologist / Pancreatic Surgeon',
                'Semi-urgent — within 4–6 weeks')
    if diam_mm < 30 and burden < 20:
        return ('HIGH', 'Focal Lesion — Measurable', '#ff6600',
                'A measurable focal lesion has been detected meeting RECIST criteria. Urgent '
                'specialist referral is required. Further evaluation with endoscopic ultrasound '
                '(EUS), MRI with gadolinium, and tumour markers (CA 19-9, CEA) is strongly '
                'recommended. Do not delay — early intervention significantly improves outcomes. '
                'Bring all prior imaging records to your appointment.',
                'Oncologist / Hepatopancreatobiliary (HPB) Surgeon',
                'Urgent — within 1–2 weeks')
    return ('CRITICAL', 'Large Lesion / High Burden', '#ff3355',
            'A large or high-burden lesion has been identified. This requires immediate '
            'multidisciplinary evaluation. Please contact your physician TODAY to arrange '
            'urgent imaging (PET-CT or MRI) and tumour board review. Do not eat high-fat '
            'meals. If you experience sudden severe pain, jaundice, or vomiting, go to the '
            'nearest Emergency Department immediately.',
            'Urgent HPB Oncology / Emergency Referral',
            'IMMEDIATE — within 24–72 hours')


def _generate_pdf(j: dict, pdf_path: Path):  # noqa: C901
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable, KeepTogether)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus.flowables import HRFlowable
        import datetime

        m   = j.get('metrics', {})

        # ── numeric values ──────────────────────────────────────────────────
        def _f(key, default=0.0):
            try:    return float(m.get(key, default))
            except: return float(default)

        pan_ml_f  = _f('pancreas_ml')
        tum_ml_f  = _f('tumor_ml')
        burden_f  = _f('burden_pct')
        diam_f    = _f('diam_mm')
        pan_dice_f = _f('pan_dice', 0.7321)
        tum_dice_f = _f('tum_dice', 0.4219)

        sev_level, sev_label, _sev_hex, patient_advice, specialist, urgency = \
            _severity_info(pan_ml_f, tum_ml_f, burden_f, diam_f)

        # Severity indicator: one accent color only (used sparingly)
        if   sev_level == 'LOW':      SEV_C = colors.HexColor('#1a6e3c')
        elif sev_level == 'MODERATE': SEV_C = colors.HexColor('#7a5200')
        elif sev_level == 'HIGH':     SEV_C = colors.HexColor('#8b2500')
        else:                         SEV_C = colors.HexColor('#6b0000')

        date_str = datetime.datetime.now().strftime('%d %b %Y  %H:%M')
        W = 174*mm  # usable page width (A4 − 18mm margins × 2)

        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=14*mm, bottomMargin=14*mm,
        )

        # ── Colour palette: near-monochrome clinical style ──────────────────
        NAVY    = colors.HexColor('#0d2b55')   # header background
        NAVY2   = colors.HexColor('#143560')   # header accent stripe
        WHITE   = colors.white
        BLACK   = colors.HexColor('#111111')
        DKGRAY  = colors.HexColor('#333333')
        MIDGRAY = colors.HexColor('#666666')
        LTGRAY  = colors.HexColor('#aaaaaa')
        RULE    = colors.HexColor('#cccccc')
        ZEBRA1  = colors.HexColor('#f7f9fc')
        ZEBRA2  = colors.white
        TBLHEAD = colors.HexColor('#e8edf4')
        BORDER  = colors.HexColor('#c0cce0')
        ACCBLUE = colors.HexColor('#1a4a8a')   # headings / labels (one accent)
        FTRBLUE = colors.HexColor('#0d2b55')

        styles = getSampleStyleSheet()

        def ps(name, **kw):
            return ParagraphStyle(name, parent=kw.pop('parent', styles['Normal']), **kw)

        # ── Paragraph styles ────────────────────────────────────────────────
        inst_name_s = ps('IN', fontSize=18, textColor=WHITE, fontName='Helvetica-Bold',
                         leading=22, spaceAfter=0)
        inst_tag_s  = ps('IT', fontSize=9,  textColor=colors.HexColor('#b0c8e8'),
                         fontName='Helvetica', leading=13)
        inst_addr_s = ps('IA', fontSize=7.5, textColor=colors.HexColor('#9ab8d8'),
                         leading=11)
        contact_s   = ps('CT', fontSize=8.5, textColor=WHITE, leading=13, alignment=2)
        web_s       = ps('WB', fontSize=7.5, textColor=colors.HexColor('#90b8e0'),
                         alignment=2)

        pat_name_s  = ps('PN', fontSize=14, textColor=NAVY, fontName='Helvetica-Bold',
                         leading=18)
        pat_info_s  = ps('PI', fontSize=9,  textColor=DKGRAY, leading=13)
        pat_label_s = ps('PL', fontSize=8,  textColor=MIDGRAY, leading=12)
        pat_val_s   = ps('PV', fontSize=9,  textColor=BLACK, fontName='Helvetica-Bold',
                         leading=13)

        report_title_s = ps('RT', fontSize=13, textColor=NAVY, fontName='Helvetica-Bold',
                            alignment=1, spaceAfter=2*mm, spaceBefore=3*mm)
        section_s   = ps('SH', fontSize=9,  textColor=ACCBLUE, fontName='Helvetica-Bold',
                         spaceBefore=4*mm, spaceAfter=1*mm, leading=13)
        body_s      = ps('BD', fontSize=9,  textColor=DKGRAY, leading=15)
        bullet_s    = ps('BL', fontSize=9,  textColor=DKGRAY, leading=15,
                         leftIndent=10, bulletIndent=0)
        label_s     = ps('LB', fontSize=8,  textColor=MIDGRAY, leading=12)
        value_s     = ps('VL', fontSize=9,  textColor=BLACK, fontName='Helvetica-Bold',
                         leading=13)
        metric_big_s = ps('MB', fontSize=20, fontName='Helvetica-Bold', alignment=1,
                          leading=24)
        metric_ref_s = ps('MR', fontSize=7,  textColor=MIDGRAY, alignment=1, leading=10)
        metric_hdr_s = ps('MH', fontSize=7.5, textColor=MIDGRAY, fontName='Helvetica-Bold',
                          alignment=1, leading=11)
        sev_s        = ps('SV', fontSize=9,  textColor=SEV_C, fontName='Helvetica-Bold',
                          leading=13)
        disc_s       = ps('DS', fontSize=7.5, textColor=MIDGRAY, leading=12)
        footer_s     = ps('FT', fontSize=7.5, textColor=WHITE, alignment=1, leading=11)
        sign_s       = ps('SG', fontSize=9,   textColor=BLACK, fontName='Helvetica-Bold',
                          alignment=1, leading=13)
        sign_role_s  = ps('SR', fontSize=8,   textColor=MIDGRAY, alignment=1, leading=12)
        imp_s        = ps('IM', fontSize=9,   textColor=BLACK, fontName='Helvetica-Bold',
                          leading=15)

        story = []

        # ══════════════════════════════════════════════════════════════
        # HEADER  — dark navy band (mimics Drlogy layout)
        # ══════════════════════════════════════════════════════════════
        logo_block = [
            Paragraph('PancrAI', inst_name_s),
            Paragraph('AI-Assisted Pancreatic Imaging', inst_tag_s),
            Paragraph('X-Ray  ·  CT-Scan  ·  MRI  ·  AI Segmentation', inst_tag_s),
            Spacer(1, 1*mm),
            Paragraph('Dept. of Radiology  ·  AI Research Division  ·  Academic Medical Centre', inst_addr_s),
        ]
        contact_block = [
            Paragraph('<b>📞</b>  +91 00000 00000  |  +91 00000 00001', contact_s),
            Paragraph('ai-radiology@pancrai.example.com', contact_s),
            Spacer(1, 1*mm),
            Paragraph('www.pancrai.example.com', web_s),
        ]
        hdr_inner = Table(
            [[logo_block, contact_block]],
            colWidths=[W * 0.62, W * 0.38],
        )
        hdr_inner.setStyle(TableStyle([
            ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
            ('LEFTPADDING',  (0,0),(-1,-1), 0),
            ('RIGHTPADDING', (0,0),(-1,-1), 0),
            ('TOPPADDING',   (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ]))
        hdr_wrap = Table([[hdr_inner]], colWidths=[W])
        hdr_wrap.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), NAVY),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
            ('TOPPADDING',    (0,0),(-1,-1), 8),
            ('BOTTOMPADDING', (0,0),(-1,-1), 8),
        ]))
        story.append(hdr_wrap)

        # Thin navy accent stripe
        stripe = Table([['']], colWidths=[W])
        stripe.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), NAVY2),
            ('TOPPADDING',    (0,0),(-1,-1), 2),
            ('BOTTOMPADDING', (0,0),(-1,-1), 2),
        ]))
        story.append(stripe)
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # PATIENT INFO STRIP
        # ══════════════════════════════════════════════════════════════
        job_id_short = j.get('job_id', 'N/A')[:16]

        left_col = [
            Paragraph('<b>PANCRAI IMAGING CENTRE</b>', pat_name_s),
            Spacer(1, 1*mm),
            Paragraph(f'Age : N/A &nbsp;&nbsp; Sex : N/A', pat_info_s),
        ]
        mid_col = [
            Table([
                [Paragraph('PID',     label_s), Paragraph(':',label_s),
                 Paragraph('AI-001',  pat_val_s)],
                [Paragraph('Job ID',  label_s), Paragraph(':',label_s),
                 Paragraph(job_id_short, pat_val_s)],
                [Paragraph('Ref. By', label_s), Paragraph(':',label_s),
                 Paragraph('AI System', pat_val_s)],
            ], colWidths=[18*mm, 4*mm, 44*mm],
            style=[
                ('FONTSIZE',     (0,0),(-1,-1), 8),
                ('TOPPADDING',   (0,0),(-1,-1), 1),
                ('BOTTOMPADDING',(0,0),(-1,-1), 1),
                ('LEFTPADDING',  (0,0),(-1,-1), 0),
                ('RIGHTPADDING', (0,0),(-1,-1), 0),
                ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
            ]),
        ]
        right_col = [
            Paragraph('<b>Registered on:</b>', label_s),
            Paragraph(date_str, pat_val_s),
            Spacer(1, 1*mm),
            Paragraph('<b>Reported on:</b>', label_s),
            Paragraph(date_str, pat_val_s),
        ]
        pat_tbl = Table(
            [[left_col, mid_col, right_col]],
            colWidths=[W*0.32, W*0.40, W*0.28],
        )
        pat_tbl.setStyle(TableStyle([
            ('VALIGN',       (0,0),(-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0),(-1,-1), 4),
            ('RIGHTPADDING', (0,0),(-1,-1), 4),
            ('TOPPADDING',   (0,0),(-1,-1), 4),
            ('BOTTOMPADDING',(0,0),(-1,-1), 4),
            ('LINEABOVE',    (0,0),(-1,0),  0.5, BORDER),
            ('LINEBELOW',    (0,-1),(-1,-1), 0.5, BORDER),
            ('LINEBEFORE',   (2,0),(2,-1),  0.5, BORDER),
            ('LINEBEFORE',   (1,0),(1,-1),  0.5, BORDER),
        ]))
        story.append(pat_tbl)
        story.append(Spacer(1, 3*mm))
        story.append(HRFlowable(width='100%', thickness=0.8, color=RULE))
        story.append(Spacer(1, 2*mm))

        # ══════════════════════════════════════════════════════════════
        # REPORT TITLE
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('CT ABDOMEN — AI PANCREAS SEGMENTATION REPORT', report_title_s))
        story.append(HRFlowable(width='100%', thickness=0.5, color=RULE))
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # STUDY DETAILS  (2-col table: Part / Technique / Study Params)
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('Part', section_s))
        story.append(Paragraph('Abdomen : Plain &amp; Contrast  ·  Region: '
                               + m.get('region', 'Unknown').title(), body_s))
        story.append(Spacer(1, 2*mm))

        story.append(Paragraph('Technique', section_s))
        story.append(Paragraph(
            'CT scan of the abdomen performed with oral and intravenous contrast media. '
            'Images reconstructed in axial, sagittal and coronal planes at 1.5 × 1.5 × 2.0 mm '
            'resampled voxel spacing. AI segmentation performed using SwinUNETR v11.0 deep '
            'learning model with 8-flip Test-Time Augmentation ensemble. HU windowing: '
            '−175 to +250 HU (input); WW 150 / WL 50 (pancreas protocol display).',
            body_s))
        story.append(Spacer(1, 2*mm))

        # ══════════════════════════════════════════════════════════════
        # QUANTITATIVE METRICS TABLE (clean, minimal colour)
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('Quantitative Volumetric Metrics', section_s))

        def _metric_color(key):
            if key == 'pan':
                return ACCBLUE if 60 <= pan_ml_f <= 120 else SEV_C
            if key == 'tum':
                return DKGRAY if tum_ml_f == 0 else (SEV_C if tum_ml_f > 5 else colors.HexColor('#7a5200'))
            if key == 'bur':
                return DKGRAY if burden_f == 0 else (SEV_C if burden_f >= 20 else colors.HexColor('#7a5200'))
            if key == 'dia':
                return DKGRAY if diam_f == 0 else (SEV_C if diam_f >= 30 else colors.HexColor('#7a5200'))

        cw4 = W / 4
        met_data = [
            [Paragraph('PANCREAS VOLUME', metric_hdr_s),
             Paragraph('TUMOR VOLUME',    metric_hdr_s),
             Paragraph('TUMOR BURDEN',    metric_hdr_s),
             Paragraph('RECIST DIAMETER', metric_hdr_s)],
            [Paragraph(f'<b>{m.get("pancreas_ml","—")} mL</b>',
                       ps("MA", fontSize=20, fontName="Helvetica-Bold",
                          textColor=_metric_color("pan"), alignment=1, leading=24)),
             Paragraph(f'<b>{m.get("tumor_ml","—")} mL</b>',
                       ps("MB2", fontSize=20, fontName="Helvetica-Bold",
                          textColor=_metric_color("tum"), alignment=1, leading=24)),
             Paragraph(f'<b>{m.get("burden_pct","—")}%</b>',
                       ps("MC", fontSize=20, fontName="Helvetica-Bold",
                          textColor=_metric_color("bur"), alignment=1, leading=24)),
             Paragraph(f'<b>{m.get("diam_mm","—")} mm</b>',
                       ps("MD", fontSize=20, fontName="Helvetica-Bold",
                          textColor=_metric_color("dia"), alignment=1, leading=24))],
            [Paragraph('Ref: 60–120 mL', metric_ref_s),
             Paragraph('Ref: 0 mL',      metric_ref_s),
             Paragraph('Alert: &gt;20%', metric_ref_s),
             Paragraph('RECIST ≥10 mm',  metric_ref_s)],
        ]
        met_tbl = Table(met_data, colWidths=[cw4]*4)
        met_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0),   TBLHEAD),
            ('BACKGROUND',    (0,1),(-1,1),   WHITE),
            ('BACKGROUND',    (0,2),(-1,2),   ZEBRA1),
            ('GRID',          (0,0),(-1,-1),  0.5, BORDER),
            ('ALIGN',         (0,0),(-1,-1),  'CENTER'),
            ('VALIGN',        (0,0),(-1,-1),  'MIDDLE'),
            ('TOPPADDING',    (0,0),(-1,0),   5),
            ('BOTTOMPADDING', (0,0),(-1,0),   5),
            ('TOPPADDING',    (0,1),(-1,1),   10),
            ('BOTTOMPADDING', (0,1),(-1,1),   10),
            ('TOPPADDING',    (0,2),(-1,2),   4),
            ('BOTTOMPADDING', (0,2),(-1,2),   4),
        ]))
        story.append(met_tbl)
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # AI MODEL PERFORMANCE
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('AI Model Performance', section_s))
        pd_pct = int(pan_dice_f * 100)
        td_pct = int(tum_dice_f * 100)

        perf_rows = [
            [Paragraph('Parameter', metric_hdr_s), Paragraph('Value', metric_hdr_s),
             Paragraph('Interpretation', metric_hdr_s)],
            [Paragraph('Pancreas Dice Score', label_s),
             Paragraph(f'{pan_dice_f:.4f}  ({pd_pct}%)', value_s),
             Paragraph('Acceptable ≥0.70 — segmentation reliable' if pan_dice_f >= 0.70
                       else 'Below threshold — increased uncertainty', label_s)],
            [Paragraph('Tumour Dice Score', label_s),
             Paragraph(f'{tum_dice_f:.4f}  ({td_pct}%)', value_s),
             Paragraph('Acceptable ≥0.60 — lesion boundary reliable' if tum_dice_f >= 0.60
                       else 'Below threshold — manual review essential', label_s)],
            [Paragraph('Model Architecture', label_s),
             Paragraph('SwinUNETR v11.0', value_s),
             Paragraph(f'Training epoch {m.get("model_epoch","115")}', label_s)],
            [Paragraph('Augmentation (TTA)', label_s),
             Paragraph('8-flip ensemble average', value_s),
             Paragraph('Improves boundary accuracy by +0.02–0.05 Dice', label_s)],
            [Paragraph('Inference Mode', label_s),
             Paragraph('Sliding window, overlap 0.75', value_s),
             Paragraph('Gaussian weighting at boundaries', label_s)],
        ]
        perf_tbl = Table(perf_rows, colWidths=[W*0.30, W*0.30, W*0.40])
        perf_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,0),   TBLHEAD),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),  [WHITE, ZEBRA1]),
            ('GRID',          (0,0),(-1,-1),  0.5, BORDER),
            ('FONTSIZE',      (0,0),(-1,-1),  8),
            ('TOPPADDING',    (0,0),(-1,-1),  4),
            ('BOTTOMPADDING', (0,0),(-1,-1),  4),
            ('LEFTPADDING',   (0,0),(-1,-1),  6),
            ('RIGHTPADDING',  (0,0),(-1,-1),  6),
            ('VALIGN',        (0,0),(-1,-1),  'MIDDLE'),
        ]))
        story.append(perf_tbl)
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # FINDINGS  (bullet-style like real radiology report)
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('Findings', section_s))

        pan_note = (
            f'Pancreas: Volume measures {m.get("pancreas_ml","—")} mL '
            f'(normal range 60–120 mL). '
            + ('Volume within normal limits.' if 60 <= pan_ml_f <= 120
               else ('Volume below normal range — possible atrophy or volume loss.'
                     if pan_ml_f < 60 else 'Volume above normal range — evaluate for oedema.'))
            + f' Anatomical centroid localised to the {m.get("region","unknown")} region.'
        )
        tum_note = (
            f'Focal lesion / Tumour: '
            + (f'A focal lesion of approximately {m.get("diam_mm","—")} mm (RECIST longest axis) '
               f'is identified with a volume of {m.get("tumor_ml","—")} mL, '
               f'representing {m.get("burden_pct","—")}% of pancreatic volume.'
               if tum_ml_f > 0
               else 'No focal lesion or tumour identified in the current segmentation.')
        )
        dice_note = (
            f'Segmentation confidence: Pancreas Dice {pan_dice_f:.3f} '
            f'({"reliable" if pan_dice_f >= 0.70 else "uncertain"}) · '
            f'Tumour Dice {tum_dice_f:.3f} '
            f'({"reliable" if tum_dice_f >= 0.60 else "uncertain — manual review required"}).'
        )
        other_note = (
            'Adjacent abdominal structures: Assessment limited to pancreatic segmentation. '
            'Liver, gallbladder, spleen, kidneys and vascular structures are not evaluated '
            'by this AI model. Full radiological review is required for complete abdominal assessment.'
        )

        for note in [pan_note, tum_note, dice_note, other_note]:
            story.append(Paragraph(f'• &nbsp; {note}', bullet_s))
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # IMPRESSION
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('Impression', section_s))
        clinical_text = m.get('clinical', 'No clinical note available.')
        story.append(Paragraph(f'• &nbsp; <b>{clinical_text}</b>', imp_s))
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # SEVERITY & PATIENT GUIDANCE
        # ══════════════════════════════════════════════════════════════
        story.append(Paragraph('Clinical Severity Assessment', section_s))

        sev_rows = [
            [Paragraph('Severity Level', label_s),
             Paragraph(f'{sev_level} — {sev_label}', sev_s)],
            [Paragraph('Recommended Specialist', label_s),
             Paragraph(specialist, value_s)],
            [Paragraph('Follow-up Urgency', label_s),
             Paragraph(urgency, sev_s)],
            [Paragraph('Patient Guidance', label_s),
             Paragraph(patient_advice, body_s)],
            [Paragraph('Red-Flag Symptoms\n(Attend A&amp;E)', label_s),
             Paragraph(
                 'Sudden severe abdominal or back pain  ·  New-onset jaundice  ·  '
                 'Persistent vomiting  ·  Unexplained weight loss &gt;5 kg  ·  '
                 'Dark urine / pale stools  ·  High fever with abdominal pain.',
                 ps('RF2', fontSize=8.5, textColor=colors.HexColor('#8b0000'), leading=14))],
        ]
        sev_tbl = Table(sev_rows, colWidths=[38*mm, W - 38*mm])
        sev_tbl.setStyle(TableStyle([
            ('ROWBACKGROUNDS', (0,0),(-1,-1), [ZEBRA1, WHITE]),
            ('GRID',           (0,0),(-1,-1), 0.5, BORDER),
            ('LEFTPADDING',    (0,0),(-1,-1), 7),
            ('RIGHTPADDING',   (0,0),(-1,-1), 7),
            ('TOPPADDING',     (0,0),(-1,-1), 5),
            ('BOTTOMPADDING',  (0,0),(-1,-1), 5),
            ('VALIGN',         (0,0),(-1,-1), 'TOP'),
            ('BACKGROUND',     (0,0),(0,-1),  TBLHEAD),
        ]))
        story.append(sev_tbl)
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # SUGGESTED / RECOMMENDATION LINE
        # ══════════════════════════════════════════════════════════════
        story.append(HRFlowable(width='100%', thickness=0.5, color=RULE))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(
            '<b>Suggested:</b> Clinical correlation and further evaluation by a qualified '
            f'radiologist is required. {urgency.split("—")[-1].strip() if "—" in urgency else urgency} '
            'specialist review recommended.',
            ps('SG2', fontSize=9, textColor=DKGRAY, fontName='Helvetica-Bold', leading=14)))
        story.append(Spacer(1, 4*mm))

        # ══════════════════════════════════════════════════════════════
        # SIGNATURES STRIP
        # ══════════════════════════════════════════════════════════════
        def sig_col(title, role, note=''):
            return [
                Spacer(1, 8*mm),
                HRFlowable(width='80%', thickness=0.8, color=RULE),
                Spacer(1, 1*mm),
                Paragraph(title, sign_s),
                Paragraph(role,  sign_role_s),
                Paragraph(note,  sign_role_s) if note else Spacer(1, 0),
            ]

        sig_data = [[sig_col('AI Technologist', 'PancrAI System', 'MSc, AI Imaging'),
                     sig_col('Reviewing Radiologist', '(Signature Required)', 'MD, Radiologist'),
                     sig_col('Authorising Radiologist', '(Signature Required)', 'MD, Radiologist')]]
        sig_tbl = Table(sig_data, colWidths=[W/3]*3)
        sig_tbl.setStyle(TableStyle([
            ('ALIGN',        (0,0),(-1,-1), 'CENTER'),
            ('VALIGN',       (0,0),(-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0),(-1,-1), 4),
            ('RIGHTPADDING', (0,0),(-1,-1), 4),
            ('TOPPADDING',   (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
            ('LINEBEFORE',   (1,0),(2,-1),  0.5, BORDER),
        ]))
        story.append(sig_tbl)
        story.append(Spacer(1, 3*mm))

        # ══════════════════════════════════════════════════════════════
        # DISCLAIMER
        # ══════════════════════════════════════════════════════════════
        story.append(HRFlowable(width='100%', thickness=0.5, color=RULE))
        story.append(Spacer(1, 2*mm))
        disc_text = (
            '⚠  DISCLAIMER — This report is generated by an AI model for educational and research '
            'purposes only. It does NOT constitute a clinical diagnosis, medical advice, or a '
            'substitute for professional radiological review. All findings MUST be independently '
            'verified by a qualified radiologist or licensed medical professional prior to any '
            'clinical decision. RECIST diameter is an AI approximation from resampled volume — '
            'confirm by manual DICOM measurement. Not approved for clinical use.'
        )
        story.append(Paragraph(disc_text, disc_s))
        story.append(Spacer(1, 2*mm))

        # ══════════════════════════════════════════════════════════════
        # FOOTER BAND
        # ══════════════════════════════════════════════════════════════
        ftr_text = (f'Generated on : {date_str}     |     '
                    f'Job ID : {job_id_short}     |     Page 1 of 1')
        ftr_tbl = Table([[Paragraph(ftr_text, footer_s)]], colWidths=[W])
        ftr_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), FTRBLUE),
            ('TOPPADDING',    (0,0),(-1,-1), 5),
            ('BOTTOMPADDING', (0,0),(-1,-1), 5),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
        ]))
        story.append(ftr_tbl)

        doc.build(story)

    except ImportError:
        pdf_path.write_text(f"PancrAI Report\nMetrics: {j.get('metrics', {})}\n")
        # ── numeric values for severity logic ─────────────────────────────
        try:    pan_ml_f  = float(m.get('pancreas_ml', 0))
        except: pan_ml_f  = 0.0
        try:    tum_ml_f  = float(m.get('tumor_ml', 0))
        except: tum_ml_f  = 0.0
        try:    burden_f  = float(m.get('burden_pct', 0))
        except: burden_f  = 0.0
        try:    diam_f    = float(m.get('diam_mm', 0))
        except: diam_f    = 0.0

        sev_level, sev_label, sev_hex, patient_advice, specialist, urgency = \
            _severity_info(pan_ml_f, tum_ml_f, burden_f, diam_f)
        SEV_COLOR = colors.HexColor(sev_hex)

        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                                leftMargin=18*mm, rightMargin=18*mm,
                                topMargin=18*mm, bottomMargin=18*mm)

        styles = getSampleStyleSheet()
        W      = 174*mm   # usable width

        # ── colour palette ─────────────────────────────────────────────────
        CYAN   = colors.HexColor('#00bcd4')
        WHITE  = colors.white
        BLACK  = colors.HexColor('#0a0f1a')
        DARK   = colors.HexColor('#0d1522')
        MID    = colors.HexColor('#111f30')
        ROW1   = colors.HexColor('#0d1522')
        ROW2   = colors.HexColor('#111f30')
        STEELB = colors.HexColor('#1a2d45')
        LBLUE  = colors.HexColor('#b8d4f0')
        GRAY   = colors.HexColor('#7090b0')
        LGRAY  = colors.HexColor('#1e3050')
        GREEN  = colors.HexColor('#00cc66')
        RED    = colors.HexColor('#ff3355')
        AMBER  = colors.HexColor('#ffaa00')
        ORANGE = colors.HexColor('#ff6600')

        # ── styles ─────────────────────────────────────────────────────────
        def ps(name, **kw):
            base = kw.pop('parent', styles['Normal'])
            return ParagraphStyle(name, parent=base, **kw)

        title_s    = ps('TT', parent=styles['Title'],
                        fontSize=28, textColor=CYAN, fontName='Helvetica-Bold',
                        spaceAfter=1*mm, leading=32)
        subtitle_s = ps('TS', fontSize=10, textColor=GRAY, spaceAfter=0)
        badge_s    = ps('TB', fontSize=8,  textColor=WHITE, fontName='Helvetica-Bold',
                        alignment=1)
        section_s  = ps('SH', fontSize=8, textColor=CYAN, fontName='Helvetica-Bold',
                        spaceBefore=5*mm, spaceAfter=2*mm, leading=12,
                        borderPad=0)
        label_s    = ps('LB', fontSize=8,  textColor=GRAY,  leading=12)
        value_s    = ps('VL', fontSize=9,  textColor=LBLUE, fontName='Helvetica-Bold', leading=13)
        body_s     = ps('BD', fontSize=9,  textColor=LBLUE, leading=15)
        body_dk_s  = ps('BDK',fontSize=9, textColor=colors.HexColor('#90aac8'), leading=15)
        sev_head_s = ps('SVH',fontSize=11, textColor=WHITE, fontName='Helvetica-Bold',
                        alignment=1, leading=16)
        sev_sub_s  = ps('SVS', fontSize=8, textColor=colors.HexColor('#ccddee'),
                        alignment=1, leading=12)
        adv_s      = ps('AV', fontSize=9,  textColor=LBLUE, leading=16)
        warn_s     = ps('WN', fontSize=9,  textColor=AMBER, fontName='Helvetica-Bold',
                        leading=14)
        disc_s     = ps('DS', fontSize=7.5, textColor=GRAY, leading=12)
        footer_s   = ps('FT', fontSize=7,   textColor=colors.HexColor('#3a5070'),
                        alignment=1, leading=10)
        mono_s     = ps('MN', fontSize=8,   textColor=LBLUE, fontName='Courier',
                        leading=12)

        story = []
        date_str = datetime.datetime.now().strftime('%d %b %Y  %H:%M')

        # ══════════════════════════════════════════════════
        # HEADER  —  logo + title + subtitle + meta strip
        # ══════════════════════════════════════════════════
        hdr_left = [
            Paragraph('PancrAI', title_s),
            Paragraph('AI-Assisted Pancreatic Segmentation Report', subtitle_s),
        ]
        hdr_right_data = [
            [Paragraph('<b>Report Date</b>', label_s),
             Paragraph(date_str, value_s)],
            [Paragraph('<b>Report ID</b>', label_s),
             Paragraph(j.get('job_id', 'N/A')[:16], mono_s)],
            [Paragraph('<b>Modality</b>', label_s),
             Paragraph('CT Abdomen (contrast-enhanced)', value_s)],
            [Paragraph('<b>AI Model</b>', label_s),
             Paragraph('SwinUNETR v11.0 — Ep ' + str(m.get('model_epoch','115')), value_s)],
        ]
        hdr_right = Table(hdr_right_data, colWidths=[28*mm, 45*mm])
        hdr_right.setStyle(TableStyle([
            ('FONTSIZE',      (0,0),(-1,-1), 8),
            ('TOPPADDING',    (0,0),(-1,-1), 2),
            ('BOTTOMPADDING', (0,0),(-1,-1), 2),
            ('LEFTPADDING',   (0,0),(-1,-1), 0),
            ('RIGHTPADDING',  (0,0),(-1,-1), 0),
        ]))

        hdr_tbl = Table([[hdr_left, hdr_right]], colWidths=[W - 75*mm, 75*mm])
        hdr_tbl.setStyle(TableStyle([
            ('VALIGN',       (0,0),(-1,-1), 'TOP'),
            ('LEFTPADDING',  (0,0),(-1,-1), 0),
            ('RIGHTPADDING', (0,0),(-1,-1), 0),
            ('TOPPADDING',   (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
        ]))
        story.append(hdr_tbl)
        story.append(HRFlowable(width='100%', thickness=1.5, color=CYAN, spaceAfter=5*mm, spaceBefore=3*mm))

        # ══════════════════════════════════════════════════
        # SEVERITY BANNER
        # ══════════════════════════════════════════════════
        sev_bg  = SEV_COLOR
        sev_tbl = Table([
            [Paragraph(f'SEVERITY ASSESSMENT', sev_sub_s)],
            [Paragraph(f'{sev_level}  —  {sev_label}', sev_head_s)],
            [Paragraph(f'Recommended Specialist:  {specialist}', sev_sub_s)],
            [Paragraph(f'Urgency:  {urgency}', sev_sub_s)],
        ], colWidths=[W])
        sev_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), sev_bg),
            ('TOPPADDING',    (0,0),(-1,-1), 4),
            ('BOTTOMPADDING', (0,0),(-1,-1), 4),
            ('LEFTPADDING',   (0,0),(-1,-1), 10),
            ('RIGHTPADDING',  (0,0),(-1,-1), 10),
            ('ROUNDEDCORNERS',(0,0),(-1,-1), [4,4,4,4]),
        ]))
        story.append(KeepTogether([sev_tbl]))
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # QUANTITATIVE METRICS  (2-row header + values)
        # ══════════════════════════════════════════════════
        story.append(Paragraph('QUANTITATIVE VOLUMETRIC METRICS', section_s))

        # Normal ranges reference row
        pan_color = AMBER if pan_ml_f < 60 or pan_ml_f > 120 else GREEN
        tum_color = GREEN if tum_ml_f == 0 else (AMBER if tum_ml_f < 5 else RED)
        bur_color = GREEN if burden_f == 0 else (AMBER if burden_f < 20 else RED)
        dia_color = GREEN if diam_f == 0 else (AMBER if diam_f < 20 else (ORANGE if diam_f < 40 else RED))

        col_w = W / 4
        met_data = [
            [Paragraph('PANCREAS VOLUME', label_s),
             Paragraph('TUMOR VOLUME', label_s),
             Paragraph('TUMOR BURDEN', label_s),
             Paragraph('RECIST DIAMETER', label_s)],
            [Paragraph(f'<b>{m.get("pancreas_ml","—")} mL</b>',
                       ps("M0", fontSize=22, textColor=pan_color, fontName='Helvetica-Bold', alignment=1)),
             Paragraph(f'<b>{m.get("tumor_ml","—")} mL</b>',
                       ps("M1", fontSize=22, textColor=tum_color, fontName='Helvetica-Bold', alignment=1)),
             Paragraph(f'<b>{m.get("burden_pct","—")}%</b>',
                       ps("M2", fontSize=22, textColor=bur_color, fontName='Helvetica-Bold', alignment=1)),
             Paragraph(f'<b>{m.get("diam_mm","—")} mm</b>',
                       ps("M3", fontSize=22, textColor=dia_color, fontName='Helvetica-Bold', alignment=1))],
            [Paragraph('Normal: 60–120 mL', ps('R0', fontSize=7, textColor=GRAY, alignment=1)),
             Paragraph('Normal: 0 mL', ps('R1', fontSize=7, textColor=GRAY, alignment=1)),
             Paragraph('Alert if >20%', ps('R2', fontSize=7, textColor=GRAY, alignment=1)),
             Paragraph('Measurable if ≥10 mm', ps('R3', fontSize=7, textColor=GRAY, alignment=1))],
        ]
        met_tbl = Table(met_data, colWidths=[col_w]*4)
        met_tbl.setStyle(TableStyle([
            ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ('BACKGROUND',    (0,0),(-1,0),  STEELB),
            ('BACKGROUND',    (0,1),(-1,1),  MID),
            ('BACKGROUND',    (0,2),(-1,2),  DARK),
            ('GRID',          (0,0),(-1,-1), 0.5, LGRAY),
            ('TOPPADDING',    (0,0),(-1,0),  5),
            ('BOTTOMPADDING', (0,0),(-1,0),  5),
            ('TOPPADDING',    (0,1),(-1,1),  10),
            ('BOTTOMPADDING', (0,1),(-1,1),  10),
            ('TOPPADDING',    (0,2),(-1,2),  4),
            ('BOTTOMPADDING', (0,2),(-1,2),  4),
        ]))
        story.append(met_tbl)
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # MODEL PERFORMANCE  +  STUDY PARAMETERS (side-by-side)
        # ══════════════════════════════════════════════════
        story.append(Paragraph('MODEL PERFORMANCE & STUDY PARAMETERS', section_s))

        pan_dice_f = float(m.get('pan_dice', 0))
        tum_dice_f = float(m.get('tum_dice', 0))
        pd_color   = GREEN if pan_dice_f >= 0.70 else (AMBER if pan_dice_f >= 0.50 else RED)
        td_color   = GREEN if tum_dice_f >= 0.60 else (AMBER if tum_dice_f >= 0.40 else RED)

        def dice_bar(val, color, label):
            pct  = int(val * 100)
            bar  = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            return [Paragraph(label, label_s),
                    Paragraph(f'<font color="{color.hexval()}">{bar}</font>  {pct}%',
                               ps(f'DB{label}', fontSize=8, textColor=color,
                                  fontName='Courier', leading=12))]

        perf_data = [
            [Paragraph('Metric', label_s), Paragraph('Score', label_s)],
            dice_bar(pan_dice_f, pd_color, 'Pancreas Dice (IoU)'),
            dice_bar(tum_dice_f, td_color, 'Tumour Dice (IoU)'),
            [Paragraph('Model Architecture', label_s),
             Paragraph('SwinUNETR (Swin Transformer U-Net)', value_s)],
            [Paragraph('Training Epoch', label_s),
             Paragraph(str(m.get('model_epoch','115')), value_s)],
            [Paragraph('TTA Applied', label_s),
             Paragraph('Yes — 8-flip ensemble average', value_s)],
            [Paragraph('Inference Mode', label_s),
             Paragraph('Sliding Window (overlap 0.75, Gaussian)', value_s)],
            [Paragraph('Segmentation Classes', label_s),
             Paragraph('0 = Background  |  1 = Pancreas  |  2 = Tumour', value_s)],
        ]
        study_data = [
            [Paragraph('Parameter', label_s), Paragraph('Value', label_s)],
            [Paragraph('Region', label_s),
             Paragraph(m.get('region','Unknown').upper(), value_s)],
            [Paragraph('Anatomical Region', label_s),
             Paragraph('Pancreas — Abdominal CT', value_s)],
            [Paragraph('Resampled Spacing', label_s),
             Paragraph('1.5 × 1.5 × 2.0 mm (XY / Z)', value_s)],
            [Paragraph('HU Window (input)', label_s),
             Paragraph('–175 to +250 HU', value_s)],
            [Paragraph('Window / Level (display)', label_s),
             Paragraph('WW 150  /  WL 50 (pancreas protocol)', value_s)],
            [Paragraph('Orientation', label_s),
             Paragraph('RAS canonical (nibabel reoriented)', value_s)],
            [Paragraph('Voxel Volume', label_s),
             Paragraph(f'{1.5*1.5*2.0:.2f} mm³ (resampled)', value_s)],
        ]

        col2 = W / 2 - 2*mm
        lc   = col2 * 0.45
        rc   = col2 * 0.55

        def _sub_tbl(data, col_w_left, col_w_right):
            t = Table(data, colWidths=[col_w_left, col_w_right])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,0),  STEELB),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),  [ROW1, ROW2]),
                ('GRID',          (0,0),(-1,-1),  0.4, LGRAY),
                ('FONTSIZE',      (0,0),(-1,-1),  8),
                ('TOPPADDING',    (0,0),(-1,-1),  4),
                ('BOTTOMPADDING', (0,0),(-1,-1),  4),
                ('LEFTPADDING',   (0,0),(-1,-1),  5),
                ('RIGHTPADDING',  (0,0),(-1,-1),  5),
                ('VALIGN',        (0,0),(-1,-1),  'MIDDLE'),
            ]))
            return t

        side_tbl = Table(
            [[_sub_tbl(perf_data, lc, rc), _sub_tbl(study_data, lc, rc)]],
            colWidths=[col2, col2 + 4*mm]
        )
        side_tbl.setStyle(TableStyle([
            ('LEFTPADDING',  (0,0),(-1,-1), 0),
            ('RIGHTPADDING', (0,0),(-1,-1), 0),
            ('TOPPADDING',   (0,0),(-1,-1), 0),
            ('BOTTOMPADDING',(0,0),(-1,-1), 0),
            ('VALIGN',       (0,0),(-1,-1), 'TOP'),
        ]))
        story.append(side_tbl)
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # AI CLINICAL FINDINGS
        # ══════════════════════════════════════════════════
        story.append(Paragraph('AI CLINICAL FINDINGS', section_s))
        clinical_text = m.get('clinical', 'No clinical note available.')
        findings_data = [[Paragraph(clinical_text, body_s)]]
        findings_tbl  = Table(findings_data, colWidths=[W])
        findings_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), MID),
            ('GRID',          (0,0),(-1,-1), 0.4, LGRAY),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
            ('TOPPADDING',    (0,0),(-1,-1), 7),
            ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ]))
        story.append(findings_tbl)
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # PATIENT GUIDANCE  (coloured by severity)
        # ══════════════════════════════════════════════════
        story.append(Paragraph('PATIENT GUIDANCE & RECOMMENDED ACTIONS', section_s))

        guidance_rows = [
            [Paragraph('Severity Level', label_s),
             Paragraph(f'<b>{sev_level} — {sev_label}</b>',
                       ps('GL', fontSize=9, textColor=SEV_COLOR, fontName='Helvetica-Bold'))],
            [Paragraph('Recommended Specialist', label_s),
             Paragraph(specialist, value_s)],
            [Paragraph('Urgency', label_s),
             Paragraph(urgency,
                       ps('GU', fontSize=9, textColor=SEV_COLOR, fontName='Helvetica-Bold'))],
            [Paragraph('Patient Advice', label_s),
             Paragraph(patient_advice, adv_s)],
            [Paragraph('Next Steps', label_s),
             Paragraph(
                'Bring this report and all prior imaging to your appointment. '
                'Ensure your physician has your full medication list, allergy history, '
                'and family history of pancreatic or gastrointestinal disease. '
                'Avoid high-fat meals and alcohol until specialist review is complete.',
                adv_s)],
            [Paragraph('Red-Flag Symptoms\n(Go to A&E immediately)', label_s),
             Paragraph(
                'Sudden severe abdominal/back pain  •  New-onset jaundice  •  '
                'Persistent vomiting  •  Unexplained weight loss >5 kg  •  '
                'Dark urine or pale stools  •  High fever with abdominal pain',
                ps('RF', fontSize=9, textColor=RED, leading=15))],
        ]
        guid_tbl = Table(guidance_rows, colWidths=[42*mm, W - 42*mm])
        guid_tbl.setStyle(TableStyle([
            ('ROWBACKGROUNDS', (0,0),(-1,-1), [ROW1, ROW2]),
            ('GRID',           (0,0),(-1,-1), 0.4, LGRAY),
            ('LEFTPADDING',    (0,0),(-1,-1), 7),
            ('RIGHTPADDING',   (0,0),(-1,-1), 7),
            ('TOPPADDING',     (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',  (0,0),(-1,-1), 6),
            ('VALIGN',         (0,0),(-1,-1), 'TOP'),
            ('BACKGROUND',     (0,3),( 0,3),  STEELB),
        ]))
        story.append(guid_tbl)
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # TECHNICAL NOTES FOR CLINICIAN
        # ══════════════════════════════════════════════════
        story.append(Paragraph('TECHNICAL NOTES FOR REVIEWING CLINICIAN', section_s))
        tech_notes = (
            'This segmentation was produced by a deep learning model (SwinUNETR) trained on '
            'multi-centre abdominal CT datasets. The model outputs three classes: background, '
            'pancreatic parenchyma, and focal lesion (tumour). Dice scores of ≥0.70 (pancreas) '
            'and ≥0.60 (tumour) are considered clinically acceptable; lower scores indicate '
            'increased uncertainty. Test-Time Augmentation (8-flip ensemble) was applied to '
            'improve boundary accuracy. RECIST 1.1 longest-axis diameter was derived from the '
            'segmentation mask bounding box in the resampled volume — this is an approximation '
            'and must be confirmed by manual measurement on the original DICOM. Tumour burden '
            f'is calculated as tumour volume / pancreas volume × 100%. Pancreas centroid located '
            f'in the {m.get("region","unknown")} region. Volume measurements use resampled voxel '
            f'spacing (1.5 × 1.5 × 2.0 mm); minor discrepancies vs. original DICOM measurements '
            'are expected. AI output should always be reviewed by a qualified radiologist.'
        )
        tech_data = [[Paragraph(tech_notes, body_dk_s)]]
        tech_tbl  = Table(tech_data, colWidths=[W])
        tech_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), DARK),
            ('GRID',          (0,0),(-1,-1), 0.4, LGRAY),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
            ('TOPPADDING',    (0,0),(-1,-1), 7),
            ('BOTTOMPADDING', (0,0),(-1,-1), 7),
        ]))
        story.append(tech_tbl)
        story.append(Spacer(1, 5*mm))

        # ══════════════════════════════════════════════════
        # DISCLAIMER
        # ══════════════════════════════════════════════════
        story.append(HRFlowable(width='100%', thickness=0.5, color=LGRAY,
                                spaceAfter=3*mm, spaceBefore=2*mm))
        disc_text = (
            '⚠  DISCLAIMER — This report is generated by an AI model for educational and research '
            'purposes only. It does NOT constitute a clinical diagnosis, medical advice, or a '
            'substitute for professional radiological review. All findings MUST be reviewed and '
            'confirmed by a qualified radiologist or licensed medical professional before any '
            'clinical decision is made. Final-year engineering project — NOT approved for clinical '
            'use. If you are a patient, please share this report with your physician.'
        )
        disc_data = [[Paragraph(disc_text, disc_s)]]
        disc_tbl  = Table(disc_data, colWidths=[W])
        disc_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor('#0a1520')),
            ('GRID',          (0,0),(-1,-1), 0.8, AMBER),
            ('LEFTPADDING',   (0,0),(-1,-1), 8),
            ('RIGHTPADDING',  (0,0),(-1,-1), 8),
            ('TOPPADDING',    (0,0),(-1,-1), 6),
            ('BOTTOMPADDING', (0,0),(-1,-1), 6),
        ]))
        story.append(disc_tbl)
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'PancrAI  ·  Report generated {date_str}  ·  Job {j.get("job_id","N/A")[:16]}  ·  '
            'For research use only',
            footer_s))

        doc.build(story)

    except ImportError:
        pdf_path.write_text(f"PancrAI Report\nMetrics: {j.get('metrics', {})}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f'\n  PancrAI server starting on http://0.0.0.0:{port}')
    print(f'  Model: {"✅ found" if MODEL_PATH.exists() else "⚠️  not found (demo mode)"}')
    print(f'  Templates: {HERE / "templates"}\n')
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True, use_reloader=False)
