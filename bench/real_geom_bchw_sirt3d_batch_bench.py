#!/usr/bin/env python3

import argparse
import json
import time
from math import cos, pi, sin
from pathlib import Path

import numpy as np

import astra

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This benchmark requires PyTorch (torch).") from e
try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    raise RuntimeError("This benchmark requires Pillow (PIL).") from e


N_SEGMENTS = 3
DEG_TO_RAD = pi / 180.0


def fov_3d_from_config(cfg: dict, n_slices: int | None) -> tuple[int, int, int, float, float, float, float, float, float]:
    fov = cfg["imaging"]["fov"]
    w = int(fov["width"] * fov.get("pixel_ratio", 1))
    h = int(fov["height"] * fov.get("pixel_ratio", 1))
    px = float(fov["pixel_size"])
    cx = float(fov.get("center_offset_x", 0.0))
    cy = float(fov.get("center_offset_y", 0.0))
    vz = float(fov.get("voxel_z", px))

    z = int(n_slices if n_slices is not None else fov.get("n_slices", 1))

    min_x = cx - (w * px) / 2.0
    max_x = cx + (w * px) / 2.0
    min_y = cy - (h * px) / 2.0
    max_y = cy + (h * px) / 2.0
    min_z = -(z * vz) / 2.0
    max_z = +(z * vz) / 2.0
    return h, w, z, min_x, max_x, min_y, max_y, min_z, max_z


def make_cone_vec_geometry(cfg: dict) -> tuple[dict, int, int, int]:
    geo = cfg["geometry"]
    src = geo["source"]
    det = geo["detector"]

    num_sources = int(src["num_sources"])
    source_interval = float(src["source_interval"])
    active = list(src["active_source_indices"])
    source_to_phantom = float(src["source_to_phantom"])

    detector_size = float(det["detector_size"])
    detector_interval = float(det["detector_interval"])
    source_to_detector = float(det["source_to_detector"])
    detector_angle = float(det["angle"])
    segment_gap = float(det["segment_gap"])

    det_height = float(det.get("detector_height", 0.0))
    det_rows = int(det.get("detector_row_count", 1))
    det_cols = int(det["num_bins"])

    arr_len = source_interval * (num_sources - 1)
    phantom_to_detector = source_to_detector - source_to_phantom

    a = detector_angle * DEG_TO_RAD
    ca = cos(a)
    sa = sin(a)

    det_row_interval = det_height / det_rows if det_rows > 0 else 0.0
    vX, vY, vZ = 0.0, 0.0, float(det_row_interval)

    dX_lr = phantom_to_detector - (detector_size / 2.0) * sa - segment_gap * sa

    vectors = np.empty((len(active) * N_SEGMENTS, 12), dtype=np.float32)
    k = 0
    for i in active:
        srcX = -source_to_phantom
        srcY = source_interval * i - (arr_len / 2.0)
        srcZ = 0.0
        dZ = 0.0

        dY_left = -(detector_size / 2.0) * (1.0 + ca) - segment_gap * ca
        uX_left = detector_interval * sa
        uY_left = detector_interval * ca
        vectors[k] = [srcX, srcY, srcZ, dX_lr, dY_left, dZ, uX_left, uY_left, 0.0, vX, vY, vZ]
        k += 1

        vectors[k] = [
            srcX,
            srcY,
            srcZ,
            phantom_to_detector,
            0.0,
            dZ,
            0.0,
            detector_interval,
            0.0,
            vX,
            vY,
            vZ,
        ]
        k += 1

        dY_right = (detector_size / 2.0) * (1.0 + ca) + segment_gap * ca
        uX_right = -detector_interval * sa
        uY_right = detector_interval * ca
        vectors[k] = [srcX, srcY, srcZ, dX_lr, dY_right, dZ, uX_right, uY_right, 0.0, vX, vY, vZ]
        k += 1

    proj_geom = astra.create_proj_geom("cone_vec", det_rows, det_cols, vectors)
    total_angles = int(vectors.shape[0])
    n_sources_active = total_angles // N_SEGMENTS
    return proj_geom, det_rows, det_cols, n_sources_active


def sino_bchw_to_astra_view(sino_bchw: torch.Tensor, det_cols: int) -> torch.Tensor:
    # BCHW: (B, det_rows, n_sources, det_cols*3) -> (B, det_rows, n_sources*3, det_cols)
    b, det_rows, n_sources, _ = sino_bchw.shape
    return sino_bchw.view(b, det_rows, n_sources, N_SEGMENTS, det_cols).reshape(b, det_rows, n_sources * N_SEGMENTS, det_cols)


def make_gpu_links_batched_3d(t: torch.Tensor, dims_xyz: tuple[int, int, int]) -> list:
    # Link each batch item by pointer offset; requires contiguous BCHW.
    if t.dtype != torch.float32 or (not t.is_cuda) or (not t.is_contiguous()) or t.ndim != 4:
        raise ValueError("expected contiguous CUDA float32 tensor with 4 dims (B, Z, Y, X)")

    b = int(t.shape[0])
    x, y, z = (int(v) for v in dims_xyz)

    base_ptr = int(t.data_ptr())
    stride0_elems = int(t.stride(0))
    elem_size = int(t.element_size())
    pitch_bytes = x * elem_size

    links = []
    for bi in range(b):
        ptr = base_ptr + bi * stride0_elems * elem_size
        links.append(astra.data3d.GPULink(ptr, x, y, z, pitch_bytes))
    return links


def to_uint8(img: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(x, p_low))
    hi = float(np.percentile(x, p_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
    y = (x - lo) / (hi - lo + 1e-12)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def save_png(path: Path, img2d: np.ndarray, p_low: float, p_high: float) -> None:
    Image.fromarray(to_uint8(img2d, p_low=p_low, p_high=p_high), mode="L").save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="/home/suxh/code/pycode/astra/astra_2d/astra_package/ct/resources/config/config_119.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--n-slices", type=int, default=1)

    p.add_argument("--bubble-dir", default="bubble", help="Dir of 512x512 .npy images. Set to '' to use random.")
    p.add_argument("--bubble-offset", type=int, default=0)

    p.add_argument("--sirt-iters", type=int, default=10)
    p.add_argument("--relax", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)

    p.add_argument("--fp-repeats", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--viz-dir", default=None, help="If set, save PNGs here.")
    p.add_argument("--viz-n", type=int, default=4)
    p.add_argument("--viz-p-low", type=float, default=1.0)
    p.add_argument("--viz-p-high", type=float, default=99.0)

    args = p.parse_args()

    cfg = json.load(open(args.config, "r", encoding="utf-8"))

    if not astra.use_cuda():
        raise RuntimeError("ASTRA reports CUDA is not available. (Run with GPU access.)")

    torch.cuda.set_device(args.gpu)
    astra.set_gpu_index(args.gpu)
    print("Device:", astra.get_gpu_info(args.gpu))

    h, w, z, min_x, max_x, min_y, max_y, min_z, max_z = fov_3d_from_config(cfg, args.n_slices)
    vol_geom = astra.create_vol_geom(h, w, z, min_x, max_x, min_y, max_y, min_z, max_z)
    proj_geom, det_rows, det_cols, n_sources = make_cone_vec_geometry(cfg)

    b = int(args.batch)
    print(f"vol[B,Z,H,W]=[{b},{z},{h},{w}]  sino[B,det_rows,n_sources,det_cols*3]=[{b},{det_rows},{n_sources},{det_cols*N_SEGMENTS}]")

    # --- volumes (B, Z, H, W) ---
    if args.bubble_dir:
        bubble_dir = Path(args.bubble_dir)
        files = sorted(bubble_dir.glob("*.npy"))
        if len(files) < args.bubble_offset + b:
            raise RuntimeError(f"Need {b} .npy files from offset {args.bubble_offset}, but only {len(files)} in {bubble_dir}")
        imgs = []
        for fpath in files[args.bubble_offset : args.bubble_offset + b]:
            a = np.load(fpath)
            if a.shape != (h, w):
                raise RuntimeError(f"{fpath} has shape {a.shape}, expected {(h, w)}")
            imgs.append(a.astype(np.float32, copy=False))
        vols_np = np.stack(imgs, axis=0)[:, None, :, :]  # (B,1,H,W)
        if z != 1:
            raise RuntimeError("bubble inputs are 2D; set --n-slices 1")
        vols_bzhw = torch.from_numpy(vols_np).contiguous().to(device="cuda")
    else:
        vols_bzhw = torch.rand((b, z, h, w), device="cuda", dtype=torch.float32)

    # --- sinogram BCHW for DL: (B, det_rows, n_sources, det_cols*3) ---
    sino_bchw = torch.zeros((b, det_rows, n_sources, det_cols * N_SEGMENTS), device="cuda", dtype=torch.float32)
    sino_astra = sino_bchw_to_astra_view(sino_bchw, det_cols)  # (B, det_rows, angles, det_cols)
    recons_bzhw = torch.zeros((b, z, h, w), device="cuda", dtype=torch.float32)

    # --- link tensors as 3D GPULink ---
    vol_links = make_gpu_links_batched_3d(vols_bzhw, (w, h, z))
    recon_links = make_gpu_links_batched_3d(recons_bzhw, (w, h, z))
    sino_links = make_gpu_links_batched_3d(sino_astra, (det_cols, n_sources * N_SEGMENTS, det_rows))

    vol_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in vol_links]
    recon_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in recon_links]
    sino_ids = [astra.data3d.link("-sino", proj_geom, lnk) for lnk in sino_links]

    try:
        # --- FP ---
        fp_cfg = astra.astra_dict("FP3D_CUDA_BATCH")
        fp_cfg["VolumeDataIds"] = vol_ids
        fp_cfg["ProjectionDataIds"] = sino_ids
        fp_id = astra.algorithm.create(fp_cfg)
        astra.algorithm.run(fp_id)  # warmup
        torch.cuda.synchronize()
        fp_times = []
        for _ in range(int(args.fp_repeats)):
            t0 = time.perf_counter()
            astra.algorithm.run(fp_id)
            torch.cuda.synchronize()
            fp_times.append(time.perf_counter() - t0)
        astra.algorithm.delete(fp_id)
        if fp_times:
            fp_med = float(np.median(fp_times))
            print(f"FP3D_CUDA_BATCH: median total={fp_med:.6f}s  per-sample={fp_med/b:.6f}s")

        # --- SIRT ---
        sirt_cfg = astra.astra_dict("SIRT3D_CUDA_GPU_BATCH")
        sirt_cfg["ProjectionDataIds"] = sino_ids
        sirt_cfg["ReconstructionDataIds"] = recon_ids
        sirt_cfg["options"] = {"Relaxation": float(args.relax), "NumWorkers": int(args.workers)}
        sirt_id = astra.algorithm.create(sirt_cfg)

        astra.algorithm.run(sirt_id, 1)  # warmup
        torch.cuda.synchronize()

        run_times = []
        for _ in range(int(args.repeats)):
            recons_bzhw.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            astra.algorithm.run(sirt_id, int(args.sirt_iters))
            torch.cuda.synchronize()
            run_times.append(time.perf_counter() - t0)

        astra.algorithm.delete(sirt_id)

        if run_times:
            med = float(np.median(run_times))
            print(
                f"SIRT3D_CUDA_GPU_BATCH: median total={med:.6f}s  per-sample={med/b:.6f}s  "
                f"per-iter-per-sample={(med/(b*int(args.sirt_iters))):.6f}s"
            )

        # --- visualize (save PNGs) ---
        if args.viz_dir:
            out_dir = Path(args.viz_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            n_save = min(b, int(args.viz_n))

            # z==1 for your training case; keep generic: visualize mid-slice.
            z_idx = z // 2
            vols_2d = vols_bzhw[:, z_idx].detach().cpu().numpy()
            recons_2d = recons_bzhw[:, z_idx].detach().cpu().numpy()

            for bi in range(n_save):
                vol = vols_2d[bi]
                recon = recons_2d[bi]
                diff = np.abs(recon - vol)
                save_png(out_dir / f"vol_b{bi:03d}.png", vol, args.viz_p_low, args.viz_p_high)
                save_png(out_dir / f"recon_b{bi:03d}.png", recon, args.viz_p_low, args.viz_p_high)
                save_png(out_dir / f"diff_b{bi:03d}.png", diff, 0.0, 99.5)
            print(f"Saved {n_save} visualizations to: {out_dir}")

        return 0
    finally:
        astra.data3d.delete(vol_ids + sino_ids + recon_ids)


if __name__ == "__main__":
    raise SystemExit(main())

