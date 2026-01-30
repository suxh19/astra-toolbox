#!/usr/bin/env python3

import argparse
import json
import statistics
import time
from math import cos, pi, sin
from typing import Any, Dict, List, Tuple

import numpy as np

import astra

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This benchmark requires PyTorch (torch).") from e


N_SEGMENTS = 3
DEG_TO_RAD = pi / 180.0


def _median(xs: List[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def calculate_fov_bounds(
    width_pixels: int,
    height_pixels: int,
    pixel_size: float,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> Tuple[float, float, float, float]:
    fov_width_mm = width_pixels * pixel_size
    fov_height_mm = height_pixels * pixel_size
    min_x = center_offset_x - fov_width_mm / 2
    max_x = center_offset_x + fov_width_mm / 2
    min_y = center_offset_y - fov_height_mm / 2
    max_y = center_offset_y + fov_height_mm / 2
    return min_x, max_x, min_y, max_y


def calculate_fov_bounds_3d(
    width_pixels: int,
    height_pixels: int,
    n_slices: int,
    pixel_size: float,
    voxel_z: float,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    min_x, max_x, min_y, max_y = calculate_fov_bounds(
        width_pixels, height_pixels, pixel_size, center_offset_x, center_offset_y
    )
    vol_z_size = n_slices * voxel_z
    min_z = -vol_z_size / 2
    max_z = vol_z_size / 2
    return min_x, max_x, min_y, max_y, min_z, max_z


def get_fov_3d(
    config: Dict[str, Any], n_slices: int | None = None
) -> Tuple[int, int, int, float, float, float, float, float, float]:
    fov = config["imaging"]["fov"]

    width_pixels = int(fov["width"] * fov["pixel_ratio"])
    height_pixels = int(fov["height"] * fov["pixel_ratio"])
    pixel_size = float(fov["pixel_size"])
    center_offset_x = float(fov.get("center_offset_x", 0))
    center_offset_y = float(fov.get("center_offset_y", 0))
    voxel_z = float(fov.get("voxel_z", pixel_size))

    if n_slices is None:
        n_slices = int(fov.get("n_slices", 1))
    else:
        n_slices = int(n_slices)

    bounds = calculate_fov_bounds_3d(
        width_pixels,
        height_pixels,
        n_slices,
        pixel_size,
        voxel_z,
        center_offset_x,
        center_offset_y,
    )
    return (
        height_pixels,
        width_pixels,
        n_slices,
        bounds[0],
        bounds[1],
        bounds[2],
        bounds[3],
        bounds[4],
        bounds[5],
    )


class ProjectionVectorGenerator:
    """
    Copy of /home/suxh/code/pycode/astra/astra_2d/astra_package/ct/geometry/vectors.py
    (3D cone_vec generator).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._parse_config()

    def _parse_config(self) -> None:
        geo = self.config["geometry"]
        source = geo["source"]
        detector = geo["detector"]

        self.num_sources = int(source["num_sources"])
        self.source_interval = float(source["source_interval"])
        self.active_source_indices: List[int] = list(source["active_source_indices"])
        self.source_to_phantom = float(source["source_to_phantom"])

        self.detector_size = float(detector["detector_size"])
        self.detector_interval = float(detector["detector_interval"])
        self.source_to_detector = float(detector["source_to_detector"])
        self.detector_angle = float(detector["angle"])
        self.segment_gap = float(detector["segment_gap"])

        self.detector_height = float(detector.get("detector_height", 0))
        self.detector_row_count = int(detector.get("detector_row_count", 1))

        self.arr_len = self.source_interval * (self.num_sources - 1)
        self.phantom_to_detector = self.source_to_detector - self.source_to_phantom

    def generate_3d(self) -> np.ndarray:
        vectors: list[list[float]] = []

        angle = self.detector_angle * DEG_TO_RAD
        gap = self.segment_gap

        cos_angle = cos(angle)
        sin_angle = sin(angle)

        det_row_interval = self.detector_height / self.detector_row_count
        vX, vY, vZ = 0.0, 0.0, float(det_row_interval)

        dX_lr = (
            self.phantom_to_detector
            - (self.detector_size / 2) * sin_angle
            - gap * sin_angle
        )

        for i in self.active_source_indices:
            srcX = -self.source_to_phantom
            srcY = self.source_interval * i - (self.arr_len / 2)
            srcZ = 0.0
            dZ = 0.0

            # left segment
            dY_left = -(self.detector_size / 2) * (1 + cos_angle) - gap * cos_angle
            uX_left = self.detector_interval * sin_angle
            uY_left = self.detector_interval * cos_angle
            vectors.append(
                [
                    srcX,
                    srcY,
                    srcZ,
                    dX_lr,
                    dY_left,
                    dZ,
                    uX_left,
                    uY_left,
                    0.0,
                    vX,
                    vY,
                    vZ,
                ]
            )

            # middle segment
            vectors.append(
                [
                    srcX,
                    srcY,
                    srcZ,
                    self.phantom_to_detector,
                    0.0,
                    dZ,
                    0.0,
                    self.detector_interval,
                    0.0,
                    vX,
                    vY,
                    vZ,
                ]
            )

            # right segment
            dY_right = (self.detector_size / 2) * (1 + cos_angle) + gap * cos_angle
            uX_right = -self.detector_interval * sin_angle
            uY_right = self.detector_interval * cos_angle
            vectors.append(
                [
                    srcX,
                    srcY,
                    srcZ,
                    dX_lr,
                    dY_right,
                    dZ,
                    uX_right,
                    uY_right,
                    0.0,
                    vX,
                    vY,
                    vZ,
                ]
            )

        return np.asarray(vectors, dtype=np.float32)


def _sino_bchw_to_astra_view(sino_bchw: torch.Tensor, det_cols: int) -> torch.Tensor:
    # BCHW where C=det_rows, H=n_sources, W=det_cols*N_SEGMENTS
    if sino_bchw.ndim != 4:
        raise ValueError(f"sino must be BCHW (4D), got shape={tuple(sino_bchw.shape)}")
    if sino_bchw.shape[-1] != det_cols * N_SEGMENTS:
        raise ValueError(
            f"sino last dim must be det_cols*{N_SEGMENTS}={det_cols*N_SEGMENTS}, got {sino_bchw.shape[-1]}"
        )
    B, det_rows, n_sources, _ = sino_bchw.shape
    return sino_bchw.view(B, det_rows, n_sources, N_SEGMENTS, det_cols).reshape(
        B, det_rows, n_sources * N_SEGMENTS, det_cols
    )


def _make_gpu_links_for_batched_3d(
    tensor_bxyz: torch.Tensor,
    dims_xyz: Tuple[int, int, int],
) -> List[astra.data3d.GPULink]:
    if tensor_bxyz.dtype != torch.float32:
        raise ValueError(f"expected float32 tensor, got {tensor_bxyz.dtype}")
    if not tensor_bxyz.is_cuda:
        raise ValueError("expected CUDA tensor")
    if not tensor_bxyz.is_contiguous():
        raise ValueError("expected contiguous tensor")
    if tensor_bxyz.ndim != 4:
        raise ValueError(f"expected 4D batched tensor, got shape={tuple(tensor_bxyz.shape)}")

    B = int(tensor_bxyz.shape[0])
    x, y, z = (int(d) for d in dims_xyz)

    # We use the slice-by-offset approach to keep a single underlying allocation.
    base_ptr = int(tensor_bxyz.data_ptr())
    stride0_elems = int(tensor_bxyz.stride(0))
    elem_size = int(tensor_bxyz.element_size())
    pitch_bytes = x * elem_size

    links: list[astra.data3d.GPULink] = []
    for b in range(B):
        ptr = base_ptr + b * stride0_elems * elem_size
        links.append(astra.data3d.GPULink(ptr, x, y, z, pitch_bytes))
    return links


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default="/home/suxh/code/pycode/astra/astra_2d/astra_package/ct/resources/config/config_119.json",
        help="Path to astra_2d JSON config (3D cone_vec geometry).",
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--n-slices", type=int, default=None)
    p.add_argument("--sirt-iters", type=int, default=10)
    p.add_argument("--relax", type=float, default=1.0)
    p.add_argument("--fp-repeats", type=int, default=10)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--workers", type=int, default=4, help="SIRT3D_CUDA_GPU_BATCH NumWorkers")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not astra.use_cuda():
        raise RuntimeError("ASTRA reports CUDA is not available. (Try running with GPU access.)")

    torch.cuda.set_device(args.gpu)
    astra.set_gpu_index(args.gpu)
    print("Device:", astra.get_gpu_info(args.gpu))

    # --- Geometry (match astra_2d) ---
    height, width, n_slices, min_x, max_x, min_y, max_y, min_z, max_z = get_fov_3d(
        config, n_slices=args.n_slices
    )
    vol_geom = astra.create_vol_geom(
        height, width, n_slices, min_x, max_x, min_y, max_y, min_z, max_z
    )

    generator = ProjectionVectorGenerator(config)
    proj_vectors = generator.generate_3d()
    det_cols = int(config["geometry"]["detector"]["num_bins"])
    det_rows = int(config["geometry"]["detector"]["detector_row_count"])
    proj_geom = astra.create_proj_geom("cone_vec", det_rows, det_cols, proj_vectors)

    # Total angles in ASTRA layout includes the 3 detector segments per source.
    total_angles = int(proj_vectors.shape[0])
    if total_angles % N_SEGMENTS != 0:
        raise RuntimeError(f"expected vectors multiple of {N_SEGMENTS}, got {total_angles}")
    n_sources = total_angles // N_SEGMENTS

    print(
        f"Shapes: vol[B,C,H,W]=[{args.batch},{n_slices},{height},{width}]  "
        f"sino[B,C,H,W]=[{args.batch},{det_rows},{n_sources},{det_cols*N_SEGMENTS}]"
    )

    # --- Allocate batched tensors in BCHW ---
    B = int(args.batch)
    vols_bchw = torch.randn((B, n_slices, height, width), device="cuda", dtype=torch.float32)

    # Projections are "C=det_rows, H=n_sources, W=det_cols*3" for deep learning (BCHW).
    sino_bchw = torch.empty((B, det_rows, n_sources, det_cols * N_SEGMENTS), device="cuda", dtype=torch.float32)
    sino_bchw.zero_()

    # Internal ASTRA view: (B, det_rows, total_angles, det_cols)
    sino_astra = _sino_bchw_to_astra_view(sino_bchw, det_cols)
    assert sino_astra.is_contiguous()

    recons_bchw = torch.zeros((B, n_slices, height, width), device="cuda", dtype=torch.float32)

    # --- Link batched tensors via GPULink (zero-copy) ---
    # ASTRA expects 3D arrays, so we link one 3D GPULink per batch item.
    vol_links = _make_gpu_links_for_batched_3d(vols_bchw, (width, height, n_slices))
    recon_links = _make_gpu_links_for_batched_3d(recons_bchw, (width, height, n_slices))
    sino_links = _make_gpu_links_for_batched_3d(sino_astra, (det_cols, total_angles, det_rows))

    vol_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in vol_links]
    recon_ids = [astra.data3d.link("-vol", vol_geom, lnk) for lnk in recon_links]
    sino_ids = [astra.data3d.link("-sino", proj_geom, lnk) for lnk in sino_links]

    try:
        # --- FP (GPU batch) ---
        fp_cfg = astra.astra_dict("FP3D_CUDA_BATCH")
        fp_cfg["VolumeDataIds"] = vol_ids
        fp_cfg["ProjectionDataIds"] = sino_ids
        fp_id = astra.algorithm.create(fp_cfg)
        # Warmup + timing for FP
        astra.algorithm.run(fp_id)
        torch.cuda.synchronize()
        fp_times: list[float] = []
        for _ in range(int(args.fp_repeats)):
            t0 = time.perf_counter()
            astra.algorithm.run(fp_id)
            torch.cuda.synchronize()
            fp_times.append(time.perf_counter() - t0)
        astra.algorithm.delete(fp_id)
        fp_med = _median(fp_times)
        print(
            f"GPU FP3D_CUDA_BATCH: median total={fp_med:.6f}s  per-sample={fp_med/B:.6f}s"
        )

        # --- SIRT (GPU batch, GPU-linked, multi-worker) ---
        sirt_cfg = astra.astra_dict("SIRT3D_CUDA_GPU_BATCH")
        sirt_cfg["ProjectionDataIds"] = sino_ids
        sirt_cfg["ReconstructionDataIds"] = recon_ids
        sirt_cfg["options"] = {"Relaxation": float(args.relax), "NumWorkers": int(args.workers)}
        sirt_id = astra.algorithm.create(sirt_cfg)

        # Warmup
        astra.algorithm.run(sirt_id, 1)
        torch.cuda.synchronize()
        recons_bchw.zero_()
        torch.cuda.synchronize()

        run_times: list[float] = []
        for _ in range(int(args.repeats)):
            recons_bchw.zero_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            astra.algorithm.run(sirt_id, int(args.sirt_iters))
            torch.cuda.synchronize()
            run_times.append(time.perf_counter() - t0)

        astra.algorithm.delete(sirt_id)

        med = _median(run_times)
        print(
            f"GPU SIRT3D_CUDA_GPU_BATCH: median total={med:.6f}s  per-sample={med/B:.6f}s  "
            f"per-iter-per-sample={(med/(B*int(args.sirt_iters))):.6f}s"
        )

        if args.verify:
            # CPU batch reference (copies sino to host, runs SIRT3D_CUDA_BATCH, compares recons).
            sino_h = sino_astra.detach().cpu().numpy()
            recons_gpu_h = recons_bchw.detach().cpu().numpy()

            # Run CPU batch SIRT on host arrays.
            sino_ids_cpu = [astra.data3d.link("-sino", proj_geom, sino_h[b]) for b in range(B)]
            recons_cpu = np.zeros((B, n_slices, height, width), dtype=np.float32)
            recon_ids_cpu = [astra.data3d.link("-vol", vol_geom, recons_cpu[b]) for b in range(B)]
            try:
                cpu_cfg = astra.astra_dict("SIRT3D_CUDA_BATCH")
                cpu_cfg["ProjectionDataIds"] = sino_ids_cpu
                cpu_cfg["ReconstructionDataIds"] = recon_ids_cpu
                cpu_cfg["options"] = {"Relaxation": float(args.relax)}
                cpu_id = astra.algorithm.create(cpu_cfg)
                astra.algorithm.run(cpu_id, int(args.sirt_iters))
                astra.algorithm.delete(cpu_id)

                max_abs = float(np.max(np.abs(recons_cpu - recons_gpu_h)))
                denom = float(np.linalg.norm(recons_cpu) + 1e-12)
                rel = float(np.linalg.norm(recons_cpu - recons_gpu_h) / denom)
                print(f"[verify] max abs diff: {max_abs:.6g}, rel l2 diff: {rel:.6g}")
            finally:
                astra.data3d.delete(sino_ids_cpu + recon_ids_cpu)

        return 0
    finally:
        astra.data3d.delete(vol_ids + sino_ids + recon_ids)


if __name__ == "__main__":
    raise SystemExit(main())
