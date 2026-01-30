#!/usr/bin/env python3

import argparse
import ctypes
import ctypes.util
import statistics
import time

import numpy as np

import astra


def _load_cudart() -> ctypes.CDLL:
    libname = ctypes.util.find_library("cudart")
    if not libname:
        # Fallback to the common default path.
        libname = "libcudart.so"
    return ctypes.CDLL(libname)


_cudart = _load_cudart()

cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int

_cudaMalloc = _cudart.cudaMalloc
_cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudaMalloc.restype = cudaError_t

_cudaFree = _cudart.cudaFree
_cudaFree.argtypes = [ctypes.c_void_p]
_cudaFree.restype = cudaError_t

_cudaMemcpy = _cudart.cudaMemcpy
_cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind]
_cudaMemcpy.restype = cudaError_t

_cudaMemset = _cudart.cudaMemset
_cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
_cudaMemset.restype = cudaError_t

_cudaDeviceSynchronize = _cudart.cudaDeviceSynchronize
_cudaDeviceSynchronize.argtypes = []
_cudaDeviceSynchronize.restype = cudaError_t

_cudaGetErrorString = _cudart.cudaGetErrorString
_cudaGetErrorString.argtypes = [cudaError_t]
_cudaGetErrorString.restype = ctypes.c_char_p

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _check(err: int, msg: str) -> None:
    if err != 0:
        s = _cudaGetErrorString(err)
        raise RuntimeError(f"{msg}: cudaError={err} ({s.decode() if s else 'unknown'})")


def cuda_malloc(nbytes: int) -> int:
    p = ctypes.c_void_p()
    _check(int(_cudaMalloc(ctypes.byref(p), nbytes)), "cudaMalloc")
    return int(p.value)


def cuda_free(ptr: int) -> None:
    _check(int(_cudaFree(ctypes.c_void_p(ptr))), "cudaFree")


def cuda_memset(ptr: int, value: int, nbytes: int) -> None:
    _check(int(_cudaMemset(ctypes.c_void_p(ptr), value, nbytes)), "cudaMemset")


def cuda_memcpy_h2d(dst_ptr: int, src: np.ndarray) -> None:
    _check(
        int(
            _cudaMemcpy(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(src.ctypes.data),
                src.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        ),
        "cudaMemcpy H2D",
    )


def cuda_memcpy_d2h(dst: np.ndarray, src_ptr: int) -> None:
    _check(
        int(
            _cudaMemcpy(
                ctypes.c_void_p(dst.ctypes.data),
                ctypes.c_void_p(src_ptr),
                dst.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        ),
        "cudaMemcpy D2H",
    )


def cuda_sync() -> None:
    _check(int(_cudaDeviceSynchronize()), "cudaDeviceSynchronize")


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _make_cone_vec(det_rows: int, det_cols: int, n_angles: int, src_dist: float) -> dict:
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False, dtype=np.float32)
    vectors = np.zeros((n_angles, 12), dtype=np.float32)
    for i, a in enumerate(angles):
        # source
        vectors[i, 0] = np.sin(a) * src_dist
        vectors[i, 1] = -np.cos(a) * src_dist
        vectors[i, 2] = 0.0
        # center of detector
        vectors[i, 3:6] = 0.0
        # detector u direction
        vectors[i, 6] = np.cos(a)
        vectors[i, 7] = np.sin(a)
        vectors[i, 8] = 0.0
        # detector v direction
        vectors[i, 9] = 0.0
        vectors[i, 10] = 0.0
        vectors[i, 11] = 1.0
    return astra.create_proj_geom("cone_vec", det_rows, det_cols, vectors)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--vol", type=int, default=64)
    p.add_argument("--det-rows", type=int, default=32)
    p.add_argument("--det-cols", type=int, default=64)
    p.add_argument("--angles", type=int, default=120)
    p.add_argument("--sirt-iters", type=int, default=5)
    p.add_argument("--relax", type=float, default=1.0)
    p.add_argument("--src-dist", type=float, default=1000.0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--workers", type=int, default=1, help="SIRT3D_CUDA_GPU_BATCH NumWorkers")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    if not astra.use_cuda():
        raise RuntimeError("ASTRA reports CUDA is not available. (Try running with GPU access.)")

    astra.set_gpu_index(args.gpu)
    print("Device:", astra.get_gpu_info(args.gpu))

    B = args.batch
    vol_geom = astra.create_vol_geom(args.vol, args.vol, args.vol)
    proj_geom = _make_cone_vec(args.det_rows, args.det_cols, args.angles, args.src_dist)

    vol_shape = (args.vol, args.vol, args.vol)  # (Z, Y, X)
    sino_shape = (args.det_rows, args.angles, args.det_cols)  # (rows, angles, cols)

    # Allocate GPU buffers (contiguous) and link them via GPULink.
    nbytes_vol = int(np.prod(vol_shape)) * 4
    nbytes_sino = int(np.prod(sino_shape)) * 4

    vol_ptrs: list[int] = []
    sino_ptrs: list[int] = []
    recon_ptrs: list[int] = []

    vol_ids: list[int] = []
    sino_ids: list[int] = []
    recon_ids: list[int] = []

    try:
        for _ in range(B):
            vp = cuda_malloc(nbytes_vol)
            sp = cuda_malloc(nbytes_sino)
            rp = cuda_malloc(nbytes_vol)
            vol_ptrs.append(vp)
            sino_ptrs.append(sp)
            recon_ptrs.append(rp)

            # pitch in bytes for contiguous layout: X * sizeof(float)
            vol_link = astra.data3d.GPULink(vp, args.vol, args.vol, args.vol, args.vol * 4)
            sino_link = astra.data3d.GPULink(sp, args.det_cols, args.angles, args.det_rows, args.det_cols * 4)
            recon_link = astra.data3d.GPULink(rp, args.vol, args.vol, args.vol, args.vol * 4)

            vol_ids.append(astra.data3d.link("-vol", vol_geom, vol_link))
            sino_ids.append(astra.data3d.link("-sino", proj_geom, sino_link))
            recon_ids.append(astra.data3d.link("-vol", vol_geom, recon_link))

        # Fill volumes with random host data, zero recon and sino.
        rng = np.random.default_rng(0)
        vols_h = [rng.standard_normal(vol_shape, dtype=np.float32) for _ in range(B)]
        for i in range(B):
            cuda_memcpy_h2d(vol_ptrs[i], vols_h[i])
            cuda_memset(sino_ptrs[i], 0, nbytes_sino)
            cuda_memset(recon_ptrs[i], 0, nbytes_vol)
        cuda_sync()

        # FP on GPU to generate sinograms.
        fp_cfg = astra.astra_dict("FP3D_CUDA_BATCH")
        fp_cfg["VolumeDataIds"] = vol_ids
        fp_cfg["ProjectionDataIds"] = sino_ids
        fp_id = astra.algorithm.create(fp_cfg)
        astra.algorithm.run(fp_id)
        astra.algorithm.delete(fp_id)

        # --- GPU SIRT batch ---
        sirt_cfg = astra.astra_dict("SIRT3D_CUDA_GPU_BATCH")
        sirt_cfg["ProjectionDataIds"] = sino_ids
        sirt_cfg["ReconstructionDataIds"] = recon_ids
        sirt_cfg["options"] = {"Relaxation": float(args.relax), "NumWorkers": int(args.workers)}
        sirt_id = astra.algorithm.create(sirt_cfg)

        # Warmup
        astra.algorithm.run(sirt_id, 1)
        cuda_sync()
        for i in range(B):
            cuda_memset(recon_ptrs[i], 0, nbytes_vol)
        cuda_sync()

        run_times: list[float] = []
        for _ in range(args.repeats):
            for i in range(B):
                cuda_memset(recon_ptrs[i], 0, nbytes_vol)
            cuda_sync()
            t0 = time.perf_counter()
            astra.algorithm.run(sirt_id, args.sirt_iters)
            cuda_sync()
            run_times.append(time.perf_counter() - t0)

        astra.algorithm.delete(sirt_id)

        print(f"GPU SIRT3D_CUDA_GPU_BATCH: median total={_median(run_times):.6f}s  per-sample={_median(run_times)/B:.6f}s")

        if args.verify:
            # Copy recon back and compare with CPU SIRT3D_CUDA_BATCH on the same sinograms.
            sinos_h = [np.empty(sino_shape, dtype=np.float32) for _ in range(B)]
            recons_gpu_h = [np.empty(vol_shape, dtype=np.float32) for _ in range(B)]
            for i in range(B):
                cuda_memcpy_d2h(sinos_h[i], sino_ptrs[i])
                cuda_memcpy_d2h(recons_gpu_h[i], recon_ptrs[i])

            # Sanity-check FP: compare GPU-linked FP output against CPU-linked FP output.
            sinos_cpu_fp = [np.zeros(sino_shape, dtype=np.float32) for _ in range(B)]
            vol_ids_cpu_fp = [astra.data3d.link("-vol", vol_geom, v) for v in vols_h]
            sino_ids_cpu_fp = [astra.data3d.link("-sino", proj_geom, s) for s in sinos_cpu_fp]
            fp_cfg_cpu = astra.astra_dict("FP3D_CUDA_BATCH")
            fp_cfg_cpu["VolumeDataIds"] = vol_ids_cpu_fp
            fp_cfg_cpu["ProjectionDataIds"] = sino_ids_cpu_fp
            fp_id_cpu = astra.algorithm.create(fp_cfg_cpu)
            astra.algorithm.run(fp_id_cpu)
            astra.algorithm.delete(fp_id_cpu)

            fp_max_abs = 0.0
            fp_max_rel = 0.0
            for a, b in zip(sinos_cpu_fp, sinos_h, strict=True):
                fp_max_abs = max(fp_max_abs, float(np.max(np.abs(a - b))))
                denom = float(np.linalg.norm(a) + 1e-12)
                fp_max_rel = max(fp_max_rel, float(np.linalg.norm(a - b) / denom))
            print(f"[verify] FP max abs diff: {fp_max_abs:.6g}, max rel l2 diff: {fp_max_rel:.6g}")

            astra.data3d.delete(vol_ids_cpu_fp + sino_ids_cpu_fp)

            # CPU SIRT
            recons_cpu_h = [np.zeros(vol_shape, dtype=np.float32) for _ in range(B)]
            sino_ids_cpu = [astra.data3d.link("-sino", proj_geom, s) for s in sinos_h]
            recon_ids_cpu = [astra.data3d.link("-vol", vol_geom, r) for r in recons_cpu_h]

            cpu_cfg = astra.astra_dict("SIRT3D_CUDA_BATCH")
            cpu_cfg["ProjectionDataIds"] = sino_ids_cpu
            cpu_cfg["ReconstructionDataIds"] = recon_ids_cpu
            cpu_cfg["options"] = {"Relaxation": float(args.relax)}
            cpu_id = astra.algorithm.create(cpu_cfg)
            astra.algorithm.run(cpu_id, args.sirt_iters)
            astra.algorithm.delete(cpu_id)

            # Compare
            max_abs = 0.0
            max_rel = 0.0
            for a, b in zip(recons_cpu_h, recons_gpu_h, strict=True):
                max_abs = max(max_abs, float(np.max(np.abs(a - b))))
                denom = float(np.linalg.norm(a) + 1e-12)
                max_rel = max(max_rel, float(np.linalg.norm(a - b) / denom))
            print(f"[verify] max abs diff: {max_abs:.6g}, max rel l2 diff: {max_rel:.6g}")
            print(
                f"[verify] norms: cpu={np.linalg.norm(recons_cpu_h[0]):.6g}  gpu={np.linalg.norm(recons_gpu_h[0]):.6g}"
            )

            astra.data3d.delete(sino_ids_cpu + recon_ids_cpu)

        astra.data3d.delete(vol_ids + sino_ids + recon_ids)
        return 0
    finally:
        for i in range(len(vol_ptrs)):
            cuda_free(vol_ptrs[i])
            cuda_free(sino_ptrs[i])
            cuda_free(recon_ptrs[i])


if __name__ == "__main__":
    raise SystemExit(main())
