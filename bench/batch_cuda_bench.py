#!/usr/bin/env python3

import argparse
import statistics
import subprocess
import time

import numpy as np

import astra


def _nvidia_smi_used_mem_mb(gpu_index: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def _make_cone_vec(det_rows: int, det_cols: int, n_angles: int, src_dist: float) -> dict:
    # Based on ASTRA sample: samples/python/s005_3d_geometry.py
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


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _format_s(secs: float) -> str:
    if secs < 1e-3:
        return f"{secs * 1e6:.2f} us"
    if secs < 1.0:
        return f"{secs * 1e3:.2f} ms"
    return f"{secs:.3f} s"


def _fp_baseline(vol_ids: list[int], sino_ids: list[int]) -> tuple[float, float, float]:
    t_create = 0.0
    t_run = 0.0
    t_delete = 0.0
    for vid, sid in zip(vol_ids, sino_ids, strict=True):
        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["VolumeDataId"] = vid
        cfg["ProjectionDataId"] = sid

        t0 = time.perf_counter()
        alg_id = astra.algorithm.create(cfg)
        t_create += time.perf_counter() - t0

        t0 = time.perf_counter()
        astra.algorithm.run(alg_id)
        t_run += time.perf_counter() - t0

        t0 = time.perf_counter()
        astra.algorithm.delete(alg_id)
        t_delete += time.perf_counter() - t0

    return t_create, t_run, t_delete


def _fp_batch(vol_ids: list[int], sino_ids: list[int]) -> tuple[float, float, float]:
    cfg = astra.astra_dict("FP3D_CUDA_BATCH")
    cfg["VolumeDataIds"] = vol_ids
    cfg["ProjectionDataIds"] = sino_ids

    t0 = time.perf_counter()
    alg_id = astra.algorithm.create(cfg)
    t_create = time.perf_counter() - t0

    t0 = time.perf_counter()
    astra.algorithm.run(alg_id)
    t_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    astra.algorithm.delete(alg_id)
    t_delete = time.perf_counter() - t0

    return t_create, t_run, t_delete


def _fp_baseline_keep_alive(vol_ids: list[int], sino_ids: list[int], repeats: int) -> tuple[float, float, float]:
    alg_ids: list[int] = []
    t0 = time.perf_counter()
    for vid, sid in zip(vol_ids, sino_ids, strict=True):
        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["VolumeDataId"] = vid
        cfg["ProjectionDataId"] = sid
        alg_ids.append(astra.algorithm.create(cfg))
    t_create = time.perf_counter() - t0

    run_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for aid in alg_ids:
            astra.algorithm.run(aid)
        run_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    for aid in alg_ids:
        astra.algorithm.delete(aid)
    t_delete = time.perf_counter() - t0

    return t_create, _median(run_times), t_delete


def _fp_batch_keep_alive(vol_ids: list[int], sino_ids: list[int], repeats: int) -> tuple[float, float, float]:
    cfg = astra.astra_dict("FP3D_CUDA_BATCH")
    cfg["VolumeDataIds"] = vol_ids
    cfg["ProjectionDataIds"] = sino_ids

    t0 = time.perf_counter()
    alg_id = astra.algorithm.create(cfg)
    t_create = time.perf_counter() - t0

    run_times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        astra.algorithm.run(alg_id)
        run_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    astra.algorithm.delete(alg_id)
    t_delete = time.perf_counter() - t0

    return t_create, _median(run_times), t_delete


def _sirt_baseline(sino_ids: list[int], recon_ids: list[int], iters: int, relax: float) -> tuple[float, float, float]:
    t_create = 0.0
    t_run = 0.0
    t_delete = 0.0
    for sid, rid in zip(sino_ids, recon_ids, strict=True):
        cfg = astra.astra_dict("SIRT3D_CUDA")
        cfg["ProjectionDataId"] = sid
        cfg["ReconstructionDataId"] = rid
        cfg["options"] = {"Relaxation": float(relax)}

        t0 = time.perf_counter()
        alg_id = astra.algorithm.create(cfg)
        t_create += time.perf_counter() - t0

        t0 = time.perf_counter()
        astra.algorithm.run(alg_id, iters)
        t_run += time.perf_counter() - t0

        t0 = time.perf_counter()
        astra.algorithm.delete(alg_id)
        t_delete += time.perf_counter() - t0

    return t_create, t_run, t_delete


def _sirt_batch(sino_ids: list[int], recon_ids: list[int], iters: int, relax: float) -> tuple[float, float, float]:
    cfg = astra.astra_dict("SIRT3D_CUDA_BATCH")
    cfg["ProjectionDataIds"] = sino_ids
    cfg["ReconstructionDataIds"] = recon_ids
    cfg["options"] = {"Relaxation": float(relax)}

    t0 = time.perf_counter()
    alg_id = astra.algorithm.create(cfg)
    t_create = time.perf_counter() - t0

    t0 = time.perf_counter()
    astra.algorithm.run(alg_id, iters)
    t_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    astra.algorithm.delete(alg_id)
    t_delete = time.perf_counter() - t0

    return t_create, t_run, t_delete


def _sirt_baseline_keep_alive(
    sino_ids: list[int],
    recon_ids: list[int],
    recon_arrays: list[np.ndarray],
    iters: int,
    relax: float,
    repeats: int,
) -> tuple[float, float, float]:
    alg_ids: list[int] = []
    t0 = time.perf_counter()
    for sid, rid in zip(sino_ids, recon_ids, strict=True):
        cfg = astra.astra_dict("SIRT3D_CUDA")
        cfg["ProjectionDataId"] = sid
        cfg["ReconstructionDataId"] = rid
        cfg["options"] = {"Relaxation": float(relax)}
        alg_ids.append(astra.algorithm.create(cfg))
    t_create = time.perf_counter() - t0

    run_times: list[float] = []
    for _ in range(repeats):
        for r in recon_arrays:
            r.fill(0.0)
        t0 = time.perf_counter()
        for aid in alg_ids:
            astra.algorithm.run(aid, iters)
        run_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    for aid in alg_ids:
        astra.algorithm.delete(aid)
    t_delete = time.perf_counter() - t0

    return t_create, _median(run_times), t_delete


def _sirt_batch_keep_alive(
    sino_ids: list[int],
    recon_ids: list[int],
    recon_arrays: list[np.ndarray],
    iters: int,
    relax: float,
    repeats: int,
) -> tuple[float, float, float]:
    cfg = astra.astra_dict("SIRT3D_CUDA_BATCH")
    cfg["ProjectionDataIds"] = sino_ids
    cfg["ReconstructionDataIds"] = recon_ids
    cfg["options"] = {"Relaxation": float(relax)}

    t0 = time.perf_counter()
    alg_id = astra.algorithm.create(cfg)
    t_create = time.perf_counter() - t0

    run_times: list[float] = []
    for _ in range(repeats):
        for r in recon_arrays:
            r.fill(0.0)
        t0 = time.perf_counter()
        astra.algorithm.run(alg_id, iters)
        run_times.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    astra.algorithm.delete(alg_id)
    t_delete = time.perf_counter() - t0

    return t_create, _median(run_times), t_delete


def _print_result(name: str, t_create: float, t_run: float, t_delete: float, bsz: int) -> None:
    total = t_create + t_run + t_delete
    per = total / bsz if bsz else float("nan")
    print(
        f"{name:>22}  create={_format_s(t_create):>10}  run={_format_s(t_run):>10}  "
        f"delete={_format_s(t_delete):>10}  total={_format_s(total):>10}  per-sample={_format_s(per):>10}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--vol", type=int, default=128, help="Cubic volume size (X=Y=Z).")
    p.add_argument("--det-rows", type=int, default=64)
    p.add_argument("--det-cols", type=int, default=128)
    p.add_argument("--angles", type=int, default=180)
    p.add_argument("--sirt-iters", type=int, default=10)
    p.add_argument("--relax", type=float, default=1.0)
    p.add_argument("--src-dist", type=float, default=1000.0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    if not astra.use_cuda():
        raise RuntimeError("ASTRA reports CUDA is not available. (Try running outside sandbox / with GPU access.)")

    astra.set_gpu_index(args.gpu)
    mem0 = _nvidia_smi_used_mem_mb(args.gpu)

    vol_geom = astra.create_vol_geom(args.vol, args.vol, args.vol)
    proj_geom = _make_cone_vec(args.det_rows, args.det_cols, args.angles, args.src_dist)

    vol_shape = (args.vol, args.vol, args.vol)  # (Z, Y, X)
    sino_shape = (args.det_rows, args.angles, args.det_cols)  # (rows, angles, cols)

    rng = np.random.default_rng(0)

    print("Device:", astra.get_gpu_info(args.gpu))
    print(
        f"Geometry: vol={vol_shape}  sino={sino_shape}  iters={args.sirt_iters}  "
        f"relax={args.relax}  src_dist={args.src_dist}"
    )
    if mem0 is not None:
        print(f"nvidia-smi memory.used (start): {mem0} MB")
    print()

    # --- correctness smoke-test on a tiny batch (optional) ---
    if args.verify:
        bsz = min(2, max(args.batch))
        vols = [rng.standard_normal(vol_shape, dtype=np.float32) for _ in range(bsz)]
        sinos_a = [np.zeros(sino_shape, dtype=np.float32) for _ in range(bsz)]
        sinos_b = [np.zeros(sino_shape, dtype=np.float32) for _ in range(bsz)]

        vol_ids = [astra.data3d.link("-vol", vol_geom, v) for v in vols]
        sino_ids_a = [astra.data3d.link("-sino", proj_geom, s) for s in sinos_a]
        sino_ids_b = [astra.data3d.link("-sino", proj_geom, s) for s in sinos_b]

        _fp_baseline(vol_ids, sino_ids_a)
        _fp_batch(vol_ids, sino_ids_b)

        fp_max = max(float(np.max(np.abs(a - b))) for a, b in zip(sinos_a, sinos_b, strict=True))
        fp_rel = max(
            float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
            for a, b in zip(sinos_a, sinos_b, strict=True)
        )
        print(f"[verify] FP max abs diff: {fp_max:.6g}, max rel l2 diff: {fp_rel:.6g}")

        # SIRT verify: use same sinos, start from zeros
        recons_a = [np.zeros(vol_shape, dtype=np.float32) for _ in range(bsz)]
        recons_b = [np.zeros(vol_shape, dtype=np.float32) for _ in range(bsz)]
        recon_ids_a = [astra.data3d.link("-vol", vol_geom, r) for r in recons_a]
        recon_ids_b = [astra.data3d.link("-vol", vol_geom, r) for r in recons_b]

        _sirt_baseline(sino_ids_a, recon_ids_a, args.sirt_iters, args.relax)
        _sirt_batch(sino_ids_a, recon_ids_b, args.sirt_iters, args.relax)

        sirt_max = max(float(np.max(np.abs(a - b))) for a, b in zip(recons_a, recons_b, strict=True))
        sirt_rel = max(
            float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
            for a, b in zip(recons_a, recons_b, strict=True)
        )
        print(f"[verify] SIRT max abs diff: {sirt_max:.6g}, max rel l2 diff: {sirt_rel:.6g}")

        astra.data3d.delete(vol_ids + sino_ids_a + sino_ids_b + recon_ids_a + recon_ids_b)
        print()

    # --- benchmarks ---
    for bsz in args.batch:
        print(f"== Batch size B={bsz} ==")

        vols = [rng.standard_normal(vol_shape, dtype=np.float32) for _ in range(bsz)]
        sinos0 = [np.zeros(sino_shape, dtype=np.float32) for _ in range(bsz)]
        sinos1 = [np.zeros(sino_shape, dtype=np.float32) for _ in range(bsz)]
        recons0 = [np.zeros(vol_shape, dtype=np.float32) for _ in range(bsz)]
        recons1 = [np.zeros(vol_shape, dtype=np.float32) for _ in range(bsz)]

        vol_ids = [astra.data3d.link("-vol", vol_geom, v) for v in vols]
        sino_ids0 = [astra.data3d.link("-sino", proj_geom, s) for s in sinos0]
        sino_ids1 = [astra.data3d.link("-sino", proj_geom, s) for s in sinos1]
        recon_ids0 = [astra.data3d.link("-vol", vol_geom, r) for r in recons0]
        recon_ids1 = [astra.data3d.link("-vol", vol_geom, r) for r in recons1]

        # Warmups to stabilize GPU clocks + JIT-ish overheads
        for _ in range(args.warmup):
            _fp_baseline(vol_ids[:1], sino_ids0[:1])
            _fp_batch(vol_ids[:1], sino_ids1[:1])
            _sirt_baseline(sino_ids0[:1], recon_ids0[:1], 1, args.relax)
            _sirt_batch(sino_ids0[:1], recon_ids1[:1], 1, args.relax)

        # FP benchmark
        fp_base = []
        fp_batch = []
        for _ in range(args.repeats):
            fp_base.append(_fp_baseline(vol_ids, sino_ids0))
            fp_batch.append(_fp_batch(vol_ids, sino_ids1))

        tcb, trb, tdb = (_median([x[i] for x in fp_base]) for i in range(3))
        tcc, trc, tdc = (_median([x[i] for x in fp_batch]) for i in range(3))

        print("-- FP3D_CUDA --")
        _print_result("baseline (loop)", tcb, trb, tdb, bsz)
        _print_result("batch (C++)", tcc, trc, tdc, bsz)
        speedup = (tcb + trb + tdb) / (tcc + trc + tdc)
        print(f"{'speedup':>22}  x{speedup:.2f}")

        # FP keep-alive benchmark: amortize create/delete across repeats
        tcb, trb, tdb = _fp_baseline_keep_alive(vol_ids, sino_ids0, args.repeats)
        tcc, trc, tdc = _fp_batch_keep_alive(vol_ids, sino_ids1, args.repeats)
        print("-- FP3D_CUDA (keep-alive) --")
        _print_result("baseline (loop)", tcb, trb, tdb, bsz)
        _print_result("batch (C++)", tcc, trc, tdc, bsz)
        speedup = (tcb + trb + tdb) / (tcc + trc + tdc)
        print(f"{'speedup':>22}  x{speedup:.2f}")

        # SIRT benchmark (use the sinograms from FP baseline to avoid random differences)
        sirt_base = []
        sirt_batch = []
        for _ in range(args.repeats):
            for r in recons0:
                r.fill(0.0)
            for r in recons1:
                r.fill(0.0)
            sirt_base.append(_sirt_baseline(sino_ids0, recon_ids0, args.sirt_iters, args.relax))
            sirt_batch.append(_sirt_batch(sino_ids0, recon_ids1, args.sirt_iters, args.relax))

        tcb, trb, tdb = (_median([x[i] for x in sirt_base]) for i in range(3))
        tcc, trc, tdc = (_median([x[i] for x in sirt_batch]) for i in range(3))

        print("-- SIRT3D_CUDA --")
        _print_result("baseline (loop)", tcb, trb, tdb, bsz)
        _print_result("batch (C++)", tcc, trc, tdc, bsz)
        speedup = (tcb + trb + tdb) / (tcc + trc + tdc)
        print(f"{'speedup':>22}  x{speedup:.2f}")

        # SIRT keep-alive benchmark: amortize init across repeats (still avoids Python loop overhead in batch)
        tcb, trb, tdb = _sirt_baseline_keep_alive(
            sino_ids0, recon_ids0, recons0, args.sirt_iters, args.relax, args.repeats
        )
        tcc, trc, tdc = _sirt_batch_keep_alive(
            sino_ids0, recon_ids1, recons1, args.sirt_iters, args.relax, args.repeats
        )
        print("-- SIRT3D_CUDA (keep-alive) --")
        _print_result("baseline (loop)", tcb, trb, tdb, bsz)
        _print_result("batch (C++)", tcc, trc, tdc, bsz)
        speedup = (tcb + trb + tdb) / (tcc + trc + tdc)
        print(f"{'speedup':>22}  x{speedup:.2f}")

        astra.data3d.delete(vol_ids + sino_ids0 + sino_ids1 + recon_ids0 + recon_ids1)

        mem1 = _nvidia_smi_used_mem_mb(args.gpu)
        if mem1 is not None:
            print(f"nvidia-smi memory.used (end): {mem1} MB")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
