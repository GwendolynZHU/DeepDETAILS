import argparse
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average DeepSHAP attribution memmaps across model replicates."
    )
    parser.add_argument("--meta-paths", nargs="+", required=True,
                        help="DeepSHAP meta .npy files from replicate runs.")
    parser.add_argument("--job", choices=("profile", "counts"), required=True,
                        help="Which attribution task to ensemble.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for ensemble .dat and meta outputs.")
    parser.add_argument("--cell-type", required=True,
                        help="Cell type name to validate against meta['ct_name'].")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Number of regions to average per chunk.")
    return parser.parse_args()


def load_meta(path):
    meta = np.load(path, allow_pickle=True)
    if isinstance(meta, np.ndarray) and meta.shape == () and isinstance(meta.item(), dict):
        meta = meta.item()
    if not isinstance(meta, dict):
        raise ValueError(f"Expected dict-like meta in {path}, got {type(meta)}")
    return meta


def normalize_for_compare(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def require(meta, key, path):
    value = meta.get(key)
    if value is None:
        raise ValueError(f"{path} is missing required meta key: {key}")
    return value


def validate_dat_file(path, dtype, shape, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    dtype = np.dtype(dtype)
    expected_size = int(np.prod(shape)) * dtype.itemsize
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise ValueError(
            f"{label} size mismatch for {path}: expected {expected_size} bytes "
            f"for shape={shape} dtype={dtype}, got {actual_size}"
        )


def load_and_validate_metas(meta_paths, job, cell_type):
    metas = [load_meta(path) for path in meta_paths]
    first = metas[0]

    if first.get("ct_name") != cell_type:
        raise ValueError(f"First meta ct_name={first.get('ct_name')} does not match --cell-type={cell_type}")

    consistency_keys = [
        "N", "L", "ct_name", "target_cell_type_idx",
        "peaks_bed", "score_regions",
    ]
    attr_name = first.get("attribution_name") or first.get("attribution_method") or "deepshap"
    for path, meta in zip(meta_paths, metas):
        if meta.get("ct_name") != cell_type:
            raise ValueError(f"{path} ct_name={meta.get('ct_name')} does not match --cell-type={cell_type}")
        meta_attr_name = meta.get("attribution_name") or meta.get("attribution_method") or "deepshap"
        if meta_attr_name != attr_name:
            raise ValueError(
                f"Do not ensemble different attribution methods together: "
                f"first={attr_name} in {meta_paths[0]} vs {meta_attr_name} in {path}"
            )
        for key in consistency_keys:
            if key in first or key in meta:
                if normalize_for_compare(first.get(key)) != normalize_for_compare(meta.get(key)):
                    raise ValueError(
                        f"Meta mismatch for {key}: first={first.get(key)} in {meta_paths[0]} "
                        f"vs {meta.get(key)} in {path}"
                    )

    N = int(require(first, "N", meta_paths[0]))
    L = int(require(first, "L", meta_paths[0]))
    dtype = np.dtype(require(first, "dtype_attr", meta_paths[0]))
    shape = (N, 4, L)

    paths = []
    for meta_path, meta in zip(meta_paths, metas):
        onehot_path = require(meta, "seq_onehot_path", meta_path)
        validate_dat_file(onehot_path, np.uint8, shape, "one-hot")
        dat_path = require(meta, f"{job}_path", meta_path)
        validate_dat_file(dat_path, dtype, shape, f"{job} attribution")
        paths.append(dat_path)

    return metas, paths, dtype, shape


def update_ensemble_meta(meta_path, base_meta, job, out_dat_path, meta_paths):
    if os.path.exists(meta_path):
        out_meta = load_meta(meta_path)
    else:
        out_meta = dict(base_meta)
        out_meta["ensemble_source_meta_paths"] = []
        out_meta["ensemble_jobs"] = []

    attr_name = base_meta.get("attribution_name") or base_meta.get("attribution_method") or "deepshap"
    out_meta.update({
        "N": int(base_meta["N"]),
        "L": int(base_meta["L"]),
        "ct_name": base_meta.get("ct_name"),
        "cell_type_names": base_meta.get("cell_type_names"),
        "target_cell_type_idx": base_meta.get("target_cell_type_idx"),
        "peaks_bed": base_meta.get("peaks_bed"),
        "val_chroms": base_meta.get("val_chroms"),
        "test_chroms": base_meta.get("test_chroms"),
        "ensemble_source_val_chroms": [meta.get("val_chroms") for meta in [load_meta(path) for path in meta_paths]],
        "ensemble_source_test_chroms": [meta.get("test_chroms") for meta in [load_meta(path) for path in meta_paths]],
        "score_regions": base_meta.get("score_regions"),
        "seq_onehot_path": base_meta.get("seq_onehot_path"),
        "dtype_attr": str(np.dtype(np.float32)),
        "attribution_method": base_meta.get("attribution_method", attr_name),
        "attribution_name": attr_name,
        f"{job}_path": out_dat_path,
        "ensemble_method": "mean",
        "ensemble_n_models": len(meta_paths),
    })

    existing_sources = list(out_meta.get("ensemble_source_meta_paths") or [])
    for path in meta_paths:
        if path not in existing_sources:
            existing_sources.append(path)
    out_meta["ensemble_source_meta_paths"] = existing_sources

    jobs = list(out_meta.get("ensemble_jobs") or [])
    if job not in jobs:
        jobs.append(job)
    out_meta["ensemble_jobs"] = jobs

    np.save(meta_path, out_meta, allow_pickle=True)
    return out_meta


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    metas, dat_paths, dtype, shape = load_and_validate_metas(
        args.meta_paths, args.job, args.cell_type
    )
    N, C, L = shape

    attr_name = metas[0].get("attribution_name") or metas[0].get("attribution_method") or "deepshap"

    os.makedirs(args.output_dir, exist_ok=True)
    out_dat_path = os.path.join(args.output_dir, f"{args.cell_type}_{args.job}_{attr_name}_ensemble_mean.dat")
    out_meta_path = os.path.join(args.output_dir, f"{args.cell_type}_{attr_name}_ensemble_meta.npy")

    arrays = [np.memmap(path, dtype=dtype, mode="r", shape=shape) for path in dat_paths]
    out = np.memmap(out_dat_path, dtype=np.float32, mode="w+", shape=shape)

    for start in range(0, N, args.chunk_size):
        end = min(N, start + args.chunk_size)
        acc = np.zeros((end - start, C, L), dtype=np.float64)
        for arr in arrays:
            acc += arr[start:end].astype(np.float64, copy=False)
        out[start:end] = (acc / len(arrays)).astype(np.float32, copy=False)

    out.flush()
    update_ensemble_meta(out_meta_path, metas[0], args.job, out_dat_path, args.meta_paths)
    print(f"Wrote ensemble attribution: {out_dat_path}")
    print(f"Wrote ensemble meta: {out_meta_path}")


if __name__ == "__main__":
    main()
