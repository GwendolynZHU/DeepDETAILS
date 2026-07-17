import os
import sys
import numpy as np
import random
import h5py
import argparse
import time
import torch

import modiscolite
import modiscolite.tfmodisco
import modiscolite.util
import functools
print = functools.partial(print, flush=True)


"""
Run TF-MoDISco on DeepSHAP attribution outputs.

The current CLI uses modisco_meta: it reads the DeepSHAP meta .npy, opens
the matching one-hot and attribution memmaps, optionally applies a saved
subset of region indices, center-crops to slice_len, converts arrays to
(N, L, 4), and passes them to modiscolite. This supports the workflow of
attributing all train+val regions once, then running motif discovery on
specific region subsets afterward.
"""

sys.path.append('/home/yz2676/DetailedProCapNet/src/ProCapNet')
from utils.misc import ensure_parent_dir_exists
from data_loading import extract_sequences, extract_observed_profiles


random.seed(0)
np.random.seed(0)
  
def set_modisco_parser(parser):
    # parser.add_argument("--genome_path", 
    #                     help="Path to genome FASTA file.",
    #                     type=str, 
    #                     required=True,
    #                     )
    
    # parser.add_argument("--chrom_sizes",
    #                     help="Path to chromosome sizes file.",
    #                     type=str,
    #                     required=True,
    #                     )
    
    # parser.add_argument("--peak_path",
    #                     help="Path to peak bed file.",
    #                     type=str,
    #                     required=True,
    #                     )
    
    parser.add_argument("--onehot_path",
                        help="Optional path to one-hot sequence .dat file. Defaults to seq_onehot_path in meta.",
                        type=str,
                        default=None,
                        )
    
    parser.add_argument("--meta_path",
                        help="Path to deepshap meta .npy (profile, count paths)",
                        type=str,
                        required=True,
                        )
    
    parser.add_argument("--modisco_results_path",
                        help="Path to save modisco results",
                        type=str,
                        required=True,
                        )
    
    parser.add_argument("--verbose",
                        help="Print verbose output",
                        action="store_true",
                        )
    
    parser.add_argument("--max_seqlets",
                        help="Maximum number of seqlets to use",
                        type=int,
                        default=1000000,
                        )
    
    parser.add_argument("--job",
                        help="Which job to run modisco on (e.g. 'profile' or 'counts')",
                        type=str,
                        required=True,
                        )

    parser.add_argument("--sliding_window_size",
                        help="sliding_window_size for modisco (default 20)",
                        type=int,
                        default=20,
                        )
    
    parser.add_argument("--flank_size",
                        help="flank_size for modisco (default 5)",
                        type=int,
                        default=5,
                        )
    
    parser.add_argument("--trim_to_window_size",
                        help="trim_to_window_size for modisco (default 24)",
                        type=int,
                        default=24,
                        )
    
    parser.add_argument("--n_leiden_runs",
                        help="Number of leiden runs for modisco (default 50)",
                        type=int,
                        default=50,
                        )
    
    parser.add_argument("--target_seqlet_fdr",
                        help="target_seqlet_fdr for modisco (default 0.2)",
                        type=float,
                        default=0.2,
                        )
    
    parser.add_argument("--subset_idx_path",
                    help="Optional .npy file of sequence indices to keep",
                    type=str,
                    default=None)
    
def _load_meta(meta_path):
    """Load dict metadata saved by np.save in deepshap.py."""

    meta = np.load(meta_path, allow_pickle=True)
    # meta might be dict saved via np.save(...)
    if isinstance(meta, np.ndarray) and meta.shape == () and isinstance(meta.item(), dict):
        meta = meta.item()
    return meta


def _require_meta_value(meta, key):
    value = meta.get(key)
    if value is None:
        raise ValueError(f"DeepSHAP meta is missing required key: {key}")
    return value


def _validate_dat_file(path, dtype, shape, label):
    if path is None:
        raise ValueError(f"{label} path is None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

    dtype = np.dtype(dtype)
    expected_size = int(np.prod(shape)) * dtype.itemsize
    actual_size = os.path.getsize(path)
    if actual_size != expected_size:
        raise ValueError(
            f"{label} file size mismatch for {path}: expected {expected_size} bytes "
            f"for shape={shape} dtype={dtype}, got {actual_size} bytes"
        )


def _resolve_onehot_path(meta, onehot_path):
    meta_onehot_path = meta.get("seq_onehot_path")
    if onehot_path is None:
        if meta_onehot_path is None:
            raise ValueError("--onehot_path is required because meta has no seq_onehot_path")
        return meta_onehot_path

    if meta_onehot_path is not None and os.path.abspath(onehot_path) != os.path.abspath(meta_onehot_path):
        print(
            "WARNING: overriding meta seq_onehot_path with --onehot_path. "
            f"meta={meta_onehot_path} override={onehot_path}"
        )
    return onehot_path


def _validate_onehot_array(onehot_seqs):
    if not np.isfinite(onehot_seqs).all():
        raise ValueError("one-hot array contains NaN or Inf")
    if not np.all((onehot_seqs == 0) | (onehot_seqs == 1)):
        raise ValueError(
            f"one-hot array has non-binary values: min={onehot_seqs.min()} max={onehot_seqs.max()}"
        )
    row_sums = onehot_seqs.sum(axis=-1)
    valid = (row_sums == 1) | (row_sums == 0)
    if not np.all(valid):
        raise ValueError(
            "one-hot columns must sum to 1 or 0 for Ns; "
            f"row_sums min={row_sums.min()} max={row_sums.max()}"
        )
    return row_sums


def _validate_subset_indices(subset_idx_path, idx, meta):
    if idx.ndim != 1:
        raise ValueError(f"subset indices must be 1D, got shape {idx.shape}")
    if len(idx) == 0:
        raise ValueError("subset index list is empty")
    if idx.min() < 0:
        raise ValueError(f"subset indices include negative value: {idx.min()}")
    if idx.max() >= meta.get("N"):
        raise ValueError(f"subset index out of range: max={idx.max()} N={meta.get('N')}")

    sidecar = subset_idx_path + ".meta.npy"
    if os.path.exists(sidecar):
        subset_meta = _load_meta(sidecar)
        for key in ("ct_name", "N", "peaks_bed", "val_chroms", "test_chroms"):
            if key in subset_meta and key in meta and subset_meta.get(key) != meta.get(key):
                raise ValueError(
                    f"subset index meta mismatch for {key}: "
                    f"subset={subset_meta.get(key)} deepshap={meta.get(key)}"
                )
    else:
        print(f"No subset sidecar meta found for {subset_idx_path}; checked bounds only.")


def _memmap_dat(meta, key_prefix):
    """Open a DeepSHAP .dat score memmap described by metadata.

    key_prefix examples:
        - "profile", "counts"
        - "profile_onehot", "counts_onehot"
    """
    N = int(_require_meta_value(meta, "N"))
    L = int(_require_meta_value(meta, "L"))
    dtype = _require_meta_value(meta, "dtype_attr")
    dat_path = _require_meta_value(meta, f"{key_prefix}_path")
    shape = (N, 4, L)
    _validate_dat_file(dat_path, dtype, shape, f"{key_prefix} scores")

    arr = np.memmap(dat_path, dtype=dtype, mode="r", shape=shape)
    return arr

def load_sequences(genome_path, chrom_sizes, peak_path, slice_len, in_window=2114, verbose=False):
    onehot_seqs = extract_sequences(genome_path, chrom_sizes, peak_path, in_window, verbose=verbose)
    
    in_width = in_window // 2
    slice_width = slice_len // 2
    
    onehot_seqs = onehot_seqs.swapaxes(1,2)
    onehot_seqs = onehot_seqs[:, (in_width - slice_width):(in_width + slice_width), :]
    assert onehot_seqs.shape[1] == slice_len and onehot_seqs.shape[2] == 4, onehot_seqs.shape

    return onehot_seqs


def load_observed_profiles(pos_bw_path, neg_bw_path, peak_path, slice_len, out_window=1000, verbose=False):
    profs = extract_observed_profiles(pos_bw_path, neg_bw_path, peak_path, out_window=out_window, verbose=verbose)
    out_width = out_window // 2
    slice_width = slice_len // 2
    
    profs = profs[..., (out_width - slice_width):(out_width + slice_width)]
    assert profs.shape[-1] == slice_len, profs.shape

    return profs    


def load_scores(scores_path, slice_len, in_window=2114):
    in_width = in_window // 2
    slice_width = slice_len // 2
    
    print("Loading file...")
    load_start = time.time()
    hyp_scores = np.load(scores_path, mmap_mode='r')
    print(f"Loaded file in {time.time() - load_start:.2f} sec. Shape: {hyp_scores.shape}")

    print("Swapping axes...")
    swap_start = time.time()
    hyp_scores = hyp_scores.swapaxes(1,2)
    print(f"Swapped axes in {time.time() - swap_start:.2f} sec")


    hyp_scores = hyp_scores[:, (in_width - slice_width):(in_width + slice_width), :]
    assert hyp_scores.shape[1] == slice_len and hyp_scores.shape[2] == 4, hyp_scores.shape

    print("finished loading scores")
    return hyp_scores

    
def _run_modisco(onehot_seqs, scores, sliding_window_size=20, 
                 flank_size=5, n_leiden_runs=50, verbose=True, 
                 max_seqlets=1000000, initial_flank_to_add=8,
                 trim_to_window_size=24, target_seqlet_fdr=0.2):
    """Run modiscolite on arrays shaped (N, L, 4)."""

    print("Running modisco(lite)...")
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=scores, one_hot=onehot_seqs,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        initial_flank_to_add=initial_flank_to_add,
        trim_to_window_size=trim_to_window_size,
        target_seqlet_fdr=target_seqlet_fdr,
        n_leiden_runs=n_leiden_runs,
        verbose=verbose,
        )
    print("Finished running modisco(lite).")
    return pos_patterns, neg_patterns

    
def modisco(genome_path, chrom_sizes, peak_path, scores_path,
             results_save_path, verbose, slice_len=1000, in_window=2114, 
             sliding_window_size=20, flank_size=5, n_leiden_runs=50, 
             max_seqlets=1000000, save=True):
    
    print("Running modisco(lite).\n")
    print("genome_path:", genome_path)
    print("chrom_sizes:", chrom_sizes)
    print("peak_path:", peak_path)
    print("scores_path:", scores_path)
    print("slice_len:", slice_len)
    print("in_window:", in_window)
    print("sliding_window_size:", sliding_window_size)
    print("flank_size:", flank_size)
    print("n_leiden_runs:", n_leiden_runs)
    print("max_seqlets:", max_seqlets)
    print("results_save_path:", results_save_path)

    
    onehot_seqs = load_sequences(genome_path, chrom_sizes, peak_path,
                                 slice_len, in_window=in_window, verbose=verbose)
    print("Finished load_sequences.")

    scores = load_scores(scores_path, slice_len, in_window=in_window)

    try:
        pos_patterns, neg_patterns = _run_modisco(onehot_seqs, scores, 
                                                  sliding_window_size=sliding_window_size, 
                                                  flank_size=flank_size, 
                                                  n_leiden_runs=n_leiden_runs, 
                                                  verbose=verbose, max_seqlets=max_seqlets,
                                                  initial_flank_to_add=flank_size,
                                                  trim_to_window_size=trim_to_window_size)
    except Exception as e:
        print("Error in _run_modisco:", e)
        raise

    if save:
        print("saving results...")
        ensure_parent_dir_exists(results_save_path)
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns, in_window)
    else:
        return pos_patterns, neg_patterns

def modisco_meta(onehot_path, meta_path, results_save_path, 
                verbose, job, slice_len=1000, sliding_window_size=20, flank_size=5,
                trim_to_window_size=24, target_seqlet_fdr=0.2, n_leiden_runs=50, 
                max_seqlets=10000000, save=True,
                subset_idx_path=None):
    """Run TF-MoDISco from DeepSHAP memmaps and metadata.

    Use subset_idx_path when attribution was computed genome/train+val-wide but
    motif discovery should focus on selected regions. The indices must refer to
    rows in the DeepSHAP memmaps described by meta_path.
    """

    if job not in {"profile", "counts"}:
        raise ValueError(f"job must be 'profile' or 'counts', got {job}")

    meta = _load_meta(meta_path)
    print(meta)

    N = int(_require_meta_value(meta, "N"))
    L = int(_require_meta_value(meta, "L"))
    onehot_path = _resolve_onehot_path(meta, onehot_path)
    _validate_dat_file(onehot_path, np.uint8, (N, 4, L), "one-hot")

    # Load full train+val one-hot and attribution arrays from .dat memmaps.
    onehot_raw = np.memmap(onehot_path, dtype=np.uint8, mode="r", shape=(N, 4, L))
    scores_raw = _memmap_dat(meta, key_prefix=job)
    scores_path = scores_raw.filename

    if onehot_raw.shape != scores_raw.shape:
        raise ValueError(f"one-hot and scores shape mismatch: {onehot_raw.shape} vs {scores_raw.shape}")

    if subset_idx_path is not None:
        # Post-attribution region selection: keep rows for the regions of interest.
        idx = np.load(subset_idx_path).astype(np.int64)
        print(f"Loaded subset indices from {subset_idx_path}, n={len(idx)}")
        _validate_subset_indices(subset_idx_path, idx, meta)

        onehot_raw = onehot_raw[idx]
        scores_raw = scores_raw[idx]
        print("Subset onehot_raw shape:", onehot_raw.shape)
        print("Subset scores_raw shape:", scores_raw.shape)

    # TF-MoDISco usually runs on a central window around the peak summit.
    in_window = onehot_raw.shape[2]
    in_width = in_window // 2
    slice_width = slice_len // 2

    onehot_seqs = onehot_raw[:, :, (in_width - slice_width):(in_width + slice_width)]
    scores = scores_raw[:, :, (in_width - slice_width):(in_width + slice_width)]

    assert onehot_seqs.shape == scores.shape
    assert onehot_seqs.shape[1] == 4
    assert onehot_seqs.shape[2] == slice_len
    print("onehot_seqs shape:", onehot_seqs.shape)
    print("Finished load_sequences.")

    try:
        # deepshap.py writes channel-first (N, 4, L); modiscolite expects (N, L, 4).
        onehot_seqs = np.swapaxes(onehot_seqs, 1, 2)  # (N, slice_len, 4)
        scores      = np.swapaxes(scores, 1, 2)       # (N, slice_len, 4)
        print("onehot_seqs:", onehot_seqs.shape, onehot_seqs.dtype)
        print("scores:", scores.shape, scores.dtype)

        row_sums = _validate_onehot_array(onehot_seqs)
        if not np.isfinite(scores).all():
            raise ValueError("scores contain NaN or Inf")

        print("onehot finite:", True)
        print("scores finite:", True)
        print("row_sums min/max/mean:", row_sums.min(), row_sums.max(), row_sums.mean())
        print("valid frac(sum==1):", np.mean(row_sums == 1.0), " zero frac(sum==0):", np.mean(row_sums == 0.0))
        print("onehot min/max:", onehot_seqs.min(), onehot_seqs.max())

        pos_patterns, neg_patterns = _run_modisco(onehot_seqs, scores, sliding_window_size=sliding_window_size, 
                                                  flank_size=flank_size, n_leiden_runs=n_leiden_runs, 
                                                  verbose=verbose, max_seqlets=max_seqlets, initial_flank_to_add=flank_size, 
                                                  trim_to_window_size=trim_to_window_size, target_seqlet_fdr=target_seqlet_fdr)
    except Exception as e:
        print("Error in _run_modisco:", e)
        raise

    if save:
        print("saving results...")
        ensure_parent_dir_exists(results_save_path)
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns, in_window)
        _annotate_modisco_output(
            results_save_path, meta, meta_path, job, scores_path,
            onehot_path, subset_idx_path
        )
    else:
        return pos_patterns, neg_patterns


def _annotate_modisco_output(results_save_path, meta, meta_path, job, scores_path,
                             onehot_path, subset_idx_path):
    """Store attribution provenance on the MoDISco HDF5 output."""

    with h5py.File(results_save_path, "a") as fh:
        fh.attrs["attribution_method"] = str(meta.get("attribution_method", "unknown"))
        fh.attrs["attribution_name"] = str(meta.get("attribution_name", meta.get("attribution_method", "unknown")))
        fh.attrs["attribution_job"] = str(job)
        fh.attrs["attribution_meta_path"] = str(meta_path)
        fh.attrs["attribution_scores_path"] = str(scores_path)
        fh.attrs["seq_onehot_path"] = str(onehot_path)
        if subset_idx_path is not None:
            fh.attrs["subset_idx_path"] = str(subset_idx_path)
        for key in (
            "ct_name",
            "target_label",
            "target_mode",
            "target_cell_type_name",
            "residual_definition",
            "film_attribution_mode",
            "film_delta_definition",
        ):
            if meta.get(key) is not None:
                fh.attrs[key] = str(meta.get(key))
        if meta.get("contrast_cell_type_idxs") is not None:
            fh.attrs["contrast_cell_type_idxs"] = np.asarray(
                meta.get("contrast_cell_type_idxs"), dtype=np.int64
            )
        if meta.get("contrast_cell_type_names") is not None:
            fh.attrs["contrast_cell_type_names"] = ",".join(
                str(x) for x in meta.get("contrast_cell_type_names")
            )
        if meta.get("ig_steps") is not None:
            fh.attrs["ig_steps"] = int(meta.get("ig_steps"))
        if meta.get("ig_baseline") is not None:
            fh.attrs["ig_baseline"] = str(meta.get("ig_baseline"))

def load_modisco_results(tfm_results_path):
    return h5py.File(tfm_results_path, "r")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    set_modisco_parser(parser)

    config = parser.parse_args()

    parent_dir = os.path.dirname(config.modisco_results_path)
    os.makedirs(parent_dir, exist_ok=True)

    # modisco(config.genome_path,
    #         config.chrom_sizes,
    #         config.peak_path,
    #         config.scores_path,
    #         config.modisco_results_path,
    #         config.verbose,
    #         max_seqlets=config.max_seqlets,
    #         save=True)
    modisco_meta(config.onehot_path,
                config.meta_path,
                config.modisco_results_path,
                config.verbose,
                job=config.job,
                n_leiden_runs=config.n_leiden_runs,
                max_seqlets=config.max_seqlets,
                sliding_window_size=config.sliding_window_size,
                flank_size=config.flank_size,
                trim_to_window_size=config.trim_to_window_size,
                target_seqlet_fdr=config.target_seqlet_fdr,
                save=True,
                subset_idx_path=config.subset_idx_path)
