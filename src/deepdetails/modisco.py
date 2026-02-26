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

sys.path.append('/fs/cbsuhy01/storage/yz2676/data/DetailedProCapNet/src/ProCapNet/')
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
                        help="Path to one-hot sequence .dat file",
                        type=str,
                        required=True,
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

def _load_meta(meta_path):
    meta = np.load(meta_path, allow_pickle=True)
    # meta might be dict saved via np.save(...)
    if isinstance(meta, np.ndarray) and meta.shape == () and isinstance(meta.item(), dict):
        meta = meta.item()
    return meta


def _memmap_dat(meta, key_prefix):
    """
    key_prefix examples:
        - "profile", "counts"
        - "profile_onehot", "counts_onehot"
    """
    N = meta.get("N")
    L = meta.get("L")
    dtype = meta.get("dtype")
    dat_path = meta.get(f"{key_prefix}_path")

    arr = np.memmap(dat_path, dtype=dtype, mode="r", shape=(N, 4, L))
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

    
def _run_modisco(onehot_seqs, scores, verbose=True, max_seqlets=1000000):
    print("Running modisco(lite)...")
    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=scores, one_hot=onehot_seqs,
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=20,
        flank_size=5,
        target_seqlet_fdr=0.05,
        n_leiden_runs=50,
        verbose=verbose,
        )
    print("Finished running modisco(lite).")
    return pos_patterns, neg_patterns

    
def modisco(genome_path, chrom_sizes, peak_path, scores_path,
             results_save_path, verbose, slice_len=1000, in_window=2114, max_seqlets=1000000, save=True):
    
    print("Running modisco(lite).\n")
    print("genome_path:", genome_path)
    print("chrom_sizes:", chrom_sizes)
    print("peak_path:", peak_path)
    print("scores_path:", scores_path)
    print("slice_len:", slice_len)
    print("in_window:", in_window)
    print("max_seqlets:", max_seqlets)
    print("results_save_path:", results_save_path)

    
    onehot_seqs = load_sequences(genome_path, chrom_sizes, peak_path,
                                 slice_len, in_window=in_window, verbose=verbose)
    print("Finished load_sequences.")

    scores = load_scores(scores_path, slice_len, in_window=in_window)

    try:
        pos_patterns, neg_patterns = _run_modisco(onehot_seqs, scores, verbose=verbose, max_seqlets=max_seqlets)
    except Exception as e:
        print("Error in _run_modisco:", e)
        return

    if save:
        print("saving results...")
        ensure_parent_dir_exists(results_save_path)
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns, in_window)
    else:
        return pos_patterns, neg_patterns

def modisco_meta(onehot_path, meta_path, results_save_path, 
                verbose, slice_len=1000, max_seqlets=10000000, save=True
                ):
    meta = _load_meta(meta_path)
    print(meta)

    # Load onehot and shap from .dat
    onehot_raw = np.memmap(onehot_path, dtype=np.uint8, mode="r", shape=(meta.get("N"), 4, meta.get("L")))
    scores_raw = _memmap_dat(meta, key_prefix="profile")

    # Slice to central slice_len (same logic you already had)
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
        onehot_seqs = np.swapaxes(onehot_seqs, 1, 2)  # (N, slice_len, 4)
        scores      = np.swapaxes(scores, 1, 2)       # (N, slice_len, 4)
        print("onehot_seqs:", onehot_seqs.shape, onehot_seqs.dtype)
        print("scores:", scores.shape, scores.dtype)

        # 1) check NaN/Inf
        print("onehot finite:", np.isfinite(onehot_seqs).all())
        print("scores finite:", np.isfinite(scores).all())

        # 2) check one-hot validity: each position should sum to 1 (or 0 if you allow Ns)
        row_sums = onehot_seqs.sum(axis=-1)  # (N, L)
        print("row_sums min/max/mean:", row_sums.min(), row_sums.max(), row_sums.mean())
        print("valid frac(sum==1):", np.mean(row_sums == 1.0), " zero frac(sum==0):", np.mean(row_sums == 0.0))

        # 3) check onehot values are near {0,1}
        print("onehot min/max:", onehot_seqs.min(), onehot_seqs.max())

        pos_patterns, neg_patterns = _run_modisco(onehot_seqs, scores, verbose=verbose, max_seqlets=max_seqlets)
    except Exception as e:
        print("Error in _run_modisco:", e)
        return

    if save:
        print("saving results...")
        ensure_parent_dir_exists(results_save_path)
        modiscolite.io.save_hdf5(results_save_path, pos_patterns, neg_patterns, in_window)
    else:
        return pos_patterns, neg_patterns

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
                max_seqlets=config.max_seqlets,
                save=True)