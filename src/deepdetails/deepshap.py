import argparse
import sys
import torch
import numba
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from captum.attr import DeepLiftShap
sys.path.insert(0, '/fs/cbsuhy01/storage/yz2676/data/DetailedProCapNet/src/DeepDETAILS/src')
# from ProCapNet.data_loading import extract_peaks
from deepdetails.data import MultiTaskSupervisedDataset
from deepdetails.model.wrapper import SupervisedDeepDETAILS

def set_parser(parser):
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained SupervisedDeepDETAILS model checkpoint.')
    
    parser.add_argument('--peaks-bed', type=str, required=True,
                        help='Path to the BED file containing peak regions for model input.')
    
    parser.add_argument('--val-chroms', type=str, nargs='+', required=True,
                        help='List of chromosomes to use for validation.')
    
    parser.add_argument('--test-chroms', type=str, nargs='+', required=True,
                        help='List of chromosomes to use for testing.')
    
    parser.add_argument('--fa-file', type=str, required=True,
                        help='Path to the reference genome FASTA file.')
    
    parser.add_argument('--acc-bw-files', type=str, nargs='+', required=True,
                        help='List of bigWig files for accessibility data for each cell type.')

    parser.add_argument('--pl-ct-bw-files', type=str, nargs='+', required=True,
                        help='List of bigWig files for plus strand PROcap data for each cell type.')
    parser.add_argument('--mn-ct-bw-files', type=str, nargs='+', required=True,
                        help='List of bigWig files for minus strand PROcap data for each cell type.')
    
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save deepSHAP results.')
    
    parser.add_argument('--cell-type-names', type=str, nargs='+', required=True,
                        help='List of cell type names corresponding to the model outputs.')
    
    parser.add_argument('--target-cell-type-idx', type=int, required=True,
                        help='Index of the target cell type for deepSHAP analysis.')
    
    parser.add_argument('--num-shuffles', type=int, default=10,
                        help='Number of dinucleotide shuffles for baseline references.')

    return parser

@numba.jit('void(int64, int64[:], int64[:], int32[:, :], int32[:,], int32[:, :], float32[:, :, :])')
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, counters, shuffled_sequences):
    """An internal function for fast shuffling using numba."""

    for i in range(n_shuffles):
        for char in chars:
            n = next_idxs_counts[char]

            next_idxs_ = np.arange(n)
            next_idxs_[:-1] = np.random.permutation(n-1)  # Keep last index same
            next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

        idx = 0
        shuffled_sequences[i, idxs[idx], 0] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = next_idxs[char, count]

            counters[i, char] += 1
            shuffled_sequences[i, idxs[idx], j] = 1

def dinuc_shuffle(sequence, n_shuffles=10, random_state=None):
    """Given a one-hot encoded sequence, dinucleotide shuffle it.
    This function takes in a one-hot encoded sequence (not a string) and
    returns a set of one-hot encoded sequences that are dinucleotide
    shuffled. The approach constructs a transition matrix between
    nucleotides, keeps the first and last nucleotide constant, and then
    randomly at uniform selects transitions until all nucleotides have
    been observed. This is a Eulerian path. Because each nucleotide has
    the same number of transitions into it as out of it (except for the
    first and last nucleotides) the greedy algorithm does not need to
    check at each step to make sure there is still a path.
    This function has been adapted to work on PyTorch tensors instead of
    numpy arrays. Code has been adapted from
    https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py
    Parameters
    ----------
    sequence: torch.tensor, shape=(k, -1)
        The one-hot encoded sequence. k is usually 4 for nucleotide sequences
        but can be anything in practice.
    n_shuffles: int, optional
        The number of dinucleotide shuffles to return. Default is 10.
    random_state: int or None or numpy.random.RandomState, optional
        The random seed to use to ensure determinism. If None, the
        process is not deterministic. Default is None. 
    Returns
    -------
    shuffled_sequences: torch.tensor, shape=(n_shuffles, k, -1)
        The shuffled sequences.
    """

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
    chars, idxs = chars.cpu().numpy(), idxs.cpu().numpy()

    next_idxs = np.zeros((len(chars), sequence.shape[1]), dtype=np.int32)
    next_idxs_counts = np.zeros(max(chars)+1, dtype=np.int32)

    for char in chars:
        next_idxs_ = np.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_sequences = np.zeros((n_shuffles, *sequence.shape), dtype=np.float32)
    counters = np.zeros((n_shuffles, len(chars)), dtype=np.int32)

    _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
        counters, shuffled_sequences)

    shuffled_sequences = torch.from_numpy(shuffled_sequences)
    return shuffled_sequences

class SeqOnlyProfileWrapper(nn.Module):
    def __init__(self, model, target_cell_type_idx):
        super().__init__()
        self.model = model
        self.target_idx = target_cell_type_idx

    def forward(self, seq, atac, per_cluster_load):
        x = (seq, atac)

        profiles_list, _, _, _ = self.model(x, per_cluster_load, return_logits=True)
        
        logits = profiles_list[self.target_idx]
        
        mean_norm_logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        probs = torch.softmax(mean_norm_logits, dim=-1)
        
        # (Batch, Strands, Length) -> Sum over Length and Strands -> (Batch,)
        return (mean_norm_logits * probs).sum(dim=(-1, -2))


class SeqOnlyCountsWrapper(nn.Module):
    def __init__(self, model, target_cell_type_idx):
        super().__init__()
        self.model = model
        self.target_idx = target_cell_type_idx

    def forward(self, seq, atac, per_cluster_load):
        x = (seq, atac)
        _, counts_list, _, _ = self.model(x, per_cluster_load)
        
        pred_counts = counts_list[self.target_idx]
        
        return pred_counts.sum(dim=-1)

def save_deepshap_results(onehot_seqs, prof_attrs, count_attrs, output_dir, ct_name):
    prof_attrs = np.concatenate(prof_attrs, axis=0)  # (N, 4, L)
    count_attrs = np.concatenate(count_attrs, axis=0)  # (N, 4, L)

    prof_onehot = prof_attrs * onehot_seqs
    count_onehot = count_attrs * onehot_seqs

    os.makedirs(output_dir, exist_ok=True)
    prof_path = os.path.join(output_dir, f"{ct_name}_profile_deepshap.npy")
    count_path = os.path.join(output_dir, f"{ct_name}_counts_deepshap.npy")

    prof_onehot_path = os.path.join(output_dir, f"{ct_name}_profile_deepshap_onehot.npy")
    count_onehot_path = os.path.join(output_dir, f"{ct_name}_counts_deepshap_onehot.npy")

    np.save(prof_path, prof_attrs)
    np.save(count_path, count_attrs)
    np.save(prof_onehot_path, prof_onehot)
    np.save(count_onehot_path, count_onehot)
    print(f"Saved deepSHAP results for {ct_name} to {output_dir}")

def run_deepshap_single_celltype(model, onehot_seqs, input_atacs, 
                               per_cluster_load, cell_type_names, 
                               target_cell_type_idx,
                               output_dir,
                               num_shuffles=10):
    """
    args:
        model: DeepDETAILS model
        onehot_seqs: (N, 4, L)
        input_atacs: (N, num_cell_types, L)
        per_cluster_load: (N, num_clusters)
        cell_type_names: list of cell type names
        target_cell_type_idx: int, index of the target cell type
        output_dir: str, directory to save outputs
        num_shuffles: int, number of dinucleotide shuffles for baseline
    """
    ct_name = cell_type_names[target_cell_type_idx]
    print(f"Starting deepSHAP for: {ct_name} (Index {target_cell_type_idx})")

    model.eval()
    model.cuda()
    
    prof_wrapper = SeqOnlyProfileWrapper(model, target_cell_type_idx=target_cell_type_idx)
    count_wrapper = SeqOnlyCountsWrapper(model, target_cell_type_idx=target_cell_type_idx)
        
    prof_explainer = DeepLiftShap(prof_wrapper)
    count_explainer = DeepLiftShap(count_wrapper)

    prof_attrs = []
    count_attrs = []

    for i in trange(len(onehot_seqs)):
        seq = torch.from_numpy(onehot_seqs[i:i+1]).float().cuda()  # (1, 4, L)
        atac = torch.from_numpy(input_atacs[i:i+1]).float().cuda()  # (1, n_ct, L)
        cluster_load = torch.from_numpy(per_cluster_load[i:i+1]).float().cuda()  # (1, n_clusters)

        ref_seqs = dinuc_shuffle(seq[0], n_shuffles=num_shuffles).cuda()    
        
        prof_scores = prof_explainer.attribute(
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load)
        )

        count_scores = count_explainer.attribute(
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load)
        )

        prof_scores = prof_scores.cpu().detach().numpy()  # (1, 4, L)
        count_scores = count_scores.cpu().detach().numpy()  # (1, 4, L)
        prof_attrs.append(prof_scores)
        count_attrs.append(count_scores)
    
    # save results
    save_deepshap_results(onehot_seqs, prof_attrs, count_attrs, output_dir, ct_name)

def run_deepshap_single_celltype_streaming(
    model, dataloader, cell_type_names, target_cell_type_idx,
    output_dir, num_shuffles=10, save_onehot_weighted=True, dtype=np.float32
):
    ct_name = cell_type_names[target_cell_type_idx]
    print(f"Starting deepSHAP for: {ct_name} (Index {target_cell_type_idx})")

    model.eval()
    model.cuda()

    prof_wrapper = SeqOnlyProfileWrapper(model, target_cell_type_idx=target_cell_type_idx).cuda()
    count_wrapper = SeqOnlyCountsWrapper(model, target_cell_type_idx=target_cell_type_idx).cuda()

    prof_explainer = DeepLiftShap(prof_wrapper)
    count_explainer = DeepLiftShap(count_wrapper)

    N = len(dataloader.dataset)

    (seqs, atacs), _, _, loads = next(iter(dataloader))
    L = seqs.shape[-1]

    os.makedirs(output_dir, exist_ok=True)

    prof_path = os.path.join(output_dir, f"{ct_name}_profile_deepshap.dat")
    count_path = os.path.join(output_dir, f"{ct_name}_counts_deepshap.dat")
    prof_mm = np.memmap(prof_path, mode="w+", dtype=dtype, shape=(N, 4, L))
    count_mm = np.memmap(count_path, mode="w+", dtype=dtype, shape=(N, 4, L))

    if save_onehot_weighted:
        prof_oh_path = os.path.join(output_dir, f"{ct_name}_profile_deepshap_onehot.dat")
        count_oh_path = os.path.join(output_dir, f"{ct_name}_counts_deepshap_onehot.dat")
        prof_oh_mm = np.memmap(prof_oh_path, mode="w+", dtype=dtype, shape=(N, 4, L))
        count_oh_mm = np.memmap(count_oh_path, mode="w+", dtype=dtype, shape=(N, 4, L))
    else:
        prof_oh_mm = count_oh_mm = None

    # Iterate through dataloader
    i = 0
    for batch in tqdm(dataloader, desc=f"DeepSHAP ({ct_name})"):
        (seqs, atacs), _, _, loads = batch  # batch_size=1
        seq = seqs.float().cuda(non_blocking=True)       # (1, 4, L)
        atac = atacs.float().cuda(non_blocking=True)     # (1, n_ct, L)
        cluster_load = loads.float().cuda(non_blocking=True)  # (1, n_clusters)

        ref_seqs = dinuc_shuffle(seq[0], n_shuffles=num_shuffles).cuda()

        prof_scores = prof_explainer.attribute(
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load)
        )
        count_scores = count_explainer.attribute(
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load)
        )

        prof_np = prof_scores.detach().cpu().numpy()[0].astype(dtype, copy=False)
        count_np = count_scores.detach().cpu().numpy()[0].astype(dtype, copy=False)
        seq_np = seq.detach().cpu().numpy()[0].astype(dtype, copy=False)

        prof_mm[i] = prof_np
        count_mm[i] = count_np

        if save_onehot_weighted:
            prof_oh_mm[i] = prof_np * seq_np
            count_oh_mm[i] = count_np * seq_np

        i += 1

    prof_mm.flush(); count_mm.flush()
    if save_onehot_weighted:
        prof_oh_mm.flush(); count_oh_mm.flush()

    meta = {
        "N": N, "L": L, "dtype": str(np.dtype(dtype)),
        "ct_name": ct_name,
        "profile_path": prof_path,
        "counts_path": count_path,
        "profile_onehot_path": prof_oh_path if save_onehot_weighted else None,
        "counts_onehot_path": count_oh_path if save_onehot_weighted else None,
    }
    np.save(os.path.join(output_dir, f"{ct_name}_deepshap_meta.npy"), meta)
    print(f"Saved deepSHAP memmaps + meta for {ct_name} in {output_dir}")


def extract_data_from_dataloader(dataloader):
    """
    Extracts inputs (Seq, ATAC, Loads) from the MultiTaskSupervisedDataset DataLoader.
    
    Args:
        dataloader: A DataLoader wrapping MultiTaskSupervisedDataset
    
    Returns:
        onehot_seqs: (N, 4, L)
        input_atacs: (N, n_cell_types, L)
        per_cluster_load: (N, n_cell_types)
    """
    all_seqs = []
    all_atacs = []
    all_loads = []

    for batch in tqdm(dataloader, desc="Extracting data"):
        (seqs, atacs), _, _, loads = batch
        all_seqs.append(seqs.detach().cpu().numpy())
        all_atacs.append(atacs.detach().cpu().numpy())
        all_loads.append(loads.detach().cpu().numpy())

    onehot_seqs = np.concatenate(all_seqs, axis=0)       # Shape: (N, 4, 4096)
    input_atacs = np.concatenate(all_atacs, axis=0)      # Shape: (N, n_cell_types, 4096)
    per_cluster_load = np.concatenate(all_loads, axis=0) # Shape: (N, n_cell_types)

    return onehot_seqs, input_atacs, per_cluster_load

def write_onehot_only(datapath_out, dataloader, dtype=np.float32, verify_onehot=True):
    """
    Write raw one-hot sequences from a DataLoader to a memmap .dat file.

    Expects batches like:
        ((seqs, atacs), _, _, loads)
    with batch_size=1 and seqs shaped (1, 4, L).

    Produces:
        - datapath_out: (N, 4, L) raw one-hot sequences
        - datapath_out + ".meta.npy": {N, L, dtype, seq_onehot_path}
    """

    N = len(dataloader.dataset)
    (seqs, _atacs), *_ = next(iter(dataloader))
    L = int(seqs.shape[-1])

    os.makedirs(os.path.dirname(datapath_out), exist_ok=True)
    mm = np.memmap(datapath_out, mode="w+", dtype=dtype, shape=(N, 4, L))

    for i, batch in enumerate(tqdm(dataloader, desc="Writing onehot")):
        (seqs, _atacs), _, _, _loads = batch
        seq_np = seqs.detach().cpu().numpy()[0].astype(dtype, copy=False)  # (4, L)

        if verify_onehot:
            # Allow Ns as all-zero columns if your pipeline uses that.
            col_sums = seq_np.sum(axis=0)
            if not (np.all((col_sums == 1) | (col_sums == 0)) and np.all((seq_np == 0) | (seq_np == 1))):
                raise ValueError(
                    f"Not one-hot at i={i}. seq min/max={seq_np.min()}/{seq_np.max()}, "
                    f"col_sums min/max={col_sums.min()}/{col_sums.max()}"
                )

        mm[i] = seq_np

    mm.flush()

    meta = {
        "N": N,
        "L": L,
        "dtype": str(np.dtype(dtype)),
        "seq_onehot_path": datapath_out,
    }
    np.save(datapath_out + ".meta.npy", meta)
    print("Wrote onehot:", datapath_out, "shape:", (N, 4, L))
    print("Meta:", datapath_out + ".meta.npy")
    return datapath_out, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_parser(parser)
    args = parser.parse_args()

    val_ds = MultiTaskSupervisedDataset(
        args.peaks_bed, args.fa_file, args.acc_bw_files, args.pl_ct_bw_files, args.mn_ct_bw_files, args.cell_type_names, y_length=1000, is_training=0,
        chromosomal_val=args.val_chroms, chromosomal_test=args.test_chroms,
    )
    val_iter = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    train_ds = MultiTaskSupervisedDataset(
        args.peaks_bed, args.fa_file, args.acc_bw_files, args.pl_ct_bw_files, args.mn_ct_bw_files, args.cell_type_names, y_length=1000, is_training=1,
        chromosomal_val=args.val_chroms, chromosomal_test=args.test_chroms,
    )
    train_iter = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    # val_onehots, val_atacs, val_loads = extract_data_from_dataloader(val_iter)
    # train_onehots, train_atacs, train_loads = extract_data_from_dataloader(train_iter)

    # onehot_seqs = np.concatenate([train_onehots, val_onehots], axis=0)
    # input_atacs = np.concatenate([train_atacs, val_atacs], axis=0)
    # per_cluster_load =  np.concatenate([train_loads, val_loads], axis=0)
    # print(onehot_seqs.shape, input_atacs.shape, per_cluster_load.shape)

    model = SupervisedDeepDETAILS.load_from_checkpoint(
        args.model_path,
        num_cell_types=len(args.cell_type_names),
        seq_only= True,
        strict=True
    )

    # run_deepshap_single_celltype(
    #     model=model,
    #     onehot_seqs=onehot_seqs,
    #     input_atacs=input_atacs,
    #     per_cluster_load=per_cluster_load,
    #     cell_type_names=args.cell_type_names,
    #     target_cell_type_idx=args.target_cell_type_idx,
    #     output_dir=args.output_dir,
    #     num_shuffles=args.num_shuffles
    # )

    combo_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])

    K = 40000
    rng = np.random.RandomState(42)
    idx = rng.choice(len(combo_ds), size=K, replace=False)
    combo_sub = torch.utils.data.Subset(combo_ds, idx)
    combo_iter = DataLoader(combo_sub, batch_size=1, shuffle=False, num_workers=0)
    
    # seq_out = os.path.join(args.output_dir, f"subset{K}_seq_onehot.dat")
    # write_onehot_only(seq_out, combo_iter, dtype=np.uint8, verify_onehot=True)
    run_deepshap_single_celltype_streaming(
        model=model,
        dataloader=combo_iter,
        cell_type_names=args.cell_type_names,
        target_cell_type_idx=args.target_cell_type_idx,
        output_dir=args.output_dir,
        num_shuffles=args.num_shuffles,
        save_onehot_weighted=True,
        dtype=np.float32
    )
