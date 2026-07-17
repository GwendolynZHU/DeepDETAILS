import argparse
import sys
from contextlib import contextmanager
import torch
import numba
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange
from captum.attr import DeepLift, DeepLiftShap
# sys.path.insert(0, '/fs/cbsuhy01/storage/yz2676/data/DetailedProCapNet/src/DeepDETAILS/src')
# from ProCapNet.data_loading import extract_peaks
from deepdetails.data import MultiTaskSupervisedDataset
from deepdetails.model.wrapper import SupervisedDeepDETAILS


"""
DeepSHAP-style sequence attribution for supervised DeepDETAILS models.

The streaming path writes raw hypothetical attributions as memmap .dat files:
    <cell>_profile_deepshap.dat, <cell>_counts_deepshap.dat
and a shared raw sequence one-hot file used later by TF-MoDISco.

Attribution is always with respect to the sequence input. ATAC tracks and load
weights are passed as fixed additional_forward_args, so the question answered is:
which bases in this sequence matter for this selected output head, given the
observed ATAC/load context?
"""

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
                        help='Directory to save attribution results.')
    
    parser.add_argument('--cell-type-names', type=str, nargs='+', required=True,
                        help='List of cell type names corresponding to the model outputs.')
    
    parser.add_argument('--target-cell-type-idx', type=int, required=True,
                        help='Index of the target cell type for attribution analysis.')

    parser.add_argument('--target-mode', type=str, default='single',
                        choices=['single', 'residual'],
                        help='Attribution target to explain. single preserves the old behavior. '
                             'residual explains target cell type minus the mean of all non-target cell types.')

    parser.add_argument('--film-attribution-mode', type=str, default='normal',
                        choices=['normal', 'identity', 'delta'],
                        help='How FiLM modulation is handled during sequence attribution. normal uses the '
                             'ATAC-derived FiLM parameters. identity forces gamma=1 and beta=0. '
                             'delta writes normal minus identity attributions, retaining DNA-space scores '
                             'for MoDISco.')
    
    parser.add_argument('--total-shuffles', type=int, default=25,
                    help='Total number of dinuc-shuffle baselines per sequence (e.g., 10, 15, 25). '
                         'Used by DeepSHAP and by integrated gradients when --ig-baseline=dinuc_shuffle.')

    parser.add_argument('--shuffle-cap-per-call', type=int, default=10,
                    help='Max number of shuffles used in a single Captum call. '
                         'Use <=10 if larger values OOM. Total-shuffles will be split into chunks.')

    parser.add_argument('--attribution-method', type=str, default='deepshap',
                        choices=['deepshap', 'input_x_gradient', 'integrated_gradients'],
                        help='Attribution method to run. Default preserves the original DeepSHAP behavior.')

    parser.add_argument('--ig-steps', type=int, default=32,
                        help='Number of interpolation steps for integrated gradients.')

    parser.add_argument('--ig-baseline', type=str, default='zero',
                        choices=['zero', 'dinuc_shuffle'],
                        help='Baseline for integrated gradients. zero uses all-zero sequence baselines; '
                             'dinuc_shuffle averages over per-sequence dinucleotide shuffles.')
    
    parser.add_argument('--use-subset', action='store_true',
                        help='If set, run attribution on a random subset of train+val sequences.')
    
    parser.add_argument('--subset-size', type=int, default=40000,
                        help='Subset size when --use-subset is set. Default: 40000.')
    
    parser.add_argument('--subset-seed', type=int, default=42,
                        help='Random seed for subset sampling. Default: 42.')
    
    parser.add_argument('--batch-size', type=int, default=1,
                    help='Dataloader batch size for attribution streaming. Suggest 4-16.')

    parser.add_argument('--attr-pair-batch-size', type=int, default=0,
                    help='Max number of (input, shuffled-baseline) pairs processed in one '
                         'Captum call. 0 means use shuffle_cap_per_call, which matches the '
                         'old single-sequence memory footprint more closely.')

    parser.add_argument('--score-regions', type=str, default='trainval',
                        choices=['trainval', 'all', 'train', 'val', 'test'],
                        help='Which rows from --peaks-bed to attribute. Use all when scoring '
                             'the same regions across multiple fold models for ensemble MoDISco.')

    return parser

@numba.jit('void(int64, int64[:], int64[:], int32[:, :, :], int32[:, :], float32[:, :, :])')
def _fast_shuffle(n_shuffles, chars, idxs, shuffled_next_idxs, counters, shuffled_sequences):
    """An internal function for fast deterministic dinucleotide shuffling."""

    for i in range(n_shuffles):
        idx = 0
        shuffled_sequences[i, idxs[idx], 0] = 1
        for j in range(1, len(idxs)):
            char = idxs[idx]
            count = counters[i, char]
            idx = shuffled_next_idxs[i, char, count]

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

    max_char = int(max(chars)) + 1
    next_idxs = np.zeros((max_char, sequence.shape[1]), dtype=np.int32)
    next_idxs_counts = np.zeros(max_char, dtype=np.int32)

    for char in chars:
        next_idxs_ = np.where(idxs[:-1] == char)[0]
        n = len(next_idxs_)

        next_idxs[char][:n] = next_idxs_ + 1
        next_idxs_counts[char] = n

    shuffled_next_idxs = np.zeros((n_shuffles, max_char, sequence.shape[1]), dtype=np.int32)
    for i in range(n_shuffles):
        for char in chars:
            n = next_idxs_counts[char]
            order = np.arange(n)
            if n > 1:
                order[:-1] = random_state.permutation(n - 1)  # Keep last index same.
            shuffled_next_idxs[i, char, :n] = next_idxs[char, :n][order]

    shuffled_sequences = np.zeros((n_shuffles, *sequence.shape), dtype=np.float32)
    counters = np.zeros((n_shuffles, max_char), dtype=np.int32)

    _fast_shuffle(n_shuffles, chars, idxs, shuffled_next_idxs, counters, shuffled_sequences)

    shuffled_sequences = torch.from_numpy(shuffled_sequences)
    return shuffled_sequences

def _expand_additional_forward_args_for_repeats(additional_forward_args, repeats):
    if additional_forward_args is None:
        return None

    expanded_args = []
    for arg in additional_forward_args:
        if torch.is_tensor(arg):
            expanded_args.append(arg.repeat_interleave(repeats, dim=0))
        else:
            expanded_args.append(arg)
    return tuple(expanded_args)

def _slice_additional_forward_args(additional_forward_args, start, end):
    if additional_forward_args is None:
        return None

    sliced_args = []
    for arg in additional_forward_args:
        if torch.is_tensor(arg):
            sliced_args.append(arg[start:end])
        else:
            sliced_args.append(arg)
    return tuple(sliced_args)

def attribute_with_paired_shuffles(
    explainer, inputs, baselines, additional_forward_args=None, pair_batch_size=None
):
    """
    Compute DeepLIFT attributions for a batch where each input example has its
    own baseline distribution.

    Captum DeepLiftShap conceptually compares every input against a baseline
    distribution and averages over baselines. For DNA, each sequence gets its
    own dinucleotide-shuffled background, because GC/dinuc composition is a
    property of that region. This helper preserves those pairings:

        inputs:    (B, ...)
        baselines: (B, M, ...)
        pairs:     input_i vs baseline_i_j for j in 1..M

    It intentionally avoids mixing one region's shuffles into another region's
    baseline pool.

    Args:
        explainer: Captum DeepLift instance.
        inputs: Tensor of shape (B, ...)
        baselines: Tensor of shape (B, M, ...)
        additional_forward_args: optional tuple whose tensor members have batch dim B

    Returns:
        Tensor of shape (B, ...) containing the mean attribution across the M
        paired baselines for each example.
    """
    if baselines.ndim != inputs.ndim + 1:
        raise ValueError(
            f"Expected baselines to have exactly one extra dimension over inputs, "
            f"got inputs.shape={tuple(inputs.shape)} baselines.shape={tuple(baselines.shape)}"
        )
    if baselines.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"Batch size mismatch between inputs and baselines: "
            f"{inputs.shape[0]} vs {baselines.shape[0]}"
        )

    B, M = baselines.shape[:2]
    repeated_inputs = inputs.repeat_interleave(M, dim=0)
    flattened_baselines = baselines.reshape(B * M, *inputs.shape[1:])
    repeated_args = _expand_additional_forward_args_for_repeats(
        additional_forward_args, repeats=M
    )

    total_pairs = repeated_inputs.shape[0]
    if pair_batch_size is None or pair_batch_size <= 0:
        pair_batch_size = total_pairs

    attrs_chunks = []
    for start in range(0, total_pairs, pair_batch_size):
        end = min(total_pairs, start + pair_batch_size)
        attrs_chunks.append(
            explainer.attribute(
                inputs=repeated_inputs[start:end],
                baselines=flattened_baselines[start:end],
                additional_forward_args=_slice_additional_forward_args(
                    repeated_args, start, end
                ),
            )
        )

    attrs = torch.cat(attrs_chunks, dim=0) if len(attrs_chunks) > 1 else attrs_chunks[0]
    return attrs.reshape(B, M, *attrs.shape[1:]).mean(dim=1)


def _call_wrapper(wrapper, seq, additional_forward_args=None):
    if additional_forward_args is None:
        return wrapper(seq)
    return wrapper(seq, *additional_forward_args)


def _compute_input_gradients(wrapper, inputs, additional_forward_args=None):
    """Return d wrapper(inputs, args) / d inputs for scalar-per-example outputs."""

    grad_inputs = inputs.detach().clone().requires_grad_(True)
    outputs = _call_wrapper(wrapper, grad_inputs, additional_forward_args)
    if outputs.ndim != 1:
        outputs = outputs.reshape(outputs.shape[0], -1).sum(dim=1)
    gradients = torch.autograd.grad(outputs.sum(), grad_inputs)[0]
    return gradients


def attribute_input_x_gradient(wrapper, inputs, additional_forward_args=None):
    """Compute input * gradient attributions for the sequence input."""

    gradients = _compute_input_gradients(wrapper, inputs, additional_forward_args)
    return inputs * gradients


def _attribute_integrated_gradients_flat(
    wrapper, inputs, baselines, additional_forward_args=None, n_steps=32
):
    """Integrated gradients for a flat batch of input-baseline pairs."""

    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if baselines.shape != inputs.shape:
        raise ValueError(
            f"IG inputs and baselines must have the same shape, got "
            f"{tuple(inputs.shape)} vs {tuple(baselines.shape)}"
        )

    total_gradients = torch.zeros_like(inputs)
    delta = inputs - baselines
    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        scaled_inputs = baselines + alpha * delta
        total_gradients += _compute_input_gradients(
            wrapper, scaled_inputs, additional_forward_args
        )

    return delta * (total_gradients / float(n_steps))


def attribute_integrated_gradients(
    wrapper, inputs, baselines=None, additional_forward_args=None,
    n_steps=32, pair_batch_size=None
):
    """Compute integrated gradients for sequence inputs.

    `baselines` can be either a single baseline per example with shape matching
    inputs, or a paired baseline distribution shaped (B, M, ...). The latter is
    averaged across M baselines after preserving each input's own baseline set.
    """

    if baselines is None:
        baselines = torch.zeros_like(inputs)

    if baselines.ndim == inputs.ndim:
        return _attribute_integrated_gradients_flat(
            wrapper, inputs, baselines, additional_forward_args, n_steps=n_steps
        )

    if baselines.ndim != inputs.ndim + 1:
        raise ValueError(
            f"Expected IG baselines to match inputs or have one extra dimension, "
            f"got inputs.shape={tuple(inputs.shape)} baselines.shape={tuple(baselines.shape)}"
        )
    if baselines.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"Batch size mismatch between inputs and IG baselines: "
            f"{inputs.shape[0]} vs {baselines.shape[0]}"
        )

    B, M = baselines.shape[:2]
    repeated_inputs = inputs.repeat_interleave(M, dim=0)
    flattened_baselines = baselines.reshape(B * M, *inputs.shape[1:])
    repeated_args = _expand_additional_forward_args_for_repeats(
        additional_forward_args, repeats=M
    )

    total_pairs = repeated_inputs.shape[0]
    if pair_batch_size is None or pair_batch_size <= 0:
        pair_batch_size = total_pairs

    attrs_chunks = []
    for start in range(0, total_pairs, pair_batch_size):
        end = min(total_pairs, start + pair_batch_size)
        attrs_chunks.append(
            _attribute_integrated_gradients_flat(
                wrapper,
                repeated_inputs[start:end],
                flattened_baselines[start:end],
                _slice_additional_forward_args(repeated_args, start, end),
                n_steps=n_steps,
            )
        )

    attrs = torch.cat(attrs_chunks, dim=0) if len(attrs_chunks) > 1 else attrs_chunks[0]
    return attrs.reshape(B, M, *attrs.shape[1:]).mean(dim=1)

RESIDUAL_DEFINITION = "target - mean(all_non_target)"


def _validate_target_spec(cell_type_names, target_cell_type_idx, target_mode):
    """Return contrast cell type indices for the requested target mode."""

    if target_mode not in {"single", "residual"}:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    n_cell_types = len(cell_type_names)
    if target_cell_type_idx < 0 or target_cell_type_idx >= n_cell_types:
        raise ValueError(
            f"target_cell_type_idx out of range: {target_cell_type_idx} "
            f"for {n_cell_types} cell types"
        )
    contrast_idxs = [i for i in range(n_cell_types) if i != target_cell_type_idx]
    if target_mode == "residual" and len(contrast_idxs) == 0:
        raise ValueError("residual target mode requires at least two cell types")
    return contrast_idxs


def _apply_target_mode(scalars, target_idx, contrast_idxs, target_mode):
    """Select a single cell target or target-minus-others residual."""

    target = scalars[:, target_idx]
    if target_mode == "single":
        return target
    if target_mode == "residual":
        contrast = scalars[:, contrast_idxs].mean(dim=1)
        return target - contrast
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _target_output_label(ct_name, target_mode):
    if target_mode == "single":
        return ct_name
    return f"{ct_name}_{target_mode}"


class SupervisedProfileWrapper(nn.Module):
    """Expose one scalar profile objective for Captum.

    The wrapped model is multi-head: each cell type has a profile head. This
    wrapper selects one cell type, requests raw profile logits, mean-normalizes
    them over positions, and returns the expected logit under the predicted
    profile distribution. That is the BPNet/ProCapNet-style profile attribution
    target.
    """

    def __init__(self, model, target_cell_type_idx, target_mode="single"):
        super().__init__()
        self.model = model
        self.target_idx = int(target_cell_type_idx)
        self.target_mode = target_mode
        cell_type_names = [str(i) for i in range(model.num_cell_types)]
        self.contrast_idxs = _validate_target_spec(
            cell_type_names, self.target_idx, self.target_mode
        )

    def forward(self, seq, atac, per_cluster_load):
        x = (seq, atac)

        profiles_list, _, _, _ = self.model(x, per_cluster_load, return_logits=True)

        profile_scalars = []
        for logits in profiles_list:
            mean_norm_logits = logits - torch.mean(logits, dim=-1, keepdim=True)
            probs = torch.softmax(mean_norm_logits, dim=-1)
            profile_scalars.append((mean_norm_logits * probs).sum(dim=(-1, -2)))

        scalars = torch.stack(profile_scalars, dim=1)
        return _apply_target_mode(
            scalars, self.target_idx, self.contrast_idxs, self.target_mode
        )


class SupervisedCountsWrapper(nn.Module):
    """Expose one scalar count objective for Captum.

    Counts are predicted per strand for the selected cell-type head. Summing
    over strands gives one scalar per example, which Captum can attribute back
    to the sequence bases.
    """

    def __init__(self, model, target_cell_type_idx, target_mode="single"):
        super().__init__()
        self.model = model
        self.target_idx = int(target_cell_type_idx)
        self.target_mode = target_mode
        cell_type_names = [str(i) for i in range(model.num_cell_types)]
        self.contrast_idxs = _validate_target_spec(
            cell_type_names, self.target_idx, self.target_mode
        )

    def forward(self, seq, atac, per_cluster_load):
        x = (seq, atac)
        _, counts_list, _, _ = self.model(x, per_cluster_load)

        count_scalars = [pred_counts.sum(dim=-1) for pred_counts in counts_list]
        scalars = torch.stack(count_scalars, dim=1)
        return _apply_target_mode(
            scalars, self.target_idx, self.contrast_idxs, self.target_mode
        )


class SeqOnlyProfileWrapper(SupervisedProfileWrapper):
    pass


class SeqOnlyCountsWrapper(SupervisedCountsWrapper):
    pass


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

def make_shuffle_plan(total: int, cap: int) -> list[int]:
    """Split total shuffles into chunks that fit in GPU memory.

    Example: total=25, cap=10 gives [10, 10, 5]. Each chunk is attributed
    separately and then combined by a weighted mean, so this changes memory
    footprint but not the intended baseline average.
    """

    if total <= 0:
        raise ValueError("total_shuffles must be > 0")
    if cap <= 0:
        raise ValueError("shuffle_cap_per_call must be > 0")
    plan = []
    remain = total
    while remain > 0:
        m = min(cap, remain)
        plan.append(m)
        remain -= m
    return plan

def _get_hparam(model, name, default=None):
    hparams = getattr(model, "hparams", {})
    if isinstance(hparams, dict):
        return hparams.get(name, default)
    return getattr(hparams, name, default)


def _as_list(value):
    return list(value) if value is not None else None


def _film_source(model):
    return getattr(model, "model", model)


def _model_has_film(model):
    return hasattr(_film_source(model), "_last_film")


@contextmanager
def _temporary_film_attribution_mode(model, mode):
    src = _film_source(model)
    old_mode = getattr(src, "_film_attribution_mode", "normal")
    setattr(src, "_film_attribution_mode", mode)
    try:
        yield
    finally:
        setattr(src, "_film_attribution_mode", old_mode)


def _film_attribution_label(attribution_method, film_attribution_mode):
    if film_attribution_mode == "normal":
        return attribution_method
    return f"film_{film_attribution_mode}_{attribution_method}"


def _prepare_streaming_explainers(model, target_cell_type_idx, target_mode="single"):
    """Move model/wrappers to CUDA and build Captum explainers.

    The streaming implementation uses Captum DeepLift directly and performs the
    DeepSHAP baseline averaging in attribute_with_paired_shuffles. This keeps
    each sequence paired only with its own dinucleotide shuffles.
    """

    model.eval()
    model.cuda()

    prof_wrapper = SupervisedProfileWrapper(
        model, target_cell_type_idx=target_cell_type_idx, target_mode=target_mode
    ).cuda()
    count_wrapper = SupervisedCountsWrapper(
        model, target_cell_type_idx=target_cell_type_idx, target_mode=target_mode
    ).cuda()

    return {
        "model": model,
        "profile_wrapper": prof_wrapper,
        "count_wrapper": count_wrapper,
        "profile_explainer": DeepLift(prof_wrapper),
        "count_explainer": DeepLift(count_wrapper),
    }


def _peek_dataloader_batch(dataloader):
    """Read one batch to infer N, sequence length, and FiLM dimensions later."""

    (seqs0, atacs0), _, _, loads0 = next(iter(dataloader))
    N = len(dataloader.dataset)
    L = int(seqs0.shape[-1])
    return N, L, seqs0, atacs0, loads0


def _open_attribution_memmaps(
    output_dir, ct_name, N, L, dtype_attr, save_onehot_weighted,
    attribution_name="deepshap",
):
    """Create output memmaps for raw and optional one-hot-weighted scores."""

    os.makedirs(output_dir, exist_ok=True)

    prof_path = os.path.join(output_dir, f"{ct_name}_profile_{attribution_name}.dat")
    count_path = os.path.join(output_dir, f"{ct_name}_counts_{attribution_name}.dat")
    prof_mm = np.memmap(prof_path, mode="w+", dtype=dtype_attr, shape=(N, 4, L))
    count_mm = np.memmap(count_path, mode="w+", dtype=dtype_attr, shape=(N, 4, L))

    prof_oh_mm = count_oh_mm = None
    prof_oh_path = count_oh_path = None
    if save_onehot_weighted:
        prof_oh_path = os.path.join(output_dir, f"{ct_name}_profile_{attribution_name}_onehot.dat")
        count_oh_path = os.path.join(output_dir, f"{ct_name}_counts_{attribution_name}_onehot.dat")
        prof_oh_mm = np.memmap(prof_oh_path, mode="w+", dtype=dtype_attr, shape=(N, 4, L))
        count_oh_mm = np.memmap(count_oh_path, mode="w+", dtype=dtype_attr, shape=(N, 4, L))

    return {
        "profile_path": prof_path,
        "counts_path": count_path,
        "profile_mm": prof_mm,
        "counts_mm": count_mm,
        "profile_onehot_path": prof_oh_path,
        "counts_onehot_path": count_oh_path,
        "profile_onehot_mm": prof_oh_mm,
        "counts_onehot_mm": count_oh_mm,
    }


def _initialize_film_outputs(
    model, profile_wrapper, first_seqs, first_atacs, first_loads,
    cell_type_names, output_dir, N, dtype_film, target_label=None
):
    """Create FiLM summary memmaps if the model exposes _last_film.

    FiLM summaries are diagnostic outputs, not sequence attributions. They
    report the ATAC-derived modulation parameters that were active for each
    region and cell type.
    """

    film_src = model.model
    state = {
        "export": hasattr(film_src, "_last_film"),
        "src": film_src,
        "gamma_mms": {},
        "beta_mms": {},
        "gamma_paths": {},
        "beta_paths": {},
        "cell_type_names": [],
        "Lm": None,
        "mode": None,
    }
    if not state["export"]:
        return state

    with torch.no_grad(), _temporary_film_attribution_mode(model, "normal"):
        seq0 = first_seqs.float().cuda(non_blocking=True)
        atac0 = first_atacs.float().cuda(non_blocking=True)
        load0 = first_loads.float().cuda(non_blocking=True)
        _ = profile_wrapper(seq0, atac0, load0)

        for ct_idx, summary_ct_name in enumerate(cell_type_names):
            gamma0 = film_src._last_film[ct_idx]["gamma"]
            if state["Lm"] is None:
                state["Lm"] = int(gamma0.shape[2])
                state["mode"] = "channel-wise" if state["Lm"] == 1 else "position-wise"
            C = gamma0.shape[1]
            path_prefix = f"{target_label}_{summary_ct_name}" if target_label else summary_ct_name
            gamma_global_path = os.path.join(output_dir, f"{path_prefix}_film_gamma_global.dat")
            beta_global_path = os.path.join(output_dir, f"{path_prefix}_film_beta_global.dat")
            state["gamma_mms"][ct_idx] = np.memmap(
                gamma_global_path, mode="w+", dtype=dtype_film, shape=(N, C)
            )
            state["beta_mms"][ct_idx] = np.memmap(
                beta_global_path, mode="w+", dtype=dtype_film, shape=(N, C)
            )
            state["gamma_paths"][summary_ct_name] = gamma_global_path
            state["beta_paths"][summary_ct_name] = beta_global_path
            state["cell_type_names"].append(summary_ct_name)

    return state


def _write_film_batch(film_state, profile_wrapper, seq, atac, cluster_load,
                      cell_type_names, start_idx, batch_size, dtype_film):
    """Save mean gamma/beta summaries for one batch."""

    if not film_state["export"]:
        return

    with torch.no_grad(), _temporary_film_attribution_mode(profile_wrapper.model, "normal"):
        _ = profile_wrapper(seq, atac, cluster_load)

        for ct_idx, _summary_ct_name in enumerate(cell_type_names):
            g = film_state["src"]._last_film[ct_idx]["gamma"]
            b = film_state["src"]._last_film[ct_idx]["beta"]

            g_global = g.mean(dim=2) - 1.0
            b_global = b.mean(dim=2)
            end_idx = start_idx + batch_size
            film_state["gamma_mms"][ct_idx][start_idx:end_idx, :] = (
                g_global.detach().cpu().numpy().astype(dtype_film, copy=False)
            )
            film_state["beta_mms"][ct_idx][start_idx:end_idx, :] = (
                b_global.detach().cpu().numpy().astype(dtype_film, copy=False)
            )


def _build_paired_baselines(seqs, batch_start_idx, n_shuffles, rep):
    """Generate B independent dinuc-shuffle baseline sets for a batch."""

    B = seqs.shape[0]
    return torch.stack([
        dinuc_shuffle(
            seqs[j],
            n_shuffles=n_shuffles,
            random_state=int(42 + (batch_start_idx + j) * 1000 + rep),
        )
        for j in range(B)
    ], dim=0).cuda(non_blocking=True)


def _attribute_deepshap_batch_over_shuffle_plan(
    explainers, seq, seqs_cpu, atac, cluster_load, batch_start_idx,
    shuffle_plan, attr_pair_batch_size
):
    """Average profile/count DeepSHAP attributions over requested shuffles."""

    prof_sum = None
    count_sum = None
    denom = 0
    for rep, m in enumerate(shuffle_plan):
        ref_seqs = _build_paired_baselines(seqs_cpu, batch_start_idx, m, rep)

        prof_m = attribute_with_paired_shuffles(
            explainers["profile_explainer"],
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load),
            pair_batch_size=attr_pair_batch_size,
        )
        count_m = attribute_with_paired_shuffles(
            explainers["count_explainer"],
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load),
            pair_batch_size=attr_pair_batch_size,
        )

        prof_sum = (prof_m * m) if prof_sum is None else (prof_sum + prof_m * m)
        count_sum = (count_m * m) if count_sum is None else (count_sum + count_m * m)
        denom += m

        del prof_m, count_m, ref_seqs

    return prof_sum / denom, count_sum / denom


def _attribute_integrated_gradients_batch_over_shuffle_plan(
    explainers, seq, seqs_cpu, atac, cluster_load, batch_start_idx,
    shuffle_plan, attr_pair_batch_size, ig_steps
):
    """Average profile/count integrated gradients over dinuc baselines."""

    prof_sum = None
    count_sum = None
    denom = 0
    for rep, m in enumerate(shuffle_plan):
        ref_seqs = _build_paired_baselines(seqs_cpu, batch_start_idx, m, rep)

        prof_m = attribute_integrated_gradients(
            explainers["profile_wrapper"],
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load),
            n_steps=ig_steps,
            pair_batch_size=attr_pair_batch_size,
        )
        count_m = attribute_integrated_gradients(
            explainers["count_wrapper"],
            inputs=seq,
            baselines=ref_seqs,
            additional_forward_args=(atac, cluster_load),
            n_steps=ig_steps,
            pair_batch_size=attr_pair_batch_size,
        )

        prof_sum = (prof_m * m) if prof_sum is None else (prof_sum + prof_m * m)
        count_sum = (count_m * m) if count_sum is None else (count_sum + count_m * m)
        denom += m

        del prof_m, count_m, ref_seqs

    return prof_sum / denom, count_sum / denom


def _attribute_batch_core(
    attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
    batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline
):
    """Dispatch one sequence attribution pass under the current FiLM mode."""

    additional_args = (atac, cluster_load)
    if attribution_method == "deepshap":
        return _attribute_deepshap_batch_over_shuffle_plan(
            explainers, seq, seqs_cpu, atac, cluster_load, batch_start_idx,
            shuffle_plan, attr_pair_batch_size
        )

    if attribution_method == "input_x_gradient":
        prof_scores = attribute_input_x_gradient(
            explainers["profile_wrapper"], seq, additional_forward_args=additional_args
        )
        count_scores = attribute_input_x_gradient(
            explainers["count_wrapper"], seq, additional_forward_args=additional_args
        )
        return prof_scores, count_scores

    if attribution_method == "integrated_gradients":
        if ig_baseline == "zero":
            zero_baselines = torch.zeros_like(seq)
            prof_scores = attribute_integrated_gradients(
                explainers["profile_wrapper"], seq, baselines=zero_baselines,
                additional_forward_args=additional_args, n_steps=ig_steps,
                pair_batch_size=attr_pair_batch_size,
            )
            count_scores = attribute_integrated_gradients(
                explainers["count_wrapper"], seq, baselines=zero_baselines,
                additional_forward_args=additional_args, n_steps=ig_steps,
                pair_batch_size=attr_pair_batch_size,
            )
            return prof_scores, count_scores

        if ig_baseline == "dinuc_shuffle":
            return _attribute_integrated_gradients_batch_over_shuffle_plan(
                explainers, seq, seqs_cpu, atac, cluster_load, batch_start_idx,
                shuffle_plan, attr_pair_batch_size, ig_steps
            )

        raise ValueError(f"Unsupported integrated gradients baseline: {ig_baseline}")

    raise ValueError(f"Unsupported attribution method: {attribution_method}")


def _attribute_batch(
    attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
    batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline,
    film_attribution_mode="normal"
):
    """Dispatch sequence attribution for one dataloader batch.

    film_attribution_mode controls whether the model uses the learned ATAC/FiLM
    parameters (normal), identity FiLM (identity), or writes the DNA-space
    difference normal - identity (delta).
    """

    model = explainers["model"]
    if film_attribution_mode == "normal":
        with _temporary_film_attribution_mode(model, "normal"):
            return _attribute_batch_core(
                attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
                batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline
            )

    if film_attribution_mode == "identity":
        with _temporary_film_attribution_mode(model, "identity"):
            return _attribute_batch_core(
                attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
                batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline
            )

    if film_attribution_mode == "delta":
        with _temporary_film_attribution_mode(model, "normal"):
            prof_normal, count_normal = _attribute_batch_core(
                attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
                batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline
            )
        with _temporary_film_attribution_mode(model, "identity"):
            prof_identity, count_identity = _attribute_batch_core(
                attribution_method, explainers, seq, seqs_cpu, atac, cluster_load,
                batch_start_idx, shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline
            )
        return prof_normal - prof_identity, count_normal - count_identity

    raise ValueError(f"Unsupported film_attribution_mode: {film_attribution_mode}")

def _write_attribution_batch(outputs, seqs, prof_scores, count_scores,
                             start_idx, batch_size, dtype_attr):
    """Copy one attribution batch from GPU tensors into output memmaps."""

    end_idx = start_idx + batch_size
    prof_np = prof_scores.detach().cpu().numpy().astype(dtype_attr, copy=False)
    count_np = count_scores.detach().cpu().numpy().astype(dtype_attr, copy=False)
    outputs["profile_mm"][start_idx:end_idx] = prof_np
    outputs["counts_mm"][start_idx:end_idx] = count_np

    if outputs["profile_onehot_mm"] is not None:
        seq_np = seqs.detach().cpu().numpy().astype(dtype_attr, copy=False)
        outputs["profile_onehot_mm"][start_idx:end_idx] = prof_np * seq_np
        outputs["counts_onehot_mm"][start_idx:end_idx] = count_np * seq_np


def _flush_stream_outputs(outputs, film_state):
    """Flush all open memmaps so long jobs leave readable partial progress."""

    outputs["profile_mm"].flush()
    outputs["counts_mm"].flush()
    if outputs["profile_onehot_mm"] is not None:
        outputs["profile_onehot_mm"].flush()
        outputs["counts_onehot_mm"].flush()
    if film_state["export"]:
        for mm in film_state["gamma_mms"].values():
            mm.flush()
        for mm in film_state["beta_mms"].values():
            mm.flush()


def _build_deepshap_meta(
    model, N, L, ct_name, target_label, cell_type_names, target_cell_type_idx,
    target_mode, contrast_cell_type_idxs,
    model_path, peaks_bed, val_chroms, test_chroms, seq_onehot_path, score_regions,
    attribution_method, attribution_name, ig_steps, ig_baseline,
    film_attribution_mode, film_delta_definition,
    total_shuffles, shuffle_cap_per_call, shuffle_plan, flush_every,
    dtype_attr, dtype_film, outputs, film_state
):
    """Collect the file/schema metadata consumed by downstream MoDISco."""

    return {
        "N": N, "L": L, "Lm": int(film_state["Lm"]) if film_state["Lm"] is not None else None,
        "ct_name": ct_name,
        "target_label": target_label,
        "cell_type_names": list(cell_type_names),
        "target_mode": target_mode,
        "target_cell_type_idx": int(target_cell_type_idx),
        "target_cell_type_name": ct_name,
        "contrast_cell_type_idxs": [int(i) for i in contrast_cell_type_idxs],
        "contrast_cell_type_names": [cell_type_names[i] for i in contrast_cell_type_idxs],
        "residual_definition": RESIDUAL_DEFINITION if target_mode == "residual" else None,
        "model_path": model_path,
        "peaks_bed": peaks_bed,
        "val_chroms": _as_list(val_chroms),
        "test_chroms": _as_list(test_chroms),
        "seq_onehot_path": seq_onehot_path,
        "score_regions": score_regions,
        "seq_only": bool(_get_hparam(model, "seq_only", False)),
        "modulation_only": bool(_get_hparam(model, "modulation_only", False)),
        "scale_function_placement": _get_hparam(model, "scale_function_placement", None),
        "attribution_method": attribution_method,
        "attribution_name": attribution_name,
        "film_attribution_mode": film_attribution_mode,
        "film_delta_definition": film_delta_definition if film_attribution_mode == "delta" else None,
        "ig_steps": int(ig_steps) if ig_steps is not None else None,
        "ig_baseline": ig_baseline,
        "total_shuffles": int(total_shuffles),
        "shuffle_cap_per_call": int(shuffle_cap_per_call),
        "shuffle_plan": shuffle_plan,
        "flush_every": int(flush_every),
        "dtype_attr": str(np.dtype(dtype_attr)),
        "dtype_film": str(np.dtype(dtype_film)),
        "profile_path": outputs["profile_path"],
        "counts_path": outputs["counts_path"],
        "profile_onehot_path": outputs["profile_onehot_path"],
        "counts_onehot_path": outputs["counts_onehot_path"],
        "film_gamma_path": film_state["gamma_paths"].get(ct_name) if film_state["export"] else None,
        "film_beta_path": film_state["beta_paths"].get(ct_name) if film_state["export"] else None,
        "film_gamma_paths": film_state["gamma_paths"] if film_state["export"] else None,
        "film_beta_paths": film_state["beta_paths"] if film_state["export"] else None,
        "film_cell_type_names": film_state["cell_type_names"] if film_state["export"] else None,
        "film_mode": film_state["mode"],
        "film_summary": "gamma files store gamma-1 channel summaries; beta files store mean beta summaries",
        "note": "Sequence attributions are conditional on fixed ATAC and loads. FiLM summaries are saved for all cell types when the model exposes _last_film.",
    }


def run_deepshap_single_celltype_streaming(
    model, dataloader, cell_type_names, target_cell_type_idx,
    output_dir, total_shuffles=25, shuffle_cap_per_call=10,
    save_onehot_weighted=True, dtype_attr=np.float32,
    dtype_film=np.float16, film_reduce="mean", flush_every=500,
    attr_pair_batch_size=0, model_path=None, peaks_bed=None,
    val_chroms=None, test_chroms=None, seq_onehot_path=None,
    score_regions="trainval", attribution_method="deepshap",
    ig_steps=32, ig_baseline="zero", target_mode="single",
    film_attribution_mode="normal",
):
    """Stream sequence attributions for one cell-type output head.

    DeepSHAP remains the default. input_x_gradient computes input * gradient
    at the observed sequence. integrated_gradients uses either all-zero
    baselines or paired dinucleotide-shuffled baselines.

    `film_reduce` is currently reserved for future FiLM summary variants; the
    existing behavior stores mean gamma-1 and mean beta summaries.
    """

    if attribution_method not in {"deepshap", "input_x_gradient", "integrated_gradients"}:
        raise ValueError(f"Unsupported attribution method: {attribution_method}")
    if film_attribution_mode not in {"normal", "identity", "delta"}:
        raise ValueError(f"Unsupported film_attribution_mode: {film_attribution_mode}")
    if film_attribution_mode != "normal" and not _model_has_film(model):
        raise ValueError(
            f"--film-attribution-mode={film_attribution_mode} requires a FiLM model exposing _last_film"
        )
    contrast_cell_type_idxs = _validate_target_spec(
        cell_type_names, target_cell_type_idx, target_mode
    )
    if ig_steps <= 0:
        raise ValueError("ig_steps must be > 0")
    if ig_baseline not in {"zero", "dinuc_shuffle"}:
        raise ValueError(f"Unsupported integrated gradients baseline: {ig_baseline}")

    attribution_name = _film_attribution_label(attribution_method, film_attribution_mode)
    film_delta_definition = "normal FiLM sequence attribution - identity FiLM sequence attribution"
    shuffle_plan = make_shuffle_plan(total_shuffles, shuffle_cap_per_call)
    if attribution_method == "deepshap" or (attribution_method == "integrated_gradients" and ig_baseline == "dinuc_shuffle"):
        print(
            f"[{attribution_name}] total_shuffles={total_shuffles}, "
            f"cap={shuffle_cap_per_call}, plan={shuffle_plan}"
        )
    if attribution_method == "integrated_gradients":
        print(f"[{attribution_name}] ig_steps={ig_steps}, ig_baseline={ig_baseline}")
    print(f"[{attribution_name}] film_attribution_mode={film_attribution_mode}")
    if film_attribution_mode == "delta":
        print(f"[{attribution_name}] film_delta_definition={film_delta_definition}")
    if attr_pair_batch_size <= 0:
        attr_pair_batch_size = shuffle_cap_per_call
    print(f"[{attribution_name}] attr_pair_batch_size={attr_pair_batch_size}")

    ct_name = cell_type_names[target_cell_type_idx]
    target_label = _target_output_label(ct_name, target_mode)
    contrast_names = [cell_type_names[i] for i in contrast_cell_type_idxs]
    if target_mode == "residual":
        print(
            f"Starting {attribution_name} residual for: {ct_name} "
            f"minus mean({contrast_names})"
        )
    else:
        print(f"Starting {attribution_name} for: {ct_name} (Index {target_cell_type_idx})")

    explainers = _prepare_streaming_explainers(
        model, target_cell_type_idx, target_mode=target_mode
    )
    N, L, seqs0, atacs0, loads0 = _peek_dataloader_batch(dataloader)
    outputs = _open_attribution_memmaps(
        output_dir, target_label, N, L, dtype_attr, save_onehot_weighted,
        attribution_name=attribution_name,
    )
    film_state = _initialize_film_outputs(
        model, explainers["profile_wrapper"], seqs0, atacs0, loads0,
        cell_type_names, output_dir, N, dtype_film, target_label=target_label
    )

    i = 0
    for batch in tqdm(dataloader, desc=f"{attribution_name} ({ct_name})"):
        (seqs, atacs), _, _, loads = batch
        B = seqs.shape[0]

        seq = seqs.float().cuda(non_blocking=True)
        atac = atacs.float().cuda(non_blocking=True)
        cluster_load = loads.float().cuda(non_blocking=True)

        _write_film_batch(
            film_state, explainers["profile_wrapper"], seq, atac, cluster_load,
            cell_type_names, i, B, dtype_film
        )
        prof_scores, count_scores = _attribute_batch(
            attribution_method, explainers, seq, seqs, atac, cluster_load, i,
            shuffle_plan, attr_pair_batch_size, ig_steps, ig_baseline,
            film_attribution_mode=film_attribution_mode
        )
        _write_attribution_batch(
            outputs, seqs, prof_scores, count_scores, i, B, dtype_attr
        )

        del prof_scores, count_scores
        i += B

        if i % flush_every == 0:
            _flush_stream_outputs(outputs, film_state)

    _flush_stream_outputs(outputs, film_state)

    meta = _build_deepshap_meta(
        model, N, L, ct_name, target_label, cell_type_names, target_cell_type_idx,
        target_mode, contrast_cell_type_idxs,
        model_path, peaks_bed, val_chroms, test_chroms, seq_onehot_path, score_regions,
        attribution_method, attribution_name, ig_steps, ig_baseline,
        film_attribution_mode, film_delta_definition,
        total_shuffles, shuffle_cap_per_call, shuffle_plan, flush_every,
        dtype_attr, dtype_film, outputs, film_state
    )
    np.save(os.path.join(output_dir, f"{target_label}_{attribution_name}_meta.npy"), meta)
    if film_state["export"]:
        print(f"Saved {attribution_name} + all-cell-type FiLM summaries for target {target_label} in {output_dir}")
    else:
        print(f"Saved {attribution_name} results for target {target_label} in {output_dir}")

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


def _load_saved_dict(path):
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and isinstance(value.item(), dict):
        return value.item()
    raise ValueError(f"Expected dict metadata in {path}, got {type(value)}")


def _existing_onehot_matches(datapath_out, N, L, dtype):
    meta_path = datapath_out + ".meta.npy"
    if not (os.path.exists(datapath_out) and os.path.exists(meta_path)):
        return None

    dtype = np.dtype(dtype)
    expected_size = int(N) * 4 * int(L) * dtype.itemsize
    if os.path.getsize(datapath_out) != expected_size:
        return None

    try:
        meta = _load_saved_dict(meta_path)
    except Exception as exc:
        print(f"Existing onehot meta is unreadable; rewriting. path={meta_path} error={exc}")
        return None

    if int(meta.get("N", -1)) != int(N):
        return None
    if int(meta.get("L", -1)) != int(L):
        return None
    if np.dtype(meta.get("dtype")) != dtype:
        return None
    return meta

def write_onehot_only(datapath_out, dataloader, dtype=np.float32, verify_onehot=True):
    """
    Write raw one-hot sequences from a DataLoader to a memmap .dat file.

    Expects batches like:
        ((seqs, atacs), _, _, loads)
    with seqs shaped (B, 4, L).

    Produces:
        - datapath_out: (N, 4, L) raw one-hot sequences
        - datapath_out + ".meta.npy": {N, L, dtype, seq_onehot_path}
    """

    N = len(dataloader.dataset)
    (seqs, _atacs), *_ = next(iter(dataloader))
    L = int(seqs.shape[-1])
    dtype = np.dtype(dtype)

    existing_meta = _existing_onehot_matches(datapath_out, N, L, dtype)
    if existing_meta is not None:
        print("Reusing existing onehot:", datapath_out, "shape:", (N, 4, L))
        return datapath_out, existing_meta

    out_dir = os.path.dirname(datapath_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = f"{datapath_out}.tmp.{os.getpid()}"
    tmp_meta_path = tmp_path + ".meta.npy"
    mm = np.memmap(tmp_path, mode="w+", dtype=dtype, shape=(N, 4, L))

    i = 0
    try:
        for batch in tqdm(dataloader, desc="Writing onehot"):
            (seqs, _atacs), _, _, _loads = batch
            B = seqs.shape[0]
            seq_np = seqs.detach().cpu().numpy().astype(dtype, copy=False)  # (B, 4, L)

            if verify_onehot:
                # Allow Ns as all-zero columns if your pipeline uses that.
                col_sums = seq_np.sum(axis=1)  # (B, L)
                if not (
                    np.all((col_sums == 1) | (col_sums == 0))
                    and np.all((seq_np == 0) | (seq_np == 1))
                ):
                    raise ValueError(
                        f"Not one-hot at i={i}. seq min/max={seq_np.min()}/{seq_np.max()}, "
                        f"col_sums min/max={col_sums.min()}/{col_sums.max()}"
                    )

            mm[i:i+B] = seq_np
            i += B

        mm.flush()
        del mm

        meta = {
            "N": N,
            "L": L,
            "dtype": str(dtype),
            "seq_onehot_path": datapath_out,
        }
        np.save(tmp_meta_path, meta)
        os.replace(tmp_path, datapath_out)
        os.replace(tmp_meta_path, datapath_out + ".meta.npy")
    except Exception:
        try:
            del mm
        except UnboundLocalError:
            pass
        for path in (tmp_path, tmp_meta_path):
            if os.path.exists(path):
                os.remove(path)
        raise

    print("Wrote onehot:", datapath_out, "shape:", (N, 4, L))
    print("Meta:", datapath_out + ".meta.npy")
    return datapath_out, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = set_parser(parser)
    args = parser.parse_args()

    split_mode_to_is_training = {
        "train": 1,
        "val": 0,
        "test": 2,
        "all": -1,
    }
    if args.score_regions == "trainval":
        val_ds = MultiTaskSupervisedDataset(
            args.peaks_bed, args.fa_file, args.acc_bw_files, args.pl_ct_bw_files, args.mn_ct_bw_files, args.cell_type_names, y_length=1000, is_training=0,
            chromosomal_val=args.val_chroms, chromosomal_test=args.test_chroms,
        )
        train_ds = MultiTaskSupervisedDataset(
            args.peaks_bed, args.fa_file, args.acc_bw_files, args.pl_ct_bw_files, args.mn_ct_bw_files, args.cell_type_names, y_length=1000, is_training=1,
            chromosomal_val=args.val_chroms, chromosomal_test=args.test_chroms,
        )
        combo_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
    else:
        combo_ds = MultiTaskSupervisedDataset(
            args.peaks_bed, args.fa_file, args.acc_bw_files, args.pl_ct_bw_files, args.mn_ct_bw_files, args.cell_type_names, y_length=1000,
            is_training=split_mode_to_is_training[args.score_regions],
            chromosomal_val=args.val_chroms, chromosomal_test=args.test_chroms,
        )

    model = SupervisedDeepDETAILS.load_from_checkpoint(
        args.model_path,
        num_cell_types=len(args.cell_type_names),
        strict=True
    )

    if args.use_subset:
        K = min(args.subset_size, len(combo_ds))
        rng = np.random.RandomState(args.subset_seed)
        idx = rng.choice(len(combo_ds), size=K, replace=False)
        combo_ds_used = torch.utils.data.Subset(combo_ds, idx)
        print(f"[{args.attribution_method}] Using {args.score_regions} subset: {K}/{len(combo_ds)} (seed={args.subset_seed})")
    else:
        combo_ds_used = combo_ds
        print(f"[{args.attribution_method}] Using {args.score_regions}: {len(combo_ds_used)} sequences")
    
    combo_iter = DataLoader(combo_ds_used, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # One-hot sequences are needed for TF-MoDISco
    if args.use_subset:
        seq_out = os.path.join(args.output_dir, f"{args.score_regions}_subset{K}_seq_onehot.dat")
    else:
        seq_out = os.path.join(args.output_dir, f"{args.score_regions}_seq_onehot_{args.target_cell_type_idx}.dat")
    seq_onehot_path, _ = write_onehot_only(seq_out, combo_iter, dtype=np.uint8, verify_onehot=True)
    run_deepshap_single_celltype_streaming(
        model=model,
        dataloader=combo_iter,
        cell_type_names=args.cell_type_names,
        target_cell_type_idx=args.target_cell_type_idx,
        output_dir=args.output_dir,
        total_shuffles=args.total_shuffles,
        shuffle_cap_per_call=args.shuffle_cap_per_call,
        save_onehot_weighted=False,
        dtype_attr=np.float32,
        dtype_film=np.float16,
        attr_pair_batch_size=args.attr_pair_batch_size,
        model_path=args.model_path,
        peaks_bed=args.peaks_bed,
        val_chroms=args.val_chroms,
        test_chroms=args.test_chroms,
        seq_onehot_path=seq_onehot_path,
        score_regions=args.score_regions,
        attribution_method=args.attribution_method,
        ig_steps=args.ig_steps,
        ig_baseline=args.ig_baseline,
        target_mode=args.target_mode,
        film_attribution_mode=args.film_attribution_mode,
    )
