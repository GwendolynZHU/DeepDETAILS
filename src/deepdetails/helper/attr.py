import os
import h5py
import pybedtools
import pyBigWig
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from collections import defaultdict
from itertools import combinations, cycle
from typing import Union, Optional
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from deepdetails.data import SequenceSignalDataset
from deepdetails.helper.utils import calc_counts_per_locus


class ReducedDataset(torch.utils.data.Dataset):
    """
    Raw dataset may have overlap regions
    ReducedDataset as the name suggests, reduces the overlap regions
    by selecting a subset regions. If training regions A, B and C overlap,
    regions A and C will be kept (always the odd indexes)
    """
    def __init__(self, base_data: SequenceSignalDataset,
                 chrom_subset: Optional[tuple] = None,
                 subset: Optional[int] = None,
                 close_threshold: int = 500):
        self.base_data = base_data
        # input regions:
        #    index     0       1       2  3
        # 0      8  chr1  113521  117617  1
        # 1     46  chr1  603470  607566  1
        # 2     51  chr1  627724  631820  1
        # 3     52  chr1  628100  632196  1
        # 4     53  chr1  631971  636067  1
        self.raw_regions = base_data.df.copy()
        if chrom_subset is not None:
            self.raw_regions = self.raw_regions.loc[self.raw_regions[0].isin(chrom_subset)]
        self.raw_regions["orig_idx"] = self.raw_regions.index.values
        mids = self.raw_regions[[1, 2]].mean(axis=1).astype(int)
        tmp_df = pd.DataFrame({0: self.raw_regions[0], 1: mids, 2: mids + 1, 3: self.raw_regions["orig_idx"]})
        tmp_bed = pybedtools.BedTool.from_dataframe(tmp_df).sort().merge(d=close_threshold, c=4, o="distinct")
        reduced_regions = tmp_bed.to_dataframe(disable_auto_names=True, header=None)
        self.final_data_indexes = []
        for candidates in reduced_regions[3].str.split(",").values:
            self.final_data_indexes.extend([int(x) for x in candidates[0::2]])
        if subset is not None and len(self.final_data_indexes) > subset:
            self.final_data_indexes = np.random.choice(self.final_data_indexes, subset, replace=False).tolist()
        self.region_file = "region_mapping.bed"
        out_df = self.base_data.df.iloc[self.final_data_indexes].copy()
        out_df["orig_idx"] = self.final_data_indexes

        out_df[[0, 1, 2, 3, "orig_idx"]].to_csv(
            self.region_file, sep="\t", index=False, header=False
        )
    @property
    def t_y(self):
        return self.base_data.t_y

    @property
    def t_x(self):
        return self.base_data.t_x

    @property
    def n_clusters(self):
        return self.base_data.n_clusters

    def __getitem__(self, index):
        return self.base_data[int(self.final_data_indexes[index])]

    def __len__(self):
        return len(self.final_data_indexes)


class ModelWithSummarization(pl.LightningModule):
    def __init__(self, base_model: Union[nn.Module, pl.LightningModule],
                 summarizer: str = "weighted_sum", contrast: Optional[str] = None,
                 sample_in_first_dim: bool = True, apply_loads_trick: bool = False):
        """Wrapper model for applying summarization to the predictions

        Parameters
        ----------
        base_model : Union[nn.Module, pl.LightningModule]
            Base model class
        summarizer : str
            Summarization method. Currently, supports "weighted_sum", "sum", and "loads"
        contrast : Optional[str]
            set values such as fc (fold change) or lfc (log fold change) to calculate the contrast between
            the predicted clusters
        sample_in_first_dim : bool, optional
            If True, return summarized predictions with samples in the first dimension,
            i.e. shape (batch_size, expected_clusters).
        apply_loads_trick : bool, optional
            Loads may not be included in the computational graph when using second pass model or models without
            active scale functions. In these cases, captum raises errors like "RuntimeError: One of the differentiated
            Tensors appears to not have been used in the graph."
            Set this to True to forcefully attach loads to the graph to address the issue above.
        """
        super(ModelWithSummarization, self).__init__()
        self.summarizer = summarizer
        self.model = base_model
        self.expected_clusters = base_model.num_cell_types
        self.fc = True if contrast is not None and contrast.upper() == "FC" else False
        self.lfc = True if contrast is not None and contrast.upper() == "LFC" else False
        self.ld = True if contrast is not None and contrast.upper() == "LOAD" else False
        self.sample_in_first_dim = sample_in_first_dim
        self.apply_loads_trick = apply_loads_trick
        self._comp_groups = tuple(combinations(np.arange(base_model.num_cell_types), 2))

        if self.fc or self.lfc:
            print("Overriding summarizer clusters with the following contrast groups")
            for i, g in enumerate(self._comp_groups):
                print(i, g)
            self.expected_clusters = len(self._comp_groups)

    def forward(self, seq: torch.Tensor, atac: torch.Tensor, loads: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        seq : torch.Tensor
            one-hot encoded sequences
        atac : torch.Tensor

        loads : torch.Tensor


        Returns
        -------
        out : torch.Tensor
            shape: batch_size x expected_clusters
        """
        model_outs = self.model([seq, atac], loads)
        pc_profiles, pc_counts, pred_loads = model_outs[:3]
        if self.apply_loads_trick:
            trick = loads.sum(axis=-1).mean()
            pc_counts = [c * trick for c in pc_counts]
        if pc_profiles is not None:
            if isinstance(pc_profiles, torch.Tensor):  # hack for pure PROcapNet
                cluster_preds = torch.exp(self.model.model.log_softmax(pc_profiles)) * torch.exp(pc_counts)[..., None]
                cluster_preds = cluster_preds[None, ...]
            else:
                cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)  # c, m, s, l
            if self.summarizer == "weighted_sum":
                weights = torch.softmax(cluster_preds, axis=-1).detach()
                out = (weights * cluster_preds).sum(axis=-1).sum(axis=-1)
            elif self.summarizer == "weighted_sum_strandless":
                logits = cluster_preds.reshape(cluster_preds.shape[0], cluster_preds.shape[1], -1)
                mean_norm_logits = logits - torch.mean(logits, axis=-1, keepdims=True)
                softmax_probs = torch.nn.Softmax(dim=-1)(mean_norm_logits.detach())
                out = (mean_norm_logits * softmax_probs).sum(axis=-1)
            elif self.summarizer == "sum":
                out = cluster_preds.sum(axis=-1).sum(axis=-1)
            elif self.summarizer == "sum-alone":
                if isinstance(pc_profiles, torch.Tensor):
                    out = pc_counts[None, ...]
                else:
                    out = torch.stack(pc_counts).sum(axis=-1)
            elif self.summarizer == "softmax-counts":
                if isinstance(pc_profiles, torch.Tensor):
                    raise NotImplementedError()
                else:
                    out = torch.softmax(torch.stack(pc_counts).sum(axis=-1), dim=0)
            elif self.summarizer == "loads":
                out = pred_loads
            else:
                raise ValueError(f"{self.summarizer} is not supported.")

            if self.sample_in_first_dim and self.summarizer != "loads":
                out = torch.swapaxes(out, 0, 1)
        else:
            out = pred_loads
        # shape of out: batch_size, n_clusters
        if self.fc or self.lfc:
            out = out + 10e-16
            contrast_out = torch.zeros(seq.shape[0], len(self._comp_groups), device=out.device, dtype=out.dtype)
            for i, (gx, gy) in enumerate(self._comp_groups):
                contrast_out[:, i] = out[:, gx] / out[:, gy]

            if self.lfc:
                contrast_out = torch.log2(contrast_out)
            out = contrast_out
        elif self.ld:
            out = out + 10e-16
            out = out / out.sum(axis=-1)[:, None]

        return out


def apply_gradient_requirements(
    inputs: tuple[torch.Tensor, ...], warn: bool = True
) -> list[bool]:
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.

    Source: captum
    """
    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        if not input.requires_grad:
            if warn:
                print(
                    f"Input Tensor {index} (shape: {input.shape}) did not already require gradients, "
                    "required_grads has been set automatically."
                )
            input.requires_grad_()
    return grad_required


def undo_gradient_requirements(
    inputs: tuple[torch.Tensor, ...], grad_required: list[bool]
) -> None:
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.

    Source: captum
    """

    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(inputs) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        if not grad_required[index]:
            input.requires_grad_(False)


def ixg(model: Union[torch.nn.Module, pl.LightningModule],
        dataset: ReducedDataset, batch_size: int,
        save_to: str = ".", abs_transform: bool = False, quiet: bool = False) -> str:
    """
    compute gradient×input attributions per cluster target and save to hdf5.

    Parameters
    ----------
    model : torch.nn.Module or pl.LightningModule
        model whose forward returns a tensor of shape (batch_size, n_clusters).
    dataset : ReducedDataset

    batch_size : int
        number of samples per batch used during attribution computation.
    save_to : str, default "."
        directory to write the output file `attr.h5`. created if missing.
    abs_transform : bool, default False
        if true, store absolute values of attributions.
    quiet : bool, default False
        if true, disables progress bars and non-critical warnings.

    Notes
    -----
    Only the sequence input attribution (first element of the input tuple) is persisted.
    This function is modified from the InputXGradient implementation in captum.

    the hdf5 file contains:
    - dataset `ohe`: shape (n_samples, 4, seq_len), integer one-hot sequences.
    - dataset `contrib`: shape (n_samples, n_clusters, 4, seq_len), float attributions.

    returns
    -------
    result_file : str
        results are written to `save_to/attr.h5`.
    """
    # care for batchnorm + dropout
    torch.set_grad_enabled(True)
    model.eval()

    data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    n_samples = len(dataset)
    seq_len = dataset.t_x
    n_clusters = model.expected_clusters
    device = getattr(model, "device", next(model.parameters()).device)

    result_file = os.path.join(save_to, "attr.h5")
    os.makedirs(save_to, exist_ok=True)

    with h5py.File(result_file, "w") as f:
        # The one-hot encoded sequences and attributions are assumed to be in length-last format,
        # i.e., have the shape (# examples, 4, sequence length).
        # this denotes the identity of the sequence
        dset_ohe = f.create_dataset("ohe", (n_samples, 4, seq_len),
                                   dtype="i", chunks=(1, 4, seq_len), compression="gzip")
        # attribution matrix
        dset_contrib = f.create_dataset("contrib", (n_samples, n_clusters, 4, seq_len),
                                        dtype="float64", chunks=(1, 1, 4, seq_len), compression="gzip")

        for batch_idx, datum in enumerate(tqdm(data_iter, disable=quiet)):
            absolute_start_coord = batch_idx * batch_size
            absolute_end_coord = min((batch_idx + 1) * batch_size, n_samples)

            # build inputs; keep batch dimension intact
            x_ohe = datum[0][0].to(device, non_blocking=True)  # shape: (B, 4, L)
            x_acc = datum[0][1].to(device, non_blocking=True)
            x_load = datum[3].to(device, non_blocking=True)
            inputs = (x_ohe, x_acc, x_load)

            # save ohe sequences for this batch
            dset_ohe[absolute_start_coord:absolute_end_coord, :, :] = x_ohe.detach().cpu().numpy()

            for target in range(n_clusters):
                # it is assumed that for all given input tensors, dim 0 is batch
                with torch.autograd.set_grad_enabled(True):
                    gradient_mask = apply_gradient_requirements(inputs, warn=False)
                    with torch.backends.cudnn.flags(enabled=False):
                        outputs = model(*inputs)  # (B, n_clusters)
                        if outputs.ndim != 2 or outputs.shape[0] != x_ohe.shape[0] or outputs.shape[1] != n_clusters:
                            raise ValueError(
                                "Attribution model must return shape "
                                f"(batch_size, n_clusters); got {tuple(outputs.shape)}."
                            )
                        selected_outputs = outputs[:, target]
                        grads = torch.autograd.grad(
                            selected_outputs.sum(), x_ohe, allow_unused=True,
                            materialize_grads=True, retain_graph=False
                        )

                    seq_grad = grads[0]
                    contrib = x_ohe * seq_grad if seq_grad is not None else torch.zeros_like(x_ohe)
                    if abs_transform:
                        contrib = contrib.abs()

                    dset_contrib[absolute_start_coord:absolute_end_coord, target, :, :] = contrib.detach().cpu().numpy()

                    undo_gradient_requirements(inputs, gradient_mask)
    return result_file


# Functions make_track_values_dict and write_scores_to_bigwigs are modified from
# https://github.com/kundajelab/nascent_RNA_models/blob/main/src/utils/write_bigwigs.py


def make_track_values_dict(scores: np.ndarray, regions: pd.DataFrame, cluster_idx: int,
                           chrom: str, verbose: bool = False) -> dict:
    """
    Prepare bigWig track for a chromosome

    Parameters
    ----------
    scores : np.ndarray
        Score values for all chromosomes. Shape: (total_regions, seq_len) or (total_regions, )
    regions : pd.DataFrame
        DataFrame with columns: 0 (chromosome names), 1 (starts), 2 (ends).
        A view of the parent dataframe which only contains regions on the specified chromosome.
        Indexes are still their old indexes as in the parent dataframe (since the df is just a view),
        and the index values correspond to the rows in all_values.
    cluster_idx : int
        Cluster index in the score
    chrom : str
        Name of a chromosome to be used for building the track.
        Value should be in coords[0]
    verbose : bool, optional
        Set verbose to True to see a progress bar.

    Returns
    -------
    track_values : dict
        Dictionary containing position as key and average value as value

    References
    ----------

    """
    chroms = regions[0].unique()
    assert len(chroms) == 1 and chrom in chroms, f"regions dataframe should only contain records for chromosomes {chrom}"

    # Use defaultdict to simplify appending values to positions
    track_values = defaultdict(list)

    # Iterate through DataFrame rows using tqdm for progress visualization
    for i, (_, start, end) in tqdm(
            regions.iterrows(), total=regions.shape[0], desc=chrom, disable=not verbose):
        values = scores[i, cluster_idx, :, :].sum(axis=0).astype("float64")

        positions = np.arange(start, end)

        # Update defaultdict with position and corresponding value
        track_values.update((pos, track_values[pos] + [val]) for pos, val in zip(positions, values))

    # take the mean at each position, so that if there was overlap, the average value is used
    track_values = {key: np.mean(vals) if len(vals) > 1 else vals[0] for key, vals in track_values.items()}
    return track_values


def write_scores_to_bigwigs(score_file: str, peaks_file: str, cluster_idx: int, save_to: str,
                            chrom_sizes_file: str, verbose: bool = False):
    """
    Write attribution scores to a bigWig file.

    Parameters
    ----------
    score_file : str
        Score values for all chromosomes. Shape: total_regions, seq_len
    peaks_file : str
        Path to a bed file containing the corresponding regions for each row in the scores array.
    cluster_idx : int
        Cluster index in the score
    save_to : str
        Full path (including file name) to save the bigWig file.
    chrom_sizes_file : str
        Path to a tab-delimited file describing the size (col 2) of each chromosome (col 1).
    verbose : bool, optional
        Set verbose to True to see a progress bar.

    Returns
    -------
    None
    """
    with h5py.File(score_file, "r") as scores_file:
        assert "contrib" in scores_file
        scores = scores_file["contrib"]
        # scores = scores.astype("float64")
        peaks = pd.read_csv(peaks_file, sep="\t", header=None)

        assert peaks.shape[0] == scores.shape[0]

        chrom_sizes = pd.read_csv(chrom_sizes_file, sep="\t", header=None)

        with pyBigWig.open(save_to, "w") as bw:
            bw.addHeader([tuple(p) for p in chrom_sizes.values.tolist()])

            # for chrom, sub_df in peaks.groupby(0): # this would create bw header no match problem (chr10 preceding chr2)
            chrom_order = chrom_sizes[0].tolist()

            for chrom in chrom_order:
                sub_df = peaks[peaks[0] == chrom]
                if sub_df.empty:
                    continue
                print(f"Adding signals on chromosome {chrom}...")

                track_values_dict = make_track_values_dict(scores, sub_df[[0, 1, 2]], cluster_idx, chrom, verbose=verbose)
                num_entries = len(track_values_dict)

                starts = sorted(list(track_values_dict.keys()))
                ends = [position + 1 for position in starts]
                scores_to_write = [track_values_dict[key] for key in starts]

                assert len(scores_to_write) == len(starts) and len(scores_to_write) == len(ends) > 0

                bw.addEntries([chrom for _ in range(num_entries)],
                              starts, ends=ends, values=scores_to_write)
                print(f"Signals on chromosome {chrom} added...")
