import argparse
import os
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm

from deepdetails.data import MultiTaskSupervisedDataset
from deepdetails.helper.attr import ModelWithSummarization, apply_gradient_requirements, undo_gradient_requirements
from deepdetails.model.wrapper import SupervisedDeepDETAILS


"""
Input x gradient attribution for supervised DeepDETAILS models.

This is a small command-line wrapper around the legacy helper.attr.py IXG logic.
It builds the same BED/fasta/bigWig datasets used by deepshap.py, optionally
keeps a subset of rows such as K562 distal train+val indices, and writes both:

  - attr.h5: datasets ohe and contrib for quick inspection / compatibility
  - modisco-compatible .dat + .npy metadata for src/deepdetails/modisco.py

The default "legacy" style uses ModelWithSummarization from helper.attr.py.
That matches the earlier input x gradient workflow more closely than the newer
DeepSHAP profile/count wrappers.
"""


class DatasetView(Dataset):
    """Attach t_x to ConcatDataset/Subset so helper-style attribution can read it."""

    def __init__(self, dataset, t_x: int):
        self.dataset = dataset
        self._t_x = int(t_x)

    @property
    def t_x(self):
        return self._t_x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class LegacySummarizedTarget(nn.Module):
    """Select one target cell type from helper.attr.ModelWithSummarization."""

    def __init__(self, base_model, target_cell_type_idx: int, summarizer: str,
                 contrast: Optional[str] = None, apply_loads_trick: bool = False):
        super().__init__()
        self.summarized = ModelWithSummarization(
            base_model,
            summarizer=summarizer,
            contrast=contrast,
            sample_in_first_dim=True,
            apply_loads_trick=apply_loads_trick,
        )
        self.target_cell_type_idx = int(target_cell_type_idx)

    def forward(self, seq, atac, loads):
        outputs = self.summarized(seq, atac, loads)
        if outputs.ndim != 2:
            raise ValueError(f"Expected summarized outputs with shape (B, targets), got {tuple(outputs.shape)}")
        return outputs[:, self.target_cell_type_idx]


class ProfileTarget(nn.Module):
    """BPNet/ProCapNet-style profile objective used by deepshap.py."""

    def __init__(self, model, target_cell_type_idx: int):
        super().__init__()
        self.model = model
        self.target_idx = int(target_cell_type_idx)

    def forward(self, seq, atac, loads):
        profiles_list, _, _, _ = self.model((seq, atac), loads, return_logits=True)
        logits = profiles_list[self.target_idx]
        mean_norm_logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        probs = torch.softmax(mean_norm_logits, dim=-1)
        return (mean_norm_logits * probs).sum(dim=(-1, -2))


class CountsTarget(nn.Module):
    """Selected cell-type counts objective used by deepshap.py."""

    def __init__(self, model, target_cell_type_idx: int):
        super().__init__()
        self.model = model
        self.target_idx = int(target_cell_type_idx)

    def forward(self, seq, atac, loads):
        _, counts_list, _, _ = self.model((seq, atac), loads)
        return counts_list[self.target_idx].sum(dim=-1)


def set_parser(parser):
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--peaks-bed', required=True)
    parser.add_argument('--val-chroms', nargs='+', required=True)
    parser.add_argument('--test-chroms', nargs='+', required=True)
    parser.add_argument('--fa-file', required=True)
    parser.add_argument('--acc-bw-files', nargs='+', required=True)
    parser.add_argument('--pl-ct-bw-files', nargs='+', required=True)
    parser.add_argument('--mn-ct-bw-files', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cell-type-names', nargs='+', required=True)
    parser.add_argument('--target-cell-type-idx', type=int, required=True)
    parser.add_argument('--score-regions', default='trainval', choices=['trainval', 'all', 'train', 'val', 'test'])
    parser.add_argument('--subset-idx-path', default=None,
                        help='Optional .npy row indices into the selected score_regions dataset.')
    parser.add_argument('--objective', choices=['counts', 'profile'], required=True)
    parser.add_argument('--style', choices=['legacy', 'deepshap-objective'], default='legacy',
                        help='legacy uses helper.attr.ModelWithSummarization; deepshap-objective uses the scalar wrappers from deepshap.py.')
    parser.add_argument('--summarizer', default=None,
                        help='Override legacy ModelWithSummarization summarizer. Defaults: counts=sum-alone, profile=weighted_sum_strandless.')
    parser.add_argument('--contrast', default=None, choices=[None, 'FC', 'LFC', 'LOAD'])
    parser.add_argument('--apply-loads-trick', action='store_true',
                        help='Attach loads to the graph for legacy IXG. Usually unnecessary for scale_function_placement=disable.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--abs-transform', action='store_true')
    parser.add_argument('--dtype-attr', default='float32')
    parser.add_argument('--write-modisco-dat', action='store_true', default=True)
    parser.add_argument('--no-write-modisco-dat', dest='write_modisco_dat', action='store_false')
    return parser


def make_dataset(args):
    split_mode_to_is_training = {
        'train': 1,
        'val': 0,
        'test': 2,
        'all': -1,
    }

    def build(is_training):
        return MultiTaskSupervisedDataset(
            args.peaks_bed,
            args.fa_file,
            args.acc_bw_files,
            args.pl_ct_bw_files,
            args.mn_ct_bw_files,
            args.cell_type_names,
            y_length=1000,
            is_training=is_training,
            chromosomal_val=args.val_chroms,
            chromosomal_test=args.test_chroms,
        )

    if args.score_regions == 'trainval':
        train_ds = build(1)
        val_ds = build(0)
        dataset = ConcatDataset([train_ds, val_ds])
        t_x = train_ds.t_x
    else:
        dataset = build(split_mode_to_is_training[args.score_regions])
        t_x = dataset.t_x

    if args.subset_idx_path is not None:
        idx = np.load(args.subset_idx_path).astype(np.int64)
        if idx.ndim != 1:
            raise ValueError(f'--subset-idx-path must contain a 1D array, got {idx.shape}')
        if len(idx) == 0:
            raise ValueError('--subset-idx-path is empty')
        if idx.min() < 0 or idx.max() >= len(dataset):
            raise ValueError(f'subset index out of range for dataset length {len(dataset)}: min={idx.min()} max={idx.max()}')
        dataset = Subset(dataset, idx.tolist())
    else:
        idx = None

    return DatasetView(dataset, t_x=t_x), idx


def default_summarizer(objective):
    if objective == 'counts':
        return 'sum-alone'
    return 'weighted_sum_strandless'


def make_target_model(args, model):
    if args.style == 'legacy':
        summarizer = args.summarizer or default_summarizer(args.objective)
        target_model = LegacySummarizedTarget(
            model,
            args.target_cell_type_idx,
            summarizer=summarizer,
            contrast=args.contrast,
            apply_loads_trick=args.apply_loads_trick,
        )
        return target_model, summarizer

    if args.summarizer is not None:
        raise ValueError('--summarizer only applies to --style legacy')
    if args.objective == 'counts':
        return CountsTarget(model, args.target_cell_type_idx), 'deepshap-counts-wrapper'
    return ProfileTarget(model, args.target_cell_type_idx), 'deepshap-profile-wrapper'


def open_memmaps(args, n_samples, seq_len, ct_name):
    if not args.write_modisco_dat:
        return None, None
    dtype_attr = np.dtype(args.dtype_attr)
    onehot_path = os.path.join(args.output_dir, f'{ct_name}_{args.objective}_ixg_onehot.dat')
    contrib_path = os.path.join(args.output_dir, f'{ct_name}_{args.objective}_ixg.dat')
    onehot_mm = np.memmap(onehot_path, mode='w+', dtype=np.uint8, shape=(n_samples, 4, seq_len))
    contrib_mm = np.memmap(contrib_path, mode='w+', dtype=dtype_attr, shape=(n_samples, 4, seq_len))
    return (onehot_mm, onehot_path), (contrib_mm, contrib_path)


def run_ixg(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    dataset, subset_idx = make_dataset(args)
    print('dataset length:', len(dataset), 't_x:', dataset.t_x)
    if subset_idx is not None:
        print('subset idx:', args.subset_idx_path, 'n:', len(subset_idx), 'min:', int(subset_idx.min()), 'max:', int(subset_idx.max()))

    model = SupervisedDeepDETAILS.load_from_checkpoint(
        args.model_path,
        num_cell_types=len(args.cell_type_names),
        strict=True,
    ).to(device).eval()

    target_model, summarizer_name = make_target_model(args, model)
    target_model = target_model.to(device).eval()
    print('target cell:', args.cell_type_names[args.target_cell_type_idx], args.target_cell_type_idx)
    print('objective:', args.objective, 'style:', args.style, 'summarizer:', summarizer_name)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    n_samples = len(dataset)
    seq_len = dataset.t_x
    ct_name = args.cell_type_names[args.target_cell_type_idx]

    h5_path = os.path.join(args.output_dir, f'{ct_name}_{args.objective}_ixg.h5')
    onehot_state, contrib_state = open_memmaps(args, n_samples, seq_len, ct_name)

    with h5py.File(h5_path, 'w') as h5:
        dset_ohe = h5.create_dataset('ohe', (n_samples, 4, seq_len), dtype='uint8', chunks=(1, 4, seq_len), compression='gzip')
        dset_contrib = h5.create_dataset('contrib', (n_samples, 4, seq_len), dtype=args.dtype_attr, chunks=(1, 4, seq_len), compression='gzip')
        if subset_idx is not None:
            h5.create_dataset('subset_idx', data=subset_idx)

        h5.attrs['model_path'] = args.model_path
        h5.attrs['peaks_bed'] = args.peaks_bed
        h5.attrs['score_regions'] = args.score_regions
        h5.attrs['subset_idx_path'] = '' if args.subset_idx_path is None else args.subset_idx_path
        h5.attrs['ct_name'] = ct_name
        h5.attrs['target_cell_type_idx'] = args.target_cell_type_idx
        h5.attrs['objective'] = args.objective
        h5.attrs['style'] = args.style
        h5.attrs['summarizer'] = summarizer_name

        offset = 0
        torch.set_grad_enabled(True)
        for batch in tqdm(loader, desc='IXG'):
            (seqs, atacs), _, _, loads = batch
            batch_size = seqs.shape[0]
            seq = seqs.to(device, non_blocking=True).float()
            atac = atacs.to(device, non_blocking=True).float()
            load = loads.to(device, non_blocking=True).float()

            inputs = (seq, atac, load)
            gradient_mask = apply_gradient_requirements(inputs, warn=False)
            try:
                with torch.backends.cudnn.flags(enabled=False):
                    outputs = target_model(seq, atac, load)
                    if outputs.ndim != 1 or outputs.shape[0] != batch_size:
                        raise ValueError(f'Target model must return shape (B,), got {tuple(outputs.shape)}')
                    grads = torch.autograd.grad(
                        outputs.sum(), seq,
                        allow_unused=True,
                        materialize_grads=True,
                        retain_graph=False,
                    )
                seq_grad = grads[0]
                contrib = seq * seq_grad if seq_grad is not None else torch.zeros_like(seq)
                if args.abs_transform:
                    contrib = contrib.abs()
            finally:
                undo_gradient_requirements(inputs, gradient_mask)

            end = offset + batch_size
            seq_np = seq.detach().cpu().numpy().astype(np.uint8, copy=False)
            contrib_np = contrib.detach().cpu().numpy().astype(args.dtype_attr, copy=False)
            dset_ohe[offset:end] = seq_np
            dset_contrib[offset:end] = contrib_np
            if onehot_state is not None:
                onehot_state[0][offset:end] = seq_np
                contrib_state[0][offset:end] = contrib_np
            offset = end

    if onehot_state is not None:
        onehot_mm, onehot_path = onehot_state
        contrib_mm, contrib_path = contrib_state
        onehot_mm.flush()
        contrib_mm.flush()
        meta = {
            'N': n_samples,
            'L': seq_len,
            'ct_name': ct_name,
            'cell_type_names': list(args.cell_type_names),
            'target_cell_type_idx': int(args.target_cell_type_idx),
            'model_path': args.model_path,
            'peaks_bed': args.peaks_bed,
            'val_chroms': list(args.val_chroms),
            'test_chroms': list(args.test_chroms),
            'score_regions': args.score_regions,
            'subset_idx_path': args.subset_idx_path,
            'attribution_method': 'input_x_gradient',
            'attribution_name': f'ixg_{args.style}',
            'objective': args.objective,
            'style': args.style,
            'summarizer': summarizer_name,
            'dtype_attr': str(np.dtype(args.dtype_attr)),
            'seq_onehot_path': onehot_path,
            'profile_path': contrib_path if args.objective == 'profile' else None,
            'counts_path': contrib_path if args.objective == 'counts' else None,
            'profile_onehot_path': None,
            'counts_onehot_path': None,
            'h5_path': h5_path,
        }
        meta_path = os.path.join(args.output_dir, f'{ct_name}_{args.objective}_ixg_meta.npy')
        np.save(meta_path, meta)
        np.save(onehot_path + '.meta.npy', {'N': n_samples, 'L': seq_len, 'dtype': 'uint8', 'seq_onehot_path': onehot_path})
        print('Wrote modisco onehot:', onehot_path)
        print('Wrote modisco scores:', contrib_path)
        print('Wrote modisco meta:', meta_path)

    print('Wrote h5:', h5_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_parser(parser)
    run_ixg(parser.parse_args())
