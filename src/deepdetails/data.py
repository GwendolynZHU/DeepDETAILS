import os
import h5py
import pyBigWig
import pyfaidx
import torch
import numpy as np
import pandas as pd
from typing import Sequence, Optional
from torch.utils.data import Dataset
from deepdetails.helper.prep_ds import seq_to_one_hot, extract_signal_from_bw
from deepdetails.par_description import PARAM_DESC


def parse_regions(hdf5_handle: h5py.File) -> pd.DataFrame:
    """
    Load regions information from a HDF5 file

    Parameters
    ----------
    hdf5_handle : h5py.File
        hdf5 object with a dataset "regions" and chromosome mapping information as attributions.
        The mappings have key "X_d" (d is the chromosome number in the dataset) and value is the
        actual chromosome name.

    Returns
    -------
    region_df : pd.DataFrame
        Region dataframe
    """
    assert "regions" in hdf5_handle
    df = pd.DataFrame(hdf5_handle["regions"][:])
    _chr_mapping = {}
    for k, v in hdf5_handle["regions"].attrs.items():
        _chr_mapping[int(k.split("_")[1])] = v
    df[0] = df[0].map(_chr_mapping)
    return df


class SequenceSignalDataset(Dataset):
    """
    Dataset class for both one-hot encoded sequences and signal tracks
    """

    def __init__(self, root: str, y_length: int = 1_000,
                 is_training: int = 1, chromosomal_val: Optional[Sequence[str]] = None,
                 chromosomal_test: Optional[Sequence[str]] = None, loads_trunc: Optional[int] = None,
                 pos_only_subset: Optional[int] = None, non_background_only: bool = False,
                 subset_seed: Optional[int] = None, enable_additional_filter: Optional[bool] = None):
        """

        Parameters
        ----------
        root : str
            {dataset}
        y_length : int
            {y_length}
        is_training : int
            0 : validation, 1 : training, 2 : testing, by default 1
        chromosomal_val : Optional[Sequence[str]]
            {chromosomal_validation}
        chromosomal_test : Optional[Sequence[str]]
            {chromosomal_testing}
        loads_trunc : Optional[int]
            {loads_trunc}
        pos_only_subset : Union[None,int]
            Set it as a positive integer if you want to sample a subset of positive regions in the training set.
            The size of the subset will be identical with the value of this argument. By default None
        non_background_only : bool
            Set it as True if you only want to use non-background regions
        subset_seed : Union[None,int]
            by default None
        enable_additional_filter : Optional[bool]
            Set it as True to select regions that are labeled as 1 in the 6th col.
            Set it as False to select regions that are labeled as 0 in the 6th col.
            Leave it as None if you don't want to apply this additional filter.

        Raises
        ------
        IOError
            If training npz data cannot be found
        ValueError

        """.format(**PARAM_DESC)
        self.expected_data_file = os.path.join(root, "data.h5")
        assert os.path.exists(self.expected_data_file)
        self.dataset = None
        self.load_groundtruth = False
        self.has_acc_norm = False
        self.prior = torch.tensor(0)

        with h5py.File(self.expected_data_file, "r") as fh:
            assert "dec" in fh
            assert "regions" in fh
            self.df = parse_regions(fh)

            self.n_targets = fh["dec"].attrs["n_targets"]
            self.n_clusters = fh["dec"].attrs["n_clusters"]
            self._t_x = fh["dec/seq"].shape[-1]

            try:
                self.cluster_names = fh["dec"].attrs["cluster_names"].split(",")
                if len(self.cluster_names) != self.n_clusters:
                    raise ValueError("len(cluster_names) != n_clusters")
            except (KeyError, ValueError):
                self.cluster_names = [f"C{i}" for i in range(self.n_clusters)]

            self.norm_factors = np.ones(self.n_clusters)
            if "scatac_norm" in fh:
                norm_df = pd.DataFrame(fh["scatac_norm"][:])
                if norm_df.shape[1] != 2:
                    raise ValueError(
                        "atac_norm_factor_file should have exactly two columns: cluster label and norm coef")
                self.norm_factors = norm_df[1].values
                self.has_acc_norm = True

            # load reference/groundtruth
            if "ref" in fh["dec"]:
                self.load_groundtruth = True

            # load prior matrix if there's one
            if "prior" in fh["dec"]:
                self.prior = fh["dec/prior"][:]

        self.y_length = y_length

        if self.y_length > self._t_x:
            raise ValueError("y_length must be shorter than or equal to self._t_x")
        self.y_truncation = (self._t_x - self.y_length) // 2
        if loads_trunc is not None:
            self.loads_trunc = loads_trunc
        else:
            self.loads_trunc = self.y_truncation

        if self.df.shape[1] < 4:
            msg = "region_file should have at least 4 cols: chr, start, end, and region_type. {self.df.head()}"
            raise ValueError(msg)

        if non_background_only:
            self.df = self.df.loc[self.df[3] == 1].copy()
        if enable_additional_filter is not None and self.df.shape[1] > 4:
            self.df = self.df.loc[self.df[4] == 1 if enable_additional_filter else 0, :].copy()
        if is_training == 1:
            v_set = set(chromosomal_val) if chromosomal_val is not None else set()
            t_set = set(chromosomal_test) if chromosomal_test is not None else set()
            vt_chromosomes = v_set.union(t_set)
            self.df = self.df.loc[~self.df[0].isin(vt_chromosomes), :]
            if isinstance(pos_only_subset, int) and pos_only_subset > 0:
                pos_df = self.df[self.df[3] == 1]
                if pos_only_subset > pos_df.shape[0]:
                    raise ValueError("Number of subset cannot be larger than the positive training set")
                self.df = pos_df.sample(
                    n=pos_only_subset, replace=False, random_state=subset_seed
                )
        elif is_training == 0 and chromosomal_val is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_val), :]
        elif is_training == 2 and chromosomal_test is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_test), :]
        self.df.reset_index(drop=False, inplace=True)

    @property
    def t_x(self):
        return self._t_x

    @property
    def t_y(self):
        return self.y_length

    @t_y.setter
    def t_y(self, value: int):
        raise NotImplementedError

    def get_weights(self, observations: torch.Tensor, small: float = 1e-16):
        safe_observations = small + observations
        weights = safe_observations / safe_observations.sum()
        return weights

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        """

        Parameters
        ----------
        idx : int
            data index

        Returns
        -------
        (seq, atac) : (torch.Tensor, torch.Tensor)
            seq: Input tensor of shape (4, self.t_x), rows are ordered by ACGT.
            atac: Input tensor of shape (self.n_clusters, self.t_x), Cluster-specific accessibility signal.
        counts : torch.Tensor
            Strand-specific read counts for each target.
            Shape: (self.n_targets)
        profiles : torch.Tensor
            Strand-specific counts profile for each target.
            Shape: (self.n_targets, self.t_y)
        groundtruth : Sequence[torch.Tensor]
            Each element in the list is a `torch.Tensor` for the ground truth for that cluster.
            Element shape: (self.n_targets, self.t_y)
            This will be returned if `self.load_groundtruth` is True and there's "ref" in the hdf5 file;
            otherwise, a constant zero will be returned.
        loads : torch.Tensor
            ATAC loads for each cluster.
            Shape: (self.n_clusters)
        misc : list
            chrom : str
            start : int
            end : int
            prior : torch.Tensor

        """
        if self.dataset is None:
            self.dataset = h5py.File(self.expected_data_file, "r")["dec"]

        hit = self.df.iloc[idx]
        abs_i = hit["index"]

        seq = torch.abs(torch.tensor(self.dataset["seq"][abs_i, :, :])).float()
        acc = torch.tensor(self.dataset["acc"][abs_i, :, :])

        if self.load_groundtruth and "ref" in self.dataset:
            ground_truth = []
            for cid in range(self.n_clusters):
                offset = self.n_targets * cid
                ground_truth.append(
                    torch.tensor(self.dataset["ref"][abs_i, offset:offset + self.n_targets,
                                 self.y_truncation:-self.y_truncation]).abs())
        else:
            ground_truth = []

        if self.y_truncation > 0:
            y = torch.tensor(self.dataset["bulk"][abs_i, :, self.y_truncation:-self.y_truncation]).abs()
        else:
            y = torch.tensor(self.dataset["bulk"][abs_i, :, :]).abs()

        loads = torch.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            if self.loads_trunc > 0:
                loads[i] = acc[i, self.loads_trunc:-self.loads_trunc].mean() * self.norm_factors[i]
            else:
                loads[i] = acc[i, :].mean() * self.norm_factors[i]

        misc_tuple = (hit[0], hit[1], hit[2], hit[3], self.prior)
        return (
            (seq, acc), y.sum(axis=1), y, ground_truth,
            self.get_weights(loads), misc_tuple)


class DynamicDataset(Dataset):
    """
    Dataset class for both one-hot encoded sequences and signal tracks
    """

    def __init__(self, fa_file: str, pl_bulk_bw_file: str, acc_bw_files: Sequence[str], regions_file: str,
                 mn_bulk_bw_file: Optional[str] = None, pl_ct_bw_files: Optional[Sequence[str]] = None,
                 mn_ct_bw_files: Optional[Sequence[str]] = None, sc_norm_file: Optional[str] = None,
                 cluster_names: Optional[Sequence[str]] = None, y_length: int = 1_000,
                 target_sliding_sum: Optional[int] = 0, t_x: int = 4096,
                 is_training: int = 1, chromosomal_val: Optional[Sequence[str]] = None,
                 chromosomal_test: Optional[Sequence[str]] = None, loads_trunc: Optional[int] = None,
                 pos_only_subset: Optional[int] = None, non_background_only: bool = False,
                 subset_seed: Optional[int] = None, enable_additional_filter: Optional[bool] = None,
                 use_bulk_constraint: Optional[bool] = False):
        """

        Parameters
        ----------
        fa_file : str
            {fa_file}
        pl_bulk_bw_file : str
            {bulk_pl}
        acc_bw_files : Sequence[str]
            {accessibility}
        regions_file : str
            {regions}
        mn_bulk_bw_file : Optional[str]
            {bulk_mn}
        pl_ct_bw_files : Optional[Sequence[str]]
            {ref_pls}
        mn_ct_bw_files : Optional[Sequence[str]]
            {ref_mns}
        sc_norm_file: Optional[str]
            {sc_norm_file}
        cluster_names : Optional[Sequence[str]]
            {ref_labels}
        y_length : int
            {y_length}
        target_sliding_sum : Optional[int]
            {target_sliding_sum}
        is_training : int
            {is_training}
        chromosomal_val : Optional[Sequence[str]]
            {chromosomal_validation}
        chromosomal_test : Optional[Sequence[str]]
            {chromosomal_testing}
        loads_trunc : Optional[int]
            {loads_trunc}
        pos_only_subset : Union[None,int]
            Set it as a positive integer if you want to sample a subset of positive regions in the training set.
            The size of the subset will be identical with the value of this argument. By default None
        non_background_only : bool
            Set it as True if you only want to use non-background regions
        subset_seed : Union[None,int]
            by default None
        enable_additional_filter : Optional[bool]
            Set it as True to select regions that are labeled as 1 in the 6th col.
            Set it as False to select regions that are labeled as 0 in the 6th col.
            Leave it as None if you don't want to apply this additional filter.
        use_bulk_constraint : Optional[bool]
            {use_bulk_constraint}

        Raises
        ------
        IOError
            If training npz data cannot be found
        ValueError

        """.format(**PARAM_DESC)
        assert os.path.exists(pl_bulk_bw_file)
        self.pl_bulk_bw_file = pl_bulk_bw_file
        if mn_bulk_bw_file is not None:
            assert os.path.exists(mn_bulk_bw_file)
        self.mn_bulk_bw_file = mn_bulk_bw_file

        assert os.path.exists(fa_file)
        self.fa_file = fa_file

        assert all([os.path.exists(f) for f in acc_bw_files])
        self.acc_bw_files = acc_bw_files

        assert os.path.exists(regions_file)
        self.df = pd.read_csv(regions_file, header=None, comment="#")

        if pl_ct_bw_files is not None:
            assert all([os.path.exists(f) for f in pl_ct_bw_files])
            self.load_groundtruth = True
        self.pl_ct_bw_files = pl_ct_bw_files
        if mn_ct_bw_files is not None:
            assert all([os.path.exists(f) for f in mn_ct_bw_files])
        self.mn_ct_bw_files = mn_ct_bw_files

        self.fa_obj = None
        self.pl_bulk_bw_obj = None
        self.acc_bw_objs = None

        self.mn_bulk_bw_obj = None
        self.pl_ct_bw_objs = None
        self.mn_ct_bw_objs = None

        self.sliding_sum = target_sliding_sum

        self.load_groundtruth = False
        self.has_acc_norm = False
        self.prior = torch.tensor(0)

        # automatically determined values
        self.n_targets = 1 if self.mn_bulk_bw_file is None else 2
        self.n_clusters = len(self.acc_bw_files)
        self.cluster_names = cluster_names if cluster_names is not None else [f"C{i}" for i in range(self.n_clusters)]
        if len(self.cluster_names) != self.n_clusters:
            raise ValueError("len(cluster_names) != n_clusters")

        self.norm_factors = np.ones(self.n_clusters)

        if sc_norm_file is not None:
            if os.path.exists(sc_norm_file):
                norm_df = pd.read_csv(sc_norm_file, header=None)
                norm_df.columns = (0, 1)
            else:
                raise IOError(f"{sc_norm_file} not found")
            if norm_df.shape[1] != 2:
                raise ValueError(
                    "atac_norm_factor_file should have exactly two columns: cluster label and norm coef")
            self.norm_factors = norm_df[1].values
            self.has_acc_norm = True

        self._t_x = t_x

        self.y_length = y_length

        if self.y_length > self._t_x:
            raise ValueError("y_length must be shorter than or equal to self._t_x")
        self.y_truncation = (self._t_x - self.y_length) // 2
        if loads_trunc is not None:
            self.loads_trunc = loads_trunc
        else:
            self.loads_trunc = self.y_truncation

        if self.df.shape[1] < 4:
            msg = "region_file should have at least 4 cols: chr, start, end, and region_type. {self.df.head()}"
            raise ValueError(msg)

        if use_bulk_constraint:
            self.bulk_constraint()

        if non_background_only:
            self.df = self.df.loc[self.df[3] == 1].copy()
        if enable_additional_filter is not None and self.df.shape[1] > 4:
            self.df = self.df.loc[self.df[4] == 1 if enable_additional_filter else 0, :].copy()
        if is_training == 1:
            v_set = set(chromosomal_val) if chromosomal_val is not None else set()
            t_set = set(chromosomal_test) if chromosomal_test is not None else set()
            vt_chromosomes = v_set.union(t_set)
            self.df = self.df.loc[~self.df[0].isin(vt_chromosomes), :]
            if isinstance(pos_only_subset, int) and pos_only_subset > 0:
                pos_df = self.df[self.df[3] == 1]
                if pos_only_subset > pos_df.shape[0]:
                    raise ValueError("Number of subset cannot be larger than the positive training set")
                self.df = pos_df.sample(
                    n=pos_only_subset, replace=False, random_state=subset_seed
                )
        elif is_training == 0 and chromosomal_val is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_val), :]
        elif is_training == 2 and chromosomal_test is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_test), :]
        self.df.reset_index(drop=False, inplace=True)

    @property
    def t_x(self):
        return self._t_x

    @property
    def t_y(self):
        return self.y_length

    @t_y.setter
    def t_y(self, value: int):
        raise NotImplementedError

    @staticmethod
    def safe_sum(bwo, row):
        raw_sum = bwo.stats(row[0], row[1], row[2], type="sum")[0]
        if raw_sum is None:
            raw_sum = 0.
        if raw_sum < 0:
            raw_sum *= -1
        return raw_sum

    def bulk_constraint(self):
        bulk_bws = [pyBigWig.open(self.pl_bulk_bw_file)]

        if self.mn_bulk_bw_file is not None:
            bulk_bws.append(pyBigWig.open(self.mn_bulk_bw_file))

        bulk_signal = self.df.apply(lambda x: sum([DynamicDataset.safe_sum(b, x) for b in bulk_bws]), axis=1)
        self.df.loc[bulk_signal == 0, 3] = 0

        for bw in bulk_bws: bw.close()

    def get_weights(self, observations: torch.Tensor, small: float = 1e-16):
        safe_observations = small + observations
        weights = safe_observations / safe_observations.sum()
        return weights

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        """

        Parameters
        ----------
        idx : int
            data index

        Returns
        -------
        (seq, atac) : (torch.Tensor, torch.Tensor)
            seq: Input tensor of shape (4, self.t_x), rows are ordered by ACGT.
            atac: Input tensor of shape (self.n_clusters, self.t_x), Cluster-specific accessibility signal.
        counts : torch.Tensor
            Strand-specific read counts for each target.
            Shape: (self.n_targets)
        profiles : torch.Tensor
            Strand-specific counts profile for each target.
            Shape: (self.n_targets, self.t_y)
        groundtruth : Sequence[torch.Tensor]
            Each element in the list is a `torch.Tensor` for the ground truth for that cluster.
            Element shape: (self.n_targets, self.t_y)
            This will be returned if `self.load_groundtruth` is True and there's "ref" in the hdf5 file;
            otherwise, a constant zero will be returned.
        loads : torch.Tensor
            ATAC loads for each cluster.
            Shape: (self.n_clusters)
        misc : list
            chrom : str
            start : int
            end : int
            prior : torch.Tensor

        """
        if self.pl_bulk_bw_obj is None:
            self.pl_bulk_bw_obj = pyBigWig.open(self.pl_bulk_bw_file)
            self.acc_bw_objs = [pyBigWig.open(f) for f in self.acc_bw_files]
        if self.mn_bulk_bw_file is not None and self.mn_bulk_bw_obj is None:
            self.mn_bulk_bw_obj = pyBigWig.open(self.mn_bulk_bw_file)
        if self.pl_ct_bw_files is not None and self.pl_ct_bw_objs is None:
            self.pl_ct_bw_objs = [pyBigWig.open(f) for f in self.pl_ct_bw_files]
        if self.mn_ct_bw_files is not None and self.mn_ct_bw_objs is None:
            self.mn_ct_bw_objs = [pyBigWig.open(f) for f in self.mn_ct_bw_files]

        if self.fa_obj is None:
            self.fa_obj = pyfaidx.Fasta(self.fa_file)

        hit = self.df.iloc[idx]

        chrom, start, end = hit[0], hit[1], hit[2]
        raw_seq = self.fa_obj[chrom][start:end].seq.upper()
        seq = torch.abs(torch.tensor(seq_to_one_hot(raw_seq))).float()

        acc = torch.stack([torch.tensor(extract_signal_from_bw(bo, chrom, start, end, sliding_sum=self.sliding_sum)) for bo in self.acc_bw_objs])
        # acc = torch.tensor(self.dataset["acc"][abs_i, :, :])

        if self.pl_ct_bw_files:
            ground_truth = []
            for cid in range(self.n_clusters):
                gt_raw = [torch.tensor(extract_signal_from_bw(self.pl_ct_bw_objs[cid], chrom, start, end, sliding_sum=self.sliding_sum)),]
                if self.mn_ct_bw_objs:
                    gt_raw.append(torch.tensor(extract_signal_from_bw(self.mn_ct_bw_objs[cid], chrom, start, end, sliding_sum=self.sliding_sum)))
                ref = torch.stack(gt_raw)
                ground_truth.append(ref[:, self.y_truncation:-self.y_truncation].abs())
        else:
            ground_truth = []

        raw_bulk = [torch.tensor(extract_signal_from_bw(self.pl_bulk_bw_obj, chrom, start, end, sliding_sum=self.sliding_sum))]
        if self.mn_bulk_bw_obj:
            raw_bulk.append(torch.tensor(extract_signal_from_bw(self.mn_bulk_bw_obj, chrom, start, end, sliding_sum=self.sliding_sum)))
        bulk = torch.stack(raw_bulk)
        if self.y_truncation > 0:
            y = bulk[:, self.y_truncation:-self.y_truncation].clone().abs()
        else:
            y = torch.tensor(bulk).abs()

        loads = torch.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            if self.loads_trunc > 0:
                loads[i] = acc[i, self.loads_trunc:-self.loads_trunc].mean() * self.norm_factors[i]
            else:
                loads[i] = acc[i, :].mean() * self.norm_factors[i]

        misc_tuple = (hit[0], hit[1], hit[2], hit[3], self.prior)
        return (
            (seq, acc), y.sum(axis=1), y, ground_truth,
            self.get_weights(loads), misc_tuple)


class MultiTaskSupervisedDataset(Dataset):
    """
    Dataset class for both one-hot encoded sequences and signal tracks
    """
    def __init__(self, region_bed: str, fa_file: str, 
                 acc_bw_files: Sequence[str],
                 pl_ct_bw_files: Sequence[str],
                 mn_ct_bw_files: Sequence[str],
                 cell_types: Sequence[str],
                 is_training: int = 1,
                 t_x: int = 4096, y_length: int = 1000,
                 chromosomal_val: Sequence[str] = ("chr22",),
                 chromosomal_test: Sequence[str] = ("chr19",),
                 pos_only_subset: Optional[int] = None,
                 subset_seed: Optional[int] = None,
                 non_background_only: bool = False):
        """
        Parameters
        ----------
        region_bed : str
            Path to the region bed file
        fa_file : str
            Path to the genome fasta file
        acc_bw_files : Sequence[str]
            List of paths to the accessibility bw files
        pl_ct_bw_files : Sequence[str]
            List of paths to the positive strand profile bw files
        mn_ct_bw_files : Sequence[str]
            List of paths to the negative strand profile bw files
        cell_types : Sequence[str]
            List of cell types corresponding to the bw files
        is_training : int
            0 : validation, 1 : training, 2 : testing, by default 1
        t_x : int
            Size of the window to extract sequences and signals
        y_length : int
            Length of the target signal to extract
        chromosomal_val : Optional[Sequence[str]]
            Chromosomes to use for validation
        chromosomal_test : Optional[Sequence[str]]
            Chromosomes to use for testing
        pos_only_subset : Optional[int]
            Set it as a positive integer if you want to sample a subset of positive regions in the training set.
            The size of the subset will be identical with the value of this argument. By default None
        subset_seed : Optional[int]
            Random seed for sampling the subset of positive regions
        non_background_only : bool
            Set it as True if you only want to use non-background regions
        """
        # Region set
        self.df = pd.read_csv(region_bed, sep="\t", header=None, usecols=[0, 1, 2])
        self.cell_types = list(cell_types)

        assert all([os.path.exists(f) for f in acc_bw_files])
        self.acc_bw_files = acc_bw_files
        assert all([os.path.exists(f) for f in pl_ct_bw_files])
        self.pl_ct_bw_files = pl_ct_bw_files
        assert all([os.path.exists(f) for f in mn_ct_bw_files])
        self.mn_ct_bw_files = mn_ct_bw_files

        # Input/output sizes
        self._t_x = t_x
        self.y_length = y_length
        if self.y_length > self._t_x:
            raise ValueError("y_length must be shorter than or equal to self.window_size")
        self.y_offset = (self._t_x - self.y_length) // 2
        self.n_targets = 1 if self.mn_ct_bw_files is None else 2
        self.n_cell_types = len(self.acc_bw_files)

        # Lazy-loaded file handles
        self.fa_file = fa_file
        self.fa_obj = None
        self.atac_bw_objs = None
        self.profile_bw_objs = None

        # Filter regions
        if non_background_only:
            # self.df = self.df.loc[self.df[3] == 1].copy()
            raise NotImplementedError("Non-background filtering is not implemented yet.\
                                       Now no GC background is applied.")
        if is_training == 1:
            v_set = set(chromosomal_val) if chromosomal_val is not None else set()
            t_set = set(chromosomal_test) if chromosomal_test is not None else set()
            vt_chromosomes = v_set.union(t_set)
            self.df = self.df.loc[~self.df[0].isin(vt_chromosomes), :]
            if pos_only_subset is not None and isinstance(pos_only_subset, int) and pos_only_subset > 0:
                if 3 not in self.df.columns:
                    raise ValueError("Fourth column (positivity label) required for pos_only_subset.")
                pos_df = self.df[self.df[3] == 1]
                if pos_only_subset > pos_df.shape[0]:
                    raise ValueError("Requested subset size exceeds available positives.")
                self.df = pos_df.sample(n=pos_only_subset, replace=False, random_state=subset_seed)

        elif is_training == 0 and chromosomal_val is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_val), :]

        elif is_training == 2 and chromosomal_test is not None:
            self.df = self.df.loc[self.df[0].isin(chromosomal_test), :]

        self.df.reset_index(drop=True, inplace=True)

    @property
    def t_x(self):
        return self._t_x

    @property
    def t_y(self):
        return self.y_length

    @t_y.setter
    def t_y(self, value: int):
        raise NotImplementedError
    
    def get_weights(self, observations: torch.Tensor, small: float = 1e-16):
        safe_observations = small + observations
        weights = safe_observations / safe_observations.sum()
        return weights
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            data index

        Returns
        -------
        (seq, atac) : (torch.Tensor, torch.Tensor)
            seq: Input tensor of shape (4, self.t_x), rows are ordered by ACGT.
            atac: Input tensor of shape (self.n_cell_types, self.t_x), Cell type-specific accessibility signal.
        counts : torch.Tensor
            Strand-specific read counts for each target.
            Shape: (self.n_cell_types, self.n_targets)
        profiles : torch.Tensor
            Strand-specific counts profile for each target.
            Shape: (self.n_cell_types, self.n_targets, self.t_y)
        loads : torch.Tensor
            ATAC loads for each cluster.
            Shape: (self.n_cell_types)
        """
        # Lazy-load I/O objects
        if self.fa_obj is None:
            self.fa_obj = pyfaidx.Fasta(self.fa_file)

        if self.atac_bw_objs is None:
            self.atac_bw_objs = {
                ct: pyBigWig.open(f) for ct, f in zip(self.cell_types, self.acc_bw_files)
            }

        if self.profile_bw_objs is None:
            self.profile_bw_objs = {
                ct: (pyBigWig.open(pl), pyBigWig.open(mn)) 
                for ct, (pl, mn) in zip(self.cell_types, zip(self.pl_ct_bw_files, self.mn_ct_bw_files))
            }

        hit = self.df.iloc[idx]
        chrom, start, end = hit[0], hit[1], hit[2]

        center = (start + end) // 2

        window_start = center - self.t_x // 2
        window_end = window_start + self.t_x
        y_start = window_start + self.y_offset
        y_end = window_end - self.y_offset

        # Sequence
        seq = self.fa_obj[chrom][window_start:window_end].seq
        seq_tensor = torch.tensor(seq_to_one_hot(seq), dtype=torch.float32)  # (4, window_size)

        # ATAC signals: stack per cell type
        atac_tensor = torch.stack([
            torch.tensor(extract_signal_from_bw(self.atac_bw_objs[ct], chrom, window_start, window_end)) 
            for ct in self.cell_types])  # (n_cell_types, 1, window_size)

        # ATAC loads
        loads = torch.zeros(self.n_cell_types)
        for i in range(self.n_cell_types):
            loads[i] = atac_tensor[i, :].mean()

        # Profiles: stack per cell type
        profile_tensors = []
        counts_tensors = []
        for ct in self.cell_types:
            pl_bw, mn_bw = self.profile_bw_objs[ct]
            profile = torch.stack([torch.tensor(extract_signal_from_bw(pl_bw, chrom, y_start, y_end)),
                      torch.tensor(extract_signal_from_bw(mn_bw, chrom, y_start, y_end))]) # (2, y_length)

            profile_tensors.append(profile.clone().detach())
            counts_tensors.append(profile.sum(axis=1).clone().detach())  # (2,)

        profiles_tensor = torch.stack(profile_tensors, dim=0)  # (n_cell_types, 2, y_length)
        counts_tensor = torch.stack(counts_tensors, dim=0)     # (n_cell_types, 2)

        return (seq_tensor, atac_tensor), counts_tensor, profiles_tensor, self.get_weights(loads)
