import gzip
import os
import h5py
import pybedtools
import pyBigWig
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional, List, Sequence
from tqdm import tqdm
from deepdetails.__about__ import __version__
from deepdetails.par_description import PARAM_DESC
from deepdetails.helper.utils import slugify, set_tmp_for_pbt, bedgraph_to_bigwig


def _midpoint_generator(bed_regions: pybedtools.BedTool):
    """

    Parameters
    ----------
    bed_regions : pybedtools.BedTool

    Returns
    -------

    """
    from pybedtools.featurefuncs import midpoint
    try:
        for region in bed_regions:
            yield midpoint(region)
    except Exception as e:
        print(e)


def extend_regions_from_mid_points(region: Union[str, pybedtools.BedTool, pd.DataFrame],
                                   extensions: Tuple[int, int], chromosome_size: str) -> pd.DataFrame:
    """
    Extend regions from their middle points

    Parameters
    ----------
    region : str or `pybedtools.BedTool` or `pd.DataFrame`
        Path to the region bed file, or a `BedTool` instance, or a `DataFrame` instance
    extensions : tuple of ints
        Two ints, first one for the upstream extension, the second one for the downstream extension
    chromosome_size : str
        Path to the chromosome size file or a name of genome release, like `hg38`.

    Returns
    -------
    extended_regions : `pd.DataFrame`
        Extended regions
    """
    if isinstance(region, pybedtools.BedTool):
        bed_obj = region
    elif isinstance(region, pd.DataFrame):
        bed_obj = pybedtools.BedTool.from_dataframe(region)
    elif isinstance(region, str) and os.path.exists(region):
        bed_obj = pybedtools.BedTool(region)
    else:
        raise ValueError("region is not supported")

    if not isinstance(chromosome_size, str):
        raise ValueError("chromosome_size must be a string")

    mid_points = pybedtools.BedTool(_midpoint_generator(bed_obj))
    if os.path.exists(chromosome_size) and os.path.isfile(chromosome_size):
        extended_regions = mid_points.slop(l=extensions[0], r=extensions[1], g=chromosome_size)
    else:
        extended_regions = mid_points.slop(l=extensions[0], r=extensions[1], genome=chromosome_size)
    return extended_regions.to_dataframe(disable_auto_names=True, header=None)


def combine_regions(region_files: Union[tuple[str,], list[str]], allowed_chrs: Optional[set] = None,
                    merge_overlap: int = 0) -> pd.DataFrame:
    """Combine regions defined in multiple files

    Parameters
    ----------
    region_files : Union[tuple[str, ...], list[str]]
        Path to the regions files, e.g. bidirectional and unidirectional peak files
    allowed_chrs : Optional[set]
        A set of allowed chromosomes. If empty, filters will not be applied.
    merge_overlap : int
        A non-negative value defining the minimum overlap between features for them to be
        merged. By default, merging is not enabled (0).

    Returns
    -------
    pd.DataFrame
        Combined regions in a DataFrame with three columns
    """
    if allowed_chrs is None:
        allowed_chrs = {}
    sub_dfs = [
        pd.read_csv(rf, sep="\t", header=None, usecols=[0, 1, 2]) for rf in region_files if os.path.exists(rf)
    ]
    df = pd.concat(sub_dfs, ignore_index=True).sort_values([0, 1])
    df = df.loc[
        (~df[0].str.contains("_")) & (~df[0].str.contains("EBV")) & (~df[0].str.contains("chrM"))
        ].copy().reset_index(drop=True)
    # drop identical regions if any
    df.drop_duplicates(subset=[0, 1, 2], inplace=True)
    if len(allowed_chrs) > 0:
        df = df.loc[df[0].isin(allowed_chrs), :]
    if merge_overlap > 0:
        df = pybedtools.BedTool.from_dataframe(df).merge(
            d=-merge_overlap).to_dataframe(disable_auto_names=True, header=None)

    return df


def generate_gc_matched_random_regions(
        input_region_file: Union[str, pd.DataFrame, pybedtools.BedTool], genome_size_file: str,
        genome_fasta_file: str, sample_scale_factor: float = 1.,
        seed: Optional[int] = None, mkwindows_stride: int = 1000, bins: int = 100,
        blacklist: Optional[str] = None, chrom_starts_with: str = "chr",
        dist_compare_plot_file: Optional[str] = None):
    """
    Generate GC content matched random regions for the input file

    Parameters
    ----------
    input_region_file : str or `pybedtools.BedTool` or `pd.DataFrame`
        Path to a bed file with n regions or a BedTool/DataFrame instance
    genome_size_file : str
        Either a path to a tab-separated file defining the size of each chromosome
        or the name of the genome build that's included in pybedtools' database
    genome_fasta_file : str
        Path to a fasta file which contains the sequence for each chromosome
    sample_scale_factor : float
        Scale factor, this algorithm tries to find sample_scale_factor times of input_region.shape[0]
        GC-matched regions. Assume there are `n` inputs in `input_region_file` then this function
        will yield about `n`*`sample_scale_factor` candidates.
    seed : Optional[int]
        Seed for sampling. If it's None, the algorithm uses timestamps for sampling.
    mkwindows_stride : None or int
        Stride for makewindows
    bins : int

    blacklist : Optional[str]
        Additional regions that should be excluded from the search space.
        Note: regions in `input_region_file` will automatically be excluded.
    chrom_starts_with : str
        Require candidate chromosomes to have this prefix
    dist_compare_plot_file : Optional[str]


    Returns
    -------
    matched_random_regions : pd.DataFrame
        Sampled regions (n) stored in a four-column DataFrame.
        0~2: coordinates
        3: GC%
    """
    __interval_key__ = "{0}:{1}-{2}"
    pybedtools.set_tempdir(".")

    if type(input_region_file) is pybedtools.BedTool:
        bed_obj = pybedtools.BedTool.from_dataframe(
            pd.read_csv(input_region_file.fn, sep="\t",
                        header=None, usecols=[0, 1, 2])
        )
    elif isinstance(input_region_file, pd.DataFrame):
        bed_obj = pybedtools.BedTool.from_dataframe(
            input_region_file[input_region_file.columns[:3]])
    else:
        bed_obj = pybedtools.BedTool.from_dataframe(
            pd.read_csv(input_region_file, sep="\t", header=None, usecols=[0, 1, 2]))

    bed_obj = bed_obj.filter(lambda x: len(x) > 0).saveas()
    # pylint: disable-next=unexpected-keyword-arg
    region_content_df = bed_obj.nucleotide_content(fi=genome_fasta_file).saveas().to_dataframe(
        names=("chrom", "start", "end", "pct_at", "pct_gc", "num_A",
               "num_C", "num_G", "num_T", "num_N", "num_oth", "seq_len"),
        comment="#")

    assert region_content_df.shape[1] == 12
    act_bins, bin_crit = pd.cut(
        region_content_df["pct_gc"], bins=bins, retbins=True, labels=np.arange(bins))
    per_bin_sampling_target = (
            act_bins.value_counts() * sample_scale_factor).to_dict()

    n_seq_lens = region_content_df["seq_len"].value_counts()

    if n_seq_lens.shape[0] == 1:
        # if lengths of input sequences are homogenous, then we use
        # makewindows to replace the sampling process for better performance
        # genomic windows overlap with input regions will be removed
        length = n_seq_lens.index.values[0]
        candidates = pybedtools.BedTool().makewindows(
            g=genome_size_file, w=length, s=mkwindows_stride
        ) if os.path.exists(genome_size_file) else pybedtools.BedTool().makewindows(
            genome=genome_size_file, w=length, s=mkwindows_stride
        ).saveas()
        # remove windows that are shorter than the requested length
        # this can happen when the windows are near the ends of chromosomes
        candidates = candidates.filter(
            lambda x: len(x) == length and x[0].startswith(chrom_starts_with)
        ).saveas()
        if isinstance(blacklist, str) and os.path.exists(blacklist):
            candidates = candidates.intersect(
                pybedtools.BedTool(blacklist), v=True).saveas()
        candidate_scope = candidates.sort().to_dataframe().drop_duplicates()
    else:
        raise ValueError("This function requires all regions to have identical length, "
                         "please use `generate_gc_matched_random_regions` instead")

    # remove regions that are not from primary assembly
    candidate_scope = candidate_scope.loc[
                      (candidate_scope.chrom.str.find("_") == -1) & (candidate_scope.chrom != "chrEBV"),
                      :]
    candidate_bed = pybedtools.BedTool.from_dataframe(candidate_scope)
    # pylint: disable-next=unexpected-keyword-arg,too-many-function-args
    candidate_regions = candidate_bed.intersect(bed_obj, v=True)

    # analyze nucleotide composition of these candidate regions
    # pylint: disable-next=unexpected-keyword-arg
    nucleotide_contents = candidate_regions.nucleotide_content(fi=genome_fasta_file).saveas().to_dataframe(
        names=("chrom", "start", "end",
               "pct_at", "pct_gc", "num_A",
               "num_C", "num_G", "num_T",
               "num_N", "num_oth", "seq_len"),
        comment="#")

    # remove regions with masked nts
    # nucleotide_contents = nucleotide_contents.loc[nucleotide_contents.num_N == 0, :]
    nucleotide_contents["bin"] = pd.cut(
        nucleotide_contents["pct_gc"], bins=bin_crit, labels=np.arange(bins))

    sampled_results = nucleotide_contents.groupby("bin").apply(
        lambda x: x.sample(
            min(int(per_bin_sampling_target[x.name]), x.shape[0]),
            random_state=seed)
    ).reset_index(drop=True)
    sampled_results = sampled_results[[
        "chrom", "start", "end", "pct_gc"]].copy()

    if dist_compare_plot_file is not None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use("Agg")
            import seaborn as sns
            ref = region_content_df[["pct_gc"]].copy()
            ref["label"] = "Reference"
            sampled = sampled_results[["pct_gc"]].copy()
            sampled["label"] = "Sampled"
            sns.kdeplot(data=pd.concat([ref, sampled], ignore_index=True), x="pct_gc", hue="label", common_norm=False)
            plt.savefig(dist_compare_plot_file, dpi=200)
            plt.close()
        except Exception as e:
            print(e)
    sampled_results.columns = (0, 1, 2, 3)
    return sampled_results


def seq_to_one_hot(x: str, non_standard_as_zero: bool = False) -> np.ndarray:
    """One-hot encoding a DNA sequence

    For non-ACGT IUPAC nucleotide symbols, the algorithm uses fraction scores

    Parameters
    ----------
    x : str
        DNA sequence
    non_standard_as_zero : bool, optional
        Whether to convert non-ACGT symbols as zeros, by default False

    Returns
    -------
    mat: np.ndarray
        np.array with a shape of (4, seq_len)
    """
    alphabet = "ACGT"
    iupac_ext = {
        "R": ("A", "G"),
        "Y": ("C", "T"),
        "S": ("G", "C"),
        "W": ("A", "T"),
        "K": ("G", "T"),
        "M": ("A", "C"),
        "B": ("C", "G", "T"),
        "D": ("A", "G", "T"),
        "H": ("A", "C", "T"),
        "V": ("A", "C", "G")
    }
    mat = np.zeros((len(alphabet), len(x)), dtype=float)
    x_treated = x.upper().replace("U", "T")
    for i in range(len(x_treated)):
        hit = x_treated[i]
        try:
            mat[alphabet.index(hit), i] = 1
        except ValueError:
            if not non_standard_as_zero and hit in iupac_ext:
                subs = iupac_ext[hit]
                for sub in subs:
                    mat[alphabet.index(sub), i] = 1 / len(subs)
    return mat


def extract_signal_from_bw(bw_obj: pyBigWig.pyBigWig, chrom: str, start: int, end: int,
                           apply_abs: bool = True, sliding_sum: int = 0) -> np.ndarray:
    """
    Extract signal values from a pyBigWig object, NaN values will be replaced with 0.

    Parameters
    ----------
    bw_obj : pyBigWig.pyBigWig
        pyBigWig object for signal extraction
    chrom : str
        name of the chromosome
    start : int
        start location of the window
    end : int
        end location of the window
    sliding_sum : int
        Apply sliding sum to the signal? If so, the size of the sliding window
    apply_abs : bool
        Whether to apply absolute function to the output

    Returns
    -------
    values : np.ndarray
        Extracted signals
    """
    if sliding_sum > 0:  # calculate sliding sum
        # shift the start coordinate and retrieve a longer sequence to make sure after calculating the sliding sum
        # we have identical number of observations as end - start
        signal_shifting = sliding_sum - 1
        values = np.lib.stride_tricks.sliding_window_view(
            np.nan_to_num(
                bw_obj.values(chrom, start - signal_shifting, end, numpy=True)
            ),
            sliding_sum).sum(axis=-1)
    else:
        values = np.nan_to_num(
            bw_obj.values(chrom, start, end, numpy=True)
        )

    return np.abs(values) if apply_abs else values


def check_bw_input(input_lst: List[Union[pyBigWig.pyBigWig, str]]):
    """
    Check BigWig input

    Parameters
    ----------
    input_lst : List[Union[pyBigWig.pyBigWig, str]]
        input files to be checked

    Returns
    -------

    """
    for i, v in enumerate(input_lst):
        if isinstance(v, str):
            if os.path.exists(v):
                input_lst[i] = pyBigWig.open(v)
            else:
                raise IOError("Cannot access file {}".format(v))
        elif not isinstance(v, pyBigWig.pyBigWig):
            raise ValueError(
                "Values in targets must be paths to bigwig files or pyBigWig instances")


def build_data_volume(regions: pd.DataFrame, target_pl_bws: List[Union[pyBigWig.pyBigWig, str]],
                      target_mn_bws: List[Union[pyBigWig.pyBigWig, str]],
                      accessibility_tracks: List[Union[pyBigWig.pyBigWig, str]],
                      t_x: int, save_to: str, sequence_fasta: str,
                      ref_pl_tracks: Optional[List[Union[pyBigWig.pyBigWig, str]]] = (),
                      ref_mn_tracks: Optional[List[Union[pyBigWig.pyBigWig, str]]] = (),
                      target_sliding_sum: int = 0):
    """
    Build data volumes for training

    Parameters
    ----------
    regions : pd.DataFrame
        three required columns: 0, 1, 2
    target_pl_bws : List[Union[pyBigWig.pyBigWig, str]]
        bulk signal (path to the file or pyBigWig object) for the forward strand
    target_mn_bws : List[Union[pyBigWig.pyBigWig, str]]
        bulk signal (path to the file or pyBigWig object) for the reverse strand
    accessibility_tracks : List[Union[pyBigWig.pyBigWig, str]]
        Accessibility (path to the file or pyBigWig object) for each psuedo-bulk cluster
    t_x : int
        Output (prediction) length
    save_to : str
        Save outputs to this folder
    sequence_fasta : str
        Genome fasta file
    ref_pl_tracks : Optional[List[Union[pyBigWig.pyBigWig, str]]]
        Reference/groundtruth signal for each cluster (+)
    ref_mn_tracks : Optional[List[Union[pyBigWig.pyBigWig, str]]]
        Reference/groundtruth signal for each cluster (-)
    target_sliding_sum : int
        Apply sliding windows (sum) to the input signal if the value is greater than 0.

    Returns
    -------

    """
    n_clusters = len(accessibility_tracks)
    accessibility_bws = list(accessibility_tracks)

    check_bw_input(target_pl_bws)
    check_bw_input(target_mn_bws)
    check_bw_input(accessibility_bws)

    with_ref = False

    if len(target_mn_bws) == 0:
        n_targets = 1
    elif len(target_mn_bws) == 1:
        n_targets = 2
    else:
        raise ValueError("target_mn_bws should have 0 (strandless) or 1 (stranded) values")

    if len(ref_pl_tracks) > 0:
        check_bw_input(ref_pl_tracks)
        check_bw_input(ref_mn_tracks)
        with_ref = True

    with h5py.File(f"{save_to}/data.h5", "w") as fh:
        # save meta info
        fh.attrs["version"] = __version__

        # save region info
        # - chromosome information is coded as object, which cannot directly write into the ds
        # - convert them as integers and write the mappings as attributes
        transformed_regions = regions.copy()
        chr_mapping = {v: k for k, v in enumerate(regions[0].unique())}
        transformed_regions[0] = transformed_regions[0].map(chr_mapping)
        dset_rg = fh.create_dataset("regions", data=transformed_regions.to_numpy())
        for chr_str, chr_idx in chr_mapping.items():
            dset_rg.attrs[f"chr_{chr_idx}"] = chr_str

        gr = fh.create_group("dec")
        # write meta info
        gr.attrs["t_x"] = t_x  # input length
        gr.attrs["n_clusters"] = n_clusters  # num of clusters
        gr.attrs["n_targets"] = n_targets  # num of clusters
        dset_seq = gr.create_dataset("seq", (regions.shape[0], 4, t_x),
                                     dtype="f", chunks=(1, 4, t_x), compression="gzip")
        dset_acc = gr.create_dataset("acc", (regions.shape[0], n_clusters, t_x),
                                     dtype="f", chunks=(1, n_clusters, t_x), compression="gzip")
        dset_bulk = gr.create_dataset("bulk", (regions.shape[0], n_targets, t_x),
                                      dtype="f", chunks=(1, n_targets, t_x), compression="gzip")

        if with_ref:
            dset_ref = gr.create_dataset("ref", (regions.shape[0], n_targets * n_clusters, t_x),
                                         dtype="f", chunks=(1, n_targets * n_clusters, t_x), compression="gzip")
        else:
            dset_ref = None

        epig_fa_df = pybedtools.BedTool.from_dataframe(regions).nuc(
            fi=sequence_fasta,
            seq=True).to_dataframe(disable_auto_names=True, header=None, skiprows=1)

        for i, row in tqdm(regions.iterrows(), total=regions.shape[0], disable=True):
            dset_seq[i, :, :] = seq_to_one_hot(
                epig_fa_df.loc[i, epig_fa_df.shape[1] - 1])

            try:
                dset_bulk[i, 0, :] = extract_signal_from_bw(
                    target_pl_bws[0], row[0], row[1], row[2], sliding_sum=target_sliding_sum)
                if n_targets == 2:
                    dset_bulk[i, 1, :] = extract_signal_from_bw(
                        target_mn_bws[0], row[0], row[1], row[2], sliding_sum=target_sliding_sum)

                for row_idx in range(n_clusters):
                    dset_acc[i, row_idx, :] = extract_signal_from_bw(
                        accessibility_bws[row_idx], row[0], row[1], row[2], sliding_sum=0)

                    if with_ref:
                        dset_ref[i, n_targets * row_idx, :] = extract_signal_from_bw(
                            ref_pl_tracks[row_idx], row[0], row[1], row[2], sliding_sum=target_sliding_sum)
                        if n_targets == 2:
                            dset_ref[i, n_targets * row_idx + 1, :] = extract_signal_from_bw(
                                ref_mn_tracks[row_idx], row[0], row[1], row[2], sliding_sum=target_sliding_sum)
            except RuntimeError as e:
                raise ValueError(
                    "Error happened when processing {0}:{1}-{2}".format(
                        row[0], row[1], row[2])
                ) from e


def bed_to_cov_bw(in_bed_path: str, out_bigwig_path: str, chrom_size_path: str,
                  rpm_norm: Optional[int] = None, report_5p_cov: bool = False,
                  limit_strand_to: Optional[str] = None):
    """
    Convert a file in bed format to bigWig format (coverage)

    Parameters
    ----------
    in_bed_path : str

    out_bigwig_path : str

    chrom_size_path : str

    rpm_norm : Optional[int]
        Give the total read counts to enable RPM normalization
    report_5p_cov : bool
        Set it as True if you only want 5' signal
    limit_strand_to : Optional[str]
        +: forward strand
        -: reverse strand
        None: no limitations

    Returns
    -------

    """
    set_tmp_for_pbt()

    # make sure chromosome sets are consistent
    _chr_df = pd.read_csv(chrom_size_path, sep="\t", header=None)
    _chr_df[2] = 0
    chrom_ranges = pybedtools.BedTool.from_dataframe(_chr_df[[0, 2, 1]])
    in_bed = pybedtools.BedTool(in_bed_path).intersect(chrom_ranges, u=True)

    gc_kwargs = {"bg": True, "g": chrom_size_path}
    if report_5p_cov:
        gc_kwargs["5"] = True
    if rpm_norm is None:
        gc_kwargs["scale"] = 1.
    else:
        gc_kwargs["scale"] = 1000. * 1000. / rpm_norm
    if limit_strand_to is not None:
        gc_kwargs["strand"] = limit_strand_to
    gc_bed = in_bed.genome_coverage(**gc_kwargs).sort(g=chrom_size_path)

    bedgraph_to_bigwig(gc_bed.fn, out_bigwig_path, chrom_size_path)

    del gc_bed


def frag_to_cut_sites(frag_df: Union[pd.DataFrame, str], save_to: str, chrom_size: str, window_size: int = 150,
                      dual_directions: bool = False):
    """Convert fragments to cut sizes

    Parameters
    ----------
    frag_df : Union[pd.DataFrame, str]

    save_to : str

    chrom_size: str

    window_size : int
        Window size for cute sites. Set as 150 to mimic the ENCODE ATAC-seq pipeline.
        Set as 400 to mimic the CellRanger ATAC (CRA) pipeline
    dual_directions : bool
        If True, the function will extract cut sites from
        both the start and end sides of the fragments, this mimics
        CRA's behavior.

    Returns
    -------

    """
    set_tmp_for_pbt()

    half_window = window_size // 2
    if isinstance(frag_df, pd.DataFrame):
        frag_bed = pybedtools.BedTool.from_dataframe(frag_df)
    else:
        frag_bed = pybedtools.BedTool(frag_df)
    start_side = frag_bed.flank(l=half_window, r=0, g=chrom_size).slop(r=half_window, l=0, g=chrom_size)
    if dual_directions:
        end_side = frag_bed.flank(r=half_window, l=0, g=chrom_size).slop(l=half_window, r=0, g=chrom_size)
        pybedtools.BedTool.cat(
            *[start_side, end_side], postmerge=False
        ).sort().saveas(save_to)
    else:
        start_side.sort().saveas(save_to)


def frag_file_to_bw(fragment_file: str, frag_proc: str, chrom_size: str, n_frags: int,
                    rpm_norm: bool = False, save_to_workdir=False) -> str:
    """

    Parameters
    ----------
    fragment_file : str
        A compressed tsv file storing fragments.
    frag_proc : str
        Additional processes to the fragment file. Allowed types:
        * naive: no additional processing
        * encode: extend 150 bp from the cut site
        * cellranger: extend 400 bp from the cut site
        * 5pi: only the 5' insert site
    chrom_size : str
        A tab-separated file with chromosome sizes.
    n_frags : int
        Number of fragments in the fragment file. Affective only when rpm_norm is True.
    rpm_norm : bool, optional
        Whether to store RPM-normalized signal in the bigWig output
    save_to_workdir : bool, optional
        Force saving the bigWig output to the working directory; otherwise, it will be written to the
        folder containing the input fragment_file.

    Returns
    -------
    save_to : str
        Saved bigWig file
    """
    report_5p_cov = False
    if frag_proc == "naive":
        csb_file = fragment_file
    elif frag_proc == "encode":
        csb_file = fragment_file.replace(".tsv.gz", ".cs.bed").replace(".tsv", ".cs.bed")
        frag_to_cut_sites(fragment_file, csb_file, chrom_size, 150, False)
    elif frag_proc == "cellranger":
        csb_file = fragment_file.replace(".tsv.gz", ".cs.bed").replace(".tsv", ".cs.bed")
        frag_to_cut_sites(fragment_file, csb_file, chrom_size, 400, True)
    elif frag_proc == "5pi":  # 5' insert sites
        csb_file = fragment_file
        report_5p_cov = True
    else:
        raise ValueError(f"Unsupported {frag_proc}")

    if save_to_workdir:
        _, tmp = os.path.split(fragment_file)
    else:
        tmp = fragment_file
    save_to = tmp.replace(".tsv.gz", ".bw").replace(".tsv", ".bw")
    if chrom_size is not None:
        bed_to_cov_bw(
            csb_file, save_to, chrom_size,
            rpm_norm=n_frags if rpm_norm else None,
            report_5p_cov=report_5p_cov)
    return save_to


def convert_bulk_frags_to_ct_frags(fragments_file: str, barcode_file: str, save_to: str,
                                   reference_labels: Optional[Sequence[str]] = None,
                                   memory_saving: Optional[bool] = False) -> tuple[dict, dict, list, pd.DataFrame]:
    """
    Convert fragments file to bigWig files for each cell type/cluster

    Parameters
    ----------
    fragments_file : str
        Fragments tsv file. The first four columns should be:
            1. chromosome
            2. start position
            3. end position
            4. barcode
    barcode_file : str
        A tsv file storing barcodes. Two columns:
            1. barcode
            2. cell type
    save_to : str
        Save all outputs to this folder.
    chrom_size : str
        A tab-separated file storing chromosome sizes
    frag_proc : str
        How to process fragments file. Allowed types:
        * naive: no additional processing
        * encode: extend 150 bp from the cut site
        * cellranger: extend 400 bp from the cut site
        * 5pi: only the 5' insert site
    keep_frags : bool
        {keep_frags}
    reference_labels : Optional[Sequence[str]]
        Reference labels for the clusters/cell types. If not provided, labels will be extracted from the barcode file.
    memory_saving : Optional[bool]
        {memory_saving}

    Returns
    -------
    out_files : dict
        key: path-safe cluster/cell-type name
        value: path to the fragment file for each cell type/cluster
    frags_per_ct : dict
        key: path-safe cluster/cell-type name
        value: fragments this cell type/cluster has
    safe_cell_types : list
        Transformed cell types that can be safely used as part of the file names
    transformed_barcodes : pd.DataFrame
        column 0: barcode
        column 1: transformed cell type label
    """.format(**PARAM_DESC)
    barcodes = pd.read_csv(barcode_file, sep="\t", header=None)
    if barcodes.shape[1] != 2:
        raise ValueError("barcode_file should have 2 columns: the cell barcode and cell type annotation")
    cell_types = sorted(barcodes[1].unique().tolist())
    print(f"Cell types in the barcode file: {cell_types}")
    safe_cell_types_mapping = {ct: slugify(ct) for ct in cell_types}
    # print(f"Safe cell types mapping: {safe_cell_types_mapping}")
    if reference_labels is not None:
        safe_ref_labels = [slugify(r) for r in reference_labels]
        # print(f"Reference labels provided: {reference_labels}")
        if safe_ref_labels != list(safe_cell_types_mapping.values()):
            raise ValueError("Reference labels provided do not match cell types")
        cell_types = reference_labels
        safe_cell_types_mapping = {ct: slugify(ct) for ct in cell_types}
    safe_cell_types = list([slugify(ct) for ct in cell_types])

    barcodes[2] = barcodes[1].map(safe_cell_types_mapping)
    bc_to_ct = barcodes.set_index(0).to_dict()[2]
    out_files = {ct: f"{save_to}/{ct}.fragments.tsv" for ct in safe_cell_types_mapping.values()}

    print("Generating pseudo-bulk fragment files based on their source cell type / cluster...")
    frags_per_ct = {ct: 0 for ct in safe_cell_types_mapping.values()}
    if memory_saving:
        frag_file_handles = {ct: open(f"{save_to}/{ct}.fragments.tsv", "w") for ct in out_files.keys()}
        in_frag_handle = gzip.open(fragments_file, "rt") if fragments_file.lower().endswith(".gz") else open(fragments_file)

        total_frags = 0
        missing_frags = 0
        for frag in in_frag_handle:
            if frag.startswith("#"):
                continue
            items = frag.split("\t")
            total_frags += 1
            try:
                source_ct = bc_to_ct[items[3]]
                fh = frag_file_handles[source_ct]
                fh.write(frag)
                frags_per_ct[source_ct] += 1
            except KeyError as e:
                missing_frags += 1
        in_frag_handle.close()

        for fh in frag_file_handles.values():
            fh.close()
    else:
        all_frags = pd.read_csv(fragments_file, sep="\t", header=None, comment="#")
        total_frags = all_frags.shape[0]
        all_frags["ctype"] = all_frags[3].map(bc_to_ct)
        all_frags.dropna(inplace=True)
        missing_frags = total_frags - all_frags.shape[0]

        for ct, sdf in all_frags.groupby("ctype"):
            frags_per_ct[ct] = sdf.shape[0]
            columns = sdf.columns
            # no need to save the cell type label
            sdf[columns[:-1]].to_csv(out_files[ct], sep="\t", index=False)
    print(
        f"{total_frags - missing_frags} / {total_frags} fragments can be associated to provided cell type annotations")
    print("Finished generating pseudo-bulk fragment files based on their source cell type / cluster...")

    tbc = barcodes[[0, 2]].copy()
    tbc.columns = (0, 1)
    return out_files, frags_per_ct, safe_cell_types, tbc
