import os
import re
import unicodedata
import pybedtools
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from datetime import datetime
from subprocess import Popen, PIPE
from typing import Tuple, List, Optional, Union, Sequence
from pytorch_lightning import loggers
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from deepdetails.par_description import PARAM_DESC


def is_valid_file(arg: str, is_dir: bool = False, create_dir: bool = False) -> str:
    """Check if the value points to a valid file / directory

    Parameters
    ----------
    arg : str
        Value of the option
    is_dir : bool
        Value should be a directory
    create_dir : bool
        Create the directory if it does not exist

    Returns
    -------
    arg : str
        Option's value if it points to a valid file
    Examples
    --------
    For CLI options that should be pointing to files

    >>> import argparse
    >>> test_parser = argparse.ArgumentParser()
    >>> test_parser.add_argument("input_file", type=lambda x: is_valid_file(x))
    """
    checks = (os.path.isfile(arg), os.path.isdir(arg))
    if not is_dir and not checks[0]:
        raise IOError('The file {} does not exist!'.format(arg))
    elif is_dir and not checks[1]:
        if create_dir:
            try:
                os.makedirs(arg, exist_ok=True)
            except Exception:
                raise IOError('The directory {} does not exist and cannot be created!'.format(arg))
        else:
            raise IOError('The directory {} does not exist!'.format(arg))
    # File/directory exists so return the name
    return arg


def run_command(cmd: str, raise_exception: bool = False):
    """Run command

    Parameters
    ----------
    cmd : str

    raise_exception : bool
        Raise an exception if the return code is not 0.

    Returns
    -------
    stdout : str

    stderr : str

    return_code : int

    """
    with Popen(cmd, shell=True, stderr=PIPE, stdout=PIPE) as p:
        stdout, stderr = p.communicate()
        stderr = stderr.decode("utf-8")
        stdout = stdout.decode("utf-8")
    if raise_exception and p.returncode != 0:
        raise RuntimeError(stderr)
    return stdout, stderr, p.returncode


def get_trainer(study_name: str, save_to: str = ".", min_delta: float = 0, earlystop_patience: int = 3,
                max_epochs: int = 200, save_top_k_model: Union[str, int] = 1, hide_progress_bar: bool = False,
                model_summary_depth: int = 6, version: Optional[str] = None, accelerator: Optional[str] = "auto",
                devices: Union[List[int], str, int] = "auto", wandb_project: Optional[str] = None,
                wandb_entity: Optional[str] = None, wandb_upload_model: Union[str, bool] = False,
                pass_mark: str = "1st") -> Tuple[pl.Trainer, str]:
    """
    Get pl.Trainer for training / inference, etc.

    Parameters
    ----------
    study_name : str
        {study_name}
    save_to : str
        {save_to}
    min_delta : float
        {min_delta}
    earlystop_patience : int
        {earlystop_patience}
    max_epochs : int
        {max_epochs}
    save_top_k_model : Union[str, int]
        {save_top_k_model}
    hide_progress_bar : bool
        {hide_progress_bar}
    model_summary_depth : int
        {max_depth}
    version : Optional[str]
        {wandb_version}
    accelerator : Optional[str]
        {accelerator}
    devices : Union[List[int], str, int]
        {devices}
    wandb_project : Optional[str]
        {wandb_project}
    wandb_entity : Optional[str]
        {wandb_entity}
    wandb_upload_model : Union[str, int]
        {wandb_upload_model}
    pass_mark

    Returns
    -------
    trainer_obj : pl.Trainer
        Lightning Trainer object
    wbl.version : str
        Final effective WandB version string
    """.format(**PARAM_DESC)
    training_readout = "train_loss"
    # avoid reuse run records
    wandb.finish()

    pass_str = f"_{pass_mark}" if pass_mark else ""
    ver_str = f"{version}{pass_str}" if version else datetime.now().strftime("%y%m%d%H%M%S")
    wbl = WandbLogger(name=f"{study_name}{pass_str}", project=wandb_project, version=ver_str,
                      reinit=True, entity=wandb_entity,
                      log_model=wandb_upload_model, save_dir=save_to, offline=True,
                      settings=wandb.Settings(start_method="fork"))
    csvl = CSVLogger(name=f"{study_name}{pass_str}", version=wbl.version, save_dir=save_to,
                     prefix=f"{study_name}{pass_str}")

    checkpoint_callback = ModelCheckpoint(monitor=training_readout, save_top_k=save_top_k_model,
                                          dirpath=os.path.join(save_to, study_name, str(wbl.version)))
    early_stop_callback = EarlyStopping(
        monitor=training_readout, min_delta=min_delta,
        patience=earlystop_patience, verbose=True, mode="min"
    )
    callbacks = [early_stop_callback,
                 checkpoint_callback,
                 ModelSummary(max_depth=model_summary_depth),]
    if accelerator == "cpu" and isinstance(devices, list):
        devices = devices[0]
    trainer_obj = pl.Trainer(
        logger=[wbl, csvl],
        enable_checkpointing=True,
        max_epochs=max_epochs,
        accelerator=accelerator, devices=devices,
        callbacks=callbacks,
        enable_progress_bar=False if hide_progress_bar else True
    )
    return trainer_obj, wbl.version


def get_suprv_trainer(study_name: str, save_to: str = ".", min_delta: float = 0, earlystop_patience: int = 3,
                max_epochs: int = 200, save_top_k_model: Union[str, int] = 1, hide_progress_bar: bool = False,
                model_summary_depth: int = 6, version: Optional[str] = None, accelerator: Optional[str] = "auto",
                devices: Union[List[int], str, int] = "auto", wandb_project: Optional[str] = None,
                wandb_entity: Optional[str] = None, wandb_upload_model: Union[str, bool] = False,
                pass_mark: str = "1st") -> Tuple[pl.Trainer, str]:
    """
    Get pl.Trainer for training / inference, etc.

    Parameters
    ----------
    study_name : str
        {study_name}
    save_to : str
        {save_to}
    min_delta : float
        {min_delta}
    earlystop_patience : int
        {earlystop_patience}
    max_epochs : int
        {max_epochs}
    save_top_k_model : Union[str, int]
        {save_top_k_model}
    hide_progress_bar : bool
        {hide_progress_bar}
    model_summary_depth : int
        {max_depth}
    version : Optional[str]
        {wandb_version}
    accelerator : Optional[str]
        {accelerator}
    devices : Union[List[int], str, int]
        {devices}
    wandb_project : Optional[str]
        {wandb_project}
    wandb_entity : Optional[str]
        {wandb_entity}
    wandb_upload_model : Union[str, int]
        {wandb_upload_model}
    pass_mark

    Returns
    -------
    trainer_obj : pl.Trainer
        Lightning Trainer object
    wbl.version : str
        Final effective WandB version string
    """.format(**PARAM_DESC)
    training_readout = "val_loss"
    # avoid reuse run records
    wandb.finish()

    pass_str = f"_{pass_mark}" if pass_mark else ""
    ver_str = f"{version}{pass_str}" if version else datetime.now().strftime("%y%m%d%H%M%S")
    wbl = WandbLogger(name=f"{study_name}{pass_str}", project=wandb_project, version=ver_str,
                      reinit=True, entity=wandb_entity,
                      log_model=wandb_upload_model, save_dir=save_to, offline=True,
                      settings=wandb.Settings(start_method="fork"))
    csvl = CSVLogger(name=f"{study_name}{pass_str}", version=wbl.version, save_dir=save_to,
                     prefix=f"{study_name}{pass_str}")

    checkpoint_callback = ModelCheckpoint(monitor=training_readout, save_top_k=save_top_k_model,
                                          dirpath=os.path.join(save_to, study_name, str(wbl.version)))
    early_stop_callback = EarlyStopping(
        monitor=training_readout, min_delta=min_delta,
        patience=earlystop_patience, verbose=True, mode="min"
    )
    callbacks = [early_stop_callback,
                 checkpoint_callback,
                 ModelSummary(max_depth=model_summary_depth),]
    if accelerator == "cpu" and isinstance(devices, list):
        devices = devices[0]
    trainer_obj = pl.Trainer(
        logger=[wbl, csvl],
        enable_checkpointing=True,
        max_epochs=max_epochs,
        accelerator=accelerator, devices=devices,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=False if hide_progress_bar else True
    )
    return trainer_obj, wbl.version


def internal_qc(metrics: list[float], pred_counts: torch.Tensor):
    """
    Run internal QC to determine if the deconvolution is sound

    Parameters
    ----------
    metrics : list[float]
        List of internal metric values
    pred_counts : torch.Tensor
        Accumulated predicted counts for each target in a strand-specific manner

    Returns
    -------
    qc_val : float
        QC value
    qc_result_brc : bool
        Branch correlation based QC result
    qc_result_sum : bool
        Total sum based QC result
    """
    qc_val = 0.
    later = 0.
    if len(metrics) > 20:
        obs = metrics[:10] + metrics[-10:]
    elif len(metrics) > 2:
        obs = metrics
    else:
        obs = None
    if obs is not None:
        n_steps = len(obs)
        early = np.mean(obs[:n_steps // 2]) + 10e-16
        later = np.mean(obs[n_steps // 2:]) + 10e-16
        qc_val = early / later

    cr_sum_collapsed = torch.isclose(pred_counts, torch.zeros_like(pred_counts), atol=0.1).sum().item() == 0
    cr_br_cor = qc_val > 1. or later < 0.4
    return (qc_val, pred_counts), cr_br_cor, cr_sum_collapsed


def calc_counts_per_locus(profiles: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
                          counts: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
                          is_per_cluster_profile: bool = False) -> torch.Tensor:
    """Calculate read counts per genomic locus

    Parameters
    ----------
    profiles : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
        List of `torch.Tensor`, each Tensor stores the unnormed predictions for a cluster
    counts : Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]
        List of `torch.Tensor`, each Tensor stores the read counts for a cluster
    is_per_cluster_profile : bool, optional
        Set this as True if you want the function to return cluster-specific predictions, by default False

    Returns
    -------
    torch.Tensor
        Transformed predictions. Shape: clusters, batch, strands, seq_len if is_per_cluster_profile is True
        batch, strands, seq_len if is_per_cluster_profile if False
    """
    profiles = torch.stack(profiles, dim=0)  # output format: n_clusters, batch, strand, seq_len
    counts = torch.stack(counts, dim=0)  # output format: n_clusters, batch, strand

    reshaped_counts = counts.repeat(1, 1, profiles.shape[-1]).view(counts.shape[0], counts.shape[1], -1,
                                                                   counts.shape[2])
    reshaped_counts = torch.swapaxes(reshaped_counts, 2, 3)

    # cluster-specific predictions
    preds = profiles * reshaped_counts

    if not is_per_cluster_profile:
        # aggregated predictions
        preds = preds.sum(dim=0)
    return preds


def rescaling_prediction(pc_profiles: list[torch.Tensor], pc_counts: list[torch.Tensor],
                         expected_bulk_counts: torch.Tensor, expected_bulk_profiles: torch.Tensor,
                         rescaling_mode: int = 0) -> torch.Tensor:
    """Rescale predictions based on the observed bulk profiles

    Parameters
    ----------
    pc_profiles : list[torch.Tensor]
        List of predicted profiles. len(pc_profiles): clusters.
        Shape of elements in the list: batch, strands, seq_len
    pc_counts : list[torch.Tensor]
        List of predicted counts. len(pc_profiles): clusters.
        Shape of elements in the list: batch, strands
    expected_bulk_counts : torch.Tensor
        Observed bulk counts. Shape: batch, strands
    expected_bulk_profiles : torch.Tensor
        Observed bulk profiles. Shape: batch, strands, seq_len
    rescaling_mode : int
        0: No rescaling
        1: Rescaled by bulk counts
        2: Rescaled by bulk profiles

    Returns
    -------
    cluster_preds: torch.Tensor
        Profile prediction for each cluster. Shape: clusters, batch, strands, seq_len
    """
    cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True).cpu().numpy()

    if rescaling_mode == 1:  # total counts
        # predicted bulk counts
        ps_counts = torch.stack(pc_counts).sum(axis=0)
        # calculate rescale factor (k)
        # k * y_hat = y
        k = expected_bulk_counts / ps_counts
        # get the rescaled counts
        pc_counts = [pcc * k for pcc in pc_counts]
        cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True).cpu().numpy()
    elif rescaling_mode == 2:  # per-bp counts
        bulk_preds = cluster_preds.sum(axis=0)
        # calculate rescale factor (k)
        # k * y_hat = y
        k = expected_bulk_profiles.cpu() / (bulk_preds + 10e-16)
        # rescale
        cluster_preds = (k * cluster_preds).numpy()
    return cluster_preds


def get_log_dir(logger: loggers.Logger):
    """
    Get log directory

    Parameters
    ----------
    logger : pytorch_lightning.loggers.Logger
        The logger instance

    Returns
    -------
    log_dir : str
        Log directory
    version : str
        Model/logger version
    """
    try:
        if isinstance(logger, loggers.WandbLogger):
            log_dir = os.path.join(logger.save_dir, logger._name, logger.version)
            version = logger.version
        elif isinstance(logger, loggers.TensorBoardLogger):
            log_dir = logger.log_dir
            version = str(logger.version)
        else:
            log_dir = "."
            version = ""
    except:
        log_dir = "."
        version = ""
    return log_dir, version


def transform_counts(values: torch.Tensor, inject_random_noise: float = 10e-16, method: str = "asinh") -> torch.Tensor:
    """
    Get asinh/log- transformed counts

    Parameters
    ----------
    values : torch.Tensor
        Values to be transformed
    inject_random_noise : float
        Scale of the random noise to be injected. If you just want to do asinh transformation, set this as 0.
    method : str
        Transformation method. Can be 'asinh' or 'log1p'

    Returns
    -------
    transformed_values: torch.Tensor
        The same shape as `values`
    """
    noises = torch.randn_like(values, device=values.device) * inject_random_noise
    func = torch.asinh if method == "asinh" else torch.log1p
    return func(values) + noises


def slugify(value: str, allow_unicode: bool = False) -> str:
    """Converts a string to a safe path string by:

        1. Converting to ASCII if `allow_unicode` is False (the default).
        2. Converting to lowercase.
        3. Removing characters that aren't alphanumerics, underscores, hyphens, or whitespace.
        4. Replacing any whitespace or repeated dashes with single dashes.
        5. Removing leading and trailing whitespace, dashes, and underscores.
    Parameters
    ----------
    value : str
        String to be converted
    allow_unicode : bool
        Convert to ASCII if `allow_unicode` is False

    References
    ----------
    - https://github.com/django/django/blob/5f180216409d75290478c71ddb0ff8a68c91dc16/django/utils/text.py#L452-L469

    Returns
    -------
    slugified_str : str
        Converted string
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def set_tmp_for_pbt(tmp_dir="."):
    current_tmp_dir = pybedtools.get_tempdir()
    # if pybedtools is using the system's default tmp dir.,
    # then switch to current working directory to avoid using up all spaces at `/`
    if current_tmp_dir in ("/tmp", "/var/tmp", "/usr/tmp", "C:\\TEMP", "C:\\TMP", "\\TMP"):
        pybedtools.set_tempdir(tmp_dir)


def bedgraph_to_bigwig(in_bedgraph_path: str, out_bigwig_path: str, chrom_size_path: str):
    """
    Convert a file in bedGraph format to bigWig format

    Parameters
    ----------
    in_bedgraph_path : str

    out_bigwig_path : str

    chrom_size_path : str


    Returns
    -------

    """
    # get chromosomes that have size info
    allowed_chromosomes = set()
    with open(chrom_size_path, "r") as csf:
        for line in csf:
            allowed_chromosomes.add(line.strip().split()[0])

    # filter bedGraph and write to output file
    tmp_file = f"{in_bedgraph_path}.1"
    with open(in_bedgraph_path, "r") as input_file, open(tmp_file, "w") as output_file:
        for line in input_file:
            parts = line.split("\t")
            chromosome = parts[0]
            if chromosome in allowed_chromosomes:
                output_file.write(line)

    # convert the filtered bedGraph file into bigWig format
    try:
        cmd = f"bedGraphToBigWig {in_bedgraph_path} {chrom_size_path} {out_bigwig_path}"
        run_command(cmd, raise_exception=True)
    except RuntimeError as e:
        if str(e).find("not case-sensitive sorted") != -1:
            cmd1 = f"LC_COLLATE=C sort -k1,1 -k2,2n {in_bedgraph_path} > {in_bedgraph_path}.sorted"
            run_command(cmd1, raise_exception=True)
            cmd = f"bedGraphToBigWig {in_bedgraph_path}.sorted {chrom_size_path} {out_bigwig_path}"
            run_command(cmd, raise_exception=True)
            os.remove(f"{in_bedgraph_path}.sorted")
        else:
            raise e

    # remove the temporary file
    os.remove(tmp_file)


def compare_dicts(dicts: Sequence[dict]) -> bool:
    """
    Compare a list of dictionaries where the values may be numpy arrays.
    Returns True if all dictionaries have identical keys and values,
    False otherwise.

    Parameters
    ----------
    dicts : Sequence[dict]

    """
    if not dicts:
        return True  # empty list of dictionaries is trivially "identical"

    # get the keys of the first dictionary to compare against
    keys = list(dicts[0].keys())

    # compare keys across all dictionaries
    for d in dicts:
        if set(d.keys()) != set(keys):
            return False

    # compare values for each key across all dictionaries
    for key in keys:
        # get the value from the first dictionary
        first_value = dicts[0][key]

        # check if the value for this key in all dictionaries is the same
        for d in dicts[1:]:
            value = d[key]
            # if both values are numpy arrays, use np.array_equal for comparison
            if isinstance(first_value, np.ndarray) and isinstance(value, np.ndarray):
                if not np.array_equal(first_value, value):
                    return False
            # otherwise, perform regular equality check
            elif first_value != value:
                return False

    return True
