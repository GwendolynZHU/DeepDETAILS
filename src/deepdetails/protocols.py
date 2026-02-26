import logging
import h5py
import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from glob import glob
from typing import Optional, Tuple, Sequence, Union
from multiprocessing import Pool
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.utilities import move_data_to_device
from deepdetails.par_description import PARAM_DESC
from deepdetails.__about__ import __version__
from deepdetails.helper.utils import get_trainer, get_suprv_trainer, slugify, internal_qc, rescaling_prediction, compare_dicts, calc_counts_per_locus
from deepdetails.model.wrapper import DeepDETAILS, SupervisedDeepDETAILS
from deepdetails.data import SequenceSignalDataset, DynamicDataset, MultiTaskSupervisedDataset
from deepdetails.helper.prep_ds import (extend_regions_from_mid_points, combine_regions, generate_gc_matched_random_regions,
                                        build_data_volume, convert_bulk_frags_to_ct_frags, frag_file_to_bw)
from deepdetails.helper.preflight import preflight_check

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s | %(asctime)s | %(message)s", level=logging.INFO)

    
def superv_contrastive(regions: str, fa_file: str, acc_bw_files: str, pl_ct_bw_files: Sequence[str],
           mn_ct_bw_files: Sequence[str], cell_types: Sequence[str], save_to: str, 
           study_name: str, batch_size: int, num_workers: int, min_delta: float,
           save_preds: bool, chrom_cv: bool, y_length: int, earlystop_patience: int,
           max_epochs: int, save_top_k_model: int, model_summary_depth: int, hide_progress_bar: bool,
           accelerator: str, devices: str, version: str, wandb_project: Optional[str], wandb_entity: Optional[str],
           gamma: float, wandb_upload_model: bool, profile_shrinkage: int, filters: int, n_non_dil_layers: int,
           non_dil_kernel_size: int, n_dilated_layers: int, dil_kernel_size: int, head_layers: int,
           conv1_kernel_size: int, gru_layers: int, gru_dropout: float, profile_kernel_size: int,
           redundancy_loss_coef: float, prior_loss_coef: float, rescaling_mode: int,
           scale_function_placement: str, learning_rate: float, betas: Tuple[float, float],
           window_size: int = 4096, all_regions: bool = True, test_pos_only: bool = True, max_retry: int = 3,
           cv: Sequence[str] = ("chr22",), ct: Sequence[str] = ("chr19",), n_times_more_embeddings: int = 2,
           seq_only: Optional[bool] = False, loads_trunc: Optional[int] = None):
    """
    Use supervised contrastive learning to train a model with DETAILS previous architecture designs.
    """
    logger.info(f"Running supervised contrastive DeepDETAILS on sample {regions} (software version: {__version__})")
    if seq_only:
        logger.info("Supervised contrastive deepDETAILS is running in sequence only mode")
    if chrom_cv:
        logger.info(f"Using {cv} for validation and {ct} for testing")
    ds = MultiTaskSupervisedDataset(
        regions, fa_file, acc_bw_files, pl_ct_bw_files, mn_ct_bw_files, cell_types, y_length=y_length, is_training=1, non_background_only=not all_regions,
        chromosomal_val=cv if chrom_cv else None, chromosomal_test=ct if chrom_cv else None,
    )
    val_ds = MultiTaskSupervisedDataset(
        regions, fa_file, acc_bw_files, pl_ct_bw_files, mn_ct_bw_files, cell_types, y_length=y_length, is_training=0,
        chromosomal_val=cv if chrom_cv else None, chromosomal_test=ct if chrom_cv else None,
    )
    test_ds = MultiTaskSupervisedDataset(
        regions, fa_file, acc_bw_files, pl_ct_bw_files, mn_ct_bw_files, cell_types, y_length=y_length, is_training=2,
        chromosomal_val=cv if chrom_cv else None, chromosomal_test=ct if chrom_cv else None,
    )
    logger.info(f"Sample {regions} has {ds.n_cell_types} cell types")
    
    # export_multitask_predictions(SupervisedDeepDETAILS, 
    #                          dataset=test_ds, 
    #                          checkpoint="/fs/cbsuhy01/storage/yz2676/data/DetailedProCapNet/wdir/deepdetails/superv_contrast/Superv_fused_with_scaling/250508102626/epoch=5-step=30276.ckpt",
    #                          batch_size=batch_size,
    #                          num_workers=num_workers,
    #                          save_to=save_to,
    #                          cell_types=cell_types,
    #                          y_length=y_length, 
    #                          study_name=study_name,
    #                          device=f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
    
    train_iter = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_iter = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    # generate roughly 10 screenshots
    test_screenshots_ratio = 10 / len(test_iter)

    retry = 0
    max_retry = max(max_retry, 1)

    while retry < max_retry:
        if retry > 0:
            version_str = f"{version}-{retry}"
        else:
            version_str = version
        trainer, ver = get_suprv_trainer(
            study_name=study_name, save_to=save_to, min_delta=min_delta, earlystop_patience=earlystop_patience,
            max_epochs=max_epochs, save_top_k_model=save_top_k_model, hide_progress_bar=hide_progress_bar,
            model_summary_depth=model_summary_depth, version=version_str, accelerator=accelerator,
            devices=devices, wandb_project=wandb_project,
            wandb_entity=wandb_entity, wandb_upload_model=wandb_upload_model, pass_mark="")

        model = SupervisedDeepDETAILS(num_cell_types=ds.n_cell_types, filters=filters,
                            n_non_dil_layers=n_non_dil_layers, non_dil_kernel_size=non_dil_kernel_size,
                            n_dil_layers=n_dilated_layers, dil_kernel_size=dil_kernel_size,
                            conv1_kernel_size=conv1_kernel_size, profile_shrinkage=profile_shrinkage,
                            profile_kernel_size=profile_kernel_size, head_mlp_layers=head_layers,
                            redundancy_loss_coef=redundancy_loss_coef, prior_loss_coef=prior_loss_coef,
                            scale_function_placement=scale_function_placement, num_tasks=ds.n_targets,
                            gru_layers=gru_layers, gru_dropout=gru_dropout,
                            n_times_more_embeddings=n_times_more_embeddings,
                            learning_rate=learning_rate, betas=betas,
                            version=ver, t_x=ds.t_x, test_screenshot_ratio=test_screenshots_ratio,
                            gamma=gamma, seq_only=seq_only)

        logger.info("Start building model...")
        trainer.fit(model, train_dataloaders=train_iter, val_dataloaders=val_iter)

        logger.info("Evaluating model's performances...")
        trainer.test(model, test_iter)
        break

        # skip QC
        # # abnormality detection
        # # we expect the similarity between predictions to have a decreasing trend
        # # if not, it suggests model may be collapsed
        # qc_val, brc_qc_res, sum_qc_res = internal_qc(model.self_qc_values, model.sum_qc_metrics)
        # # only use sum_qc_res when test steps were called
        # if not model.enable_sum_qc_metrics:
        #     logger.warning("Bypassing sum-based QC since test loop was not triggered")
        #     final_sum_qc_res = True
        # else:
        #     final_sum_qc_res = sum_qc_res
        # if brc_qc_res and final_sum_qc_res:
        #     logger.info(f"Model passed self QC (self QC value: {qc_val})")
        #     break
        # else:
        #     logger.warning(f"Model collapsed (self QC value: {qc_val[0]} - {brc_qc_res}; {qc_val[1]} - {sum_qc_res})")
        #     ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        #     if ckpt_path is not None and os.path.exists(ckpt_path):
        #         logger.warning(
        #             f"Deleting model file {ckpt_path} from the collapsed run")
        #         os.remove(ckpt_path)
        #     logger.info("Trying to rerun the deconvolution process...")
        # retry += 1
    
    if save_preds:
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if os.path.exists(ckpt_path):
            logger.info(f"Exporting predictions using checkpoint from {ckpt_path}...")
            export_multitask_predictions(SupervisedDeepDETAILS, test_ds, ckpt_path, batch_size, num_workers=num_workers,
                           save_to=save_to, cell_types=cell_types, study_name=study_name,
                           y_length=y_length, device=f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
        else:
            logger.warning(f"Checkpoint file {ckpt_path} doesn't exist anymore... Maybe model collapsed?")


def export_multitask_predictions(model: pl.LightningModule, dataset: Union[callable, str], checkpoint: str,
                                  batch_size: int, num_workers: int, save_to: str, cell_types: Sequence[str],
                                  y_length: int, study_name: str, device: str, save_npy: bool = True):

    model = model.load_from_checkpoint(checkpoint).to(device).eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # One list per cell type
    profile_preds = [[] for _ in cell_types]
    count_preds = [[] for _ in cell_types]
    region_names = []

    for batch_idx, batch in enumerate(dataloader):
        (seq, atac), _, _, loads = batch
        seq, atac, loads = seq.to(device), atac.to(device), loads.to(device)

        pc_profiles, pc_counts, _, _ = model((seq, atac), loads)

        ct_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)

        # track region coordinates from dataset
        start_idx = batch_idx * batch_size
        for i in range(seq.shape[0]):
            row = dataset.df.iloc[start_idx + i]
            chrom, start, end = row[0], row[1], row[2]
            center = (start + end) // 2
            y_offset = y_length // 2
            region_names.append(f"{chrom}:{center-y_offset}-{center+y_offset}")

        for i, ct in enumerate(cell_types):
            # Append to the list
            profile_preds[i].append(ct_preds[i].detach().cpu().numpy())
            count_preds[i].append(pc_counts[i].detach().cpu().numpy())  

    # Save
    combined_counts_dict = {}
    for i, ct in enumerate(cell_types):
        prof_np = np.concatenate(profile_preds[i], axis=0)  # (n_regions, 2, 1000)
        count_np = np.concatenate(count_preds[i], axis=0)   # (n_regions, 2)

        # Save npy
        np.save(os.path.join(save_to, f"{study_name}.{ct}.profiles.npy"), prof_np)
        np.save(os.path.join(save_to, f"{study_name}.{ct}.counts.npy"), count_np)

        total_counts = np.sum(count_np, axis=1)
        combined_counts_dict[ct] = total_counts

    # Save counts to CSV
    df_all_counts = pd.DataFrame(combined_counts_dict, index=region_names)
    counts_file = os.path.join(
        save_to,
        "counts.csv.gz" if study_name == "" else f"{slugify(study_name)}.counts.csv.gz"
    )
    df_all_counts.to_csv(counts_file)


def deconv(dataset: str, save_to: str, study_name: str, batch_size: int, num_workers: int, min_delta: float,
           save_preds: bool, chrom_cv: bool, y_length: int, earlystop_patience: int,
           max_epochs: int, save_top_k_model: int, model_summary_depth: int, hide_progress_bar: bool,
           accelerator: str, devices: str, version: str, wandb_project: Optional[str], wandb_entity: Optional[str],
           gamma: float, wandb_upload_model: bool, profile_shrinkage: int, filters: int, n_non_dil_layers: int,
           non_dil_kernel_size: int, n_dilated_layers: int, dil_kernel_size: int, head_layers: int,
           conv1_kernel_size: int, gru_layers: int, gru_dropout: float, profile_kernel_size: int,
           redundancy_loss_coef: float, prior_loss_coef: float, rescaling_mode: int,
           scale_function_placement: str, learning_rate: float, betas: Tuple[float, float],
           all_regions: bool = False, test_pos_only: bool = True, max_retry: int = 3,
           cv: Optional[Tuple[str, ...]] = None, ct: Optional[Tuple[str, ...]] = None, n_times_more_embeddings: int = 2,
           loads_trunc: Optional[int] = None, seq_only: Optional[bool] = False):
    """
    Deconvolve a bulk sequencing library with DETAILS
    """
    logger.info(f"Running DeepDETAILS deconvolution on sample {dataset} (software version: {__version__})")
    if seq_only:
        logger.info("DeepDETAILS is running in sequence only mode")

    ds = SequenceSignalDataset(
        root=dataset, y_length=y_length, is_training=True, non_background_only=not all_regions,
        chromosomal_val=cv if chrom_cv else None, chromosomal_test=ct if chrom_cv else None,
        loads_trunc=loads_trunc
    )
    test_ds = SequenceSignalDataset(
        root=dataset, y_length=y_length, is_training=2,
        chromosomal_val=cv if chrom_cv else None, chromosomal_test=ct if chrom_cv else None,
        non_background_only=test_pos_only,
        loads_trunc=loads_trunc
    )
    logger.info(f"Sample {dataset} has {ds.n_clusters} clusters/cell types")
    _aux_info = {"ground truth": ds.load_groundtruth, "acc norm": ds.has_acc_norm, "prior": ds.prior.dim() == 2}
    logger.info("Additional information in the dataset: {}".format(
        '\t'.join(["{0}: {1}".format(k, v) for k, v in _aux_info.items()])))

    train_iter = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    test_iter = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    # generate roughly 10 screenshots
    test_screenshots_ratio = 10 / len(test_iter)

    retry = 0
    max_retry = max(max_retry, 1)

    while retry < max_retry:
        if retry > 0:
            version_str = f"{version}-{retry}"
        else:
            version_str = version
        trainer, ver = get_trainer(
            study_name=study_name, save_to=save_to, min_delta=min_delta, earlystop_patience=earlystop_patience,
            max_epochs=max_epochs, save_top_k_model=save_top_k_model, hide_progress_bar=hide_progress_bar,
            model_summary_depth=model_summary_depth, version=version_str, accelerator=accelerator,
            devices=devices, wandb_project=wandb_project,
            wandb_entity=wandb_entity, wandb_upload_model=wandb_upload_model, pass_mark="")

        model = DeepDETAILS(expected_clusters=ds.n_clusters, filters=filters,
                            n_non_dil_layers=n_non_dil_layers, non_dil_kernel_size=non_dil_kernel_size,
                            n_dil_layers=n_dilated_layers, dil_kernel_size=dil_kernel_size,
                            conv1_kernel_size=conv1_kernel_size, profile_shrinkage=profile_shrinkage,
                            profile_kernel_size=profile_kernel_size, head_mlp_layers=head_layers,
                            redundancy_loss_coef=redundancy_loss_coef, prior_loss_coef=prior_loss_coef,
                            scale_function_placement=scale_function_placement, num_tasks=ds.n_targets,
                            gru_layers=gru_layers, gru_dropout=gru_dropout,
                            n_times_more_embeddings=n_times_more_embeddings,
                            learning_rate=learning_rate, betas=betas,
                            version=ver, t_x=ds.t_x, test_screenshot_ratio=test_screenshots_ratio,
                            gamma=gamma, seq_only=seq_only)

        logger.info("Start building model...")
        trainer.fit(model, train_dataloaders=train_iter, val_dataloaders=None)

        logger.info("Evaluating model's performances...")
        trainer.test(model, test_iter)

        # abnormality detection
        # we expect the similarity between predictions to have a decreasing trend
        # if not, it suggests model may be collapsed
        qc_val, brc_qc_res, sum_qc_res = internal_qc(model.self_qc_values, model.sum_qc_metrics)
        # only use sum_qc_res when test steps were called
        if not model.enable_sum_qc_metrics:
            logger.warning("Bypassing sum-based QC since test loop was not triggered")
            final_sum_qc_res = True
        else:
            final_sum_qc_res = sum_qc_res
        if brc_qc_res and final_sum_qc_res:
            logger.info(f"Model passed self QC (self QC value: {qc_val})")
            break
        else:
            logger.warning(f"Model collapsed (self QC value: {qc_val[0]} - {brc_qc_res}; {qc_val[1]} - {sum_qc_res})")
            ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
            if ckpt_path is not None and os.path.exists(ckpt_path):
                logger.warning(
                    f"Deleting model file {ckpt_path} from the collapsed run")
                os.remove(ckpt_path)
            logger.info("Trying to rerun the deconvolution process...")
        retry += 1

    if save_preds:
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if os.path.exists(ckpt_path):
            logger.info(f"Exporting predictions using checkpoint from {ckpt_path}...")
            export_results(DeepDETAILS, dataset, ckpt_path, batch_size, num_workers=num_workers,
                           save_to=save_to, study_name=study_name,
                           y_length=y_length, rescaling_mode=rescaling_mode, loads_trunc=loads_trunc,
                           pos_only=test_pos_only, device=f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu")
        else:
            logger.warning(f"Checkpoint file {ckpt_path} doesn't exist anymore... Maybe model collapsed?")


def _export_results(model: pl.LightningModule, dataset: callable, checkpoint: str, batch_size: int,
                    num_workers: int, save_to: str, study_name: str = "", y_length: int = 1000,
                    rescaling_mode: int = 0, device: str = "cpu", merge_strands: bool = False):
    """
    Core function for exporting results to a hdf5 file, use it via export_results or export_wg_results

    Parameters
    ----------
    model : pl.LightningModule

    dataset : callable
        {dataset}
    checkpoint : str
        {checkpoint}
    batch_size : int
        {batch_size}
    num_workers : int
        {num_workers}
    save_to : str
        {save_to}
    study_name : str, optional
        {study_name}
    y_length : int, optional
        {y_length}
    rescaling_mode : int, optional
        {rescaling_mode}
    device : str
        {device}
    merge_strands : bool, optional
        {merge_strands}

    Returns
    -------

    """.format(**PARAM_DESC)
    trained_model = model.load_from_checkpoint(checkpoint).to(device=device)

    pred_iter = DataLoader(dataset, batch_size=batch_size * 4, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    # disable grads + batchnorm + dropout
    torch.set_grad_enabled(False)
    trained_model.eval()

    n_regions = len(dataset)
    regions_df = dataset.df.copy()

    index_offset = 0
    data_file = os.path.join(save_to, "predictions.h5") if study_name == "" else os.path.join(save_to,
                                                                                              f"{slugify(study_name)}.predictions.h5")

    counts_lst = [torch.zeros(n_regions) for _ in range(dataset.n_clusters)]

    with h5py.File(data_file, "w") as f:
        # first save regions
        truncation = (dataset.t_x - y_length) // 2
        transformed_regions = regions_df.copy().drop(columns="index")
        transformed_regions[1] += truncation
        transformed_regions[2] -= truncation
        _chr_mapping = {v: k for k, v in enumerate(transformed_regions[0].unique())}
        transformed_regions[0] = transformed_regions[0].map(_chr_mapping)
        dset_rg = f.create_dataset("regions", data=transformed_regions.to_numpy())
        for chr_str, chr_idx in _chr_mapping.items():
            dset_rg.attrs[f"chr_{chr_idx}"] = chr_str

        # then save the predictions
        out_targets = 1 if merge_strands and dataset.n_targets > 1 else dataset.n_targets
        ds = f.create_dataset("preds", (dataset.n_clusters, n_regions, out_targets, y_length),
                              dtype="f", chunks=(1, 1, out_targets, y_length), compression="gzip")
        ds.attrs["n_clusters"] = dataset.n_clusters
        ds.attrs["n_targets"] = out_targets
        ds.attrs["cluster_names"] = dataset.cluster_names

        for batch_idx, batch in enumerate(pred_iter):
            x, expected_counts, expected_profiles, _, loads, misc = move_data_to_device(batch, device=device)
            bs = x[0].shape[0]

            model_outs = trained_model(x, loads)
            pc_profiles, pc_counts = model_outs[:2]

            cluster_preds = rescaling_prediction(
                pc_profiles, pc_counts, expected_counts, expected_profiles, rescaling_mode)

            if cluster_preds.shape[2] > out_targets:
                cluster_preds = cluster_preds.sum(axis=2, keepdims=True)

            ds[:, index_offset:index_offset + bs, :, :] = cluster_preds
            agg_counts = cluster_preds.sum(axis=-1).sum(axis=-1)
            for i, ac in enumerate(agg_counts): counts_lst[i][index_offset:index_offset + bs] = torch.from_numpy(ac)
            index_offset += bs

    region_name = regions_df[0] + ":" + transformed_regions[1].map(str) + "-" + transformed_regions[2].map(str)

    counts_file = os.path.join(save_to, "counts.csv.gz") if study_name == "" else os.path.join(save_to,
                                                                                               f"{slugify(study_name)}.counts.csv.gz")
    pd.DataFrame({dataset.cluster_names[k]: v for k, v in enumerate(counts_lst)}, index=region_name.values).to_csv(
        counts_file)


def export_results(model: pl.LightningModule, dataset: Union[callable, str], checkpoint: str, batch_size: int,
                   num_workers: int, save_to: str, loads_trunc: Optional[int] = None,
                   study_name: str = "", y_length: int = 1000, rescaling_mode: int = 0,
                   pos_only: bool = True, device: str = "cpu"):
    """
    Exports results to a hdf5 file

    Parameters
    ----------
    model : pl.LightningModule

    dataset : str
        {dataset}
    checkpoint : str
        {checkpoint}
    batch_size : int
        {batch_size}
    num_workers : int
        {num_workers}
    loads_trunc : Optional[int]
        {loads_trunc}
    save_to : str
        {save_to}
    study_name : str, optional
        {study_name}
    y_length : int, optional
        {y_length}
    rescaling_mode : int, optional
        {rescaling_mode}
    pos_only : bool, optional
        {test_pos_only}

    Returns
    -------

    """.format(**PARAM_DESC)
    pred_ds = SequenceSignalDataset(
        root=dataset, y_length=y_length, is_training=-1, loads_trunc=loads_trunc,
        chromosomal_val=None, chromosomal_test=None, non_background_only=pos_only
    )
    _export_results(
        model, pred_ds, checkpoint, batch_size, num_workers, save_to,
        study_name, y_length, rescaling_mode, device
    )


def export_wg_results(model: pl.LightningModule, checkpoint: str, fa_file: str, pl_bulk_bw_file: str,
                      acc_bw_files: Sequence[str], regions_file: str, batch_size: int,
                      num_workers: int, save_to: str, mn_bulk_bw_file: Optional[str] = None,
                      pl_ct_bw_files: Optional[Sequence[str]] = None, mn_ct_bw_files: Optional[Sequence[str]] = None,
                      sc_norm_file: Optional[str] = None, cluster_names: Optional[Sequence[str]] = None,
                      y_length: int = 1_000, target_sliding_sum: Optional[int] = 0, window_size: int = 4096,
                      is_training: int = 1, chromosomal_val: Optional[Sequence[str]] = None,
                      chromosomal_test: Optional[Sequence[str]] = None, loads_trunc: Optional[int] = None,
                      study_name: str = "", rescaling_mode: int = 0, pos_only: bool = True,
                      use_bulk_constraint: Optional[bool] = False, merge_strands: bool = False, device: str = "cpu"):
    """
    Exports whole genome results to a hdf5 file

    Parameters
    ----------
    model : pl.LightningModule

    dataset : str
        {dataset}
    checkpoint : str
        {checkpoint}
    batch_size : int
        {batch_size}
    num_workers : int
        {num_workers}
    loads_trunc : Optional[int]
        {loads_trunc}
    save_to : str
        {save_to}
    study_name : str, optional
        {study_name}
    y_length : int, optional
        {y_length}
    rescaling_mode : int, optional
        {rescaling_mode}
    pos_only : bool, optional
        {test_pos_only}
    use_bulk_constraint : bool, optional
        {use_bulk_constraint}
    merge_strands : bool, optional
        {merge_strands}
    device : str, optional
        {device}

    Returns
    -------

    """.format(**PARAM_DESC)
    pred_ds = DynamicDataset(
        fa_file=fa_file, pl_bulk_bw_file=pl_bulk_bw_file, acc_bw_files=acc_bw_files,
        regions_file=regions_file, mn_bulk_bw_file=mn_bulk_bw_file, pl_ct_bw_files=pl_ct_bw_files,
        mn_ct_bw_files=mn_ct_bw_files, sc_norm_file=sc_norm_file, cluster_names=cluster_names,
        y_length=y_length, is_training=is_training, loads_trunc=loads_trunc, t_x=window_size,
        chromosomal_val=chromosomal_val, chromosomal_test=chromosomal_test, non_background_only=pos_only,
        target_sliding_sum=target_sliding_sum, use_bulk_constraint=use_bulk_constraint
    )
    _export_results(
        model, pred_ds, checkpoint, batch_size, num_workers, save_to,
        study_name, y_length, rescaling_mode, device, merge_strands
    )


def pred_to_bw(pred_file: str, save_to: str, chrom_size: str, min_abs_val: float = 10e-3, num_workers: int = 8,
               skip_sort_merge: bool = False, binning: Optional[int] = 0):
    """
    Convert predictions to signals in BigWig files

    Parameters
    ----------
    pred_file : str

    save_to : str
        {save_to}
    chrom_size : str
        {chrom_size}
    min_abs_val : float
        {min_abs_val}
    num_workers : int
        {num_workers}
    skip_sort_merge : bool
        {skip_sort_merge}
    binning : Optional[int]
        {out_binning}

    Returns
    -------

    """.format(**PARAM_DESC)
    from deepdetails.helper.export import preds_to_bg_star, bg_to_bw_core, STRAND_LABELS, STRAND_COEFF

    with h5py.File(pred_file, "r") as f:
        n_clusters = f["preds"].attrs["n_clusters"]
        n_strands = f["preds"].attrs["n_targets"]
        cluster_names = f["preds"].attrs["cluster_names"]

    pairs = [(i, j) for i in range(n_clusters) for j in range(n_strands)]

    jobs = []
    for ci, si in pairs:
        jobs.append((pred_file, ci, si, save_to, min_abs_val, binning))

    logger.info("Converting predictions to bedGraph files")
    with Pool(num_workers) as p:
        p.starmap(preds_to_bg_star, jobs)

    logger.info("Converting bedGraph files to bigWig files")
    jobs = []
    for idx_cluster in range(n_clusters):
        safe_cluster_name = slugify(cluster_names[idx_cluster])
        for idx_strand in range(n_strands):
            bg_file = os.path.join(save_to, f"C{idx_cluster}.{STRAND_LABELS[idx_strand]}.bg")
            if not os.path.exists(bg_file):
                raise IOError(f"Could not find {bg_file}")
            jobs.append((
                bg_file,
                os.path.join(save_to, f"{safe_cluster_name}.{STRAND_LABELS[idx_strand]}"),
                chrom_size, STRAND_COEFF[idx_strand], skip_sort_merge
            ))

    with Pool(num_workers) as p:
        p.starmap(bg_to_bw_core, jobs)


def prepare_dataset(regions: Sequence[str], bulk_pl: str, save_to: str, window_size: int,
                    chrom_size: str, genome_fa: str, background_sampling_ratio: float = 0.,
                    fragments: Optional[str] = None, barcodes: Optional[str] = None,
                    accessibility: Optional[Sequence[str]] = None, bulk_mn: Optional[str] = None,
                    ref_labels: Optional[Sequence[str]] = None, ref_pls: Optional[Sequence[str]] = None,
                    ref_mns: Optional[Sequence[str]] = None, background_blacklist: Optional[str] = None,
                    final_regions: bool = False, merge_overlap_peaks: int = 0, keep_frags: bool = False,
                    target_sliding_sum: int = 0, seed: Optional[int] = None, skip_preflight: bool = False,
                    accessible_peaks: Optional[str] = None, preflight_cutoff: float = 0.035, nu: float = 0.85,
                    candidate_qval: float = 0.01, candidate_fc: float = 2, max_top_n: int = 1000, n_aggs: int = 5,
                    min_cells_required: Optional[int] = 20, memory_saving: Optional[bool] = False,
                    use_qnorm: Optional[bool] = False):
    """Build a dataset for DETAILS

    Parameters
    ----------
    regions : Sequence[str]
        {regions}
    bulk_pl : str
        {bulk_pl}
    save_to : str
        {save_to}
    window_size : int
        {t_x}
    chrom_size : str
        {chrom_size}
    genome_fa : str
        {genome_fa}
    background_sampling_ratio : float, optional
        {background_sampling_ratio}, by default 0.
    fragments : Optional[str]
        {fragments}
    barcodes : Optional[str]
        {barcodes}
    accessibility : Optional[Sequence[str]]
        {accessibility}
    bulk_mn : Optional[str], optional
        {bulk_mn}, by default None
    ref_labels : Sequence[str]
        {ref_labels}
    ref_pls : Sequence[str]
        {ref_pls}
    ref_mns : Sequence[str]
        {ref_mns}
    background_blacklist : Optional[str]
        {background_blacklist}, by default None
    final_regions : bool, optional
        {final_regions}
    merge_overlap_peaks : int, optional
        {merge_overlap_peaks}
    keep_frags : bool, optional
        {keep_frags}
    target_sliding_sum : Optional[int]
        {target_sliding_sum}
    seed : Optional[int]
        {seed}, by default None
    skip_preflight : Optional[bool]
        {skip_preflight}, by default False
    accessible_peaks : Optional[str]
        {accessible_peaks}
    preflight_cutoff : Optional[float]
        {preflight_cutoff}
    candidate_qval : Optional[float]
        {qval_cutoff}    
    candidate_fc : Optional[float]
        {fc_cutoff}
    nu : Optional[float]
        {preflight_nu}
    max_top_n : Optional[int]
        {max_top_n}
    n_aggs : Optional[int]
        {n_aggs}
    min_cells_required : Optional[int]
        {min_cells_required}
    memory_saving : Optional[bool]
        {memory_saving}
    use_qnorm : Optional[bool]
        {use_qnorm}
    """.format(**PARAM_DESC)
    norm_factors = None
    if accessibility is None and fragments is None:
        raise ValueError("Either accessibility or fragments must be provided")

    if not final_regions:
        chrom_size_file = genome_fa + ".fai"
        if os.path.exists(chrom_size_file):
            chr_size = pd.read_csv(chrom_size_file, sep="\t", header=None)
            chrs_in_fa = set(chr_size[0].unique())
        else:
            raise IOError(f"Expecting chromosome size file {chrom_size_file}."
                          f"You can generate it with `samtools faidx`")
        extended_regions = extend_regions_from_mid_points(
            combine_regions(regions, chrs_in_fa, merge_overlap=merge_overlap_peaks),
            extensions=(window_size // 2, window_size // 2 - 1),
            chromosome_size=chrom_size)
        extended_regions["region_type"] = 1

        # remove regions on mitochondria or scaffold chromosomes
        extended_regions = extended_regions.loc[
            (extended_regions[0] != "chrM") & (~extended_regions[0].str.contains("_"))
            ].copy().reset_index(drop=True)

        # check if all regions are extended successfully, if not, remove failed ones
        d = extended_regions[2] - extended_regions[1]
        if (d == window_size).sum() != d.shape[0]:
            n_before = extended_regions.shape[0]
            extended_regions = extended_regions.loc[d == window_size].copy().reset_index(drop=True)
            n_after = extended_regions.shape[0]
            logger.warning(f"{n_after - n_before} regions removed because of their lengths")

        # sample background regions
        if background_sampling_ratio > 0.:
            logger.info("Sampling background regions...")
            background_regions = generate_gc_matched_random_regions(
                extended_regions, chrom_size, genome_fa,
                seed=seed, sample_scale_factor=background_sampling_ratio,
                blacklist=background_blacklist,
                dist_compare_plot_file=os.path.join(save_to, "background_sampling.png"))
            background_regions["region_type"] = 0
            logger.info(f"Sampled {background_regions.shape[0]} background regions")

            final_regions = pd.concat(
                [extended_regions, background_regions[[0, 1, 2, "region_type"]]],
                ignore_index=True
            ).sort_values([0, 1]).reset_index(drop=True)
        else:
            final_regions = extended_regions.copy().sort_values([0, 1]).reset_index(drop=True)
        final_regions = final_regions[[0, 1, 2, "region_type"]]
        final_regions.to_csv(os.path.join(save_to, "regions.csv"), header=False, index=False)
    else:
        final_regions = pd.read_csv(regions if isinstance(regions, str) else regions[0],
                                    comment="#", header=None, sep="\t")
    print(final_regions)

    if fragments is not None:
        if barcodes is None:
            raise ValueError("Either barcodes or fragments must be provided")
        else:
            logger.info("Generating pseudo-bulk accessibility profiles for each cluster in the reference...")
            _ct_frag_dict, _frags_per_ct, ref_labels, _tbc = convert_bulk_frags_to_ct_frags(
                fragments, barcodes, save_to, reference_labels=ref_labels, memory_saving=memory_saving)
            logger.info("Finished generating pseudo-bulk accessibility profiles.")

            if not skip_preflight:
                if accessible_peaks is not None and os.path.exists(accessible_peaks):
                    logger.info("Preflight check...")
                    accessible_regions = pd.read_csv(accessible_peaks, sep="\t", header=None)
                    to_exclude = preflight_check(
                        _ct_frag_dict, _tbc, accessible_regions,(bulk_pl, bulk_mn),
                        max_top_n=max_top_n, preflight_cutoff=preflight_cutoff, nu=nu,
                        qval_cutoff=candidate_qval, fc_cutoff=candidate_fc, n_aggs=n_aggs,
                        min_cells_required=min_cells_required, use_qnorm=use_qnorm, save_to=save_to
                    )
                    for c in to_exclude:
                        logger.warning(f"Cluster {c} is excluded. If you want to keep it, bypass the preflight check.")
                        del _ct_frag_dict[c]
                        ref_labels.remove(c)
                else:
                    raise IOError(f"Expecting {accessible_peaks} for preflight check")

            logger.info("Generating pseudo-bulk bigWig files based on their source cell type / cluster...")
            for ct, ctf in _ct_frag_dict.items():
                frag_file_to_bw(ctf, frag_proc="5pi", chrom_size=chrom_size, n_frags=_frags_per_ct[ct])
            logger.info("Finished generating pseudo-bulk bigWig files based on their source cell type / cluster...")

            accessibility = []
            norm_factors = []
            for ct in ref_labels:
                accessibility.append(f"{save_to}/{ct}.fragments.bw")
                assert os.path.exists(accessibility[-1])
                norm_factors.append((ct, _frags_per_ct[ct]))
            if not keep_frags:
                for ct in glob(f"{save_to}/*.fragments.tsv"):
                    os.remove(ct)

    barcodes = pd.read_csv(barcodes, sep="\t", header=None)
    if barcodes.shape[1] != 2:
        raise ValueError("barcode_file should have 2 columns: the cell barcode and cell type annotation")
    cell_types = sorted(barcodes[1].unique().tolist())
    ref_labels = list([slugify(ct) for ct in cell_types])
    accessibility = []
    for ct in ref_labels:
        accessibility.append(f"{save_to}/{ct}.fragments.bw")
    # reference
    pl_refs = []
    mn_refs = []
    if ref_labels is not None and ref_pls is not None:
        if len(ref_labels) == len(ref_pls) == len(ref_mns):
            for (label, plf, mnf) in zip(ref_labels, ref_pls, ref_mns):
                logger.info(f"Adding ground truth for {label}: {plf} {mnf}")
                pl_refs.append(plf)
                mn_refs.append(mnf)

    print("window_size", window_size)
    build_data_volume(final_regions, [bulk_pl, ], [bulk_mn, ] if bulk_mn is not None else [],
                      accessibility, window_size, save_to, genome_fa, pl_refs, mn_refs,
                      target_sliding_sum=target_sliding_sum)

    # save RPM-norm factors
    if norm_factors is not None and ref_labels is not None:
        norm_factors = pd.DataFrame(norm_factors)
        norm_factors[1] = (1000. * 1000.) / norm_factors[1]
        pd.DataFrame(norm_factors).to_csv(os.path.join(save_to, "scatac.norm.csv"), header=False, index=False)

        transformed_norm = norm_factors.copy()
        _mapping = {v: k for k, v in enumerate(norm_factors[0].unique())}
        transformed_norm[0] = transformed_norm[0].map(_mapping)

        with h5py.File(f"{save_to}/data.h5", "a") as fh:
            # write cluster labels
            gr = fh["dec"]
            gr.attrs["cluster_names"] = ",".join(ref_labels)

            # save norm factor
            if transformed_norm is not None:
                if transformed_norm is not None:
                    ds = fh.create_dataset("scatac_norm", data=transformed_norm.to_numpy())
                    for cluster_str, cluster_idx in _mapping.items():
                        ds.attrs[f"c_{cluster_idx}"] = cluster_str


def merge_rep_preds(in_pred_files: Sequence[str], save_to: str, keep_old: bool = False, quiet: bool = False):
    """
    Merge predictions from multiple replicate runs
    
    Parameters
    ----------
    in_pred_files : Sequence[str]
        {preds}
    save_to : str
        {save_to}
    keep_old : bool
        {keep_old_preds}
    quiet : bool
        {quiet}

    Returns
    -------

    """.format(**PARAM_DESC)
    logger.info(f"Merging (averaging) {in_pred_files} into {save_to}")
    out = h5py.File(save_to, "w")
    ins = [h5py.File(f, "r") for f in in_pred_files]

    # check regions
    regions = [i["regions"][:] for i in ins]
    region_checks = [np.array_equal(r, regions[0]) for r in regions[1:]]
    if not all(region_checks):
        logger.error("Expecting regions to be consistent across all input files")
        logger.error(region_checks)
        exit(1)

    region_attrs = [dict(i["regions"].attrs.items()) for i in ins]
    if not all([attr == region_attrs[0] for attr in region_attrs[1:]]):
        logger.error("Expecting regions to be consistent across all input files")
        logger.error(region_attrs)
        exit(1)

    # copy region definitions
    dset_rg = out.create_dataset("regions", data=regions[0])
    for attr_name, attr_value in ins[0]["regions"].attrs.items():
        dset_rg.attrs[attr_name] = attr_value

    # check pred shapes
    pred_shapes = [i["preds"].shape for i in ins]
    if not all([s == pred_shapes[0] for s in pred_shapes[1:]]):
        logger.error("Expecting predictions to be the same shape across all input files")
        logger.error(pred_shapes)
        exit(1)

    # check pred attrs
    pred_attrs = [dict(i["preds"].attrs.items()) for i in ins]
    if not compare_dicts(pred_attrs):
        logger.error("Expecting predictions to have the same set of attributions across all input files")
        logger.error(pred_attrs)
        exit(1)

    # create dataset for the predictions
    ds = out.create_dataset("preds",
                            (pred_shapes[0][0], pred_shapes[0][1], pred_shapes[0][2], pred_shapes[0][3]),
                            dtype="f", chunks=(1, 1, pred_shapes[0][2], pred_shapes[0][3]), compression="gzip")
    ds.attrs["n_clusters"] = ins[0]["preds"].attrs["n_clusters"]
    ds.attrs["n_targets"] = ins[0]["preds"].attrs["n_targets"]
    ds.attrs["cluster_names"] = ins[0]["preds"].attrs["cluster_names"]

    # calculate the average
    for idx in tqdm(range(pred_shapes[0][1]), disable=quiet):
        per_rep_values = np.stack([i["preds"][:, idx, :, :] for i in ins])
        ds[:, idx, :, :] = per_rep_values.mean(axis=0)

    # close files
    for f in ins: f.close()
    out.close()
    logger.info("Merging completed")

    if not keep_old:
        logger.info("Removing predictions from replicate runs")
        for f in in_pred_files:
            os.remove(f)
