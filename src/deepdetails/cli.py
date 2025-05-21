import argparse
import os
from glob import glob
from deepdetails.__about__ import __version__
from deepdetails.par_description import PARAM_DESC
from deepdetails.protocols import deconv, superv_contrastive, prepare_dataset, export_results, export_wg_results, pred_to_bw, merge_rep_preds
from deepdetails.model.wrapper import DeepDETAILS
from deepdetails.helper.utils import is_valid_file


def _general_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("General")
    group.add_argument("--study-name", default="DeepDETAILS", type=str, help=PARAM_DESC["study_name"])
    group.add_argument("--save-to", default=".", type=lambda _x: is_valid_file(_x, is_dir=True, create_dir=True),
                       help=PARAM_DESC["save_to"])
    group.add_argument("--y-length", default=1000, type=int, help=PARAM_DESC["y_length"])
    group.add_argument("--num-workers", help=PARAM_DESC["num_workers"], type=int, default=16)
    group.add_argument("--batch-size", help=PARAM_DESC["batch_size"],
                       type=int, default=32)
    group.add_argument("--hide-progress-bar", action="store_true", help=PARAM_DESC["hide_progress_bar"])
    group.add_argument("--version", action="version", version=__version__)


def _supervised_training_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Supervised Training")
    group.add_argument("--regions", required=True, type=str,
                        help=PARAM_DESC["regions"])
    group.add_argument("--fa_file", required=True, type=str,
                        help=PARAM_DESC["fa_file"])
    group.add_argument("--accessibility", dest="acc_bw_files", nargs="+",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["accessibility"])
    group.add_argument("--ref-pls", dest="pl_ct_bw_files", nargs="+",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_pls"])
    group.add_argument("--ref-mns", dest="mn_ct_bw_files", nargs="+",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_mns"])
    group.add_argument("--ref-labels", dest="cell_types", nargs="+",
                       type=str, help=PARAM_DESC["ref_labels"])
    group.add_argument("--window-size", default=4096, type=int, help=PARAM_DESC["t_x"])
    group.add_argument("--chrom-cv", action="store_true", 
                       help=PARAM_DESC["chrom_cv"], default=True)
    group.add_argument("--chromosomal-validation", "--cv", dest="cv", default=("chr22",),
                       help=PARAM_DESC["chromosomal_validation"], nargs="+")
    group.add_argument("--chromosomal-testing", "--ct", dest="ct", default=("chr19",),
                       help=PARAM_DESC["chromosomal_testing"], nargs="+")
    group.add_argument("--accelerator", type=str, default="auto",
                       choices=("gpu", "tpu", "auto", "cpu", "ipu"), help=PARAM_DESC["accelerator"])
    group.add_argument("--devices", help=PARAM_DESC["devices"],
                       nargs="*", type=int, default=(0,))
    group.add_argument("--earlystop-patience", help=PARAM_DESC["earlystop_patience"],
                       type=int, default=2)
    group.add_argument("--min-delta", help=PARAM_DESC["min_delta"],
                       type=float, default=0.0001)
    group.add_argument("--max-epochs", help=PARAM_DESC["max_epochs"],
                       type=int, default=50)
    group.add_argument("--save-top-k-model", help=PARAM_DESC["save_top_k_model"],
                       default=1, type=int)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--save-preds", action="store_true", default=True, help=PARAM_DESC["save_preds"])
    g.add_argument("--no-preds", action="store_false", dest="save_preds", help=PARAM_DESC["no_preds"])
    group.add_argument("--redundancy-loss-coef", help=PARAM_DESC["redundancy_loss_coef"],
                       type=float, default=1.0)
    group.add_argument("--prior-loss-coef", help=PARAM_DESC["prior_loss_coef"],
                       type=float, default=1.0)
    group.add_argument("--gamma", help=PARAM_DESC["gamma"],
                       type=float, default=1e-8)
    group.add_argument("--learning-rate", help=PARAM_DESC["learning_rate"],
                       type=float, default=1e-3)
    group.add_argument("--betas", help=PARAM_DESC["betas"],
                       type=float, default=(0.9, 0.999), nargs=2)
    group.add_argument("--model-summary-depth", help=PARAM_DESC["max_depth"],
                       type=int, default=1)
    group.add_argument("--max-retry", help=PARAM_DESC["max_retry"],
                       type=int, default=3)
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=(0, 1, 2), default=1, type=int)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--all-regions", action="store_true", help=PARAM_DESC["all_regions"], default=True)
    g.add_argument("--peak-only", action="store_false", dest="all_regions", help=PARAM_DESC["peak_only"])
    group.add_argument("--test-all-regions", dest="test_pos_only", action="store_false",
                        help=PARAM_DESC["test_pos_only"])
    group.add_argument("-v", "--version-tag", dest="version", help=PARAM_DESC["wandb_version"],
                       type=str, default="")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _training_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Training")
    group.add_argument("--dataset", required=True, type=lambda _x: is_valid_file(_x, is_dir=True),
                       help=PARAM_DESC["dataset"])
    group.add_argument("--chrom-cv", action="store_true", help=PARAM_DESC["chrom_cv"])
    group.add_argument("--chromosomal-validation", "--cv", dest="cv", default=("chr22",),
                       help=PARAM_DESC["chromosomal_validation"], nargs="*")
    group.add_argument("--chromosomal-testing", "--ct", dest="ct", default=("chr19",),
                       help=PARAM_DESC["chromosomal_testing"], nargs="*")
    group.add_argument("--accelerator", type=str, default="auto",
                       choices=("gpu", "tpu", "auto", "cpu", "ipu"), help=PARAM_DESC["accelerator"])
    group.add_argument("--devices", help=PARAM_DESC["devices"],
                       nargs="*", type=int, default=(0,))
    group.add_argument("--earlystop-patience", help=PARAM_DESC["earlystop_patience"],
                       type=int, default=2)
    group.add_argument("--min-delta", help=PARAM_DESC["min_delta"],
                       type=float, default=0.0001)
    group.add_argument("--max-epochs", help=PARAM_DESC["max_epochs"],
                       type=int, default=50)
    group.add_argument("--save-top-k-model", help=PARAM_DESC["save_top_k_model"],
                       default=1, type=int)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--save-preds", action="store_true", default=True, help=PARAM_DESC["save_preds"])
    g.add_argument("--no-preds", action="store_false", dest="save_preds", help=PARAM_DESC["no_preds"])
    group.add_argument("--redundancy-loss-coef", help=PARAM_DESC["redundancy_loss_coef"],
                       type=float, default=1.0)
    group.add_argument("--prior-loss-coef", help=PARAM_DESC["prior_loss_coef"],
                       type=float, default=1.0)
    group.add_argument("--gamma", help=PARAM_DESC["gamma"],
                       type=float, default=1e-8)
    group.add_argument("--learning-rate", help=PARAM_DESC["learning_rate"],
                       type=float, default=1e-3)
    group.add_argument("--betas", help=PARAM_DESC["betas"],
                       type=float, default=(0.9, 0.999), nargs=2)
    group.add_argument("--model-summary-depth", help=PARAM_DESC["max_depth"],
                       type=int, default=1)
    group.add_argument("--max-retry", help=PARAM_DESC["max_retry"],
                       type=int, default=3)
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=(0, 1, 2), default=1, type=int)
    # for backward compatibility
    g = group.add_mutually_exclusive_group()
    g.add_argument("--all-regions", action="store_true", help=PARAM_DESC["all_regions"], default=True)
    g.add_argument("--peak-only", action="store_false", dest="all_regions", help=PARAM_DESC["peak_only"])
    group.add_argument("--test-all-regions", dest="test_pos_only", action="store_false",
                        help=PARAM_DESC["test_pos_only"])
    group.add_argument("-v", "--version-tag", dest="version", help=PARAM_DESC["wandb_version"],
                       type=str, default="")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _model_conf_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Model Configuration")
    group.add_argument("--profile-shrinkage", help=PARAM_DESC["profile_shrinkage"],
                       type=int, default=8, required=False)
    group.add_argument("--filters", help=PARAM_DESC["filters"], type=int, default=512)
    group.add_argument("--n-non-dilated-layers", dest="n_non_dil_layers",
                       type=int, default=1, help=PARAM_DESC["n_non_dil_layers"])
    group.add_argument("--non-dilated-kernel-size", dest="non_dil_kernel_size",
                       type=int, default=3, help=PARAM_DESC["non_dil_kernel_size"])
    group.add_argument("--n-dilated-layers", help=PARAM_DESC["n_dilated_layers"],
                       type=int, default=9)
    group.add_argument("--dilated-kernel-size", dest="dil_kernel_size",
                       type=int, default=4, help=PARAM_DESC["dil_kernel_size"])
    group.add_argument("--head-mlp-layers", dest="head_layers",
                       type=int, default=3, help=PARAM_DESC["head_mlp_layers"])
    group.add_argument("--conv1-kernel-size", help=PARAM_DESC["conv1_kernel_size"],
                       type=int, default=21)
    group.add_argument("--gru-layers", help=PARAM_DESC["gru_layers"],
                       type=int, default=2)
    group.add_argument("--gru-dropout", help=PARAM_DESC["gru_dropout"],
                       type=float, default=0.1)
    group.add_argument("--profile-kernel-size", help=PARAM_DESC["profile_kernel_size"],
                       type=int, default=9)
    group.add_argument("--scale-function-placement", choices=("early", "late", "late-ch", "disable"),
                       help=PARAM_DESC["scale_function_placement"], default="late-ch")
    group.add_argument("--seq", dest="seq_only", action="store_true", required=False,
                       help=PARAM_DESC["seq_only"])
    group.add_argument("--n-times-more-embeddings", help=PARAM_DESC["n_times_more_embeddings"],
                       type=int, default=2)


def _wandb_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("WandB")
    group.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT"))
    group.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    group.add_argument("--wandb-upload-model", action="store_true", required=False)


def _prep_dataset_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("")
    group.add_argument("--regions", nargs="+", required=True, type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["regions"])
    group.add_argument("--bulk-pl", required=True, type=lambda _x: is_valid_file(_x), help=PARAM_DESC["bulk_pl"])
    group.add_argument("--bulk-mn", type=lambda _x: is_valid_file(_x), help=PARAM_DESC["bulk_mn"])
    group.add_argument("--save-to", default=".", type=lambda _x: is_valid_file(_x, is_dir=True),
                       help=PARAM_DESC["save_to"])
    group.add_argument("--genome-fa", type=lambda _x: is_valid_file(_x), required=True, help=PARAM_DESC["genome_fa"])
    group.add_argument("--chrom-size", type=lambda _x: is_valid_file(_x), required=True, help=PARAM_DESC["chrom_size"])
    group.add_argument("--window-size", default=4096, type=int, help=PARAM_DESC["t_x"])
    group.add_argument("--seed", type=int, default=1234567, help=PARAM_DESC["seed"])
    group.add_argument("--ref-labels", nargs="*", type=str, help=PARAM_DESC["ref_labels"])
    group.add_argument("--ref-pls", nargs="*", type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_pls"])
    group.add_argument("--ref-mns", nargs="*", type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_mns"])
    group.add_argument("--background-sampling-ratio", help=PARAM_DESC["background_sampling_ratio"],
                       default=0., type=float)
    group.add_argument("--background-blacklist", type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["background_blacklist"])
    group.add_argument("--final-regions", action="store_true", help=PARAM_DESC["final_regions"])
    group.add_argument("--keep-frags", action="store_true", help=PARAM_DESC["keep_frags"])
    group.add_argument("--memory-saving", action="store_true", help=PARAM_DESC["memory_saving"])
    group.add_argument("--merge-overlap-peaks", help=PARAM_DESC["merge_overlap_peaks"],
                       type=int, default=0)
    group.add_argument("--target-sliding-sum", help=PARAM_DESC["target_sliding_sum"],
                       default=0, type=int)
    group0 = group.add_mutually_exclusive_group(required=True)
    group0.add_argument("--accessibility", nargs="+", type=lambda _x: is_valid_file(_x),
                        help=PARAM_DESC["accessibility"])
    group0.add_argument("--fragments", type=lambda _x: is_valid_file(_x), help=PARAM_DESC["fragments"])
    group.add_argument("--barcodes", type=lambda _x: is_valid_file(_x), help=PARAM_DESC["barcodes"])


def _preflight_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Preflight")
    group.add_argument("--skip-preflight", action="store_true", help=PARAM_DESC["skip_preflight"])
    group.add_argument("--accessible-regions", dest="accessible_peaks", required=False,
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["accessible_peaks"])
    group.add_argument("--preflight-cutoff", help=PARAM_DESC["preflight_cutoff"],
                       default=0.035, type=float)
    group.add_argument("--nu", help=PARAM_DESC["preflight_nu"],
                       default=0.85, type=float)
    group.add_argument("--use-qnorm", action="store_true", help=PARAM_DESC["use_qnorm"])
    group.add_argument("--candidate-qval", help=PARAM_DESC["qval_cutoff"],
                       default=0.01, type=float)
    group.add_argument("--candidate-fc", help=PARAM_DESC["fc_cutoff"],
                       default=2., type=float)
    group.add_argument("--max-top-n", help=PARAM_DESC["max_top_n"],
                       type=int, default=1000)
    group.add_argument("--min-cells-required", help=PARAM_DESC["min_cells_required"],
                       type=int, default=20)
    group.add_argument("--n-aggs", help=PARAM_DESC["n_aggs"],
                       type=int, default=5)


def _export_pred_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export HDF5")
    group.add_argument("--dataset", required=True, type=lambda _x: is_valid_file(_x, is_dir=True),
                       help=PARAM_DESC["dataset"])
    group.add_argument("-m", "--checkpoint", required=True, type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["checkpoint"])
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=(0, 1, 2), default=1, type=int)
    group.add_argument("--all-regions", dest="pos_only", action="store_false",
                       help=PARAM_DESC["all_regions"])
    group.add_argument("--merge-strands", action="store_true",
                       help=PARAM_DESC["merge_strands"])
    group.add_argument("--device", help=PARAM_DESC["device"], type=str, default="cuda")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _export_wg_pred_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export whole-genome HDF5")
    group.add_argument("--genome-fa", dest="fa_file", required=True,
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["genome_fa"])
    group.add_argument("--regions-file", type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["regions"])
    group.add_argument("--bulk-pl", dest="pl_bulk_bw_file", required=True,
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["bulk_pl"])
    group.add_argument("--bulk-mn", dest="mn_bulk_bw_file",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["bulk_mn"])
    group.add_argument("--accessibility", dest="acc_bw_files", nargs="+",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["accessibility"])
    group.add_argument("--ref-pls", dest="pl_ct_bw_files", nargs="*",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_pls"])
    group.add_argument("--ref-mns", dest="mn_ct_bw_files", nargs="*",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["ref_mns"])

    group.add_argument("--sc-norm", dest="sc_norm_file",
                       type=lambda _x: is_valid_file(_x), help=PARAM_DESC["sc_norm_file"])
    group.add_argument("--ref-labels", dest="cluster_names", nargs="*",
                       type=str, help=PARAM_DESC["ref_labels"])
    group.add_argument("--window-size", default=4096, type=int, help=PARAM_DESC["t_x"])
    group.add_argument("--target-sliding-sum", help=PARAM_DESC["target_sliding_sum"],
                       default=0, type=int)
    group.add_argument("--dataset-mode", dest="is_training", help=PARAM_DESC["is_training"],
                       choices=(0, 1, 2, -1), default=-1, type=int)

    group.add_argument("-m", "--checkpoint", required=True, type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["checkpoint"])
    group.add_argument("--rescaling-mode", dest="rescaling_mode", help=PARAM_DESC["rescaling_mode"],
                       choices=(0, 1, 2), default=1, type=int)
    group.add_argument("--all-regions", dest="pos_only", action="store_false",
                       help=PARAM_DESC["all_regions"])
    group.add_argument("--merge-strands", action="store_true",
                       help=PARAM_DESC["merge_strands"])
    group.add_argument("--use-bulk-constraint", action="store_true", required=False,
                       help=PARAM_DESC["use_bulk_constraint"])
    group.add_argument("--device", help=PARAM_DESC["device"], type=str, default="cuda")
    group.add_argument("--loads-trunc", required=False, type=int, help=PARAM_DESC["loads_trunc"])


def _build_bw_parser(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Export BigWig")
    group.add_argument("-p", "--pred-file", required=True, type=lambda _x: is_valid_file(_x),
                       help=PARAM_DESC["pred_file"])
    group.add_argument("-s", "--save-to", default=".", type=lambda _x: is_valid_file(_x, is_dir=True),
                       help=PARAM_DESC["save_to"])
    group.add_argument("-c", "--chrom-size", type=lambda _x: is_valid_file(_x), required=True,
                       help=PARAM_DESC["chrom_size"])
    group.add_argument("--min-abs-val", action="store", help=PARAM_DESC["min_abs_val"],
                       type=float, default=10e-3)
    group.add_argument("-f", "--fast", dest="skip_sort_merge", action="store_true",
                       help=PARAM_DESC["skip_sort_merge"])
    group.add_argument("--out-binning", dest="binning",
                       help=PARAM_DESC["out_binning"], type=int, default=0)
    group.add_argument("--num-workers", help=PARAM_DESC["num_workers"], type=int, default=16)


def _merge_preds(parent_parser: argparse.ArgumentParser):
    group = parent_parser.add_argument_group("Merge predictions from multipe replicate runs")
    g = group.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred-dir", type=lambda _x: is_valid_file(_x, is_dir=True), help=PARAM_DESC["pred_dir"])
    g.add_argument("--preds", type=lambda _x: is_valid_file(_x), nargs="+", help=PARAM_DESC["preds"])
    group.add_argument("--save-to", type=str, required=True, help=PARAM_DESC["save_to"])
    group.add_argument("--keep-old-preds", action="store_true", help=PARAM_DESC["keep_old_preds"])
    group.add_argument("--quiet", action="store_true", help=PARAM_DESC["quiet"])


def deepdetails():
    parser = argparse.ArgumentParser(description="DeepDETAILS", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-v", "--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(title="Available functions", dest="function")

    # Subparser for supervised training
    parser_train = subparsers.add_parser("superv_contrast", help="Train DeepDETAILS with reference datasets")
    _general_parser(parser_train)
    _supervised_training_parser(parser_train)
    _model_conf_parser(parser_train)
    _wandb_parser(parser_train)

    # Subparser for deconvolution
    parser_deconv = subparsers.add_parser("deconv", help="Using DeepDETAILS to deconvolve a bulk sample")
    _general_parser(parser_deconv)
    _training_parser(parser_deconv)
    _model_conf_parser(parser_deconv)
    _wandb_parser(parser_deconv)

    # Subparser for dataset preparation
    parser_prep_data = subparsers.add_parser("prep-data",
                                             help="Create a dataset for DeepDETAILS to deconvolve a bulk sample")
    _prep_dataset_parser(parser_prep_data)
    _preflight_parser(parser_prep_data)

    # Subparser for export prediction hdf5
    parser_export_pred = subparsers.add_parser("export-pred", help="Export predictions to a hdf5 file")
    _general_parser(parser_export_pred)
    _export_pred_parser(parser_export_pred)

    parser_gw_export_pred = subparsers.add_parser("export-gw-pred", help="Export whole-genome predictions to a hdf5 file")
    _general_parser(parser_gw_export_pred)
    _export_wg_pred_parser(parser_gw_export_pred)

    # Subparser for export bigwigs
    parser_build_bw = subparsers.add_parser("build-bw", help="Store predictions to BigWig files")
    _build_bw_parser(parser_build_bw)

    # Subparser for merging predictions from replicate runs
    parser_merge_preds = subparsers.add_parser("merge-preds", help="Merge predictions from multiple replicate runs")
    _merge_preds(parser_merge_preds)

    args = parser.parse_args()
    args_dict = vars(args).copy()
    args_dict.pop("function")

    if args.function == "deconv":
        deconv(**args_dict)
    elif args.function == "superv_contrast":
        superv_contrastive(**args_dict)
    elif args.function == "prep-data":
        if args.fragments and args.barcodes is None:
            parser.error("--fragments requires --barcodes to be specified")
        prepare_dataset(**args_dict)
    elif args.function == "export-pred":
        [args_dict.pop(k) for k in {"hide_progress_bar", "version"} if k in args_dict]
        export_results(model=DeepDETAILS, **args_dict)
    elif args.function == "export-gw-pred":
        [args_dict.pop(k) for k in {"hide_progress_bar", "version"} if k in args_dict]
        export_wg_results(model=DeepDETAILS, **args_dict)
    elif args.function == "build-bw":
        pred_to_bw(**args_dict)
    elif args.function == "merge-preds":
        pred_files = None
        if args.pred_dir:
            pred_files = glob(os.path.join(args.pred_dir, "*predictions.h5"))
        elif args.preds:
            pred_files = args.preds
        merge_rep_preds(
            in_pred_files=pred_files, save_to=args.save_to,
            keep_old=args.keep_old_preds, quiet=args.quiet,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    deepdetails()
