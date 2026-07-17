PARAM_DESC = {
    "dataset": "Path to the dataset folder",
    "expected_clusters": "Number of expected clusters/cell types in the bulk sample.",
    "num_tasks": "Number of predicted strands",
    "t_x": "Length of the inputs",
    "y_length": "Length of the final prediction",
    "first_pass": "First pass of the training",
    "seq_only": "Use sequence only model",
    "modulation_only": "Use chromatin to modulate sequence embeddings without direct chromatin fusion into prediction heads",
    "redundancy_loss_coef": "Redundancy loss coefficient",
    "prior_loss_coef": "Prior loss coefficient",
    "disable_final_rescaling": "By default, DETAILS adjusts the final exported predictions so they have scales identical"
                               " to the bulk library. By setting this switch, you can disable this adjustment.",
    "rescaling_mode": "0: No rescaling, 1: Counts-level rescaling, 2: BP-level rescaling",
    "gamma": "This small scalar is applied to the injected linear pattern in calculating redundancy "
             "reduction terms for preventing trivial solutions.",
    "save_preds": "Set this switch to save final predictions. Specify --no-preds if you don't want to save predictions.",
    "no_preds": "Do not export predictions.",
    "all_regions": "Train models with both peak and background regions. This is the default behavior. "
                   "Specify --peak-only if you want to only use peak regions",
    "peak_only": "Train models with peak regions only.",
    "test_pos_only": "Evaluate models only on non-background regions.",
    "chrom_cv": "Use chromosomal cross-validation.",
    "chromosomal_validation": "Chromosomes kept for validation. Only effective when chrom_cv is specified.",
    "chromosomal_testing": "Chromosomes kept for testing. Only effective when chrom_cv is specified.",
    "test_screenshot_ratio": "Ratio of screenshots across batches",
    # parameters for the dataset
    "regions": "Peak regions. Multiple files allowed.",
    "bulk_pl": "Bulk signal for the forward strand.",
    "bulk_mn": "Bulk signal for the reverse strand (optional).",
    "accessibility": "Cluster/cell type specific accessibility tracks, preferably depth normalized.",
    "fragments": "Fragment file for the sc/snATAC experiment (in tsv format).",
    "barcodes": "A tab-separated file describing cell type annotations for cells captured in the sc/snATAC experiment."
                "It should have two columns (no header): "
                "* the first column stores the cell barcode "
                "* the second column stores the cluster label. "
                "The barcodes in this file should match barcodes in fragments.",
    "chrom_size": "Path to the chromosome size file",
    "genome_fa": "Path to the genome sequence file",
    "background_sampling_ratio": "Sample background regions according to the number of peak regions. If ratio is 1, equal number will be sampled.",
    "ref_labels": "Label for each cell type/cluster",
    "ref_pls": "Forward strand signal (in BigWig file) for each cell type/cluster",
    "ref_mns": "Reverse strand signal (in BigWig file) for each cell type/cluster",
    "background_blacklist": "Blacklist regions to be excluded from the dataset",
    "final_regions": "Whether or not the regions are final. Final means no extension and no more background sampling.",
    "target_sliding_sum": "Apply sliding sum to the target signals if the value is greater than 0.",
    "seed": "Random seed",
    "merge_overlap_peaks": "Minimum overlap between features allowed for features to be merged. 0 for not to merge.",
    "loads_trunc": "Whether to set separate truncation value for calculating loads. If None, y_truncation will be used.",
    "keep_frags": "Keep extracted fragments for each cell type/cluster",
    "skip_preflight": "Skip checking clusters in the snATAC experiment that may not present in the bulk library",
    "accessible_peaks": "Accessible regions identified from the snATAC experiment.",
    "memory_saving": "Try to use solutions that more memory friendly.",
    "preflight_nu": "nu for NuSVR, see sklearn's documentation for more information. Range: [0., 1.]",
    "preflight_cutoff": "Clusters with diagnostic values higher than this will be excluded. Range: [0., 1.]",
    "min_cells_required": "Minimum number of cells required for a cell type in the reference set to be considered.",
    "use_qnorm": "Apply quantile normalization to the single-cell reference",
    "n_aggs": "Number of aggregated pseudo-replicates for identifying differentially accessible regions.",
    "max_top_n": "Maximum number of signature candidates to consider for each cluster",
    "qval_cutoff": "Q-value cutoff for a region to be considered a signature",
    "fc_cutoff": "Fold-change cutoff for a region to be considered a signature",
    # parameters for a dynamic dataset
    "fa_file": "Genome fasta file, should have a companion index file (.fai)",
    "sc_norm_file": "Path to a CSV file with two columns: cluster name and normalization factor. No header, no index.",
    "is_training": "0 : validation, 1 : training, 2 : testing, by default 1",
    "use_bulk_constraint": "Mark all regions with no bulk signal as background regions",
    # parameters for the model
    "profile_shrinkage": "# GRU filters for the accessibility profiles will be # CNN filters / profile_shrinkage",
    "filters": "# CNN filters",
    "n_non_dil_layers": "# non-dilation CNN layers",
    "non_dil_kernel_size": "Filter size for non-dilation CNN layers",
    "n_dilated_layers": "# dilated CNN layers",
    "dil_kernel_size": "Filter size for dilated CNN layers",
    "head_mlp_layers": "# layers for head MLPs",
    "conv1_kernel_size": "Filter size for the first CNN layer",
    "gru_layers": "# GRU layers",
    "gru_dropout": "Dropout for GRU layers",
    "profile_kernel_size": "Filter size for the final profile output",
    "scale_function_placement": "Placement for the scale function",
    "n_times_more_embeddings": "upscale sequence embedding by this factor",
    "max_retry": "The maximum retry if a model fails internal QC",
    # parameters for exporting predictions
    "checkpoint": "Checkpoint for a trained model",
    "pred_file": "Path to a hdf5 file containing all the predictions",
    "min_abs_val": "Minimum absolute value for a predicted value to be exported",
    "merge_strands": "Merge strands when exporting predictions",
    "out_binning": "If given a positive value (the number of bins), export the binned stats instead of the original bp-resolution predictions",
    "skip_sort_merge": "If the input bg_file is sorted and there are no overlapping regions (or they get merged before), set this as True to speed up computation.",
    # parameters for Pytorch Lightning and WandB (descriptions taken from their docs)
    "study_name": "Study name. Avoid spaces and special characters.",
    "save_to": "Path where data is saved.",
    "batch_size": "Batch size",
    "num_workers": "How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    "min_delta": "Minimum change in the monitored quantity to qualify as an improvement,"
                 "i.e. an absolute change of less than or equal to min_delta, will count as no improvement.",
    "earlystop_patience": "Number of checks with no improvement after which training will be stopped."
                          "Under the default configuration, one check happens after every training epoch.",
    "max_epochs": "Stop training once this number of epochs is reached.",
    "learning_rate": "Learning rate",
    "betas": "Coefficients used for computing running averages of gradient and its square",
    "save_top_k_model": "The best k models according to the quantity monitored will be saved. "
                        "If k == 0, no models are saved. If k == -1, all models are saved.",
    "hide_progress_bar": "Whether to hide the progress bar.",
    "max_depth": "The maximum depth of layer nesting that the summary will include.",
    "accelerator": "Accelerator for training / inference. Values can be 'gpu', 'tpu', 'auto', 'cpu', or 'ipu'.",
    "devices": "The devices to use. Can be set to a positive number (int), a sequence of device indices"
               "(list), the value -1 to indicate all available devices should be used, or 'auto' for"
               "automatic selection based on the chosen accelerator.",
    "device": "Use this device to do calculation",
    "wandb_version": "Version for WandB logging",
    "wandb_project": "The name of the project to which this run will belong."
                     "If not set, the environment variable WANDB_PROJECT will be used as a fallback."
                     "If both are not set, it defaults to 'lightning_logs'.",
    "wandb_entity": "WandB username or team name. This entity must exist before you can send runs to WandB's server.",
    "wandb_upload_model": "Log checkpoints as W&B artifacts. Latest and best aliases are automatically set."
                          "'all': checkpoints are logged during training."
                          "True: checkpoints are logged at the end of training, except when save_top_k_model == -1"
                          "which also logs every checkpoint during training."
                          "False: (default), no checkpoint is logged.",
    # parameters for merging predictions
    "pred_dir": "Path to the folder storing all replicate predictions",
    "preds": "Path to hdf5 files for predictions from each replicate run",
    "keep_old_preds": "Keep predictions from each replicate run",
    "quiet": "Disable progress bar.",
}
