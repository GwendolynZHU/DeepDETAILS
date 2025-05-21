import torch
import torchmetrics
import pytorch_lightning as pl
from typing import Tuple, Optional
from einops import rearrange
from deepdetails.helper.inspection import bulk_visual_inspection, per_cluster_visual_inspection, gt_visual_inspection
from deepdetails.model.loss import RMSLELoss, off_diagonal
from deepdetails.model.deconvolution import Regressor, SeqOnlyRegressor, SupervisedRegressor
from deepdetails.helper.utils import transform_counts, calc_counts_per_locus
from deepdetails.par_description import PARAM_DESC


class SupervisedDeepDETAILS(pl.LightningModule):
    def __init__(self, num_cell_types: int, profile_shrinkage: int = 1, filters: int = 512,
                 n_non_dil_layers: int = 0, non_dil_kernel_size: int = 3, n_dil_layers: int = 8,
                 dil_kernel_size: int = 3, conv1_kernel_size: int = 21, gru_layers: int = 1,
                 gru_dropout: float = 0.1, profile_kernel_size: int = 75, head_mlp_layers: int = 3,
                 num_tasks: int = 2, redundancy_loss_coef: float = 1.,
                 prior_loss_coef: float = 1., learning_rate: float = 1e-3, version: str = "",
                 scale_function_placement: str = "disable", t_x: int = 4096, test_screenshot_ratio: float = 0.002,
                 gamma: float = 1e-8, n_times_more_embeddings: int = 2, betas: Tuple[float, float] = (0.9, 0.999),
                 seq_only: Optional[bool] = False, first_pass: Optional[bool] = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_cell_types = num_cell_types
        self.example_input_array = (
            (torch.swapaxes(
                torch.nn.functional.one_hot(
                    torch.randint(0, 4, size=(8, t_x))
                ), 1, 2).float(),
             torch.rand(8, num_cell_types, t_x)),
            torch.nn.functional.softmax(torch.rand(8, num_cell_types), dim=0),
        )
        self.first_pass = first_pass

        if seq_only:
            raise ValueError("Not implemented yet")
        else:
            self.model = SupervisedRegressor(
                num_cell_types=num_cell_types, filters=filters,
                n_non_dil_layers=n_non_dil_layers, non_dil_kernel_size=non_dil_kernel_size,
                n_dil_layers=n_dil_layers, dil_kernel_size=dil_kernel_size, profile_shrinkage=profile_shrinkage,
                conv1_kernel_size=conv1_kernel_size, profile_kernel_size=profile_kernel_size,
                gru_layers=gru_layers, gru_dropout=gru_dropout, n_times_more_embeddings=n_times_more_embeddings,
                counts_head_mlp_layers=head_mlp_layers, num_tasks=num_tasks,
                scale_function_placement=scale_function_placement)
            
        self.profile_loss_func = RMSLELoss()
        self.redundancy_loss_coef = redundancy_loss_coef
        self.learning_rate = learning_rate
        self.betas = betas
        self.pearsonr = torchmetrics.PearsonCorrCoef()
        self.val_pearsonr = torchmetrics.PearsonCorrCoef()
        self.test_pearsonr = torchmetrics.PearsonCorrCoef()
        self.test_pc_pearsons = torch.nn.ModuleList([torchmetrics.PearsonCorrCoef() for _ in range(num_cell_types)])
        self.version = version
        self.test_screenshot_ratio = test_screenshot_ratio
        self.gamma = gamma
        self.self_qc_values = []
        self.sum_qc_metrics = torch.zeros(num_cell_types, num_tasks)
        self.enable_sum_qc_metrics = False

    def forward(self, x, loads):
        return self.model(x, loads)
    
    def training_step(self, batch, batch_idx):
        x, expected_counts, expected_profiles, loads = batch
        pc_profiles, pc_counts, _, _ = self.model(x, loads)

        ct_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)

        losses = [
            self.profile_loss_func(ct_preds[i], expected_profiles[:, i])
            for i in range(self.num_cell_types)
        ]

        cell_type_corrs = []
        for i in range(self.num_cell_types):
            pred = ct_preds[i]  # shape: [B, 2, L]
            target = expected_profiles[:, i, :, :]
            corr = self.pearsonr(pred.flatten(), target.flatten())
            self.log(f"tr_corr_{i}", corr, on_epoch=True, prog_bar=False)
            cell_type_corrs.append(corr)
        train_mean_corr = torch.stack(cell_type_corrs).mean()
        self.log("train_corr", train_mean_corr, on_epoch=True, prog_bar=True)

        msle_loss = torch.stack(losses).mean()

        reshaped = rearrange(ct_preds, "c b s l -> c b (s l)")
        reshaped = reshaped + torch.arange(reshaped.shape[-1], device=self.device) * self.gamma

        if reshaped.shape[1] > 1:
            branch_corrs = torch.tensor(
                [off_diagonal(torch.corrcoef(sample)).pow_(2).mean() 
                 for sample in reshaped]
            ).mean()
            if batch_idx % 50 == 0:
                self.self_qc_values.append(branch_corrs.item())
        else:
            branch_corrs = torch.tensor(0)

        self.log("train_msle_loss", msle_loss, on_epoch=True, prog_bar=False)
        self.log("train_br_cor", branch_corrs, on_epoch=True, prog_bar=False)

        loss = msle_loss + branch_corrs * self.redundancy_loss_coef
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, expected_counts, expected_profiles, loads = batch # profiles: (B, C, S, L)
        pc_profiles, pc_counts, _, _ = self.model(x, loads)

        ct_preds = calc_counts_per_locus(pc_profiles, pc_counts, True) # (C, B, S, L)

        losses = [
            self.profile_loss_func(ct_preds[i], expected_profiles[:, i])
            for i in range(self.num_cell_types)
        ]
        msle_loss = torch.stack(losses).mean() # keeps loss magnitude consistent

        self.log("val_msle_loss", msle_loss, prog_bar=True)
        loss = msle_loss

        self.log("val_loss", loss, prog_bar=True)
        
        cell_type_corrs = []
        for i in range(self.num_cell_types):
            pred = ct_preds[i]                   # shape: [B, 2, L]
            target = expected_profiles[:, i]     # shape: [B, 2, L]

            corr = self.val_pearsonr(pred.flatten(), target.flatten())
            self.log(f"vcorr_{i}", corr, prog_bar=True)
            cell_type_corrs.append(corr)
        valid_corrs = [c for c in cell_type_corrs if not torch.isnan(c)]
        if valid_corrs:
            val_mean_corr = torch.stack(valid_corrs).mean()
        else:
            val_mean_corr = torch.tensor(0.0, device=self.device)

        self.log("val_corr", val_mean_corr, prog_bar=True)

        if torch.rand(1)[0] < 0.005:
            gt_visual_inspection(y_hats=ct_preds,
                                   y=expected_profiles.permute(1, 0, 2, 3),
                                   prefix=f"e{self.current_epoch}.b{x[0].sum().item():.4f}.s",
                                   logger=self.logger,
                                   cell_type_names=[f"ct_{i}" for i in range(self.num_cell_types)])

        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        x, expected_counts, expected_profiles, loads = batch
        
        pc_profiles, pc_counts, _, _ = self.model(x, loads)

        ct_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)

        # routine evaluation
        losses = [
            self.profile_loss_func(ct_preds[i], expected_profiles[:, i])
            for i in range(self.num_cell_types)
        ]
        msle_loss = torch.stack(losses).mean()
        self.log("test_loss", msle_loss, on_epoch=True)

        cell_type_corrs = []
        for i in range(self.num_cell_types):
            pred = ct_preds[i]                   # shape: [B, 2, L]
            target = expected_profiles[:, i]     # shape: [B, 2, L]

            corr = self.test_pearsonr(pred.flatten(), target.flatten())
            self.log(f"testcorr_{i}", corr, prog_bar=True)
            cell_type_corrs.append(corr)

        test_mean_corr = torch.stack(cell_type_corrs).mean()
        self.log("test_corr", test_mean_corr, prog_bar=True)

        # groundtruth-based evaluation
        if torch.rand(1)[0] < self.test_screenshot_ratio:
            gt_visual_inspection(y_hats=ct_preds,
                                   y=expected_profiles.permute(1, 0, 2, 3),
                                   prefix=f"preview{batch_idx}.{dataloader_idx}.{x[0].sum().item():.4f}.s",
                                   logger=self.logger,
                                   cell_type_names=[f"ct_{i}" for i in range(self.num_cell_types)])
                

        # per_celltype_total = torch.stack(pc_counts).clone().detach().sum(dim=0).to(self.sum_qc_metrics.device)  # (C,)
        # self.sum_qc_metrics += per_celltype_total
        # self.enable_sum_qc_metrics = True


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=self.betas)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class DeepDETAILS(pl.LightningModule):
    def __init__(self, expected_clusters: int, profile_shrinkage: int = 1, filters: int = 512,
                 n_non_dil_layers: int = 0, non_dil_kernel_size: int = 3, n_dil_layers: int = 8,
                 dil_kernel_size: int = 3, conv1_kernel_size: int = 21, gru_layers: int = 1,
                 gru_dropout: float = 0.1, profile_kernel_size: int = 75, head_mlp_layers: int = 3,
                 num_tasks: int = 2, first_pass: Optional[bool] = None, redundancy_loss_coef: float = 1.,
                 prior_loss_coef: float = 1., learning_rate: float = 1e-3, version: str = "",
                 scale_function_placement: str = "late-ch", t_x: int = 4096, test_screenshot_ratio: float = 0.002,
                 gamma: float = 1e-8, n_times_more_embeddings: int = 2, betas: Tuple[float, float] = (0.9, 0.999),
                 seq_only: Optional[bool] = False) -> None:
        """

        Parameters
        ----------
        expected_clusters : int
            {expected_clusters}
        profile_shrinkage : int
            {profile_shrinkage}
        filters : int
            {filters}
        n_non_dil_layers : int
            {n_non_dil_layers}
        non_dil_kernel_size : int
            {non_dil_kernel_size}
        n_dil_layers : int
            {n_dilated_layers}
        dil_kernel_size : int
            {dil_kernel_size}
        conv1_kernel_size : int
            {conv1_kernel_size}
        gru_layers : int
            {gru_layers}
        gru_dropout : float
            {gru_dropout}
        profile_kernel_size : int
            {profile_kernel_size}
        head_mlp_layers : int
            {head_mlp_layers}
        num_tasks : int
            {num_tasks}
        first_pass : Optional[bool]
            {first_pass}
        redundancy_loss_coef : float
            {redundancy_loss_coef}
        prior_loss_coef : float
            {prior_loss_coef}
        learning_rate : float
            {learning_rate}
        version : str
            {wandb_version}
        scale_function_placement : str
            {scale_function_placement}
        t_x : int
            {t_x}
        test_screenshot_ratio : float
            {test_screenshot_ratio}
        gamma : float
            {gamma}
        n_times_more_embeddings : int
            {n_times_more_embeddings}
        betas : Tuple[float, float]
            {betas}
        seq_only : Optional[bool]
            {seq_only}
        """.format(**PARAM_DESC)
        super().__init__()
        self.save_hyperparameters()
        self.expected_clusters = expected_clusters
        self.example_input_array = (
            (torch.swapaxes(
                torch.nn.functional.one_hot(
                    torch.randint(0, 4, size=(8, t_x))
                ), 1, 2).float(),
             torch.rand(8, expected_clusters, t_x)),
            torch.nn.functional.softmax(torch.rand(8, expected_clusters), dim=0),
        )
        self.first_pass = first_pass

        if seq_only:
            self.model = SeqOnlyRegressor(
                expected_clusters=expected_clusters, filters=filters,
                n_non_dil_layers=n_non_dil_layers, non_dil_kernel_size=non_dil_kernel_size,
                n_dil_layers=n_dil_layers, dil_kernel_size=dil_kernel_size,
                conv1_kernel_size=conv1_kernel_size, profile_kernel_size=profile_kernel_size,
                counts_head_mlp_layers=head_mlp_layers, num_tasks=num_tasks,
                scale_function_placement=scale_function_placement
            )
        else:
            self.model = Regressor(
                expected_clusters=expected_clusters, filters=filters,
                n_non_dil_layers=n_non_dil_layers, non_dil_kernel_size=non_dil_kernel_size,
                n_dil_layers=n_dil_layers, dil_kernel_size=dil_kernel_size, profile_shrinkage=profile_shrinkage,
                conv1_kernel_size=conv1_kernel_size, profile_kernel_size=profile_kernel_size,
                gru_layers=gru_layers, gru_dropout=gru_dropout, n_times_more_embeddings=n_times_more_embeddings,
                counts_head_mlp_layers=head_mlp_layers, num_tasks=num_tasks,
                scale_function_placement=scale_function_placement)

        self.profile_loss_func = RMSLELoss()

        self.redundancy_loss_coef = redundancy_loss_coef
        self.prior_loss_coef = prior_loss_coef
        self.learning_rate = learning_rate
        self.betas = betas
        self.pearsonr = torchmetrics.PearsonCorrCoef()
        self.val_pearsonr = torchmetrics.PearsonCorrCoef()
        # ct: counts, r: Pearson's R, rr: "Rank" R, sl: strandless
        self.test_pearsonr = torchmetrics.PearsonCorrCoef()
        self.test_pc_pearsons = torch.nn.ModuleList([torchmetrics.PearsonCorrCoef() for _ in range(expected_clusters)])
        self.version = version
        self.test_screenshot_ratio = test_screenshot_ratio
        self.gamma = gamma

        self.self_qc_values = []
        self.sum_qc_metrics = torch.zeros(expected_clusters * num_tasks)
        self.enable_sum_qc_metrics = False

    def forward(self, x, loads):
        return self.model(x, loads)

    def training_step(self, batch, batch_idx):
        x, expected_counts, expected_profiles, _, loads, misc = batch

        pc_profiles, pc_counts, _, _ = self.model(x, loads)

        cs_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)
        preds = cs_preds.sum(dim=0)

        msle_loss = self.profile_loss_func(preds, expected_profiles)
        self.log("train_msle_loss", msle_loss, prog_bar=True, on_step=True)

        reshaped = rearrange(cs_preds, "c b s l -> c b (s l)")
        reshaped = reshaped + torch.arange(reshaped.shape[-1], device=self.device) * self.gamma

        if reshaped.shape[1] > 1:
            branch_corrs = torch.tensor(
                [off_diagonal(torch.corrcoef(sample)).pow_(2).mean() for sample in reshaped]).mean()
            if batch_idx % 50 == 0:
                self.self_qc_values.append(branch_corrs.item())
        else:
            branch_corrs = torch.tensor(0)
        self.log("train_br_cor", branch_corrs, on_step=True)

        if misc[-1].dim() == 2:
            prior = misc[-1][:, :]
            cluster_preds = rearrange(
                cs_preds + torch.arange(cs_preds.shape[-1], device=self.device) * self.gamma,
                "c b s l -> c (b s l)")
            observed_corrs = torch.corrcoef(cluster_preds)
            prior_loss = (prior - observed_corrs).pow_(2).mean()
            self.log("train_prior_loss", prior_loss, on_step=True)
            loss = msle_loss + branch_corrs * self.redundancy_loss_coef + prior_loss * self.prior_loss_coef
        else:
            loss = msle_loss + branch_corrs * self.redundancy_loss_coef

        cor = self.pearsonr(
            transform_counts(preds.flatten()),
            transform_counts(expected_profiles.flatten())
        )

        self.log("train_loss", loss, on_epoch=True,
                 on_step=True, prog_bar=True)

        self.log("train_corr", cor, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, expected_counts, expected_profiles, _, loads, misc = batch

        pc_profiles, pc_counts, _, _ = self.model(x, loads)
        preds = calc_counts_per_locus(pc_profiles, pc_counts, False)

        msle_loss = self.profile_loss_func(preds, expected_profiles)
        self.log("val_msle_loss", msle_loss, prog_bar=True)
        loss = msle_loss

        self.log("val_loss", loss, prog_bar=True)

        val_cor = self.val_pearsonr(
            transform_counts(preds.flatten()),
            transform_counts(expected_profiles.flatten())
        )

        if torch.rand(1)[0] < 0.1:
            bulk_visual_inspection(preds, expected_profiles, calc_counts_per_locus(pc_profiles, pc_counts, True),
                                   f"e{self.current_epoch}.b{x[0].sum().item():.4f}.s", logger=self.logger)

        self.log("val_corr", val_cor, prog_bar=True)

        return loss

    def _groundtruth_based_eval(self, per_cluster_y: torch.Tensor, per_cluster_y_hat: torch.Tensor):
        """

        Parameters
        ----------
        per_cluster_y : torch.Tensor
            shape: (# cluster, batch, strand, seq_len)
        per_cluster_y_hat : torch.Tensor
            shape: (# cluster, batch, strand, seq_len)

        Returns
        -------

        """
        y_hats_list = []
        if per_cluster_y.shape[0] == per_cluster_y_hat.shape[0]:
            for i, real_profiles in enumerate(per_cluster_y):
                y_hats = per_cluster_y_hat[i]
                y_hats_list.append(y_hats)
                x = torch.stack([
                    transform_counts(y_hats.flatten()),
                    transform_counts(real_profiles.flatten())
                ])
                test_cor = torch.corrcoef(x)[0, 1]
                if torch.isnan(test_cor):
                    test_cor = 0.
                self.log(f"test_corr_{i}", test_cor, on_epoch=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        x, expected_counts, expected_profiles, expected_per_cluster_profiles, loads, _ = batch

        pc_profiles, pc_counts, pc_weights, _ = self.model(x, loads)

        preds = calc_counts_per_locus(pc_profiles, pc_counts, False)
        cluster_preds = calc_counts_per_locus(pc_profiles, pc_counts, True)  # cluster, batch, strand, seq_len

        # routine evaluation
        msle_loss = self.profile_loss_func(preds, expected_profiles)

        test_cor = self.test_pearsonr(
            transform_counts(preds.flatten()),
            transform_counts(expected_profiles.flatten())
        )

        # groundtruth-based evaluation
        if len(expected_per_cluster_profiles) > 0:
            self._groundtruth_based_eval(torch.stack(expected_per_cluster_profiles), cluster_preds)

            if torch.rand(1)[0] < self.test_screenshot_ratio:
                if isinstance(pc_weights, tuple):
                    pc_weights = torch.zeros(loads.shape[0], loads.shape[1])
                per_cluster_visual_inspection(
                    preds, cluster_preds, expected_profiles,
                    torch.stack(expected_per_cluster_profiles, dim=0),
                    loads, pc_weights,
                    f"preview{batch_idx}.{dataloader_idx}.{x[0].sum().item():.4f}.s", logger=self.logger)
        else:  # no groundtruth, only plot preds
            if torch.rand(1)[0] < self.test_screenshot_ratio:
                bulk_visual_inspection(preds, expected_profiles, cluster_preds,
                                       f"preview{batch_idx}.{dataloader_idx}.{x[0].sum().item():.4f}.s",
                                       logger=self.logger)

        self.log("test_loss", msle_loss, on_epoch=True)
        self.log("test_corr", test_cor, prog_bar=True, on_epoch=True)

        per_cluster_per_strand_total = torch.stack(pc_counts).clone().detach().sum(axis=1).flatten().to(
            self.sum_qc_metrics.device)
        self.sum_qc_metrics = self.sum_qc_metrics + per_cluster_per_strand_total
        self.enable_sum_qc_metrics = True

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=self.betas)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
