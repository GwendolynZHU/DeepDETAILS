from math import gamma
import torch
from typing import Tuple
from torch import nn
from . import ResidualConv, ResidualConvWithXProjection


class BaseRegressor(nn.Module):
    def __init__(self, filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
                 n_dil_layers=8, dil_kernel_size=3, profile_kernel_size=75,
                 counts_head_mlp_layers=3, num_tasks=1) -> None:
        super().__init__()
        self.body = nn.Sequential()

        for _ in range(n_non_dil_layers):
            self.body.append(
                ResidualConv(filters=filters, kernel_size=non_dil_kernel_size,
                             dilation_rate=1)
            )
        for i in range(n_dil_layers):
            self.body.append(
                ResidualConv(filters=filters, kernel_size=dil_kernel_size,
                             dilation_rate=2 ** (i + 1))
            )

        self.shape_head = nn.LazyConv1d(
            num_tasks, kernel_size=profile_kernel_size, padding="valid")
        self.counts_head = nn.Sequential()
        for _ in range(counts_head_mlp_layers):
            self.counts_head.append(nn.LazyLinear(filters))
            self.counts_head.append(nn.ReLU())
        self.counts_head.append(nn.LazyLinear(num_tasks))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x : torch.Tensor
            Shape: batch, channels, seq_len

        Returns
        -------
        profile : torch.Tensor
            Shape: batch, num_tasks, seq_len_1
        counts : torch.Tensor
            Shape: batch, num_tasks
        """
        body = self.body(x)  # (batch, filters, remaining_length)
        shape = self.shape_head(body)  # (batch, strands, target_length)
        # shape = shape.squeeze()
        body_gap = body.mean(axis=2)  # (batch, filters)
        counts = self.counts_head(body_gap)
        return nn.functional.softmax(shape, dim=2), nn.functional.softplus(counts)


class SeqOnlyBaseRegressor(nn.Module):
    def __init__(self, filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
                 n_dil_layers=8, dil_kernel_size=3):
        super().__init__()
        self.body = nn.Sequential()

        for _ in range(n_non_dil_layers):
            self.body.append(
                ResidualConv(filters=filters, kernel_size=non_dil_kernel_size,
                             dilation_rate=1)
            )
        for i in range(n_dil_layers):
            self.body.append(
                ResidualConv(filters=filters, kernel_size=dil_kernel_size,
                             dilation_rate=2 ** (i + 1))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape: batch, channels, seq_len
        Returns
        -------
        body : torch.Tensor
            Shape: batch, filters, seq_len_1
        """
        body = self.body(x)  # (batch, filters, remaining_length)
        return body
    

class SeqOnlyRegressor(nn.Module):
    def __init__(
            self, expected_clusters: int,
            filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
            n_dil_layers=8, dil_kernel_size=3, conv1_kernel_size=21, profile_kernel_size=75,
            counts_head_mlp_layers=3, num_tasks=1, n_times_more_embeddings=2,
            scale_function_placement: str = "late-ch") -> None:

        super().__init__()
        self.expected_clusters = expected_clusters

        # first convolution without dilation
        self.motif_detector = nn.Conv1d(
            4, filters, kernel_size=conv1_kernel_size, padding="valid")

        self.cluster_regressor = nn.ModuleList()

        for _ in range(self.expected_clusters):
            self.cluster_regressor.append(
                BaseRegressor(
                    filters=filters,
                    n_non_dil_layers=n_non_dil_layers,
                    non_dil_kernel_size=non_dil_kernel_size,
                    n_dil_layers=n_dil_layers,
                    dil_kernel_size=dil_kernel_size,
                    profile_kernel_size=profile_kernel_size,
                    counts_head_mlp_layers=counts_head_mlp_layers,
                    num_tasks=num_tasks)
            )

        self.scale_function_placement = scale_function_placement

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], per_cluster_load: torch.Tensor) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """Forward propagation

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            seq: Shape: batch, ATCG, window_size
            atac: Shape: batch, n_clusters, window_size
        per_cluster_load : torch.Tensor
            Prior about per cluster loads

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]
            1. First element is per cluster profile (Shape: batch, strands, window_size).
            2. Second element is per cluster counts (Shape: batch, strands).
            3. Per cluster weights (Shape: batch, n_clusters)
            4. Per cluster motif gates (Shape: batch, filters)
            Elements in per cluster profiles can be directly used for the imputation of
            pseudo-bulk initiation patterns.
        """
        seq, atac = x
        motifs = self.motif_detector(seq)  # shape: batch, filters_1, seq_len

        cluster_weights = per_cluster_load
        per_cluster_profiles = []
        per_cluster_counts = []
        per_cluster_activations = []

        for cluster_id in range(self.expected_clusters):
            cw = cluster_weights[:, cluster_id]

            if self.scale_function_placement == "early":
                cluster_profile, cluster_counts = self.cluster_regressor[cluster_id](
                    motifs * cw[:, None, None])
            else:
                cluster_profile, cluster_counts = self.cluster_regressor[cluster_id](motifs)
            if self.scale_function_placement == "late":
                per_cluster_profiles.append(cluster_profile * cw[:, None, None])
                per_cluster_counts.append(cluster_counts * cw[:, None])
            elif self.scale_function_placement == "late-ch":
                per_cluster_profiles.append(cluster_profile)
                per_cluster_counts.append(cluster_counts * cw[:, None])
            else:
                per_cluster_profiles.append(cluster_profile)
                per_cluster_counts.append(cluster_counts)

        return per_cluster_profiles, per_cluster_counts, cluster_weights, per_cluster_activations


class PerClusterHead(nn.Module):
    def __init__(self, shape_filters=512, profile_kernel_size=75, counts_head_mlp_layers=3, num_tasks=1) -> None:
        super().__init__()

        self.shape_head = nn.Sequential()
        for _ in range(counts_head_mlp_layers):
            self.shape_head.append(nn.LazyConv1d(shape_filters, kernel_size=1))
            self.shape_head.append(nn.ELU())
        self.shape_head.append(nn.LazyConv1d(num_tasks, kernel_size=profile_kernel_size, padding="valid"))
        self.counts_head = nn.Sequential()
        for _ in range(counts_head_mlp_layers):
            self.counts_head.append(nn.LazyLinear(shape_filters, bias=False))
            self.counts_head.append(nn.ELU())
        self.counts_head.append(nn.LazyLinear(num_tasks))
        self.shape_act = nn.Softmax(dim=2)
        self.counts_act = nn.Softplus()

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x : (torch.Tensor, torch.Tensor)
            Seq: shape: batch, channels_1, seq_len
            ATAC: shape: batch, channels_2, seq_len

        Returns
        -------
        profile : torch.Tensor
            Shape: batch, num_tasks, seq_len_1
        counts : torch.Tensor
            Shape: batch, num_tasks
        """
        seq_gap = x[0].mean(axis=2)

        shape = self.shape_head(torch.hstack([x[0], x[1]]))  # (batch, strands, target_length)
        counts = self.counts_head(torch.hstack([seq_gap, x[1].mean(axis=2)]))
        return self.shape_act(shape), self.counts_act(counts)


class PerClusterHeadSeqOnly(nn.Module):
    def __init__(self, shape_filters=512, profile_kernel_size=75, 
                 head_layers=3, num_tasks=1) -> None:
        super().__init__()

        self.shape_head = nn.Sequential()
        for _ in range(head_layers):
            self.shape_head.append(nn.LazyConv1d(shape_filters, kernel_size=1))
            self.shape_head.append(nn.ELU())
        self.shape_head.append(nn.LazyConv1d(num_tasks, kernel_size=profile_kernel_size, padding="valid"))
        self.counts_head = nn.Sequential()
        for _ in range(head_layers):
            self.counts_head.append(nn.LazyLinear(shape_filters, bias=False))
            self.counts_head.append(nn.ELU())
        self.counts_head.append(nn.LazyLinear(num_tasks))
        self.shape_act = nn.Softmax(dim=2)
        self.counts_act = nn.Softplus()

    def forward(self, x: torch.Tensor, return_logits=False) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x : torch.Tensor
            Seq: shape: batch, channels_1, seq_len

        Returns
        -------
        profile : torch.Tensor
            Shape: batch, num_tasks, seq_len_1
        counts : torch.Tensor
            Shape: batch, num_tasks
        """
        seq_gap = x.mean(axis=2)

        shape = self.shape_head(x)  # (batch, strands, target_length)
        counts = self.counts_head(seq_gap)
        if return_logits:
            return shape, counts
        else:
            return self.shape_act(shape), self.counts_act(counts)
    

class Regressor(nn.Module):
    def __init__(
            self, expected_clusters: int,
            profile_shrinkage=1, filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
            n_dil_layers=8, dil_kernel_size=3, conv1_kernel_size=21, profile_kernel_size=75,
            counts_head_mlp_layers=3, num_tasks=1, gru_layers=1, gru_dropout=0.1, n_times_more_embeddings=2,
            scale_function_placement: str = "late-ch") -> None:

        super().__init__()
        self.expected_clusters = expected_clusters
        n_profile_filters = int(filters / profile_shrinkage)

        # first convolution without dilation
        self.motif_detector = nn.Sequential(nn.Conv1d(
            4, filters, kernel_size=conv1_kernel_size, padding="valid"))
        self.filter_gates = nn.ModuleList(
            [nn.LazyLinear(filters * n_times_more_embeddings, bias=False) for _ in range(expected_clusters)])
        self.profile_refiner = nn.GRU(input_size=1, hidden_size=n_profile_filters,
                                      num_layers=gru_layers, dropout=gru_dropout,
                                      batch_first=True, bidirectional=True)

        for _ in range(n_non_dil_layers):
            self.motif_detector.append(
                ResidualConv(filters=filters, kernel_size=non_dil_kernel_size,
                             dilation_rate=1)
            )

        for i in range(n_dil_layers):
            if i < n_dil_layers - 1:
                self.motif_detector.append(
                    ResidualConv(filters=filters, kernel_size=dil_kernel_size,
                                 dilation_rate=2 ** (i + 1))
                )
            else:
                self.motif_detector.append(
                    ResidualConvWithXProjection(filters=filters * n_times_more_embeddings, kernel_size=dil_kernel_size,
                                                dilation_rate=2 ** (i + 1))
                )

        self.per_cluster_preds = nn.ModuleList()
        for _ in range(self.expected_clusters):
            self.per_cluster_preds.append(
                PerClusterHead(
                    shape_filters=filters, profile_kernel_size=profile_kernel_size,
                    counts_head_mlp_layers=counts_head_mlp_layers, num_tasks=num_tasks
                ))

        self.scale_function_placement = scale_function_placement

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], per_cluster_load: torch.Tensor) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """Forward propagation

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            seq: Shape: batch, ATCG, window_size
            atac: Shape: batch, n_clusters, window_size
        per_cluster_load : torch.Tensor
            Prior about per cluster loads

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]
            1. First element is per cluster profile (Shape: batch, strands, window_size).
            2. Second element is per cluster counts (Shape: batch, strands).
            3. Per cluster weights (Shape: batch, n_clusters)
            4. Per cluster motif gates (Shape: batch, filters)
            Elements in per cluster profiles can be directly used for the imputation of
            pseudo-bulk initiation patterns.
        """
        seq, atac = x
        motifs = self.motif_detector(seq)  # shape: batch, filters_1, seq_len
        motifs_gap = motifs.mean(axis=2)
        atac_truncation = (atac.shape[-1] - motifs.shape[-1]) // 2

        cluster_weights = per_cluster_load
        per_cluster_profiles = []
        per_cluster_counts = []
        per_cluster_activations = []

        for cluster_id in range(self.expected_clusters):
            cw = cluster_weights[:, cluster_id]
            profiles, _ = self.profile_refiner(atac[:, cluster_id, :].unsqueeze(2))  # shape: batch, filters_2, seq_len
            profiles = torch.swapaxes(profiles, 1, 2)[:, :, atac_truncation:-atac_truncation]

            gates = torch.sigmoid(self.filter_gates[cluster_id](motifs_gap))
            filtered_activations = motifs * gates[:, :, None]

            if self.scale_function_placement == "early":
                cluster_profile, cluster_counts = self.per_cluster_preds[cluster_id](
                    (filtered_activations * cw[:, None, None], profiles))
            else:
                cluster_profile, cluster_counts = self.per_cluster_preds[cluster_id]((filtered_activations, profiles))
            if self.scale_function_placement == "late":
                per_cluster_profiles.append(cluster_profile * cw[:, None, None])
                per_cluster_counts.append(cluster_counts * cw[:, None])
            elif self.scale_function_placement == "late-ch":
                per_cluster_profiles.append(cluster_profile)
                per_cluster_counts.append(cluster_counts * cw[:, None])
            else:
                per_cluster_profiles.append(cluster_profile)
                per_cluster_counts.append(cluster_counts)
            per_cluster_activations.append(gates)

        return per_cluster_profiles, per_cluster_counts, cluster_weights, per_cluster_activations


class SupervisedRegressor(nn.Module):
    def __init__(self, num_cell_types: int, profile_shrinkage=8, filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
                 n_dil_layers=8, dil_kernel_size=3, conv1_kernel_size=21, profile_kernel_size=75,
                 counts_head_mlp_layers=3, num_tasks=2, gru_layers=1, gru_dropout=0.1,
                 n_times_more_embeddings=2, scale_function_placement: str = "disable") -> None:
        
        super().__init__()
        self.num_cell_types = num_cell_types
        n_profile_filters = int(filters / profile_shrinkage)
        
        # Shared motif detector
        self.motif_detector = nn.Sequential(nn.Conv1d(
            4, filters, kernel_size=conv1_kernel_size, padding="valid"))
        
        for _ in range(n_non_dil_layers):
            self.motif_detector.append(
                ResidualConv(filters=filters, kernel_size=non_dil_kernel_size,
                             dilation_rate=1)
            )

        for i in range(n_dil_layers):
            if i < n_dil_layers - 1:
                self.motif_detector.append(
                    ResidualConv(filters=filters, kernel_size=dil_kernel_size,
                                 dilation_rate=2 ** (i + 1))
                )
            else:
                self.motif_detector.append(
                    ResidualConvWithXProjection(filters=filters * n_times_more_embeddings, kernel_size=dil_kernel_size,
                                                dilation_rate=2 ** (i + 1))
                )
        
        # Shared GRU for profile refinement
        self.filter_gates = nn.ModuleList(
            [nn.LazyLinear(filters * n_times_more_embeddings, bias=False) for _ in range(num_cell_types)])

        self.profile_refiner = nn.GRU(input_size=1, hidden_size=n_profile_filters,
                                      num_layers=gru_layers, dropout=gru_dropout,
                                      batch_first=True, bidirectional=True)
        
        # Per cell type heads
        self.per_cell_type_heads = nn.ModuleList()
        for _ in range(self.num_cell_types):
            self.per_cell_type_heads.append(
                PerClusterHead(
                    shape_filters=filters, profile_kernel_size=profile_kernel_size,
                    counts_head_mlp_layers=counts_head_mlp_layers, num_tasks=num_tasks
                ))
            
        self.scale_function_placement = scale_function_placement

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], per_cluster_load: torch.Tensor) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward propagation

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            seq: Shape: batch, ATCG, window_size
            atac: Shape: batch, n_cell_types, window_size
        per_cluster_load : torch.Tensor
            Prior about per cluster loads

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            1. First element is per cell type profile (Shape: batch, strands, window_size).
            2. Second element is per cell type counts (Shape: batch, strands).
            3. Per cell type motif gates (Shape: batch, filters)
            Elements in per cell type profiles can be directly used for the imputation of
            pseudo-bulk initiation patterns.
        """
        seq, atac = x
        motifs = self.motif_detector(seq) # shape: batch, filter_1 (1024), seq_len
        motifs_gap = motifs.mean(axis=2) # shape: batch, filter_1 (1024)
        atac_truncation = (atac.shape[-1] - motifs.shape[-1]) // 2

        cluster_weights = per_cluster_load
        per_cell_type_profiles = []
        per_cell_type_counts = []
        per_cell_type_activations = []

        for cell_type_id in range(self.num_cell_types):
            cw = cluster_weights[:, cell_type_id]
            profiles, _ = self.profile_refiner(atac[:, cell_type_id, :].unsqueeze(2)) # shape: batch, seq_len, filters_2 (128)
            profiles = torch.swapaxes(profiles, 1, 2)[:, :, atac_truncation:-atac_truncation]
            
            gates = torch.sigmoid(self.filter_gates[cell_type_id](motifs_gap)) # shape: batch, filters_1 (1024)
            filtered_activations = motifs * gates[:, :, None]
            
            if self.scale_function_placement == "early":
                per_cell_type_profile, per_cell_type_count = self.per_cell_type_heads[cell_type_id](
                    (filtered_activations * cw[:, None, None], profiles))
            else:
                per_cell_type_profile, per_cell_type_count = self.per_cell_type_heads[cell_type_id](
                    (filtered_activations, profiles))
            cw = cw.to(per_cell_type_count.device)
            if self.scale_function_placement == "late":
                per_cell_type_profiles.append(per_cell_type_profile * cw[:, None, None])
                per_cell_type_counts.append(per_cell_type_count * cw[:, None])
            elif self.scale_function_placement == "late-ch":
                per_cell_type_profiles.append(per_cell_type_profile)
                per_cell_type_counts.append(per_cell_type_count * cw[:, None])
            else:
                per_cell_type_profiles.append(per_cell_type_profile)
                per_cell_type_counts.append(per_cell_type_count)
            per_cell_type_activations.append(gates)

        return per_cell_type_profiles, per_cell_type_counts, cluster_weights, per_cell_type_activations


class SupervisedSeqOnlyRegressor(nn.Module):
    def __init__(self, num_cell_types: int, 
                 filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
                 n_dil_layers=8, dil_kernel_size=3, conv1_kernel_size=21,
                 profile_kernel_size=75, counts_head_mlp_layers=3,
                 num_tasks=1, 
                 scale_function_placement: str = "late-ch") -> None:

        super().__init__()
        self.num_cell_types = num_cell_types

        # first convolution without dilation
        self.motif_detector = nn.Conv1d(
            4, filters, kernel_size=conv1_kernel_size, padding="valid")

        self.encoder = SeqOnlyBaseRegressor(
            filters=filters, n_non_dil_layers=n_non_dil_layers,
            non_dil_kernel_size=non_dil_kernel_size,
            n_dil_layers=n_dil_layers, dil_kernel_size=dil_kernel_size)
        
        self.per_cluster_heads = nn.ModuleList()
        for _ in range(self.num_cell_types):
            self.per_cluster_heads.append(
                PerClusterHeadSeqOnly(
                    shape_filters=filters, profile_kernel_size=profile_kernel_size,
                    head_layers=counts_head_mlp_layers, num_tasks=num_tasks
                ))

        self.scale_function_placement = scale_function_placement

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], per_cluster_load: torch.Tensor, return_logits=False) -> tuple[
        list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """Forward propagation

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            seq: Shape: batch, ATCG, window_size
            atac: Shape: batch, n_cell_types, window_size
        per_cluster_load : torch.Tensor
            Prior about per cluster loads
        return_logits : bool
            Whether to return raw logits before activation functions
            Default is False.
        Returns
        -----------
        tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, list[torch.Tensor]]
            1. First element is per cluster profile (Shape: batch, strands, window_size).
            2. Second element is per cluster counts (Shape: batch, strands).
            3. Per cluster weights (Shape: batch, n_clusters)
            4. Per cluster motif gates (Shape: batch, filters)
        """
        seq, atac = x
        motifs = self.motif_detector(seq)  # shape: batch, filters_1, seq_len
        body = self.encoder(motifs)  # shape: batch, filters, seq_len

        cluster_weights = per_cluster_load
        per_cell_type_profiles = []
        per_cell_type_counts = []
        per_cell_type_activations = []

        for cell_type_id in range(self.num_cell_types):
            cw = cluster_weights[:, cell_type_id]

            if self.scale_function_placement == "early":
                cell_type_profile, cell_type_count = self.per_cluster_heads[cell_type_id](
                    body * cw[:, None, None])
            else:
                cell_type_profile, cell_type_count = self.per_cluster_heads[cell_type_id](body)
            
            if return_logits:
                per_cell_type_profiles.append(cell_type_profile)
                per_cell_type_counts.append(cell_type_count)
            else:
                if self.scale_function_placement == "late":
                    per_cell_type_profiles.append(cell_type_profile * cw[:, None, None])
                    per_cell_type_counts.append(cell_type_count * cw[:, None])
                elif self.scale_function_placement == "late-ch":
                    per_cell_type_profiles.append(cell_type_profile)
                    per_cell_type_counts.append(cell_type_count * cw[:, None])
                else:
                    per_cell_type_profiles.append(cell_type_profile)
                    per_cell_type_counts.append(cell_type_count)
        
        return per_cell_type_profiles, per_cell_type_counts, cluster_weights, per_cell_type_activations

class ATACToProfilesAndFiLM(nn.Module):
    """
    Encode pseudo-bulk ATAC-seq profiles into FiLM parameters for the modulation of sequence embeddings.
        pseudo-bulk ATAC: (B, 1, L)
        profiles: (B, P, L)
        gamma, beta: (B, C, L)
    
    P = profile_channels, C = motif_embedding_channels, L = sequence_length_after_convolution
    """
    def __init__(
        self,
        profile_channels: int,
        motif_channels: int,
        hidden: int = 64,
        k1: int = 25,
        k2: int = 9,
        dropout: float = 0.0,
        film_tanh_scale: float = 0.1,
    ):
        super().__init__()
        self.film_tanh_scale = film_tanh_scale
        self.encoder = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=k1, padding=k1//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=k2, padding=k2//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gamma_head = nn.Conv1d(hidden, motif_channels, kernel_size=1)
        self.beta_head = nn.Conv1d(hidden, motif_channels, kernel_size=1)
        self.profile_head = nn.Conv1d(hidden, profile_channels, kernel_size=1)

    def forward(self, atac: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        atac: torch.Tensor
            Shape: (B, 1, L)

        Returns
        -------
        gamma: torch.Tensor
            Shape: (B, C, L)
        beta: torch.Tensor
            Shape: (B, C, L)
        """
        x = self.encoder(atac)  # (B, hidden, L)
        profile = self.profile_head(x)  # (B, P, L)

        gamma = self.gamma_head(x)  # (B, C, L)
        beta = self.beta_head(x)  # (B, C, L)

        gamma = 1.0 + torch.tanh(gamma) * self.film_tanh_scale
        beta = torch.tanh(beta) * self.film_tanh_scale
        
        return profile, gamma, beta

class SupervisedFiLMRegressor(nn.Module):
    def __init__(self, num_cell_types: int, profile_shrinkage=8, filters=512, n_non_dil_layers=0, non_dil_kernel_size=3,
                 n_dil_layers=8, dil_kernel_size=3, conv1_kernel_size=21, profile_kernel_size=75,
                 counts_head_mlp_layers=3, num_tasks=2, n_times_more_embeddings=2,
                 atac_hidden=64, atac_dropout=0.0, film_tanh_scale=0.1) -> None:
        
        super().__init__()
        self.num_cell_types = num_cell_types
        n_profile_filters = int(filters / profile_shrinkage)
        
        # Shared motif detector
        self.motif_detector = nn.Sequential(nn.Conv1d(
            4, filters, kernel_size=conv1_kernel_size, padding="valid"))
        
        for _ in range(n_non_dil_layers):
            self.motif_detector.append(
                ResidualConv(filters=filters, kernel_size=non_dil_kernel_size,
                             dilation_rate=1)
            )

        for i in range(n_dil_layers):
            if i < n_dil_layers - 1:
                self.motif_detector.append(
                    ResidualConv(filters=filters, kernel_size=dil_kernel_size,
                                 dilation_rate=2 ** (i + 1))
                )
            else:
                self.motif_detector.append(
                    ResidualConvWithXProjection(filters=filters * n_times_more_embeddings, kernel_size=dil_kernel_size,
                                                dilation_rate=2 ** (i + 1))
                )
        
        # motif gate
        self.filter_gates = nn.ModuleList(
            [nn.LazyLinear(filters * n_times_more_embeddings, bias=False) for _ in range(num_cell_types)])

        self.atac_conditioners = nn.ModuleList([
            ATACToProfilesAndFiLM(
                profile_channels=2 * n_profile_filters, # because of bidirectional GRU
                motif_channels=filters * n_times_more_embeddings,
                hidden=atac_hidden,
                dropout=atac_dropout,
                film_tanh_scale=film_tanh_scale,
            )
            for _ in range(num_cell_types)
        ])
        
        # Per cell type heads
        self.per_cell_type_heads = nn.ModuleList()
        for _ in range(self.num_cell_types):
            self.per_cell_type_heads.append(
                PerClusterHead(
                    shape_filters=filters, profile_kernel_size=profile_kernel_size,
                    counts_head_mlp_layers=counts_head_mlp_layers, num_tasks=num_tasks
                ))
        
        self._last_film = {}

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], per_cluster_load: torch.Tensor,
                return_logits: bool = False) -> tuple[
        list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Forward propagation

        Parameters
        ----------
        x : Tuple[torch.Tensor, torch.Tensor]
            seq: Shape: batch, ATCG, window_size
            atac: Shape: batch, n_cell_types, window_size
        per_cluster_load : torch.Tensor
            Prior about per cluster loads

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            1. First element is per cell type profile (Shape: batch, strands, window_size).
            2. Second element is per cell type counts (Shape: batch, strands).
            3. Per cell type motif gates (Shape: batch, filters)
            Elements in per cell type profiles can be directly used for the imputation of
            pseudo-bulk initiation patterns.
        """
        seq, atac = x
        motifs = self.motif_detector(seq) # shape: batch, filter_1 (1024), seq_len
        motifs_gap = motifs.mean(axis=2) # shape: batch, filter_1 (1024)
        atac_truncation = (atac.shape[-1] - motifs.shape[-1]) // 2

        cluster_weights = per_cluster_load
        per_cell_type_profiles = []
        per_cell_type_counts = []
        per_cell_type_activations = []

        N = 200
        if not hasattr(self, "_dbg_batch_step"):
            self._dbg_batch_step = 0
        self._dbg_batch_step += 1
        do_print = (self._dbg_batch_step % N == 0)

        for cell_type_id in range(self.num_cell_types):
            cw = cluster_weights[:, cell_type_id]
            
            atac_ct = atac[:, cell_type_id, :].unsqueeze(1) # shape: batch, 1, seq_len
            profiles, gamma, beta = self.atac_conditioners[cell_type_id](atac_ct)

            self._last_film[cell_type_id] = {
                "gamma": gamma.detach(),
                "beta": beta.detach(),
            }

            profiles = profiles[:, :, atac_truncation:-atac_truncation]  # (B, P, motifs_len)
            gamma    = gamma[:, :, atac_truncation:-atac_truncation]     # (B, C, motifs_len)
            beta     = beta[:, :, atac_truncation:-atac_truncation]      # (B, C, motifs_len)

            gates = torch.sigmoid(self.filter_gates[cell_type_id](motifs_gap)) # shape: batch, filters_1 (1024)
            filtered_activations = motifs * gates[:, :, None]
            
            filtered_activations = filtered_activations * gamma + beta
            
            per_cell_type_profile, per_cell_type_count = self.per_cell_type_heads[cell_type_id](
                (filtered_activations, profiles))

            per_cell_type_profiles.append(per_cell_type_profile)
            per_cell_type_counts.append(per_cell_type_count)
            per_cell_type_activations.append(gates)

            if do_print:
                with torch.no_grad():
                    # gamma/beta: [B, C(=1024), L]
                    g = gamma.detach()
                    b = beta.detach()

                    abs_g = (g - 1.0).abs()
                    abs_b = b.abs()

                    # 95th percentile
                    p95_g = torch.quantile(abs_g.reshape(-1), 0.95).item()
                    p95_b = torch.quantile(abs_b.reshape(-1), 0.95).item()

                    mean_g = abs_g.mean().item()
                    mean_b = abs_b.mean().item()

                    # saturation fraction near bounds (因为你 scale=0.1，所以边界是 0.9 和 1.1)
                    sat = ((g <= 0.9001) | (g >= 1.0999)).float().mean().item()

                    print(f"[FiLM batch={self._dbg_batch_step} ct={cell_type_id}] |gamma-1| mean={mean_g:.4f} p95={p95_g:.4f} sat={sat:.3f} |beta| mean={mean_b:.4f} p95={p95_b:.4f}")
                    if cell_type_id in [0, 3]:
                        print(f"[Profile batch={self._dbg_batch_step} ct={cell_type_id}] count mean={per_cell_type_count.mean().item():.4f} std={per_cell_type_count.std().item():.4f}")

        return per_cell_type_profiles, per_cell_type_counts, cluster_weights, per_cell_type_activations
