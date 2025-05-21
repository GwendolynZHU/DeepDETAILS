import os
import torch
import matplotlib.pyplot as plt
from typing import Union, Optional
from deepdetails.helper.utils import get_log_dir


def _get_scale(a=1):
    def forward(x):
        x = (x >= 0) * x + (x < 0) * x * a
        return x

    def inverse(x):
        x = (x >= 0) * x + (x < 0) * x / a
        return x

    return forward, inverse


def _adjust_y_ranges(axes):
    y_start, y_end = axes.get_ylim()
    if y_start < 0 < y_end:
        forward, inverse = _get_scale(-1 * (y_end / y_start))
        axes.set_yscale("function", functions=(forward, inverse))
        ranges = [y_start, 0, y_end]
        axes.set_ylim((y_start, y_end))
        axes.yaxis.set_ticks(ranges)


def _share_ylim(axes, ax_ids):
    y_lims = [axes[i].get_ylim() for i in ax_ids]
    for i in ax_ids:
        axes[i].set_ylim(min(y_lims[0][0], *[ylim[0] for ylim in y_lims]),
                         max(y_lims[0][1], *[ylim[1] for ylim in y_lims]))
        _adjust_y_ranges(axes[i])


def bulk_visual_inspection(y_hats: torch.Tensor, y: torch.Tensor,
                           per_cluster_y_hats: Union[None, torch.Tensor],
                           prefix: str = ".", logger: Union[None, callable] = None):
    """Generate a snapshot of the predicted bulk vs. input bulk

    Parameters
    ----------
    y_hats : torch.Tensor
        Predictions. Shape: cluster, batch, strands, seq_len
    y : torch.Tensor
        Truth (learning targets)
    per_cluster_y_hats : Union[None, torch.Tensor]

    prefix : str
        Prefix to the output file, default ""
    logger : Union[None, callable]
        Logger instances

    Returns
    -------

    """
    sample_id = torch.randint(y_hats.size(0), (1, 1))[0, 0].item()
    detached_y = y.detach().cpu().numpy()
    detached_y_hat = y_hats.detach().cpu().numpy()
    detached_per_cluster_y_hats = per_cluster_y_hats.detach().cpu().numpy()

    fig, axs = plt.subplots(2 + per_cluster_y_hats.shape[0] if per_cluster_y_hats is not None else 0,
                            1, sharex=True)

    plot_data = [detached_y[sample_id, :, :], detached_y_hat[sample_id, :, :], ]
    plot_label = ["Y", r"$\hat{Y}$", ]
    for i in range(per_cluster_y_hats.size(0)):
        plot_data.append(detached_per_cluster_y_hats[i, sample_id, :, :])
        plot_label.append(r"$\hat{{Y}}_{}$".format(i))

    for i, (data, label) in enumerate(zip(plot_data, plot_label)):
        ax = axs[i]
        ax.plot(data[0, :], color="#FF0D57")
        ax.plot(data[1, :] * -1, color="#1E88E5")
        if i < 2:
            _adjust_y_ranges(ax)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    rows_sharing_y = tuple(range(2, len(axs)))
    _share_ylim(axs, rows_sharing_y)

    fig.align_ylabels()
    plt.tight_layout()

    dest_folder, version = get_log_dir(logger)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest = os.path.join(dest_folder, f"{prefix}{sample_id}{version}.png")

    plt.savefig(dest, dpi=200)
    plt.close()


def gt_visual_inspection(y_hats: torch.Tensor, y: torch.Tensor,
                         prefix: str = ".", logger: Union[None, callable] = None,
                         cell_type_names: Optional[list] = None):
    """Generate a snapshot of the predicted bulk vs. input bulk

    Parameters
    ----------
    y_hats : torch.Tensor
        Predictions. Shape: cluster, batch, strands, seq_len
    y : torch.Tensor
        Truth (learning targets)

    prefix : str
        Prefix to the output file, default ""
    logger : Union[None, callable]
        Logger instances
    cell_type_names : Optional[list]
        List of cell type names for the y_hats tensor. If None, the default names will be used.

    Returns
    -------

    """
    assert y.shape == y_hats.shape, "Shape mismatch between y and y_hats"
    n_cell_types, batch_size, strands, seq_len = y_hats.shape
    sample_id = torch.randint(batch_size, (1, 1))[0, 0].item()
    detached_y = y.detach().cpu().numpy()
    detached_y_hat = y_hats.detach().cpu().numpy()

    fig, axs = plt.subplots(n_cell_types * 2, 1, sharex=True, figsize=(5, 0.6 * n_cell_types * 2))
    pred_rows = []
    truth_rows = []
    for i in range(n_cell_types):
        true_signal = detached_y[i, sample_id, :, :]
        pred_signal = detached_y_hat[i, sample_id]

        # Ground truth
        ax_true = axs[2 * i]
        truth_rows.append(2 * i)
        ax_true.plot(true_signal[0], color="#FF0D57")
        ax_true.plot(-true_signal[1], color="#1E88E5")
        label = cell_type_names[i] if cell_type_names else f"CT {i}"
        ax_true.set_ylabel(f"Y\n{label}")
        ax_true.spines["top"].set_visible(False)
        ax_true.spines["right"].set_visible(False)

        # Prediction
        ax_pred = axs[2 * i + 1]
        pred_rows.append(2 * i + 1)
        ax_pred.plot(pred_signal[0], color="#FF0D57")
        ax_pred.plot(-pred_signal[1], color="#1E88E5")
        ax_pred.set_ylabel(r"$\hat{Y}$" + "\n" + label)
        ax_pred.spines["top"].set_visible(False)
        ax_pred.spines["right"].set_visible(False)

    _share_ylim(axs, pred_rows)
    _share_ylim(axs, truth_rows)

    fig.align_ylabels()
    plt.tight_layout()

    # Save the figure
    dest_folder, version = get_log_dir(logger)
    os.makedirs(dest_folder, exist_ok=True)

    dest = os.path.join(dest_folder, f"{prefix}{sample_id}{version}.png")
    
    plt.savefig(dest, dpi=200)
    plt.close()

    
def per_cluster_visual_inspection(
        y_hats: torch.Tensor, per_cluster_y_hats: torch.Tensor,
        y: torch.Tensor, per_cluster_y: torch.Tensor,
        loads: torch.Tensor, weights: torch.Tensor,
        prefix: str = ".", logger: Union[None, callable] = None):
    """Generate a snapshot of the per-cluster prediction vs. truth

    Parameters
    ----------
    y_hats : torch.Tensor
        Aggregated predictions. Shape: batch, strands, seq_len
    per_cluster_y_hats : torch.Tensor
        Cluster-specific predictions. Shape: n_clusters, batch, strands, seq_len
    y: torch.Tensor
        Aggregated true profiles. Shape: batch, strands, seq_len
    per_cluster_y : torch.Tensor
        Cluster-specific truth. Shape: n_clusters, batch, strands, seq_len
    loads : torch.Tensor
        Cluster loads. Shape: batch, n_clusters
    weights : torch.Tensor
        Cluster weights. Shape: batch, n_clusters
    prefix : str
        Prefix to the output file, default ""
    logger : Union[None, callable]
        Logger instances

    Returns
    -------

        """
    sample_id = torch.randint(per_cluster_y_hats.size(1), (1, 1))[0, 0].item()
    detached_y = y.detach().cpu().numpy()
    detached_per_cluster_y_hats = per_cluster_y_hats.detach().cpu().numpy()
    detached_y_hat = y_hats.detach().cpu().numpy()
    detached_per_cluster_y = per_cluster_y.detach().cpu().numpy()
    detached_loads = loads.detach().cpu().numpy()
    detached_weights = weights.detach().cpu().numpy()

    n_rows = 2 + per_cluster_y_hats.size(0) * 2
    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=(5, 0.6 * n_rows))
    plot_data = [detached_y[sample_id, :, :], detached_y_hat[sample_id, :, :], ]
    plot_label = ["Y", r"$\hat{Y}$", ]
    pred_rows = []
    truth_rows = []
    for i in range(per_cluster_y_hats.size(0)):
        plot_data.append(detached_per_cluster_y[i, sample_id, :, :])
        plot_label.append(r"$Y_{}$".format(i) + "\n({:.3f})".format(detached_loads[sample_id, i].item()))
        truth_rows.append(2 * (i + 1))
        plot_data.append(detached_per_cluster_y_hats[i, sample_id, :, :])
        plot_label.append(r"$\hat{{Y}}_{}$".format(i) + "\n({:.3f})".format(detached_weights[sample_id, i].item()))
        pred_rows.append(2 * (i + 1) + 1)

    for i, (data, label) in enumerate(zip(plot_data, plot_label)):
        ax = axs[i]
        ax.plot(data[0, :], color="#FF0D57")
        if data.shape[0] == 2:  # only draw stranded signal when available
            ax.plot(data[1, :] * -1, color="#1E88E5")
        if i < 2:
            _adjust_y_ranges(ax)
        ax.set_ylabel(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _share_ylim(axs, pred_rows)
    _share_ylim(axs, truth_rows)

    fig.align_ylabels()
    plt.tight_layout()

    dest_folder, version = get_log_dir(logger)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    dest = os.path.join(dest_folder, f"{prefix}{sample_id}{version}.png")

    plt.savefig(dest, dpi=200)
    plt.close()
