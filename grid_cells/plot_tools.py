import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List
from scipy.ndimage import gaussian_filter


def activity_map(
    position: np.ndarray,
    activity: np.ndarray,
    nbins: Union[Tuple[int], int] = 25,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Compute mean activity map binned by position.

    Args:
        position: Position data, shape (n_samples,) or (n_samples, n_dims).
        activity: Activity data, shape (n_samples,).
        nbins: Number of bins per dimension. Default is 25.
        sigma: Gaussian smoothing sigma. If None, no smoothing applied.

    Returns:
        mean_activity: Mean activity in each bin.
        counts: Number of samples in each bin.
        edges: Bin edges for each dimension.
    """
    if position.ndim == 2:
        ndim = position.shape[1]
    elif position.ndim == 1:
        ndim = 1
    else:
        raise ValueError("WTF?")
    if isinstance(nbins, int):
        nbins = (nbins,) * ndim

    pos_min = position.min(axis=0)
    pos_max = position.max(axis=0)
    if ndim > 1:
        edges = [np.linspace(pos_min[i], pos_max[i], nbins[i] + 1) for i in range(ndim)]
    else:
        edges = [np.linspace(pos_min, pos_max, nbins[0] + 1)]

    activity_sum, edges = np.histogramdd(position, bins=edges, weights=activity)
    counts, _ = np.histogramdd(position, bins=edges)

    mean_activity = np.divide(
        activity_sum,
        counts,
        out=np.zeros_like(activity_sum),
        where=counts > 0,
    )
    if sigma:
        mean_activity = gaussian_filter(mean_activity, sigma=sigma)

    return mean_activity, counts, edges


def get_plot_grid(n_plots, figsize=(4, 4), **kwargs):
    """
    Create a grid of subplots for plotting.

    Args:
        n_plots: Number of subplots to create.
        figsize: Base figure size for each subplot. Default is (4, 4).
        **kwargs: Additional keyword arguments passed to plt.subplots.

    Returns:
        fig: The matplotlib figure object.
        axes: Array of matplotlib axes objects reshaped to (n_rows, n_cols).
    """
    if n_plots <= 0:
        raise ValueError("n_plots must be a positive integer")

    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows), **kwargs
    )
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    return fig, axes


def wrap_to_pi(angle):
    """Wraps any angle to the [-pi, pi] range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angular_error(decoded, true):
    return (
        np.rad2deg(
            np.arctan2(
                np.sin(np.deg2rad(decoded - true)), np.cos(np.deg2rad(decoded - true))
            )
        )
        / 180
    )


def smooth_ts(array: np.ndarray, interval: int):
    return (
        array[: (len(array) // interval) * interval]
        .reshape((-1,) + (interval,) + array.shape[1:])
        .mean(axis=1)
    )
