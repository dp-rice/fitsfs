"""Fit a piecewise constant population size to a site frequency spectrum."""
from typing import Iterator

import autograd.numpy as np
import autoptim
from autograd.scipy.special import gammaln


def expected_sfs(sizes: np.ndarray, times: np.ndarray, final_size: float, n: int):
    """
    Compute the expected SFS for a piecewise-constant populations size.

    Parameters
    ----------
    sizes: np.ndarray
        The population size in each epoch starting with the present
    times: np.ndarray
        The start time (backwards in time) of each epoch
    final_size: float
        The population size in the final (earliest) epoch
    n: int
        The (haploid) sample size

    Returns
    -------
    np.ndarray
        The expected site frequency spectrum for the specified model.
    """
    intervals = np.concatenate(([times[0]], np.diff(times)))
    V = _precompute_V(n)
    W = _precompute_W(n)
    return _sfs_exp(n, sizes, intervals, final_size, V, W)


def fit_sfs(
    sfs_obs: np.ndarray,
    k_max: int,
    num_epochs: int,
    size_bounds: tuple[float, float],
    interval_bounds: tuple[float, float],
    final_size: float,
    num_restarts: int,
    options: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a piecewise-constant population size to a site frequency spectrum.

    A partial reimplimentation of fastNeutrino (Bhaskar et al 2015).
    Uses L-BFGS-B to minimize the KL divergence between the expected and observed SFS,
    with automatic differentiation to compute the gradient.

    Parameters
    ----------
    sfs_obs : np.ndarray
        The observed SFS to fit normalized to 1.
        `sfs_obs[0]` is the fraction of singletons.
    k_max : int
        The allele frequency cutoff for fitting.
        All higher-count alleles are lumped together.
    num_epochs : int
        The number of epochs (including the present) in the piecewise-constant model.
    size_bounds : tuple[float, float]
        Bounds on the population sizes to consider.
    interval_bounds : tuple[float, float]
        Bounds on the epoch lengths to consider.
    final_size : float
        The population size in the final (earliest) epoch.
    num_restarts : int
        The number of random starting points to sample.
    options: dict
        Dictionary of options for scipy.minimize.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        `(sizes_fit, times_fit, sfs_fit, kl_divergence)`
        Returns the best fit epoch sizes and start times,
        the expected SFS, and KL(sfs_obs || sfs_exp).
    """
    n = len(sfs_obs) + 1
    V = _precompute_V(n)
    W = _lump(_precompute_W(n), k_max, axis=0)
    target = _lump(sfs_obs, k_max)

    def loss(sizes, times) -> float:
        return _cross_entropy(target, _sfs_exp(n, sizes, times, final_size, V, W))

    sample_starts = _sample_starts(
        size_bounds, interval_bounds, num_epochs, num_restarts
    )

    minima = (
        autoptim.minimize(
            loss,
            start,
            bounds=(size_bounds, interval_bounds),
            method="L-BFGS-B",
            options=options,
        )[0]
        for start in sample_starts
    )
    sizes_fit, intervals_fit = min(minima, key=lambda x: loss(*x))
    times_fit = np.cumsum(intervals_fit)
    sfs_fit = _sfs_exp(n, sizes_fit, intervals_fit, final_size, V, W)
    kld = _kl_div(target, sfs_fit)
    return sizes_fit, times_fit, sfs_fit, kld


def _sample_starts(
    size_bounds: tuple[float, float],
    interval_bounds: tuple[float, float],
    num_epochs: int,
    num_restarts: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for i in range(num_restarts):
        size_starts = _log_sample(*size_bounds, size=num_epochs - 1)
        interval_starts = _log_sample(*interval_bounds, size=num_epochs - 1)
        yield size_starts, interval_starts


def _log_sample(
    lower_bound: float,
    upper_bound: float,
    size: int,
) -> float:
    return np.exp(
        np.random.uniform(np.log(lower_bound), np.log(upper_bound), size=size)
    )


def _lump(a: np.ndarray, k_max: int, axis: int = 0):
    left = a.take(indices=range(k_max), axis=axis)
    right = a.take(indices=range(k_max, a.shape[axis]), axis=axis)
    partial_sum = np.sum(right, axis=axis, keepdims=True)
    return np.concatenate((left, partial_sum), axis=axis)


def _sfs_exp(n, sizes, intervals, final_size, V, W):
    c = _c_integral(n, sizes=sizes, intervals=intervals, final_size=1.0)
    return np.dot(W, c) / np.dot(V, c)


def _c_integral(n: int, sizes, intervals, final_size) -> np.ndarray:
    s = np.pad(sizes, ((0, 1)), mode="constant", constant_values=(final_size,))
    r = np.pad(
        np.cumsum(intervals / s[:-1]),
        ((1, 0)),
        mode="constant",
        constant_values=(0,),
    )
    m = np.arange(2, n + 1)
    bincoeff = m * (m - 1) / 2
    r_exp = np.pad(
        np.exp(-r[:, None] * bincoeff),
        ((0, 1), (0, 0)),
        mode="constant",
        constant_values=(0,),
    )
    return np.dot(s, -np.diff(r_exp, axis=0)) / bincoeff


def _precompute_V(n: int) -> np.ndarray:
    m = np.arange(2, n + 1)
    return (
        (2 * m - 1)
        * np.exp(gammaln(n + 1) + gammaln(n) - gammaln(n + m) - gammaln(n - m + 1))
        * (1 + (-1) ** m)
    )


def _precompute_W(n: int) -> np.ndarray:
    i = np.arange(1, n)
    W = np.zeros((n - 1, n + 1))
    W[:, 2] = 6 / (n + 1)
    W[:, 3] = 30 * (n - 2 * i) / ((n + 1) * (n + 2))
    for m in range(2, n - 1):
        W[:, m + 2] = (
            -(1 + m) * (3 + 2 * m) * (n - m) / (m * (2 * m - 1) * (n + m + 1)) * W[:, m]
            + (3 + 2 * m) * (n - 2 * i) / (m * (n + m + 1)) * W[:, m + 1]
        )
    return W[:, 2:]


def _cross_entropy(p, q):
    return -np.sum(p * np.log(q))


def _kl_div(p, q):
    return -np.sum(p * np.log(q / p))
