"""Fit a piecewise constant population size to a site frequency spectrum."""
from typing import Iterator

import autograd.numpy as np
import autoptim
from autograd.scipy.special import gammaln


def expected_sfs(sizes: np.ndarray, times: np.ndarray, initial_size: float, n: int):
    """
    Compute the expected SFS for a piecewise-constant populations size.

    Parameters
    ----------
    sizes: np.ndarray
        The population size in each epoch after the initial one
    times: np.ndarray
        The start time (backwards in time) of each epoch
    initial_size: float
        The population size in the present
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
    return _sfs_exp(sizes, intervals, initial_size, V, W)


def fit_sfs(
    sfs_obs: np.ndarray,
    k_max: int,
    num_epochs: int,
    size_bounds: tuple[float, float],
    interval_bounds: tuple[float, float],
    initial_size: float,
    num_restarts: int,
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
    initial_size : float
        The initial (present) population size.
    num_restarts : int
        The number of random starting points to sample.

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
        return _cross_entropy(target, _sfs_exp(sizes, times, initial_size, V, W))

    sample_starts = _sample_starts(
        size_bounds, interval_bounds, num_epochs, num_restarts
    )

    minima = (
        autoptim.minimize(
            loss,
            start,
            bounds=(size_bounds, interval_bounds),
            method="L-BFGS-B",
            options={"gtol": 1e-6},
        )[0]
        for start in sample_starts
    )
    sizes_fit, intervals_fit = min(minima, key=lambda x: loss(*x))
    times_fit = np.cumsum(intervals_fit)
    sfs_fit = _sfs_exp(sizes_fit, intervals_fit, initial_size, V, W)
    kld = _kl_div(target, sfs_fit)
    return sizes_fit, times_fit, sfs_fit, kld


def _sample_starts(
    size_bounds: tuple[float, float],
    interval__bounds: tuple[float, float],
    num_epochs: int,
    num_restarts: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for i in range(num_restarts):
        size_starts = np.random.uniform(*size_bounds, size=num_epochs - 1)
        interval_starts = np.random.uniform(*interval_bounds, size=num_epochs - 1)
        yield size_starts, interval_starts


def _lump(a: np.ndarray, k_max: int, axis: int = 0):
    left = a.take(indices=range(k_max), axis=axis)
    right = a.take(indices=range(k_max, a.shape[axis]), axis=axis)
    partial_sum = np.sum(right, axis=axis, keepdims=True)
    return np.concatenate((left, partial_sum), axis=axis)


def _sfs_exp(sizes, intervals, initial_size, V, W):
    c = _c_integral(n, sizes=sizes, intervals=intervals, initial_size=1.0)
    return np.dot(W, c) / np.dot(V, c)


def _c_integral(n: int, sizes, intervals, initial_size) -> np.ndarray:
    s = np.pad(sizes, ((1, 0)), mode="constant", constant_values=(initial_size,))
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


if __name__ == "__main__":
    n = 100
    k_max = 40

    initial_size = 1.0
    true_sizes = np.array([0.25, 1.00])
    true_times = np.array([0.5, 0.6])
    true_sfs = expected_sfs(true_sizes, true_times, initial_size, n)
    print(_lump(true_sfs, k_max))

    num_epochs = 3
    size_bounds = (1e-1, 10.0)
    interval_bounds = (1e-1, 1.0)
    num_restarts = 100
    fitted = fit_sfs(
        true_sfs, k_max, num_epochs, size_bounds, interval_bounds, initial_size, 100
    )
    for f in fitted:
        print(f)
