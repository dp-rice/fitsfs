"""Fit a piecewise constant population size to a site frequency spectrum."""
import json
from dataclasses import InitVar, dataclass, field
from typing import Iterator

import autograd.numpy as np
import autoptim
from autograd.scipy.special import gammaln


@dataclass
class FittedPWCModel:
    samples: int
    sizes: list[float] = field(init=False)
    sizes_iter: InitVar[Iterator[float]]
    times: list[float] = field(init=False)
    times_iter: InitVar[Iterator[float]]
    sfs_obs: list[float] = field(init=False)
    sfs_iter: InitVar[Iterator[float]]
    sfs_exp: list[float] = field(init=False)
    kl_div: float = field(init=False)
    num_epochs: int = field(init=False)
    k_max: int = field(init=False)
    folded: bool

    def __post_init__(self, sizes_iter, times_iter, sfs_iter):  # noqa D105
        self.sizes = list(sizes_iter)
        self.times = list(times_iter)
        self.sfs_obs = list(sfs_iter)
        self.num_epochs = len(self.times)
        if len(self.sizes) != self.num_epochs + 1:
            raise ValueError("`len(sizes)` must equal `len(times) + 1`")
        self.k_max = len(self.sfs_obs)
        self.sfs_exp = list(
            _lump(
                expected_sfs(self.sizes, self.times, self.samples, self.folded),
                self.k_max,
            )
        )
        self.kl_div = _kl_div(np.array(self.sfs_obs), np.array(self.sfs_exp))

    def toJson(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self, default=lambda o: o.__dict__)


def expected_sfs(
    sizes: np.ndarray, times: np.ndarray, samples: int, folded: bool = False
):
    """
    Compute the expected SFS for a piecewise-constant populations size.

    Parameters
    ----------
    sizes: np.ndarray
        The population size in each epoch starting with the present
    times: np.ndarray
        The start time (backwards in time) of each epoch
    samples: int
        The (haploid) sample size

    Returns
    -------
    np.ndarray
        The expected site frequency spectrum for the specified model.
    """
    intervals = np.concatenate(([times[0]], np.diff(times)))
    V = _precompute_V(samples)
    W = _precompute_W(samples, folded)
    return _sfs_exp(samples, sizes, intervals, V, W)


def fit_sfs(
    sfs_obs: np.ndarray,
    folded: bool,
    k_max: int,
    num_epochs: int,
    penalty_coef: float = 1e-4,
    num_restarts: int = 100,
    size_bounds: tuple[float, float] = (1e-2, 1e2),
    interval_bounds: tuple[float, float] = (1e-2, 1e2),
    options: dict = {"ftol": 1e-10, "gtol": 1e-12},
) -> FittedPWCModel:
    """
    Fit a piecewise-constant population size to a site frequency spectrum.

    A partial reimplimentation of fastNeutrino (Bhaskar et al 2015).
    Uses L-BFGS-B to minimize the KL divergence between the expected and observed SFS,
    with automatic differentiation to compute the gradient.

    Uses L^2 norm regularization on the log population sizes to keep them on unit scale.

    Parameters
    ----------
    sfs_obs : np.ndarray
        The observed SFS to fit normalized to 1.
        `sfs_obs[0]` is the fraction of singletons.
    folded : bool
        If True, fold the SFS.
    k_max : int
        The allele frequency cutoff for fitting.
        All higher-count alleles are lumped together.
    num_epochs : int
        The number of epochs (including the present) in the piecewise-constant model.
    penalty_coef: float
        The penalty coefficient on `log(sizes)`.
        Used to keep sizes on unit scale. (default = 1e-4)
    num_restarts : int
        The number of random starting points to sample. (default = 100)
    size_bounds : tuple[float, float]
        Bounds on the population sizes to consider. (default = (1e-2, 1e2))
    interval_bounds : tuple[float, float]
        Bounds on the epoch lengths to consider. (default = (1e-2, 1e2))
    options: dict
        Dictionary of options for scipy.minimize.

    Returns
    -------
    FittedPWCModel

    """
    samples = len(sfs_obs) + 1
    V = _precompute_V(samples)
    W = _lump(_precompute_W(samples, folded), k_max, axis=0)
    if folded:
        target = _lump(_fold(sfs_obs), k_max)
    else:
        target = _lump(sfs_obs, k_max)

    def penalty(sizes, intervals) -> float:
        return penalty_coef * np.sum(np.log(sizes) ** 2)

    def loss(sizes, intervals) -> float:
        return _cross_entropy(
            target, _sfs_exp(samples, sizes, intervals, V, W)
        ) + penalty(sizes, intervals)

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
    return FittedPWCModel(samples, sizes_fit, times_fit, target, folded)


def _sample_starts(
    size_bounds: tuple[float, float],
    interval_bounds: tuple[float, float],
    num_epochs: int,
    num_restarts: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for i in range(num_restarts):
        size_starts = _log_sample(*size_bounds, size=num_epochs)
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


def _fold(a: np.ndarray, axis: int = 0):
    n_fold = a.shape[0] // 2
    folded = np.zeros_like(a)
    folded[:-n_fold] = a[:-n_fold]
    folded[:n_fold] += a[: -(n_fold + 1) : -1]
    return folded


def _lump(a: np.ndarray, k_max: int, axis: int = 0):
    left = a.take(indices=range(k_max - 1), axis=axis)
    right = a.take(indices=range(k_max - 1, a.shape[axis]), axis=axis)
    partial_sum = np.sum(right, axis=axis, keepdims=True)
    return np.concatenate((left, partial_sum), axis=axis)


def _sfs_exp(n, sizes, intervals, V, W):
    c = _c_integral(n, sizes=sizes, intervals=intervals)
    return np.dot(W, c) / np.dot(V, c)


def _c_integral(n: int, sizes, intervals) -> np.ndarray:
    r = np.pad(
        np.cumsum(intervals / sizes[:-1]), (1, 0), mode="constant", constant_values=(0,)
    )
    m = np.arange(2, n + 1)
    bincoeff = m * (m - 1) / 2
    r_exp = np.pad(
        np.exp(-r[:, None] * bincoeff),
        ((0, 1), (0, 0)),
        mode="constant",
        constant_values=(0,),
    )
    return np.dot(sizes, -np.diff(r_exp, axis=0)) / bincoeff


def _precompute_V(n: int) -> np.ndarray:
    m = np.arange(2, n + 1)
    return (
        (2 * m - 1)
        * np.exp(gammaln(n + 1) + gammaln(n) - gammaln(n + m) - gammaln(n - m + 1))
        * (1 + (-1) ** m)
    )


def _precompute_W(n: int, folded: bool = False) -> np.ndarray:
    i = np.arange(1, n)
    W = np.zeros((n - 1, n + 1))
    W[:, 2] = 6 / (n + 1)
    W[:, 3] = 30 * (n - 2 * i) / ((n + 1) * (n + 2))
    for m in range(2, n - 1):
        W[:, m + 2] = (
            -(1 + m) * (3 + 2 * m) * (n - m) / (m * (2 * m - 1) * (n + m + 1)) * W[:, m]
            + (3 + 2 * m) * (n - 2 * i) / (m * (n + m + 1)) * W[:, m + 1]
        )
    if folded:
        return _fold(W[:, 2:])
    else:
        return W[:, 2:]


def _cross_entropy(p, q):
    return -np.sum(p * np.log(q))


def _kl_div(p, q):
    return -np.sum(p * np.log(q / p))
