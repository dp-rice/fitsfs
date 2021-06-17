import autograd.numpy as np
import autoptim
from autograd.scipy.special import gammaln


def fit_sfs(
    sfs_obs: np.ndarray,
    k_max: int,
    num_epochs: int,
    interval_bounds: tuple[float, float],
    size_bounds: tuple[float, float],
    initial_size: float,
    num_restarts: int,
):
    n = len(sfs_obs) + 1
    V = precompute_V(n)
    W = lump(precompute_W(n), k_max, axis=0)
    target = lump(sfs_obs, k_max)

    def loss(sizes, times) -> float:
        return cross_entropy(target, _sfs_exp(sizes, times, initial_size, V, W))

    size_starts = np.random.uniform(*size_bounds, size=(num_restarts, num_epochs - 1))
    interval_starts = np.random.uniform(
        *interval_bounds, size=(num_restarts, num_epochs - 1)
    )
    minima = (
        autoptim.minimize(
            loss,
            [sizes_0, interval_0],
            bounds=bounds,
            method="L-BFGS-B",
        )[0]
        for sizes_0, interval_0 in zip(size_starts, interval_starts)
    )
    sizes_fit, intervals_fit = min(minima, key=lambda x: loss(*x))
    times_fit = np.cumsum(intervals_fit)
    sfs_fit = _sfs_exp(sizes_fit, intervals_fit, initial_size, V, W)
    kld = kl_div(target, sfs_fit)
    return sizes_fit, times_fit, sfs_fit, kld


def lump(a: np.ndarray, k_max: int, axis: int = 0):
    left = a.take(indices=range(k_max), axis=axis)
    right = a.take(indices=range(k_max, a.shape[axis]), axis=axis)
    partial_sum = np.sum(right, axis=axis, keepdims=True)
    return np.concatenate((left, partial_sum), axis=axis)


def sfs_exp(sizes, times, initial_size: float, n: int):
    intervals = np.concatenate(([times[0]], np.diff(times)))
    V = precompute_V(n)
    W = precompute_W(n)
    return _sfs_exp(sizes, intervals, initial_size, V, W)


def _sfs_exp(sizes, intervals, initial_size, V, W):
    c = c_integral(n, sizes=sizes, intervals=intervals, initial_size=1.0)
    return np.dot(W, c) / np.dot(V, c)


def cross_entropy(p, q):
    return -np.sum(p * np.log(q))


def kl_div(p, q):
    return -np.sum(p * np.log(q / p))


def c_integral(n: int, sizes, intervals, initial_size) -> np.ndarray:
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


def precompute_V(n: int) -> np.ndarray:
    m = np.arange(2, n + 1)
    return (
        (2 * m - 1)
        * np.exp(gammaln(n + 1) + gammaln(n) - gammaln(n + m) - gammaln(n - m + 1))
        * (1 + (-1) ** m)
    )


def precompute_W(n: int) -> np.ndarray:
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


if __name__ == "__main__":
    n = 100
    k_max = 20

    true_sizes = np.array([0.5, 0.25])
    true_times = np.array([0.5, 1.0])
    true_sfs = sfs_exp(true_sizes, true_times, 1.0, 100)
    print(lump(true_sfs, k_max))

    num_epochs = 3
    bounds = ((1e-1, 10.0), (1e-1, 1.0))
    num_restarts = 100
    fitted = fit_sfs(true_sfs, k_max, num_epochs, bounds[0], bounds[1], 1.0, 100)
    for f in fitted:
        print(f)

    print(lump(fitted[2], k_max))
