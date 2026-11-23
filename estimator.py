"""estimators.py - conventional channel estimators used as baselines (Task 2).

LS + 2D interpolation lives in data_utils.make_xy (the network input IS the LS+interpolation estimate).
This file adds the MMSE estimator of the paper (Eqs. 3-5):  h_d_hat = A_MMSE * h_p_LS,
    A_MMSE = R_dp (R_pp + sigma^2 (X X^H)^-1)^-1,   and with unit-modulus pilots (X X^H)^-1 = I.
The correlation matrices are estimated from the TRAINING channels, so this is the 'MMSE with
known channel statistics' benchmark (the paper's ideal MMSE), not something a real receiver has.
"""
import numpy as np

from data_utils import NSC, NSYM, SPLIT, load_ls, load_truth


def grid_vectors(chan, sl):
    """True channel as complex vectors in MATLAB's column-major grid order: (n, 1008), index = slot*72 + subcarrier."""
    H = load_truth(chan, sl)                                   # (n, 72, 14, 2)
    Hc = H[..., 0].astype(np.float64) + 1j * H[..., 1].astype(np.float64)
    return Hc.transpose(0, 2, 1).reshape(Hc.shape[0], -1)


def full_correlation(chan, chunk=4000):
    """R[i, j] = E[h_i conj(h_j)] over the 1008 grid points, estimated from the training split."""
    tr = SPLIT["train"]
    R = np.zeros((NSC * NSYM, NSC * NSYM), dtype=np.complex128)
    n = 0
    for a in range(tr.start, tr.stop, chunk):
        Hv = grid_vectors(chan, slice(a, min(a + chunk, tr.stop)))
        R += Hv.T @ Hv.conj()
        n += Hv.shape[0]
    return R / n


def mmse_matrix(R, pIdx, snr_db):
    """A_MMSE (1008 x nP) for pilots at 1-based grid indices pIdx and noise variance 10^(-SNR/10)."""
    pi = np.asarray(pIdx) - 1
    Rdp, Rpp = R[:, pi], R[np.ix_(pi, pi)]
    sigma2 = 10.0 ** (-snr_db / 10.0)
    return Rdp @ np.linalg.inv(Rpp + sigma2 * np.eye(len(pi)))


def mmse_mse(R, chan, nP, snr, n_test):
    """Per-element complex MSE of the MMSE estimate on the first n_test test samples."""
    sl = slice(SPLIT["test"].start, SPLIT["test"].start + n_test)
    re_p, im_p, pIdx = load_ls(chan, nP, snr)
    Hls = re_p[:, sl].astype(np.float64) + 1j * im_p[:, sl].astype(np.float64)   # (nP, n)
    Hhat = mmse_matrix(R, pIdx, snr) @ Hls                                         # (1008, n)
    return float(np.mean(np.abs(Hhat.T - grid_vectors(chan, sl)) ** 2))
