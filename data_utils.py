"""data_utils.py - Python side of the Assignment 3 pipeline (Step 7).

Reads the MATLAB files written by run_all.m and builds network inputs/targets:
  Y = true channel H            (N, 72, 14, 2)   [..., 0] = Re, [..., 1] = Im
  X = LS + 2D interpolation     (N, 72, 14, 2)   = W @ (noisy LS values at the pilots)
Both are divided by one global constant c (std of the training ground truth).
"""
import json
import os

import h5py
import numpy as np
from scipy.io import loadmat

# Folder written by run_all.m (override with the A3_DATA environment variable, e.g. on Colab)
DATA = os.environ.get("A3_DATA", r"D:\ANTARIKSH\MLWC_3\data")

NSC, NSYM = 72, 14
SPLIT = {"train": slice(0, 32000), "val": slice(32000, 36000), "test": slice(36000, 40000)}


def _path(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} not found. Check DATA in data_utils.py (currently {DATA}).")
    return p


def load_truth(chan, sl=slice(0, None)):
    """Ground-truth channel for the samples in `sl`: (n, 72, 14, 2) float32.
    Only that slice is read from the HDF5 file, which keeps RAM use small."""
    with h5py.File(_path(f"H_{chan}.mat"), "r") as f:
        re, im = f["Hre"][sl], f["Him"][sl]
    assert re.shape[1:] == (NSYM, NSC), re.shape           # h5py shows MATLAB's axes reversed
    return np.stack([re, im], -1).transpose(0, 2, 1, 3).astype(np.float32)


def load_ls(chan, nP, snr):
    """Noisy LS values at the pilots: (nP, N) real part, (nP, N) imaginary part, 1-based pilot indices."""
    d = loadmat(_path(f"LS_{chan}_P{nP}_snr{snr}.mat"))
    return (d["HlsRe"].astype(np.float32), d["HlsIm"].astype(np.float32),
            d["pIdx"].ravel().astype(int))


def load_W(nP, method="linear"):
    """Interpolation matrix, shape (1008, nP)."""
    return loadmat(_path(f"W_{method}_P{nP}.mat"))["W"].astype(np.float32)


def interpolate(W, Hre_p, Him_p):
    """Full-grid LS+interpolation estimate, shape (N, 72, 14, 2)."""
    out = []
    for p in (Hre_p, Him_p):
        g = (W @ p).reshape(NSC, NSYM, -1, order="F")      # MATLAB stores the grid column-major
        out.append(g.transpose(2, 0, 1))
    return np.stack(out, -1)


def get_c(chan):
    """Global scale = std of the training-set ground truth (cached in norm_<chan>.json)."""
    fn = os.path.join(DATA, f"norm_{chan}.json")
    if os.path.exists(fn):
        try:
            with open(fn) as f:
                return json.load(f)["c"]
        except (ValueError, KeyError):
            pass                                   # empty/corrupt cache: recompute below
    c = float(load_truth(chan, SPLIT["train"]).std())   # compute FIRST, then write
    with open(fn, "w") as f:
        json.dump({"c": c}, f)
    return c


def make_xy(chan, nP, snr, split, method="linear", n=0):
    """Normalised network input X and target Y for one split, plus the scale c.
    n > 0 keeps only the first n samples of the split (only those are loaded and interpolated)."""
    sl, c = SPLIT[split], get_c(chan)
    if n:
        sl = slice(sl.start, min(sl.stop, sl.start + n))
    re_p, im_p, _ = load_ls(chan, nP, snr)
    X = interpolate(load_W(nP, method), re_p[:, sl], im_p[:, sl]) / c
    Y = load_truth(chan, sl) / c
    return X.astype(np.float32), Y.astype(np.float32), c


def mse_complex(Xhat, Y, c):
    """Per-element complex MSE in ORIGINAL units: mean(Re^2 + Im^2) of the error."""
    return 2.0 * float(np.mean((Xhat - Y) ** 2)) * c ** 2


def to_single_channel(A):
    """(N,72,14,2) -> (2N,72,14,1): real parts first, then imaginary parts (as in the paper)."""
    return np.concatenate([A[..., :1], A[..., 1:]], axis=0)


def from_single_channel(A):
    """Inverse of to_single_channel."""
    n = A.shape[0] // 2
    return np.concatenate([A[:n], A[n:]], axis=-1)




# """data_utils.py - Python side of the Assignment 3 pipeline (Step 7).

# Reads the MATLAB files written by run_all.m and builds network inputs/targets:
#   Y = true channel H            (N, 72, 14, 2)   [..., 0] = Re, [..., 1] = Im
#   X = LS + 2D interpolation     (N, 72, 14, 2)   = W @ (noisy LS values at the pilots)
# Both are divided by one global constant c (std of the training ground truth).
# """
# import json
# import os
# from functools import lru_cache

# import h5py
# import numpy as np
# from scipy.io import loadmat

# # Folder written by run_all.m (override with the A3_DATA environment variable, e.g. on Colab)
# DATA = os.environ.get("A3_DATA", r"D:\ANTARIKSH\MLWC_3\data")

# NSC, NSYM = 72, 14
# SPLIT = {"train": slice(0, 32000), "val": slice(32000, 36000), "test": slice(36000, 40000)}


# def _path(name):
#     p = os.path.join(DATA, name)
#     if not os.path.exists(p):
#         raise FileNotFoundError(f"{p} not found. Check DATA in data_utils.py (currently {DATA}).")
#     return p


# @lru_cache(maxsize=2)
# def load_truth(chan):
#     """Ground-truth channel, shape (N, 72, 14, 2), float32."""
#     with h5py.File(_path(f"H_{chan}.mat"), "r") as f:
#         re, im = np.array(f["Hre"]), np.array(f["Him"])
#     assert re.shape[1:] == (NSYM, NSC), re.shape           # h5py shows MATLAB's axes reversed
#     return np.stack([re, im], -1).transpose(0, 2, 1, 3).astype(np.float32)


# def load_ls(chan, nP, snr):
#     """Noisy LS values at the pilots: (nP, N) real part, (nP, N) imaginary part, 1-based pilot indices."""
#     d = loadmat(_path(f"LS_{chan}_P{nP}_snr{snr}.mat"))
#     return (d["HlsRe"].astype(np.float32), d["HlsIm"].astype(np.float32),
#             d["pIdx"].ravel().astype(int))


# def load_W(nP, method="linear"):
#     """Interpolation matrix, shape (1008, nP)."""
#     return loadmat(_path(f"W_{method}_P{nP}.mat"))["W"].astype(np.float32)


# def interpolate(W, Hre_p, Him_p):
#     """Full-grid LS+interpolation estimate, shape (N, 72, 14, 2)."""
#     out = []
#     for p in (Hre_p, Him_p):
#         g = (W @ p).reshape(NSC, NSYM, -1, order="F")      # MATLAB stores the grid column-major
#         out.append(g.transpose(2, 0, 1))
#     return np.stack(out, -1)


# def get_c(chan):
#     """Global scale = std of the training-set ground truth (cached in norm_<chan>.json)."""
#     fn = os.path.join(DATA, f"norm_{chan}.json")
#     if os.path.exists(fn):
#         try:
#             with open(fn) as f:
#                 return json.load(f)["c"]
#         except (ValueError, KeyError):
#             pass                                   # empty/corrupt cache: recompute below
#     c = float(load_truth(chan)[SPLIT["train"]].std())   # compute FIRST, then write
#     with open(fn, "w") as f:
#         json.dump({"c": c}, f)
#     return c


# def make_xy(chan, nP, snr, split, method="linear"):
#     """Normalised network input X and target Y for one split, plus the scale c."""
#     sl, c = SPLIT[split], get_c(chan)
#     re_p, im_p, _ = load_ls(chan, nP, snr)
#     X = interpolate(load_W(nP, method), re_p[:, sl], im_p[:, sl]) / c
#     Y = load_truth(chan)[sl] / c
#     return X.astype(np.float32), Y.astype(np.float32), c


# def mse_complex(Xhat, Y, c):
#     """Per-element complex MSE in ORIGINAL units: mean(Re^2 + Im^2) of the error."""
#     return 2.0 * float(np.mean((Xhat - Y) ** 2)) * c ** 2


# def to_single_channel(A):
#     """(N,72,14,2) -> (2N,72,14,1): real parts first, then imaginary parts (as in the paper)."""
#     return np.concatenate([A[..., :1], A[..., 1:]], axis=0)


# def from_single_channel(A):
#     """Inverse of to_single_channel."""
#     n = A.shape[0] // 2
#     return np.concatenate([A[:n], A[n:]], axis=-1)