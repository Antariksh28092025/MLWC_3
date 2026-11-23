"""check_step7.py - verifies the Python pipeline against the MATLAB baseline."""
import numpy as np
from scipy.io import loadmat
from data_utils import *

base = loadmat(os.path.join(DATA, "baseline_LS_interp.mat"))
mse_matlab = base["mseSNR"]                              # channel x method x SNR (MATLAB)
snr_all = base["snrAll"].ravel().astype(int).tolist()
chans = ["VehA", "SUI5"]

print("--- A) LS+interp MSE on the test split: Python vs MATLAB (48 pilots, linear) ---")
worst = 0.0
for ci, chan in enumerate(chans):
    for snr in (12, 22):
        X, Y, c = make_xy(chan, 48, snr, "test")
        mp = mse_complex(X, Y, c)
        mm = float(mse_matlab[ci, 0, snr_all.index(snr)])
        rel = abs(mp - mm) / mm
        worst = max(worst, rel)
        print(f"{chan} {snr:2d} dB | X {X.shape} Y {Y.shape} | c={c:.4f} | "
              f"python={mp:.5f} matlab={mm:.5f} | rel.diff={rel:.1e}")
print("A) PASS" if worst < 1e-4 else "A) CHECK: Python and MATLAB disagree")

print("--- B) pilot pass-through: the input image must equal the LS values at the pilots ---")
X, Y, c = make_xy("VehA", 48, 22, "val")
re_p, im_p, pIdx = load_ls("VehA", 48, 22)
sub, slot = (pIdx - 1) % NSC, (pIdx - 1) // NSC
sl = SPLIT["val"]
err_re = np.abs(X[:, sub, slot, 0] * c - re_p[:, sl].T).max()
err_im = np.abs(X[:, sub, slot, 1] * c - im_p[:, sl].T).max()
print(f"max error at the pilots: Re {err_re:.1e}, Im {err_im:.1e}")
print("B) PASS" if max(err_re, err_im) < 1e-4 else "B) CHECK: grid ordering is wrong")

print("--- C) single-channel helpers ---")
Xs = to_single_channel(X)
print("to_single_channel:", X.shape, "->", Xs.shape, "| round trip exact:",
      np.array_equal(from_single_channel(Xs), X))