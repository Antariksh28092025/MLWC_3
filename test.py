"""test.py - evaluates the trained ChannelNet models and the conventional estimator, and makes the figures.

Example:  python test.py                      (both channels, 1000 test samples per point)
          python test.py --n_test 4000        (the whole test split; slower)
Outputs (in --out): fig3_VehA..., fig4_SUI5..., high-SNR figures, fig5 (MSE vs pilots), and .csv/.json tables.
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import make_xy, mse_complex, to_single_channel
from models import SRCNN, DnCNN, predict, mean_sq
from estimator import full_correlation, mmse_mse

p = argparse.ArgumentParser()
p.add_argument("--chans", nargs="+", default=["VehA", "SUI5"])
p.add_argument("--snrs", nargs="+", type=int, default=[0, 5, 10, 12, 15, 20, 22, 25, 30])
p.add_argument("--pilot_counts", nargs="+", type=int, default=[16, 24, 32, 48])
p.add_argument("--pilot_snr", type=int, default=22, help="fixed SNR for the MSE-vs-pilots figure")
p.add_argument("--train_snrs", nargs="+", type=int, default=[12, 22])
p.add_argument("--n_test", type=int, default=1000, help="test samples per point (max 4000)")
p.add_argument("--ckpt", default="checkpoints")
p.add_argument("--out", default="results")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.out, exist_ok=True)
LS_NAME = "LS + interpolation"
MMSE_NAME = "MMSE (statistics from training set)"
net_name = lambda s: f"ChannelNet (trained at {s} dB)"
MARK = {LS_NAME: "s", MMSE_NAME: "^"}


def short(name):
    """Short label for console output."""
    if name == LS_NAME: return "LS"
    if name == MMSE_NAME: return "MMSE"
    return "Net@" + name.split("at ")[1].split(" ")[0]


def load_net(chan, nP, snr):
    """Returns (SRCNN, DnCNN) or None if the checkpoints do not exist."""
    base = os.path.join(args.ckpt, f"{chan}_P{nP}_snr{snr}")
    if not (os.path.exists(base + "_srcnn.pt") and os.path.exists(base + "_dncnn.pt")):
        return None
    ck_sr = torch.load(base + "_srcnn.pt", map_location="cpu")
    ck_dn = torch.load(base + "_dncnn.pt", map_location="cpu")
    sr, dn = SRCNN(), DnCNN(depth=ck_dn["args"]["dn_depth"])          # depth is stored in the checkpoint
    sr.load_state_dict(ck_sr["state"]); dn.load_state_dict(ck_dn["state"])
    return sr.to(dev).eval(), dn.to(dev).eval()


def to_tensor(A):
    """(N,72,14,2) -> (2N,1,72,14): real parts first, then imaginary parts."""
    return torch.from_numpy(np.ascontiguousarray(to_single_channel(A).transpose(0, 3, 1, 2))).to(dev)


def net_mse(nets, X, Y, c):
    sr, dn = nets
    est = predict(dn, predict(sr, to_tensor(X)))
    return 2.0 * mean_sq(est, to_tensor(Y)) * c ** 2                   # per-element complex MSE, original units


def line_plot(x, series, xlabel, title, fname, vlines=()):
    plt.figure(figsize=(6.4, 4.6))
    for name, ys in series.items():
        pts = [(a, b) for a, b in zip(x, ys) if b is not None]
        if pts:
            xs, yy = zip(*pts)
            plt.semilogy(xs, yy, marker=MARK.get(name, "o"), label=name)
    for v in vlines:
        plt.axvline(v, ls=":", color="gray", lw=0.8)
    plt.xlabel(xlabel); plt.ylabel("MSE (per resource element)"); plt.title(title)
    plt.grid(True, which="both", alpha=0.4); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.out, fname), dpi=200); plt.close()


def write_csv(fname, xname, x, series):
    with open(os.path.join(args.out, fname), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([xname] + list(series))
        for i, xv in enumerate(x):
            w.writerow([xv] + ["" if s[i] is None else f"{s[i]:.6f}" for s in series.values()])


all_results = {}
for chan in args.chans:
    print(f"\n===== {chan} =====")
    nets = {s: load_net(chan, 48, s) for s in args.train_snrs}
    for s, n in nets.items():
        print(f"ChannelNet trained at {s} dB:", "loaded" if n else "MISSING (skipped)")
    print("estimating channel correlation matrix from the training set ...")
    R = full_correlation(chan)

    # ---- MSE versus SNR (48 pilots) ----
    snr_series = {LS_NAME: [], MMSE_NAME: [], **{net_name(s): [] for s in args.train_snrs}}
    for snr in args.snrs:
        X, Y, c = make_xy(chan, 48, snr, "test", n=args.n_test)
        snr_series[LS_NAME].append(mse_complex(X, Y, c))
        snr_series[MMSE_NAME].append(mmse_mse(R, chan, 48, snr, args.n_test))
        for s in args.train_snrs:
            snr_series[net_name(s)].append(net_mse(nets[s], X, Y, c) if nets[s] else None)
        print(f"  SNR {snr:2d} dB: " + " | ".join(
            f"{short(k)} {v[-1]:.5f}" for k, v in snr_series.items() if v[-1] is not None))
    fig_no = {"VehA": "fig3", "SUI5": "fig4"}.get(chan, "fig_snr")
    line_plot(args.snrs, snr_series, "SNR (dB)", f"{chan}: MSE vs SNR, 48 pilots",
              f"{fig_no}_{chan}_mse_vs_snr.png", vlines=args.train_snrs)
    high = max(args.train_snrs)
    line_plot(args.snrs, {k: v for k, v in snr_series.items() if k in (LS_NAME, MMSE_NAME, net_name(high))},
              "SNR (dB)", f"{chan}: deep high-SNR case (trained at {high} dB)",
              f"highsnr_{chan}_mse_vs_snr.png", vlines=[high])
    write_csv(f"{chan}_mse_vs_snr.csv", "SNR_dB", args.snrs, snr_series)

    # ---- MSE versus number of pilots (fixed SNR) ----
    pil_series = {LS_NAME: [], MMSE_NAME: [], net_name(args.pilot_snr): []}
    for nP in args.pilot_counts:
        X, Y, c = make_xy(chan, nP, args.pilot_snr, "test", n=args.n_test)
        pil_series[LS_NAME].append(mse_complex(X, Y, c))
        pil_series[MMSE_NAME].append(mmse_mse(R, chan, nP, args.pilot_snr, args.n_test))
        nn = load_net(chan, nP, args.pilot_snr)
        pil_series[net_name(args.pilot_snr)].append(net_mse(nn, X, Y, c) if nn else None)
        print(f"  pilots {nP:2d} @ {args.pilot_snr} dB: " + " | ".join(
            f"{short(k)} {v[-1]:.5f}" for k, v in pil_series.items() if v[-1] is not None))
    line_plot(args.pilot_counts, pil_series, "number of pilots", f"{chan}: MSE vs pilots at {args.pilot_snr} dB",
              f"fig5_{chan}_mse_vs_pilots.png")
    write_csv(f"{chan}_mse_vs_pilots.csv", "pilots", args.pilot_counts, pil_series)
    all_results[chan] = {"snr": args.snrs, "mse_vs_snr": snr_series,
                         "pilots": args.pilot_counts, "mse_vs_pilots": pil_series}

with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump({"n_test": args.n_test, "results": all_results}, f, indent=1)
print("\nfigures and tables saved in", args.out)
