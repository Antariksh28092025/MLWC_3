"""train.py - trains ChannelNet (SRCNN stage, then DnCNN stage) for one channel model and one training SNR.

Example:  python train.py --chan VehA --snr 12
Outputs (in --out): <chan>_P<pilots>_snr<snr>_srcnn.pt / _dncnn.pt / _curves.json / _curves.png
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data_utils import make_xy, to_single_channel
from models import SRCNN, DnCNN, predict, mean_sq

p = argparse.ArgumentParser()
p.add_argument("--chan", required=True, choices=["VehA", "SUI5"])
p.add_argument("--snr", type=int, required=True, help="training SNR in dB (assignment: 12 and 22)")
p.add_argument("--pilots", type=int, default=48)
p.add_argument("--epochs_sr", type=int, default=30)
p.add_argument("--epochs_dn", type=int, default=30)
p.add_argument("--batch", type=int, default=128)          # paper
p.add_argument("--lr", type=float, default=1e-3)          # paper
p.add_argument("--patience", type=int, default=5, help="early stopping on validation loss")
p.add_argument("--dn_depth", type=int, default=20)        # paper: 20
p.add_argument("--max_train", type=int, default=0, help="use only the first N training samples (0 = all 32000)")
p.add_argument("--max_eval", type=int, default=0, help="use only the first N val/test samples (0 = all 4000)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="checkpoints")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.makedirs(args.out, exist_ok=True)
tag = f"{args.chan}_P{args.pilots}_snr{args.snr}"
print(f"device: {dev} | {tag} | seed {args.seed}")


def to_tensor(A, limit=0):
    """(N,72,14,2) numpy -> (2N,1,72,14) tensor: real parts first, then imaginary parts (as in the paper)."""
    if limit:
        A = A[:limit]
    A = to_single_channel(A).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(A)).to(dev)


# ---------------- data ----------------
Xtr, Ytr, c = make_xy(args.chan, args.pilots, args.snr, "train", n=args.max_train)
Xva, Yva, _ = make_xy(args.chan, args.pilots, args.snr, "val", n=args.max_eval)
Xte, Yte, _ = make_xy(args.chan, args.pilots, args.snr, "test", n=args.max_eval)
Xtr, Ytr = to_tensor(Xtr, args.max_train), to_tensor(Ytr, args.max_train)
Xva, Yva = to_tensor(Xva, args.max_eval), to_tensor(Yva, args.max_eval)
Xte, Yte = to_tensor(Xte, args.max_eval), to_tensor(Yte, args.max_eval)
print(f"images: train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}  | scale c = {c:.4f}")


def to_original(ms):
    """Tensor MSE (normalised units, real/imag stacked) -> per-element complex MSE in original units."""
    return 2.0 * ms * c ** 2


# ---------------- generic training loop ----------------
def fit(model, Xt, Yt, Xv, Yv, epochs, name):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    n = Xt.shape[0]
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        tr = tot / n
        va = mean_sq(predict(model, Xv), Yv)
        hist["train"].append(tr)
        hist["val"].append(va)
        print(f"[{name}] epoch {ep:3d}/{epochs} | train {tr:.5f} | val {va:.5f} | {time.time() - t0:.1f}s")
        if va < best:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{name}] early stop (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return hist


# ---------------- stage 1: SRCNN ----------------
sr = SRCNN().to(dev)
hist_sr = fit(sr, Xtr, Ytr, Xva, Yva, args.epochs_sr, "SRCNN")
torch.save({"state": sr.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_srcnn.pt")

# ---------------- stage 2: DnCNN on the (frozen) SRCNN output ----------------
Ztr, Zva, Zte = predict(sr, Xtr), predict(sr, Xva), predict(sr, Xte)
dn = DnCNN(depth=args.dn_depth).to(dev)
hist_dn = fit(dn, Ztr, Ytr, Zva, Yva, args.epochs_dn, "DnCNN")
torch.save({"state": dn.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_dncnn.pt")

# ---------------- test-set summary at the training SNR ----------------
res = {
    "LS + interpolation": to_original(mean_sq(Xte, Yte)),
    "SRCNN only": to_original(mean_sq(Zte, Yte)),
    "ChannelNet (SRCNN + DnCNN)": to_original(mean_sq(predict(dn, Zte), Yte)),
}
print(f"\nTest MSE at {args.snr} dB ({args.chan}, {args.pilots} pilots):")
for k, v in res.items():
    print(f"  {k:28s}: {v:.5f}")

with open(f"{args.out}/{tag}_curves.json", "w") as f:
    json.dump({"srcnn": hist_sr, "dncnn": hist_dn, "test_mse": res, "args": vars(args)}, f, indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, h, nm in zip(ax, (hist_sr, hist_dn), ("SRCNN stage", "DnCNN stage")):
        a.semilogy(range(1, len(h["train"]) + 1), h["train"], label="train")
        a.semilogy(range(1, len(h["val"]) + 1), h["val"], label="validation")
        a.set_title(f"{nm} ({tag})"); a.set_xlabel("epoch"); a.set_ylabel("MSE (normalised)")
        a.grid(True); a.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}/{tag}_curves.png", dpi=150)
except ImportError:
    print("matplotlib not installed: loss curves saved only in the json file")
print("saved checkpoints and curves in", args.out)




# """train.py - trains ChannelNet (SRCNN stage, then DnCNN stage) for one channel model and one training SNR.

# Example:  python train.py --chan VehA --snr 12
# Outputs (in --out): <chan>_P<pilots>_snr<snr>_srcnn.pt / _dncnn.pt / _curves.json / _curves.png
# """
# import argparse
# import copy
# import json
# import os
# import time

# import numpy as np
# import torch
# import torch.nn as nn

# from data_utils import make_xy, to_single_channel
# from models import SRCNN, DnCNN, predict, mean_sq

# p = argparse.ArgumentParser()
# p.add_argument("--chan", required=True, choices=["VehA", "SUI5"])
# p.add_argument("--snr", type=int, required=True, help="training SNR in dB (assignment: 12 and 22)")
# p.add_argument("--pilots", type=int, default=48)
# p.add_argument("--epochs_sr", type=int, default=30)
# p.add_argument("--epochs_dn", type=int, default=30)
# p.add_argument("--batch", type=int, default=128)          # paper
# p.add_argument("--lr", type=float, default=1e-3)          # paper
# p.add_argument("--patience", type=int, default=5, help="early stopping on validation loss")
# p.add_argument("--dn_depth", type=int, default=20)        # paper: 20
# p.add_argument("--max_train", type=int, default=0, help="use only the first N training samples (0 = all 32000)")
# p.add_argument("--max_eval", type=int, default=0, help="use only the first N val/test samples (0 = all 4000)")
# p.add_argument("--seed", type=int, default=0)
# p.add_argument("--out", default="checkpoints")
# args = p.parse_args()

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# dev = "cuda" if torch.cuda.is_available() else "cpu"
# torch.backends.cudnn.benchmark = True
# os.makedirs(args.out, exist_ok=True)
# tag = f"{args.chan}_P{args.pilots}_snr{args.snr}"
# print(f"device: {dev} | {tag} | seed {args.seed}")


# def to_tensor(A, limit=0):
#     """(N,72,14,2) numpy -> (2N,1,72,14) tensor: real parts first, then imaginary parts (as in the paper)."""
#     if limit:
#         A = A[:limit]
#     A = to_single_channel(A).transpose(0, 3, 1, 2)
#     return torch.from_numpy(np.ascontiguousarray(A)).to(dev)


# # ---------------- data ----------------
# Xtr, Ytr, c = make_xy(args.chan, args.pilots, args.snr, "train")
# Xva, Yva, _ = make_xy(args.chan, args.pilots, args.snr, "val")
# Xte, Yte, _ = make_xy(args.chan, args.pilots, args.snr, "test")
# Xtr, Ytr = to_tensor(Xtr, args.max_train), to_tensor(Ytr, args.max_train)
# Xva, Yva = to_tensor(Xva, args.max_eval), to_tensor(Yva, args.max_eval)
# Xte, Yte = to_tensor(Xte, args.max_eval), to_tensor(Yte, args.max_eval)
# print(f"images: train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}  | scale c = {c:.4f}")


# def to_original(ms):
#     """Tensor MSE (normalised units, real/imag stacked) -> per-element complex MSE in original units."""
#     return 2.0 * ms * c ** 2


# # ---------------- generic training loop ----------------
# def fit(model, Xt, Yt, Xv, Yv, epochs, name):
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_fn = nn.MSELoss()
#     hist = {"train": [], "val": []}
#     best, best_state, bad = float("inf"), None, 0
#     n = Xt.shape[0]
#     for ep in range(1, epochs + 1):
#         t0 = time.time()
#         model.train()
#         perm = torch.randperm(n, device=dev)
#         tot = 0.0
#         for i in range(0, n, args.batch):
#             idx = perm[i:i + args.batch]
#             loss = loss_fn(model(Xt[idx]), Yt[idx])
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#             tot += loss.item() * idx.numel()
#         tr = tot / n
#         va = mean_sq(predict(model, Xv), Yv)
#         hist["train"].append(tr)
#         hist["val"].append(va)
#         print(f"[{name}] epoch {ep:3d}/{epochs} | train {tr:.5f} | val {va:.5f} | {time.time() - t0:.1f}s")
#         if va < best:
#             best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
#         else:
#             bad += 1
#             if bad >= args.patience:
#                 print(f"[{name}] early stop (no val improvement for {args.patience} epochs)")
#                 break
#     model.load_state_dict(best_state)
#     return hist


# # ---------------- stage 1: SRCNN ----------------
# sr = SRCNN().to(dev)
# hist_sr = fit(sr, Xtr, Ytr, Xva, Yva, args.epochs_sr, "SRCNN")
# torch.save({"state": sr.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_srcnn.pt")

# # ---------------- stage 2: DnCNN on the (frozen) SRCNN output ----------------
# Ztr, Zva, Zte = predict(sr, Xtr), predict(sr, Xva), predict(sr, Xte)
# dn = DnCNN(depth=args.dn_depth).to(dev)
# hist_dn = fit(dn, Ztr, Ytr, Zva, Yva, args.epochs_dn, "DnCNN")
# torch.save({"state": dn.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_dncnn.pt")

# # ---------------- test-set summary at the training SNR ----------------
# res = {
#     "LS + interpolation": to_original(mean_sq(Xte, Yte)),
#     "SRCNN only": to_original(mean_sq(Zte, Yte)),
#     "ChannelNet (SRCNN + DnCNN)": to_original(mean_sq(predict(dn, Zte), Yte)),
# }
# print(f"\nTest MSE at {args.snr} dB ({args.chan}, {args.pilots} pilots):")
# for k, v in res.items():
#     print(f"  {k:28s}: {v:.5f}")

# with open(f"{args.out}/{tag}_curves.json", "w") as f:
#     json.dump({"srcnn": hist_sr, "dncnn": hist_dn, "test_mse": res, "args": vars(args)}, f, indent=1)

# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     for a, h, nm in zip(ax, (hist_sr, hist_dn), ("SRCNN stage", "DnCNN stage")):
#         a.semilogy(range(1, len(h["train"]) + 1), h["train"], label="train")
#         a.semilogy(range(1, len(h["val"]) + 1), h["val"], label="validation")
#         a.set_title(f"{nm} ({tag})"); a.set_xlabel("epoch"); a.set_ylabel("MSE (normalised)")
#         a.grid(True); a.legend()
#     fig.tight_layout()
#     fig.savefig(f"{args.out}/{tag}_curves.png", dpi=150)
# except ImportError:
#     print("matplotlib not installed: loss curves saved only in the json file")
# print("saved checkpoints and curves in", args.out)

# '''"""train.py - trains ChannelNet (SRCNN stage, then DnCNN stage) for one channel model and one training SNR.

# Example:  python train.py --chan VehA --snr 12
# Outputs (in --out): <chan>_P<pilots>_snr<snr>_srcnn.pt / _dncnn.pt / _curves.json / _curves.png
# """
# import argparse
# import copy
# import json
# import os
# import time

# import numpy as np
# import torch
# import torch.nn as nn

# from data_utils import make_xy, to_single_channel
# from models import SRCNN, DnCNN, predict, mean_sq

# p = argparse.ArgumentParser()
# p.add_argument("--chan", required=True, choices=["VehA", "SUI5"])
# p.add_argument("--snr", type=int, required=True, help="training SNR in dB (assignment: 12 and 22)")
# p.add_argument("--pilots", type=int, default=48)
# p.add_argument("--epochs_sr", type=int, default=30)
# p.add_argument("--epochs_dn", type=int, default=30)
# p.add_argument("--batch", type=int, default=128)          # paper
# p.add_argument("--lr", type=float, default=1e-3)          # paper
# p.add_argument("--patience", type=int, default=5, help="early stopping on validation loss")
# p.add_argument("--dn_depth", type=int, default=20)        # paper: 20
# p.add_argument("--max_train", type=int, default=0, help="use only the first N training samples (0 = all 32000)")
# p.add_argument("--max_eval", type=int, default=0, help="use only the first N val/test samples (0 = all 4000)")
# p.add_argument("--seed", type=int, default=0)
# p.add_argument("--out", default="checkpoints")
# args = p.parse_args()

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# dev = "cuda" if torch.cuda.is_available() else "cpu"
# torch.backends.cudnn.benchmark = True
# os.makedirs(args.out, exist_ok=True)
# tag = f"{args.chan}_P{args.pilots}_snr{args.snr}"
# print(f"device: {dev} | {tag} | seed {args.seed}")


# def to_tensor(A, limit=0):
#     """(N,72,14,2) numpy -> (2N,1,72,14) tensor: real parts first, then imaginary parts (as in the paper)."""
#     if limit:
#         A = A[:limit]
#     A = to_single_channel(A).transpose(0, 3, 1, 2)
#     return torch.from_numpy(np.ascontiguousarray(A)).to(dev)


# # ---------------- data ----------------
# Xtr, Ytr, c = make_xy(args.chan, args.pilots, args.snr, "train", n=args.max_train)
# Xva, Yva, _ = make_xy(args.chan, args.pilots, args.snr, "val", n=args.max_eval)
# Xte, Yte, _ = make_xy(args.chan, args.pilots, args.snr, "test", n=args.max_eval)
# Xtr, Ytr = to_tensor(Xtr, args.max_train), to_tensor(Ytr, args.max_train)
# Xva, Yva = to_tensor(Xva, args.max_eval), to_tensor(Yva, args.max_eval)
# Xte, Yte = to_tensor(Xte, args.max_eval), to_tensor(Yte, args.max_eval)
# print(f"images: train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}  test {tuple(Xte.shape)}  | scale c = {c:.4f}")


# def to_original(ms):
#     """Tensor MSE (normalised units, real/imag stacked) -> per-element complex MSE in original units."""
#     return 2.0 * ms * c ** 2


# # ---------------- generic training loop ----------------
# def fit(model, Xt, Yt, Xv, Yv, epochs, name):
#     opt = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_fn = nn.MSELoss()
#     hist = {"train": [], "val": []}
#     best, best_state, bad = float("inf"), None, 0
#     n = Xt.shape[0]
#     for ep in range(1, epochs + 1):
#         t0 = time.time()
#         model.train()
#         perm = torch.randperm(n, device=dev)
#         tot = 0.0
#         for i in range(0, n, args.batch):
#             idx = perm[i:i + args.batch]
#             loss = loss_fn(model(Xt[idx]), Yt[idx])
#             opt.zero_grad(set_to_none=True)
#             loss.backward()
#             opt.step()
#             tot += loss.item() * idx.numel()
#         tr = tot / n
#         va = mean_sq(predict(model, Xv), Yv)
#         hist["train"].append(tr)
#         hist["val"].append(va)
#         print(f"[{name}] epoch {ep:3d}/{epochs} | train {tr:.5f} | val {va:.5f} | {time.time() - t0:.1f}s")
#         if va < best:
#             best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
#         else:
#             bad += 1
#             if bad >= args.patience:
#                 print(f"[{name}] early stop (no val improvement for {args.patience} epochs)")
#                 break
#     model.load_state_dict(best_state)
#     return hist


# # ---------------- stage 1: SRCNN ----------------
# sr = SRCNN().to(dev)
# hist_sr = fit(sr, Xtr, Ytr, Xva, Yva, args.epochs_sr, "SRCNN")
# torch.save({"state": sr.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_srcnn.pt")

# # ---------------- stage 2: DnCNN on the (frozen) SRCNN output ----------------
# Ztr, Zva, Zte = predict(sr, Xtr), predict(sr, Xva), predict(sr, Xte)
# dn = DnCNN(depth=args.dn_depth).to(dev)
# hist_dn = fit(dn, Ztr, Ytr, Zva, Yva, args.epochs_dn, "DnCNN")
# torch.save({"state": dn.state_dict(), "c": c, "args": vars(args)}, f"{args.out}/{tag}_dncnn.pt")

# # ---------------- test-set summary at the training SNR ----------------
# res = {
#     "LS + interpolation": to_original(mean_sq(Xte, Yte)),
#     "SRCNN only": to_original(mean_sq(Zte, Yte)),
#     "ChannelNet (SRCNN + DnCNN)": to_original(mean_sq(predict(dn, Zte), Yte)),
# }
# print(f"\nTest MSE at {args.snr} dB ({args.chan}, {args.pilots} pilots):")
# for k, v in res.items():
#     print(f"  {k:28s}: {v:.5f}")

# with open(f"{args.out}/{tag}_curves.json", "w") as f:
#     json.dump({"srcnn": hist_sr, "dncnn": hist_dn, "test_mse": res, "args": vars(args)}, f, indent=1)

# try:
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots(1, 2, figsize=(10, 4))
#     for a, h, nm in zip(ax, (hist_sr, hist_dn), ("SRCNN stage", "DnCNN stage")):
#         a.semilogy(range(1, len(h["train"]) + 1), h["train"], label="train")
#         a.semilogy(range(1, len(h["val"]) + 1), h["val"], label="validation")
#         a.set_title(f"{nm} ({tag})"); a.set_xlabel("epoch"); a.set_ylabel("MSE (normalised)")
#         a.grid(True); a.legend()
#     fig.tight_layout()
#     fig.savefig(f"{args.out}/{tag}_curves.png", dpi=150)
# except ImportError:
#     print("matplotlib not installed: loss curves saved only in the json file")
# print("saved checkpoints and curves in", args.out)
# '''