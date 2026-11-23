"""models.py - SRCNN and DnCNN exactly as described in the ChannelNet paper (Sec. III-B)."""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """Super-resolution stage: 9x9x64 -> ReLU -> 1x1x32 -> ReLU -> 5x5x1 (one 72x14 image in, one out)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class DnCNN(nn.Module):
    """Denoising stage: `depth` conv layers (paper: 20).
    Layer 1: 3x3x1->64 + ReLU; layers 2..depth-1: 3x3x64 + BatchNorm + ReLU; last: 3x3x64->1.
    Residual learning: the body predicts the noise, the estimate is input - predicted noise."""

    def __init__(self, depth=20, width=64):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(width, 1, 3, padding=1, bias=False)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.body(x)


@torch.no_grad()
def predict(model, X, bs=250):
    """Run a model over a big tensor in batches (eval mode, no gradients)."""
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, X.shape[0], bs)])


def mean_sq(a, b):
    """Mean squared error over all elements (a real-valued tensor MSE)."""
    return torch.mean((a - b) ** 2).item()

# """models.py - SRCNN and DnCNN exactly as described in the ChannelNet paper (Sec. III-B)."""
# import torch
# import torch.nn as nn


# class SRCNN(nn.Module):
#     """Super-resolution stage: 9x9x64 -> ReLU -> 1x1x32 -> ReLU -> 5x5x1 (one 72x14 image in, one out)."""

#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 1, kernel_size=5, padding=2),
#         )

#     def forward(self, x):
#         return self.net(x)


# class DnCNN(nn.Module):
#     """Denoising stage: `depth` conv layers (paper: 20).
#     Layer 1: 3x3x1->64 + ReLU; layers 2..depth-1: 3x3x64 + BatchNorm + ReLU; last: 3x3x64->1.
#     Residual learning: the body predicts the noise, the estimate is input - predicted noise."""

#     def __init__(self, depth=20, width=64):
#         super().__init__()
#         layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
#         for _ in range(depth - 2):
#             layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
#                        nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
#         layers += [nn.Conv2d(width, 1, 3, padding=1, bias=False)]
#         self.body = nn.Sequential(*layers)

#     def forward(self, x):
#         return x - self.body(x)


# @torch.no_grad()
# def predict(model, X, bs=1000):
#     """Run a model over a big tensor in batches (eval mode, no gradients)."""
#     model.eval()
#     return torch.cat([model(X[i:i + bs]) for i in range(0, X.shape[0], bs)])


# def mean_sq(a, b):
#     """Mean squared error over all elements (a real-valued tensor MSE)."""
#     return torch.mean((a - b) ** 2).item()
