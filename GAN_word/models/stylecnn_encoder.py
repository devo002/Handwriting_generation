# models/stylecnn_encoder.py
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, Union, List

# ---- fixed defaults ----
DEFAULT_IN_CHANNELS = 50
DEFAULT_FINAL_SIZE: Tuple[int, int] = (8, 27)


# ---------------- utils (self-contained) ----------------
def primeFactors(n: Union[int, np.integer, torch.Tensor]) -> Iterable[int]:
    """Return prime factorization with multiplicities (e.g., 36 -> [2,2,3,3])."""
    if hasattr(n, "item"):
        try:
            n = int(n.item())
        except Exception:
            n = int(n)
    else:
        n = int(n)

    if n < 2:
        return []

    factors: List[int] = []
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            factors.append(f)
            n //= f
        f += 2
    if n > 1:
        factors.append(n)
    return factors


def getGroupSize(channels: int) -> int:
    goalSize = 8 if channels >= 32 else 4
    if channels % goalSize == 0:
        return goalSize
    factors = primeFactors(channels)
    bestDist = 9999
    bestGroup = 1
    for f in factors:
        if abs(f - goalSize) <= bestDist:  # favor larger when equal
            bestDist = abs(f - goalSize)
            bestGroup = f
    return int(bestGroup)


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride,
        padding=0,
        norm="none",
        activation="relu",
        pad_type="zero",
        transpose=False,
        reverse=False,
    ):
        super().__init__()
        self.reverse = reverse
        self.use_bias = True

        # padding
        if transpose:
            self.pad = lambda x: x
        elif pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise AssertionError(f"Unsupported padding type: {pad_type}")

        # normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "group":
            self.norm = nn.GroupNorm(getGroupSize(norm_dim), norm_dim)
        elif norm in ("none", "sn"):
            self.norm = None
        elif norm == "ln":
            raise NotImplementedError("LayerNorm not wired in this repo")
        elif norm == "adain":
            raise NotImplementedError("AdaIN not wired in this repo")
        else:
            raise AssertionError(f"Unsupported normalization: {norm}")

        # activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "logsoftmax":
            self.activation = nn.LogSoftmax(dim=1)
        elif activation == "none":
            self.activation = None
        else:
            raise AssertionError(f"Unsupported activation: {activation}")

        # convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(
                input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=padding
            )
        elif norm == "sn":
            raise NotImplementedError("SpectralNorm path not implemented here")
        else:
            self.conv = nn.Conv2d(
                input_dim, output_dim, kernel_size, stride, bias=self.use_bias, padding=0
            )

    def forward(self, x):
        if not self.reverse:
            x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        if self.reverse:
            x = self.conv(self.pad(x))
        return x


# ---------------- encoder ----------------
class ImageEncoderStyleCNN(nn.Module):
    """
    Drop-in replacement for enc_image that mimics your VGG pyramid using lightweight CNN blocks.
    - Accepts DEFAULT_IN_CHANNELS (50) by default
    - Returns a list: [f1, f2, f3, f4, f5, f6]
    - f6 is resized to DEFAULT_FINAL_SIZE (8, 27) for generator compatibility
    Shapes (approx) for input ~ (B, 50, 48, 540):
      f1: (B,  64, 48, 540)
      f2: (B, 128, 48, 540)
      f3: (B, 256, 24, 270)
      f4: (B, 512, 12, 135)
      f5: (B, 512,  6,  67)
      f6: (B, 512,  8,  27)  # after interpolate
    """

    def __init__(self, in_channels: int = DEFAULT_IN_CHANNELS, final_size: Tuple[int, int] = DEFAULT_FINAL_SIZE):
        super().__init__()
        self.final_size = final_size

        # Stage 1: keep spatial (→ f1: 64, H, W)
        self.enc1 = nn.Sequential(
            Conv2dBlock(in_channels, 64, 5, 1, 2, pad_type="reflect", norm="in", activation="relu")
        )

        # Stage 2: keep spatial (→ f2: 128, H, W)
        self.enc2 = nn.Sequential(
            Conv2dBlock(64, 128, 3, 1, 1, pad_type="reflect", norm="in", activation="relu"),
            Conv2dBlock(128, 128, 3, 1, 1, pad_type="reflect", norm="in", activation="relu"),
        )

        # Stage 3: downsample (2,2) then a 3x3 with asymmetric pad (→ f3: 256, H/2, W/2)
        self.enc3 = nn.Sequential(
            Conv2dBlock(128, 256, 4, 2, 1, pad_type="reflect", norm="in", activation="relu"),
            nn.ReflectionPad2d((1, 1, 0, 0)),
            Conv2dBlock(256, 256, 3, 1, 0, pad_type="reflect", norm="in", activation="relu"),
        )

        # Stage 4: downsample (2,2) then a 3x3 with asymmetric pad (→ f4: 512, H/4, W/4)
        self.enc4 = nn.Sequential(
            Conv2dBlock(256, 512, 4, 2, 1, pad_type="reflect", norm="in", activation="relu"),
            nn.ReflectionPad2d((1, 1, 0, 0)),
            Conv2dBlock(512, 512, 3, 1, 0, pad_type="reflect", norm="in", activation="relu"),
        )

        # Stage 5: anisotropic stride (2,1) + light width pooling (→ f5: 512, ~H/8, ~W/8)
        self.enc5 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 0, 0)),
            Conv2dBlock(512, 512, (4, 4), (2, 1), 0, pad_type="reflect", norm="in", activation="relu"),
            nn.AvgPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # ~135 → ~67
        )

        # Stage 6: refine and then force to (8,27)
        self.enc6 = nn.Sequential(
            Conv2dBlock(512, 512, 3, 1, 1, pad_type="reflect", norm="in", activation="relu"),
        )

    def encode_with_intermediate(self, x: torch.Tensor):
        r1 = self.enc1(x)   # (B,  64, H,   W)
        r2 = self.enc2(r1)  # (B, 128, H,   W)
        r3 = self.enc3(r2)  # (B, 256, H/2, W/2)
        r4 = self.enc4(r3)  # (B, 512, H/4, W/4)
        r5 = self.enc5(r4)  # (B, 512, ~H/8, ~W/8)
        r6 = self.enc6(r5)  # (B, 512, ~H/8, ~W/8)
        r6 = F.interpolate(r6, size=self.final_size, mode="bilinear", align_corners=False)  # -> (B,512,8,27)
        return [r1, r2, r3, r4, r5, r6]

    def forward(self, x: torch.Tensor):
        return self.encode_with_intermediate(x)
