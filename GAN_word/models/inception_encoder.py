# models/inception_encoder.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from typing import Tuple, List, Optional

# ---- fixed defaults ----
DEFAULT_IN_CHANNELS = 50
DEFAULT_FINAL_SIZE: Tuple[int, int] = (8, 27)
DEFAULT_PROBE_SIZE: Tuple[int, int] = (48, 540)
DEFAULT_AUX_LOGITS = False
DEFAULT_SOFTEN_DOWNSAMPLING = True

# ✅ your local pretrained weights path
DEFAULT_WEIGHTS = "/home/woody/iwi5/iwi5333h/model/inception_v3_imagenet1k_v1.pth"


class ImageEncoderInceptionV3(nn.Module):
    """
    Inception v3 encoder that returns 5 intermediate feature maps reduced to 512 channels:
      taps = ["Mixed_5c", "Mixed_5d", "Mixed_6b", "Mixed_6e", "Mixed_7c"]
    Supports arbitrary in_channels (default 50) and short-height inputs (e.g., 48xW).
    Last feature map is resized to DEFAULT_FINAL_SIZE for your decoder path.
    """

    def __init__(
        self,
        in_channels: int = DEFAULT_IN_CHANNELS,
        final_size: Tuple[int, int] = DEFAULT_FINAL_SIZE,
        weight_path: Optional[str] = DEFAULT_WEIGHTS,
        aux_logits: bool = DEFAULT_AUX_LOGITS,
        soften_downsampling: bool = DEFAULT_SOFTEN_DOWNSAMPLING,
        probe_size: Tuple[int, int] = DEFAULT_PROBE_SIZE,
    ):
        super().__init__()
        self.output_dim = 512
        self.final_size = final_size
        self.probe_size = probe_size

        # ---- backbone ----
        self.model = inception_v3(weights=None, aux_logits=aux_logits, transform_input=False)

        # ---- load local pretrained weights ----
        if weight_path and os.path.isfile(weight_path):
            sd = torch.load(weight_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            elif isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)

        # ---- modify first conv to accept custom in_channels ----
        first = self.model.Conv2d_1a_3x3.conv
        new_first = nn.Conv2d(
            in_channels,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=(first.bias is not None),
        )
        with torch.no_grad():
            if first.weight.shape[1] == 3:
                new_first.weight[:, :3] = first.weight
                if in_channels > 3:
                    rep = first.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
                    new_first.weight[:, 3:] = rep
            else:
                nn.init.kaiming_normal_(new_first.weight, mode="fan_out", nonlinearity="relu")
                if new_first.bias is not None:
                    nn.init.zeros_(new_first.bias)
        self.model.Conv2d_1a_3x3.conv = new_first

        # ---- handle pooling layers ----
        pool1 = getattr(self.model, "MaxPool_3a_3x3", None) or getattr(self.model, "maxpool1")
        pool2 = getattr(self.model, "MaxPool_5a_3x3", None) or getattr(self.model, "maxpool2")
        self.pool1 = pool1
        self.pool2 = pool2

        # ---- soften early downsampling for low-height inputs ----
        if soften_downsampling:
            self.model.Conv2d_1a_3x3.conv.stride = (1, 1)
            if hasattr(self.pool1, "pool"):
                self.pool1.pool.stride = (1, 1)
            else:
                self.pool1.stride = (1, 1)

        # ---- ordered forward graph ----
        self.blocks = nn.ModuleDict({
            "Conv2d_1a_3x3": self.model.Conv2d_1a_3x3,
            "Conv2d_2a_3x3": self.model.Conv2d_2a_3x3,
            "Conv2d_2b_3x3": self.model.Conv2d_2b_3x3,
            "pool1": self.pool1,
            "Conv2d_3b_1x1": self.model.Conv2d_3b_1x1,
            "Conv2d_4a_3x3": self.model.Conv2d_4a_3x3,
            "pool2": self.pool2,
            "Mixed_5b": self.model.Mixed_5b,
            "Mixed_5c": self.model.Mixed_5c,
            "Mixed_5d": self.model.Mixed_5d,
            "Mixed_6a": self.model.Mixed_6a,
            "Mixed_6b": self.model.Mixed_6b,
            "Mixed_6c": self.model.Mixed_6c,
            "Mixed_6d": self.model.Mixed_6d,
            "Mixed_6e": self.model.Mixed_6e,
            "Mixed_7a": self.model.Mixed_7a,
            "Mixed_7b": self.model.Mixed_7b,
            "Mixed_7c": self.model.Mixed_7c,
        })
        self.block_order = list(self.blocks.keys())

        # ---- feature collection points ----
        self.collect_points = ["Mixed_5c", "Mixed_5d", "Mixed_6b", "Mixed_6e", "Mixed_7c"]

        # ---- determine in_channels for reducers using dummy forward ----
        with torch.no_grad():
            H, W = self.probe_size
            dummy = torch.zeros(1, in_channels, H, W)
            ch_list: List[int] = []
            x = dummy
            for name in self.block_order:
                x = self.blocks[name](x)
                if name in self.collect_points:
                    ch_list.append(x.shape[1])
            if len(ch_list) != len(self.collect_points):
                raise RuntimeError(
                    f"Expected {len(self.collect_points)} taps, got {len(ch_list)}"
                )

        # ---- 1x1 conv reducers to 512 ----
        self.reduce_layers = nn.ModuleList([nn.Conv2d(c, 512, kernel_size=1) for c in ch_list])

    def encode_with_intermediate(self, x: torch.Tensor):
        results: List[torch.Tensor] = []
        red_idx = 0
        h = x
        for name in self.block_order:
            h = self.blocks[name](h)
            if name in self.collect_points:
                results.append(self.reduce_layers[red_idx](h))
                red_idx += 1

        # resize final map
        results[-1] = F.interpolate(results[-1], size=self.final_size, mode="bilinear", align_corners=False)
        return results  # list of 5 tensors

    def forward(self, x: torch.Tensor):
        return self.encode_with_intermediate(x)
