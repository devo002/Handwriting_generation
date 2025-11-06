import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_l

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientNetImageEncoder(nn.Module):
    """
    EfficientNetV2-L backbone, adapted for 50-channel input and generator compatibility.
    Outputs list of 5 feature maps, each 512 channels, final resized to (8, 27).
    """
    def __init__(self, in_channels=50, weight_path="/home/woody/iwi5/iwi5333h/model/efficientnet_v2_l-59c71312.pth"):
        super().__init__()
        self.output_dim = 512
        self.model = efficientnet_v2_l(weights=None)

        if weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # Replace first conv layer to handle multiple channels
        stem_conv = self.model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = stem_conv.weight
            if in_channels > 3:
                new_conv.weight[:, 3:] = stem_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)

        self.model.features[0][0] = new_conv
        self.features = nn.Sequential(*list(self.model.features.children()))

        # Channel reducers
        self.reduce_layers = nn.ModuleList()
        for i, block in enumerate(self.features):
            if i in [1, 2, 3, 4, 5]:
                out_channels = next(m.out_channels for m in reversed(list(block.modules())) if isinstance(m, nn.Conv2d))
                self.reduce_layers.append(nn.Conv2d(out_channels, 512, kernel_size=1))

    def encode_with_intermediate(self, x):
        results, ridx = [], 0
        for i, block in enumerate(self.features):
            x = block(x)
            if i in [1, 2, 3, 4, 5]:
                reduced = self.reduce_layers[ridx](x)
                results.append(reduced)
                ridx += 1
        results[-1] = F.interpolate(results[-1], size=(8, 27), mode="bilinear", align_corners=False)
        return results[-5:]

    def forward(self, x):
        return self.encode_with_intermediate(x)
