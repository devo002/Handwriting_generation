import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResNetImageEncoder(nn.Module):
    """
    ResNet50 backbone, adapted to accept 50 input channels and return
    a list of 5 feature maps (each 512 channels).
    Compatible with your generator pipeline.
    """
    def __init__(self, in_channels=50, weight_path="/home/woody/iwi5/iwi5333h/model/resnet50-0676ba61.pth"):
        super().__init__()
        self.output_dim = 512

        # Load ResNet50
        self.model = resnet50(weights=None)
        if weight_path:
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        # Modify first conv to accept N channels
        original_conv = self.model.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            if in_channels > 3:
                new_conv.weight[:, 3:] = original_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
        self.model.conv1 = new_conv

        # Get intermediate outputs
        return_nodes = {
            "relu": "feat1",
            "layer1": "feat2",
            "layer2": "feat3",
            "layer3": "feat4",
            "layer4": "feat5",
        }
        self.extractor = create_feature_extractor(self.model, return_nodes=return_nodes)

        # Reduce channels to 512
        self.reduces = nn.ModuleList([
            nn.Conv2d(64, 512, 1),
            nn.Conv2d(256, 512, 1),
            nn.Conv2d(512, 512, 1),
            nn.Conv2d(1024, 512, 1),
            nn.Conv2d(2048, 512, 1),
        ])

    def encode_with_intermediate(self, x):
        feats = self.extractor(x)
        keys = ["feat1", "feat2", "feat3", "feat4", "feat5"]
        outs = [r(feats[k]) for r, k in zip(self.reduces, keys)]
        outs[-1] = F.interpolate(outs[-1], size=(8, 27), mode="bilinear", align_corners=False)
        return outs

    def forward(self, x):
        return self.encode_with_intermediate(x)
