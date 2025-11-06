# models/vgg_encoder.py
import torch
from torch import nn
from vgg_tro_channel3_modi import vgg19_bn

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vgg19_bn(False)
        self.output_dim = 512

        enc_layers = list(self.model.features.children())
        enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:3]).to(gpu))
        enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[3:9]).to(gpu))
        enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[9:16]).to(gpu))
        enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[16:29]).to(gpu))
        enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[29:42]).to(gpu))
        enc_6 = nn.DataParallel(nn.Sequential(*enc_layers[42:]).to(gpu))
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5, enc_6]

    def encode_with_intermediate(self, x):
        results = [x]
        for i in range(6):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, x):
        return self.encode_with_intermediate(x)
