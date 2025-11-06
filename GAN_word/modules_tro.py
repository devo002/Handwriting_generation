# main_gan.py
import os
import importlib, inspect, os, sys
import yaml
import inspect
import cv2
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import Conv2dBlock, ResBlocks          
from load_data import OUTPUT_MAX_LEN, IMG_HEIGHT, IMG_WIDTH, vocab_size, index2letter, num_tokens, tokens
from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
from recognizer.models.encoder_vgg import Encoder as rec_encoder
from recognizer.models.decoder import Decoder as rec_decoder
from recognizer.models.seq2seqnew2 import Seq2Seq as rec_seq2seq
from recognizer.models.attention import locationAttention as rec_attention

gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def build_image_encoder(name: str) -> nn.Module:
    name = (name or "").strip().lower()
    mapping = {
        "vgg":         ("models.vgg_encoder",        "VGGImageEncoder"),
        "resnet":      ("models.resnet_encoder",     "ResNetImageEncoder"),
        "efficientnet":("models.efficientnet_encoder","EfficientNetImageEncoder"),
        "dino":        ("models.dino_encoder",       "ImageEncoderDINOv2"),
        "inception":   ("models.inception_encoder",  "ImageEncoderInceptionV3"),
        "stylecnn":    ("models.stylecnn_encoder",   "ImageEncoderStyleCNN"),
    }
    if name not in mapping:
        raise ValueError(f"Unknown image encoder '{name}'")

    module_path, class_name = mapping[name]
    #print(f"[factory] loading {module_path}.{class_name}", flush=True)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    inst = cls().to(gpu)

    try:
        src = inspect.getfile(cls)
    except Exception:
        src = "<unknown>"
    #print(f"[factory] built: {cls.__name__} from {cls.__module__} ({src})", flush=True)
    return inst

class TextEncoder_FC(nn.Module):
    def __init__(self, text_max_len: int):
        super().__init__()
        embed_size = 64
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
            nn.Linear(text_max_len * embed_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=False),
            nn.Linear(2048, 4096),
        )
        self.linear = nn.Linear(embed_size, 512)

    def forward(self, x, f_xs_shape):
        # x: (B, T)
        xx = self.embed(x)  

        batch_size = xx.shape[0]
        xxx = xx.reshape(batch_size, -1)  
        out = self.fc(xxx)           

       
        xx_new = self.linear(xx)          
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = max(1, f_xs_shape[-1] // ts)

        tiles = []
        for i in range(ts):
            tok = xx_new[:, i:i+1]                 
            tiles.append(tok.repeat(1, width_reps, 1))  

        padding_reps = f_xs_shape[-1] % ts
        if padding_reps:
            pad_tok = self.embed(torch.full((1, 1), tokens['PAD_TOKEN'], dtype=torch.long, device=xx.device))
            pad_tok = self.linear(pad_tok).repeat(batch_size, padding_reps, 1)
            tiles.append(pad_tok)

        res = torch.cat(tiles, dim=1)            
        res = res.permute(0, 2, 1).unsqueeze(2)  
        final_res = res.repeat(1, 1, height_reps, 1)  
        return out, final_res



class Decoder(nn.Module):
    def __init__(self, ups=3, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super().__init__()
        layers = []
        layers += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        for _ in range(ups):
            layers += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='in', activation=activ, pad_type=pad_type),
            ]
            dim //= 2
        layers += [Conv2dBlock(dim, out_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GenModel_FC(nn.Module):
    """
    - enc_image: backbone picked by config (vgg/resnet/efficientnet/dino/inception/stylecnn)
    - enc_text: text embedding path producing (B,4096) and (B,512,H,W)
    - dec: AdaIN-based decoder (same as your implementation)
    - mix(): concatenates last image feat with text grid and reduces to 512
    """
    def __init__(self, text_max_len: int, encoder_name: str):
        super().__init__()
        self.enc_image = build_image_encoder(encoder_name)   
        self.enc_text = TextEncoder_FC(text_max_len).to(gpu)
        self.dec = Decoder().to(gpu)

        self.linear_mix = nn.Linear(1024, 512)
        self.max_conv = nn.MaxPool2d(kernel_size=2, stride=2)
        
    # def __init__(self, text_max_len: int, encoder_name: str):
    #     super().__init__()
    #     if not encoder_name:
    #         raise ValueError("encoder_name must be provided (e.g. 'vgg', 'dino', 'resnet', ...)")

    #     # build encoder and loudly report what we got
    #     self.enc_image = build_image_encoder(encoder_name)

    #     # extra, redundant proof (in case factory logs are missed)
    #     try:
    #         import inspect
    #         cls = self.enc_image.__class__
    #         src = inspect.getfile(cls)
    #         print(
    #             f"[GenModel_FC] encoder_name='{encoder_name}' -> {cls.__name__} "
    #             f"from {cls.__module__} ({src})",
    #             flush=True,
    #         )
    #     except Exception as e:
    #         print(f"[GenModel_FC] encoder introspection failed: {e}", flush=True)

        self.enc_text = TextEncoder_FC(text_max_len).to(gpu)
        self.dec = Decoder().to(gpu)

        self.linear_mix = nn.Linear(1024, 512)
        self.max_conv = nn.MaxPool2d(kernel_size=2, stride=2)

    def assign_adain_params(self, adain_params, results, embed):
        i = 0
        for m in self.dec.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std  = adain_params[:, m.num_features:2*m.num_features]
                m.bias   = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                m.con    = embed
                if i == 1 and len(results) >= 4:
                    m.input = self.max_conv(results[3])
                elif i == 3 and len(results) >= 5:
                    m.input = results[4]
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
                i += 1

    def decode(self, content, results, embed, adain_params):
        self.assign_adain_params(adain_params, results, embed)
        return self.dec(content)

    def mix(self, results, feat_embed):
        feat_mix = torch.cat([results[-1], feat_embed], dim=1) 
        f = feat_mix.permute(0, 2, 3, 1)                    
        ff = self.linear_mix(f)                            
        return ff.permute(0, 3, 1, 2)                

    def forward_image_features(self, x):
        return self.enc_image(x)

    def forward_text_features(self, tokens, f_shape):
        return self.enc_text(tokens, f_shape)


# class DisModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.n_layers = 6
#         self.final_size = 1024
#         nf = 16
#         cnn_f = [Conv2dBlock(1, nf, 7, 1, 3, pad_type='reflect', norm='none', activation='none')]
#         for _ in range(self.n_layers - 1):
#             nf_out = min(nf * 2, 1024)
#             cnn_f += [Conv2dBlock(nf, nf, 3, 1, 1, pad_type='reflect', norm='none', activation='lrelu')]
#             cnn_f += [Conv2dBlock(nf, nf_out, 3, 1, 1, pad_type='reflect', norm='none', activation='lrelu')]
#             cnn_f += [nn.ReflectionPad2d(1)]
#             cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
#             nf = min(nf * 2, 1024)
#         nf_out = min(nf * 2, 1024)
#         cnn_f += [Conv2dBlock(nf, nf, 3, 1, 1, pad_type='reflect', norm='none', activation='lrelu')]
#         cnn_f += [Conv2dBlock(nf, nf_out, 3, 1, 1, pad_type='reflect', norm='none', activation='lrelu')]
#         cnn_c = [Conv2dBlock(nf_out,
#                              self.final_size,
#                              IMG_HEIGHT // (2 ** (self.n_layers - 1)),
#                              IMG_WIDTH // (2 ** (self.n_layers - 1)) + 1,
#                              norm='none', activation='lrelu', pad_type='reflect')]
#         self.cnn_f = nn.Sequential(*cnn_f)
#         self.cnn_c = nn.Sequential(*cnn_c)
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, x):
#         feat = self.cnn_f(x)
#         out = self.cnn_c(feat)
#         return out.squeeze(-1).squeeze(-1)

#     def calc_dis_fake_loss(self, input_fake):
#         label = torch.zeros(input_fake.shape[0], self.final_size, device=gpu)
#         resp_fake = self.forward(input_fake)
#         return self.bce(resp_fake, label)

#     def calc_dis_real_loss(self, input_real):
#         label = torch.ones(input_real.shape[0], self.final_size, device=gpu)
#         resp_real = self.forward(input_real)
#         return self.bce(resp_real, label)

#     def calc_gen_loss(self, input_fake):
#         label = torch.ones(input_fake.shape[0], self.final_size, device=gpu)
#         resp_fake = self.forward(input_fake)
#         return self.bce(resp_fake, label)
    
class DisModel(nn.Module):
    def __init__(self):
        super(DisModel, self).__init__()
        self.n_layers = 6
        self.final_size = 1024
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, self.final_size, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        return out.squeeze(-1).squeeze(-1) # b,1024   maybe b is also 1, so cannnot out.squeeze()

    def calc_dis_fake_loss(self, input_fake):
        label = torch.zeros(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

    def calc_dis_real_loss(self, input_real):
        label = torch.ones(input_real.shape[0], self.final_size).to(gpu)
        resp_real = self.forward(input_real)
        real_loss = self.bce(resp_real, label)
        return real_loss

    def calc_gen_loss(self, input_fake):
        label = torch.ones(input_fake.shape[0], self.final_size).to(gpu)
        resp_fake = self.forward(input_fake)
        fake_loss = self.bce(resp_fake, label)
        return fake_loss

class WriterClaModel(nn.Module):
    def __init__(self, num_writers):
        super(WriterClaModel, self).__init__()
        self.n_layers = 6
        nf = 16
        cnn_f = [Conv2dBlock(1, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, num_writers, IMG_HEIGHT//(2**(self.n_layers-1)), IMG_WIDTH//(2**(self.n_layers-1))+1,
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat) # b,310,1,1
        loss = self.cross_entropy(out.squeeze(-1).squeeze(-1), y)
        return loss

class RecModel(nn.Module):
    def __init__(self, pretrain=False):
        super(RecModel, self).__init__()
        hidden_size_enc = hidden_size_dec = 512
        embed_size = 60
        self.enc = rec_encoder(hidden_size_enc, IMG_HEIGHT, IMG_WIDTH, True, None, False).to(gpu)
        self.dec = rec_decoder(hidden_size_dec, embed_size, vocab_size, rec_attention, None).to(gpu)
        self.seq2seq = rec_seq2seq(self.enc, self.dec, OUTPUT_MAX_LEN, vocab_size).to(gpu)
        if pretrain:
            model_file = 'recognizer/save_weights/seq2seq-72.model_5.79.bak'
            print('Loading RecModel', model_file)
            self.seq2seq.load_state_dict(torch.load(model_file))

    def forward(self, img, label, img_width):
        self.seq2seq.train()
        img = torch.cat([img,img,img], dim=1) # b,1,64,128->b,3,64,128
        output, attn_weights = self.seq2seq(img, label, img_width, teacher_rate=False, train=False, beam_size=3)
        return output.permute(1, 0, 2) # t,b,83->b,t,83
    
def normalize(tar):
    tar = (tar - tar.min())/(tar.max()-tar.min())
    tar = tar * 255
    tar = tar.astype(np.uint8)
    return tar

def fine(label_list):
    if type(label_list) != type([]):
        return [label_list]
    else:
        return label_list
    
def write_image(xg, pred_label, gt_img, gt_label, tr_imgs, xg_swap, pred_label_swap, gt_label_swap, title, num_tr=2):
    folder = '/home/woody/iwi5/iwi5333h/img4'
    if not os.path.exists(folder):
        os.makedirs(folder)
    batch_size = gt_label.shape[0]
    tr_imgs = tr_imgs.cpu().numpy()
    xg = xg.cpu().numpy()
    xg_swap = xg_swap.cpu().numpy()
    gt_img = gt_img.cpu().numpy()
    gt_label = gt_label.cpu().numpy()
    gt_label_swap = gt_label_swap.cpu().numpy()
    pred_label = torch.topk(pred_label, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label = pred_label.cpu().numpy()
    pred_label_swap = torch.topk(pred_label_swap, 1, dim=-1)[1].squeeze(-1) # b,t,83 -> b,t,1 -> b,t
    pred_label_swap = pred_label_swap.cpu().numpy()
    tr_imgs = tr_imgs[:, :num_tr, :, :]
    outs = list()
    for i in range(batch_size):
        src = tr_imgs[i].reshape(num_tr*IMG_HEIGHT, -1)
        gt = gt_img[i].squeeze()
        tar = xg[i].squeeze()
        tar_swap = xg_swap[i].squeeze()
        src = normalize(src)
        gt = normalize(gt)
        tar = normalize(tar)
        tar_swap = normalize(tar_swap)
        gt_text = gt_label[i].tolist()
        gt_text_swap = gt_label_swap[i].tolist()
        pred_text = pred_label[i].tolist()
        pred_text_swap = pred_label_swap[i].tolist()

        gt_text = fine(gt_text)
        gt_text_swap = fine(gt_text_swap)
        pred_text = fine(pred_text)
        pred_text_swap = fine(pred_text_swap)

        for j in range(num_tokens):
            gt_text = list(filter(lambda x: x!=j, gt_text))
            gt_text_swap = list(filter(lambda x: x!=j, gt_text_swap))
            pred_text = list(filter(lambda x: x!=j, pred_text))
            pred_text_swap = list(filter(lambda x: x!=j, pred_text_swap))


        gt_text = ''.join([index2letter[c-num_tokens] for c in gt_text])
        gt_text_swap = ''.join([index2letter[c-num_tokens] for c in gt_text_swap])
        pred_text = ''.join([index2letter[c-num_tokens] for c in pred_text])
        pred_text_swap = ''.join([index2letter[c-num_tokens] for c in pred_text_swap])
        gt_text_img = np.zeros_like(tar)
        gt_text_img_swap = np.zeros_like(tar)
        pred_text_img = np.zeros_like(tar)
        pred_text_img_swap = np.zeros_like(tar)
        cv2.putText(gt_text_img, gt_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(gt_text_img_swap, gt_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img, pred_text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(pred_text_img_swap, pred_text_swap, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        out = np.vstack([src, gt, gt_text_img, tar, pred_text_img, gt_text_img_swap, tar_swap, pred_text_img_swap])
        outs.append(out)
    final_out = np.hstack(outs)
    cv2.imwrite(folder+'/'+title+'.png', final_out)
    
    

def load_cfg(path: str) -> dict:
    """
    Expects YAML like:
      generator:
        image_encoder: vgg  # or resnet/efficientnet/dino/inception/stylecnn
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)



if __name__ == "__main__":
    cfg_path = os.environ.get("CFG", "config.yaml")
    cfg = load_cfg(cfg_path)
    encoder_name = cfg["generator"]["image_encoder"]

    gen = GenModel_FC(text_max_len=OUTPUT_MAX_LEN, encoder_name=encoder_name).to(gpu)
    print(f"[GenModel_FC] Using image encoder: {gen.enc_image.__class__.__name__}")
    print(f"[GenModel_FC] encoder class: {gen.enc_image.__class__.__name__}")
    print(f"[GenModel_FC] encoder module: {gen.enc_image.__class__.__module__}")
    print(f"[GenModel_FC] encoder defined in: {inspect.getfile(gen.enc_image.__class__)}")

    dis = DisModel().to(gpu)

    B, C, H, W = 2, 50, IMG_HEIGHT, IMG_WIDTH  
    dummy_img = torch.randn(B, C, H, W, device=gpu)
    feats = gen.forward_image_features(dummy_img)         
    text_tokens = torch.randint(low=0, high=vocab_size, size=(B, OUTPUT_MAX_LEN), device=gpu)
    _, text_grid = gen.forward_text_features(text_tokens, feats[-1].shape)
    mixed = gen.mix(feats, text_grid)                

    
    adain_params = torch.randn(B, 512 * 2, device=gpu)   
    out_imgs = gen.decode(mixed, feats, embed=_, adain_params=adain_params)  

    print(f"[OK] Enc feats: {[tuple(f.shape) for f in feats]}")
    print(f"[OK] Mixed: {tuple(mixed.shape)}, Out: {tuple(out_imgs.shape)}")
