
import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
from modules_tro import normalize
import os
import time
from tqdm import tqdm, trange
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

'''Take turns to open the comments below to run 4 scenario experiments'''

folder_wids = '/home/woody/iwi5/iwi5333h/data'
# img_base = '/home/WeiHongxi/WangHeng/project/dataset/Iam_database/words/'
img_base = '/home/woody/iwi5/iwi5333h/data'
folder_pre = '/home/woody/iwi5/iwi5333h/inception'
# folder_pre = 'test_single_writer.4_scenarios_average/'
#epoch = 5000
epoch = 3500

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--epoch', default=epoch, type=int,
                    help=('Epoch/model checkpoint number, used to pick model path '
                          'like contran-<epoch>.model'))
parser.add_argument(
    '--writers',
    nargs='+',
    default=None,
    help="One or more writer IDs to process (e.g., a01 a02). If omitted, all writers in the target file are processed."
)

'''data preparation'''
def pre_data(data_dict, target_file):
    with open(target_file, 'r') as _f:
        data = _f.readlines()
        lables = [i.split(' ')[1].replace('\n', '').replace('\r', '') for i in data]
        data = [i.split(' ')[0] for i in data]
        wids = [i.split(',')[0] for i in data]
        imgnames = [i.split(',')[1] for i in data]

    for wid, imgname, lable in zip(wids, imgnames, lables):
        index = []
        index.append(imgname)
        index.append(lable)
        if wid in data_dict.keys():
            data_dict[wid].append(index)
        else:
            data_dict[wid] = [index]

    '''Try on different datasets'''
    # folder = 'res_img_gw'
    # img_base = '/home/lkang/datasets/WashingtonDataset_words/words/'
    # target_file = 'gw_total_mas50.gt.azAZ'

    # folder = 'res_img_parzival'
    # img_base = '/home/lkang/datasets/ParzivalDataset_German/data/word_images_normalized/'
    # target_file = 'parzival_mas50.gt.azAZ'

    # folder = 'res_img_esp'
    # img_base = '/home/lkang/datasets/EsposallesOfficial/words_lines.official.old/'
    # target_file = 'esposalles_total.gt.azAZ'

    return data_dict

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_writer(wid, model_file, folder, text_corpus, data_dict):
    def read_image(file_name, thresh=None):
        subfolder = file_name.split('-')[0]  # gets 'a01'
        parent = '-'.join(file_name.split('-')[:2])  # gets 'a01-000u'
        url = os.path.join(img_base, subfolder, parent, file_name + '.png')

        if not os.path.exists(url):
            print(f"⚠️ Image not found: {url}")
            return None

        img = cv2.imread(url, 0)
        if img is None:
            print(f"⚠️ Failed to read image (cv2.imread returned None): {url}")
            return None

        if thresh:
            # img[img>thresh] = 255
            pass

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
        img = img / 255.  # 0-255 -> 0-1

        img = 1. - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
            outImg[:, :img_width] = img
        outImg = outImg.astype('float32')

        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal

    def label_padding(labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        new_label_len.append(len(ll) + 2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        num = OUTPUT_MAX_LEN - len(ll)
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num)  # replace PAD_TOKEN
        return ll

    '''data preparation'''
    # collect images for this writer, skipping missing files
    raw_items = data_dict[wid]
    imgs_list = []
    for item in raw_items:
        im = read_image(item[0])
        if im is not None:
            imgs_list.append(im)

    if len(imgs_list) == 0:
        print(f"⚠️ No valid images for writer {wid}, skipping.")
        return

    random.shuffle(imgs_list)
    final_imgs = imgs_list[:50]
    if len(final_imgs) < 50:
        while len(final_imgs) < 50 and len(imgs_list) > 0:
            num_cp = min(50 - len(final_imgs), len(imgs_list))
            final_imgs = final_imgs + imgs_list[:num_cp]

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu)  # 1,50,64,216

    with open(text_corpus, 'r') as _f:
        texts = _f.read().split()
    labels = torch.from_numpy(np.array([np.array(label_padding(label, num_tokens)) for label in texts])).to(gpu)

    '''model loading'''
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    model.load_state_dict(torch.load(model_file, map_location=gpu))  # load
    model.eval()
    num = 0
    with torch.no_grad():
        f_xss = model.gen.enc_image(imgs)
        f_xs = f_xss[-1]
        for label in labels:
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xss, f_embed)
            xg = model.gen.decode(f_mix, f_xss, f_embed, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            label = label.squeeze().cpu().numpy().tolist()
            pred = torch.topk(pred, 1, dim=-1)[1].squeeze()
            pred = pred.cpu().numpy().tolist()
            for j in range(num_tokens):
                label = list(filter(lambda x: x != j, label))
                pred = list(filter(lambda x: x != j, pred))
            label = ''.join([index2letter[c - num_tokens] for c in label])
            pred = ''.join([index2letter[c - num_tokens] for c in pred])
            ed_value = Lev.distance(pred, label)
            if ed_value <= 100:
                num += 1
                xg_np = xg.cpu().numpy().squeeze()
                xg_np = normalize(xg_np)
                xg_np = 255 - xg_np
                img_folder = folder
                if not os.path.exists(img_folder):
                    os.makedirs(img_folder, exist_ok=True)
                out_path = os.path.join(img_folder, f'{wid}-{num}.{label}-{pred}.png')
                ret = cv2.imwrite(out_path, xg_np)
                if not ret:
                    import pdb; pdb.set_trace()

if __name__ == '__main__':
    args = parser.parse_args()
    model_epoch = str(args.epoch)

    for i in range(1):
        if i == 0:
            folder = folder_pre + model_epoch + '/res_4.oo_vocab_te_writer'
            target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/Groundtruth/gan.iam.tr_va.gt.filter27'
            #text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/corpora_english/copy2.57'
            text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/corpora_english/in_vocab.subset.tro.37'

        # elif i == 1:
        #     folder = folder_pre + model_epoch + '/res_1.in_vocab_tr_writer'
        #     target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/Groundtruth/gan.iam.tr_va.gt.filter27'
        #     text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/corpora_english/in_vocab.subset.tro.37'

        # elif i == 2:
        #     folder = folder_pre + model_epoch + '/res_2.in_vocab_te_writer'
        #     target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/Groundtruth/gan.iam.test.gt.filter27'
        #     text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/corpora_english/in_vocab.subset.tro.37'

        # elif i == 3:
        #     folder = folder_pre + model_epoch + '/res_3.oo_vocab_tr_writer'
        #     target_file = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/Groundtruth/gan.iam.tr_va.gt.filter27'
        #     text_corpus = '/home/woody/iwi5/iwi5333h/AFFGanWriting/GAN_word/corpora_english/oov.common_words'

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(folder)

        data_dict = dict()
        data_dict = pre_data(data_dict, target_file)

        # Build the set of all writers from the target file
        with open(target_file, 'r') as _f:
            data = _f.readlines()
        all_wids = list(set([i.split(',')[0] for i in data]))

        # If --writers provided, filter; otherwise process all
        if args.writers:
            requested = set(args.writers)
            available = set(all_wids)
            missing = requested - available
            if missing:
                print(f"⚠️ Requested writer(s) not found in {target_file}: {sorted(missing)}")
            wids = [w for w in all_wids if w in requested]
        else:
            wids = all_wids

        # Iterate
        wids = tqdm(wids)
        for wid in wids:
            model_path = f'/home/vault/iwi5/iwi5333h/save_weights4/contran-{model_epoch}.model'
            test_writer(wid, model_path, folder, text_corpus, data_dict)