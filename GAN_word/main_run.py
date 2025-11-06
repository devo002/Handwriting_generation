import os
from datetime import datetime
import torch
import glob
from torch import optim
import numpy as np
import time
import argparse
import logging
from load_data import NUM_WRITERS
from network_tro import ConTranModel
from load_data import loadData as load_data_func
from loss_tro import CER
import torch.backends.cudnn as cudnn
import optuna

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
parser.add_argument('--optuna', action='store_true', help='Run Optuna hyperparameter tuning')
args = parser.parse_args()

gpu = torch.device('cuda')

# Hyperparams and Configs
OOV = True
NUM_THREAD = 2
EARLY_STOP_EPOCH = 20
EVAL_EPOCH = 20
MODEL_SAVE_EPOCH = 100
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 8
lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec =1 * 1e-5
lr_cla = 1 * 1e-5

CurriculumModelID = args.start_epoch
model_name = 'aaa'
run_id = datetime.strftime(datetime.now(), '%m-%d-%H-%M')
base_logdir = '/home/woody/iwi5/iwi5333h'
logdir = os.path.join(base_logdir, 'log4', model_name + '-' + str(run_id))
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter(logdir)

def log(msg):
    logger = logging.getLogger("GanWriting")
    handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.info(msg)
    logger.removeHandler(handler)
    logger.removeHandler(console)

def all_data_loader():
    data_train, data_test = load_data_func(OOV)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_THREAD, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_THREAD, pin_memory=True)
    return train_loader, test_loader

def sort_batch(batch):
    train_domain, train_wid, train_idx = [], [], []
    train_img, train_img_width, train_label = [], [], []
    img_xts, label_xts, label_xts_swap = [], [], []

    for domain, wid, idx, img, img_width, label, img_xt, label_xt, label_xt_swap in batch:
        if wid >= NUM_WRITERS:
            print('error!')
        train_domain.append(domain)
        train_wid.append(wid)
        train_idx.append(idx)
        train_img.append(img)
        train_img_width.append(img_width)
        train_label.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    return (
        np.array(train_domain),
        torch.tensor(train_wid, dtype=torch.int64),
        np.array(train_idx),
        torch.tensor(np.array(train_img), dtype=torch.float32),
        torch.tensor(np.array(train_img_width), dtype=torch.int64),
        torch.tensor(np.array(train_label), dtype=torch.int64),
        torch.tensor(np.array(img_xts), dtype=torch.float32),
        torch.tensor(np.array(label_xts), dtype=torch.int64),
        torch.tensor(np.array(label_xts_swap), dtype=torch.int64)
        
        # torch.tensor(train_img, dtype=torch.float32),
        # torch.tensor(train_img_width, dtype=torch.int64),
        # torch.tensor(train_label, dtype=torch.int64),
        # torch.tensor(img_xts, dtype=torch.float32),
        # torch.tensor(label_xts, dtype=torch.int64),
        # torch.tensor(label_xts_swap, dtype=torch.int64)
    )

def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):
    model.train()
    loss_dis, loss_dis_tr = [], []
    loss_cla, loss_cla_tr = [], []
    loss_l1, loss_rec, loss_rec_tr = [], [], []

    time_s = time.time()
    cer_tr = CER()
    cer_te = CER()
    cer_te2 = CER()

    for train_data_list in train_loader:
        rec_opt.zero_grad()
        l_rec_tr = model(train_data_list, epoch, 'rec_update', cer_tr)
        rec_opt.step()

        cla_opt.zero_grad()
        l_cla_tr = model(train_data_list, epoch, 'cla_update')
        cla_opt.step()

        dis_opt.zero_grad()
        l_dis_tr = model(train_data_list, epoch, 'dis_update')
        dis_opt.step()

        gen_opt.zero_grad()
        l_total, l_dis, l_cla, l_l1, l_rec = model(train_data_list, epoch, 'gen_update', [cer_te, cer_te2])
        gen_opt.step()

        loss_dis.append(l_dis.cpu().item())
        loss_dis_tr.append(l_dis_tr.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_cla_tr.append(l_cla_tr.cpu().item())
        loss_l1.append(l_l1.cpu().item())
        loss_rec.append(l_rec.cpu().item())
        loss_rec_tr.append(l_rec_tr.cpu().item())

    res_cer_tr = cer_tr.fin()
    res_cer_te = cer_te.fin()
    res_cer_te2 = cer_te2.fin()

    writer.add_scalars("train", {
        "fl_dis_tr": np.mean(loss_dis_tr), "fl_dis": np.mean(loss_dis),
        "fl_cla_tr": np.mean(loss_cla_tr), "fl_cla": np.mean(loss_cla),
        "fl_rec_tr": np.mean(loss_rec_tr), "fl_rec": np.mean(loss_rec),
        "fl_l1": np.mean(loss_l1),
        "res_cer_tr": res_cer_tr, "res_cer_te": res_cer_te, "res_cer_te2": res_cer_te2
    }, epoch)

    log(('epo%d <tr>-<gen>: l_dis=%.2f-%.2f, l_cla=%.2f-%.2f, l_rec=%.2f-%.2f, l1=%.2f, cer=%.2f-%.2f-%.2f, time=%.1f' %
         (epoch, np.mean(loss_dis_tr), np.mean(loss_dis), np.mean(loss_cla_tr), np.mean(loss_cla),
          np.mean(loss_rec_tr), np.mean(loss_rec), np.mean(loss_l1),
          res_cer_tr, res_cer_te, res_cer_te2, time.time() - time_s)))
    
    return res_cer_te + res_cer_te2

def test(test_loader, epoch, modelFile_o_model):
    if isinstance(modelFile_o_model, str):
        model = ConTranModel(NUM_WRITERS, show_iter_num, OOV).to(gpu)
        print('Loading ' + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model))
    else:
        model = modelFile_o_model
    model.eval()

    loss_dis, loss_cla, loss_rec = [], [], []
    time_s = time.time()
    cer_te = CER()
    cer_te2 = CER()

    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, 'eval', [cer_te, cer_te2])
        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())

    test_cer = cer_te.fin() + cer_te2.fin()

    writer.add_scalars("EVAL", {
        "fl_dis": np.mean(loss_dis), "fl_cla": np.mean(loss_cla),
        "fl_rec": np.mean(loss_rec),
        "res_cer_te": cer_te.fin(), "res_cer_te2": cer_te2.fin()
    }, epoch)

    log(('EVAL: l_dis=%.3f, l_cla=%.3f, l_rec=%.3f, cer=%.2f-%.2f, time=%.1f' %
         (np.mean(loss_dis), np.mean(loss_cla), np.mean(loss_rec),
          cer_te.fin(), cer_te2.fin(), time.time() - time_s)))
    
    return test_cer    


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        #self.val_loss_min = np.Inf
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss        
        

def main(train_loader, test_loader, num_writers):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)
    folder_weights = '/home/vault/iwi5/iwi5333h/save_weights4'
    os.makedirs(folder_weights, exist_ok=True)

    if CurriculumModelID > 0:
        model_file = os.path.join(folder_weights, f'contran-{CurriculumModelID}.model')
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file))

    dis_opt = optim.Adam(filter(lambda p: p.requires_grad, model.dis.parameters()), lr=lr_dis)
    gen_opt = optim.Adam(filter(lambda p: p.requires_grad, model.gen.parameters()), lr=lr_gen)
    rec_opt = optim.Adam(filter(lambda p: p.requires_grad, model.rec.parameters()), lr=lr_rec)
    cla_opt = optim.Adam(filter(lambda p: p.requires_grad, model.cla.parameters()), lr=lr_cla)

    early_stopping = EarlyStopping(patience=EARLY_STOP_EPOCH if EARLY_STOP_EPOCH else 20, verbose=True)

    for epoch in range(CurriculumModelID, 50001):
        if epoch > 4000:
            global MODEL_SAVE_EPOCH
            MODEL_SAVE_EPOCH = 20

        if epoch % 20 == 0 and epoch != 0:
            train_loader, test_loader = all_data_loader()

        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % EVAL_EPOCH == 0:
            test_cer = test(test_loader, epoch, model)
            early_stopping(test_cer)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                model_path = os.path.join(folder_weights, f'contran-best.model')
                torch.save(model.state_dict(), model_path)
                break

        if epoch % MODEL_SAVE_EPOCH == 0:
            model_path = os.path.join(folder_weights, f'contran-{epoch}.model')
            torch.save(model.state_dict(), model_path)
            
def train_with_custom_lr(train_loader, test_loader, num_writers, lr_dis, lr_gen, lr_rec, lr_cla):
    model = ConTranModel(num_writers, show_iter_num, OOV).to(gpu)

    dis_opt = optim.Adam(filter(lambda p: p.requires_grad, model.dis.parameters()), lr=lr_dis)
    gen_opt = optim.Adam(filter(lambda p: p.requires_grad, model.gen.parameters()), lr=lr_gen)
    rec_opt = optim.Adam(filter(lambda p: p.requires_grad, model.rec.parameters()), lr=lr_rec)
    cla_opt = optim.Adam(filter(lambda p: p.requires_grad, model.cla.parameters()), lr=lr_cla)

    for epoch in range(160):  # Fixed 100 epochs
        train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % 20 == 0 and epoch != 0:
            val_cer = test(test_loader, epoch, model)
            print(f"[Trial Eval @Epoch {epoch}] CER: {val_cer:.2f}")

    final_cer = test(test_loader, 160, model)
    return final_cer

def optuna_objective(trial):
 
    lr_gen = trial.suggest_loguniform("lr_gen", 1e-5, 1e-3)
    lr_rec = trial.suggest_loguniform("lr_rec", 1e-6, 1e-4)

   
    lr_dis = 1e-4
    lr_cla = 1e-5

    train_loader, test_loader = all_data_loader()

    final_cer = train_with_custom_lr(train_loader, test_loader, NUM_WRITERS,
                                     lr_dis, lr_gen, lr_rec, lr_cla)

    return final_cer


def rm_old_model(index, folder_weights):
    models = glob.glob(os.path.join(folder_weights, '*.model'))
    for m in models:
        epoch = int(m.split('-')[-1].split('.')[0])
        if epoch < index:
            os.remove(m)

if __name__ == '__main__':
    print(time.ctime())
    train_loader, test_loader = all_data_loader()
    
    if args.optuna:
        storage_path = "sqlite:///optuna_effnetv2llr_study.db"
        study = optuna.create_study(study_name="gen_rec_lr_tuning",
                                storage=storage_path,
                                direction="minimize",
                                load_if_exists=True)
        study.optimize(optuna_objective, n_trials=20)

        print("Best trial:")
        print(f"  CER: {study.best_trial.value}")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        main(train_loader, test_loader, NUM_WRITERS)

    print(time.ctime())
    #main(train_loader, test_loader, NUM_WRITERS)
    #print(time.ctime())
