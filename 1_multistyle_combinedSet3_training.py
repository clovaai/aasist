import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list_prevDbs, genSpoof_list,
                        Dataset_train_prevDbs, Dataset_PrevDbs, Dataset_cs2, Dataset_train_cs2)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import numpy as np
from evaluation import compute_eer

warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    config = {
        "database_path": '/data/nfsshare/rishith/datasets/smaller_dbs/restructured/combinedSet3/',
    #     "asv_score_path": "ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
        "model_path": "./models/weights/AASIST-L.pth",
        "batch_size": 24,
        "num_epochs": 100,
        "loss": "CCE",
        "track": "LA",
        "eval_all_best": "True",
        "eval_output": "eval_scores_cs3_cs3.txt",
        "cudnn_deterministic_toggle": "True",
        "cudnn_benchmark_toggle": "False",
        "model_config": {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": 128,
            "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
            "gat_dims": [24, 32],
            "pool_ratios": [0.4, 0.5, 0.7, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0]
        },
        "optim_config": {
            "optimizer": "adam", 
            "amsgrad": "False",
            "base_lr": 0.0001,
            "lr_min": 0.000005,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0001,
            "scheduler": "cosine"
        }
    }


    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]


    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(1234, config)

    output_dir = Path('./exp_result')
    database_path = Path(config["database_path"])
    dev_trial_path = Path('./protocols/combinedSet3_dev.txt')

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        'CS3',
        'AASIST-L',
        config["num_epochs"], config["batch_size"])
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    """Make PyTorch DataLoaders for train / developement / evaluation"""

    trn_database_path = database_path 
    dev_database_path = database_path 

    trn_list_path = Path('./protocols/combinedSet3_train.txt')
    dev_trial_path = './protocols/combinedSet3_dev.txt'
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_train_cs2(list_IDs=file_train,
                                        labels=d_label_trn,
                                        base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(1234)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_cs2(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    eval_trial_path = Path('/data/nfsshare/rishith/datasets/asvSpoof2019/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
    eval_database_path = Path('/data/nfsshare/rishith/datasets/asvSpoof2019/DS_10283_3336/LA/ASVspoof2019_LA_eval/')

    file_eval = genSpoof_list1(dir_meta=eval_trial_path,
                          is_train=False,
                          is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                            base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")


    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                scheduler, config)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer = getEER(metric_path/"dev_score.txt")
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
            running_loss, dev_eer))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                    model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                dev_eer = getEER(metric_path/"dev_score.txt")
                eval_eer = getEER(eval_score_path)

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(),
                            model_save_path / "best.pth")
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    if eval_tdcf <= best_eval_tdcf:
        best_eval_tdcf = eval_tdcf
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))


def getEER(score_path):
    cm_data = np.genfromtxt(score_path, dtype=str)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]
    return eer_cm * 100

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

def genSpoof_list1(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, _, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} \n".format(utt_id, key, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss



if __name__ == "__main__":
    main()