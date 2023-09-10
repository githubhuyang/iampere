import os
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, download_url
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import shutil
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

from meta import Meta
from nn import PiApproximationWithNN


class MyLoss(nn.Module):
    def __init__(self, weight):
      super().__init__()
      self.bce = nn.BCELoss(weight=weight)

    def forward(self, y_pred, y_true):
        return self.bce(y_pred, y_true)


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        l = list(os.listdir(self.processed_dir))
        l = sorted(l)
        return l

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return data


def train_pi(task_set_name, epoch_num=100, load_model=False, is_cpu=False):
    if not os.path.isdir(f"{Meta.data_dir}/pt_model"):
        os.makedirs(f"{Meta.data_dir}/pt_model")

    # data
    dataset = MyOwnDataset(f"{Meta.data_dir}/pyg/{task_set_name}")

    ts = int(len(dataset) * 4 // 5)
    train_set = dataset[:ts]
    valid_set = dataset[ts:]

    # model
    pi = PiApproximationWithNN(lr=Meta.lr)
    if load_model:
        pi.load(f"{Meta.data_dir}/pt_model/pi-best.pt")

    if is_cpu:
        pi.gnn.cpu()
    
    train_loss_lst = []

    ps_diff_train_lst = []
    ps_diff_valid_lst = []
    prec_train_lst = []
    prec_valid_lst = []
    recall_train_lst = []
    recall_valid_lst = []

    epoch_lst = list(range(epoch_num))
    best_f1 = 0
    
    for epoch in epoch_lst:
        pi.gnn.train()
        y_true = []
        y_pred = []

        loss_value = 0
        patch_size_diff = 0

        edge_cnt = 0
        with tqdm(total=len(train_set), desc="itr", leave=False) as pbar_itr:
            for data in train_set:
                
                if not is_cpu:
                    data = data.cuda()

                tp_cnt = torch.sum(data.y.detach())
                tn_cnt = data.y.shape[0] - tp_cnt
                weight = torch.clone(data.y.view(-1, 1)).detach()
                weight[weight == 1] = 2 * tn_cnt / tp_cnt # 2 * 
                weight[weight == 0] = 1
                my_loss = MyLoss(weight=weight) # nn.BCELoss(weight=weight) # nn.L1Loss() # nn.MSELoss()

                pi.optimizer.zero_grad()
                out = pi.gnn(data.x, data.edge_index, data.edge_attr)[:data.y.shape[0]]
                
                loss = my_loss(out, data.y.view(-1, 1))
                loss.backward()
                pi.optimizer.step()

                with torch.no_grad():
                    loss_value += data.y.shape[0] * loss.item()
                    edge_cnt += data.y.shape[0]

                    y_true += data.y.cpu().numpy().tolist()
                    y_pred += (out.flatten() > 0.5).int().cpu().numpy().tolist()

                pbar_itr.update()

        train_ps_diff = min(sum(y_pred), sum(y_true)) / max(sum(y_pred), sum(y_true)) # len(y_true)
        train_prec = precision_score(y_true, y_pred, average='binary', pos_label=1)
        train_recall = recall_score(y_true, y_pred, average='binary', pos_label=1)

        ps_diff_train_lst.append(train_ps_diff)
        recall_train_lst.append(train_recall)
        prec_train_lst.append(train_prec)

        train_loss = loss_value / edge_cnt
        train_loss_lst.append(train_loss)

        pi.gnn.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in valid_set:
                if not is_cpu:
                    data = data.cuda()

                pred = pi.gnn(data.x, data.edge_index, data.edge_attr)[:data.y.shape[0]]
                
                y_true += data.y.cpu().numpy().tolist()
                y_pred += (pred.flatten() > 0.5).int().cpu().numpy().tolist()

        valid_ps_diff = min(sum(y_pred), sum(y_true)) / max(sum(y_pred), sum(y_true)) # / len(y_true)
        valid_prec = precision_score(y_true, y_pred, average='binary', pos_label=1)
        valid_recall = recall_score(y_true, y_pred, average='binary', pos_label=1)
        
        if valid_prec > 0 or valid_recall > 0:
            valid_f1 = 2 * valid_prec * valid_recall / (valid_prec + valid_recall)

            ps_diff_valid_lst.append(valid_ps_diff)
            prec_valid_lst.append(valid_prec)
            recall_valid_lst.append(valid_recall)

            if best_f1 < valid_f1:
                best_f1 = valid_f1
                pi.save(f"{Meta.data_dir}/pt_model/pi-best.pt")

        print(f"epoch={epoch}")
        print(f"train_loss={train_loss}")
        print(f"train_ps_sim={train_ps_diff}, valid_ps_sim={valid_ps_diff}")
        print(f"train_prec={train_prec}, valid_prec={valid_prec}")
        print(f"train_recall={train_recall}, valid_recall={valid_recall}")
    
    df = pd.DataFrame({'epoch': epoch_lst,
                'loss-train': train_loss_lst,
                'ps-sim-train': ps_diff_train_lst,
                'ps-sim-valid': ps_diff_valid_lst,
                'prec-train':   prec_train_lst,
                'prec-valid':   prec_valid_lst,
                'recall-train': recall_train_lst,
                'recall-valid': recall_valid_lst})
    df.to_csv(f'{Meta.csv_dir}/{task_set_name}_gnn-train.csv', index=False)


if __name__ == '__main__':
    train_pi(Meta.train_task_set_name, epoch_num=Meta.epoch_num, load_model=False, is_cpu=False)