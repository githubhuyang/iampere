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
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
import time

from meta import Meta
from nn import PiApproximationWithNN
from train import MyOwnDataset
from interact_env import InteractEnv


def predict_parallel(task_set_name):
    cp_dir_path = f"{Meta.data_dir}/itm_patches/{task_set_name}"
    if not os.path.isdir(cp_dir_path):
        os.makedirs(cp_dir_path)

    arg_lst_lst = []
    dataset = MyOwnDataset(f"{Meta.data_dir}/pyg/{task_set_name}")
    for i, data in enumerate(dataset):
        pt_file_name = dataset.processed_file_names[i]
        task_idx = int(pt_file_name.split(".")[0].split("-")[-1])

        cp_dir_path = f"{Meta.data_dir}/itm_patches/{task_set_name}"
        if os.path.isfile(f"{cp_dir_path}/cp-{task_idx}.pkl"):
            continue
        
        arg_lst_lst.append([task_set_name, task_idx, pt_file_name])

    task_idx_lst = []
    hw_lst = []
    tc_lst = []
    ps_lst = []
    max_patch_cnt = 0
    with tqdm(total=len(arg_lst_lst), desc="task", leave=False) as pbar:
        with Pool(Meta.n_cpu) as p:
            for i, (task_idx, hw, tc, ps, is_max_patch) in enumerate(p.imap_unordered(predict_single_cpu, arg_lst_lst)):
                task_idx_lst.append(task_idx)
                hw_lst.append(hw)
                tc_lst.append(tc)
                ps_lst.append(ps)

                if is_max_patch:
                    max_patch_cnt += 1

                pbar.update(1)

    # print(f"# max patch: {max_patch_cnt}")

    df = pd.DataFrame({'task': task_idx_lst,
                'hw': hw_lst,
                'ps': ps_lst,
                'tc': tc_lst})
    df = df.sort_values(by=['task'])

    df.to_csv(f'{Meta.csv_dir}/{task_set_name}_gnn-predict.csv', index=False)

    print("Done!")


def predict_single_cpu(arg_lst, timelim=300):
    task_set_name, task_idx, pt_file_name = arg_lst[0], arg_lst[1], arg_lst[2]

    pi = PiApproximationWithNN(lr=Meta.lr, cuda=False)
    pi.load_cpu(f"{Meta.data_dir}/pt_model/pi-best.pt")
    pi.gnn = pi.gnn.cpu()
    pi.gnn.eval()

    data = torch.load(f"{Meta.data_dir}/pyg/{task_set_name}/processed/{pt_file_name}")
    
    start_time = time.time()

    with torch.no_grad():
        pred = pi.gnn(data.x, data.edge_index, data.edge_attr)
    pred = pred.flatten().numpy().tolist()

    env = InteractEnv(task_set_name, task_idx)
    env.reset()

    action_cnt = 0

    for f in env.W:
        if env.W[f]:
            action_cnt += 1

    for e in env.A:
        action_cnt += len(env.A[e])

    edge_idx_lst = [j for j in range(action_cnt)] # len(pred)
    edge_idx_lst = sorted(edge_idx_lst, key=lambda j: pred[j], reverse=True)
    
    final_action_patch = None
    init_patch = [j for j in range(action_cnt) if pred[j] > 0.5]
    if len(init_patch) == len(edge_idx_lst):
        final_action_patch = []
    else:
        patch_lst = []
        qt = int(len(edge_idx_lst) // 10)
                    
        patch_lst.append(init_patch)
        patch_lst.append(edge_idx_lst[:qt])
        patch_lst.append(edge_idx_lst[:2 * qt])
        patch_lst.append(edge_idx_lst[:4 * qt])
        patch_lst.append(edge_idx_lst[:8 * qt])
        patch_lst.append(edge_idx_lst)

        patch_lst = sorted(patch_lst, key = lambda patch: len(patch))

        for patch in patch_lst:
            if len(patch) == len(edge_idx_lst):
                final_action_patch = []
                break

            if time.time() - start_time > timelim:
                final_action_patch = []
                break

            env = InteractEnv(task_set_name, task_idx)
            action_space = env.reset() 

            terminated = False
            final_action_patch = []
            while not terminated:
                if time.time() - start_time > timelim:
                    break

                selected_edge_idx_lst = [j for j in patch if j in action_space]
                if len(selected_edge_idx_lst) == 0:
                    break

                selected_edge_idx_lst = sorted(selected_edge_idx_lst, key=lambda j: pred[j], reverse=True)

                selected_action_lst = [action_space[edge_idx] for edge_idx in selected_edge_idx_lst]
                
                action_space, terminated, effect_action_cnt = env.step(selected_action_lst)

                final_action_patch += selected_action_lst[:effect_action_cnt]

            if terminated:
                break

            if time.time() - start_time > timelim:
                final_action_patch = []
                break

    ps = None
    is_max_patch = False
    if len(final_action_patch) == 0:
        ps = -1
        is_max_patch = True
    else:
        ps = len(final_action_patch)

    cp_dir_path = f"{Meta.data_dir}/itm_patches/{task_set_name}"

    if len(final_action_patch) > 0:
        with open(f"{cp_dir_path}/cp-{task_idx}.pkl", "wb") as pickle_out:
            pickle.dump(final_action_patch, pickle_out)

    tc = time.time() - start_time
    if tc > timelim:
        tc = timelim

    return task_idx, "cpu", tc, ps, is_max_patch


if __name__ == '__main__':
    predict_parallel(Meta.test_task_set_name)