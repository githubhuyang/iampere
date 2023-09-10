from torch_geometric.data import Data
import torch
import os
import pickle
import psutil
from multiprocessing import Pool
from tqdm import tqdm

from meta import Meta
from maxsat import maxsat_repair_single
from utils import reduce_graph


def gen_pyg_data(task_set_name, for_training=True):
    data_obj_dir = f"{Meta.data_dir}/pyg/{task_set_name}/processed"
    if not os.path.isdir(data_obj_dir):
        os.makedirs(data_obj_dir)
        os.makedirs(f"{Meta.data_dir}/pyg/{task_set_name}/raw")

    args_lst = []
    for file_name in os.listdir(f"{Meta.data_dir}/tasks/{task_set_name}"):
        task_idx = int(file_name.split(".")[0].split("-")[-1])

        if not os.path.isfile(f"{data_obj_dir}/data-{task_idx}.pt"):
            args_lst.append((task_set_name, task_idx, for_training))

    with Pool(Meta.n_cpu) as p: # Meta.n_cpu
        with tqdm(total=len(args_lst)) as pbar:
            for _ in p.imap_unordered(gen_pyg_data_single, args_lst):
                pbar.update()


def gen_pyg_data_single(args):
    task_set_name, task_idx, for_training = args[0], args[1], args[2]
    
    # use complete MaxSAT
    repair = None
    if for_training:
        repair = maxsat_repair_single([task_set_name, task_idx, None, True], return_repair=True)
        
        if repair == None:
            print(f"fail to gen data.y for {task_set_name} {task_idx}")
            return

    repair_A = None
    repair_W = None
    if repair != None:
        repair_A = repair[2]
        repair_W = repair[3]

    with open(f"{Meta.data_dir}/tasks/{task_set_name}/pt-{task_idx}.pkl", "rb") as f:
        E, F, A, W, P, u_lst, l_lst = pickle.load(f)
    E, F, A, W, P, u_lst, l_lst = reduce_graph(E, F, A, W, P, u_lst, l_lst, save_enabled_flow_perm=True)

    X = []
    ## entity node
    e_to_idx = {}
    for i, e in enumerate(E):
        e_to_idx[e] = i

        feature = [1, 0] # entity node
        
        # compromised entity or not
        if e in u_lst:
            feature.append(1)
        else:
            feature.append(0)

        X.append(feature)

    if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
        print(f"run out of memory {task_set_name} {task_idx}")
        return

    p_to_idx = {}
    for perm_idx in range(len(P)):

        p_to_idx[perm_idx] = len(e_to_idx) + perm_idx

        feature = [0, 1] # perm node

        # target perm or not
        if perm_idx in l_lst:
            feature.append(1)
        else:
            feature.append(0)

        X.append(feature)

    if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
        print(f"run out of memory {task_set_name} {task_idx}")
        return

    X = torch.tensor(X, dtype=torch.float)

    # edges
    ## entity-entity edges
    edge_index = []
    edge_attr = []
    edge_y = []
    edge_idx_to_action = {}
    for (e1, e2) in F:
        e1_idx = e_to_idx[e1]
        e2_idx = e_to_idx[e2]

        if W[(e1, e2)]:
            edge_idx_to_action[len(edge_index)] = ("ee", e1, e2)
            edge_index.append([e1_idx, e2_idx])

            # edge feature
            feature = [1, 0, 0, 0, 0, 0]
            edge_attr.append(feature)

            if for_training:
                if W[(e1, e2)] and not repair_W[(e1, e2)]:
                    edge_y.append(1)
                else:
                    edge_y.append(0)

    if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
        print(f"run out of memory {task_set_name} {task_idx}")
        return

    ## perm-entity edges
    for e in E:
        for perm_idx in range(len(P)):
            node_idx_e = e_to_idx[e]
            node_idx_p = p_to_idx[perm_idx]

            if perm_idx in A[e]:
                can_remove = True
                for (e1, e2) in F:
                    if e2 == e and W[(e1, e2)] and perm_idx in A[e1]:
                        can_remove = False
                        break

                if can_remove:
                    edge_idx_to_action[len(edge_index)] = ("ep", e, perm_idx)

                edge_index.append([node_idx_p, node_idx_e])

                # edge feature
                feature = [0, 1, 0, 0, 0, 0]
                edge_attr.append(feature)

                if for_training:
                    if perm_idx in A[e] and perm_idx not in repair_A[e]:
                        edge_y.append(1)
                    else:
                        edge_y.append(0)

    if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
        print(f"run out of memory {task_set_name} {task_idx}")
        return

    ## entity-perm-entity edges
    for perm_idx in range(len(P)):
        node_idx_p = p_to_idx[perm_idx]

        perm = P[perm_idx]
        if perm[0] == "type-i":
            e1 = perm[1]
            e2 = perm[2]

            node_idx_e1 = e_to_idx[e1]
            node_idx_e2 = e_to_idx[e2]

            feature = [0, 0, 1, 0, 0, 0] # source
            edge_index.append([node_idx_e1, node_idx_p])
            edge_attr.append(feature)

            feature = [0, 0, 0, 1, 0, 0] # sink
            edge_index.append([node_idx_e2, node_idx_p])
            edge_attr.append(feature)
        elif perm[0] == "type-ii":
            feature = [0, 0, 0, 0, 1, 0] # perm to perm
            for perm_idx_b in range(len(P)):
                if perm_idx_b != perm_idx:
                    node_idx_p_b = p_to_idx[perm_idx_b]

                    edge_index.append([node_idx_p_b, node_idx_p])
                    edge_attr.append(feature)

            e = perm[1]
            feature = [0, 0, 0, 0, 0, 1] # perm to entity
            node_idx_e = e_to_idx[e]

            edge_index.append([node_idx_e, node_idx_p])
            edge_attr.append(feature)

    if psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
        print(f"run out of memory {task_set_name} {task_idx}")
        return

    assert(len(edge_index) == len(edge_attr))

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = None
    if for_training:
        edge_y = torch.tensor(edge_y, dtype=torch.float)
        data = Data(x=X, y=edge_y, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)
    else:
        data = Data(x=X, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr)

    data_obj_dir = f"{Meta.data_dir}/pyg/{task_set_name}/processed"
    torch.save(data, f"{data_obj_dir}/data-{task_idx}.pt")


if __name__ == '__main__':
    gen_pyg_data(Meta.train_task_set_name, for_training=True)
    gen_pyg_data(Meta.test_task_set_name, for_training=False)