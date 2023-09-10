from torch_geometric.data import Data
import pickle
import torch
import copy

from fpi_verify import FPIVerifier
from utils import reduce_graph, fixed_point_iteration
from env import ConcreteEnv
from meta import Meta


class InteractEnv:
    def __init__(self, task_set_name, task_id):
        super(InteractEnv, self).__init__()
        self.task_set_name = task_set_name
        self.task_idx = task_id
        
    def reset(self):
        with open(f"{Meta.data_dir}/tasks/{self.task_set_name}/pt-{self.task_idx}.pkl", "rb") as f:
            E, F, A, W, P, u_lst, l_lst = pickle.load(f)
        self.E, self.F, self.A, self.W, self.P, self.u_lst, self.l_lst = reduce_graph(E, F, A, W, P, u_lst, l_lst, save_enabled_flow_perm=True)
        
        self.A_copy = copy.deepcopy(self.A)
        return self.observe_action_space()

    def step(self, action_lst):
        terminated = False
        eftv_action_cnt = 0
        for action in action_lst:
            if action[0] == "ep":
                e = action[1]
                perm_idx = action[2]

                assert(perm_idx in self.A[e])
                self.A[e].remove(perm_idx)
            else:
                assert(action[0] == "ee")
                e1 = action[1]
                e2 = action[2]

                self.W[(e1, e2)] = False
            
            eftv_action_cnt += 1

            concrete_env = ConcreteEnv(None, None, None)
            E = copy.deepcopy(self.E)
            F = copy.deepcopy(self.F)
            A = copy.deepcopy(self.A)
            W = copy.deepcopy(self.W)
            P = copy.deepcopy(self.P)
            u_lst = copy.deepcopy(self.u_lst)
            l_lst = copy.deepcopy(self.l_lst)
            
            concrete_env.set_internal_config(E, F, A, W, P, u_lst, l_lst, reduce_config=True)
            verifier = FPIVerifier(concrete_env)
            msg, fpi_cnt = verifier.verify()

            if msg == "vulnerable":
                pass
            else:
                terminated = True
                break

        action_space = self.observe_action_space()
        return action_space, terminated, eftv_action_cnt

    def observe_action_space(self):
        edge_idx_to_action = {}

        edge_cnt = 0
        for (e1, e2) in self.F:
            if self.W[(e1, e2)]:
                edge_idx_to_action[edge_cnt] = ("ee", e1, e2)
                edge_cnt += 1

        for e in self.E:
            for perm_idx in range(len(self.P)):
                if perm_idx in self.A_copy[e]:
                    if perm_idx in self.A[e]:
                        can_remove = True
                        for (e1, e2) in self.F:
                            if e2 == e and self.W[(e1, e2)] and perm_idx in self.A[e1]:
                                can_remove = False
                                break

                        if can_remove:
                            edge_idx_to_action[edge_cnt] = ("ep", e, perm_idx)

                    edge_cnt += 1

        return edge_idx_to_action