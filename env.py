import torch
from torch_geometric.data import Data
from enum import Enum, auto
import pickle
import copy

from meta import Meta
from utils import fixed_point_iteration, reduce_graph


class ConcreteEnv:
    def __init__(self, task_set_name, task_idx, reduce_config):
        if task_set_name == None or task_idx == None:
            return

        self.task_set_name = task_set_name
        self.task_idx = task_idx

        with open(f"{Meta.data_dir}/tasks/{self.task_set_name}/pt-{self.task_idx}.pkl", "rb") as f:
            E, F, A, W, P, u_lst, l_lst = pickle.load(f)

        # no simplify
        self.set_internal_config(E, F, A, W, P, u_lst, l_lst, reduce_config=False) # True

    def set_internal_config(self, E, F, A, W, P, u_lst, l_lst, reduce_config):
        if reduce_config:
            self.E, self.F, self.A, self.W, self.P, self.u_lst, self.l_lst = reduce_graph(E, F, A, W, P, u_lst, l_lst)
        else:
            self.E, self.F, self.A, self.W, self.P, self.u_lst, self.l_lst = E, F, A, W, P, u_lst, l_lst

    def _has_perm(self, e, perm_idx):
        return perm_idx in self.A[e]

    def _update_by_perm(self, perm_idx, enable_fpi):
        perm = self.P[perm_idx]
        if perm[0] == "type-i":
            e0 = perm[1]
            e1 = perm[2]

            self.W[(e0, e1)] = True
        elif perm[0] == "type-ii":
            e = perm[1]

            self.A[e] = set([i for i in range(len(self.P))]) # copy.deepcopy(self.P)
        else: # perms with no affects on config
            assert(perm[0] == "other")
            pass

        if enable_fpi:
            self.fixed_point_itr()

    def fixed_point_itr(self):
        self.A, fpi_cnt = fixed_point_iteration(self.A, self.W)
        return fpi_cnt

    def execute(self, u, perm_idx, enable_fpi=True):
        assert(u in self.u_lst)
        
        acd = False # access control decision
        if self._has_perm(u, perm_idx):
            acd = True
            self._update_by_perm(perm_idx, enable_fpi)
        
        return acd

    #def get_prior_knowledge(self):
    #    return self.E, self.F, self.P, self.u_lst, self.l_lst