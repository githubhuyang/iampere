import copy
import csv
import pickle
import random
import os
from multiprocessing import Pool
from tqdm import tqdm
import json
import shutil

from meta import Meta
from env import ConcreteEnv


class FPIVerifier:
    def __init__(self, concrete_env):
        self.concrete_env = concrete_env

    def verify(self):
        fpi_cnt = 2
        while True:
            self.reset()
            for u, perm_idx in self.requests:
                self.concrete_env.execute(u, perm_idx, enable_fpi=False)
                
            fpi_cnt = fpi_cnt + 1 + self.concrete_env.fixed_point_itr()

            for u in self.concrete_env.u_lst:
                for l in self.concrete_env.l_lst:
                    if l in self.concrete_env.A[u]:
                        return "vulnerable", fpi_cnt

            if all(len(self.old_A[e]) == len(self.concrete_env.A[e]) for e in self.concrete_env.E):
                break

        return "no issue detected", fpi_cnt


    def reset(self):
        self.old_A = copy.deepcopy(self.concrete_env.A)

        self.requests = []
        for u in self.concrete_env.u_lst:
            for perm_idx in self.concrete_env.A[u]:
                perm = self.concrete_env.P[perm_idx]
                if perm[0] == "other":
                    continue
                elif perm[0] == "type-i" and self.concrete_env.W[(perm[1], perm[2])] == True:
                    continue
                elif perm[0] == "type-ii" and len(self.concrete_env.A[perm[1]]) == len(self.concrete_env.P):
                    continue
                else:
                    self.requests.append((u, perm_idx))