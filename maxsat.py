import pickle
import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from colorama import Fore, Style
import copy
import random
import psutil
from tqdm import tqdm

from utils import reduce_graph, fixed_point_iteration
from env import ConcreteEnv
from fpi_verify import FPIVerifier
from meta import Meta


def fpi_simp(config):
    assert(len(config) == 7)

    E, F, A, W, P, u_lst, l_lst = config[0], config[1], config[2], config[3], config[4], config[5], config[6]
    concrete_env = ConcreteEnv(None, None, None)
    concrete_env.set_internal_config(E, F, A, W, P, u_lst, l_lst, reduce_config=True)

    verifier = FPIVerifier(concrete_env)
    msg, fpi_cnt = verifier.verify()

    if msg == "vulnerable":
        return False, fpi_cnt
    return True, fpi_cnt


class ConcreteModel:
    def __init__(self, task_set_name, task_idx, fpi_only=False):
        self.task_set_name = task_set_name
        self.task_idx = task_idx

        if not fpi_only:
            with open(f"{Meta.data_dir}/tasks/{task_set_name}/pt-{task_idx}.pkl", "rb") as f:
                E, F, A, W, P, u_lst, l_lst = pickle.load(f)

            reduce_start = time.time()
        
            self.E, self.F, self.A, self.W, self.P, self.u_lst, self.l_lst = reduce_graph(E, F, A, W, P, u_lst, l_lst)

            self.reduce_tc = time.time() - reduce_start

            # debug
            self.e_reduce = ((len(E) - len(self.E)) / len(E))
            self.f_reduce = ((len(F) - len(self.F)) / len(F))
            self.p_reduce = ((len(P) - len(self.P)) / len(P))

            # count max patch size
            self.max_patch_size = 0
            for e in self.A:
                self.max_patch_size += len(self.A[e])

            for (e1, e2) in self.W:
                if self.W[(e1, e2)]:
                    self.max_patch_size += 1

    def fpi(self, reduce_config=True):
        concrete_env = ConcreteEnv(self.task_set_name, self.task_idx, reduce_config)
        verifier = FPIVerifier(concrete_env)
        msg, fpi_cnt = verifier.verify()

        if msg == "vulnerable":
            return False, fpi_cnt
        return True, fpi_cnt

    def maxsat_repair(self, bound, time_budget, cand_patch=None, use_complete_solver=True, rm_cnf_file=True):
        assert(len(self.A) > 0 and len(self.W) > 0)

        gen_start = time.time()

        soft_vars = self.gen_repair_clauses(bound, time_budget, cand_patch=cand_patch, save_as_file=True)
        
        time_budget = time_budget - (time.time() - gen_start)
        
        if soft_vars == None or time_budget <= 0:

            if rm_cnf_file and os.path.isfile(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf"):
                os.remove(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf")
            return None, None, -1

        solve_start = time.time()
        
        command = None

        if use_complete_solver:
            # use complete solver
            command = ["timeout", "-s", "15", f"{time_budget}", "cashwmaxsatcoreplus", f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf"]
        else:
            # TODO: use incomplete solver
            raise NotImplementedError()

        p = None
        try:
            p = subprocess.run(command, capture_output=True)
        except Exception as e:
            print("bug:", e)

        if rm_cnf_file:
            os.remove(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf")
         
        if time.time() - solve_start >= time_budget or p == None:
            return None, None, -1

        patch_size = 0
        o_found = False
        v_found = False
        s = p.stdout.decode().strip()

        repair_A = copy.deepcopy(self.A)
        repair_W = copy.deepcopy(self.W)

        for line in reversed(s.split("\n")):
            if line.startswith("o"):
                o_found = True

            if line.startswith("v"):
                # print(line)
                
                patch_msg = ""
                patch_msg += f"{Fore.WHITE} semantic patch ({self.task_set_name} {self.task_idx}):\n"

                eles = line.split(" ")[1:]
                if len(eles) == 1:
                    solution = eles[0]
                    # binary digit format
                    for var_id in soft_vars:
                        if solution[var_id - 1] == "0":
                            mypair = soft_vars[var_id]

                            if isinstance(mypair[1], int):
                                e = mypair[0]
                                perm_idx = mypair[1]

                                perm_info = self.P[perm_idx][:-1]
                                patch_msg += f"revoke ({e}, {perm_info})\n"
                                
                                repair_A[e].remove(perm_idx)
                            else:
                                e1 = mypair[0]
                                e2 = mypair[1]
                                patch_msg += f"revoke ({e1}, {e2})\n"
                                
                                repair_W[(e1, e2)] = False

                            patch_size += 1

                else:
                    # lit format
                    solution = [int(lit_str) for lit_str in eles]
                    for var_id in soft_vars:
                        if solution[var_id - 1] < 0:
                            mypair = soft_vars[var_id]

                            if isinstance(mypair[1], int):
                                e = mypair[0]
                                perm_idx = mypair[1]

                                perm_info = self.P[perm_idx][:-1]
                                patch_msg += f"revoke ({e}, {perm_info})\n"
                                
                                repair_A[e].remove(perm_idx)
                            else:
                                e1 = mypair[0]
                                e2 = mypair[1]
                                patch_msg += f"revoke ({e1}, {e2})\n"
                                
                                repair_W[(e1, e2)] = False

                            patch_size += 1

                print("\n" + patch_msg + "\n")

                if patch_size == 0:
                    print(f"{Fore.RED} fail to generate patch for", self.task_set_name, self.task_idx)

                v_found = True

            if line.startswith("s OPTIMUM FOUND"):
                # print(line)
                pass
            elif line.startswith("s UNSATISFIABLE"):
                print(f"{Fore.RED} Error: {self.task_idx}")
                print(line)
                patch_size = -1

                break

                
            if o_found and v_found:
                break

        return repair_A, repair_W, patch_size


    def gen_repair_clauses(self, bound, time_budget, cand_patch=None, save_as_file=True):
        assert(bound > 1)
        if os.path.isfile(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf"):
            os.remove(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf")

        start_time = time.time()

        # Step 1: allocate boolean var id
        var_cnt = 0

        ep_pair2var_id = {}
        for e in self.E:
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None

            for perm_idx in range(len(self.P)):
                if perm_idx in self.l_lst or self.P[perm_idx][0] != "other":
                    var_id = var_cnt + 1
                    var_cnt += 1

                    ep_pair2var_id[(e, perm_idx)] = var_id

        ee_pair2var_id = {}
        for (e1, e2) in self.F:
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None

            var_id = var_cnt + 1
            var_cnt += 1

            ee_pair2var_id[(e1, e2)] = var_id

        eep_tuple2var_id = {}
        for (e1, e2) in self.F:
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None
            for perm_idx in range(len(self.P)):
                if perm_idx in self.l_lst or self.P[perm_idx][0] != "other":
                    var_id = var_cnt + 1
                    var_cnt += 1

                    eep_tuple2var_id[(e1, e2, perm_idx)] = var_id

        ep_old_and_ep_new2var_id = {}
        for e in self.E:
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None
            for perm_idx in range(len(self.P)):
                if perm_idx in self.l_lst or self.P[perm_idx][0] != "other":
                    var_id = var_cnt + 1
                    var_cnt += 1

                    ep_old_and_ep_new2var_id[(e, perm_idx)] = var_id

        var_id = var_cnt + 1
        var_cnt += 1
        use_perm_init_var = var_id
        
        # Step 2. gen clauses for initial state
        can_remove_lst = None
        if cand_patch != None:
            can_remove_lst = []
            for action in cand_patch:
                # print(action)
                if action[0] == "ep":
                    e = action[1]
                    perm_idx = action[2]
                    can_remove_lst.append((e, perm_idx))
                else:
                    assert(action[0] == "ee")
                    e1 = action[1]
                    e2 = action[2]

                    can_remove_lst.append((e1, e2))
            assert(len(can_remove_lst) > 0)

        clauses_for_init_state = []
        clauses_for_init_state.append(['h', use_perm_init_var]) # use perm

        soft_vars = {}
        for e, perm_idx_set in self.A.items():
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None
            for perm_idx in range(len(self.P)):
                if perm_idx in self.l_lst or self.P[perm_idx][0] != "other":
                    var_id = ep_pair2var_id[(e, perm_idx)]

                    c = None
                    if perm_idx in perm_idx_set:
                        w = 1 # w = 5
                        if e in self.u_lst: # prioritize revoking permissions assigned to comprimised entity
                            w = 1

                        # c = [w, var_id] # soft clause
                        # soft_vars[var_id] = (e, perm_idx)
                        if can_remove_lst == None or (e, perm_idx) in can_remove_lst:
                            c = [w, var_id] # soft clause
                            soft_vars[var_id] = (e, perm_idx)
                        else:
                            c = ['h', var_id]
                    else:
                        c = ['h', -var_id] # hard clause
                        
                    clauses_for_init_state.append(c)

        for (e1, e2), enabled in self.W.items():
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None

            var_id = ee_pair2var_id[(e1, e2)]

            c = None
            if enabled:
                w = 1 # w = 3
                if e2 in self.u_lst: # prioritize revoking permission flows towards compromised entities
                    w = 1

                if can_remove_lst == None or (e1, e2) in can_remove_lst:
                    c = [w, var_id] # soft clause
                    soft_vars[var_id] = (e1, e2)
                else:
                    c = ['h', var_id]
            else:
                c = ['h', -var_id] # hard clause

            clauses_for_init_state.append(c)

            for perm_idx in range(len(self.P)):
                if perm_idx in self.l_lst or self.P[perm_idx][0] != "other":
                    #    (a and b) <-> x
                    # := (a and b) -> x and x -> (a and b)
                    # := (-a or -b or x) and (-x or (a and b))
                    # := (-a or -b or x) and (-x or a) and (-x or b)
                    var_id_a = ep_pair2var_id[(e1, perm_idx)]
                    var_id_b = var_id
                    var_id_x = eep_tuple2var_id[(e1, e2, perm_idx)]

                    c1 = ['h', -var_id_a, -var_id_b, var_id_x]
                    clauses_for_init_state.append(c1)

                    c2 = ['h', -var_id_x, var_id_a]
                    clauses_for_init_state.append(c2)

                    c3 = ['h', -var_id_x, var_id_b]
                    clauses_for_init_state.append(c3)

                    # make sure it is currently in the fixed point
                    var_id_c = ep_pair2var_id[(e2, perm_idx)]
                    c4 = ['h', -var_id_x, var_id_c]
                    clauses_for_init_state.append(c4)

        # Step 3. gen clauses for states transitions
        other_clauses = []
        for b in range(bound - 1):
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None

            curr_use_perm_init_var = use_perm_init_var + b * var_cnt
            next_use_perm_init_var = use_perm_init_var + (b + 1) * var_cnt

            # speed up via p'
            # rule 0.1:
            #    p' -> -p
            # := -p' or -p
            c = ['h', -curr_use_perm_init_var, -next_use_perm_init_var]
            other_clauses.append(c)

            # rule 0.2
            #    (-a' and a and -p') -> -p
            # := a' or -a or p' or -p
            for init_var_id in ep_pair2var_id.values():
                curr_var_id = init_var_id + b * var_cnt
                next_var_id = init_var_id + (b + 1) * var_cnt

                c = ['h', curr_var_id, -next_var_id, curr_use_perm_init_var, -next_use_perm_init_var]
                other_clauses.append(c)

            # rule 0.3
            #    ((-a' -> -a) and (-b' -> -b) and -p') -> p
            # := ((a' or -a) and (b' or -b) and -p') -> p
            # := (-a' and a) or (-b' and b) or p' or p
            # := y1' or y2' or p' or p
            c = ['h', curr_use_perm_init_var, next_use_perm_init_var]
            for init_var_id in ep_old_and_ep_new2var_id.values():
                curr_var_id = init_var_id + b * var_cnt
                c.append(curr_var_id)

            other_clauses.append(c)

            # rule 0.4
            # y1' <-> (-a' and a)
            # (-y1' or (-a' and a)) and (a' or -a or y1')
            # (-y1' or -a') and (-y1' and a) and (a' or -a or y1')
            for (e, p), init_var_id_y in ep_old_and_ep_new2var_id.items():
                curr_var_id_y = init_var_id_y + b * var_cnt

                init_var_id_a = ep_pair2var_id[(e, p)]
                curr_var_id_a = init_var_id_a + b * var_cnt
                next_var_id_a = init_var_id_a + (1 + b) * var_cnt

                c1 = ['h', -curr_var_id_y, -curr_var_id_a]
                other_clauses.append(c1)

                c2 = ['h', -curr_var_id_y, next_var_id_a]
                other_clauses.append(c2)

                c3 = ['h', curr_var_id_a, -next_var_id_a, curr_var_id_y]
                other_clauses.append(c3)


            # rule 1.1: curr_var_id -> next_var_id (for entity)
            for init_var_id in ep_pair2var_id.values():
                curr_var_id = init_var_id + b * var_cnt
                next_var_id = init_var_id + (1 + b) * var_cnt

                c = ['h', -curr_var_id, next_var_id]
                other_clauses.append(c)

            # rule 1.2: curr_var_id -> next_var_id (for flow)
            for init_var_id in ee_pair2var_id.values():
                curr_var_id = init_var_id + b * var_cnt
                next_var_id = init_var_id + (1 + b) * var_cnt

                c = ['h', -curr_var_id, next_var_id]
                other_clauses.append(c)

            # rule 2: perm used to enable a flow or assign perms to other entity 
            for (e, perm_idx) in ep_pair2var_id:
                if e in self.u_lst:
                    init_var_id_1 = ep_pair2var_id[(e, perm_idx)]
                    curr_var_id = init_var_id_1 + b * var_cnt 

                    perm = self.P[perm_idx]
                    if perm[0] == "type-i":
                        e1 = perm[1]
                        e2 = perm[2]

                        init_var_id_2 = ee_pair2var_id[(e1, e2)]
                        next_var_id = init_var_id_2 + (1 + b) * var_cnt

                        #    (p' and curr_var_id) -> next_var_id
                        # := -p' or -curr_var_id or next_var_id
                        c = ['h', -curr_use_perm_init_var, -curr_var_id, next_var_id]
                        other_clauses.append(c)
                    elif perm[0] == "type-ii":
                        e1 = perm[1]
                        for pid in range(len(self.P)):
                            if pid in self.l_lst or self.P[pid][0] != "other":
                                init_var_id_2 = ep_pair2var_id[(e1, pid)]
                                next_var_id = init_var_id_2 + (1 + b) * var_cnt

                                #    (p' and curr_var_id) -> next_var_id
                                # := -p' or -curr_var_id or next_var_id
                                c = ['h', -curr_use_perm_init_var, -curr_var_id, next_var_id]
                                other_clauses.append(c)
                    else:
                        pass

            # rule 2.1:
            #    (p' and -g' and -h' and -e') -> -e
            # := -(p' and -g' and -h' and -e') or -e
            # := -p' or g' or h' or e' or -e
            for (e, perm_idx), init_var_id_e in ep_pair2var_id.items():
                c = ['h', -curr_use_perm_init_var]
                for pid in range(len(self.P)):
                    perm = self.P[pid]
                    if perm[0] == "type-ii" and perm[1] == e:
                        for u in self.u_lst:
                            init_var_id_u = ep_pair2var_id[(u, pid)]
                            curr_var_id_u = init_var_id_u + b * var_cnt
                            
                            c.append(curr_var_id_u)

                curr_var_id_e = init_var_id_e + b * var_cnt
                next_var_id_e = init_var_id_e + (1 + b) * var_cnt
                c += [curr_var_id_e, -next_var_id_e]

                other_clauses.append(c)

            # rule 2.2:
            #    (p' and -u1' and -u2' and -f') -> -f (for flow)
            # := -(p' and -u1' and -u2' and -f') or -f
            # := -p' or u1' or u2' or f' or -f
            for (e1, e2), init_var_id_f in ee_pair2var_id.items():
                past_time = time.time() - start_time
                if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                    return None

                curr_var_id_f = init_var_id_f + b * var_cnt
                next_var_id_f = init_var_id_f + (1 + b) * var_cnt

                c = ['h', -curr_use_perm_init_var, curr_var_id_f, -next_var_id_f]
                for pid in range(len(self.P)):
                    perm = self.P[pid]
                    if perm[0] == "type-i" and perm[1] == e1 and perm[2] == e2:

                        for u in self.u_lst:
                            init_var_id_u = ep_pair2var_id[(u, pid)]
                            curr_var_id_u = init_var_id_u + b * var_cnt
                            
                            c.append(curr_var_id_u)

                other_clauses.append(c)

            # rule 3: enabled flow sync
            for (e1, e2), init_var_id_0 in ee_pair2var_id.items():
                past_time = time.time() - start_time
                if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                    return None

                curr_var_id_0 = init_var_id_0 + b * var_cnt

                for pid in range(len(self.P)):
                    if pid in self.l_lst or self.P[pid][0] != "other":
                        init_var_id_1 = ep_pair2var_id[(e1, pid)]
                        curr_var_id_1 = init_var_id_1 + b * var_cnt

                        init_var_id_2 = ep_pair2var_id[(e2, pid)]
                        next_var_id_2 = init_var_id_2 + (1 + b) * var_cnt

                        #    (-p' and a' and b') -> c
                        # := -(-p' and a' and b') or c
                        # := p' or -a' or -b' or c
                        c = ['h', curr_use_perm_init_var, -curr_var_id_0, -curr_var_id_1, next_var_id_2]
                        other_clauses.append(c)

            # rule 3.1:
            #    (-p' and -f') -> -f
            # := p' or f' or -f
            for init_var_id_f in ee_pair2var_id.values():
                past_time = time.time() - start_time
                if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                    return None

                curr_var_id_f = init_var_id_f + b * var_cnt
                next_var_id_f = init_var_id_f + (1 + b) * var_cnt

                c = ['h', curr_use_perm_init_var, curr_var_id_f, -next_var_id_f]
                other_clauses.append(c)

            # rule 3.2:
            #    (-p' and (-a' or -b') and (-c' or -d') and -e') -> -e
            # := -(-p' and (-a' or -b'') and (-c' or -d') and -e') or -e
            # := p' or -(-a' or -b') or -(-c' or -d') or e' or -e
            # := p' or (a' and b') or (c' and d') or e' or -e
            # := p' or x1' or x2' or e' or -e
            sink2srcs = {}
            for e in self.E:
                sink2srcs[e] = set()

            for (e1, e2) in ee_pair2var_id:
                sink2srcs[e2].add(e1)

            for sink, srcs in sink2srcs.items():
                past_time = time.time() - start_time
                if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                    return None

                if len(srcs) == 0:
                    for pid in range(len(self.P)):
                        if pid in self.l_lst or self.P[pid][0] != "other":
                            init_var_id = ep_pair2var_id[(sink, pid)]

                            curr_var_id = init_var_id + b * var_cnt
                            next_var_id = init_var_id + (1 + b) * var_cnt
                            c = ['h', curr_use_perm_init_var, curr_var_id, -next_var_id]

                            other_clauses.append(c)
                else:
                    for pid in range(len(self.P)):
                        if pid in self.l_lst or self.P[pid][0] != "other":
                            init_var_id = ep_pair2var_id[(sink, pid)]

                            curr_var_id = init_var_id + b * var_cnt
                            next_var_id = init_var_id + (1 + b) * var_cnt
                            c = ['h', curr_use_perm_init_var, curr_var_id, -next_var_id]

                            for src in srcs:
                                init_var_id_another = eep_tuple2var_id[(src, sink, pid)]

                                curr_var_id_another = init_var_id_another + b * var_cnt
                                c.append(curr_var_id_another)
                    
                            other_clauses.append(c)

            # rule 4:
            #    (a and b) <-> x
            # := (a and b) -> x and x -> (a and b)
            # := (-a or -b or x) and (-x or (a and b))
            # := (-a or -b or x) and (-x or a) and (-x or b)
            for sink, srcs in sink2srcs.items():
                past_time = time.time() - start_time
                if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                    return None

                for src in srcs:
                    for pid in range(len(self.P)):
                        if pid in self.l_lst or self.P[pid][0] != "other":
                            init_var_id_a = ep_pair2var_id[(src, pid)]
                            next_var_id_a = init_var_id_a + (1 + b) * var_cnt

                            init_var_id_b = ee_pair2var_id[(src, sink)]
                            next_var_id_b = init_var_id_b + (1 + b) * var_cnt

                            init_var_id_x = eep_tuple2var_id[(src, sink, pid)]
                            next_var_id_x = init_var_id_x + (1 + b) * var_cnt

                            c1 = ['h', -next_var_id_a, -next_var_id_b, next_var_id_x]
                            c2 = ['h', -next_var_id_x, next_var_id_a]
                            c3 = ['h', -next_var_id_x, next_var_id_b]
                            
                            other_clauses.append(c1)
                            other_clauses.append(c2)
                            other_clauses.append(c3)
        
        # no error: u never has target perm
        no_error_clauses = []
        for b in range(-1, bound - 1):
            for u in self.u_lst:
                for l_idx in self.l_lst:
                    init_var_id = ep_pair2var_id[(u, l_idx)]
                    next_var_id = init_var_id + (1 + b) * var_cnt

                    no_error_clauses.append(['h', -next_var_id])

        clauses = clauses_for_init_state + other_clauses + no_error_clauses

        if save_as_file:
            past_time = time.time() - start_time
            if past_time > time_budget or psutil.Process().memory_info().rss / (1024 * 1024 * 1024) > Meta.max_mem_cost:
                return None

            self.write_wcnf(clauses)

        return soft_vars

    def write_wcnf(self, clauses):
        with open(f"{Meta.data_dir}/cnf/{self.task_set_name}-{self.task_idx}.wcnf", "w") as f:
            for c in clauses:
                c_w_end = [str(lit) for lit in c] + ["0"]
                c_str = " ".join(c_w_end)
                f.write(c_str + "\n")


def maxsat_repair_single(args, return_repair=False):
    start_time = time.time()

    task_set_name, task_idx, cand_patch, use_complete_solver = args[0], args[1], args[2], args[3]

    m0 = ConcreteModel(task_set_name, task_idx, fpi_only=True)
    
    is_safe, fpi_cnt = m0.fpi(reduce_config=False)

    patch_size = 0
    repair = None

    m1 = None
    fpi_cnt_lst = [fpi_cnt]
    while not is_safe:
        left_time_budget = Meta.solver_timeout - (time.time() - start_time)
        if left_time_budget <= 0:
            patch_size = -1
            break

        m1 = ConcreteModel(task_set_name, task_idx)
        
        repair_A, repair_W, patch_size = m1.maxsat_repair(bound=fpi_cnt, time_budget=left_time_budget, cand_patch=cand_patch, use_complete_solver=use_complete_solver)

        if patch_size == -1:
            break

        repair = (m1.E, m1.F, repair_A, repair_W, m1.P, m1.u_lst, m1.l_lst)

        is_safe, fpi_cnt_prime = fpi_simp(repair)

        if not is_safe:
            fpi_cnt = max(fpi_cnt + 1, fpi_cnt_prime)
            fpi_cnt_lst.append(fpi_cnt)

    solve_tc = time.time() - start_time
    if solve_tc > Meta.solver_timeout or patch_size == -1:
        solve_tc = Meta.solver_timeout

        if cand_patch != None:
            patch_size = len(cand_patch)
        else:
            patch_size = -1

    maxsat_tc_str = '%.4f' % solve_tc
    if patch_size != -1:
        print(f"{Fore.GREEN} task_id={task_idx}, maxsat={maxsat_tc_str}s (bound={fpi_cnt_lst})", flush=True)
    else:
        print(f"{Fore.YELLOW} task_id={task_idx}, maxsat timeout (bound={fpi_cnt_lst})", flush=True)

    if return_repair:
        if patch_size == -1:
            repair = None

        return repair

    return task_idx, m1.reduce_tc, m1.e_reduce, m1.f_reduce, m1.p_reduce, patch_size, m1.max_patch_size, solve_tc


def maxsat_repair_parallel(task_set_name, read_cand_patch, use_complete_solver):
    print(f"maxsat_repair_parallel ({task_set_name}, read_cand_patch={read_cand_patch}, use_complete_solver={use_complete_solver}):")

    args_lst = []
    if read_cand_patch:
        for file_name in os.listdir(f"{Meta.data_dir}/tasks/{task_set_name}"):
            task_idx = int(file_name.split(".")[0].split("-")[-1])

            cand_patch_path = f"{Meta.data_dir}/cand_patches/{task_set_name}/cp-{task_idx}.pkl"
            
            cand_patch = None
            if os.path.isfile(cand_patch_path):
                with open(cand_patch_path, "rb") as f:
                    cand_patch = pickle.load(f)

                    if len(cand_patch) == 0:
                        cand_patch = None

            args_lst.append((task_set_name, task_idx, cand_patch, use_complete_solver))
    else:
        for file_name in os.listdir(f"{Meta.data_dir}/tasks/{task_set_name}"):
            task_idx = int(file_name.split(".")[0].split("-")[-1])
            args_lst.append((task_set_name, task_idx, None, use_complete_solver))

    with tqdm(total=len(args_lst), desc="task", leave=False) as pbar:
        with Pool(Meta.n_cpu) as p:
            task_idx_lst = []
            reduce_tc_lst = []
            e_reduce_lst = []
            f_reduce_lst = []
            p_reduce_lst = []
            patch_size_lst = []
            max_patch_size_lst = []
            maxsat_solve_tc_lst = []

            for i, (task_idx, reduce_tc, e_reduce, f_reduce, p_reduce, patch_size, max_patch_size, solve_tc) in enumerate(p.imap_unordered(maxsat_repair_single, args_lst)):
                 task_idx_lst.append(task_idx)
                 reduce_tc_lst.append(reduce_tc)
                 e_reduce_lst.append(e_reduce)
                 f_reduce_lst.append(f_reduce)
                 p_reduce_lst.append(p_reduce)

                 patch_size_lst.append(patch_size)
                 max_patch_size_lst.append(max_patch_size)

                 maxsat_solve_tc_lst.append(solve_tc)

                 pbar.update()

            df = pd.DataFrame({'task': task_idx_lst,
                    'ps': patch_size_lst,
                    'max-ps': max_patch_size_lst,
                    'tc': maxsat_solve_tc_lst})

            df = df.sort_values(by=['task'])
            
            s1 = ""
            if read_cand_patch:
                s1 = "wcp"
            else:
                s1 = "wocp"
            
            s2 = ""
            if use_complete_solver:
                s2 = "c"
            else:
                s2 = "i"

            df.to_csv(f'{Meta.csv_dir}/{task_set_name}_{s1}_{s2}.csv', index=False)

if __name__ == '__main__':
    maxsat_repair_parallel(Meta.test_task_set_name, read_cand_patch=True, use_complete_solver=True)