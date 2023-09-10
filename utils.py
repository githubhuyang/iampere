import copy

def fixed_point_iteration(A, W):
    fpi_cnt = 1

    while True:
        # old_A = copy.deepcopy(A)
        changed = False
        for (e0, e1) in W:
            if W[(e0, e1)]: # enabled
                old_size = len(A[e1])
                A[e1] = A[e1].union(A[e0])

                if not changed and len(A[e1]) > old_size:
                    changed = True

        if not changed:
            break

        fpi_cnt += 1

    return A, fpi_cnt


def reduce_graph(E, F, A, W, P, u_lst, l_lst, save_enabled_flow_perm=True):
    while True:
        # step 1: recognize entities which may reach comprimised entities
        sink2srcs = {}
        for e in E:
            sink2srcs[e] = set()
            
        for (e1, e2) in F:
            sink2srcs[e2].add(e1)

        E_reach = copy.deepcopy(u_lst)
        while True:
            E_added = set()
            for e in E_reach:
                E_added = E_added.union(sink2srcs[e])

            old_size = len(E_reach)
            E_reach = E_reach.union(E_added)
            if len(E_reach) == old_size:
                break
        
        for l in l_lst:
            if P[l][0] == "type-i":
                E_reach.add(P[l][1])
                E_reach.add(P[l][2])
            elif P[l][0] == "type-ii":
                E_reach.add(P[l][1])

        # setp 2: recognize flows which have chances to be enabled
        F_prime = set()
        for (e1, e2) in F:
            if e1 in E_reach and e2 in E_reach:
                if W[(e1, e2)]:
                    F_prime.add((e1, e2))
                elif any(P[l][0] == "type-i" and P[l][1] == e1 and P[l][2] == e2 for l in l_lst):
                    F_prime.add((e1, e2))
                else:
                    for e in E_reach:
                        perm_idx_set = A[e]
                        
                        if any((P[perm_idx][0] == "type-i" and P[perm_idx][1] == e1 and P[perm_idx][2] == e2) or \
                            (P[perm_idx][0] == "type-ii" and P[perm_idx][1] in E_reach) for perm_idx in perm_idx_set):
                            F_prime.add((e1, e2))
                            break

        # step 3: update flow states
        W_prime = {}
        for f in F_prime:
            W_prime[f] = W[f]
        
        # step 4: update perm space
        P_prime = []
        P_old_idx_to_new_idx = {}
        for perm_idx, perm in enumerate(P):
            if perm_idx in l_lst:
                P_old_idx_to_new_idx[perm_idx] = len(P_prime)
                P_prime.append(perm)
            elif perm[0] == "type-i":
                e1, e2 = perm[1], perm[2]
                if (e1, e2) in F_prime:
                    if not W_prime[(e1, e2)]:
                        P_old_idx_to_new_idx[perm_idx] = len(P_prime)
                        P_prime.append(perm)
                    elif save_enabled_flow_perm:
                        P_old_idx_to_new_idx[perm_idx] = len(P_prime)
                        P_prime.append(perm)

            elif perm[0] == "type-ii":
                if perm[1] in E_reach:
                    P_old_idx_to_new_idx[perm_idx] = len(P_prime)
                    P_prime.append(perm)

        
        # step 5: update perm assignment
        A_prime = {}
        for e in E_reach:
            A_prime[e] = set()
            perm_idx_set = A[e]

            for perm_idx in perm_idx_set:
                if perm_idx in P_old_idx_to_new_idx:
                    perm_idx_prime = P_old_idx_to_new_idx[perm_idx]
                    A_prime[e].add(perm_idx_prime)
                
        # step 6: update l_lst
        l_lst_prime = set()
        for l in l_lst:
            l_prime = P_old_idx_to_new_idx[l]
            l_lst_prime.add(l_prime)

        if len(E_reach) < len(E) or len(F_prime) < len(F) or len(P_prime) < len(P):
            if len(A_prime) > 0 and len(W_prime) > 0:
                E, F, A, W, P, l_lst = E_reach, F_prime, A_prime, W_prime, P_prime, l_lst_prime
            else:
                break
        else:
            break

    return E, F, A, W, P, u_lst, l_lst