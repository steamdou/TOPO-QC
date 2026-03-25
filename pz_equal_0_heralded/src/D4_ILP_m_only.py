import os, sys
import numpy as np
from scipy.sparse import coo_matrix
import gurobipy as gp
from gurobipy import GRB

sys.path.append(os.path.dirname(__file__))
from tableaux import *

def ILP_decode(V, E1_list, E2_list, T1_list, T2_list,
               Gamma1, Gamma2, w1, w2, env=None):
    """
    Multi-species ILP decoder for the global string-network formulation.

    Inputs:
        V          : list of vertex indices
        E1_list    : list of 3 lists, E1_list[a] = list of S1 edge indices of color a
        E2_list    : list of 3 lists, E2_list[b] = list of S2 edge indices of color b
        T1_list    : list of S1 terminal sets (per species), using vertex indices
        T2_list    : list of S2 terminal sets (per species), using vertex indices
        Gamma1     : dict[a][v] -> list of S1 edges incident on v
        Gamma2     : dict[b][v] -> list of S2 edges incident on v
        w1, w2     : global edge weight arrays, shape (3*|V|,) or similar

    Returns:
        x_sol : binary solution vector (numpy array)
        y1_idx, y2_idx : dictionaries mapping (a,e) / (b,e) to variable indices in x_sol
    """

    V = list(V)
    nS1 = len(E1_list)
    nS2 = len(E2_list)

    # -------------------------------------------------------------------------
    # Data: tau_v(2,b) = 1 if v is an S2 terminal of species b, else 0
    # -------------------------------------------------------------------------
    tau2 = {}
    for b in range(nS2):
        for v in V:
            tau2[(b, v)] = 1 if v in T2_list[b] else 0

    # S1 terminal indicators T1_data[a,v]
    T1_data = {}
    for a in range(nS1):
        for v in V:
            T1_data[(a, v)] = 1 if v in T1_list[a] else 0

    # -------------------------------------------------------------------------
    # Variable indexing
    # -------------------------------------------------------------------------
    offset = 0

    # S1: edge activations y1[a,e]
    y1_idx = {}
    for a in range(nS1):
        for e in E1_list[a]:
            y1_idx[(a, e)] = offset
            offset += 1

    # S1: midpoints x1[a,v]
    x1_idx = {}
    for a in range(nS1):
        for v in V:
            x1_idx[(a, v)] = offset
            offset += 1

    # S2: edge activations y2[b,e]
    y2_idx = {}
    for b in range(nS2):
        for e in E2_list[b]:
            y2_idx[(b, e)] = offset
            offset += 1

    # S2: midpoints x2[b,v]
    x2_idx = {}
    for b in range(nS2):
        for v in V:
            x2_idx[(b, v)] = offset
            offset += 1

    # S2: terminal state T2[b,v] and absorption z2[b,v]
    T2_idx = {}
    z2_idx = {}
    for b in range(nS2):
        for v in V:
            T2_idx[(b, v)] = offset
            offset += 1
            z2_idx[(b, v)] = offset
            offset += 1

    # S1–S2 contact indicators g[b,v,a], BUT ONLY for S2 terminals v ∈ T2_list[b]
    g_idx = {}
    for b in range(nS2):
        for v in T2_list[b]:
            for a in range(nS1):
                g_idx[(b, v, a)] = offset
                offset += 1

    nvar = offset
    # print(f"Total variables: {nvar}")

    # -------------------------------------------------------------------------
    # Objective: cost on y1 and y2
    # -------------------------------------------------------------------------
    c = np.zeros(nvar)
    w1 = np.asarray(w1)
    w2 = np.asarray(w2)

    for a in range(nS1):
        for e in E1_list[a]:
            c[y1_idx[(a, e)]] = w1[e]

    for b in range(nS2):
        for e in E2_list[b]:
            c[y2_idx[(b, e)]] = w2[e]

    # -------------------------------------------------------------------------
    # Sparse constraint helpers
    # -------------------------------------------------------------------------
    eq_i, eq_j, eq_val, b_eq = [], [], [], []
    ub_i, ub_j, ub_val, b_ub = [], [], [], []
    row_eq = 0
    row_ub = 0

    def add_eq_idxvals(indices, values, rhs):
        nonlocal row_eq
        if not indices:
            row_eq += 1
            b_eq.append(float(rhs))
            return
        eq_i.extend([row_eq] * len(indices))
        eq_j.extend(indices)
        eq_val.extend([float(v) for v in values])
        b_eq.append(float(rhs))
        row_eq += 1

    def add_ub_idxvals(indices, values, rhs):
        nonlocal row_ub
        if not indices:
            row_ub += 1
            b_ub.append(float(rhs))
            return
        ub_i.extend([row_ub] * len(indices))
        ub_j.extend(indices)
        ub_val.extend([float(v) for v in values])
        b_ub.append(float(rhs))
        row_ub += 1

    # -------------------------------------------------------------------------
    # 1) S1 degree constraints:
    #    sum_{e in Gamma1[a][v]} y1[a,e] = 2 * x1[a,v] + T1_data[a,v]
    # -------------------------------------------------------------------------
    for a in range(nS1):
        for v in V:
            incident_edges = Gamma1[a].get(v, [])
            idxs = [y1_idx[(a, e)] for e in incident_edges] + [x1_idx[(a, v)]]
            vals = [1.0] * len(incident_edges) + [-2.0]
            rhs = float(T1_data[(a, v)])
            add_eq_idxvals(idxs, vals, rhs)

    # -------------------------------------------------------------------------
    # 2) S2 contact and absorption with S1
    #
    # Only for S2 terminals v ∈ T2_list[b]:
    #   sum_e y1[a,e] <= deg(1,a)(v) * g[b,v,a]
    #   sum_e y1[a,e] >= g[b,v,a]
    #
    # For each S2 terminal (b,v):
    #   z2[b,v] <= sum_a g[b,v,a]
    #   z2[b,v] >= g[b,v,a] for all a
    # -------------------------------------------------------------------------
    for b in range(nS2):
        for v in T2_list[b]:
            for a in range(nS1):
                g_var = g_idx[(b, v, a)]
                incident_edges = Gamma1[a].get(v, [])
                deg1 = len(incident_edges)

                # sum_e y1[a,e] - deg1 * g <= 0
                idxs = [y1_idx[(a, e)] for e in incident_edges] + [g_var]
                vals = [1.0] * len(incident_edges) + [-float(deg1)]
                add_ub_idxvals(idxs, vals, 0.0)

                # -sum_e y1[a,e] + g <= 0  (sum_e y1 >= g)
                idxs = [y1_idx[(a, e)] for e in incident_edges] + [g_var]
                vals = [-1.0] * len(incident_edges) + [1.0]
                add_ub_idxvals(idxs, vals, 0.0)

        # Absorption rule per S2 terminal:
        for v in T2_list[b]:
            z_var = z2_idx[(b, v)]
            g_vars = [g_idx[(b, v, a)] for a in range(nS1)]

            # z2[b,v] <= sum_a g[b,v,a]
            idxs = [z_var] + g_vars
            vals = [1.0] + [-1.0] * len(g_vars)
            add_ub_idxvals(idxs, vals, 0.0)

            # z2[b,v] >= g[b,v,a]  => g[b,v,a] - z2[b,v] <= 0
            for a in range(nS1):
                g_var = g_idx[(b, v, a)]
                idxs = [g_var, z_var]
                vals = [1.0, -1.0]
                add_ub_idxvals(idxs, vals, 0.0)

    # -------------------------------------------------------------------------
    # 3) S2 terminal state:
    #    For each (b,v): T2[b,v] + z2[b,v] = tau2[b,v]
    # -------------------------------------------------------------------------
    for b in range(nS2):
        for v in V:
            idxs = [T2_idx[(b, v)], z2_idx[(b, v)]]
            vals = [1.0, 1.0]
            rhs = float(tau2[(b, v)])
            add_eq_idxvals(idxs, vals, rhs)

    # -------------------------------------------------------------------------
    # 4) S2 degree constraints with big-M gating by S1 presence
    #
    # If you want to *temporarily* ignore S2 degree constraints,
    # you can comment out this whole block to debug S1+absorption only.
    # -------------------------------------------------------------------------
    for b in range(nS2):
        for v in V:
            incident_edges = Gamma2[b].get(v, [])
            deg2 = len(incident_edges)
            M = float(max(deg2, 3))  # safe-ish big-M

            # sum_T1 = sum of S1 terminal indicators at v (data)
            sum_T1 = sum(T1_data[(a, v)] for a in range(nS1))

            # Constraint 1: deg2 - 2 x2[b,v] - T2[b,v] - M * sum_a x1[a,v] <= M * sum_T1
            idxs = [y2_idx[(b, e)] for e in incident_edges]
            vals = [1.0] * len(incident_edges)

            idxs.append(x2_idx[(b, v)])
            vals.append(-2.0)

            idxs.append(T2_idx[(b, v)])
            vals.append(-1.0)

            for a in range(nS1):
                idxs.append(x1_idx[(a, v)])
                vals.append(-M)

            rhs = M * float(sum_T1)
            add_ub_idxvals(idxs, vals, rhs)

            # Constraint 2: -deg2 + 2 x2[b,v] + T2[b,v] - M * sum_a x1[a,v] <= M * sum_T1
            idxs = [y2_idx[(b, e)] for e in incident_edges]
            vals = [-1.0] * len(incident_edges)

            idxs.append(x2_idx[(b, v)])
            vals.append(2.0)

            idxs.append(T2_idx[(b, v)])
            vals.append(1.0)

            for a in range(nS1):
                idxs.append(x1_idx[(a, v)])
                vals.append(-M)

            rhs = M * float(sum_T1)
            add_ub_idxvals(idxs, vals, rhs)

    # -------------------------------------------------------------------------
    # Build & solve with Gurobi
    # -------------------------------------------------------------------------
    A_eq = coo_matrix((eq_val, (eq_i, eq_j)), shape=(row_eq, nvar))
    b_eq_arr = np.array(b_eq, dtype=float)
    A_ub = coo_matrix((ub_val, (ub_i, ub_j)), shape=(row_ub, nvar))
    b_ub_arr = np.array(b_ub, dtype=float)

    # CRITICAL FIX: Use the provided environment instead of default
    if env is not None:
        m = gp.Model("ILP_decode_two_layer_S1", env=env)
    else:
        m = gp.Model("ILP_decode_two_layer_S1")
    m.Params.OutputFlag = 0

    m.Params.Threads = 1
    
    # m.Params.TimeLimit = 300.0
    # m.Params.MIPGap = 0.005

    m.Params.MIPGap = 0.0
    m.Params.MIPGapAbs = 0.0
    # m.Params.OptimalityTol = 1e-9
    # m.Params.FeasibilityTol = 1e-9
    m.Params.TimeLimit = GRB.INFINITY
    
    m.Params.ConcurrentMIP = 1

    xvars = m.addMVar(shape=nvar, vtype=GRB.BINARY, name="x")
    m.setObjective(c @ xvars, GRB.MINIMIZE)
    if A_eq.shape[0] > 0:
        m.addMConstr(A_eq, xvars, "=", b_eq_arr)
    if A_ub.shape[0] > 0:
        m.addMConstr(A_ub, xvars, "<", b_ub_arr)

    m.optimize()

    if m.Status != GRB.OPTIMAL:
        print(f"Solver status: {m.Status} (non-optimal)")
        m.dispose()
        return None, None, None

    x_sol = np.rint(xvars.X).astype(int)
    m.dispose()
    return x_sol, y1_idx, y2_idx

def build_ILP_structure(linear_size, cn_dict, w1, w2):
    """
    Build ILP data structures from D4-style connectivity.

    linear_size = (Nx, Ny)  # Nx = #columns, Ny = #rows

    cn_dict:
        'edge_color_arr': color of edges, 0-red , 1-green, 2-blue
        'endpoint_vertex': list of length 3*Nx*Ny, each entry is a dict with keys 'pv', 'qv'
        'star_syndromes':  np.array of shape (3*Nx*Ny, 2)

    Returns:
        V          : list of vertex indices [0 .. Nx*Ny - 1]
        E1_list    : list of 3 lists; E1_list[a] is list of S1 edge indices of color a
        E2_list    : list of 3 lists; E2_list[b] is list of S2 edge indices of color b
        Gamma1     : dict[a][v] -> list of S1 edge indices incident on vertex v
        Gamma2     : dict[b][v] -> list of S2 edge indices incident on vertex v
        w1_arr     : np.array of shape (3*Nx*Ny,), filled with uniform S1 edge weight w1
        w2_arr     : np.array of shape (3*Nx*Ny,), filled with uniform S2 edge weight w2
    """
    Nx, Ny = linear_size

    N_vert = Nx * Ny
    N_edge = 3 * Nx * Ny

    V = list(range(N_vert))

    # Colors: col_arr[e] in {0,1,2} for edge index e
    col_arr = cn_dict['edge_color_arr']
    endpoint_vertex = cn_dict['endpoint_vertex']  # list of dicts with 'pv', 'qv'
    star_syndromes  = cn_dict['star_syndromes']   # (N_edge, 2)

    # Endpoint arrays for S1 and S2 (used only to build Gamma1/Gamma2)
    E1_u = np.empty(N_edge, dtype=int)
    E1_v = np.empty(N_edge, dtype=int)
    for k in range(N_edge):
        E1_u[k] = endpoint_vertex[k]['pv']
        E1_v[k] = endpoint_vertex[k]['qv']

    E2_u = star_syndromes[:, 0]
    E2_v = star_syndromes[:, 1]

    # Species split by color: 0 = red, 1 = green, 2 = blue
    E1_list = [[] for _ in range(3)]
    E2_list = [[] for _ in range(3)]
    for e in range(N_edge):
        c = col_arr[e]
        E1_list[c].append(e)
        E2_list[c].append(e)

    # Adjacency: Gamma1[a][v], Gamma2[b][v]
    Gamma1 = {a: {v: [] for v in V} for a in range(3)}
    Gamma2 = {b: {v: [] for v in V} for b in range(3)}

    for e in range(N_edge):
        c = col_arr[e]

        u1, v1 = E1_u[e], E1_v[e]
        Gamma1[c][u1].append(e)
        Gamma1[c][v1].append(e)

        u2, v2 = E2_u[e], E2_v[e]
        Gamma2[c][u2].append(e)
        Gamma2[c][v2].append(e)

    # Uniform weights (arrays over all edges)
    w1_arr = np.full(N_edge, float(w1))
    w2_arr = np.full(N_edge, float(w2))

    return V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr

def extract_active_edges(x, y1_idx, y2_idx):
    """
    From ILP solution x and index maps y1_idx, y2_idx,
    return sets of active S1 and S2 edges, grouped by species.
    """
    # S1 edges: species a = 0,1,2
    active_S1_by_species = {a: set() for a in range(3)}
    for (a, e), idx in y1_idx.items():
        if x[idx] == 1:
            active_S1_by_species[a].add(e)

    # S2 edges: species b = 0,1,2
    active_S2_by_species = {b: set() for b in range(3)}
    for (b, e), idx in y2_idx.items():
        if x[idx] == 1:
            active_S2_by_species[b].add(e)

    # If you also want the union over species:
    active_S1_all = set().union(*active_S1_by_species.values())
    active_S2_all = set().union(*active_S2_by_species.values())

    return active_S1_by_species, active_S2_by_species, active_S1_all, active_S2_all

class D4_Code:
    def __init__(self, l, encode_x, cn_dict, V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr, env=None, rng=None):
        self.encode_x = encode_x                                       #which logical-x is used as initial state stabilizer (0,1,2) = vertical logical operators of three colors
        self.Nx=3*l
        self.Ny=3*l                                                    #(X,Y)-dimension  '(Nx,Ny) must be multiples of 3'
        self.Nq = 3*self.Nx*self.Ny                                    #number of qubits = number of edges
        self.env = env
        self.rng = rng
        
        HH_star = star_check_matrix((self.Nx,self.Ny))
        self.edge_color_arr = get_edge_color_array((self.Nx,self.Ny))

        self.SS = ground_state_stabilizer((self.Nx,self.Ny),self.edge_color_arr) #star stabilizers tableaux, Nx*Ny stabilizers, last 3 rows are products \Pi_c A_pc
        self.DD = ground_state_destabilizer((self.Nx,self.Ny),HH_star)           #destabilizer tableaux,     Nx*Ny-3 destabilizers
        self.RR = np.zeros(self.SS.shape[0])                                     #sign of the stabilizers,   shape Nx*Ny array, all 0
        self.bL = np.zeros(self.Nx*self.Ny)                                      #triangular stabilizers, |>, Nx*Ny stabilizers
        self.bR = np.zeros(self.Nx*self.Ny)                                      #triangular stabilizers, <|, Nx*Ny stabilizers
        self.LZ = np.zeros(12)                                                   #sign of logical Z operators, all initialized to 0, index: [Rv,Gv,Bv,Rh,Gh,Bh, (Rv,Gv,Bv,Rh,Gh,Bh)']
                                                                                 #The last six primed operators are to be used for updating logical X operators, but they are equivalent to the unprimed counterpart
        self.LX_vec = np.zeros((6,4*self.Nx*self.Ny))                            #dressing of logical X operators, rows: [Rv,Gv,Bv,Rh,Gh,Bh]
        self.LX_sign = 2*np.ones(np.shape(self.LX_vec)[0])
        if self.encode_x.size > 0:
            self.LX_sign[self.encode_x] = 0                                      #set logical X stabilizers to 0, non-existent logical X to 2
        #initialized independently to improve speed
        self.cn_dict, self.V, self.E1_list, self.E2_list, self.Gamma1, self.Gamma2, self.w1_arr, self.w2_arr = cn_dict, V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr
        
        self.X_error_edges = np.zeros(self.Nq, dtype=bool)
        self.Z_error_edges = np.zeros(self.Nq, dtype=bool)
        self.X_correction_edges = np.zeros(self.Nq, dtype=bool)
        self.Z_correction_edges = np.zeros(self.Nq, dtype=bool)
        self.step2_weight = np.ones(self.Nq)
        self.step2_correction_edges = np.zeros(self.Nq, dtype=bool)

        self.T1 = [[],[],[]]
        self.T2 = [[],[],[]]

    def X_errors(self, rate):
        assert 0 <= rate <= 1
        for idx in range(self.Nq):
            if self.rng.random() <= rate:
                self.X_error_edges[idx] = True
                apply_X(idx,self.bL,self.bR,self.SS,self.DD,self.RR,self.LZ,self.LX_vec,self.LX_sign,(self.Nx,self.Ny),self.cn_dict)

    def Z_errors(self, rate):
        assert 0 <= rate <= 1
        for idx in range(self.Nq):
            if self.rng.random() <= rate:
                self.Z_error_edges[idx] = True
                apply_Z(idx,(self.Nx,self.Ny),self.SS,self.RR,self.LX_vec,self.LX_sign,self.cn_dict)
    
    def single_edge_X(self, edge):
        assert 0 <= edge <= 3*self.Nx*self.Ny
        apply_X(edge,self.bL,self.bR,self.SS,self.DD,self.RR,self.LZ,self.LX_vec,self.LX_sign,(self.Nx,self.Ny),self.cn_dict)
    
    def measure_e_anyons(self):
        bL_color = self.cn_dict['bL_vertex_color_arr']
        bR_color = self.cn_dict['bR_vertex_color_arr']
        star_color = self.cn_dict['star_color_arr']
        for idx in range(self.Nx*self.Ny):
            if self.bL[idx] == 1:
                self.T1[bL_color[idx]].append(idx)
            if self.bR[idx] == 1:
                self.T1[bR_color[idx]].append(idx)
            m = self.bL[idx] + self.bR[idx]
            if m==0:
                measurement_output = measure_A(idx,(self.Nx,self.Ny),self.SS,self.DD,self.RR,self.bL,self.bR,self.LZ,self.LX_vec,self.LX_sign,self.cn_dict,rng=self.rng)
                if measurement_output == 5:
                    return 5
                elif measurement_output == 1:
                    self.T2[star_color[idx]].append(idx)
        return (self.T1, self.T2)
    
    def flux_correction(self):
        x, e1, e2 = ILP_decode(self.V, self.E1_list, self.E2_list, self.T1, self.T2, self.Gamma1, self.Gamma2, self.w1_arr, self.w2_arr, env=self.env)
        if x is None:
            raise RuntimeError("ILP_decode failed")

        active_S1_by_species, active_S2_by_species, active_S1, active_S2 = extract_active_edges(x, e1, e2)
        for S1_idx in active_S1:
            apply_X(S1_idx,self.bL,self.bR,self.SS,self.DD,self.RR,self.LZ,self.LX_vec,self.LX_sign,(self.Nx,self.Ny),self.cn_dict)
            self.X_correction_edges[S1_idx] = True
        # for S2_idx in active_S2:
        #     apply_Z(S2_idx,(self.Nx,self.Ny),self.SS,self.RR,self.LX_vec,self.LX_sign,self.cn_dict)
        #     self.Z_correction_edges[S2_idx] = True
        assert np.array_equal(self.bL, np.zeros(self.Nx*self.Ny))
        assert np.array_equal(self.bR, np.zeros(self.Nx*self.Ny))
        return active_S1_by_species, active_S2_by_species, active_S1, active_S2

    def correct_e_anyons(self):
        # measure e syndromes
        a_syndrome = np.zeros(self.Nx*self.Ny)
        for kv in range(self.Nx*self.Ny):
            outcome = measure_A(kv,(self.Nx,self.Ny),self.SS,self.DD,self.RR,self.bL,self.bR,self.LZ,self.LX_vec,self.LX_sign,self.cn_dict,rng=self.rng)
            if outcome == 5:
                return 5
            else:
                a_syndrome[kv]=outcome
        # build the matching graph
        for i in range(self.Nq):
            dc = self.cn_dict['endpoint_vertex'][i]
            pe0 = dc['pe'][0]
            pe1 = dc['pe'][1]
            qe0 = dc['qe'][0]
            qe1 = dc['qe'][1]
            if self.X_correction_edges[pe0] and self.X_correction_edges[pe1]:
                self.step2_weight[i] = 0
            if self.X_correction_edges[qe0] and self.X_correction_edges[qe1]:
                self.step2_weight[i] = 0
        self.e_graph = Matching.from_check_matrix(self.cn_dict['HH_star'], self.step2_weight)
        # correct errors
        try:
            z_correction_locations = np.nonzero(self.e_graph.decode(a_syndrome))[0]        
            for z_edge in z_correction_locations:
                apply_Z(z_edge,(self.Nx,self.Ny),self.SS,self.RR,self.LX_vec,self.LX_sign,self.cn_dict)
                self.step2_correction_edges[z_edge] = True
            #check that all errors are corrected
            assert np.array_equal(self.bL, np.zeros(self.Nx*self.Ny))
            assert np.array_equal(self.bR, np.zeros(self.Nx*self.Ny))
            for site in range(self.Nx*self.Ny):
                outcome = measure_A(site,(self.Nx,self.Ny),self.SS,self.DD,self.RR,self.bL,self.bR,self.LZ,self.LX_vec,self.LX_sign,self.cn_dict,rng=self.rng)
                if outcome != 0:
                    raise ValueError("anyons not corrected")
            return 0
        except ValueError: # No matching found, odd number of e-anyon of any color
            return 3 #return 3 if there are odd number of e-anyons for each color
    
    def decode_X_logicals(self): 
        lx_total_sign = self.LX_sign.copy()

        # If no logical X was initially encoded, nothing to check
        if self.encode_x.size == 0:
            return False

        # iterate over initial x-logicals
        for i in self.encode_x:
            if np.any(self.LX_vec[i,:3*self.Nx*self.Ny]==1) and self.LX_sign[i]<=1: # has Z dressing and active
                # check if current decorated logical-x includes non-contractible z-loop
                x_flipped = np.nonzero((self.cn_dict['HH_log_x']@self.LX_vec[i,:3*self.Nx*self.Ny])%2)[0] # logical Xs that anti-commute with Z dressings
                for ind in x_flipped: # anti-commuting logical X due to non-contractible z-loop
                    if ind in self.encode_x:
                        return 5
                    else: 
                        # non-trivial z-loop does not anticummute with the initial x-logical: so it is separately stabilizer
                        # note lx_total_sign[i]<=1 due to the topmost if
                        lx_total_sign[i] = (lx_total_sign[i]+self.LZ[(ind+3)%6])%2
        lx_out = np.any(lx_total_sign[self.encode_x]>=1) #if any of the initially encoded x-logical failed?
        return lx_out
