import os, sys
import numpy as np
from scipy.sparse import coo_matrix
import gurobipy as gp
from gurobipy import GRB

sys.path.append(os.path.dirname(__file__))
from tableaux import *

def ILP_decode(V_color, E1_list, E2_list, T1_list, T2_list,
               Gamma1, Gamma2, w1, w2, w_site, env=None):
    """Solve the reduced, site-weighted D4 kagome ILP.

    Vertex labels are implicit: ``V == range(len(V_color))``.  At vertex v,
    only S1 colors different from ``V_color[v]`` and the site-color S2 species
    are supported.  ``w_site`` is charged once whenever one or both supported
    S1 midpoint variables are active at a measured-S2 or vacuum site.
    """
    V_color = np.asarray(V_color, dtype=int)
    if V_color.ndim != 1 or np.any((V_color < 0) | (V_color > 2)):
        raise ValueError("V_color must be a one-dimensional array with values in {0,1,2}")
    V = list(range(V_color.size))
    nS1, nS2 = len(E1_list), len(E2_list)
    if nS1 != 3 or nS2 != 3:
        raise ValueError("The D4 kagome formulation requires exactly three colors")

    T1_sets = [set(x) for x in T1_list]
    T2_sets = [set(x) for x in T2_list]
    vertex_set = set(V)
    if any(not x <= vertex_set for x in T1_sets + T2_sets):
        raise ValueError("A terminal list contains an invalid vertex")
    if any(V_color[v] == a for a, terminals in enumerate(T1_sets) for v in terminals):
        raise ValueError("An S1 terminal cannot have the same color as its site")
    if any(V_color[v] != b for b, terminals in enumerate(T2_sets) for v in terminals):
        raise ValueError("An S2 terminal must have the same color as its site")

    S_m = set().union(*T1_sets)
    S_e = set().union(*T2_sets)
    if S_m & S_e:
        raise ValueError("S1 and S2 terminals cannot occupy the same site")
    S_0 = vertex_set - S_m - S_e

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    if np.ndim(w_site) == 0:
        w_site = np.full(len(V), float(w_site))
    else:
        w_site = np.asarray(w_site, dtype=float)
        if w_site.shape != (len(V),):
            raise ValueError("w_site must be a scalar or have one entry per site")

    offset = 0
    y1_idx = {}
    for a in range(3):
        for e in E1_list[a]:
            y1_idx[(a, e)] = offset
            offset += 1

    x1_idx = {}
    for v in V:
        for a in range(3):
            if a != V_color[v]:
                x1_idx[(a, v)] = offset
                offset += 1

    y2_idx = {}
    for b in range(3):
        for e in E2_list[b]:
            y2_idx[(b, e)] = offset
            offset += 1

    x2_idx = {}
    for v in V:
        x2_idx[v] = offset
        offset += 1

    z_e_idx = {}
    for v in sorted(S_e):
        z_e_idx[v] = offset
        offset += 1

    z_0_idx = {}
    for v in sorted(S_0):
        z_0_idx[v] = offset
        offset += 1

    nvar = offset
    c = np.zeros(nvar)
    for key, idx in y1_idx.items():
        c[idx] = w1[key[1]]
    for key, idx in y2_idx.items():
        c[idx] = w2[key[1]]
    for v, idx in z_e_idx.items():
        c[idx] = w_site[v]
    for v, idx in z_0_idx.items():
        c[idx] = w_site[v]

    eq_i, eq_j, eq_val, b_eq = [], [], [], []
    ub_i, ub_j, ub_val, b_ub = [], [], [], []
    row_eq = row_ub = 0

    def add_eq(indices, values, rhs):
        nonlocal row_eq
        eq_i.extend([row_eq] * len(indices))
        eq_j.extend(indices)
        eq_val.extend(float(x) for x in values)
        b_eq.append(float(rhs))
        row_eq += 1

    def add_ub(indices, values, rhs):
        nonlocal row_ub
        ub_i.extend([row_ub] * len(indices))
        ub_j.extend(indices)
        ub_val.extend(float(x) for x in values)
        b_ub.append(float(rhs))
        row_ub += 1

    # S1 degree: sum y1 = 2*x1 + tau1, only for colors != site color.
    for v in V:
        for a in range(3):
            if a == V_color[v]:
                continue
            incident = Gamma1[a].get(v, [])
            add_eq(
                [y1_idx[(a, e)] for e in incident] + [x1_idx[(a, v)]],
                [1.0] * len(incident) + [-2.0],
                int(v in T1_sets[a]),
            )

    # z is the OR of the two supported S1 midpoint variables.
    for vertices, z_map in ((S_e, z_e_idx), (S_0, z_0_idx)):
        for v in sorted(vertices):
            midpoints = [x1_idx[(a, v)] for a in range(3) if a != V_color[v]]
            z = z_map[v]
            add_ub(midpoints + [z], [1.0, 1.0, -2.0], 0.0)
            add_ub(midpoints + [z], [-1.0, -1.0, 1.0], 0.0)

    # Site-color S2 parity at measured-e sites.  If z=0, sum y2=2*x2+1;
    # if z=1, the pair of inequalities is relaxed.
    for v in sorted(S_e):
        b = int(V_color[v])
        incident = Gamma2[b].get(v, [])
        M = float(len(incident))
        z = z_e_idx[v]
        add_ub(
            [y2_idx[(b, e)] for e in incident] + [x2_idx[v], z],
            [1.0] * len(incident) + [-2.0, 1.0 - M],
            1.0,
        )
        add_ub(
            [y2_idx[(b, e)] for e in incident] + [x2_idx[v], z],
            [-1.0] * len(incident) + [2.0, -1.0 - M],
            -1.0,
        )

    # Site-color S2 parity at measured-vacuum sites.  If z=0, sum y2=2*x2.
    for v in sorted(S_0):
        b = int(V_color[v])
        incident = Gamma2[b].get(v, [])
        M = float(len(incident))
        z = z_0_idx[v]
        add_ub(
            [y2_idx[(b, e)] for e in incident] + [x2_idx[v], z],
            [1.0] * len(incident) + [-2.0, -M],
            0.0,
        )
        add_ub(
            [y2_idx[(b, e)] for e in incident] + [x2_idx[v], z],
            [-1.0] * len(incident) + [2.0, -M],
            0.0,
        )

    A_eq = coo_matrix((eq_val, (eq_i, eq_j)), shape=(row_eq, nvar))
    A_ub = coo_matrix((ub_val, (ub_i, ub_j)), shape=(row_ub, nvar))
    model = gp.Model("D4_reduced_site_weighted_ILP", env=env) if env is not None else gp.Model("D4_reduced_site_weighted_ILP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPGap = 0.0
    model.Params.MIPGapAbs = 0.0
    model.Params.TimeLimit = GRB.INFINITY
    model.Params.ConcurrentMIP = 1
    variables = model.addMVar(shape=nvar, vtype=GRB.BINARY, name="x")
    model.setObjective(c @ variables, GRB.MINIMIZE)
    if row_eq:
        model.addMConstr(A_eq, variables, "=", np.asarray(b_eq))
    if row_ub:
        model.addMConstr(A_ub, variables, "<", np.asarray(b_ub))
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        print(f"Solver status: {model.Status} (non-optimal)")
        model.dispose()
        return None, None, None
    solution = np.rint(variables.X).astype(int)
    model.dispose()
    return solution, y1_idx, y2_idx, z_e_idx, z_0_idx


def build_ILP_structure(linear_size, cn_dict, w1, w2, w_site):
    """
    Build ILP data structures from D4-style connectivity.

    linear_size = (Nx, Ny)  # Nx = #columns, Ny = #rows

    cn_dict:
        'edge_color_arr': color of edges, 0-red , 1-green, 2-blue
        'endpoint_vertex': list of length 3*Nx*Ny, each entry is a dict with keys 'pv', 'qv'
        'star_syndromes':  np.array of shape (3*Nx*Ny, 2)

    Returns:
        V_color    : np.array of site colors indexed by vertex
        E1_list    : list of 3 lists; E1_list[a] is list of S1 edge indices of color a
        E2_list    : list of 3 lists; E2_list[b] is list of S2 edge indices of color b
        Gamma1     : dict[a][v] -> list of S1 edge indices incident on vertex v
        Gamma2     : dict[b][v] -> list of S2 edge indices incident on vertex v
        w1_arr     : np.array of shape (3*Nx*Ny,), filled with uniform S1 edge weight w1
        w2_arr     : np.array of shape (3*Nx*Ny,), filled with uniform S2 edge weight w2
        w_site_arr : np.array of shape (Nx*Ny,), containing site penalties
    """
    Nx, Ny = linear_size

    N_vert = Nx * Ny
    N_edge = 3 * Nx * Ny

    V = list(range(N_vert))

    # Colors: col_arr[e] in {0,1,2} for edge index e
    V_color = np.asarray(cn_dict['star_color_arr'], dtype=int).copy()
    if V_color.shape != (N_vert,):
        raise ValueError("cn_dict['star_color_arr'] has the wrong shape")
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

    # Validate the color-support assumptions used by the reduced ILP.
    for v in V:
        site_color = int(V_color[v])
        if Gamma1[site_color][v]:
            raise ValueError("A site has incident S1 links of its own color")
        for b in range(3):
            if b != site_color and Gamma2[b][v]:
                raise ValueError("A site has incident S2 links of a non-site color")

    # Uniform weights (arrays over all edges)
    w1_arr = np.full(N_edge, float(w1))
    w2_arr = np.full(N_edge, float(w2))
    if np.ndim(w_site) == 0:
        w_site_arr = np.full(N_vert, float(w_site))
    else:
        w_site_arr = np.asarray(w_site, dtype=float).copy()
        if w_site_arr.shape != (N_vert,):
            raise ValueError("w_site must be a scalar or have one entry per site")

    return V_color, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr, w_site_arr

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
    def __init__(self, l, encode_x, cn_dict, V_color, E1_list, E2_list, Gamma1, Gamma2,
                 w1_arr, w2_arr, w_site_arr, env=None, rng=None):
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
        self.cn_dict = cn_dict
        self.V_color = np.asarray(V_color, dtype=int).copy()
        self.E1_list, self.E2_list = E1_list, E2_list
        self.Gamma1, self.Gamma2 = Gamma1, Gamma2
        self.w1_arr, self.w2_arr = w1_arr.copy(), w2_arr.copy()
        self.w_site_arr = np.asarray(w_site_arr, dtype=float).copy()
        
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
                # measure e_anyon
                measurement_output = measure_A(idx,(self.Nx,self.Ny),self.SS,self.DD,self.RR,self.bL,self.bR,self.LZ,self.LX_vec,self.LX_sign,self.cn_dict,rng=self.rng)
                if measurement_output == 5:
                    return 5
                elif measurement_output == 1:
                    self.T2[star_color[idx]].append(idx)
        return (self.T1, self.T2)
    
    def flux_correction(self):
        x, e1, e2, _, _ = ILP_decode(
            self.V_color, self.E1_list, self.E2_list, self.T1, self.T2,
            self.Gamma1, self.Gamma2, self.w1_arr, self.w2_arr,
            self.w_site_arr, env=self.env
        )
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


