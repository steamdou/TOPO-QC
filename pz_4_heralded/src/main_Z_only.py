import multiprocessing as mp
import os, sys, time
import numpy as np
import gurobipy as gp

sys.path.append(os.path.dirname(__file__))
from .tableaux import *
from .D4_eff import *

# Global variable to store the Gurobi environment for each worker process
_worker_env = None

# Global variables for random number generator
_worker_rng = None

def init_worker():
    """Initialize each worker process with its own Gurobi environment and NumPy Generator RNG"""
    global _worker_env, _worker_rng

    # Create a separate Gurobi environment for this worker process
    _worker_env = gp.Env()

    # Combine PID and current time (in ms) to get a unique 32-bit seed
    seed = (int(time.time() * 1000) & 0xFFFFFFFF) ^ os.getpid()
    _worker_rng = np.random.default_rng(seed)

def run_simulation(args):
    global _worker_env, _worker_rng

    l_index, p_index, L, p, stop = args
    px = p[p_index]
    pz = 0.04
    w1 = 1
    w2 = 1.1
    #0.09 worse
    #0.18
    
    tot_count = 0
    error_count = 0
    # odd_count = 0
    cn_dict = connection_dict((L[l_index]*3,L[l_index]*3))
    V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr = build_ILP_structure((L[l_index]*3,L[l_index]*3), cn_dict, w1, w2)

    while tot_count < stop:
        tot_count += 1
        code = D4_Code(L[l_index], np.array([]), cn_dict, V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr, env=_worker_env, rng=_worker_rng)
        code.X_errors(px)
        code.Z_errors(pz)
        s = code.measure_e_anyons()
        if (np.isscalar(s) and s == 5):
            raise ValueError('collapsed X logical')
        else:
            code.flux_correction()
            output = code.correct_e_anyons()
            if output == 3:
                # return 3 if there are odd number of e-anyons for any color
                # odd_count += 1
                error_count += 1
            elif output == 5:
                raise ValueError('collapsed X logical')
            elif output == 0:
                if not np.array_equal(code.LZ, np.zeros(12)):
                    error_count += 1
            else:
                raise ValueError('unexpected output')
    # print(odd_count, error_count)
    error_rate = error_count / tot_count
    return l_index, p_index, error_rate, tot_count

# Main script
if __name__ == "__main__":
    # ---- start timer ----
    start_time = time.time()

    L = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]
    p = [0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
    #[0.158, 0.16, 0.162, 0.164, 0.166, 0.168, 0.17, 0.172]
    #[0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    # [0.143, 0.146, 0.15, 0.154, 0.158, 0.162]
    #[0.152, 0.154, 0.156, 0.158, 0.16, 0.162, 0.164, 0.166]
    #[0.14, 0.142, 0.144, 0.146, 0.148, 0.15, 0.152, 0.144]
    #[0.073, 0.083, 0.093, 0.103, 0.113, 0.123, 0.133, 0.143]
    stop = 8000

    error_rate = np.zeros((len(L), len(p)))
    counter = np.zeros((len(L), len(p)))

    # Prepare arguments for parallel execution
    args = [(l, i, L, p, stop) for l in range(len(L)) for i in range(len(p))]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=32, initializer=init_worker) as pool:
        results = pool.map(run_simulation, args)

    # Collect results
    for l, i, rate, count in results:
        error_rate[l][i] = rate
        counter[l][i] = count

    # ---- stop timer ----
    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_hr = elapsed_sec / 3600.0

    # -------- Save to txt file --------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"/project/liangjiang/aubreyz/pz_logicalZs/ILP_eff_Z/pz_4_heralded_Z/output1.1_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write("Simulation parameters:\n")
        f.write(f"L = {L}\n")
        f.write(f"p = {p}\n")
        f.write(f"stop = {stop}\n\n")

        f.write("Error Rate:\n")
        np.savetxt(f, error_rate, fmt="%.6f")
        f.write("\nCounter:\n")
        np.savetxt(f, counter, fmt="%d")

        f.write(f"\nTotal runtime: {elapsed_sec:.2f} seconds ({elapsed_hr:.3f} hours)\n")