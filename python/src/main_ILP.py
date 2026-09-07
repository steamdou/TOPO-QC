import multiprocessing as mp
import os, sys, time
import numpy as np
import gurobipy as gp
import argparse
import yaml

sys.path.append(os.path.dirname(__file__))
from tableaux import *
from D4_new import *

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

    l_index, p_index, config = args
    L = config['lattice_sizes']
    px_vals = config['px_vals']
    #[0.1628, 0.1648, 0.1668, 0.1688, 0.1708, 0.1728, 0.1748, 0.1768]
    #[0.1698]
    px = px_vals[p_index]
    pz = config['pz_val']
    stop = config['iterations']

    decoder_type = config['decoder_type']

    if decoder_type == 'P(E)' or decoder_type == 'effective':
        w_site = 0

        if decoder_type == 'effective':
            w1 = config['effective_w1']
            w2 = config['effective_w2']
            
        elif decoder_type == 'P(E)' :
            w1 = np.log(1/px - 1)
            if pz > 0:
                w2 = np.log(1/pz - 1)
            else:
                w2 = 27*L[l_index]*L[l_index]

    if decoder_type == 'conditional':
        w_site = np.log(2)
        
        w1 = np.log(1/px - 1)
        if pz > 0:
            w2 = np.log(1/pz - 1)
        else:
            w2 = 27*L[l_index]*L[l_index]

    
    tot_count = 0
    error_count = 0
    cn_dict = connection_dict((L[l_index]*3,L[l_index]*3))
    V_color, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr, w_site_arr = build_ILP_structure(
        (L[l_index]*3, L[l_index]*3), cn_dict, w1, w2, w_site
    )

    while tot_count < stop:
        tot_count += 1
        code = D4_Code(
            L[l_index], np.array(config['X_stabilizers']), cn_dict, V_color,
            E1_list, E2_list, Gamma1, Gamma2,
            w1_arr, w2_arr, w_site_arr,
            env=_worker_env, rng=_worker_rng
        )
        code.X_errors(px)
        code.Z_errors(pz)
        s = code.measure_e_anyons()

        if len(config['X_stabilizers']) > 0:
            if (np.isscalar(s) and s == 5):
                error_count += 1
            else:
                code.flux_correction()
                output = code.correct_e_anyons()
                if (np.isscalar(output) and output == 5):
                    error_count += 1
                else:
                    lx_out = code.decode_X_logicals()
                    if lx_out==5:
                        error_count += 1
                    elif lx_out:
                        error_count += 1
        else:
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
 
    error_rate = error_count / tot_count
    return l_index, p_index, error_rate, tot_count

# Main function
def main(): # filename: path to config yaml file with program and simulation settings

    parser = argparse.ArgumentParser(
        description="Run decoder program."
    )
    parser.add_argument("config", help="The program YAML config file")

    # ---- start timer ----
    start_time = time.time()

    config_filename = parser.config
    with open(config_filename, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    L = config['lattice_sizes']
    p = config['px_vals']
    stop = config['iterations']

    error_rate = np.zeros((len(L), len(p)))
    counter = np.zeros((len(L), len(p)))

    # Prepare arguments for parallel execution
    args = [(l, i, config) for l in range(len(L)) for i in range(len(p))]

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
    output_file = config['output_path'] + f"/output_{timestamp}.txt"

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

if __name__ == "__main__":
    main()