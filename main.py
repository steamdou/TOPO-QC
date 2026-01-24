from multiprocessing import Pool
import os, sys, time
import numpy as np

sys.path.append(os.path.dirname(__file__))
from tableaux import *
from D4 import *

def init_worker():
    # Combine PID and current time (in ms) to get a unique 32-bit seed
    seed = (int(time.time() * 1000) & 0xFFFFFFFF) ^ os.getpid()
    np.random.seed(seed)

def run_simulation(args):
    l_index, p_index, L, p, stop = args
    px = p[p_index]
    pz = 0.03
    w1 = np.log(1/px - 1)
    w2 = np.log(1/pz - 1)
    
    tot_count = 0
    error_count = 0
    flux_count = 0
    charge_count = 0
    cn_dict = connection_dict((L[l_index]*3,L[l_index]*3))
    V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr = build_ILP_structure((L[l_index]*3,L[l_index]*3), cn_dict, w1, w2)

    while tot_count < stop:
        tot_count += 1
        code = D4_Code(L[l_index], np.array([0,1,2]), cn_dict, V, E1_list, E2_list, Gamma1, Gamma2, w1_arr, w2_arr)
        code.X_errors(px)
        code.Z_errors(pz)
        s = code.measure_e_anyons()
        if (np.isscalar(s) and s == 5):
            error_count += 1
            flux_count += 1
        else:
            code.flux_correction()
            output = code.correct_e_anyons()
            if (np.isscalar(output) and output == 5):
                error_count += 1
                flux_count += 1
            else:
                lx_out = code.decode_X_logicals()
                if lx_out==5 or (code.LZ[0] != 0) or (code.LZ[1] != 0) or (code.LZ[2] != 0):
                    error_count += 1
                    flux_count += 1
                elif lx_out:
                    error_count += 1
                    charge_count += 1
 
    error_rate = error_count / tot_count
    return l_index, p_index, error_rate, tot_count, flux_count/tot_count, charge_count/tot_count

# Main script
if __name__ == "__main__":
    # ---- start timer ----
    start_time = time.time()

    L = [4,4,4,4,
         5,5,5,5,
         6,6,6,6,
         7,7,7,7]
    p = [0.134, 0.136, 0.138, 0.14, 0.142, 0.144, 0.146, 0.148]
    stop = 10000

    error_rate = np.zeros((len(L), len(p)))
    counter = np.zeros((len(L), len(p)))
    flux_count = np.zeros((len(L), len(p)))
    charge_count = np.zeros((len(L), len(p)))

    # Prepare arguments for parallel execution
    args = [(l, i, L, p, stop) for l in range(len(L)) for i in range(len(p))]

    with Pool(processes=32, initializer=init_worker) as pool:
        results = pool.map(run_simulation, args)

    # Collect results
    for l, i, rate, count, flux, charge in results:
        error_rate[l][i] = rate
        counter[l][i] = count
        flux_count[l][i] = flux
        charge_count[l][i] = charge

    # ---- stop timer ----
    end_time = time.time()
    elapsed_sec = end_time - start_time
    elapsed_hr = elapsed_sec / 3600.0

    # -------- Save to txt file --------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"../output_{timestamp}.txt"

    with open(output_file, "w") as f:
        f.write("Simulation parameters:\n")
        f.write(f"L = {L}\n")
        f.write(f"p = {p}\n")
        f.write(f"stop = {stop}\n\n")

        f.write("Error Rate:\n")
        np.savetxt(f, error_rate, fmt="%.6f")
        f.write("\nFlux Rate:\n")
        np.savetxt(f, flux_count, fmt="%.6f")
        f.write("\nCharge Rate:\n")
        np.savetxt(f, charge_count, fmt="%.6f")
        f.write("\nCounter:\n")
        np.savetxt(f, counter, fmt="%d")

        f.write(f"\nTotal runtime: {elapsed_sec:.2f} seconds ({elapsed_hr:.3f} hours)\n")
