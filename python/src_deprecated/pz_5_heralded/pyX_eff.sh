#!/bin/bash
#SBATCH --job-name=d4_sim_python          
#SBATCH --account=pi-liangjiang
#SBATCH --output=d4_sim-%J.out           
#SBATCH --error=d4_sim-%J.err           
#SBATCH --time=36:00:00                  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32             
#SBATCH --mem=64GB
#SBATCH --partition=caslake

# ---- Modules / environment ----
module load python/anaconda-2022.05
module load gurobi/11.0
source activate /project/liangjiang/aubreyz/anaconda/topoqc
# source activate pymatch

# Avoid over-subscription from BLAS/MKL using too many threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TOPO_QC='/home/aubreyz/TOPO-QC'


# ---- Go to the directory where you run sbatch ----
cd "$TOPO_QC/pz_5_heralded"

# If your main.py is in ./src/main.py:
python -m src.main_logicalX_eff
