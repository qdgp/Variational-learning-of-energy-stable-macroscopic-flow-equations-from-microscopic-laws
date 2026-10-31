#!/bin/bash --login
#SBATCH --job-name=DeePN_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=multiscaleml
#SBATCH --array=0-11
#SBATCH --output=slurm-%x-%A_%a.out

set -e
ml purge
source /mnt/ufs18/nodr/home/gepei/anaconda3/etc/profile.d/conda.sh
conda activate vscode_gpu
n=$SLURM_ARRAY_TASK_ID
cd /mnt/research/MultiscaleML_group/PeiGe/DeePN_entropy/N_bead_github/N_bead
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

chain_raw=/mnt/gs21/scratch/gepei/DeePN_chain
star_raw=/mnt/gs21/scratch/gepei/DeePN_star

# One independent generation case per array task. Outputs are written directly.

# 0: Chain training, kBT 0.22
if [ "$n" -eq 0 ]; then
    python3 MD_chain_N7/main_train_set.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.22" --start 1200000 --stop 2200000 --seed 0
fi

# 1: Chain training, kBT 0.25
if [ "$n" -eq 1 ]; then
    python3 MD_chain_N7/main_train_set.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.25" --start 1200000 --stop 2200000 --seed 0
fi

# 2: Chain training, kBT 0.30
if [ "$n" -eq 2 ]; then
    python3 MD_chain_N7/main_train_set.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.30" --start 1200000 --stop 2200000 --seed 0
fi

# 3: Star training
if [ "$n" -eq 3 ]; then
    python3 MD_star_N7/main_train_set.py "$star_raw/box_60_60_100_fx_0.016" --start 0 --stop 1000000 --seed 0
fi

# 4: Chain time series, kBT 0.22
if [ "$n" -eq 4 ]; then
    python3 MD_chain_N7/main_train_set_time_series.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.22" --start 1200000 --stop 2200000 --seed 0
fi

# 5: Chain time series, kBT 0.25
if [ "$n" -eq 5 ]; then
    python3 MD_chain_N7/main_train_set_time_series.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.25" --start 1200000 --stop 2200000 --seed 0
fi

# 6: Chain time series, kBT 0.30
if [ "$n" -eq 6 ]; then
    python3 MD_chain_N7/main_train_set_time_series.py "$chain_raw/box_60_60_100_fx_0.02_T_800_kBT_0.30" --start 1200000 --stop 2200000 --seed 0
fi

# 7: Chain macro, T800, 20 bins
if [ "$n" -eq 7 ]; then
    python3 MD_chain_N7/main_macro_set.py "$chain_raw/box_60_60_100_fx_0.016_T_800" --nbin 20 --start 400000 --stop 1400000
fi

# 8: Chain macro, T800, 40 bins
if [ "$n" -eq 8 ]; then
    python3 MD_chain_N7/main_macro_set.py "$chain_raw/box_60_60_100_fx_0.016_T_800" --nbin 40 --start 400000 --stop 1400000
fi

# 9: Chain macro, T2000, 20 bins, symmetry extension
if [ "$n" -eq 9 ]; then
    python3 MD_chain_N7/main_macro_set.py "$chain_raw/box_60_60_100_fx_0.016_T_2000" --nbin 20 --start 0 --stop 2600000 --step 1000 --extend
fi

# 10: Star macro, 20 bins
if [ "$n" -eq 10 ]; then
    python3 MD_star_N7/main_macro_set.py "$star_raw/box_60_60_100_fx_0.016" --nbin 20 --start 0 --stop 1000000 --step 2000
fi

# 11: Star macro, 40 bins
if [ "$n" -eq 11 ]; then
    python3 MD_star_N7/main_macro_set.py "$star_raw/box_60_60_100_fx_0.016" --nbin 40 --start 0 --stop 1000000 --step 2000
fi
