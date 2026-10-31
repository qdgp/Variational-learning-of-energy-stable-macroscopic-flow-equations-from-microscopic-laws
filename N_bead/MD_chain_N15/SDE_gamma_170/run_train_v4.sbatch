#!/bin/bash --login
#SBATCH --job-name=fene_gdot
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32G
#SBATCH --time=32:00:00
#SBATCH --array=0-199
#SBATCH --account=data-machine

set -euo pipefail

module purge
module load MATLAB/2025b

cd '/mnt/research/MultiscaleML_group/PeiGe/DeePN_entropy/N_bead_github/N_bead/MD_chain_N15/SDE_gamma_170'


task=${SLURM_ARRAY_TASK_ID}
gdotIdx=$(( task % 40 + 1 ))          # 1..40
edotIdx=$(( task / 40 ))      # 0..4

gammaDot=$(awk -v a="${gdotIdx}" 'BEGIN{printf "%.6e", a*1e-5}')

# Ensure results folder exists (MAT files will go here)
OUTDIR="/mnt/research/MultiscaleML_group/PeiGe/DeePN_entropy/N_bead_github/N_bead/MD_chain_N15/SDE_gamma_170/results_v4/set_${edotIdx}"
mkdir -p "${OUTDIR}"


echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${task}"
echo "gammaDot=${gammaDot}"
echo "Working dir: $(pwd)"
echo "Saving MAT files to: $(pwd)/results_v4"

# Run MATLAB (expects fene_elongation_fixed_gamma.m in this directory)
matlab -batch "try, addpath('/mnt/research/MultiscaleML_group/PeiGe/DeePN_entropy/N_bead_github/N_bead/MD_chain_N15/SDE_gamma_170'); fene_elongation_fixed_gamma(${gammaDot}, ${edotIdx}, '${OUTDIR}'); catch ME, disp(getReport(ME,'extended')); exit(1); end"

