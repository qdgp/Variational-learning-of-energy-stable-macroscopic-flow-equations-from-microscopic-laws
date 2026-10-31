# Variational learning of energy-stable macroscopic flow equations from microscopic laws

Code for the DeePN project: training and simulating macroscopic polymer-flow models from microscopic chain and star configurations.

This repository contains source code only. Jupyter notebooks, training datasets, trained-model checkpoints, simulation results, figures, and machine-specific validation reports are excluded.

## Code layout

- `N_bead/solver*.py`: microscopic encoders and macroscopic constitutive models.
- `N_bead/train_*.py`: regular, stochastic, and comparison-model training.
- `N_bead/main_simulation_*.py` and `model_simulation*.py`: flow and stochastic simulations.
- `N_bead/main_test_*.py` and `model_test_set.py`: microscopic/model comparisons.
- `N_bead/md_data.py`: regular MD preparation routines, including periodic molecular COM reconstruction.
- `N_bead/MD_chain_N7/` and `N_bead/MD_star_N7/`: Python training/macro data generators and LAMMPS launch inputs.
- `N_bead/MD_chain_N7/SDE_data_set_v2/` and `N_bead/MD_chain_N15/SDE_gamma_170/`: microscopic SDE simulation and training-set assembly in MATLAB.
- `N_bead/data_io.py`: single-precision MAT output.

## Environment and inputs

The observed working Python environment is Python 3.7.16, TensorFlow 2.9.1, NumPy 1.21.5, SciPy 1.7.3, h5py 3.7.0, and Matplotlib 3.5.3. See `requirements.txt`; it records the tested environment, rather than a newly resolved dependency lock. The Python dependencies can be installed in a suitable environment with:

```bash
python -m pip install -r requirements.txt
```

MATLAB and the original custom LAMMPS executable are required only for the corresponding microscopic simulation workflows.

Training data, microscopic trajectories, reference ensembles, and pretrained weights must be supplied separately. The scripts currently contain the author's explicit HPC paths and Slurm environment setup. Edit the input roots in `N_bead/load_set.py`, individual scripts, and the `run_*` files for your installation. Simulation/model paths are relative to `N_bead/`.

## Regular training and macro data generation

On the author's HPC system:

```bash
cd N_bead
sbatch run_generate_sets.sh
```

This submits a Slurm array with one explicit generation case per task:

| Task | Case |
| --- | --- |
| 0–2 | Chain training at kBT 0.22, 0.25, 0.30 |
| 3 | Star training |
| 4–6 | Chain time-series training at the three temperatures |
| 7–8 | Chain T800 macro data with 20 / 40 bins |
| 9 | Chain T2000 macro data, 20 bins, symmetry extension |
| 10–11 | Star macro data with 20 / 40 bins |

Examples of individual generators:

```bash
python MD_chain_N7/main_train_set.py /path/to/chain/dumps --start 1200000 --stop 2200000 --seed 0
python MD_chain_N7/main_train_set_time_series.py /path/to/chain/dumps --start 1200000 --stop 2200000 --seed 0
python MD_star_N7/main_train_set.py /path/to/star/dumps --start 0 --stop 1000000 --seed 0
python MD_chain_N7/main_macro_set.py /path/to/chain/dumps --nbin 40
python MD_star_N7/main_macro_set.py /path/to/star/dumps --nbin 40
```

The regular generators assume valid input and write final HDF5 files directly. Existing output files are overwritten. They retain `.mat` filenames for the existing Python readers; these files are read with h5py. Floating-point outputs are float32, while counters and MD steps remain integer.

Velocity loading is shared in `load_set.load_velocity`. The existing 20-step velocity averages are dated at their sampling-window centers (first center 0.021 for MD timestep 0.002 and record interval 0.04). The three-temperature data are used for enhanced sampling. Historical macro `c0` diagnostics are retained separately from the learned conformation definitions.

## Model training and simulations

The explicit Slurm templates are `N_bead/run_gpu_train`, `run_cpu_sim`, `run_cpu_random`, and `run_gpu_test`. They include active commands and commented alternatives. Select the intended model/seed and configure the paths/resources before submitting. No checkpoints or datasets are downloaded automatically.
