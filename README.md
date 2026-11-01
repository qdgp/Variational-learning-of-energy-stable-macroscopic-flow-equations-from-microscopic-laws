# Variational learning of energy-stable macroscopic flow equations from microscopic laws

Code for constructing and evaluating macroscopic constitutive models of polymeric fluids, including seven-bead chain and star polymers.

The observed working environment is Python 3.7.16, TensorFlow 2.9.1, NumPy 1.21.5, SciPy 1.7.3, h5py 3.7.0, and Matplotlib 3.5.3. Dependencies are recorded in `requirements.txt`. Microscopic trajectories, reference configurations, prepared datasets, and trained checkpoints are supplied separately.

All file paths below are relative to `N_bead/`. Model names in filenames use underscores, corresponding to the paper’s VAR-Law and PDE-Law notation.

## 1. Micro-to-macro learning

This part constructs macroscopic constitutive models from microscopic polymer configurations. Microscopic encoders define generalized conformation tensors,

$$
\mathbf c_i=\left\langle\mathbf b_i(\mathbf r)\right\rangle,
$$

which serve as the resolved structural variables.

**Microscopic training-data preparation**

| File | Description |
|---|---|
| `MD_chain_N7/main_train_set.py`, `MD_star_N7/main_train_set.py` | Generate spatially sampled training ensembles for seven-bead chain and star polymers, saving connector vectors, FENE potential gradients, and polymer stress. |
| `MD_chain_N7/main_train_set_time_series.py` | Generates paired configuration ensembles, stresses, and local velocity-gradient data. These samples provide the time-series estimates of constitutive evolution used to train VAR and PDE. |

**Model definitions**

| File | Description |
|---|---|
| `solver.py` | Shared implementation of microscopic encoders, conformation tensors, encoder derivatives, analytical microscopic closure targets, tensor invariants and bases, thermodynamic-potential evaluation, polymer stress, and checkpoint operations. |
| `solver_chain_VAR_Law.py` | Defines the chain VAR-Law model, including the variational representations of relaxation and flow coupling. Uses a prescribed end-to-end first conformation tensor and learned additional encoders, with averaging over chain reversal. |
| `solver_star_VAR_Law.py` | Defines the star-polymer VAR-Law model, including molecular connectivity, trainable encoders, variational constitutive functions, and averaging over equivalent arm permutations. |
| `solver_chain_VAR.py` | Defines the chain VAR model. Retains the variational relationship between entropy, mobility, relaxation, and stress, with standard upper-convected transport and no additional flow-coupling tensor. |
| `solver_chain_PDE.py` | Defines the chain PDE model. Represents polymer stress and constitutive relaxation independently using invariant features and tensor bases, together with standard upper-convected transport. |
| `solver_chain_PDE_Law.py` | Defines the chain PDE-Law model. Independently represents stress, relaxation contributions, and flow coupling using an eigenframe-based construction, averaging over 24 equivalent frame transformations. |

**Training scripts**

| File | Description |
|---|---|
| `train_VAR_Law_chain.py` | Jointly trains the chain’s trainable encoders and variational constitutive ingredients by matching analytical microscopic relaxation, flow-coupling, and stress targets. Subsequently freezes the encoders, refines the constitutive networks, and performs a flow simulation. |
| `train_VAR_Law_star.py` | Performs the corresponding joint encoder and variational-closure learning for star polymers, followed by constitutive refinement and a flow simulation. |
| `train_VAR.py` | Reuses the encoder from a parent VAR-Law checkpoint. Estimates relaxation targets from paired configurations after subtracting the prescribed transport, then fits the variational relaxation and stress predictions while holding the encoder fixed. |
| `train_PDE.py` | Reuses the parent encoder and constructs trajectory-based relaxation targets. Trains the independently parameterized relaxation and stress functions while holding the encoder fixed. |
| `train_PDE_law.py` | Uses a trained chain VAR-Law encoder to evaluate conformation tensors and analytical microscopic targets. Fits the PDE-Law stress, relaxation contributions, and flow-coupling functions while holding the encoder fixed. |

## 2. Macroscopic flow simulation

This part evolves the coupled macroscopic velocity and conformation fields. Polymer stress enters the incompressible momentum equation, while the local velocity gradient drives the constitutive dynamics.

The simulation scripts consider reverse Poiseuille flow under an imposed body force, followed by relaxation after the forcing is removed.

| File | Description |
|---|---|
| `model_simulation.py` | Shared spatial discretization and time integration for the coupled velocity and conformation equations. Provides constitutive-model interfaces, forcing updates, and collection of velocity, conformation, stress, and thermodynamic diagnostics. |
| `main_simulation_RPF.py` | Loads a selected trained model and simulates reverse Poiseuille flow followed by relaxation after removing the forcing. |
| `main_simulation_RPF_all.py` | Compares the configured chain VAR-Law, VAR, PDE, and PDE-Law models during prolonged forcing and subsequent relaxation. For PDE and PDE-Law trajectories, thermodynamic diagnostics are evaluated using the VAR-Law potential. |
| `main_simulation_RPF_fene_p.py` | Simulates the FENE-P reference model using its analytical stress and relaxation expressions within the shared flow-simulation framework. |
