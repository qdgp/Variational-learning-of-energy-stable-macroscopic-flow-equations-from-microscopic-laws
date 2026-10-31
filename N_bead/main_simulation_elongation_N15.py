import sys, time, h5py, os, scipy.io
import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_simulation import SimulationModel
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rand_seed=int(sys.argv[1])
save_folder=sys.argv[2]
md,save_prex=get_solver(save_folder)

Nbin=20
T=120000
model = md()
model.load_model(Nbin,save_prex+save_folder,f'train_result_seed_{rand_seed}.mat')

gamma = 86
e_rate=5e-5*np.arange(1,Nbin+1)
if 'gamma-170' in save_folder:
    gamma = 170
    e_rate=3e-5*np.arange(Nbin)
if 'gamma-40' in save_folder:
    gamma = 40
    e_rate=1e-4*np.arange(Nbin)
    
sim_model=SimulationModel(fx=0.01,dt=0.02,T=T,ES=False,Nbin=Nbin,rand_seed=rand_seed,save_freq=20,gamma=gamma)
sim_model.save_name=f'elongation_{rand_seed}_T_{T}_dt_{sim_model.dt}_gamma_{sim_model.gamma}.mat'
sim_model.main_elongation(model,tf,e_rate)
sim_model.save_data(model.save_folder)