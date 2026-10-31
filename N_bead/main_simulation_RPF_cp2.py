import sys, time, os, scipy.io
import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_simulation import SimulationModel

rand_seed=int(sys.argv[1])
save_folder=sys.argv[2]
md,save_prex=get_solver(save_folder)

fx=0.016
dt=0.02
Nbin=60
stop_time = 800
# gamma = 170


model = md()
model.load_model(Nbin,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')

for gamma in [170, 175, 180]:
    total_time=stop_time*4
    sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                                rand_seed=rand_seed,save_freq=8,gamma=gamma,stop_time=stop_time)
    sim_model.main(model,stop_time,tf)

