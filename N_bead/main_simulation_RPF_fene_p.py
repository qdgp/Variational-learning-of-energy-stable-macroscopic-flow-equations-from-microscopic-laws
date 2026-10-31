import numpy as np
from model_simulation import SimulationModel
import time, sys
import scipy.io

fx=0.016
dt=0.02
Nbin=int(sys.argv[1]) if len(sys.argv)>1 else 120
stop_time=800
total_time=stop_time*4

save_folder='model_chain_N7/model_fenep'
sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                            rand_seed=1,save_freq=1,gamma=170,stop_time=stop_time)
b=35
kBT=0.25

I=np.reshape(np.eye(3),[1,1,3,3])
I=np.tile(I, [sim_model.Nbin,1,1,1])
sim_model.init_c0(np.reshape(np.eye(3),[1,1,3,3]))
print(sim_model.c_tmp[0,0,:,:])

def get_pred_data(c):
    trc=np.trace(c,axis1=2,axis2=3).reshape((sim_model.Nbin,1,1,1))
    detc=np.linalg.det(c).reshape((sim_model.Nbin,1,1,1))
    h_trc=b/(b+3-trc); 
    S= - 0.5*kBT*(-b*np.log(h_trc)+np.log(detc)).reshape((sim_model.Nbin,1))
    stress=(kBT*h_trc*c).reshape((sim_model.Nbin,3,3))
    ML_H1H2=2*kBT*(I-h_trc*c)*((1.0/1.1575)**2)
    ds_dc=0.5*kBT*(np.linalg.inv(c)-h_trc*I)
    ML_E=np.zeros((sim_model.Nbin,1,9,9))
    return ML_E, ML_H1H2, stress, S, ds_dc

ML_E, ML_H1H2, stress, S, ds_dc=get_pred_data(sim_model.c_tmp)
sim_model.init_ES(S,sim_model.ES)

start_time=time.time() 
for loop in range(sim_model.steps):
    if loop==int(stop_time/sim_model.dt):
        sim_model.reset_fx(0)
    ML_E, ML_H1H2, stress, S, ds_dc=get_pred_data(sim_model.c_tmp)
    

    if (loop % sim_model.save_step) == 0:
        sim_model.collect_data(ML_E,ML_H1H2,stress,S)
        if loop % (sim_model.steps//10) == 0:
            print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
    
    sim_model.one_step(ML_E,ML_H1H2,stress,S)
    sim_model.c_tmp=(sim_model.c_tmp+sim_model.c_tmp.transpose([0,1,3,2]))/2.0

sim_model.save_data(save_folder)
