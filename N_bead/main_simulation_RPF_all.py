import sys, time, os, scipy.io
import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_simulation import SimulationModel

rand_seed=int(sys.argv[1])
# fx=float(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx=0.016
dt=0.02
Nbin=int(sys.argv[2]) if len(sys.argv)>2 else 120
stop_time=2000
total_time=stop_time*4
save_freq = 1

save_folder = "model5_macro_chain-v11_c3_temp-bin2_net_100_100_N1920"
from solver_chain_v11 import StressModel as md
save_prex='model_chain_N7/'
model_std = md()
model_std.load_model(Nbin,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')
try:
    sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                              rand_seed=rand_seed,save_freq=save_freq,gamma=170,stop_time=stop_time)
    sim_model.main(model_std,stop_time,tf)
except Exception as e:
    print(save_folder+'   ' + sim_model.save_name  +' Failed')
    print(e)
    sim_model.save_data(model_std.save_folder)


save_folder = "model5_macro_chain-v11_c3_temp-bin2_net_100_100_N1920_chain-v10-VAR_net_100_100"
from solver_chain_v10 import StressModel as md
save_prex='model_chain_N7/'
model = md()
model.load_model(Nbin,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')
try:
    sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                              rand_seed=rand_seed,save_freq=save_freq,gamma=170,stop_time=stop_time)
    sim_model.main(model,stop_time,tf)
except Exception as e:
    print(save_folder+'   ' + sim_model.save_name  +' Failed')
    print(e)
    sim_model.save_data(model.save_folder)





save_folder = "model5_macro_chain-v11_c3_temp-bin2_net_100_100_N1920_chain-v9-PDE_net_100_100"
from solver_chain_v9 import StressModel as md
save_prex='model_chain_N7/'
model = md()
model.load_model(Nbin,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')
try:
    sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                              rand_seed=rand_seed,save_freq=save_freq,gamma=170,stop_time=stop_time)
    ds_dc_star, S0=model_std.get_S0()
    c_init=model.get_c_init()
    sim_model.save['dn']=model.ver+f'-c{model.c_num}'
    sim_model.init_c0(c_init)
    sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

    @tf.function
    def get_pred_data(c):
        ML_H1H2, ML_E, ML_pi_p, S, ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
        
        S, ds_dc=model_std.get_S(c,ds_dc_star, S0)

        I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
        ML_stress=model.kBT*I+ML_pi_p
        return ML_E, ML_H1H2, ML_stress, S, ds_dc
    
    start_time=time.time() 
    for loop in range(sim_model.steps):
        if loop==int(stop_time/sim_model.dt):
            sim_model.reset_fx(0)
        c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
        ML_E, ML_H1H2, ML_stress, S, ds_dc=get_pred_data(c)
        ML_E, ML_H1H2, ML_stress, S, ds_dc=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(), ds_dc.numpy()
        
        if (loop % sim_model.save_step) == 0:
            sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
            if loop % (sim_model.steps//10) == 0:
                print(f"step:{loop}, time:{time.time() - start_time}",flush=True)

        sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)
        sim_model.c_tmp=(sim_model.c_tmp+sim_model.c_tmp.transpose([0,1,3,2]))/2.0

    sim_model.save_data(model.save_folder)
except Exception as e:
    print(save_folder+'   ' + sim_model.save_name  +' Failed')
    print(e)
    sim_model.save_data(model.save_folder)









save_folder = "model5_macro_chain-v11_c3_temp-bin2_net_100_100_N1920_chain-v8-24eign_net_100_100"
from solver_chain_v8 import StressModel as md
save_prex='model_chain_N7/'
model = md()
model.load_model(Nbin,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')
try:
    sim_model=SimulationModel(fx=fx,dt=dt,T=total_time,ES=False,Nbin=Nbin,
                              rand_seed=rand_seed,save_freq=save_freq,gamma=170,stop_time=stop_time)
    ds_dc_star, S0=model_std.get_S0()
    c_init=model.get_c_init()
    sim_model.save['dn']=model.ver+f'-c{model.c_num}'
    sim_model.init_c0(c_init)
    sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

    
    Rc_weight=model.weights['c0_weight']/tf.norm(model.weights['c0_weight'],axis = 1,keepdims=True)/model.scale
    @tf.function
    def get_pred_data(c,c0U):
        U_T_c_U_flat=model.get_c_with_eigen(c, c0U)
            
        ML_stress=model.get_stress(U_T_c_U_flat, c0U)
        ML_E=model.get_ML_E(U_T_c_U_flat, c0U)
        ML_H2=model.get_ML_H2(U_T_c_U_flat, c0U)
        ML_H1=model.get_ML_H1(U_T_c_U_flat, c0U,Rc_weight)
        ML_H1H2=model.kBT*ML_H1-ML_H2
        return ML_E, ML_H1H2, ML_stress

    start_time=time.time() 
    for loop in range(sim_model.steps):
        if loop==int(stop_time/sim_model.dt):
            sim_model.reset_fx(0)

        c_0_gamma, c_0_U = np.linalg.eig(sim_model.c_tmp[:,0])        # (?, 3) unsorted eigenvalues, (?, 3, 3) unsorted eigenvectors
        c0U=tf.constant(c_0_U,dtype=model.tf_dtype)
        c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
        
        ML_E, ML_H1H2, ML_stress=get_pred_data(c,c0U)
        S, ds_dc=model_std.get_S(c,ds_dc_star, S0)

        ML_E, ML_H1H2, ML_stress= ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy()
        S, ds_dc = S.numpy(), ds_dc.numpy()

        if (loop % sim_model.save_step) == 0:
            sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
            if loop % (sim_model.save_step*1000) == 0:
                print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
        sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)
        sim_model.c_tmp=(sim_model.c_tmp+sim_model.c_tmp.transpose([0,1,3,2]))/2.0
        
    sim_model.save_data(model.save_folder)
except Exception as e:
    print(save_folder+'   ' + sim_model.save_name  +' Failed')
    print(e)
    sim_model.save_data(model.save_folder)
