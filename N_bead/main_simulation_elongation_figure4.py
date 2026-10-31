import sys, time, os, scipy.io
import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_simulation import SimulationModel

rand_seed=int(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dt=0.2
Nbin=40
stop_time = 40000
total_time = 120000

save_folder1 = "model8_macro_chain-v17_c2_sde-v2-bin32_net_40_40_N1600"
save_folder2 = "model8_macro_chain-v17_c2_sde-v2-bin32_net_40_40_N1600_chain-v8-24eign_net_100_100"

print(save_folder1+' start')
md,save_prex=get_solver(save_folder1)
model_std = md()
model_std.load_model(Nbin,save_prex+save_folder1,f'train_result_seed_{rand_seed}.mat')

sim_model=SimulationModel(fx=0.01,dt=dt,T=total_time,ES=False,Nbin=Nbin,rand_seed=rand_seed,save_freq=20,gamma=170)
sim_model.save_name=f'elongation_{rand_seed}_dt_{sim_model.dt}_figure4.mat'
e_rate=5e-5*np.arange(Nbin)

ds_dc_star, S0=model_std.get_S0()
c_init=model_std.get_c_init()
sim_model.save['dn']=model_std.ver+f'-c{model_std.c_num}'
sim_model.init_c0(c_init)
sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

sim_model.save['elongation_rate']=e_rate
sim_model.kappa[:,0,0]=sim_model.save['elongation_rate']
sim_model.kappa[:,1,1]=-sim_model.save['elongation_rate']

@tf.function
def get_pred_data(c):
    ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model_std.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
    I=tf.eye(3, batch_shape = [model_std.batch_size],dtype=tf.float32)
    ML_stress=model_std.kBT*I+ML_pi_p
    return ML_E, ML_H1H2, ML_stress, S, ds_dc

start_time=time.time() 
for loop in range(sim_model.steps):
    if loop==int(stop_time/sim_model.dt):
        sim_model.kappa=np.zeros([sim_model.Nbin,3,3])
    c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
    ML_E, ML_H1H2, ML_stress, S, ds_dc=get_pred_data(c)
    ML_E, ML_H1H2, ML_stress, S, ds_dc=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(), ds_dc.numpy()

    dc=ML_H1H2/sim_model.gamma
    dc_advection=np.reshape(sim_model.kappa,[sim_model.Nbin,1,3,3]) @ sim_model.c_tmp
    dc_source = np.reshape(np.reshape(sim_model.kappa,[sim_model.Nbin,1,1,9]) @ ML_E,[sim_model.Nbin,sim_model.c_num,3,3]) 
    dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 

    if (loop % sim_model.save_step) == 0:
        sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
        if loop % (sim_model.save_step*200) == 0:
            print(f"T:{loop*sim_model.dt}, time:{time.time() - start_time}",flush=True)

    sim_model.c_tmp=sim_model.c_tmp+dc*sim_model.dt

sim_model.save_data(model_std.save_folder)







# print(save_folder2+' start')
# md,save_prex=get_solver(save_folder2)
# model = md()
# model.load_model(Nbin,save_prex+save_folder2,f'train_result_seed_{rand_seed}.mat')

# sim_model=SimulationModel(fx=0.01,dt=dt,T=total_time,ES=False,Nbin=Nbin,rand_seed=rand_seed,save_freq=20,gamma=170)
# sim_model.save_name=f'elongation_{rand_seed}_dt_{sim_model.dt}_figure4.mat'
# ds_dc_star, S0=model_std.get_S0()
# c_init=model.get_c_init()
# sim_model.save['dn']=model.ver+f'-c{model.c_num}'
# sim_model.init_c0(c_init)
# sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

# Rc_weight=model.weights['c0_weight']/tf.norm(model.weights['c0_weight'],axis = 1,keepdims=True)/model.scale
# sim_model.save['elongation_rate']=e_rate
# sim_model.kappa[:,0,0]=sim_model.save['elongation_rate']
# sim_model.kappa[:,1,1]=-sim_model.save['elongation_rate']

# @tf.function
# def get_pred_data(c,c0U):
#     U_T_c_U_flat=model.get_c_with_eigen(c, c0U)
        
#     ML_stress=model.get_stress(U_T_c_U_flat, c0U)
#     ML_E=model.get_ML_E(U_T_c_U_flat, c0U)
#     ML_H2=model.get_ML_H2(U_T_c_U_flat, c0U)
#     ML_H1=model.get_ML_H1(U_T_c_U_flat, c0U, Rc_weight)
#     ML_H1H2=model.kBT*ML_H1-ML_H2
#     return ML_E, ML_H1H2, ML_stress


# try:
#     start_time=time.time() 
#     for loop in range(sim_model.steps):
#         if loop==int(stop_time/sim_model.dt):
#             sim_model.kappa=np.zeros([sim_model.Nbin,3,3])

#         c_0_gamma, c_0_U = np.linalg.eig(sim_model.c_tmp[:,0])
#         c0U=tf.constant(c_0_U,dtype=model.tf_dtype)
#         c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
        
#         ML_E, ML_H1H2, ML_stress=get_pred_data(c,c0U)
#         S, ds_dc=model_std.get_S(c,ds_dc_star, S0)

#         ML_E, ML_H1H2, ML_stress= ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy()
#         S, ds_dc = S.numpy(), ds_dc.numpy()

#         if (loop % sim_model.save_step) == 0:
#             sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
#             if loop % (sim_model.save_step*200) == 0:
#                 sim_model.save_data(model.save_folder)
#                 print(f"T:{loop*sim_model.dt}, time:{time.time() - start_time}",flush=True)

#         dc=ML_H1H2/sim_model.gamma
#         dc_advection=np.reshape(sim_model.kappa,[sim_model.Nbin,1,3,3]) @ sim_model.c_tmp
#         dc_source = np.reshape(np.reshape(sim_model.kappa,[sim_model.Nbin,1,1,9]) @ ML_E,[sim_model.Nbin,sim_model.c_num,3,3]) 
#         dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 
#         sim_model.c_tmp=sim_model.c_tmp+dc*sim_model.dt

#     sim_model.save_data(model.save_folder)
# except Exception as e:
#     print(save_folder2+'   ' + sim_model.save_name  +' Failed')
#     print(e)
#     sim_model.save_data(model.save_folder)