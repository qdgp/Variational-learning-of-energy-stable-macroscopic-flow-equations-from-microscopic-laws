import os, sys, time, h5py, scipy.io
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from z_fun import get_solver
import numpy as np
import tensorflow as tf
from model_simulation import SimulationModel
from load_set import load
from model_test_set import TestModel 

save_folder, rand_seed, data_set=sys.argv[1], int(sys.argv[2]), sys.argv[3]
md,save_prex=get_solver(save_folder)
model = md()
model.load_model(1,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')

xr, ff, stress, Np, data_set_name=load.load(save_folder,data_set)
model.batch_size,model.Np=50,Np
test_model=TestModel.main_train_micro(model,xr,ff,tf)

c_num=model.c_num
net_size=[100,100]
xr_ref=load.init('chain')
from solver_chain_v8 import StressModel as md_eigen
config={'c_num':c_num, 'batch_size':10,
        'E':net_size,'stress':net_size,'H2':net_size,'H1':net_size,
        'save_folder':save_prex+save_folder+'_'+md_eigen.ver+f'_net_{"_".join(map(str,net_size))}'}
model_eigen=md_eigen(Np)
print(config['save_folder'])

model_eigen.init_for_training(config,seed=rand_seed,stddev=0.01)
if not os.path.exists(model_eigen.save_folder):
    os.makedirs(model_eigen.save_folder,exist_ok=True)
model_eigen.init_rr_ref(xr_ref)

bs=20
model_eigen.batch_size,model_eigen.Np=bs,Np

c0=test_model.save['c'][:,0]
direction100 = np.array([1, 0, 0])                                              # (3,) any direction
c_0_gamma, c_0_U = np.linalg.eig(c0)                                           # (?, 3) unsorted eigenvalues, (?, 3, 3) unsorted eigenvectors
idx = (c_0_gamma).argsort()[:, : : -1]                                          # (?, 3) 0, 1, 2
for i in range(idx.shape[0]):
    c_0_gamma[i, :] = c_0_gamma[i, idx[i]]                                      # (3,) sort eigenvalues (all positive) from large to small
    c_0_U[i, :, :] = c_0_U[i, :, :].T[idx[i], :].T                              # (3, 3) sort eigenvectors columns respectively
    c_0_U[i, :, :] = np.sign(direction100 @ c_0_U[i, :, :]) * c_0_U[i, :, :]    # (3, 3) choose directions


macro_tf={'c':tf.constant(test_model.save['c'],dtype=model.tf_dtype),
          'c0U':tf.constant(c_0_U,dtype=model.tf_dtype),
          'MD_H1':tf.constant(test_model.save['MD_H1'],dtype=model.tf_dtype),
          'MD_H2':tf.constant(test_model.save['MD_H2'],dtype=model.tf_dtype),
          'MD_E':tf.constant(test_model.save['MD_E'],dtype=model.tf_dtype),
          'stress':tf.constant(stress,dtype=model.tf_dtype)}

model_eigen.batch_size=test_model.save['c'].shape[0]

Rc_weight=model.Rc_weight
model_eigen.weights.pop('c0_weight', None)
model_eigen.weights.pop('weight_pos', None)

lr=tf.keras.optimizers.schedules.ExponentialDecay(1e-2, 20, 0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)

@tf.function
def run_optimization(macro_tf):
    with tf.GradientTape() as tape:  
        U_T_c_U_flat=model_eigen.get_c_with_eigen(macro_tf['c'], macro_tf['c0U'])
        
        ML_stress=model_eigen.get_stress(U_T_c_U_flat, macro_tf['c0U'])
        ML_E=model_eigen.get_ML_E(U_T_c_U_flat, macro_tf['c0U'])
        ML_H2=model_eigen.get_ML_H2(U_T_c_U_flat, macro_tf['c0U'])
        ML_H1=model_eigen.get_ML_H1(U_T_c_U_flat, macro_tf['c0U'],Rc_weight)

        H1_loss = tf.norm(ML_H1 - macro_tf['MD_H1'])
        H2_loss = tf.norm(ML_H2 - macro_tf['MD_H2'])
        E_loss = tf.norm(ML_E - macro_tf['MD_E'])
        stress_loss = tf.norm(ML_stress - macro_tf['stress'])
        loss = 0.1*H1_loss + 0.1*H2_loss + 0.6* stress_loss + 0.2*E_loss

    stress_rmse=stress_loss/tf.norm(macro_tf['stress'])
    H1_rmse=H1_loss/tf.norm(macro_tf['MD_H1'])
    H2_rmse=H2_loss/tf.norm(macro_tf['MD_H2'])
    E_rmse=E_loss/tf.norm(macro_tf['MD_E'])

    trainable_variables = list(model_eigen.weights.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, H1_loss, H1_rmse, H2_loss, H2_rmse, E_loss, E_rmse, stress_loss, stress_rmse

for step in range(1000+1):
    loss, H1_loss, H1_rmse, H2_loss, H2_rmse, E_loss, E_rmse, stress_loss, stress_rmse = run_optimization(macro_tf)
    if step % 10 == 0:
        print("step: %4u, loss: %.3f, H1: %.3f(%.3f), H2: %.3f(%.3f), E: %.3f(%.3f), stress: %.3f(%.3f)" %(step, loss, H1_loss, H1_rmse, H2_loss, H2_rmse, E_loss, E_rmse, stress_loss, stress_rmse),flush=True)

model_eigen.weights['c0_weight'] = Rc_weight
model_eigen.save_var['loss']=np.array([E_loss, E_rmse, stress_loss, stress_rmse, H1_loss+H2_loss, (H1_rmse+H2_rmse)/2])
model_eigen.record_save(f'train_result_seed_{rand_seed}.mat')

Nbin=20
model_eigen.batch_size=Nbin
sim_model=SimulationModel(fx=0.016,dt=0.2,T=3600,gamma=185, ES=False,Nbin=Nbin,rand_seed=rand_seed,stop_time=1200)
sim_model.main_eigen(model_eigen,1200,tf)

sim_model=SimulationModel(fx=0.016,dt=0.2,T=3600,gamma=170, ES=False,Nbin=Nbin,rand_seed=rand_seed,stop_time=1200)
sim_model.main_eigen(model_eigen,1200,tf)