from data_io import savemat
import sys, time, os, scipy.io
import numpy as np
import tensorflow as tf

rand_seed=4
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

save_folder = "model5_macro_chain-VAR_Law_c3_temp-bin2_net_100_100_N1920_chain-VAR_net_100_100"
from solver_chain_VAR import StressModel as md
save_prex='model_chain_N7/'
model = md()
model.load_model(1,save_prex+'/'+save_folder,f'train_result_seed_{rand_seed}.mat')


Folder,Name='model_chain_N7/',f'/sim_data_{rand_seed}_0.016_dt_0.02_Nbin_60_stop_2000_gamma_170.mat'
VAR_DeePN_Folder=Folder+'model5_macro_chain-VAR_Law_c3_temp-bin2_net_100_100_N1920'
VAR=scipy.io.loadmat(VAR_DeePN_Folder+'_chain-VAR_net_100_100'+Name)
c_tmp = VAR['c'][-1,[1]]
print(c_tmp.shape)
ds_dc_star, S0=model.get_S0()
@tf.function
def get_pred_data(c):
    ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
    I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
    ML_stress=model.kBT*I+ML_pi_p
    return ML_E, ML_H1H2, ML_stress, S, ds_dc

start_time=time.time() 
c_all = np.zeros([800,3,3,3])
dt=100
c_all[0] = c_tmp
for loop in range(1,800):
    for i in range(400):
        c=tf.constant(c_tmp,dtype=tf.float32)
        ML_E, ML_H1H2, ML_stress, S, ds_dc=get_pred_data(c)
        ML_E, ML_H1H2, ML_stress, S, ds_dc=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(), ds_dc.numpy()

        c_tmp=c_tmp+ML_H1H2/170*dt
    print(c_tmp[0,0,0,0],c_tmp[0,0,2,2])
    c_all[loop] = c_tmp

savemat(VAR_DeePN_Folder+'_chain-VAR_net_100_100'+f'/sim_data_{rand_seed}_0.016_dt_0.02_Nbin_60_stop_2000_gamma_170_relax.mat',{'c':c_all})