import os, sys, time, h5py, scipy.io
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
save_folder, rand_seed=sys.argv[1], int(sys.argv[2])
from solver_chain_v10 import StressModel as md
import numpy as np
import tensorflow as tf
from model_simulation import SimulationModel
from load_set import load, resolve_data
from model_test_set import TestModel 

if '_c2_' in save_folder:
    c_num=2
elif '_c3_' in save_folder:
    c_num=3

save_prex='model_chain_N7/'

def load_file(folder):
    data_file =h5py.File(resolve_data('MD_chain_N7/'+folder+'/train_time_series_bin_2_Np_1920.mat'),'r')
    xr1 = np.array(data_file['train_rr1'],dtype=np.float32).transpose((0,2,3,1)) # (Nsample,Np,Nbonds,3)
    xr2 = np.array(data_file['train_rr2'],dtype=np.float32).transpose((0,2,3,1)) # (Nsample,Np,Nbonds,3)
    stress1 = np.array(data_file['train_stress1'],dtype=np.float32) # (Nsample,3,3)
    stress2 = np.array(data_file['train_stress2'],dtype=np.float32) # (Nsample,3,3)
    stress = (stress1+stress2)/2.0
    kappa = np.array(data_file['train_kappa'],dtype=np.float32) # (Nsample,3,3)
    return xr1,xr2,stress,kappa

def train_set():
    xr1_1,xr2_1,stress_1,kappa_1 = load_file('box_60_60_100_fx_0.02_T_800_kBT_0.22')
    xr1_2,xr2_2,stress_2,kappa_2 = load_file('box_60_60_100_fx_0.02_T_800_kBT_0.25')
    xr1_3,xr2_3,stress_3,kappa_3 = load_file('box_60_60_100_fx_0.02_T_800_kBT_0.30')
    
    xr1=np.concatenate([xr1_1,xr1_2,xr1_3],axis=0)
    xr2=np.concatenate([xr2_1,xr2_2,xr2_3],axis=0)
    stress=np.concatenate([stress_1,stress_2,stress_3],axis=0)
    kappa=np.concatenate([kappa_1,kappa_2,kappa_3],axis=0)
    return xr1,xr2,stress,kappa

xr1,xr2,stress,kappa=train_set()

Np=xr1.shape[1]

bs=40
net_size=[100,100]
xr_ref=load.init('chain')
config={'c_num':c_num,
        'encoder':[100,100,100],'S':net_size,'B':net_size,'E':net_size,'Sl':net_size,'Br':net_size,'Er':net_size,'batch_size':bs,
        'save_folder':save_prex+save_folder+'_'+md.ver+f'_net_{"_".join(map(str,net_size))}'}
model=md(Np)
print(config['save_folder'])

model.init_for_training(config,seed=rand_seed,stddev=0.01)
model.load_model_weight_compatible(save_prex+save_folder,f'train_result_seed_{rand_seed}.mat')
model.record_save(f'train_result_seed_{rand_seed}.mat')
if not os.path.exists(model.save_folder):
    os.makedirs(model.save_folder,exist_ok=True)
model.init_rr_ref(xr_ref)

model.batch_size,model.Np=bs,Np
Nsample=int(xr1.shape[0]/bs)*bs

macro={'c':np.zeros([Nsample,model.c_num,3,3],dtype=model.np_dtype),
       'MD_H1H2':np.zeros([Nsample,model.c_num,3,3],dtype=model.np_dtype)}

gamma=170
dt=0.4
np_kappa=np.zeros([bs,1,3,3])

for i in range(int(Nsample/bs)):
    np_kappa[:,0,0,2]=-kappa[i*bs:(i+1)*bs,0]

    xr_tmp=tf.constant(xr1[i*bs:(i+1)*bs],dtype=model.tf_dtype) # (bs,Np,Nbonds,3)
    c_1=( model.get_c_from_xr(xr_tmp)+ model.get_c_from_xr(tf.reverse(xr_tmp,axis=[2])) )/2
    xr_tmp=tf.constant(xr2[i*bs:(i+1)*bs],dtype=model.tf_dtype) # (bs,Np,Nbonds,3)
    c_2=( model.get_c_from_xr(xr_tmp)+ model.get_c_from_xr(tf.reverse(xr_tmp,axis=[2])) )/2

    c = (c_2.numpy()+c_1.numpy())/2    

    dc_advection = np_kappa @ c
    dc = ( (c_2.numpy()-c_1.numpy())/dt - dc_advection - dc_advection.transpose((0,1,3,2)) ) * gamma

    macro['c'][i*bs:(i+1)*bs] = c
    macro['MD_H1H2'][i*bs:(i+1)*bs] = dc

# print(np.linalg.norm(macro['MD_H1H2']))
# print(np.linalg.norm(macro['c']))

# xr1,xr2,xr3,ff1,ff2,ff3,stress1,stress2,stress3=[],[],[],[],[],[],[],[],[]
# xr, ff, xr_tmp, ff_tmp = [], [], [], []

macro_tf={'c':tf.constant(macro['c'],dtype=model.tf_dtype),
          'MD_H1H2':tf.constant(macro['MD_H1H2'],dtype=model.tf_dtype),
          'stress':tf.constant(stress,dtype=model.tf_dtype)}

model.batch_size=Nsample
weights={}
keys=list(model.weights.keys())
for key in keys:
    weights[key]=model.weights[key].numpy()
    if key.startswith('encoder'):
        model.weights.pop(key, None)

Rc_weight = np.ones([1,model.Nbonds])/np.sqrt(6.0)/model.scale
Rc_weight = tf.constant(Rc_weight,dtype=model.tf_dtype)
# Rc_weight=model.weights['c0_weight']/tf.norm(model.weights['c0_weight'],axis = 1,keepdims=True)/model.scale
model.weights.pop('c0_weight', None)
model.weights.pop('weight_pos', None)

lr=tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 20, 0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)

@tf.function
def run_optimization(macro_tf):
    with tf.GradientTape() as tape:  
        ML_H1H2, ML_E, ML_pi_p, S, ds_dc=model.get_from_macro(macro_tf['c'])
        I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
        ML_stress=model.kBT*I+ML_pi_p

        H1H2_loss = tf.norm(ML_H1H2 - macro_tf['MD_H1H2'])
        stress_loss = tf.norm(ML_stress - macro_tf['stress'])
        loss = 0.3*H1H2_loss + 0.7* stress_loss

    stress_rmse=stress_loss/tf.norm(macro_tf['stress'])
    H1H2_rmse=H1H2_loss/tf.norm(macro_tf['MD_H1H2'])

    trainable_variables = list(model.weights.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, H1H2_loss, H1H2_rmse, stress_loss, stress_rmse

for step in range(1000+1):
    loss, H1H2_loss, H1H2_rmse, stress_loss, stress_rmse = run_optimization(macro_tf)
    if step % 10 == 0:
        print("step: %4u, loss: %.3f, H1H2: %.3f(%.3f), stress: %.3f(%.3f)" %(step, loss, H1H2_loss, H1H2_rmse, stress_loss, stress_rmse),flush=True)

for key in model.weights:
    weights[key]=model.weights[key].numpy()
model.weights={}
for key in weights:
    model.weights[key]=tf.Variable(weights[key],dtype=model.tf_dtype,trainable=True,name=key)

model.save_var['loss']=np.array([0, 0, stress_loss, stress_rmse, H1H2_loss, H1H2_rmse])
model.record_save(f'train_result_seed_{rand_seed}.mat')

Nbin=20
model.batch_size=Nbin
sim_model=SimulationModel(fx=0.016,dt=0.2,T=2400,gamma=185, ES=False,Nbin=Nbin,rand_seed=rand_seed,stop_time=1200)
sim_model.main(model,1200,tf)
sim_model=SimulationModel(fx=0.016,dt=0.2,T=2400,gamma=170, ES=False,Nbin=Nbin,rand_seed=rand_seed,stop_time=1200)
sim_model.main(model,1200,tf)
