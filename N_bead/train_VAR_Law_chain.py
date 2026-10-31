from data_io import savemat
import os, sys, time, h5py, scipy.io
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rand_seed,c_num,data_set=int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]

from solver_chain_VAR_Law import StressModel as md
save_prex='model_chain_N7/'
import numpy as np
import tensorflow as tf
from model_simulation import SimulationModel
from load_set import load
from model_test_set import TestModel 

net_size=list(map(int, sys.argv[4:]))
print(f'rand seed: {rand_seed}, c: {c_num}, net: {net_size}')

xr, ff, stress,Np,data_set_name=load.load(md.ver,data_set)
Nsample=xr.shape[0]
bs = 50

xr_ref=load.init(md.ver)
config={'c_num':c_num,
        'encoder':[100,100,100],'S':net_size,'B':net_size,'E':net_size,'Sl':net_size,'Br':net_size,'Er':net_size,
        'start_rate' : 1e-03,'decay_step':int(Nsample/bs),'decay_ratio' : 0.95,
        'step':80*int(Nsample/bs),'batch_size':bs,
        'loss_k':[0.1,0.6,0.3],
        'save_folder':save_prex+'model5_macro_'+md.ver+f'_c{c_num}_{data_set_name}_net_{"_".join(map(str,net_size))}_N{Np}',
        'load_folder':''}
model=md(Np)
print(config['save_folder'])

model.init_for_training(config,seed=rand_seed,stddev=0.01)
if config['load_folder']:
    model.load_model_weight_compatible(config['load_folder'],f'train_result_seed_{rand_seed}.mat')
model.init_rr_ref(xr_ref)
with tf.device("CPU"):
    train_dataset = tf.data.Dataset.from_tensor_slices({"xr":xr, "ff": ff, "stress":stress,"list":np.arange(xr.shape[0])})
    train_dataset = train_dataset.shuffle(xr.shape[0]).repeat().batch(config['batch_size'])
train_iter = iter(train_dataset)

lr=tf.keras.optimizers.schedules.ExponentialDecay(config['start_rate'], config['decay_step'], config['decay_ratio'], staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)

def get_pred_data(train_data):
    c, MD_H1, MD_H2 , MD_E=model.get_from_micro(train_data['xr'],train_data['ff'])
    ML_H1H2, ML_E, ML_pi_p, S, ds_dc= model.get_from_macro(c,MD_E) 
    MD_H1H2=model.kBT*MD_H1-MD_H2
    I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
    ML_stress=model.kBT*I+ML_pi_p
    return c, ML_stress, MD_H1H2, ML_H1H2, MD_E, ML_E, S   

@tf.function
def run_optimization(train_data):
    with tf.GradientTape() as tape:  
        c, ML_stress, MD_H1H2, ML_H1H2, MD_E, ML_E, S=get_pred_data(train_data)
        I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
        E_tmp=(MD_E - ML_E)/(tf.norm(MD_E,axis=[-2,-1],keepdims=True)+0.01)
        stress_tmp=(train_data['stress'] - ML_stress)/(tf.norm(model.kBT*I-train_data['stress'],axis=[-2,-1],keepdims=True)+0.01)
        H1H2_tmp=(ML_H1H2 - MD_H1H2)/(tf.norm(MD_H1H2,axis=[-2,-1],keepdims=True)+0.01)
        E_loss = tf.norm(E_tmp)/model.batch_size
        pi_p_loss = tf.norm(stress_tmp)/model.batch_size
        H1H2_loss = tf.norm(H1H2_tmp)/model.batch_size
        loss = config['loss_k'][0] * E_loss+config['loss_k'][1] * pi_p_loss +config['loss_k'][2]*H1H2_loss

    E_l=tf.norm(MD_E - ML_E)
    E_rmse = E_l / tf.norm(MD_E)
    pi_p_l=tf.norm(train_data['stress'] - ML_stress)
    pi_p_rmse = pi_p_l / tf.norm(model.kBT*I-train_data['stress'])
    H1H2_l=tf.norm(ML_H1H2 - MD_H1H2) 
    H1H2_rmse =H1H2_l / tf.norm(MD_H1H2)
    
    trainable_variables = list(model.weights.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    loss_all=tf.stack([loss, E_loss, E_rmse, pi_p_loss, pi_p_rmse, H1H2_loss, H1H2_rmse, E_l, pi_p_l, H1H2_l],axis=0)
    return loss_all

training_history=np.zeros([config['step']+1,12])
start_time = time.time()

print("Total steps: %3u" %(config['step']),flush=True) 

for step in range(config['step']+1):
    train_data=next(train_iter)
    loss_all=run_optimization(train_data)
    
    elapsed_time = time.time() - start_time
    training_history[step]=np.array([step]+loss_all.numpy().tolist()+[elapsed_time])
    if step == 0:
        print("build time: %3u" %(time.time() - start_time),flush=True) 
    if step % config['decay_step'] == 0:
        print("step: %4u, loss: %.2f, E: %.2f(%.2f), stress: %.2f(%.2f),H1H2: %.2f(%.2f), time %3u" %
            (step,*loss_all.numpy()[:7].tolist(),elapsed_time),flush=True)   
savemat(config['save_folder']+f'/training_history_seed_{rand_seed}.mat', 
                {'training_history':np.array(training_history)})
model.record_save(f'train_result_seed_{rand_seed}.mat')

test_model=TestModel.main_train(model,xr,ff,tf) #E_loss, E_rmse, stress_loss, stress_rmse, H1H2_loss, H1H2_rmse
print("Final loss: E: %.2f(%.2f), stress: %.2f(%.2f),H1H2: %.2f(%.2f)" % tuple(test_model.save['loss']),flush=True)
############################################################################
train_data=[]

ds_dc_star, S0=model.get_S0()
Nsample=int(xr.shape[0]/bs)*bs
xr=xr[:Nsample]
ff=ff[:Nsample]
stress=stress[:Nsample]

macro={'stress':stress,'c':np.zeros([Nsample,model.c_num,3,3],dtype=model.np_dtype),
       'MD_H1H2':np.zeros([Nsample,model.c_num,3,3],dtype=model.np_dtype),
       'MD_E':np.zeros([Nsample,model.c_num,9,9],dtype=model.np_dtype)}

for i in range(int(Nsample/bs)):
    xr_tmp=tf.constant(xr[i*bs:(i+1)*bs],dtype=model.tf_dtype) # (bs,Np,Nbonds,3)
    ff_tmp=tf.constant(ff[i*bs:(i+1)*bs],dtype=model.tf_dtype) # (bs,Np,Nbonds,3)

    c_tmp, MD_H1, MD_H2 , MD_E_tmp=model.get_from_micro(xr_tmp,ff_tmp)
    MD_H1H2_tmp=model.kBT*MD_H1-MD_H2
    macro['c'][i*bs:(i+1)*bs],macro['MD_H1H2'][i*bs:(i+1)*bs],macro['MD_E'][i*bs:(i+1)*bs]=c_tmp.numpy(),MD_H1H2_tmp.numpy(),MD_E_tmp.numpy()

macro_tf={'c':tf.constant(macro['c'],dtype=model.tf_dtype),
          'MD_H1H2':tf.constant(macro['MD_H1H2'],dtype=model.tf_dtype),
          'MD_E':tf.constant(macro['MD_E'],dtype=model.tf_dtype),
          'stress':tf.constant(stress,dtype=model.tf_dtype)}
xr_tmp, ff_tmp = [], []

model.batch_size=Nsample
weights={}
keys=list(model.weights.keys())
for key in keys:
    weights[key]=model.weights[key].numpy()
    if not key.startswith('E'):
        model.weights.pop(key, None)
keys=model.weights.keys()

lr=tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 100, 0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)

@tf.function
def run_optimization(c,MD_E):
    with tf.GradientTape() as tape:  
        E_x,E_basis=model.get_E_basis(c)
        ML_E=model.get_ML_E(E_basis, E_x)
        E_loss = tf.norm(MD_E - ML_E)
    E_rmse = E_loss / tf.norm(MD_E)
    
    trainable_variables = list(model.weights.values())
    gradients = tape.gradient(E_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return E_loss, E_rmse, ML_E

for step in range(4000+1):
    E_loss, E_rmse, ML_E=run_optimization(macro_tf['c'],macro_tf['MD_E'])
    if step % 500 == 0:
        print("step: %4u, E: %.3f(%.3f)" %(step, E_loss, E_rmse),flush=True)
macro_tf['MD_E']=[]
macro_tf['ML_E']=ML_E

for key in model.weights:
    weights[key]=model.weights[key].numpy()
model.weights={}
for key in weights:
    if key.startswith('B') or key.startswith('S'):
        model.weights[key]=tf.Variable(weights[key],dtype=model.tf_dtype,trainable=True,name=key)
keys=model.weights.keys()

lr=tf.keras.optimizers.schedules.ExponentialDecay(1e-4, 100, 0.9, staircase=False)
optimizer = tf.keras.optimizers.Adam(lr)
@tf.function
def run_optimization(c,MD_H1H2,ML_E,stress):
    with tf.GradientTape() as tape:  
        ds_dc_star, S0=model.get_S0()
        S, ds_dc=model.get_S(c,ds_dc_star, S0)
        E_x,E_basis=model.get_E_basis(c)
        ML_H1H2=-model.get_ML_H1H2(E_basis, E_x, ds_dc)

        ML_pi_p=model.get_pi_p(c,ds_dc,ML_E)
        I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
        ML_stress=model.kBT*I+ML_pi_p
        
        H1H2_loss = tf.norm(ML_H1H2 - MD_H1H2)
        stress_loss = tf.norm(ML_stress - stress)
        loss = H1H2_loss/model.c_num + stress_loss
    H1H2_rmse = H1H2_loss / tf.norm(MD_H1H2)
    stress_rmse = stress_loss / tf.norm(model.kBT*I-stress)
    
    trainable_variables = list(model.weights.values())
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return H1H2_loss, H1H2_rmse, stress_loss, stress_rmse, ML_H1H2, ML_stress

stress_tf=tf.constant(stress,model.tf_dtype)
for step in range(4000+1):
    H1H2_loss, H1H2_rmse, stress_loss, stress_rmse, ML_H1H2, ML_stress=run_optimization(macro_tf['c'],macro_tf['MD_H1H2'],macro_tf['ML_E'],macro_tf['stress'])
    if step % 500 == 0:
        print("step: %4u, H1H2: %.2f(%.2f), stress: %.2f(%.2f)" %(step, H1H2_loss, H1H2_rmse, stress_loss, stress_rmse),flush=True)

for key in model.weights:
    weights[key]=model.weights[key].numpy()
model.weights={}
for key in weights:
    model.weights[key]=tf.Variable(weights[key],dtype=model.tf_dtype,trainable=True,name=key)

model.record_save(f'train_result_seed_{rand_seed}.mat')
model.batch_size=20
test_loss=np.array([E_loss, E_rmse, stress_loss, stress_rmse, H1H2_loss, H1H2_rmse])
print("Final loss: E: %.2f(%.2f), stress: %.2f(%.2f),H1H2: %.2f(%.2f)" % tuple(test_loss),flush=True)

model.save_var['loss']=test_loss
model.record_save(f'train_result_seed_{rand_seed}.mat')


############################################################################
Nbin=20
model.batch_size=Nbin
stop_time=1600
try:
    sim_model=SimulationModel(fx=0.016,dt=0.2,T=stop_time*4,ES=False,Nbin=Nbin,rand_seed=rand_seed,save_freq=4)
    sim_model.save_name = f'sim_data_{rand_seed}_{sim_model.fx}_dt_{sim_model.dt}_Nbin_{Nbin}_stop_{stop_time}.mat'
    sim_model.main(model,stop_time,tf)
except:
    print('simulation fx 0.016 Failed')
############################################################################

from model_simulation_RollMill import SimulationModel as SMRM

stop_time=6000
Nbin=50
model.batch_size=Nbin**2
try:
    sim_model=SMRM(0.1, stop_time, f_para=0.006, sigma=0.0, Nbin=Nbin, gamma=170, rand_seed=rand_seed, save_freq=8)
    sim_model.save_name = f'sim_data_RollMill_{rand_seed}_{sim_model.fx}_dt_{sim_model.dt}_Nbin_{sim_model.Nbin}_gamma_{sim_model.gamma}.mat'
    sim_model.main(model,tf)
except Exception as e:
    print(e)
    sim_model.save_data(model.save_folder)