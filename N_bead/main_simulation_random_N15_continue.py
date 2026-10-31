import sys, time, os, scipy.io
import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_simulation import SimulationModel

arg1=int(sys.argv[1])
rand_seed=int(sys.argv[2])
save_folder=sys.argv[3]

N_sample=16000
Nbin=200
dt=0.01

gamma = 86
e_rate=5e-5*np.arange(1,Nbin+1)
if 'gamma-170' in save_folder:
    gamma = 170
    e_rate=3e-5*np.arange(Nbin)
if 'gamma-40' in save_folder:
    gamma = 40
    e_rate=1e-4*np.arange(Nbin)


# save_folder = "model8_macro_chain-v11_c3_set-v2_net_100_100_N1600"
print(save_folder+f' rand_seed {rand_seed} start')
md,save_prex=get_solver(save_folder)
model = md()
model.load_model(1,save_prex+save_folder,f'train_result_seed_{rand_seed}.mat')


rate=e_rate[arg1-1]
e_rate=rate*np.arange(1,1+1)
save=scipy.io.loadmat(save_prex+save_folder+f'/sim_random_{rand_seed}_dt_{dt}_N{N_sample}_rate_{rate:.5f}.mat')
c_init=np.array(save['c'][-1])

model.batch_size=Nbin
ds_dc_star, S0=model.get_S0()
sim_model=SimulationModel(fx=0.0,dt=dt,T=12000,ES=False,Nbin=Nbin,rand_seed=rand_seed,save_freq=20, gamma=gamma)
sim_model.save_name = f'sim_random_{rand_seed}_dt_{dt}_N{N_sample}_rate_{rate:.5f}_t2.mat'
print(sim_model.save_name)
sim_model.save['dn']=model.ver+f'-c{model.c_num}'
sim_model.init_c0_only(c_init)
sim_model.c_tmp=c_init

sim_model.save['elongation_rate']=np.repeat(e_rate,Nbin)
sim_model.kappa[:,0,0]=sim_model.save['elongation_rate']
sim_model.kappa[:,1,1]=-sim_model.save['elongation_rate']

def get_M(self, c_basis, Tr_c):
    c_basis = tf.reshape(c_basis,[self.batch_size,1,self.Eb_num,9])   
    Tr_c = tf.reshape(Tr_c,[self.batch_size,1,1,self.Ex_num])  

    M_matrix = self.subnetwork(Tr_c, 'B') # (batch_size, c_num,1, Eb_num^2)
    M_matrix = tf.reshape(M_matrix,[self.batch_size,self.c_num,self.Eb_num,self.Eb_num])
    Gamma = tf.matmul(M_matrix,c_basis)  # (batch_size, c_num,Eb_num, 9)

    M_matrix = tf.matmul(M_matrix,M_matrix,transpose_a=True)
    M_tmp = tf.matmul(c_basis,M_matrix,transpose_a=True) @ c_basis # (batch_size, c_num,9,9)
    return M_tmp, Gamma

@tf.function
def get_pred_data(self,c,ds_dc_star,S0):
    S, ds_dc=self.get_S(c,ds_dc_star, S0)

    with tf.GradientTape(watch_accessed_variables=False,persistent=False) as tape:
        tape.watch(c)
        E_x,E_basis=self.get_E_basis(c)
        M_tmp, Gamma = get_M(model, E_basis, E_x)
    Gradient_M = tape.batch_jacobian(M_tmp, c)
    Gradient_M = tf.reshape(Gradient_M,[self.batch_size,self.c_num,9,9,self.c_num,9])
    Gradient_M = tf.einsum('abcdbc->abd',Gradient_M)

    ML_E=self.get_ML_E(E_basis, E_x)
    ML_pi_p=self.get_pi_p(c,ds_dc,ML_E)

    ds_dc = tf.reshape(ds_dc,[self.batch_size,self.c_num,9,1])
    ML_H1H2 = -tf.reshape(M_tmp @ ds_dc,[self.batch_size,self.c_num,3,3])
    Gradient_M = tf.reshape(Gradient_M,[self.batch_size,self.c_num,3,3])

    I = tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
    ML_stress = model.kBT*I+ML_pi_p
    return ML_E, ML_H1H2, ML_stress, S, ds_dc, Gamma, Gradient_M

rng = np.random.default_rng(rand_seed)
start_time=time.time() 
kBT_N_gamma=model.kBT / N_sample / sim_model.gamma /14.0 # 14 for N bonds
for loop in range(sim_model.steps):
    c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
    ML_E, ML_H1H2, ML_stress, S, ds_dc, Gamma, Gradient_M = get_pred_data(model,c,ds_dc_star,S0)
    ML_E, ML_H1H2, Gamma, Gradient_M = ML_E.numpy(), ML_H1H2.numpy(), Gamma.numpy(), Gradient_M.numpy()
    
    Wt=rng.normal(0, 1, [Nbin,model.c_num,1,model.Eb_num])/np.sqrt(sim_model.dt)
    dc_advection = np.reshape(sim_model.kappa,[sim_model.Nbin,1,3,3]) @ sim_model.c_tmp
    dc_source = np.reshape(np.reshape(sim_model.kappa,[sim_model.Nbin,1,1,9]) @ ML_E,[sim_model.Nbin,sim_model.c_num,3,3])
    dc_random = np.sqrt(2 * kBT_N_gamma) * np.reshape(Wt @ Gamma,[Nbin,model.c_num,3,3])
    dc_gradent = Gradient_M * kBT_N_gamma
    dc = ML_H1H2/sim_model.gamma + dc_source + dc_advection + dc_advection.transpose((0,1,3,2)) + dc_random + dc_gradent

    sim_model.c_tmp=sim_model.c_tmp+dc*sim_model.dt

    if (loop % sim_model.save_step) == 0:
        sim_model.save_i = sim_model.save_i+1
        sim_model.collect_other_data('c',sim_model.c_tmp)
        sim_model.collect_other_data('stress',ML_stress.numpy())
        sim_model.collect_other_data('S_list',S.numpy())

        if loop % (sim_model.save_step*20) == 0:
            print(f"T:{loop*sim_model.dt}, time:{time.time() - start_time}",flush=True)
            sim_model.save_data(model.save_folder)
sim_model.save_data(model.save_folder)