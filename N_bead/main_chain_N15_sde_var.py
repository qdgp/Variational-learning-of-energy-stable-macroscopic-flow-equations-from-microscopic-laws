from data_io import savemat
import numpy as np
import os, h5py, scipy.io, sys, time
from z_fun import get_solver
import tensorflow as tf

rand_seed=int(sys.argv[1])
save_folder=sys.argv[2]
md,save_prex=get_solver(save_folder)

model = md()
model.load_model(rand_seed,save_prex+save_folder,f'train_result_seed_{rand_seed}.mat')


if 'sde-bin16-v3' in save_folder:
    Np=400
    SDE_Folder='/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_var/results/'
    c=np.zeros([14,48,3000,model.c_num,3,3],dtype=np.float32)
    def get_xr(SDE_Folder, id, i):
        data_file =h5py.File(SDE_Folder+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i+1:02d}.mat','r')
        xr=np.array(data_file['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3)
        return xr
elif 'sde-gamma-40' in save_folder:
    Np=1600
    SDE_Folder='/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_40/results_var/'
    c=np.zeros([20,12,3000,model.c_num,3,3],dtype=np.float32)
    def get_xr(SDE_Folder, id, i):
        data_file =h5py.File(SDE_Folder+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i+1}.mat','r')
        xr=np.array(data_file['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3) (3000, 3, 1600, 14)
        xr=np.transpose(xr,[0,2,3,1])
        return xr
elif ('gamma-170' in save_folder) and ('chain-N15' in save_folder):
    Np=1600
    SDE_Folder='/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_170/results_var_v2/'
    c=np.zeros([20,12,6000,model.c_num,3,3],dtype=np.float32)
    def get_xr(SDE_Folder, id, i):
        SDE_Folder1='/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_170/results_var/'
        SDE_Folder2='/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_170/results_var_v2/'
        data_file1 =h5py.File(SDE_Folder1+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i}.mat','r')
        data_file2 =h5py.File(SDE_Folder2+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i}.mat','r')
        xr1 = np.array(data_file1['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3) (3000, 3, 1600, 14)
        xr2 = np.array(data_file2['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3) (3000, 3, 1600, 14)
        xr=np.concatenate([xr1,xr2],axis=0)
        xr=np.transpose(xr,[0,2,3,1])
        return xr

elif 'chain-N15' not in save_folder:
    Np=1600
    SDE_Folder='/mnt/gs21/scratch/gepei/DeePN_chain/SDE_data_set_v2/results_var/'
    c=np.zeros([32,24,3000,model.c_num,3,3],dtype=np.float32)
    def get_xr(SDE_Folder, id, i):
        data_file =h5py.File(SDE_Folder+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i+1}.mat','r')
        xr=np.array(data_file['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3) (3000, 3, 1600, 14)
        xr=np.transpose(xr,[0,2,3,1])
        return xr


start_time = time.time()
for id in range(c.shape[0]):
    for i in range(c.shape[1]):
        xr = get_xr(SDE_Folder, id, i)
        model.Np, model.batch_size = xr.shape[1], xr.shape[0]
        c_tmp = model.get_c_from_xr(xr)
        c[id,i] = c_tmp
    elapsed_time = time.time() - start_time
    print(id,elapsed_time,flush=True)

savemat(save_prex+save_folder + f'/test_sde_{rand_seed}.mat', {"c": c,})
