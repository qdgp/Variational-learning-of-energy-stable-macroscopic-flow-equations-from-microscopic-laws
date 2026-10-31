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

Np=1600
SDE_Folder='/mnt/gs21/scratch/gepei/DeePN_chain/SDE_data_set_v2/results_var/'
c=np.zeros([32,96,3000,model.c_num,3,3],dtype=np.float32)
def get_xr(SDE_Folder, id, i):
    data_file =h5py.File(SDE_Folder+f'gdot_{id:02d}/Qsave_gdot_{id:02d}_replica_{i+1}.mat','r')
    xr=np.array(data_file['Qsave'],dtype=np.float32)[1:] # (Nsample,Np,Nbonds,3) (3000, 3, 1600, 14)
    xr=np.transpose(xr,[0,2,3,1])
    return xr


start_time = time.time()
for id in range(32):
    for i in range(24):
        xr = get_xr(SDE_Folder, id, i)
        xr = xr.reshape([3000, 400, 4, 6, 3])
        model.Np, model.batch_size = xr.shape[1], xr.shape[0]
        for j in range(4):
            c_tmp = model.get_c_from_xr(xr[:,:,j])
            c[id,i*4+j] = c_tmp
    elapsed_time = time.time() - start_time
    print(id,elapsed_time,flush=True)

savemat(save_prex+save_folder + f'/test_sde_{rand_seed}_Np400.mat', {"c": c,})
