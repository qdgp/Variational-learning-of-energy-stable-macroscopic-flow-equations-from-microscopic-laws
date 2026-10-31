from data_io import savemat
import numpy as np
import scipy.io
import h5py

class TestModel(object):
    kBT=0.25
    def __init__(self,batch_size,c_num,data_size):
        self.batch_size=batch_size
        self.c_num=c_num
        self.data_size=data_size
        self.save={}
        self.init_model()

    def init_model(self):
        c_num=self.c_num     
        self.save['MD_E']=np.zeros(self.data_size+[c_num,9,9],dtype=np.float32)
        self.save['ML_E']=np.zeros(self.data_size+[c_num,9,9],dtype=np.float32)
        self.save['ML_stress']=np.zeros(self.data_size+[3,3],dtype=np.float32)
        self.save['MD_H1H2']=np.zeros(self.data_size+[c_num,3,3],dtype=np.float32)
        self.save['ML_H1H2']=np.zeros(self.data_size+[c_num,3,3],dtype=np.float32)
        self.save['S']=np.zeros(self.data_size+[1],dtype=np.float32)
        self.save['dS_dc']=np.zeros(self.data_size+[c_num,3,3],dtype=np.float32)
        self.save['c']=np.zeros(self.data_size+[c_num,3,3],dtype=np.float32)

    def load_test_set_box60(self):
        data_file =h5py.File('MD_sample/box_60_60_80/test_bin_0.625.mat','r')
        self.xr = np.array(data_file['train_rr']).transpose((0,1,3,2)) # (Nsample,Nbin,Np,3)
        self.Af = np.array(data_file['train_ff']).transpose((0,1,3,2)) # (Nsample,Nbin,Np,3)
        self.Np=self.xr.shape[2]
        self.c0 = np.transpose(self.xr,(0,1,3,2)) @ self.xr / self.Np # (Nsample,3,3)
        self.N_sample=1000
        self.bin_num=self.xr.shape[1]
        self.data_size=[self.N_sample,self.bin_num]
        
    def collect_macro_data(self,i,j,MD_E,ML_E,MD_H1H2,ML_H1H2,ML_stress,S,dS_dc,c):
        bs=self.batch_size
        self.save['MD_E'][i*bs:(i+1)*bs,j]=MD_E
        self.save['ML_E'][i*bs:(i+1)*bs,j]=ML_E
        self.save['ML_H1H2'][i*bs:(i+1)*bs,j]=ML_H1H2
        self.save['MD_H1H2'][i*bs:(i+1)*bs,j]=MD_H1H2
        self.save['ML_stress'][i*bs:(i+1)*bs,j]=ML_stress
        self.save['S'][i*bs:(i+1)*bs,j]=S
        self.save['dS_dc'][i*bs:(i+1)*bs,j]=dS_dc
        self.save['c'][i*bs:(i+1)*bs,j]=c

    def collect_macro_data_1d(self,i,MD_E,ML_E,MD_H1H2,ML_H1H2,ML_stress,S,dS_dc,c):
        bs=self.batch_size
        self.save['MD_E'][i*bs:(i+1)*bs]=MD_E
        self.save['ML_E'][i*bs:(i+1)*bs]=ML_E
        self.save['ML_H1H2'][i*bs:(i+1)*bs]=ML_H1H2
        self.save['MD_H1H2'][i*bs:(i+1)*bs]=MD_H1H2
        self.save['ML_stress'][i*bs:(i+1)*bs]=ML_stress
        self.save['S'][i*bs:(i+1)*bs]=S
        self.save['dS_dc'][i*bs:(i+1)*bs]=dS_dc
        self.save['c'][i*bs:(i+1)*bs]=c

    def collect_other_data_1d(self,i,name,var):
        bs=self.batch_size
        if i==0:
            self.save[name]=np.zeros(self.data_size+list(var.shape)[1:],dtype=np.float32)
        self.save[name][i*bs:(i+1)*bs]=var

    def collect_other_data(self,i,j,name,var):
        bs=self.batch_size
        if i==0 and j==0:
            self.save[name]=np.zeros(self.data_size+list(var.shape)[1:],dtype=np.float32)
        self.save[name][i*bs:(i+1)*bs,j]=var

    def evalute_loss(self):
        error_E=self.save['MD_E']-self.save['ML_E']
        error_H1H2=self.save['MD_H1H2']-self.save['ML_H1H2']
        error_stress=self.save['ML_stress']-self.save['MD_stress']
        
        E_loss = np.linalg.norm(error_E)
        H1H2_loss = np.linalg.norm(error_H1H2)
        stress_loss = np.linalg.norm(error_stress)
        for i in self.data_size:
            E_loss=E_loss/float(i)
            H1H2_loss=H1H2_loss/float(i)
            stress_loss=stress_loss/float(i)

        I = np.tile(np.eye(3), self.data_size + [1, 1])

        E_rmse = np.linalg.norm(error_E)/np.linalg.norm(self.save['MD_E'])
        H1H2_rmse = np.linalg.norm(error_H1H2)/np.linalg.norm(self.save['MD_H1H2'])
        stress_rmse = np.linalg.norm(error_stress)/np.linalg.norm(self.kBT*I -self.save['MD_stress'])
        self.save['loss']=np.array([E_loss, E_rmse, stress_loss, stress_rmse, H1H2_loss, H1H2_rmse])


    def save_data(self,save_name):
        savemat(save_name,self.save)

    def sort_eigen(c_0):
        c_0_gamma, c_0_U = np.linalg.eig(c_0)                                           # (?, 3) unsorted eigenvalues, (?, 3, 3) unsorted eigenvectors
        return c_0_U
    
    def main_train(model,xr,ff,tf):
        bs=model.batch_size
        data_size=[bs*(xr.shape[0]//bs)]
        model.Np=xr.shape[1]
        
        self=TestModel(bs,model.c_num,data_size)

        ds_dc_star, S0=model.get_S0()
        def evaluate_bin(i):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
            stress=tf.reduce_mean(tf.matmul(xr_tmp,ff_tmp,transpose_a=True),axis=1)/model.Nbonds

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
            MD_H1H2=model.kBT*MD_H1-MD_H2
            ML_stress= model.kBT*tf.eye(3, batch_shape = [bs],dtype=tf.float32)+ML_pi_p
            self.collect_macro_data_1d(i,MD_E.numpy(),ML_E.numpy(),MD_H1H2.numpy(),ML_H1H2.numpy(),
                                    ML_stress.numpy(),S.numpy(),ds_dc.numpy(),c.numpy())
            self.collect_other_data_1d(i,'MD_stress',stress)
            return
        for i in range(xr.shape[0]//bs):
            evaluate_bin(i)
            # print(i,flush=True)
        self.evalute_loss()
        return self

    def main_train_micro(model,xr,ff,tf):
        bs=model.batch_size
        data_size=[bs*(xr.shape[0]//bs)]
        model.Np=xr.shape[1]
        
        self=TestModel(bs,model.c_num,data_size)
        def evaluate_bin(i):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            self.collect_other_data_1d(i,'c',c.numpy())
            self.collect_other_data_1d(i,'MD_H1',MD_H1.numpy())
            self.collect_other_data_1d(i,'MD_H2',MD_H2.numpy())
            self.collect_other_data_1d(i,'MD_E',MD_E.numpy())
            return
        for i in range(xr.shape[0]//bs):
            evaluate_bin(i)
            print(i,flush=True)
        return self

    def main_train_eigen(model,xr,ff,tf):
        bs=model.batch_size
        data_size=[bs*(xr.shape[0]//bs),1]
        self=TestModel(bs,model.c_num,data_size)

        def evaluate_bin(i):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
            stress=tf.reduce_mean(tf.matmul(xr_tmp,ff_tmp,transpose_a=True),axis=1)/model.Nbonds

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            c0U=tf.constant(TestModel.sort_eigen(c.numpy()[:,0,:,:]),dtype=tf.float32)
            ML_H1, ML_H2, ML_E, ML_stress= model.get_from_macro(c,c0U)
            MD_H1H2=model.kBT*MD_H1-MD_H2
            ML_H1H2=model.kBT*ML_H1-ML_H2

            self.collect_macro_data(i,0,MD_E.numpy(),ML_E.numpy(),MD_H1H2.numpy(),ML_H1H2.numpy(),
                                    ML_stress.numpy(),0,0,c.numpy())
            self.collect_other_data(i,0,'MD_stress',stress)
            return
        for i in range(xr.shape[0]//bs):
                evaluate_bin(i)
        self.evalute_loss()
        return self


    # def main_train_eigen(model,xr,ff,tf):
    #     bs=model.batch_size
    #     data_size=[bs*(xr.shape[0]//bs),1]
    #     self=TestModel(bs,model.c_num,data_size)

    #     ds_dc_star, S0=model.get_S0()
    #     def evaluate_bin(i):
    #         xr_tmp=tf.constant(xr[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
    #         ff_tmp=tf.constant(ff[i*bs:(i+1)*bs],dtype=tf.float32) # (bs,Np,Nbonds,3)
    #         stress=tf.reduce_mean(tf.matmul(xr_tmp,ff_tmp,transpose_a=True),axis=1)/model.Nbonds

    #         c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
    #         c0U=tf.constant(TestModel.sort_eigen(c.numpy()[:,0,:,:]),dtype=tf.float32)
    #         ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,c0U,ds_dc_star=ds_dc_star,S0=S0)
    #         MD_H1H2=model.kBT*MD_H1-MD_H2
    #         ML_stress= model.kBT*tf.eye(3, batch_shape = [bs],dtype=tf.float32)+ML_pi_p
    #         self.collect_macro_data(i,0,MD_E.numpy(),ML_E.numpy(),MD_H1H2.numpy(),ML_H1H2.numpy(),
    #                                 ML_stress.numpy(),S.numpy(),ds_dc.numpy(),c.numpy())
    #         self.collect_other_data(i,0,'MD_stress',stress)
    #         return
    #     for i in range(xr.shape[0]//bs):
    #             evaluate_bin(i)
    #     self.evalute_loss()
    #     return self.save['loss']

    def main_test(model,xr,ff,data_size,tf):
        bs=model.batch_size
        test_model=TestModel(bs,model.c_num,data_size)
        ds_dc_star, S0=model.get_S0()
        def evaluate_bin(i,j):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)
            stress=tf.reduce_mean(tf.matmul(xr_tmp,ff_tmp,transpose_a=True),axis=1)/model.Nbonds

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
            MD_H1H2=model.kBT*MD_H1-MD_H2
            ML_stress= model.kBT*tf.eye(3, batch_shape = [bs],dtype=tf.float32)+ML_pi_p
            test_model.collect_macro_data(i,j,MD_E.numpy(),ML_E.numpy(),MD_H1H2.numpy(),ML_H1H2.numpy(),
                                    ML_stress.numpy(),S.numpy(),ds_dc.numpy(),c.numpy())
            test_model.collect_other_data(i,j,'MD_stress',stress)
            return

        for i in range(int(data_size[0]/bs)):
            for j in range(data_size[1]):
                evaluate_bin(i,j)
        test_model.evalute_loss()
        return test_model



    def main_test_eigen(model,xr,ff,data_size,tf):
        bs=model.batch_size
        test_model=TestModel(bs,model.c_num,data_size)
        def evaluate_bin(i,j):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)
            stress=tf.reduce_mean(tf.matmul(xr_tmp,ff_tmp,transpose_a=True),axis=1)/model.Nbonds

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            c0U=tf.constant(TestModel.sort_eigen(c.numpy()[:,0,:,:]),dtype=tf.float32)
            ML_H1, ML_H2, ML_E, ML_stress= model.get_from_macro(c,c0U)
            MD_H1H2=model.kBT*MD_H1-MD_H2
            ML_H1H2=model.kBT*ML_H1-ML_H2

            test_model.collect_macro_data(i,j,MD_E.numpy(),ML_E.numpy(),MD_H1H2.numpy(),ML_H1H2.numpy(),
                                    ML_stress.numpy(),0,0,c.numpy())
            test_model.collect_other_data(i,j,'MD_stress',stress)
            return

        for i in range(int(data_size[0]/bs)):
            for j in range(data_size[1]):
                evaluate_bin(i,j)
        test_model.evalute_loss()
        return test_model

    def main_test_restart(model,xr, ff, Np, data_size,step, rand_seed,tf):
        bs=2
        model.batch_size=bs
        model.Np=Np
        test_model=TestModel(bs,model.c_num,data_size)
        test_model.save={}
        ds_dc_star, S0=model.get_S0()

        def evaluate_bin(i,j):
            xr_tmp=tf.constant(xr[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)
            ff_tmp=tf.constant(ff[i*bs:(i+1)*bs,j],dtype=tf.float32) # (bs,Np,Nbonds,3)

            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,ff_tmp)
            S, ds_dc=model.get_S(c,ds_dc_star, S0)
            E_x,E_basis=model.get_E_basis(c)
            ML_H1H2=-model.get_ML_H1H2(E_basis, E_x, ds_dc)    
            MD_H1H2=model.kBT*MD_H1-MD_H2

            test_model.collect_other_data(i,j,'c',c.numpy())
            test_model.collect_other_data(i,j,'ML_H1H2',ML_H1H2.numpy())
            test_model.collect_other_data(i,j,'MD_H1H2',MD_H1H2.numpy())
            return

        for i in range(int(data_size[0]/bs)):
            for j in range(data_size[1]):
                evaluate_bin(i,j)
        test_model.save_data(model.save_folder+f'/test_seed_{rand_seed}_{step}_fx_0.016_restart.mat')
        return test_model