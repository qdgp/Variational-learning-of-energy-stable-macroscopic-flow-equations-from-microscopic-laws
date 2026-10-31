from data_io import savemat
import numpy as np
import scipy.io
import h5py

class SimulationModel(object):
    ver='chain'
    def __init__(self,fx,dt,T,Nbin=20,gamma=185,ES=False,rand_seed=None,save_freq=0.8,stop_time=800):
        self.gamma = gamma
        self.dt = dt
        self.rho = 6.1
        self.rho_s = 4
        self.fx = fx
        self.gx = fx*self.rho_s
        self.n_p = 0.3 * 6
        self.vis_sol = 55
        self.Z = 100
        self.Nbin = Nbin
        self.Nbin_half = int(self.Nbin/2)
        self.dx = self.Z/self.Nbin
        self.dx_sq = self.dx*self.dx
        self.steps=int(T/self.dt)+1
        self.ES=ES
        
        if ES:
            self.save_name = f'sim_data_ES_{rand_seed}_{self.fx}_dt_{self.dt}_Nbin_{self.Nbin}.mat'
        else:
            self.save_name = f'sim_data_{rand_seed}_{fx}_dt_{dt}_Nbin_{Nbin}_stop_{stop_time}_gamma_{gamma}.mat'
        
        self.save_steps=int(T/save_freq)+1
        self.save_step=int(save_freq/self.dt)
        self.save={}
        self.save['t']=np.arange(self.save_steps)*save_freq
        self.save['dt']=self.dt
        self.save_i=0

    def init_c0_only(self,c_init):
        self.c_tmp=np.tile(c_init,[self.Nbin, 1, 1, 1])
        self.c_num=c_init.shape[1]
        self.kappa = np.zeros([self.Nbin,3,3])
        self.U_tmp = np.zeros([self.Nbin]) 
        
        self.f_body = np.zeros([self.Nbin])
        self.f_body[0:self.Nbin_half] = self.gx
        self.f_body[self.Nbin_half:self.Nbin] = -self.gx

    def init_c0(self,c_init):
        self.c_tmp=np.tile(c_init,[self.Nbin, 1, 1, 1])
        self.c_num=c_init.shape[1]
        self.save['c']=np.zeros([self.save_steps,self.Nbin,self.c_num,3,3])
        self.save['H1H2']=np.zeros([self.save_steps,self.Nbin,self.c_num,3,3])
        self.save['E'] = np.zeros([self.save_steps,self.Nbin,self.c_num,9,9])
        self.save['stress'] = np.zeros([self.save_steps,self.Nbin,3,3])
        self.save['U_list'] = np.zeros([self.save_steps,self.Nbin])
        self.save['kenergy'] = np.zeros([self.save_steps])
        self.save['S_list'] = np.zeros([self.save_steps,self.Nbin])
        self.save['Senergy'] = np.zeros([self.save_steps])
        self.save['r'] = np.zeros([self.save_steps])

        self.f_body = np.zeros([self.Nbin])
        self.f_body[0:self.Nbin_half] = self.gx
        self.f_body[self.Nbin_half:self.Nbin] = -self.gx
                
        self.kappa = np.zeros([self.Nbin,3,3])
        self.U_tmp = np.zeros([self.Nbin]) 

    def init_ES(self,S,ES=False):
        self.r = self.get_Es(S)
        self.Es_sqrt = self.get_Es(S)

        
    def one_step(self,ML_E,ML_H1H2,stress,S=0):    
        stress_p_xy = stress[:,0,2] * self.n_p

        dc=ML_H1H2/self.gamma
        dc_advection=np.reshape(self.kappa,[self.Nbin,1,3,3]) @ self.c_tmp
        # dc_advection=self.c_tmp @ np.reshape(self.kappa,[self.Nbin,1,3,3])
        dc_source = np.reshape(np.reshape(self.kappa,[self.Nbin,1,1,9]) @ ML_E,[self.Nbin,self.c_num,3,3]) 
        dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 

        self.c_tmp=self.c_tmp+dc*self.dt

        grad_stress_s_tmp=self.vis_sol*self.D2Dx2(self.U_tmp)
        grad_stress_p_tmp=self.DDx(stress_p_xy)

        self.U_tmp = self.U_tmp + (grad_stress_s_tmp + grad_stress_p_tmp + self.f_body)/self.rho * self.dt; 
        self.kappa[:,0,2] = self.DDx(self.U_tmp)
        self.Es_sqrt = self.get_Es(S)
        self.r = self.get_Es(S)

    def one_step_elongation(self,ML_E,ML_H1H2,stress,S=0):    
        dc=ML_H1H2/self.gamma
        dc_advection=np.reshape(self.kappa,[self.Nbin,1,3,3]) @ self.c_tmp
        dc_source = np.reshape(np.reshape(self.kappa,[self.Nbin,1,1,9]) @ ML_E,[self.Nbin,self.c_num,3,3]) 
        dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 

        self.c_tmp=self.c_tmp+dc*self.dt
        self.Es_sqrt = self.get_Es(S)
        self.r = self.get_Es(S)

    def one_step_ES(self,ML_E,ML_H1H2,stress,S,ds_dc):
        self.ML_H1H2=ML_H1H2
        self.ML_E=ML_E
        self.ds_dc=ds_dc
        grad_stress = self.DDx(stress[:,0,2] * self.n_p)
        self.Es_sqrt = self.get_Es(S)

        res=1
        u_tilde,r,dc=self.gaussian_iteration_one_step(grad_stress,self.U_tmp,self.r)
        while res>0.0001:
            u_tilde_new,r_new,dc=self.gaussian_iteration_one_step(grad_stress,u_tilde,r)
            res=np.linalg.norm(u_tilde_new-u_tilde)+np.abs(r_new-r)
            u_tilde,r,dc=self.gaussian_iteration_one_step(grad_stress,u_tilde_new,r_new)
        self.U_tmp,self.r,dc=self.gaussian_iteration_one_step(grad_stress,u_tilde,r)
        self.c_tmp=self.c_tmp+dc*self.dt

    def gaussian_iteration_one_step(self,grad_stress,u_tilde,r):
        grad_stress=grad_stress*r/self.Es_sqrt
        u_tilde=(self.rho*self.U_tmp+self.dt*self.f_body+self.dt*self.vis_sol/self.dx_sq*
                    (np.roll(u_tilde,-1,axis=0)+np.roll(u_tilde,1,axis=0))+self.dt*grad_stress)/(self.rho+2*self.dt*self.vis_sol/self.dx_sq)                       
        self.kappa[:,0,2] = self.DDx(u_tilde)

        dc=self.ML_H1H2/self.gamma*r/self.Es_sqrt
        dc_advection=np.reshape(self.kappa,[self.Nbin,1,3,3]) @ self.c_tmp
        dc_source = np.reshape(np.reshape(self.kappa,[self.Nbin,1,1,9]) @ self.ML_E,[self.Nbin,self.c_num,3,3]) 
        dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 

        r=self.r+self.dt/2/self.Es_sqrt*np.sum(self.ds_dc*dc)*self.dx
        return u_tilde,r,dc

    def get_Es(self,S):
        return np.sqrt(np.sum(S+10)*self.dx)
    
    def save_data(self,save_folder): 
        savemat(save_folder+'/'+self.save_name,self.save)
        
    def reset_fx(self,fx): 
        self.fx = fx
        self.gx = fx*self.rho_s
        self.f_body[0:self.Nbin_half] = self.gx
        self.f_body[self.Nbin_half:self.Nbin] = -self.gx        
    
    def DDx(self,w):
        return   (np.roll(w,-1,axis=0)-np.roll(w,1,axis=0))/self.dx/2.0
    def D2Dx2(self,w):
        return   (np.roll(w,-1,axis=0)-2*w+np.roll(w,1,axis=0))/self.dx_sq

    def collect_data(self,ML_E,ML_H1H2,stress,S=None):
        self.save['c'][self.save_i]=self.c_tmp
        self.save['stress'][self.save_i] = stress
        self.save['H1H2'][self.save_i]=ML_H1H2
        self.save['E'][self.save_i]=ML_E
        self.save['U_list'][self.save_i]=self.U_tmp
        self.save['r'][self.save_i]=self.r
        self.save['kenergy'][self.save_i]=1/2*self.rho*np.sum(self.U_tmp**2)*self.dx
        if S is not None:
            self.save['S_list'][self.save_i]=S.reshape(self.Nbin)
            self.save['Senergy'][self.save_i]=self.get_Es(S)**2
        self.save_i=self.save_i+1

    def collect_other_data(self,name,var):
        if self.save_i==1:
            self.save[name]=np.zeros([self.save_steps]+list(var.shape))
        self.save[name][self.save_i-1]=var

    def get_c(self):
        return self.c_tmp
    
    def sort_eigen(c_0):
        # try to determine the direction of c_0_U
        # r, (?, self.N_p, 3)
        direction100 = np.array([1, 0, 0])                                              # (3,) any direction
        c_0_gamma, c_0_U = np.linalg.eig(c_0)                                           # (?, 3) unsorted eigenvalues, (?, 3, 3) unsorted eigenvectors
        idx = (c_0_gamma).argsort()[:, : : -1]                                          # (?, 3) 0, 1, 2
        for i in range(idx.shape[0]):
            c_0_gamma[i, :] = c_0_gamma[i, idx[i]]                                      # (3,) sort eigenvalues (all positive) from large to small
            c_0_U[i, :, :] = c_0_U[i, :, :].T[idx[i], :].T                              # (3, 3) sort eigenvectors columns respectively
            c_0_U[i, :, :] = np.sign(np.sign(direction100 @ c_0_U[i, :, :])+0.1) * c_0_U[i, :, :]    # (3, 3) choose directions
        return c_0_U                                                    # (?, 3, 3) <r \otimes r>, (?, 3) sorted eigenvalues, (?, 3, 3) sorted eigenvectors
    
    def main(sim_model,model,stop_time,tf):
        import time
        ds_dc_star, S0=model.get_S0()
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)
        sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

        @tf.function
        def get_pred_data(c):
            ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
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
        return sim_model

    def main_MD_M(sim_model,model,stop_time,tf):
        import time
        ds_dc_star, S0=model.get_S0()
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)
        sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

        @tf.function
        def get_pred_data(c):
            ML_H1H2, ML_E, ML_pi_p, S, ds_dc, ML_M= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
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
            if sim_model.ES:
                sim_model.one_step_ES(ML_E,ML_H1H2,ML_stress,S,ds_dc)
            else:
                sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)
            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
                if loop % (sim_model.save_steps//10) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
                    sim_model.save_data(model.save_folder)
        sim_model.save_data(model.save_folder)
        return sim_model


    def simulation_restart(model,rand_seed,step,tf,stop_time=400):
        import time
        ds_dc_star, S0=model.get_S0()
        c_init=model.get_c_init()
        data_file =h5py.File(f'MD_sample/box_60_60_100_fx_0.016_relax/test_bin_5.mat','r')
        xr = np.array(data_file['train_rr'][0],dtype=np.float32).transpose((0,2,3,1)) # (Nbin,Np,Nbonds,3)
        Np,Nbin=xr.shape[1],xr.shape[0]

        sim_model=SimulationModel(fx=0,dt=0.2,T=stop_time,ES=False,Nbin=Nbin,rand_seed=rand_seed)
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.save_name=sim_model.save_name[:-4]+f'_{step}_restart.mat'

        sim_model.init_c0(c_init)
        sim_model.init_ES(np.tile(S0.numpy(),[Nbin,1]),sim_model.ES)

        model.batch_size=1
        model.Np=Np
        for i in range(Nbin):
            xr_tmp=tf.constant(xr[[i]])
            c, MD_H1, MD_H2 , MD_E=model.get_from_micro(xr_tmp,xr_tmp)
            sim_model.c_tmp[[i]]=c.numpy()
        model.batch_size=Nbin

        @tf.function
        def get_pred_data(c):
            S, ds_dc=model.get_S(c,ds_dc_star, S0)
            E_x,E_basis=model.get_E_basis(c)
            ML_H1H2=-model.get_ML_H1H2(E_basis, E_x, ds_dc)    
            return ML_H1H2
        
        start_time=time.time() 
        for loop in range(sim_model.steps):
            if loop==int(stop_time/sim_model.dt):
                sim_model.reset_fx(0)
            c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
            ML_H1H2=get_pred_data(c).numpy()

            dc=2*ML_H1H2/sim_model.gamma
            sim_model.c_tmp=sim_model.c_tmp+dc*sim_model.dt

            if (loop % sim_model.save_step) == 0:
                sim_model.save['c'][sim_model.save_i]=sim_model.c_tmp
                sim_model.save['H1H2'][sim_model.save_i]=ML_H1H2
                sim_model.save_i+=1
                
                if loop % (sim_model.save_step*1000) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
                    sim_model.save_data(model.save_folder)
        sim_model.save_data(model.save_folder)
        return sim_model
    

    def main_eigen(sim_model,model,stop_time,tf):
        import time
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)
        S=np.zeros((sim_model.Nbin,))
        sim_model.r = 0
        sim_model.Es_sqrt = 0

        
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
            ML_E, ML_H1H2, ML_stress= ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy()


            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress,None)
                if loop % (sim_model.save_step*1000) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
            sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)
            sim_model.c_tmp=(sim_model.c_tmp+sim_model.c_tmp.transpose([0,1,3,2]))/2.0
            
        sim_model.save_data(model.save_folder)
        return sim_model
    
    def main_eigen_old(sim_model,model,stop_time,tf):
        import time
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init[[0]])
        S=np.zeros([sim_model.Nbin,])
        sim_model.init_ES(S,sim_model.ES)

        @tf.function
        def get_pred_data(c,c0U):
            ML_H1H2, ML_E, ML_stress= model.get_from_macro(c,c0U)
            return ML_E, ML_H1H2, ML_stress
        
        start_time=time.time() 
        for loop in range(sim_model.steps):
            if loop==int(stop_time/sim_model.dt):
                sim_model.reset_fx(0)
            c0U=SimulationModel.sort_eigen(sim_model.c_tmp[:,0,:,:])
            c0U=tf.constant(c0U,dtype=tf.float32)
            c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
            ML_E, ML_H1H2, ML_stress=get_pred_data(c,c0U)
            ML_E, ML_H1H2, ML_stress=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy()
            sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)
            
            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
                if loop % (sim_model.save_step*1000) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
        sim_model.save_data(model.save_folder)
        return sim_model
    
    def main_elongation(sim_model,model,tf,e_rate):
        import time
        ds_dc_star, S0=model.get_S0()
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)
        sim_model.init_ES(np.tile(S0.numpy(),[sim_model.Nbin,1]),sim_model.ES)

        sim_model.save['elongation_rate']=e_rate
        sim_model.kappa[:,0,0]=sim_model.save['elongation_rate']
        sim_model.kappa[:,1,1]=-sim_model.save['elongation_rate']

        @tf.function
        def get_pred_data(c):
            ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
            I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
            ML_stress=model.kBT*I+ML_pi_p
            return ML_E, ML_H1H2, ML_stress, S, ds_dc
        
        start_time=time.time() 
        for loop in range(sim_model.steps):
            c=tf.constant(sim_model.c_tmp,dtype=tf.float32)
            ML_E, ML_H1H2, ML_stress, S, ds_dc=get_pred_data(c)
            ML_E, ML_H1H2, ML_stress, S, ds_dc=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(), ds_dc.numpy()
        
            dc=ML_H1H2/sim_model.gamma
            dc_advection=np.reshape(sim_model.kappa,[sim_model.Nbin,1,3,3]) @ sim_model.c_tmp
            dc_source = np.reshape(np.reshape(sim_model.kappa,[sim_model.Nbin,1,1,9]) @ ML_E,[sim_model.Nbin,sim_model.c_num,3,3]) 
            dc = dc + dc_source+dc_advection+ dc_advection.transpose((0,1,3,2)) 

            sim_model.c_tmp=sim_model.c_tmp+dc*sim_model.dt

            sim_model.Es_sqrt = sim_model.get_Es(S)
            sim_model.r = sim_model.get_Es(S)

            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
                if loop % (sim_model.save_step*200) == 0:
                    print(f"T:{loop*sim_model.dt}, time:{time.time() - start_time}",flush=True)