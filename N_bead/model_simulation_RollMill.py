from data_io import savemat
import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
import h5py
# import hdf5storage

class SimulationModel(object):
    ver='chain'
    def __init__(self,dt,T,f_para=0.006,sigma=0.0,Nbin=40,gamma=185,rand_seed=None,save_freq=1):
        self.Nbin=Nbin
        self.nx = Nbin
        self.ny = Nbin
        self.x_length = 100
        self.y_length = 100
        self.dx = self.x_length / self.nx
        self.dy = self.y_length / self.ny
        self.dx_sq = self.dx*self.dx
        self.dy_sq = self.dy*self.dy
        self.x = np.linspace(0, self.x_length, self.nx + 1) + self.dx / 2
        self.y = np.linspace(0, self.y_length, self.ny + 1) + self.dy / 2
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing = 'ij')

        self.gamma = gamma
        self.dt = dt
        self.rho = 6.1
        self.rho_s = 4

        self.fx = f_para
        self.f_coef = self.fx * self.rho_s / self.rho # body force coefficient

        self.n_p = 0.3*6
        self.eta = 55
        self.vis_sol = 55
        self.nu = self.eta / self.rho

        self.steps=int(T/self.dt)+1
        self.save_name = f'sim_data_RollMill_{rand_seed}_{self.fx}_dt_{self.dt}_Nbin_{self.Nbin}.mat'

        self.save_steps=int(T/save_freq)+1
        self.save_step=int(save_freq/self.dt)
        self.save={}
        self.save['t']=np.arange(self.save_steps)*save_freq
        self.save['dt']=self.dt
        self.save_i=0


        self.u_apprx = self.f_coef / self.nu * max(self.x_length, self.y_length) ** 2 / (2 * np.pi) ** 2
        dt_suggest = min(2 * self.nu / self.u_apprx, min(self.dx, self.dy) ** 2 / 2 / self.nu) / 2   
        print('Re = ' + str(self.u_apprx * np.max((self.x_length, self.y_length)) / self.nu))
        self.sigma=sigma #self.dt/self.dx/2/5
        print('sigma = ' + str(self.sigma))

        self.time_step = 0
        self.u0 = - np.sin(self.X[0 : self.nx, 0 : self.ny] / self.x_length * 2 * np.pi) * np.cos(self.Y[0 : self.nx, 0 : self.ny] / self.y_length * 2 * np.pi) * self.u_apprx
        self.v0 = + np.cos(self.X[0 : self.nx, 0 : self.ny] / self.x_length * 2 * np.pi) * np.sin(self.Y[0 : self.nx, 0 : self.ny] / self.y_length * 2 * np.pi) * self.u_apprx
        self.u = np.zeros((self.nx, self.ny)) # velocity
        self.v = np.zeros((self.nx, self.ny)) # velocity
        self.w = np.zeros((self.nx, self.ny)) # Newtonian stress
        self.kappa = np.zeros((self.nx, self.ny, 3, 3)) # velocity gradient
        self.f = + 2 * np.sin(self.X[0 : self.nx, 0 : self.ny] / self.x_length * 2 * np.pi) * np.cos(self.Y[0 : self.nx, 0 : self.ny] / self.y_length * 2 * np.pi) * self.f_coef # external force
        self.g = - 2 * np.cos(self.X[0 : self.nx, 0 : self.ny] / self.x_length * 2 * np.pi) * np.sin(self.Y[0 : self.nx, 0 : self.ny] / self.y_length * 2 * np.pi) * self.f_coef # external force

        self.Sxp = lambda fun: np.concatenate((fun[1 : self.nx, :], fun[0 : 1, :]), axis = 0)
        self.Sxm = lambda fun: np.concatenate((fun[self.nx - 1 : self.nx, :], fun[0 : self.nx - 1, :]), axis = 0)
        self.Syp = lambda fun: np.concatenate((fun[:, 1 : self.ny], fun[:, 0 : 1]), axis = 1)
        self.Sym = lambda fun: np.concatenate((fun[:, self.ny - 1 : self.ny], fun[:, 0 : self.ny - 1]), axis = 1)
        self.Sext = lambda fun: np.concatenate((np.concatenate((fun, fun[0 : 1, :]), axis = 0), np.concatenate((fun[:, 0 : 1], fun[0 : 1, 0 : 1]), axis = 0)), axis = 1)
    
    def upwind(self, fun, wind_x, wind_y):
        # wind_x, wind_y: (self.nx, self.ny, 1, 1)
        # fun: (self.nx, self.ny, 3, 3)
        epsilon = 1e-5
        fun_DDxp = (self.Sxp(fun) - fun) / self.dx # wind_x < -epsilon
        fun_DDxm = (fun - self.Sxm(fun)) / self.dx # wind_x > +epsilon
        fun_DDxc = self.DDx(fun)
        fun_DDyp = (self.Syp(fun) - fun) / self.dy # wind_y < -epsilon
        fun_DDym = (fun - self.Sym(fun)) / self.dy # wind_y > +epsilon
        fun_DDyc = self.DDy(fun)
        wind_xp = (wind_x < -epsilon)
        wind_xm = (wind_x > +epsilon)
        wind_xc = ((wind_x <= +epsilon) & (wind_x >= -epsilon))
        wind_yp = (wind_y < -epsilon)
        wind_ym = (wind_y > +epsilon)
        wind_yc = ((wind_y <= +epsilon) & (wind_y >= -epsilon))
        upwind_fun_DDx = wind_xp * fun_DDxp + wind_xm * fun_DDxm + wind_xc * fun_DDxc
        upwind_fun_DDy = wind_yp * fun_DDyp + wind_ym * fun_DDym + wind_yc * fun_DDyc
        return upwind_fun_DDx, upwind_fun_DDy

    def init_c0(self,c_init):
        self.c_num=c_init.shape[1]
        c_init=np.reshape(c_init,[1,1,self.c_num,3,3])
        self.c_tmp=np.tile(c_init,[self.nx,self.ny, 1, 1, 1])
        
        self.save['c']        = np.zeros([self.save_steps,self.nx,self.ny,self.c_num,3,3])
        # self.save['H1H2']     = np.zeros([self.save_steps,self.nx,self.ny,self.c_num,3,3])
        # self.save['E']        = np.zeros([self.save_steps,self.nx,self.ny,self.c_num,9,9])
        self.save['stress']   = np.zeros([self.save_steps,self.nx,self.ny,3,3])
        self.save['u']        = np.zeros([self.save_steps,self.nx,self.ny])
        self.save['v']        = np.zeros([self.save_steps,self.nx,self.ny])
        self.save['kenergy']  = np.zeros([self.save_steps])
        self.save['S_list']   = np.zeros([self.save_steps,self.nx,self.ny])
                
        self.kappa = np.zeros([self.nx,self.ny,3,3])
        self.u = np.zeros([self.nx,self.ny]) 
        self.v = np.zeros([self.nx,self.ny]) 

    def one_step(self,ML_E,ML_H1H2,stress,S=0):
        # ML_E    (nx, ny, c_num, 9, 9)
        # ML_H1H2 (nx, ny, c_num, 3, 3)
        # stress  (nx, ny, 3, 3)

        ###################  velocity_update   ######################################
        u_old = self.u.copy()
        v_old = self.v.copy()
        self.u \
            = u_old - self.dt * (u_old * self.DDx(u_old) + v_old * self.DDy(u_old)) \
            + self.dt * (self.nu * (self.D2Dx2(u_old) + self.D2Dy2(u_old)) + self.f)
        self.v \
            = v_old - self.dt * (u_old * self.DDx(v_old) + v_old * self.DDy(v_old)) \
            + self.dt * (self.nu * (self.D2Dx2(v_old) + self.D2Dy2(v_old)) + self.g)

        self.kappa[:, :, 0, 0] = self.DDx(self.u).copy()
        self.kappa[:, :, 0, 1] = self.DDy(self.u).copy()
        self.kappa[:, :, 1, 0] = self.DDx(self.v).copy()
        self.kappa[:, :, 1, 1] = self.DDy(self.v).copy()

        ###################  encoder_update   ######################################
        self.tau = stress / self.rho * self.n_p  
        dc_RHS = ML_H1H2/self.gamma
        dc_advection = np.reshape(self.kappa,[self.nx, self.ny,1,3,3]) @ self.c_tmp
        dc_source = np.reshape(np.reshape(self.kappa,[self.nx, self.ny,1,1,9]) @ ML_E,[self.nx, self.ny,self.c_num,3,3])
        
        u_tmp=np.reshape(self.u,[self.Nbin,self.Nbin,1,1,1])
        v_tmp=np.reshape(self.v,[self.Nbin,self.Nbin,1,1,1])
        upwind_c_DDx, upwind_c_DDy = self.upwind(self.c_tmp, u_tmp, v_tmp) 
        dc_material = u_tmp * upwind_c_DDx + v_tmp * upwind_c_DDy
        dc_diffusion=self.sigma*(self.D2Dx2(self.c_tmp)+self.D2Dy2(self.c_tmp))
        
        dc = dc_RHS + dc_advection + dc_advection.transpose((0, 1, 2, 4, 3)) + dc_source - dc_material +dc_diffusion

        self.c_tmp=self.c_tmp+dc*self.dt


        ###################  pressure_RHS   ######################################
        self.k \
            = (self.DDx(self.u) + self.DDy(self.v)) / self.dt \
                - self.DDx(self.u) ** 2 - 2 * self.DDy(self.u) * self.DDx(self.v) - self.DDy(self.v) ** 2 \
                + (self.D2Dx2(self.tau[:, :, 0, 0]) + 2 * self.D2Dxy(self.tau[:, :, 0, 1]) + self.D2Dy2(self.tau[:, :, 1, 1])) \
                + self.DDx(self.f) + self.DDy(self.g)
        
        ###################  pressure_Poisson_direct   ######################################
        def Poisson_periodic(nx, ny, dx, dy, f):
            # \begin{align}
            #   \nabla w & = f, & \Omega, \\
            #   \left. w \right| _{\partial _{-x} \Omega} & = \left. w \right| _{\partial _{+x} \Omega}, & \\
            #   \left. w \right| _{\partial _{-y} \Omega} & = \left. w \right| _{\partial _{+y} \Omega}. &
            # \end{align}
            # \begin{align}
            #   & \quad \( v ^{2} \kron{J _{M}}{I _{N}} + u ^{2} \kron{I _{M}}{J _{N}} - 2 \( u ^{2} + v ^{2} \) \kron{I _{M}}{I _{N}} \) w \nonumber \\
            #   & = u ^{2} v ^{2} f.
            # \end{align}
            def I(k):
                return scipy.sparse.eye(k)
            def J(k):
                data = np.ones((4, k))
                offsets = np.array([1, -1, k - 1, 1 - k])
                return scipy.sparse.spdiags(data, offsets, k, k)
            LHS \
                = dy ** 2 * scipy.sparse.kron(J(nx), I(ny)) \
                + dx ** 2 * scipy.sparse.kron(I(nx), J(ny)) \
                - 2 * (dx ** 2 + dy ** 2) * scipy.sparse.kron(I(nx), I(ny))
            RHS = dx ** 2 * dy ** 2 * f.ravel()
            w = scipy.sparse.linalg.spsolve(LHS, RHS)
            # w = scipy.linalg.solve(LHS.toarray(), RHS, overwrite_a = True, overwrite_b = True, assume_a = 'sym')
            # w = w - np.mean(w)
            w = w.reshape((nx, ny))
            return w
        self.w = Poisson_periodic(self.nx, self.ny, self.dx, self.dy, self.k)
        # self.w = self.w - np.mean(self.w)
        self.w = self.w - self.w[self.nx // 2, self.ny // 2]

        u_old = self.u.copy()
        v_old = self.v.copy()
        self.u = u_old - self.dt * self.DDx(self.w) + self.dt * (self.DDx(self.tau[:, :, 0, 0]) + self.DDy(self.tau[:, :, 0, 1]))
        self.v = v_old - self.dt * self.DDy(self.w) + self.dt * (self.DDx(self.tau[:, :, 0, 1]) + self.DDy(self.tau[:, :, 1, 1]))

    def save_data(self,save_folder): 
        savemat(save_folder+'/'+self.save_name,self.save)
        # hdf5storage.savemat(save_folder+'/'+self.save_name, self.save, format='7.3', oned_as='column')

    def DDx(self,w):
        return   (np.roll(w,-1,axis=0)-np.roll(w,1,axis=0))/self.dx/2.0
    def DDy(self,w):
        return   (np.roll(w,-1,axis=1)-np.roll(w,1,axis=1))/self.dy/2.0
    def D2Dx2(self,w):
        return   (np.roll(w,-1,axis=0)-2*w+np.roll(w,1,axis=0))/self.dx_sq
    def D2Dy2(self,w):
        return   (np.roll(w,-1,axis=1)-2*w+np.roll(w,1,axis=1))/self.dy_sq
    def D2Dxy(self,w):
        return   self.DDx(self.DDy(w))
   
    def collect_data(self,ML_E,ML_H1H2,stress,S=None):
        self.save['c'][self.save_i]=self.c_tmp
        self.save['stress'][self.save_i] = stress
        # self.save['H1H2'][self.save_i]=ML_H1H2
        # self.save['E'][self.save_i]=ML_E
        self.save['u'][self.save_i]=self.u
        self.save['v'][self.save_i]=self.v
        self.save['kenergy'][self.save_i]=1/2*self.rho*np.sum(self.u**2+self.v**2)*self.dx
        if S is not None:
            self.save['S_list'][self.save_i]=S
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
    
    def main(sim_model,model,tf):
        import time
        ds_dc_star, S0=model.get_S0()
        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)

        @tf.function
        def get_pred_data(c):
            ML_H1H2, ML_E, ML_pi_p, S,ds_dc= model.get_from_macro(c,ds_dc_star=ds_dc_star,S0=S0)
            I=tf.eye(3, batch_shape = [model.batch_size],dtype=tf.float32)
            ML_stress=model.kBT*I+ML_pi_p
            return ML_E, ML_H1H2, ML_stress, S, ds_dc
        
        Nbin=sim_model.Nbin
        xy=[Nbin,Nbin,sim_model.c_num]
        start_time=time.time() 
        for loop in range(sim_model.steps):
            c_tmp=np.reshape(sim_model.c_tmp,[Nbin*Nbin,sim_model.c_num,3,3])

            c=tf.constant(c_tmp,dtype=tf.float32)
            ML_E, ML_H1H2, ML_stress, S, ds_dc=get_pred_data(c)
            # ML_E, ML_H1H2, ML_stress, S, ds_dc=ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(), ds_dc.numpy()
            ML_E, ML_H1H2, ML_stress, S = ML_E.numpy(), ML_H1H2.numpy(), ML_stress.numpy(), S.numpy()
            ML_E, ML_H1H2, ML_stress, S = ML_E.reshape(xy+[9,9]), ML_H1H2.reshape(xy+[3,3]), ML_stress.reshape([Nbin,Nbin,3,3]), S.reshape([Nbin,Nbin])

            sim_model.one_step(ML_E,ML_H1H2,ML_stress,S)

            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress,S)
                if loop % (sim_model.save_step*100) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
        sim_model.save_data(model.save_folder)
        return sim_model

    def main_eigen(sim_model,model,tf):
        import time
        from solver_chain_v8 import StressModel as md
        model=md()

        c_init=model.get_c_init()
        sim_model.save['dn']=model.ver+f'-c{model.c_num}'
        sim_model.init_c0(c_init)

        @tf.function
        def get_pred_data(c,c_0_U):
            ML_H1, ML_H2, ML_E, stress= model.get_from_macro(c, c_0_U)
            ML_H1H2=model.kBT*ML_H1-ML_H2
            return ML_H1H2, ML_E, stress
        
        Nbin=sim_model.Nbin
        xy=[Nbin,Nbin,sim_model.c_num]
        start_time=time.time() 
        for loop in range(sim_model.steps):
            c_tmp=np.reshape(sim_model.c_tmp,[Nbin*Nbin,sim_model.c_num,3,3])

            c_0_gamma, c_0_U = np.linalg.eig(sim_model.c_tmp[:,0])        # (?, 3) unsorted eigenvalues, (?, 3, 3) unsorted eigenvectors
            c0U=tf.constant(c_0_U,dtype=model.tf_dtype)

            c=tf.constant(c_tmp,dtype=model.tf_dtype)
            ML_H1H2, ML_E, ML_stress=get_pred_data(c,c0U)
            ML_H1H2, ML_E, ML_stress = ML_H1H2.numpy(), ML_E.numpy(), ML_stress.numpy()
            ML_E, ML_H1H2, ML_stress = ML_E.reshape(xy+[9,9]), ML_H1H2.reshape(xy+[3,3]), ML_stress.reshape([Nbin,Nbin,3,3])

            sim_model.one_step(ML_E,ML_H1H2,ML_stress)

            if (loop % sim_model.save_step) == 0:
                sim_model.collect_data(ML_E,ML_H1H2,ML_stress)
                if loop % (sim_model.save_step*100) == 0:
                    print(f"step:{loop}, time:{time.time() - start_time}",flush=True)
        sim_model.save_data(model.save_folder)
        return sim_model