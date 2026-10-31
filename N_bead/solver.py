from data_io import savemat
import numpy as np
import tensorflow as tf
import scipy.io
import os
import h5py

class solver(object):
    tf_dtype=tf.float32
    np_dtype=np.float32
    kBT=0.25
    Nbonds=6
    ver='base'
    scale=np.sqrt(0.2475)
    g1 = tf.random.Generator.from_seed(1234)

    def __init__(self,Np=2400):
        self.Np=Np
        self.weights={}
        self.S0=0.0
        S=np.array([[1,-1,0,0,0,0,0],
                    [0,1,-1,0,0,0,0],
                    [1,0,0,-1,0,0,0],
                    [0,0,0,1,-1,0,0],
                    [1,0,0,0,0,-1,0],
                    [0,0,0,0,0,1,-1]],dtype=np.float32)
        A_tmp=S@S.T
        self.A=tf.constant(A_tmp[:self.Nbonds,:self.Nbonds],dtype=self.tf_dtype)

        Rc_weight=np.ones([1,self.Nbonds])/np.sqrt(4.4543)
        self.Rc_weight =  tf.constant(Rc_weight,dtype=self.tf_dtype)
        self.MD_H1_0 = 2*tf.reduce_sum((self.Rc_weight @ self.A) * self.Rc_weight)

    def init_basis_var(self,config=None):
        c_init=tf.eye(3, batch_shape = [1,self.c_num],dtype=self.tf_dtype)
        S_x=self.get_S_basis(c_init)
        E_x,E_basis=self.get_E_basis(c_init)

        self.Sx_num=S_x.shape[1]
        self.Bx_num=E_x.shape[1]
        self.Ex_num=E_x.shape[1]
        self.Eb_num=E_basis.shape[1]
        self.Gamma_num=np.sum(np.arange(2,self.c_num+1))+1
        self.c_star=c_init

        il1 = np.tril_indices(self.c_num)
        gather_list=np.zeros([self.c_num,self.c_num],dtype=np.int32)+self.Gamma_num
        gather_list[il1]=np.arange(np.sum(np.arange(self.c_num+1)))
        self.gather_list=np.reshape(gather_list,[-1,]).tolist()

        E0=np.zeros([9,9],dtype=self.np_dtype)
        for i in [0,4,8]:
            E0[i,i]=1.0/3.0
        for i in [1,2,3,5,6,7]:
            E0[i,i]=1.0/4.0
        E0[0,4]=-1.0/3.0
        E0[0,8]=-1.0/3.0
        E0[1,3]=1.0/2.0
        E0[2,6]=1.0/2.0
        E0[4,8]=-1.0/3.0
        E0[5,7]=1.0/2.0
        E0=E0+E0.T
        E0=E0.reshape([1,9,9])

        E1=np.eye(3).reshape([1,9,1]) * np.eye(3).reshape([1,1,9])
        
        E=np.concatenate([E0,E1],axis=0)
        self.E0=tf.constant(E.reshape([2,81]),dtype=self.tf_dtype)
        self.save_var.update({'E0':E})

    def init_for_training(self,config,seed=123,stddev=0.01,B=True,E=True,S=True):
        self.seed=seed
        self.save_folder=config['save_folder']
        self.batch_size=config['batch_size']
        self.c_num=config['c_num']
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder,exist_ok=True)
        self.save_var=config.copy()  
        
        self.init_basis_var(config)
        
        self.rand = tf.initializers.RandomNormal(mean=0.0, stddev=stddev,seed=seed)
        tf.random.set_seed(seed)
        self.g1 = tf.random.Generator.from_seed(seed)

        if 'encoder' in config:
            self.init_net_encoder(config)
        if S:
            self.init_net_S(config)
        if B:
            self.init_net_B(config)
        if E:
            self.init_net_E(config)

        self.init_net_weights(config)

    def init_net_encoder(self,config):
        n_hidden=[self.Nbonds * 3 - 3]+config['encoder']+[self.c_num-1]  #  (Nbonds * 3 - 3) or (1) for dumbbell
        for i in range(len(n_hidden)-1):
            self.weights[f'encoder_h{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'encoder_h{i}')
            self.weights[f'encoder_b{i}']=tf.Variable(tf.zeros([1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'encoder_h{i}')

    def init_net_S(self,config):
        if 'S' in config:
            n_hidden=[self.Sx_num]+config['S']+[1]
            self.weights['S_y0']=tf.Variable(self.rand([n_hidden[0], n_hidden[1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name='S_y0')
            self.weights['S_b0']=tf.Variable(tf.zeros([1,n_hidden[1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name='S_b0')
            for i in range(1,len(n_hidden)-1):
                self.weights[f'S_z{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_z{i}')
                self.weights[f'S_y{i}']=tf.Variable(self.rand([n_hidden[0], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_y{i}')
                self.weights[f'S_b{i}']=tf.Variable(tf.zeros([1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_b{i}')
        if 'Sl' in config:
            n_hidden=[self.c_num]+config['Sl']+[1]
            self.weights['Sl_y0']=tf.Variable(self.rand([n_hidden[0], n_hidden[1]],dtype=self.tf_dtype),trainable=True,name='Sl_y0')
            self.weights['Sl_b0']=tf.Variable(tf.zeros([1,n_hidden[1]],dtype=self.tf_dtype),trainable=True,name='Sl_b0')
            for i in range(1,len(n_hidden)-1):
                self.weights[f'Sl_z{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'Sl_z{i}')
                self.weights[f'Sl_y{i}']=tf.Variable(self.rand([n_hidden[0], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'Sl_y{i}')
                self.weights[f'Sl_b{i}']=tf.Variable(tf.zeros([1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'Sl_b{i}')
    
    def init_net_E(self,config):
        if 'E' in config:
            n_hidden=[self.Ex_num]+config['E']+[self.Eb_num*self.Eb_num]
            for i in range(len(n_hidden)-1):
                self.weights[f'E_h{i}']=tf.Variable(self.rand([1,2*(self.c_num-1),n_hidden[i],n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'E_h{i}')
                self.weights[f'E_b{i}']=tf.Variable(tf.zeros([1,2*(self.c_num-1),1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'E_h{i}')

    def init_net_B(self,config):
        if 'B' in config:
            n_hidden=[self.Ex_num]+config['B']+[self.Eb_num*self.Eb_num]
            for i in range(len(n_hidden)-1):
                self.weights[f'B_h{i}']=tf.Variable(self.rand([1,2*self.c_num,n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_h{i}')
                self.weights[f'B_b{i}']=tf.Variable(tf.zeros([1,2*self.c_num,1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_b{i}')
            
    def init_net_weights(self,config):    
        self.weights['weight_pos']=tf.Variable(self.rand([self.c_num-1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'weight_pos')
        
        
    def get_c_init(self,xr_init=None):
        c_init=np.reshape(np.eye(3,dtype=self.np_dtype),[1,1,3,3])
        c_init=np.tile(c_init,[1,self.c_num,1,1])
        return c_init
    
    def init_rr_ref(self,xr): # [1,Np,Nbonds,3]
        rr1 = np.linalg.norm(xr, axis = 3)  # [batch_size,Np,Nbonds]
        rr2 = np.linalg.norm(xr[:,:,:-1,:] + xr[:,:,1:,:], axis = 3)  # [batch_size,Np,Nbonds-1]
        rr3 = np.linalg.norm(xr[:,:,:-2,:] + xr[:,:,1:-1,:] + xr[:,:,2:,:], axis = 3)  # [batch_size,Np,Nbonds-2]
        rr_list = np.concatenate([rr1, rr2, rr3], axis = 2)    # (batch_size, Np, Nbonds * 3 - 3)

        self.save_var.update({'xr_ref':xr})
        self.xr_ref={'rr':tf.constant(rr_list,dtype=self.tf_dtype),'xr':tf.constant(xr,dtype=self.tf_dtype),'Np':xr.shape[1]}


    def record_save(self,save_name=None):
        for key in self.weights:
            self.save_var['Net_'+key]=self.weights[key].numpy()
        if save_name is not None:
            savemat(self.save_folder+'/'+save_name, self.save_var)
        else:
            savemat(self.save_folder+'/train_result.mat', self.save_var)

    def load_model(self,batch_size,load_folder,load_file='train_result.mat'):
        data=scipy.io.loadmat(load_folder+'/'+load_file)
        self.save_folder=load_folder
        self.save_var=data.copy()

        self.c_num=int(data['c_num'])
        self.batch_size=batch_size
        for key in data:
            if key.startswith('Net_'):
                self.weights[key[4:]]=tf.Variable(np.array(data[key]),dtype=self.tf_dtype,trainable=True,name=key[4:])

        self.init_basis_var(data)
        self.init_rr_ref(np.array(data['xr_ref']))

        return data
    
    def load_model_weight(self,load_folder,load_file='train_result.mat'):
        data=scipy.io.loadmat(load_folder+'/'+load_file)
        for key in data:
            if key.startswith('Net_'):
                self.weights[key[4:]]=tf.Variable(np.array(data[key]),dtype=self.tf_dtype,trainable=True,name=key[4:])

    def load_model_weight_compatible(self,load_folder,load_file='train_result.mat'):
        data=scipy.io.loadmat(load_folder+'/'+load_file)
        for key in self.weights:
            if 'Net_'+key in data:
                key_data=np.array(data['Net_'+key])
                if self.weights[key].shape==data['Net_'+key].shape :
                    print(key,' ',self.weights[key].shape,'==',key_data.shape)
                    self.weights[key]=tf.Variable(key_data,dtype=self.tf_dtype,trainable=True,name=key)
                else:
                    print(key,' ',self.weights[key].shape,'!=',key_data.shape)

    def get_g(self,xr): # xr= [batch_size,Np,Nbonds,3]
        rr1 = tf.linalg.norm(xr, axis = 3)  # [batch_size,Np,Nbonds]
        rr2 = tf.linalg.norm(xr[:,:,:-1,:] + xr[:,:,1:,:], axis = 3)  # [batch_size,Np,Nbonds-1]
        rr3 = tf.linalg.norm(xr[:,:,:-2,:] + xr[:,:,1:-1,:] + xr[:,:,2:,:], axis = 3)  # [batch_size,Np,Nbonds-2]
        rr_list = tf.concat([rr1, rr2, rr3], axis = 2)    # (batch_size, Np, Nbonds * 3 - 3)

        g_r = self.subnetwork(rr_list, 'encoder')   # (batch_size, Np, c_num - 1)

        g_r_ref = self.subnetwork(self.xr_ref['rr'], 'encoder') # (1, Np, c_num - 1)
        weight_pos = self.weights['weight_pos'] / tf.norm(self.weights['weight_pos'],axis = 1,keepdims=True)/self.scale # (c_num - 1, Nbonds)
        yr=weight_pos @ self.xr_ref['xr']  # (1 , Np, c_num-1, 3)
        g_r_ref = tf.norm(yr*tf.expand_dims(g_r_ref,axis=-1), axis = 1,keepdims=True) / (self.xr_ref['Np'] ** 0.5) # (1 , 1, c_num-1, 3) 
        g_r_ref = tf.reduce_mean(g_r_ref,axis=-1)
        g_r = g_r / g_r_ref 
        return g_r # (batch_size , Np, c_num - 1)

    def get_g_test(self,xr): # xr= [batch_size,Np,Nbonds,3]
        # rr1 = tf.linalg.norm(xr, axis = 3)  # [batch_size,Np,Nbonds]
        # rr2 = tf.linalg.norm(xr[:,:,:-1,:] + xr[:,:,1:,:], axis = 3)  # [batch_size,Np,Nbonds-1]
        # rr3 = tf.linalg.norm(xr[:,:,:-2,:] + xr[:,:,1:-1,:] + xr[:,:,2:,:], axis = 3)  # [batch_size,Np,Nbonds-2]
        # rr_list = tf.concat([rr1, rr2, rr3], axis = 2)    # (batch_size, Np, Nbonds * 3 - 3)

        # g_r = tf.reduce_sum(rr_list*rr_list,axis=2,keepdims=True)   # (batch_size, Np, num_encoder - 1)
        # g_r_ref = tf.reduce_sum(self.xr_ref['rr']*self.xr_ref['rr'],axis=2,keepdims=True) # (1, Np, num_encoder - 1)

        # g_r_ref = tf.norm(g_r_ref, axis = 1,keepdims=True) / (self.xr_ref['Np'] ** 0.5) # (num_encoder - 1)            
        # g_r = g_r / g_r_ref 
        g_r=tf.ones([self.batch_size,self.Np,self.c_num-1],dtype=self.tf_dtype)
        return g_r # (batch_size , Np, c_num - 1)

    def get_dg(self,xr,g_r,dg):
        # xr= [batch_size,Np,Nbonds,3]
        # g_r (batch_size , Np, c_num - 1)
        dg_dxr   = tf.zeros((self.batch_size,self.Np,0,self.Nbonds,3), dtype = self.tf_dtype) 
        dg2_dxr  = tf.zeros((self.batch_size,self.Np,0,self.Nbonds,self.Nbonds), dtype = self.tf_dtype)
        for ii in range(1, self.c_num):
            dg_dxr_tmp = dg.gradient(g_r[:,:, ii - 1],xr, unconnected_gradients='zero')  # (batch_size,Np,Nbonds,3)
            dg_dxr = tf.concat([dg_dxr, tf.expand_dims(dg_dxr_tmp,axis=2)], axis = 2) 
            dg2_dxr_tmp = tf.zeros((self.batch_size,self.Np,0,self.Nbonds), dtype = self.tf_dtype)
            for jj in range(self.Nbonds):
                dg_sq_2nd_dx = dg.gradient(dg_dxr_tmp[:,:,jj,0],xr, unconnected_gradients='zero') # (batch_size,Np,Nbonds,3)
                dg_sq_2nd_dy = dg.gradient(dg_dxr_tmp[:,:,jj,1],xr, unconnected_gradients='zero') # (batch_size,Np,Nbonds,3)
                dg_sq_2nd_dz = dg.gradient(dg_dxr_tmp[:,:,jj,2],xr, unconnected_gradients='zero') # (batch_size,Np,Nbonds,3)
                dg_sq_2nd = dg_sq_2nd_dx[:,:,:,0] + dg_sq_2nd_dy[:,:,:,1] + dg_sq_2nd_dz[:,:,:,2] 
                dg2_dxr_tmp = tf.concat([dg2_dxr_tmp,tf.expand_dims(dg_sq_2nd,axis=2)],axis=2)
            dg2_dxr = tf.concat([dg2_dxr,tf.expand_dims(dg2_dxr_tmp,axis=2)],axis=2)
        return dg_dxr, dg2_dxr  # dg_dxr (batch_size,Np,c_num-1,Nbonds,3) dg2_dxr (batch_size,Np,c_num-1,Nbonds,Nbonds)

    def get_MD_H1(self,yr,g_r,dg_dxr,dg2_dxr,weight_pos): 
        # yr      (batch_size,Np,c_num  ,3)
        # g_r     (batch_size,Np,c_num-1)
        # dg_dxr  (batch_size,Np,c_num-1,Nbonds,3)
        # dg2_dxr (batch_size,Np,c_num-1,Nbonds,Nbonds)
        # weight_pos (c_num - 1, Nbonds)
        yr_tmp=yr[:,:,1:,:] # (batch_size,Np,c_num-1,3)
        dg_drk_dg_drj = tf.matmul(dg_dxr, dg_dxr,transpose_b=True) # (batch_size,Np,c_num-1,Nbonds,Nbonds)
        A_dg_drk_dg_drj = tf.reduce_sum(self.A*dg_drk_dg_drj,axis=[-1,-2]) # (batch_size,Np,c_num-1)
        A_gdg2_dxr = g_r * tf.reduce_sum(self.A*dg2_dxr,axis=[-1,-2]) # (batch_size,Np,c_num-1)

        w_dg_dr = 4* tf.expand_dims(weight_pos @ self.A,axis=2) * dg_dxr  # (batch_size,Np,c_num-1,Nbonds,3)
        w_dg_dr = tf.reduce_sum(w_dg_dr,axis=3) # (batch_size,Np,c_num-1,3)

        MD_H1_half_left = (tf.expand_dims(A_dg_drk_dg_drj + A_gdg2_dxr,axis=3) * yr_tmp + 
                           tf.expand_dims(g_r,axis=3) * w_dg_dr)  # (batch_size,Np,c_num-1,3)
        MD_H1_half = tf.matmul(tf.transpose(MD_H1_half_left,[0,2,3,1]),tf.transpose(yr_tmp,[0,2,1,3]))/float(self.Np) # (batch_size,c_num-1,3,3)
        
        w_A_w=2*tf.reduce_sum((weight_pos @ self.A) * weight_pos,axis=1)  # (c_num-1,)
        w_A_w_g_r2=w_A_w* tf.reduce_mean(g_r*g_r,axis=1) # (batch_size,c_num-1)
        w_A_w_g_r2_I=tf.reshape(w_A_w_g_r2,[self.batch_size,self.c_num-1,1,1]) *tf.eye(3,batch_shape=[1,1],dtype=self.tf_dtype)
        MD_H1_cn = MD_H1_half +tf.transpose(MD_H1_half,[0,1,3,2]) + w_A_w_g_r2_I  # (batch_size, self.c_num-1,3,3)
        MD_H1=tf.concat([self.MD_H1_0*tf.eye(3,batch_shape=[self.batch_size,1],dtype=self.tf_dtype),MD_H1_cn],axis=1)
        return MD_H1 # (batch_size, self.c_num,3,3)
    
    def get_MD_H2(self,yr,Af,g_r,dg_dxr,weight_pos_full):
        # yr      (batch_size,Np,c_num  ,3)
        # Af      (batch_size,Np,Nbonds ,3)
        # g_r     (batch_size,Np,c_num-1)
        # dg_dxr  (batch_size,Np,c_num-1,Nbonds,3)
        # weight_pos_full (c_num , Nbonds)
        
        Af_dg_dxr=tf.reduce_sum(tf.expand_dims(Af,axis=2)*dg_dxr,axis=[3,4])  #(batch_size,Np,c_num-1)
        w_Af=weight_pos_full @ Af #(batch_size,Np,c_num,3)

        Af_dg_dxr_expand=tf.concat([tf.zeros([self.batch_size,self.Np,1],dtype=self.tf_dtype),Af_dg_dxr],axis=2) #(batch_size,Np,c_num)
        g_r_expand=tf.concat([tf.ones([self.batch_size,self.Np,1],dtype=self.tf_dtype),g_r],axis=2) #(batch_size,Np,c_num)
        MD_H2_half_left = (tf.expand_dims(g_r_expand*Af_dg_dxr_expand,axis=3) * yr + 
                           tf.expand_dims(g_r_expand*g_r_expand,axis=3) * w_Af)  # (batch_size,Np,c_num,3)
        MD_H2_half = tf.matmul(tf.transpose(MD_H2_half_left,[0,2,3,1]),tf.transpose(yr,[0,2,1,3]))/float(self.Np) # (batch_size,c_num,3,3)
        
        MD_H2 = MD_H2_half +tf.transpose(MD_H2_half,[0,1,3,2])
        return MD_H2 # (batch_size, c_num,3,3)

    def get_MD_E(self, xr, yr, g_r, dg_dxr): 
        # xr      (batch_size,Np,Nbonds ,3)
        # yr      (batch_size,Np,c_num  ,3)        
        # g_r     (batch_size,Np,c_num-1)
        # dg_dxr  (batch_size,Np,c_num-1,Nbonds,3)

        xr_dg_dxr= tf.matmul(tf.expand_dims(xr,axis=2),dg_dxr,transpose_a=True) # (batch_size,Np,c_num-1, 3, 3)
        g_r_xr_dg_dxr= tf.expand_dims(g_r,axis=3) * tf.reshape(xr_dg_dxr,[self.batch_size,self.Np,self.c_num-1, 9])

        yr_tmp=tf.expand_dims(yr[:,:,1:,:],axis=4) # (batch_size,Np,c_num-1,3,1)
        yrT_yr=tf.reshape(tf.matmul(yr_tmp,yr_tmp,transpose_b=True),[self.batch_size,self.Np,self.c_num-1, 9])

        # MD_E_ci=2*tf.reduce_mean(tf.expand_dims(g_r_xr_dg_dxr,axis=4)*
        #                          tf.expand_dims(yrT_yr,axis=3),axis=1) # (batch_size, c_num-1,9,9)
        MD_E_ci=2*(tf.transpose(g_r_xr_dg_dxr,[0,2,3,1]) @ tf.transpose(yrT_yr,[0,2,1,3]))/float(self.Np)
        MD_E=tf.concat([tf.zeros([self.batch_size,1,9,9],dtype=self.tf_dtype),MD_E_ci],axis=1)
        return MD_E #[batch_size,c_num,9,9]

    def get_c(self,yr,g_r):  # yr=(batch_size,Np,c_num,3), g_r=(batch_size,Np,c_num-1)
        g_r_extend = tf.concat([tf.ones([self.batch_size,self.Np,1],dtype = self.tf_dtype),g_r],axis=-1)# (batch_size, Np, c_num) 
        gr_yr = tf.transpose(tf.expand_dims(g_r_extend,axis=3) * yr, [0,2,1,3]) # (batch_size,c_num,Np,3)
        c=tf.matmul(gr_yr,gr_yr,transpose_a=True)/self.Np  # (batch_size,c_num,3,3)
        return c # (batch_size, c_num, 3, 3)
    
    def get_c_from_xr(self,xr):
        g_r=self.get_g(xr)  # (batch_size , Np, c_num - 1)
        weight_pos = self.weights['weight_pos'] / tf.norm(self.weights['weight_pos'],axis = 1,keepdims=True)/self.scale # (c_num - 1, Nbonds)
        weight_pos_full = tf.concat([self.Rc_weight, weight_pos], axis = 0) # (c_num, Nbonds)
        yr=weight_pos_full @ xr  # (batch_size , Np, c_num, 3)

        c=self.get_c(yr,g_r)  # (batch_size, c_num, 3, 3)
        return c

    def get_S0(self, c=None):
        if c is None:
            c=self.c_star
        with tf.GradientTape(watch_accessed_variables=False) as dS:
            dS.watch(c)
            det_c=tf.linalg.det(c)
            S_x_star=self.get_S_basis(c)
            S = self.kBT*(self.subnetwork_convex(S_x_star, 'S')
                +self.subnetwork_s_linear(tf.linalg.trace(c)) 
                -1/2*tf.reduce_mean(tf.math.log(det_c),axis=-1,keepdims=True))  # (1 , 1)
        ds_dc_star = dS.gradient(S,c) # (1 , c_num,3,3)
        return ds_dc_star, S
    
    def get_S(self, c , ds_dc_star, S0): 
        # c (batch_size,c_num,3,3)
        ds_dc, S = self.get_S0(c)
        ds_dc = ds_dc-ds_dc_star # (batch_size , c_num,3,3)
        S=S-tf.expand_dims(tf.reduce_sum(ds_dc_star*(c-self.c_star),axis=[1,2,3]),1)-S0
        return S, ds_dc # s=(batch_size,1) ds_dc=(batch_size,c_num,3,3)

    def get_ML_H1H2(self, B_basis, B_x, ds_dc):
        # ds_dc (batch_size,c_num,3,3)
        # B_x (batch_size,Bx_num)
        # B_basis (batch_size,Bb_num,3,3)

        B_basis=tf.reshape(B_basis,[self.batch_size,1,self.Eb_num,9])   
        B_x = tf.reshape(B_x,[self.batch_size,1,1,self.Bx_num])  

        M_matrix = self.subnetwork(B_x, 'B') # (batch_size , c_num, 1, Bb_num^2)
        M_matrix = tf.reshape(M_matrix,[self.batch_size,2*self.c_num,self.Eb_num,self.Eb_num])
        
        M_ijkl = tf.matmul(B_basis,M_matrix,transpose_a=True) @ B_basis # (batch_size , 2*c_num, 9, 9)
        M_ijkl = tf.reshape(M_ijkl,[self.batch_size, 2, self.c_num, 3, 3, 3, 3])   
        M_list = M_ijkl[:,0] + tf.transpose(M_ijkl[:,1],[0,1,2,4,3,5]) + tf.transpose(M_ijkl[:,1],[0,1,4,2,3,5])
        
        M_list = tf.reshape(M_list,[self.batch_size,self.c_num,9,9])    
        M_list = tf.matmul(M_list,M_list,transpose_a=True)
        
        ds_dc = tf.reshape(ds_dc,[self.batch_size,self.c_num,9,1])
        ML_H1H2 = tf.reshape(M_list @ ds_dc,[self.batch_size,self.c_num,3,3])
        return ML_H1H2 # (batch_size, self.c_num,3,3)



    # def get_ML_H1H2(self, B_basis, B_x, ds_dc):
    #     # ds_dc (batch_size,c_num,3,3)
    #     # B_x (batch_size,Bx_num)
    #     # B_basis (batch_size,Bb_num,3,3)

    #     B_basis=tf.reshape(B_basis,[self.batch_size,1,self.Eb_num,9])   
    #     B_x = tf.reshape(B_x,[self.batch_size,1,1,self.Bx_num])  

    #     M_matrix = self.subnetwork(B_x, 'B') # (batch_size , c_num, 1, Bb_num^2)
    #     M_matrix = tf.reshape(M_matrix,[self.batch_size,2*self.c_num,self.Eb_num,self.Eb_num])
        
    #     M_matrix = tf.matmul(M_matrix,M_matrix,transpose_a=True)

    #     M_ijkl = tf.matmul(B_basis,M_matrix,transpose_a=True) @ B_basis # (batch_size , c_num, 9, 9)   
    #     M_ijkl = tf.reshape(M_ijkl,[self.batch_size , 2, self.c_num, 3, 3, 3, 3])
    #     M_list = M_ijkl[:,0] + tf.transpose(M_ijkl[:,1],[0,1,2,4,3,5]) # (batch_size , c_num, 3, 3, 3, 3)   
    #     M_list = tf.reshape(M_list,[self.batch_size,self.c_num,9,9])    

    #     ds_dc = tf.reshape(ds_dc,[self.batch_size,self.c_num,9,1])
    #     ML_H1H2 = tf.reshape(M_list @ ds_dc,[self.batch_size,self.c_num,3,3])
    #     return ML_H1H2 # (batch_size, self.c_num,3,3)

    def get_ML_E(self, B_basis, B_x): # c_basis=(batch_size,c_basis_num,3,3)
        B_basis=tf.reshape(B_basis,[self.batch_size,1,self.Eb_num,9])   
        B_x = tf.reshape(B_x,[self.batch_size,1,1,self.Ex_num])  

        E_matrix = self.subnetwork(B_x, 'E') # (batch_size , c_num-1, 1, Bb_num^2)
        E_matrix = tf.reshape(E_matrix,[self.batch_size,2*(self.c_num-1),self.Eb_num,self.Eb_num])
        E_ijkl = tf.matmul(B_basis,E_matrix,transpose_a=True) @ B_basis # (batch_size , c_num-1, 9, 9)   
        
        E_ijkl = tf.reshape(E_ijkl,[self.batch_size, 2, self.c_num-1, 3, 3, 3, 3])
        E_list = (E_ijkl[:,0]   + tf.transpose(E_ijkl[:,1],[0,1,2,4,3,5]) 
                                + tf.transpose(E_ijkl[:,1],[0,1,4,2,5,3])
                                + tf.transpose(E_ijkl[:,1],[0,1,2,4,5,3])
                                + tf.transpose(E_ijkl[:,1],[0,1,4,2,3,5]))
        E_list = tf.reshape(E_list,[self.batch_size, self.c_num-1, 9, 9])  

        E_list=tf.concat([tf.zeros([self.batch_size,1,9,9],dtype=self.tf_dtype),E_list],axis=1)
        return E_list # (batch_size, c_num,9 ,9)

    def get_pi_p(self,c,ds_dc,E_list):
        E_dsdc=tf.reduce_sum(E_list @ tf.reshape(ds_dc,[self.batch_size,self.c_num,9,1]),axis=1)
        ML_pi_p=2.0*tf.reduce_sum(c @ ds_dc,axis=1)+tf.reshape(E_dsdc,[self.batch_size,3,3])
        return ML_pi_p # (batch_size,3,3)

    # def get_S_basis(self,c):
    #     tr_c, det_c, c_inv=self.get_inv_c(c)
    #     x2=tf.linalg.trace(c_inv)
    #     x3=-tf.math.log(det_c)
    #     S_x=tf.concat([tr_c,x2,x3], axis = -1) # (batch_size , c_num*3)         
    #     return S_x
    
    # def get_E_basis(self,c):
    #     I=tf.eye(3, batch_shape = [c.shape[0],1],dtype=self.tf_dtype)
    #     cx=c
    #     for i in range(c.shape[1]-1):
    #         c_tmp=tf.expand_dims(c[:,i,:,:], axis=1)-c[:,i+1:,:,:]
    #         cx=tf.concat([cx,I+c_tmp @ c_tmp], axis=1)
    #     tr_cx, det_cx, cx_inv=self.get_inv_c(cx)
    #     E_basis=tf.concat([I,cx,cx_inv],axis=1)
    #     E_x=tf.concat([tr_cx,tf.linalg.trace(cx_inv),-tf.math.log(det_cx)], axis = -1)    
    #     return E_x,E_basis

    def get_S_basis(self,c):
        I=tf.eye(3, batch_shape = [c.shape[0],1],dtype=self.tf_dtype)
        tr_c=tf.linalg.trace(c)
        tr_c2=tf.linalg.trace((c-I) @ (c-I)) / 100.0
        tr_c3=tf.linalg.trace(c @ c @ c) / 1000.0

        for i in range(c.shape[1]-1):
            c_tmp= tf.expand_dims(c[:,i,:,:], axis=1) - c[:,i+1:,:,:]
            tr_c2=tf.concat([tr_c2,tf.linalg.trace(c_tmp @ c_tmp)/100.0], axis=1)

        S_x=tf.concat([tr_c,tr_c2,tr_c3], axis = -1)    
        return S_x

    def get_E_basis(self,c):
        I=tf.eye(3, batch_shape = [c.shape[0],1],dtype=self.tf_dtype)
        
        tr_c=tf.linalg.trace(c)-3.0
        c2 = (c-I) @ (c-I)
        tr_c2=tf.linalg.trace(c2)
        c3 = (c-I) @ (c-I) @ (c-I)
        tr_c3=tf.linalg.trace(c3)

        cx=tf.zeros([c.shape[0],0,3,3],dtype=self.tf_dtype)
        for i in range(c.shape[1]-1):
            c_tmp= (tf.expand_dims(c[:,i,:,:], axis=1)-I) @ (c[:,i+1:,:,:]-I)
            cx=tf.concat([cx,(c_tmp + tf.transpose(c_tmp,[0,1,3,2]))/2.0], axis=1)
        E_basis=tf.concat([I,c,c2,cx],axis=1)
        E_x=tf.concat([tr_c,tr_c2,tf.linalg.trace(cx)], axis = -1)    
        return E_x,E_basis

    def get_tr_c(self,c):
        shape=c.shape[:-2]+[1,1]
        I=tf.eye(3,batch_shape=c.shape[:-2],dtype=self.tf_dtype)
        tr_c=tf.linalg.trace(c)
        tr_c2=tf.linalg.trace(c @ c)
        x2=0.5*tr_c**2-tr_c2
        c2= tf.reshape(tr_c**2,shape)*I/3 - (c @ c)
        return tr_c, x2, c2

    def get_inv_c(self,c):
        shape=c.shape[:-2]+[1,1]
        I=tf.eye(3,batch_shape=c.shape[:-2],dtype=self.tf_dtype)
        tr_c=tf.linalg.trace(c)
        tr_c2=tf.linalg.trace(c @ c)
        det_c=tf.linalg.det(c)
        c_inv=(c @ c-tf.reshape(tr_c,shape)*c+
                0.5*tf.reshape(tr_c**2-tr_c2,shape)*I)/tf.reshape(det_c,shape)
        return tr_c, det_c, c_inv

    def get_from_micro_inner(self,xr,Af):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as dg:
            dg.watch(xr)
            g_r=self.get_g(xr)  # (batch_size , Np, c_num - 1)
            dg_dxr, dg2_dxr=self.get_dg(xr,g_r,dg) # (batch_size , Np, c_num - 1)

        weight_pos = self.weights['weight_pos'] / tf.norm(self.weights['weight_pos'],axis = 1,keepdims=True)/self.scale # (c_num - 1, Nbonds)
        weight_pos_full = tf.concat([self.Rc_weight, weight_pos], axis = 0) # (c_num, Nbonds)
        yr=weight_pos_full @ xr  # (batch_size , Np, c_num, 3)

        c=self.get_c(yr,g_r)  # (batch_size, c_num, 3, 3)

        MD_H1=self.get_MD_H1(yr, g_r, dg_dxr, dg2_dxr, weight_pos)
        MD_H2=self.get_MD_H2(yr, Af, g_r, dg_dxr, weight_pos_full)  
        MD_E=self.get_MD_E(xr, yr, g_r, dg_dxr)
        return c, MD_H1, MD_H2 , MD_E

    def get_from_micro(self,xr,ff):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        
        Af=self.A @ ff
        c, MD_H1, MD_H2 , MD_E=self.get_from_micro_inner(xr,Af)
        # index=[ [0,1,2,3,4,5],
        #         [0,1,4,5,2,3],
        #         [2,3,0,1,4,5],
        #         [2,3,4,5,0,1],
        #         [4,5,0,1,2,3],
        #         [4,5,2,3,0,1]]
        
        # c,MD_H1,MD_H2,MD_E=0,0,0,0
        # for i in index:        
        #     xr_new=tf.gather(xr, i, axis=2)
        #     Af=self.A @ tf.gather(ff, i, axis=2)
        #     c_1, MD_H1_1, MD_H2_1 , MD_E_1=self.get_from_micro_inner(xr_new,Af)
        #     c = c+c_1/6.0
        #     MD_H1 = MD_H1+MD_H1_1/6.0
        #     MD_H2 = MD_H2+MD_H2_1/6.0
        #     MD_E = MD_E+MD_E_1/6.0

        return c, MD_H1, MD_H2 , MD_E










    def get_from_macro(self,c,MD_E=None,ds_dc_star=None,S0=None):
        if S0 is None:
            ds_dc_star, S0=self.get_S0()
        S, ds_dc=self.get_S(c,ds_dc_star, S0)

        E_x,E_basis=self.get_E_basis(c)
        ML_E=self.get_ML_E(E_basis, E_x)
        ML_H1H2=-self.get_ML_H1H2(E_basis, E_x, ds_dc)

        if MD_E is not None:
            ML_pi_p=self.get_pi_p(c,ds_dc,MD_E)
        else:
            ML_pi_p=self.get_pi_p(c,ds_dc,ML_E)
        return ML_H1H2, ML_E, ML_pi_p, S, ds_dc

    def subnetwork(self, hiddens, key):
        i=0
        while key+'_h{}'.format(i+1) in self.weights:
            hiddens = tf.add(tf.matmul(hiddens, self.weights[key+'_h{}'.format(i)]), self.weights[key+'_b{}'.format(i)])
            hiddens = tf.nn.tanh(hiddens)
            i=i+1
        if key+'_b{}'.format(i) in self.weights:
            output = tf.matmul(hiddens, self.weights[key+'_h{}'.format(i)]) + self.weights[key+'_b{}'.format(i)]
        else:
             output = tf.matmul(hiddens, self.weights[key+'_h{}'.format(i)])           
        return output 
    
    def subnetwork_convex(self, y, key):
        i=0
        hiddens = (y @ tf.abs(self.weights[key+f'_y{i}'])+ 
                   self.weights[key+f'_b{i}'])
        hiddens = tf.nn.softplus(hiddens)
        i=i+1
        while key+'_b{}'.format(i) in self.weights:
            hiddens = (hiddens @ tf.abs(self.weights[key+f'_z{i}'])+
                       y @ tf.abs(self.weights[key+f'_y{i}'])+
                       self.weights[key+f'_b{i}'])
            hiddens = tf.nn.softplus(hiddens)
            i=i+1
        output=hiddens
        return output
    
    def subnetwork_s_linear(self, y):
        hiddens = tf.nn.softplus(y @ self.weights['Sl_y0']+self.weights['Sl_b0'])
        i=1
        while f'Sl_b{i}' in self.weights:
            hiddens = tf.nn.softplus(hiddens @ tf.abs(self.weights[f'Sl_z{i}'])+
                       y @ self.weights[f'Sl_y{i}']+
                       self.weights[f'Sl_b{i}'])
            i=i+1
        return hiddens