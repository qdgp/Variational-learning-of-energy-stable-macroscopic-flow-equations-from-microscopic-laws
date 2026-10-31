import numpy as np
import tensorflow as tf
import scipy.io, os, h5py
from solver import solver

class StressModel(solver):
    kBT=0.25
    Nbonds=6
    ver='chain-v17'
    scale=np.sqrt(0.744)

    def __init__(self,Np=2400):
        self.Np=Np
        self.weights={}
        self.S0=0.0
        S=np.eye(self.Nbonds+1,dtype=self.np_dtype)-np.diagflat(np.ones([self.Nbonds,1],dtype=self.np_dtype),1)
        S=S[:self.Nbonds]
        self.A=tf.constant(S@S.T,dtype=self.np_dtype)

        Rc_weight=np.ones([1,self.Nbonds])/np.sqrt(self.Nbonds)/self.scale
        self.Rc_weight =  tf.constant(Rc_weight,dtype=self.tf_dtype)
        self.MD_H1_0 = 2*tf.reduce_sum((self.Rc_weight @ self.A) * self.Rc_weight)

    def init_net_B(self,config):
        if 'B' in config:
            n_hidden=[self.Ex_num]+config['B']+[self.Eb_num*self.Eb_num]
            for i in range(len(n_hidden)-1):
                self.weights[f'B_h{i}']=tf.Variable(self.rand([1,self.c_num,n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_h{i}')
                self.weights[f'B_b{i}']=tf.Variable(tf.zeros([1,self.c_num,1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_b{i}')

    def init_net_S(self,config):
        if 'S' in config:
            n_hidden=[self.Sx_num]+config['S']+[1]
            self.weights['S_y0']=tf.Variable(self.rand([1, self.c_num, n_hidden[0], n_hidden[1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name='S_y0')
            self.weights['S_b0']=tf.Variable(tf.zeros([1, self.c_num, 1,n_hidden[1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name='S_b0')
            for i in range(1,len(n_hidden)-1):
                self.weights[f'S_z{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_z{i}')
                self.weights[f'S_y{i}']=tf.Variable(self.rand([n_hidden[0], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_y{i}')
                self.weights[f'S_b{i}']=tf.Variable(tf.zeros([1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'S_b{i}')

    def init_net_weights(self,config):    
        self.weights['weight_pos']=tf.Variable(self.rand([self.c_num-1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'weight_pos')
        self.weights['S_weight']=tf.Variable(tf.ones([self.c_num,1],dtype=self.tf_dtype)/self.c_num,dtype=self.tf_dtype,trainable=True,name=f'S_weight')
    
    def get_from_micro(self,xr,ff):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        Af=self.A @ ff
        c_1, MD_H1_1, MD_H2_1 , MD_E_1=self.get_from_micro_inner(xr,Af)
        
        xr=tf.reverse(xr,axis=[2])
        ff=tf.reverse(ff,axis=[2])
        Af=self.A @ ff
        c_2, MD_H1_2, MD_H2_2 , MD_E_2=self.get_from_micro_inner(xr,Af)
        c = (c_1+c_2)/2.0
        MD_H1 = (MD_H1_1+MD_H1_2)/2.0
        MD_H2 = (MD_H2_1+MD_H2_2)/2.0
        MD_E = (MD_E_1+MD_E_2)/2.0
        return c, MD_H1, MD_H2 , MD_E

    def get_ML_H1H2(self, c_basis, Tr_c, ds_dc):
        # ds_dc=(batch_size,c_num,3,3) c_basis=(batch_size,Eb_num,3,3)
        c_basis = tf.reshape(c_basis,[self.batch_size,1,self.Eb_num,9])   
        Tr_c = tf.reshape(Tr_c,[self.batch_size,1,1,self.Ex_num])  

        M_matrix = self.subnetwork(Tr_c, 'B') # (batch_size, c_num,1, Eb_num^2)
        M_matrix = tf.reshape(M_matrix,[self.batch_size,self.c_num,self.Eb_num,self.Eb_num])
        M_matrix = tf.matmul(M_matrix,M_matrix,transpose_a=True)
        M_tmp = tf.matmul(c_basis,M_matrix,transpose_a=True) @ c_basis # (batch_size, c_num,9,9)

        ds_dc = tf.reshape(ds_dc,[self.batch_size,self.c_num,9,1])
        ML_H1H2 = tf.reshape(M_tmp @ ds_dc,[self.batch_size,self.c_num,3,3])
        return ML_H1H2 # (batch_size, self.c_num,3,3)

    def get_S0(self, c=None):
        if c is None:
            c=self.c_star
        with tf.GradientTape(watch_accessed_variables=False) as dS:
            dS.watch(c)
            det_c=tf.linalg.det(c)
            S_x_star=self.get_S_basis(c)
            S_x_star=tf.expand_dims(S_x_star,axis=-2)
            S_x= self.subnetwork_convex(S_x_star, 'S') # (bs , c_num, 1, 1)
            S = self.kBT*( tf.math.reduce_sum(S_x,axis=[1,2]) 
                -1/2*(tf.math.log(det_c) @ tf.abs(self.weights['S_weight'])) )  # (bs , 1)
        ds_dc_star = dS.gradient(S,c) # (bs , c_num,3,3)
        return ds_dc_star, S

    def get_S_basis(self,c):
        I=tf.eye(3, batch_shape = [c.shape[0],1],dtype=self.tf_dtype)
        tr_c=tf.expand_dims(tf.linalg.trace(c), axis=-1)
        tr_c2=tf.expand_dims(tf.linalg.trace((c-I) @ (c-I)) / 100.0, axis=-1)
        S_x=tf.concat([tr_c,tr_c2], axis = -1)
        return S_x

    def get_E_basis(self,c):
        I=tf.eye(3, batch_shape = [c.shape[0],1],dtype=self.tf_dtype)
        
        tr_c=tf.linalg.trace(c)-3.0
        c2 = (c-I) @ (c-I) /100.0
        tr_c2=tf.linalg.trace(c2)
        c3 = ((c-I) @ (c-I) @ (c-I))/1000.0
        tr_c3 = tf.linalg.trace(c3)

        cx=tf.zeros([c.shape[0],0,3,3],dtype=self.tf_dtype)
        for i in range(c.shape[1]-1):
            c_tmp= (tf.expand_dims(c[:,i,:,:], axis=1)-I) @ (c[:,i+1:,:,:]-I) / 100.0
            cx=tf.concat([cx,(c_tmp + tf.transpose(c_tmp,[0,1,3,2]))/2.0], axis=1)
        E_basis=tf.concat([I,c,c2,c3,cx],axis=1)
        E_x=tf.concat([tr_c,tr_c2,tr_c3,tf.linalg.trace(cx)], axis = -1)    
        return E_x,E_basis
    
    def get_from_micro_with_MD_M(self,xr,ff):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        Af=self.A @ ff
        c_1, MD_H1_1, MD_H2_1 , MD_E_1, MD_M_1=self.get_from_micro_inner_with_MD_M(xr,Af)
        
        xr=tf.reverse(xr,axis=[2])
        ff=tf.reverse(ff,axis=[2])
        Af=self.A @ ff
        c_2, MD_H1_2, MD_H2_2 , MD_E_2, MD_M_2=self.get_from_micro_inner_with_MD_M(xr,Af)
        c = (c_1+c_2)/2.0
        MD_H1 = (MD_H1_1+MD_H1_2)/2.0
        MD_H2 = (MD_H2_1+MD_H2_2)/2.0
        MD_E = (MD_E_1+MD_E_2)/2.0
        MD_M = (MD_M_1+MD_M_2)/2.0
        return c, MD_H1, MD_H2, MD_E, MD_M

    def get_MD_M(self,yr,g_r,dg_dxr,weight_pos_full):
        # yr      (batch_size,Np,c_num  ,3)
        # g_r     (batch_size,Np,c_num-1)
        # dg_dxr  (batch_size,Np,c_num-1,Nbonds,3)
        # weight_pos_full (c_num , Nbonds)

        # 2 g_p dg_{p}dr_{mn} y_{p,i} y_{p,j} + 
        # g_{p}^2 \omega_{p,m} \delta_{in} \mb{y}_{p,j}+ 
        # g_{p}^2 \omega_{p,m} \delta_{jn} \mb{y}_{p,i}
        # db_mn = [batch_size,Np,c_num,Nbonds,3,3,3]
        # db_pmn = [             p    ,m     ,n]

        g_r_full = tf.concat([tf.ones([self.batch_size,self.Np,1],dtype=self.tf_dtype),g_r],axis=2) #(batch_size,Np,c_num)
        dg_dxr_full = tf.concat([tf.zeros([self.batch_size,self.Np,1,self.Nbonds,3],dtype=self.tf_dtype),dg_dxr],axis=2) #(batch_size,Np,c_num)
       
        yr = tf.reshape(yr,[self.batch_size,self.Np,self.c_num,1,1,3])
        g_r_full = tf.reshape(g_r_full,[self.batch_size,self.Np,self.c_num,1,1,1,1])
        dg_dxr_full = tf.reshape(dg_dxr_full,[self.batch_size,self.Np,self.c_num,self.Nbonds,3,1,1])

        I = tf.eye(3,batch_shape=[1,1,1,1],dtype=self.tf_dtype)
        g_r_dg_dxr = 2 * g_r_full * dg_dxr_full * tf.expand_dims(yr,axis=5) * tf.expand_dims(yr,axis=6)
        g2_w_iy = (g_r_full * g_r_full * tf.reshape(weight_pos_full,[1,1,self.c_num,self.Nbonds,1,1,1])
                   *(tf.expand_dims(I,axis=5)* tf.expand_dims(yr,axis=6) + tf.expand_dims(I,axis=6)* tf.expand_dims(yr,axis=5)))
        
        db = tf.reshape(g_r_dg_dxr + g2_w_iy,[self.batch_size,self.Np,self.c_num,self.Nbonds,3,9]) # [batch_size,Np,c_num,Nbonds,3,9]
        
        MD_M = tf.zeros([self.batch_size, 9, 9],dtype=self.tf_dtype)
        for i in range(3):
            Adb = self.A @ db[:,:,1,:,i,:] # [batch_size,Np,Nbonds,9]
            db_Adb = tf.matmul(db[:,:,0,:,i,:], Adb,transpose_a=True) # [batch_size,Np,9,9]
            MD_M = MD_M +  tf.reduce_mean(db_Adb,axis=1) # [batch_size,9,9]
        return MD_M # (batch_size, c_num,c_num,9,9)