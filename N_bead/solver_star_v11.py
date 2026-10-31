import numpy as np
import tensorflow as tf
import scipy.io, os, h5py
from solver import solver

class StressModel(solver):
    kBT=0.25
    Nbonds=6
    ver='star-v11'
    scale=np.sqrt(0.74239)

    def __init__(self,Np=2400):
        self.Np=Np
        self.weights={}
        self.S0=0.0
        S=np.array([[1,-1,0,0,0,0,0],
                    [0,1,-1,0,0,0,0],
                    [1,0,0,-1,0,0,0],
                    [0,0,0,1,-1,0,0],
                    [1,0,0,0,0,-1,0],
                    [0,0,0,0,0,1,-1]],dtype=self.np_dtype)
        self.A=tf.constant(S@S.T,dtype=self.tf_dtype)


    def init_net_B(self,config):
        if 'B' in config:
            n_hidden=[self.Ex_num]+config['B']+[self.Eb_num*self.Eb_num]
            for i in range(len(n_hidden)-1):
                self.weights[f'B_h{i}']=tf.Variable(self.rand([1,self.c_num,n_hidden[i], n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_h{i}')
                self.weights[f'B_b{i}']=tf.Variable(tf.zeros([1,self.c_num,1,n_hidden[i+1]],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'B_b{i}')

    def init_net_weights(self,config):    
        self.weights['weight_pos']=tf.Variable(self.rand([self.c_num-1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'weight_pos')
        self.weights['c0_weight']=tf.Variable(self.rand([1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'c0_weight')
        self.weights['S_weight']=tf.Variable(tf.ones([self.c_num,1],dtype=self.tf_dtype)/self.c_num,dtype=self.tf_dtype,trainable=True,name=f'S_weight')

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
        
        Rc_weight=self.weights['c0_weight']/tf.norm(self.weights['c0_weight'],axis = 1,keepdims=True)/self.scale
        MD_H1_0 = 2*tf.reduce_sum((Rc_weight @ self.A) * Rc_weight)
        
        MD_H1=tf.concat([MD_H1_0*tf.eye(3,batch_shape=[self.batch_size,1],dtype=self.tf_dtype),MD_H1_cn],axis=1)
        return MD_H1 # (batch_size, self.c_num,3,3)

    def get_from_micro_inner(self,xr,Af):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        with tf.GradientTape(watch_accessed_variables=False,persistent=True) as dg:
            dg.watch(xr)
            g_r=self.get_g(xr)  # (batch_size , Np, c_num - 1)
            dg_dxr, dg2_dxr=self.get_dg(xr,g_r,dg) # (batch_size , Np, c_num - 1)

        Rc_weight=self.weights['c0_weight']/tf.norm(self.weights['c0_weight'],axis = 1,keepdims=True)/self.scale
        weight_pos = self.weights['weight_pos'] / tf.norm(self.weights['weight_pos'],axis = 1,keepdims=True)/self.scale # (c_num - 1, Nbonds)
        weight_pos_full = tf.concat([Rc_weight, weight_pos], axis = 0) # (c_num, Nbonds)
        yr=weight_pos_full @ xr  # (batch_size , Np, c_num, 3)

        c=self.get_c(yr,g_r)  # (batch_size, c_num, 3, 3)

        MD_H1=self.get_MD_H1(yr, g_r, dg_dxr, dg2_dxr, weight_pos)
        MD_H2=self.get_MD_H2(yr, Af, g_r, dg_dxr, weight_pos_full)  
        MD_E=self.get_MD_E(xr, yr, g_r, dg_dxr)
        return c, MD_H1, MD_H2 , MD_E

    
    def get_from_micro(self,xr,ff):
        # xr= [batch_size,Np,Nbonds,3]
        # Af= [batch_size,Np,Nbonds,3]
        rand=self.g1.uniform([2],minval=0,maxval=6,dtype=tf.dtypes.int32,name=None)
        index=tf.constant([ [0,1,2,3,4,5],
                            [0,1,4,5,2,3],
                            [2,3,0,1,4,5],
                            [2,3,4,5,0,1],
                            [4,5,0,1,2,3],
                            [4,5,2,3,0,1]],dtype=tf.int32)
        
        Af=self.A @ tf.gather(ff, index[rand[0]], axis=2)
        xr_tmp=tf.gather(xr, index[int(rand[0])], axis=2)
        c_1, MD_H1_1, MD_H2_1 , MD_E_1=self.get_from_micro_inner(xr_tmp,Af)

        Af=self.A @ tf.gather(ff, index[rand[1]], axis=2)
        xr_tmp=tf.gather(xr, index[int(rand[1])], axis=2)
        c_2, MD_H1_2, MD_H2_2 , MD_E_2=self.get_from_micro_inner(xr_tmp,Af)
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

    def get_c_from_xr(self,xr):
        g_r=self.get_g(xr)  # (batch_size , Np, c_num - 1)
        Rc_weight=self.weights['c0_weight']/tf.norm(self.weights['c0_weight'],axis = 1,keepdims=True)/self.scale
        weight_pos = self.weights['weight_pos'] / tf.norm(self.weights['weight_pos'],axis = 1,keepdims=True)/self.scale # (c_num - 1, Nbonds)
        weight_pos_full = tf.concat([Rc_weight, weight_pos], axis = 0) # (c_num, Nbonds)
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
                -1/2*(tf.math.log(det_c) @ tf.abs(self.weights['S_weight'])) )  # (1 , 1)
        ds_dc_star = dS.gradient(S,c) # (1 , c_num,3,3)
        return ds_dc_star, S

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
        c3 = ((c-I) @ (c-I) @ (c-I))
        tr_c3 = tf.linalg.trace(c3)

        cx=tf.zeros([c.shape[0],0,3,3],dtype=self.tf_dtype)
        for i in range(c.shape[1]-1):
            c_tmp= (tf.expand_dims(c[:,i,:,:], axis=1)-I) @ (c[:,i+1:,:,:]-I)
            cx=tf.concat([cx,(c_tmp + tf.transpose(c_tmp,[0,1,3,2]))/2.0], axis=1)
        E_basis=tf.concat([I,c,c2,c3,cx],axis=1)
        E_x=tf.concat([tr_c,tr_c2,tr_c3,tf.linalg.trace(cx)], axis = -1)    
        return E_x,E_basis