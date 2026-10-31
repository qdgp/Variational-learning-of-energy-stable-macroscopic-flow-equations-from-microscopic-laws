import numpy as np
import tensorflow as tf
import scipy.io, os, h5py
from solver import solver

class StressModel(solver):
    kBT=0.25
    Nbonds=6
    ver='chain-v11'
    scale=np.sqrt(0.744)

    def __init__(self,Np=2400):
        self.Np=Np
        self.weights={}
        self.S0=0.0
        S=np.eye(7,dtype=self.np_dtype)-np.diagflat(np.ones([6,1],dtype=self.np_dtype),1)
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
