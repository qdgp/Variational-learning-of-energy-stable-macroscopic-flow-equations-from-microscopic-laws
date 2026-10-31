import numpy as np
import tensorflow as tf
import scipy.io, os, h5py
from solver import solver

class StressModel(solver):
    kBT=0.25
    Nbonds=6
    ver='chain-v8-24eign' # 24 eigen
    scale=np.sqrt(0.744)

    def __init__(self,Np=2400):
        self.Np=Np
        self.weights={}
        self.S0=0.0
        S=np.eye(7,dtype=self.np_dtype)-np.diagflat(np.ones([6,1],dtype=self.np_dtype),1)
        A_tmp=S@S.T
        self.A=tf.constant(A_tmp[:self.Nbonds,:self.Nbonds],dtype=self.np_dtype)

        eorder = np.array([[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,1,0],[2,0,1]])
        esign =  np.array([[1,1,1],[1,-1,1],[1,1,-1],[1,-1,-1]])
        self.eigen_order, self.eigen_order_T, self.eigen_sign,self.eigen_sign_T, self.eigen_num=self.get_eigen(eorder,esign)

    def init_net_S(self,config):
        return
    
    def init_net_E(self,config):
        if 'E' in config:
            n_hidden=[self.c_num*6]+config['E']+[72]
            for ii in range(1,self.c_num):
                for i in range(0, len(n_hidden)-1):
                    self.weights[f'E{ii}_h{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]]),trainable=True,name=f'E{ii}_h{i}')
                    self.weights[f'E{ii}_b{i}']=tf.Variable(tf.zeros([n_hidden[i+1]]),trainable=True,name=f'E{ii}_h{i}')

    def init_net_B(self,config):
        key = 'stress'
        n_hidden=[self.c_num*6]+config[key]+[6]
        for i in range(0, len(n_hidden)-1):
            self.weights[key+f'_h{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]]),trainable=True,name=key+f'_h{i}')
            self.weights[key+f'_b{i}']=tf.Variable(tf.zeros([n_hidden[i+1]]),trainable=True,name=key+f'_b{i}')
        key = 'H2'
        n_hidden=[self.c_num*6]+config[key]+[6*self.c_num]
        for i in range(0, len(n_hidden)-1):
            self.weights[key+f'_h{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]]),trainable=True,name=key+f'_h{i}')
            self.weights[key+f'_b{i}']=tf.Variable(tf.zeros([n_hidden[i+1]]),trainable=True,name=key+f'_b{i}')
        key = 'H1'
        n_hidden=[self.c_num*6]+config[key]+[6*(self.c_num-1)]
        for i in range(0, len(n_hidden)-1):
            self.weights[key+f'_h{i}']=tf.Variable(self.rand([n_hidden[i], n_hidden[i+1]]),trainable=True,name=key+f'_h{i}')
            self.weights[key+f'_b{i}']=tf.Variable(tf.zeros([n_hidden[i+1]]),trainable=True,name=key+f'_b{i}')


    def init_net_weights(self,config):    
        self.weights['weight_pos']=tf.Variable(self.rand([self.c_num-1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'weight_pos')
        self.weights['c0_weight']=tf.Variable(self.rand([1, self.Nbonds],dtype=self.tf_dtype),dtype=self.tf_dtype,trainable=True,name=f'c0_weight')

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

    def get_from_macro(self, c, c_0_U):
        U_T_c_U_flat=self.get_c_with_eigen(c, c_0_U)
        
        stress=self.get_stress(U_T_c_U_flat,c_0_U)
        ML_E=self.get_ML_E(U_T_c_U_flat,c_0_U)
        ML_H2=self.get_ML_H2(U_T_c_U_flat,c_0_U)
        ML_H1=self.get_ML_H1(U_T_c_U_flat,c_0_U)
        return ML_H1, ML_H2, ML_E, stress

    def get_eigen(self,eorder,esign):
        eorder_num=eorder.shape[0]
        esign_num=esign.shape[0]
        eigen_num=eorder_num*esign_num

        eigen_order=np.zeros((eigen_num,6))
        eigen_sign=np.zeros((eigen_num,6))
        eigen_order_T=np.zeros((eigen_num,6))
        eigen_sign_T=np.zeros((eigen_num,6))

        C=np.array([[0,1,2],[1,4,5],[2,5,8]])
        CT=np.array([[0,1,2],[1,3,4],[2,4,5]])
        for i in range(eorder_num):
            for j in range(esign_num):
                U=np.eye(3)
                U=np.matmul(U[:,eorder[i,:]],np.diag(esign[j,:]))
                UTCU=U.T @ C @ U
                eigen_order[i*esign_num+j,:]=np.abs([UTCU[0,0],UTCU[1,0],UTCU[2,0],UTCU[1,1],UTCU[2,1],UTCU[2,2]])
                eigen_sign[i*esign_num+j,:]=np.sign([UTCU[0,0]+0.5,UTCU[1,0]+0.5,UTCU[2,0]+0.5,UTCU[1,1]+0.5,UTCU[2,1]+0.5,UTCU[2,2]+0.5])     
                
                # UCUT=U @ CT @ U.T    
                UCUT=U @ CT @ U.T  
                eigen_order_T[i*esign_num+j,:]=np.abs([UCUT[0,0],UCUT[1,0],UCUT[2,0],UCUT[1,1],UCUT[2,1],UCUT[2,2]])
                eigen_sign_T[i*esign_num+j,:]=np.sign([UCUT[0,0]+0.5,UCUT[1,0]+0.5,UCUT[2,0]+0.5,UCUT[1,1]+0.5,UCUT[2,1]+0.5,UCUT[2,2]+0.5])

        eigen_order = tf.constant(eigen_order, dtype = tf.int32)
        eigen_order_T = tf.constant(eigen_order_T, dtype = tf.int32)
        eigen_sign = tf.constant(eigen_sign, dtype = self.tf_dtype)
        eigen_sign_T = tf.constant(eigen_sign_T, dtype = self.tf_dtype)  
        
        return eigen_order, eigen_order_T, eigen_sign,eigen_sign_T, eigen_num
    
    def get_c_with_eigen(self, c, c_0_U):
        c_0_U=tf.expand_dims(c_0_U,axis=1)
        U_T_c_U = tf.matmul(c_0_U, c, transpose_a = True) @ c_0_U  # (batch_size, c_num, 3, 3)
        U_T_c_U_flat = tf.zeros((0,6*self.c_num), dtype = self.tf_dtype)
        for jj in range(self.eigen_num):
            flat = tf.gather(tf.reshape(U_T_c_U,[self.batch_size, self.c_num, 9]),
                             self.eigen_order[jj,:], axis = -1)*self.eigen_sign[jj,:] # (batch_size,c_num,6)
            flat=tf.reshape(flat,[self.batch_size, 6*self.c_num])
            U_T_c_U_flat = tf.concat([U_T_c_U_flat, flat], axis = 0) # (eigen_num*batch_size,c_num*6) 
        return U_T_c_U_flat

    def get_stress(self, U_T_c_U_flat,c_0_U):
        stress=tf.zeros((self.batch_size, 6), dtype = self.tf_dtype)
        stress_out=self.subnetwork(U_T_c_U_flat, 'stress') # (eigen_num*batch_size, 6) 
        stress_out=tf.reshape(stress_out,[self.eigen_num,self.batch_size,6])
        for jj in range(self.eigen_num):
            stress = stress + tf.gather(stress_out[jj,:,:], self.eigen_order_T[jj,:], axis = -1)*tf.reshape(self.eigen_sign_T[jj,:],[1,6])
        stress = c_0_U @ tf.matmul(self.flat2mtx(stress/self.eigen_num), c_0_U, transpose_b = True)
        return stress # (batch_size, 3,3)
    
    def get_ML_H2(self, U_T_c_U_flat,c_0_U):
        ML_H2_out=self.subnetwork(U_T_c_U_flat, 'H2') # (batch_size*eigen_num,6*c_num)
        ML_H2_out=tf.reshape(ML_H2_out,[self.eigen_num,self.batch_size,self.c_num,6])

        ML_H2=tf.zeros((self.batch_size,self.c_num,6), dtype = self.tf_dtype)
        for jj in range(self.eigen_num):
            ML_H2=ML_H2+tf.gather(ML_H2_out[jj,:,:,:], self.eigen_order_T[jj,:], axis = -1)*tf.reshape(self.eigen_sign_T[jj,:],[1,1,6])
        ML_H2=ML_H2/self.eigen_num # (batch_size,c_num,6)
        
        c_0_U=tf.expand_dims(c_0_U,axis=1)
        ML_H2=c_0_U @ tf.matmul(self.flat2mtx(ML_H2), c_0_U, transpose_b = True)   
        return ML_H2 # (batch_size,c_num,3,3)
    
    def get_ML_H1(self, U_T_c_U_flat, c_0_U, Rc_weight=None):
        ML_H1_out=self.subnetwork(U_T_c_U_flat, 'H1') # (batch_size*eigen_num,6*(c_num-1))
        ML_H1_out=tf.reshape(ML_H1_out,[self.eigen_num,self.batch_size,(self.c_num-1),6])
        
        ML_H1=tf.zeros((self.batch_size,self.c_num-1,6), dtype = self.tf_dtype)
        for jj in range(self.eigen_num):
            ML_H1=ML_H1+tf.gather(ML_H1_out[jj,:,:,:], self.eigen_order_T[jj,:], axis = -1)*tf.reshape(self.eigen_sign_T[jj,:],[1,1,6])
        ML_H1=ML_H1/self.eigen_num # (batch_size,(c_num-1),6)
        c_0_U=tf.expand_dims(c_0_U,axis=1)
        ML_H1=c_0_U @ tf.matmul(self.flat2mtx(ML_H1), c_0_U, transpose_b = True)  
        
        if Rc_weight is None:
            Rc_weight=self.weights['c0_weight']/tf.norm(self.weights['c0_weight'],axis = 1,keepdims=True)/self.scale
        MD_H1_0 = 2*tf.reduce_sum((Rc_weight @ self.A) * Rc_weight)
        ML_H1 = tf.concat([MD_H1_0*tf.eye(3,batch_shape=[self.batch_size,1],dtype=self.tf_dtype),ML_H1],axis=1)
        return ML_H1 # (batch_size,c_num-1,3,3)


    def get_ML_E(self, U_T_c_U_flat,c_0_U):
        ML_E=tf.zeros([self.batch_size,1,9,9],dtype=self.tf_dtype)
        for ii in range(1, self.c_num):
            E_ii_out = self.subnetwork(U_T_c_U_flat, f'E{ii}')  # (batch_size*eigen_num, 6 * 12) 
            E_ii_out=tf.reshape(E_ii_out,[self.eigen_num,self.batch_size,12,6])

            ML_E_tmp = tf.zeros((self.batch_size, 9, 9), dtype = self.tf_dtype)                                 # (batch_size, 6, 6)   0, will store the 4th order tensor for individual encoder moment
            for jj in range(self.eigen_num):
                for kk in range(0, 6):
                    r_dr_c_vector_rot_1 = tf.gather(E_ii_out[jj,:, 2 * kk,:], self.eigen_order_T[jj,:], axis = -1)*tf.reshape( self.eigen_sign_T[jj,:],[1,6])
                    r_dr_c_vector_rot_1 = c_0_U @ tf.matmul(self.flat2mtx(r_dr_c_vector_rot_1), c_0_U, transpose_b = True) # (batch_size, 3, 3)
                    
                    r_dr_c_vector_rot_2 = tf.gather(E_ii_out[jj,:, 2 * kk+1,:], self.eigen_order_T[jj,:], axis = -1)*tf.reshape( self.eigen_sign_T[jj,:],[1,6])
                    r_dr_c_vector_rot_2 = c_0_U @ tf.matmul(self.flat2mtx(r_dr_c_vector_rot_2), c_0_U, transpose_b = True)                
                    
                    r_dr_c_vector_rot_1 = tf.reshape(r_dr_c_vector_rot_1, [self.batch_size, 9, 1])
                    r_dr_c_vector_rot_2 = tf.reshape(r_dr_c_vector_rot_2, [self.batch_size, 1, 9])
                    ML_E_tmp = ML_E_tmp + r_dr_c_vector_rot_1 @ r_dr_c_vector_rot_2                   # (batch_size, 9, 9)
                    
            ML_E_tmp=tf.expand_dims(ML_E_tmp/self.eigen_num,axis=1)      
            ML_E = tf.concat([ML_E, ML_E_tmp], axis = 1)  # (batch_size, c_num,9,9)  
        return ML_E # (batch_size, c_num,9,9)  

    def flat2mtx(self, flat):
        """upper part flat to square matrix, (?, 6) to (?, 3, 3)"""
        mtx = tf.reshape(tf.gather(flat, [0, 1, 2, 1, 3, 4, 2, 4, 5], axis = -1), flat.shape[0:-1]+3+3)
        return mtx
    def mtx2flat(self, mtx):
        """square matrix to upper part flat, (?, 3, 3) to (?, 6)"""
        flat = tf.gather(tf.reshape(mtx, mtx.shape[0:-2]+9), [0, 1, 2, 4 ,5, 8], axis = -1)
        return flat
