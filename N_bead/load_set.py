import numpy as np
import os, h5py, sys
import scipy.io
Folder='/mnt/research/MultiscaleML_group/PeiGe/DeePN_entropy/N_bead/'

def resolve_data(file_name):
    """Prefer locally generated data; fall back to the original archive."""
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    return local if os.path.isfile(local) else os.path.join(Folder, file_name)

def check_ver(ver,file_name):
    if 'star' in ver:
        MD_folder='MD_star_N7/'
    elif 'chain' in ver:
        MD_folder='MD_chain_N7/'

    return resolve_data(MD_folder+file_name)

def load_file(file_name):
    with h5py.File(resolve_data(file_name),'r') as data_file:
        xr = np.array(data_file['train_rr'],dtype=np.float32).transpose((0,2,3,1)) # (Nsample,Np,Nbonds,3)
        ff = np.array(data_file['train_ff'],dtype=np.float32).transpose((0,2,3,1)) # (Nsample,Np,Nbonds,3)
    return xr,ff

def load_velocity(path, nbin=40, sample_dt=0.04, timestep=0.002):
    """Load u_x(t,z), dating each record at its averaging-window center.

    stat/vel averages MD steps 1..20. With dt=0.002, the first record is
    centered at 0.021, followed by records spaced by 0.04.
    """
    if h5py.is_hdf5(str(path)):
        with h5py.File(str(path), 'r') as data:
            velocity = np.array(data['v'])[0]
            times = np.array(data['t']).ravel()
    elif str(path).endswith('.mat'):
        data = scipy.io.loadmat(str(path))
        velocity = data['v'][:, :, 0].T
        times = data['t'].ravel()
    else:
        velocity = np.loadtxt(str(path), skiprows=2, usecols=3).reshape(-1, nbin)
        times = np.arange(len(velocity)) * sample_dt
    times = times - times[0] + (sample_dt + timestep) / 2
    return times, velocity

def get_ff_from_xr(xr):
    Ks = 0.1
    l0 =2.3
    l0_sq = l0*l0
    
    r_bond=np.linalg.norm(xr,axis=-1,keepdims=True)
    rr_r0 = 1.0 - r_bond*r_bond/l0_sq
    rr_r0[rr_r0<0.01]=0.01
    ff=Ks*xr/rr_r0
    return ff

class load(object):
    def load(ver,data_set):
        if 'star' in ver:
            if 'bin2' in data_set:
                return load.star_bin(bin=2,Np=1920)
        elif 'chain' in ver:
            if 'temp-bin2' in data_set:
                return load.chain_temp(bin=2,Np=1920)
        sys.exit('data set not found')



    def chain_temp(bin,Np):
        data_set_name=f'temp-bin{bin}'

        xr1, ff1 = load_file(f'MD_chain_N7/box_60_60_100_fx_0.02_T_800_kBT_0.22/train_bin_{bin}_Np_{Np}.mat')
        xr2, ff2 = load_file(f'MD_chain_N7/box_60_60_100_fx_0.02_T_800_kBT_0.25/train_bin_{bin}_Np_{Np}.mat')
        xr3, ff3 = load_file(f'MD_chain_N7/box_60_60_100_fx_0.02_T_800_kBT_0.30/train_bin_{bin}_Np_{Np}.mat')

        xr=np.concatenate([xr1[1:],xr2[1:],xr3[1:]],axis=0)
        ff=np.concatenate([ff1[1:],ff2[1:],ff3[1:]],axis=0)
        Np, Nbonds = xr.shape[1], xr.shape[2]
        stress=np.mean(ff.transpose((0,1,3,2)) @ xr,axis=1)/float(Nbonds)
        print('xr:',xr.shape,'ff:',ff.shape,'stress:',stress.shape)
        return xr, ff, stress, Np, data_set_name

    def star_bin(bin,Np):
        data_set_name=f'bin{bin}'
        xr1, ff1 = load_file(f'MD_star_N7/box_60_60_100_fx_0.016/train_bin_{bin}_Np_{Np}.mat')
        xr, ff=xr1[1:], ff1[1:]
        Np, Nbonds = xr.shape[1], xr.shape[2]
        stress=np.mean(ff.transpose((0,1,3,2)) @ xr,axis=1)/float(Nbonds)
        print('xr:',xr.shape,'ff:',ff.shape,'stress:',stress.shape)
        return xr, ff, stress, Np, data_set_name

    def init(ver, dtype=np.float32):
        if 'star' in ver:
            data_file =h5py.File(Folder+'MD_star_N7/box_60_60_100_fx_0.016/init.mat','r')
            xr = np.array(data_file['rr'],dtype=dtype).transpose((1,2,0)) # (Np,Nbonds,3)
            xr = np.expand_dims(xr,axis=0)
        elif 'chain' in ver:
            data_file =h5py.File(Folder+'MD_chain_N7/box_60_60_100_init_2/init.mat','r')
            xr = np.array(data_file['xr'],dtype=dtype).transpose((0,2,3,1)) # (Nsample,Np,Nbonds,3)
            xr = xr[[50]]
            xr = np.concatenate([xr,-np.flip(xr,axis=2)],axis=1)
        print('xr-ref:',xr.shape)
        return xr

    def load_macro(ver,fx='0.01',Data_Folder=Folder):
        box_folder=f'box_60_60_100_fx_{fx}'
        if 'chain' in ver and '0.016' in fx:
            box_folder=f'box_60_60_100_fx_{fx}_T_800'
            
        path=check_ver(ver,box_folder+'/macro_Nbin_40.mat')
        fs=h5py.File(path)

        bin40={'t':np.array(fs['t']),'c0':np.array(fs['c0']),'stress':np.array(fs['stress'])}
        
        path=check_ver(ver,box_folder+'/v_Nbin_40.mat')
        t,v=load_velocity(path)
        Nbin,Nbin2,Nbin4=40,20,10
        v=(v[:,:Nbin4]+v[:,Nbin2-1:Nbin4-1:-1]-v[:,Nbin2:3*Nbin4]-v[:,Nbin-1:3*Nbin4-1:-1])/4.0
        bin40.update({'vt':t,'v':v})
        
        path=check_ver(ver,box_folder+'/macro_Nbin_20.mat')
        fs=h5py.File(path)
        bin20={'t':np.array(fs['t']),'c0':np.array(fs['c0']),'stress':np.array(fs['stress'])}
        bin20.update({'vt':t,'v':(v[:,::2]+v[:,1::2])/2.0})
        return bin40, bin20

    def load_macro_chain(ver,fx='0.016',T='800'):
        box_folder=f'box_60_60_100_fx_{fx}_T_{T}'
            
        path=check_ver(ver,box_folder+'/macro_Nbin_20_extend.mat')
        fs=h5py.File(path)
        bin20={'t':np.array(fs['t']),'c0':np.array(fs['c0']),'stress':np.array(fs['stress'])}

        path=check_ver(ver,box_folder+'/v_Nbin_20.mat')
        t,v=load_velocity(path,nbin=20)
        Nbin,Nbin2,Nbin4=20,10,5
        v=(v[:,:Nbin4]+v[:,Nbin2-1:Nbin4-1:-1]-v[:,Nbin2:3*Nbin4]-v[:,Nbin-1:3*Nbin4-1:-1])/4.0
        bin20.update({'vt':t,'v':v})
        return bin20
