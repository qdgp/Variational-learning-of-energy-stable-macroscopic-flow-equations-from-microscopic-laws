import sys, time, os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import tensorflow as tf
from z_fun import get_solver
from model_test_set import TestModel


save_folder = 'model9_macro_star-VAR_Law_c2_bin2_net_40_40_N1920'
lammps_folder = '/mnt/gs21/scratch/gepei/DeePN_star/box_60_60_100_fx_0.016'

L = np.array([60.0, 60.0, 100.0], dtype=np.float32)
step_min = 0
step_max = 1000000
step = 2000
dt = 4.0
Nbin = 60

Ks = 0.1
l0 = 2.3


def load_lammps_data(file_name):
    data = np.loadtxt(file_name, skiprows=9, dtype=np.float32)
    data = data[np.argsort(data[:, 0])]
    xyz = data[:, 1:4].reshape((-1, 7, 3)).transpose((1, 0, 2))

    rr = xyz[[0, 1, 0, 3, 0, 5]] - xyz[[1, 2, 3, 4, 5, 6]]
    rr -= L.reshape((1, 1, 3)) * (rr >= L.reshape((1, 1, 3)) / 2.0)
    rr += L.reshape((1, 1, 3)) * (rr <= -L.reshape((1, 1, 3)) / 2.0)

    weights = np.array([2, 1, 2, 1, 2, 1], dtype=np.float32).reshape((6, 1))
    poly_zpos = xyz[0, :, 2] - np.sum(weights * rr[:, :, 2], axis=0) / 7.0
    poly_zpos -= (poly_zpos > L[2]) * L[2]
    poly_zpos += (poly_zpos < 0.0) * L[2]

    return rr.transpose((1, 0, 2)), poly_zpos


def apply_symmetry(xr, poly_zpos):
    signs = np.array([
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
    ], dtype=np.float32)

    xr = np.concatenate([xr * sign for sign in signs], axis=0)
    poly_zpos = np.concatenate([
        poly_zpos,
        poly_zpos + L[2] / 2.0,
        L[2] - poly_zpos,
        L[2] - poly_zpos + L[2] / 2.0,
    ])
    poly_zpos = np.mod(poly_zpos, L[2])
    return xr, poly_zpos


def get_snapshot_bins(file_name):
    xr, poly_zpos = load_lammps_data(file_name)
    xr, poly_zpos = apply_symmetry(xr, poly_zpos)

    bin_length = L[2] / Nbin
    eval_bin_num = Nbin // 4
    bin_id = np.floor(poly_zpos / bin_length).astype(np.int32)
    bin_indices = [np.flatnonzero(bin_id == i) for i in range(eval_bin_num)]
    Np = min(len(indices) for indices in bin_indices)

    return [xr[indices[:Np]] for indices in bin_indices], Np


def get_force(xr):
    r_bond = np.linalg.norm(xr, axis=-1, keepdims=True)
    rr_r0 = 1.0 - r_bond * r_bond / (l0 * l0)
    rr_r0[rr_r0 < 0.01] = 0.01
    return Ks * xr / rr_r0


rand_seed = int(sys.argv[1])

StressModel, save_prex = get_solver(save_folder)
model = StressModel()
model.load_model(1, save_prex + save_folder, f'train_result_seed_{rand_seed}.mat')

eval_bin_num = Nbin // 4
snapshot_num = (step_max - step_min) // step + 1
data_size = [snapshot_num, eval_bin_num]
test_model = TestModel(model.batch_size, model.c_num, data_size)
Np_all = np.zeros(snapshot_num, dtype=np.int32)

ds_dc_star, S0 = model.get_S0()

for i, lammps_step in enumerate(range(step_min, step_max + 1, step)):
    start_time = time.time()
    file_name = f'{lammps_folder}/dump.{lammps_step}.lammpstrj'
    xr_bins, Np = get_snapshot_bins(file_name)
    Np_all[i] = Np
    model.Np = Np

    for j, xr in enumerate(xr_bins):
        ff = get_force(xr)
        xr = tf.constant(xr[np.newaxis], dtype=tf.float32)
        ff = tf.constant(ff[np.newaxis], dtype=tf.float32)
        stress = tf.reduce_mean(
            tf.matmul(xr, ff, transpose_a=True), axis=1
        ) / model.Nbonds

        c, MD_H1, MD_H2, MD_E = model.get_from_micro(xr, ff)
        ML_H1H2, ML_E, ML_pi_p, S, ds_dc = model.get_from_macro(
            c, ds_dc_star=ds_dc_star, S0=S0
        )
        MD_H1H2 = model.kBT * MD_H1 - MD_H2
        ML_stress = model.kBT * tf.eye(
            3, batch_shape=[model.batch_size], dtype=tf.float32
        ) + ML_pi_p

        test_model.collect_macro_data(
            i, j, MD_E.numpy(), ML_E.numpy(), MD_H1H2.numpy(),
            ML_H1H2.numpy(), ML_stress.numpy(), S.numpy(),
            ds_dc.numpy(), c.numpy()
        )
        test_model.collect_other_data(i, j, 'MD_stress', stress)

    print(
        f'snapshot {i + 1}/{snapshot_num}, step {lammps_step}, '
        f'Np {Np}, time {time.time() - start_time:.2f} s',
        flush=True,
    )

test_model.evalute_loss()
test_model.save['t'] = np.arange(snapshot_num, dtype=np.float64) * dt
test_model.save['z'] = (np.arange(eval_bin_num) + 0.5) * L[2] / Nbin
test_model.save['Np'] = Np_all
test_model.save_data(
    model.save_folder + f'/test_all_bin_seed_{rand_seed}.mat'
)
