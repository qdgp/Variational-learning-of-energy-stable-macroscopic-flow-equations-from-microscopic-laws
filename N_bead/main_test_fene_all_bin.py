import time

import numpy as np
from model_test_set import TestModel


lammps_folder = '/mnt/gs21/scratch/gepei/DeePN_chain/box_60_60_100_fx_0.016_T_800'

L = np.array([60.0, 60.0, 100.0], dtype=np.float32)
step_min = 400000
step_max = 1400000
step = 1000
dt = 2.0
Nbin = 60


def load_lammps_data(file_name):
    data = np.loadtxt(file_name, skiprows=9, dtype=np.float32)
    data = data[np.argsort(data[:, 0])]
    xyz = data[:, 1:4].reshape((-1, 7, 3)).transpose((1, 0, 2))

    rr = xyz[:-1] - xyz[1:]
    rr -= L.reshape((1, 1, 3)) * (rr >= L.reshape((1, 1, 3)) / 2.0)
    rr += L.reshape((1, 1, 3)) * (rr <= -L.reshape((1, 1, 3)) / 2.0)

    weights = np.arange(6, 0, -1, dtype=np.float32).reshape((6, 1))
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


bs = 1
eval_bin_num = Nbin // 4
snapshot_num = (step_max - step_min) // step + 1
test_model = TestModel(bs, 1, [snapshot_num, eval_bin_num])

b = 35
kBT = 0.25
I = np.reshape(np.eye(3), [1, 1, 3, 3])
I = np.tile(I, [bs, 1, 1, 1])


def get_pred_data(c):
    trc = np.trace(c, axis1=2, axis2=3).reshape((bs, 1, 1, 1))
    detc = np.linalg.det(c).reshape((bs, 1, 1, 1))
    h_trc = b / (b + 3 - trc)
    S = -0.5 * kBT * (-b * np.log(h_trc) + np.log(detc)).reshape((bs, 1))
    stress = (kBT * h_trc * c).reshape((bs, 3, 3))
    ML_H1H2 = 2 * kBT * (I - h_trc * c) * ((1.0 / 1.1575) ** 2)
    ds_dc = 0.5 * kBT * (np.linalg.inv(c) - h_trc * I)
    ML_E = np.zeros((bs, 1, 9, 9))
    return ML_E, ML_H1H2, stress, S, ds_dc


for i, lammps_step in enumerate(range(step_min, step_max + 1, step)):
    start_time = time.time()
    file_name = f'{lammps_folder}/dump.{lammps_step}.lammpstrj'
    xr_bins, Np = get_snapshot_bins(file_name)

    for j, xr in enumerate(xr_bins):
        xr = xr[np.newaxis]
        yr = np.sum(xr, axis=2)
        c0 = yr.transpose([0, 2, 1]) @ yr / xr.shape[1] / 6 / 0.744
        c0 = np.reshape(c0, [bs, 1, 3, 3])

        ML_E, ML_H1H2, stress, S, ds_dc = get_pred_data(c0)

        test_model.collect_other_data(i, j, 'S', S)
        test_model.collect_other_data(i, j, 'ML_stress', stress)
        test_model.collect_other_data(i, j, 'c', c0)

    print(
        f'snapshot {i + 1}/{snapshot_num}, step {lammps_step}, '
        f'Np {Np}, time {time.time() - start_time:.2f} s',
        flush=True,
    )

test_model.save = {
    name: test_model.save[name] for name in ['S', 'ML_stress', 'c']
}
test_model.save['t'] = np.arange(snapshot_num, dtype=np.float64) * dt
test_model.save_data('model_chain_N7/model_fenep/test_all_bin_seed_1.mat')
