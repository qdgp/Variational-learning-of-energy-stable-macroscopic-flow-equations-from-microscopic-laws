"""Streaming preparation of regular seven-bead chain and star MD datasets.

Conformations use (polymer, bond, Cartesian) axes in memory. HDF5 files retain
the existing sample-first layouts used by load_set.py and the figure code.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from load_set import load_velocity


@dataclass
class Snapshot:
    step: int
    bonds: np.ndarray
    center_z: np.ndarray
    lower: np.ndarray
    box: np.ndarray


def read_snapshot(path, topology):
    """Read one orthorhombic LAMMPS dump; reconstruct the equal-mass COM.

    Connectors point from child to parent. Unwrapping along the molecular
    tree before averaging bead positions handles molecules crossing a boundary.
    """
    with Path(path).open() as stream:
        header = [stream.readline().strip() for _ in range(9)]
        bounds = np.array([list(map(float, line.split())) for line in header[5:8]])
        lower, box = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
        columns = header[8].split()[2:]
        coordinates = next(keys for keys in [('x', 'y', 'z'), ('xu', 'yu', 'zu'), ('xs', 'ys', 'zs')]
                           if all(key in columns for key in keys))
        usecols = [columns.index(key) for key in ('id',) + coordinates]
        atoms = np.fromstring(stream.read(), sep=' ').reshape(-1, len(columns))[:, usecols]
    atoms = atoms[np.argsort(atoms[:, 0])]
    xyz = atoms[:, 1:].reshape(-1, 7, 3)
    if coordinates[0] == 'xs':
        xyz = lower + xyz * box
    parents = [0, 1, 2, 3, 4, 5] if topology == 'chain' else [0, 1, 0, 3, 0, 5]
    children = [1, 2, 3, 4, 5, 6]
    bonds = xyz[:, parents] - xyz[:, children]
    bonds -= box * np.floor(bonds / box + 0.5)
    unwrapped = np.empty_like(xyz)
    unwrapped[:, 0] = xyz[:, 0]
    for bond, (parent, child) in enumerate(zip(parents, children)):
        unwrapped[:, child] = unwrapped[:, parent] - bonds[:, bond]
    center_z = (unwrapped.mean(axis=1)[:, 2] - lower[2]) % box[2]
    return Snapshot(int(header[1]), bonds, center_z, lower, box)


def fene_force(bonds):
    """FENE potential gradient with the original 0.01 denominator cutoff."""
    denominator = np.maximum(1.0 - np.sum(bonds**2, axis=-1, keepdims=True) / 2.3**2, 0.01)
    return 0.1 * bonds / denominator


def polymer_stress(bonds, force):
    """Mean force-connector tensor per bond (no isotropic offset)."""
    return np.einsum('nbi,nbj->ij', force, bonds) / (len(bonds) * bonds.shape[1])


def find_snapshots(path, start=None, stop=None, step=200):
    available = {int(file.name.split('.')[1]): file for file in Path(path).glob('dump.*.lammpstrj')}
    first = min(available) if start is None else start
    last = max(available) if stop is None else stop
    return [(s, available[s]) for s in range(first, last + 1, step)], min(available)


def pick_bin(centers, length, width, npoly, rng, attempts=20):
    """Choose one common spatial bin, then the first npoly sorted molecules."""
    for left in rng.uniform(0.0, length - width, attempts):
        selected = [np.flatnonzero((z > left) & (z < left + width)) for z in centers]
        if all(len(index) >= npoly for index in selected):
            return left, [index[:npoly] for index in selected]
    raise ValueError('No candidate bin contains {} molecules in every snapshot'.format(npoly))


def output_file(path, args, topology):
    """Open the final HDF5 output directly for writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = h5py.File(str(path), 'w')
    data.attrs.update(format='DeePN-HDF5', topology=topology,
                      source_path=str(args.path.resolve()),
                      center_of_mass='mean of tree-unwrapped bead positions',
                      spring_constant=np.float32(0.1), max_extension=np.float32(2.3),
                      force_denominator_cutoff=np.float32(0.01), timestep=np.float32(args.timestep))
    return data


def tensor_dataset(data, name, samples, npoly):
    array = data.create_dataset(name, (samples, 3, npoly, 6), dtype='f4',
                                chunks=(1, 3, npoly, 6), compression='gzip',
                                compression_opts=1, shuffle=True)
    array.attrs['axes'] = 'sample,cartesian,polymer,bond'
    return array


def base_parser(description, step=200):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path', type=Path, help='Directory containing LAMMPS dumps')
    parser.add_argument('--output', type=Path, help='Output HDF5 file (legacy .mat suffix)')
    parser.add_argument('--start', type=int, help='First dump step; default: first available')
    parser.add_argument('--stop', type=int, help='Last dump step, inclusive; default: last available')
    parser.add_argument('--step', type=int, default=step, help='Spacing between dump steps')
    parser.add_argument('--timestep', type=float, default=0.002, help='MD integration timestep')
    return parser


def training_parser(description):
    parser = base_parser(description)
    parser.add_argument('--bin-length', type=float, default=2.0)
    parser.add_argument('--npoly', type=int, default=1920)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=20)
    return parser


def output_path(args, topology, filename):
    folder = 'MD_chain_N7' if topology == 'chain' else 'MD_star_N7'
    return args.output or Path(__file__).parent / folder / args.path.resolve().name / filename


def add_time(data, snapshots, timestep):
    steps = np.array([step for step, _ in snapshots], dtype=np.int64)
    data.create_dataset('steps', data=steps)
    data.create_dataset('t', data=((steps - steps[0]) * timestep)[:, None], dtype='f4')


def train_main(topology):
    args = training_parser('Generate spatially binned {} training data'.format(topology)).parse_args()
    snapshots, _ = find_snapshots(args.path, args.start, args.stop, args.step)
    name = 'train_bin_{:g}_Np_{}.mat'.format(args.bin_length, args.npoly)
    rng = np.random.default_rng(args.seed)
    with output_file(output_path(args, topology, name), args, topology) as data:
        data.attrs.update(seed=args.seed, candidates=args.candidates)
        data.create_dataset('bin_length', data=[[args.bin_length]], dtype='f4')
        add_time(data, snapshots, args.timestep)
        rr = tensor_dataset(data, 'train_rr', len(snapshots), args.npoly)
        ff = tensor_dataset(data, 'train_ff', len(snapshots), args.npoly)
        stress = data.create_dataset('train_stress', (len(snapshots), 3, 3), dtype='f4')
        lefts = data.create_dataset('bin_left', (len(snapshots),), dtype='f4')
        for i, (step, path) in enumerate(snapshots):
            frame = read_snapshot(path, topology)
            left, (index,) = pick_bin([frame.center_z], frame.box[2], args.bin_length,
                                     args.npoly, rng, args.candidates)
            bonds = frame.bonds[index]
            force = fene_force(bonds)
            rr[i], ff[i] = bonds.transpose(2, 0, 1), force.transpose(2, 0, 1)
            stress[i], lefts[i] = polymer_stress(bonds, force), left
            if i % 100 == 0 or i == len(snapshots) - 1:
                print('snapshot {}/{}'.format(i + 1, len(snapshots)), flush=True)


def time_series_main():
    parser = training_parser('Generate adjacent-snapshot chain training pairs')
    parser.add_argument('--velocity', type=Path, help='Velocity MAT file or .series.plt; default: input directory text file')
    parser.add_argument('--velocity-nbin', type=int, default=40)
    parser.add_argument('--velocity-dt', type=float, default=0.04)
    parser.add_argument('--velocity-origin-step', type=int, help='MD step at velocity t=0; default: first available dump')
    args = parser.parse_args()
    snapshots, origin = find_snapshots(args.path, args.start, args.stop, args.step)
    if args.velocity_origin_step is not None:
        origin = args.velocity_origin_step
    velocity_path = args.velocity or args.path / 'vel1D_Nbin_{}.series.plt'.format(args.velocity_nbin)
    vt, velocity = load_velocity(velocity_path, args.velocity_nbin, args.velocity_dt, args.timestep)
    midtimes = np.array([(a[0] + b[0]) / 2 - origin for a, b in zip(snapshots[:-1], snapshots[1:])]) * args.timestep
    name = 'train_time_series_bin_{:g}_Np_{}.mat'.format(args.bin_length, args.npoly)
    rng = np.random.default_rng(args.seed)
    previous = read_snapshot(snapshots[0][1], 'chain')
    dz = previous.box[2] / args.velocity_nbin
    # Existing VAR/PDE readers negate train_kappa when building kappa_xz.
    negative_gradient = (np.roll(velocity, 1, axis=1) - np.roll(velocity, -1, axis=1)) / (2 * dz)
    grid = (np.arange(args.velocity_nbin + 2) - 0.5) * dz
    with output_file(output_path(args, 'chain', name), args, 'chain') as data:
        data.attrs.update(seed=args.seed, candidates=args.candidates,
                          velocity_path=str(velocity_path.resolve()),
                          velocity_origin_step=origin, train_kappa_convention='negative_du_x_dz')
        data.create_dataset('bin_length', data=[[args.bin_length]], dtype='f4')
        add_time(data, snapshots[:-1], args.timestep)
        data.create_dataset('steps2', data=[s for s, _ in snapshots[1:]])
        data.create_dataset('pair_dt', data=np.diff([s for s, _ in snapshots]) * args.timestep, dtype='f4')
        data.create_dataset('midpoint_t', data=midtimes, dtype='f4')
        count = len(snapshots) - 1
        arrays = {key: tensor_dataset(data, key, count, args.npoly)
                  for key in ['train_rr1', 'train_rr2', 'train_ff1', 'train_ff2']}
        stress1 = data.create_dataset('train_stress1', (count, 3, 3), dtype='f4')
        stress2 = data.create_dataset('train_stress2', (count, 3, 3), dtype='f4')
        kappa = data.create_dataset('train_kappa', (count, 1), dtype='f4')
        lefts = data.create_dataset('bin_left', (count,), dtype='f4')
        for i, (step, path) in enumerate(snapshots[1:]):
            current = read_snapshot(path, 'chain')
            left, selected = pick_bin([previous.center_z, current.center_z], current.box[2],
                                      args.bin_length, args.npoly, rng, args.candidates)
            for suffix, frame, index, stress in [('1', previous, selected[0], stress1),
                                                ('2', current, selected[1], stress2)]:
                bonds = frame.bonds[index]
                force = fene_force(bonds)
                arrays['train_rr' + suffix][i] = bonds.transpose(2, 0, 1)
                arrays['train_ff' + suffix][i] = force.transpose(2, 0, 1)
                stress[i] = polymer_stress(bonds, force)
            gradient = np.array([np.interp(midtimes[i], vt, column) for column in negative_gradient.T])
            gradient = np.concatenate([gradient[-1:], gradient, gradient[:1]])
            kappa[i, 0] = np.interp(left + args.bin_length / 2, grid, gradient)
            lefts[i] = left
            previous = current
            if i % 100 == 0 or i == count - 1:
                print('pair {}/{}'.format(i + 1, count), flush=True)


def macro_main(topology):
    parser = base_parser('Generate binned {} polymer stress and conformation'.format(topology),
                         step=200 if topology == 'chain' else 2000)
    parser.add_argument('--nbin', type=int, required=True, help='Number of bins over the full box')
    parser.add_argument('--extend', action='store_true', help='Apply the original four flow-symmetry copies')
    args = parser.parse_args()
    snapshots, _ = find_snapshots(args.path, args.start, args.stop, args.step)
    name = 'macro_Nbin_{}{}.mat'.format(args.nbin, '_extend' if args.extend else '')
    with output_file(output_path(args, topology, name), args, topology) as data:
        data.attrs.update(nbin=args.nbin, extend=args.extend,
                          tensor_components='xx,yx,zx,xy,yy,zy,xz,yz,zz',
                          c0_definition='end_bead_to_com / sqrt(6*0.225)' if topology == 'chain'
                          else 'mean bond covariance / 0.693726786866425')
        add_time(data, snapshots, args.timestep)
        stress = data.create_dataset('stress', (len(snapshots), 9, args.nbin), dtype='f4')
        c0 = data.create_dataset('c0', stress.shape, dtype='f4')
        counts = data.create_dataset('count', (len(snapshots), args.nbin), dtype='i8')
        for i, (step, path) in enumerate(snapshots):
            frame = read_snapshot(path, topology)
            bonds, z = frame.bonds, frame.center_z
            if args.extend:
                signs = np.array([[1, 1, 1], [-1, 1, 1], [-1, 1, -1], [1, 1, -1]])
                bonds = np.concatenate([bonds * sign for sign in signs])
                z = np.concatenate([z, z + frame.box[2] / 2, frame.box[2] - z,
                                    frame.box[2] - z + frame.box[2] / 2]) % frame.box[2]
            bins = np.floor(z / frame.box[2] * args.nbin).astype(int)
            for j in range(args.nbin):
                selected = bonds[bins == j]
                tensor = polymer_stress(selected, fene_force(selected))
                if topology == 'chain':
                    weights = np.arange(6, 0, -1) / (7 * np.sqrt(6 * 0.225))
                    y = np.einsum('b,nbi->ni', weights, selected)
                    conformation = y.T @ y / len(y)
                else:
                    conformation = polymer_stress(selected, selected) / 0.693726786866425
                stress[i, :, j] = tensor.ravel(order='F')
                c0[i, :, j] = conformation.ravel(order='F')
                counts[i, j] = len(selected)
            if i % 100 == 0 or i == len(snapshots) - 1:
                print('snapshot {}/{}'.format(i + 1, len(snapshots)), flush=True)
