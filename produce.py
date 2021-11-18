import hexagonal_lattice as hl
import pickle
import argparse
import random as rn
import numpy as np


def export_pickle(dim, dv, gtol=1.e-3, percentile=0, converge=True, seed=None, n=1, d=1):
    for i in range(n):
        if seed is None:
            seed = rn.randint(0, 1000000000)
        try:
            result = hl.run_absolute_displacement(dim, dv, tol=gtol, percentile=percentile, true_convergence=converge,
                                                  x0=True, seed=seed, d=d, k=10/d**2)
        except FileNotFoundError:
            print('Did not found x0')
            result = hl.run_absolute_displacement(dim, dv, tol=gtol, percentile=percentile, true_convergence=converge,
                                                  x0=None, d=d, k=10/d**2)

        path = f'./current_measurements/dim={dim}_dv={dv}_perc={percentile}_{seed}.pickle'
        pickle_out = open(path, 'wb')
        pickle.dump(result, pickle_out)
        pickle_out.close()
        print(f'pickle with dim={dim}, dv={dv}, d={d} and dilution={percentile}, seed={seed} successfully exported.'
              f' {i + 1} out of {n}')
        seed = None


def export_pickle_sphere(dim, r, dv, gtol=1.e-3, percentile=0, converge=True, seed=None, n=1, d=1):
    for i in range(n):
        if seed is None:
            seed = rn.randint(0, 1000000000)

        result = hl.run_sphere(dim, rad=r, dv=dv, tol=gtol, percentile=percentile, true_convergence=converge,
                                   x0=True, seed=seed, d=d, k=10/d**2)

        path = f'./current_measurements/dim={dim}_dv={dv}_perc={percentile}_{seed}.pickle'
        pickle_out = open(path, 'wb')
        pickle.dump(result, pickle_out)
        pickle_out.close()
        print(f'pickle with dim={dim}, dv={dv}, d={d} and dilution={percentile}, seed={seed} successfully exported.'
              f' {i + 1} out of {n}')
        seed = None


parser = argparse.ArgumentParser(
    description='Calculates the energy of a displaced honeycomb lattice and exports this as a pickle')
parser.add_argument('-dim', type=int, help='Variable that describes the size of the lattice')
parser.add_argument('-dv', type=float, help='displacement value')
parser.add_argument('-p', type=float, help='percentile by which the lattice is diluted')
parser.add_argument('--gtol', type=float, default=1.e-3,
                    help='for successful convergence the gradient norm must be smaller than gtol ')
parser.add_argument('--r', type=float, default=None, help='radius of the sphere')
parser.add_argument('--conv', choices=('True', 'False'), default='True',
                    help='if set to false gtol will not be generated automatically. Passing gtol'
                         'than becomes necessary')
parser.add_argument('--n', type=int, default=1, help='number of times the minimization should happen')
parser.add_argument('--s', type=int, default=None, help='Seed for dilution')
parser.add_argument('--loop', type=int, choices=(0, 1), default=0,
                    help='0 for False, 1 for True, runs over all seeds in '
                         'seed_list.txt')
parser.add_argument('--d', type=int, help='Horizontal distance between two blue nodes')
args = parser.parse_args()

if __name__ == '__main__':
    if args.loop == 0 and args.r is None:
        export_pickle(args.dim, args.dv*args.d, args.gtol, args.p, args.conv == 'True', args.s, args.n, args.d)
    elif args.loop == 1 and args.r is None:
        dvs = list(np.arange(args.dv, 2.5 - .5, -.5))
        for i in dvs:
            export_pickle(args.dim, i*args.d, args.gtol, args.p, args.conv == 'True', args.s, args.n, args.d)
    elif args.loop == 0 and args.r is not None:
        export_pickle_sphere(args.dim, args.r*args.d, args.dv*args.d, args.gtol, args.p, args.conv == 'True', args.s,
                             args.n, args.d)
    elif args.loop == 1 and args.r is not None:
        dvs = list(np.arange(args.dv, 0-.5, -.5))
        for i in dvs:
            export_pickle_sphere(args.dim, args.r*args.d, i*args.d, args.gtol, args.p, args.conv == 'True', args.s,
                                 args.n, args.d)
