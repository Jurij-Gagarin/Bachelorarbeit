import hexagonal_lattice as hl
import pickle
import argparse


def export_pickle(dim, dv, gtol=1.e-3, percentile=0, converge=True, n=1):
    for i in range(n):
        path = f'./current_measurements/dim={dim}_dv={dv}_perc={percentile}_{i + 1}.pickle'
        try:
            result = hl.run_absolute_displacement(dim, dv, tol=gtol, percentile=percentile, true_convergence=converge,
                                                  x0=True)
        except FileNotFoundError:
            print('Did not found x0')
            result = hl.run_absolute_displacement(dim, dv, tol=gtol, percentile=percentile, true_convergence=converge,
                                                  x0=False)

        pickle_out = open(path, 'wb')
        pickle.dump(result, pickle_out)
        pickle_out.close()
        print(f'pickle with dim={dim}, dv={dv} and dilution={percentile} successfully exported. {i+1} out of {n}')


parser = argparse.ArgumentParser(
    description='Calculates the energy of a displaced honeycomb lattice and exports this as a pickle')
parser.add_argument('-dim', type=int, help='Variable that describes the size of the lattice')
parser.add_argument('-dv', type=float, help='displacement value')
parser.add_argument('-p', type=int, help='percentile by which the lattice is diluted')
parser.add_argument('--gtol', type=float, default=1.e-3,
                    help='for successful convergence the gradient norm must be smaller than gtol ')
parser.add_argument('--conv', choices=('True', 'False'), default='True',
                    help='if set to false gtol will not be generated automatically. Passing gtol'
                         'than becomes necessary')
parser.add_argument('--n', type=int, default=1, help='number of times the minimization should happen')
args = parser.parse_args()

if __name__ == '__main__':
    export_pickle(args.dim, args.dv, args.gtol, args.p, args.conv == 'True', args.n)
