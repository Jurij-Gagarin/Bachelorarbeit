import hexagonal_lattice as hl
import pickle
import argparse


def export_pickle(dim, dv, gtol=1.e-10, percentile=0):
    path = f'dim={dim}_dv={dv}_gtol={gtol}_perc={percentile}.pickle'
    result = hl.run_absolute_displacement(dim, dv, plot=False, gtol=gtol, percentile=percentile)
    pickle_out = open(path, 'wb')
    pickle.dump(result, pickle_out)
    pickle_out.close()


parser = argparse.ArgumentParser(
    description='Calculates the energy of a displaced honeycomb lattice and exports this as a pickle')
parser.add_argument('-dim', type=int, help='Variable that describes the size of the lattice')
parser.add_argument('-dv', type=float, help='displacement value')
parser.add_argument('-p', type=int, help='percentile by which the lattice is diluted')
parser.add_argument('--gtol', type=float,
                    help='for successful convergence the gradient norm must be smaller than gtol ')
args = parser.parse_args()

if __name__ == '__main__':
    export_pickle(args.dim, args.dv, percentile=args.p)
    if args.gtol is not None:
        export_pickle(args.dim, args.dv, args.gtol, args.p)
    print(f'pickle with dim={args.dim}, dv={args.dv} and dilution={args.p} successfully exported')
