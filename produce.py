import hexagonal_lattice as hl
import pickle
import argparse


def export_pickle(dim, dv, gtol=1.e-10):
    path = f'dim={dim}_dv={dv}_gtol={gtol}.pickle'
    result = hl.run_absolute_displacement(dim, dv, plot=False, gtol=gtol)
    pickle_out = open(path, 'wb')
    pickle.dump(result, pickle_out)
    pickle_out.close()


parser = argparse.ArgumentParser(description='Calculates the energy of a displaced honeycomb lattice and exports this as a pickle')
parser.add_argument('-dim', type=int, help='Variable that describes the size of the lattice')
parser.add_argument('-dv', type=float, help='displacement value')
args = parser.parse_args()
if __name__ == '__main__':
    export_pickle(args.dim, args.dv)
    print(f'pickle with dim={args.dim} and dv={args.dv} successfully exported')
