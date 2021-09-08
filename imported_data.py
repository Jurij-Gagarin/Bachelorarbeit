import measurements as m
import hexagonal_lattice as hl
import numpy as np
import plot
import pickle


def single_plot_from_pickle(dim, dv, path, gtol=1.e-10, perc=0, d=1):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=dv)
    matrices = hl.dilute_lattice(hl.adjacency_matrix(l), perc)
    A = np.add(matrices[0], matrices[1])

    pos = {}
    list_mobile_coords = pickle.load(open(path, 'rb')).x

    for i in range(0, len(l)):
        if l[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not l[i].return_mobility():
            vector = l[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])

    plot.draw_initial_graph(A, 22, pos, l)


def print_convergence(dim, dv, gtol=1.e-10, perc=0):
    obj = m.import_pickle(dim, dv, gtol, perc)
    # print(gtol, obj.fun, obj.message)
    print(obj)


# single_plot_from_pickle(30, 5, '/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_5/'
#                               'dim=30_dv=5.0_perc=0.pickle')
