import measurements as m
import hexagonal_lattice as hl
import numpy as np
import plot


def single_plot_from_pickle(dim, lattice, displace_value=1, gtol=1.e-10, perc=0):
    pos = {}
    list_mobile_coords = m.import_pickle(dim, displace_value, gtol, perc).x

    for i in range(0, len(lattice)):
        if lattice[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not lattice[i].return_mobility():
            vector = lattice[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])
    return pos


def do_plot(dim, dv, gtol=1.e-10, perc=0, d=1, k=2):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=dv)
    matrices = hl.dilute_lattice(hl.adjacency_matrix(l), perc)
    A = np.add(matrices[0], matrices[1])

    plot.draw_initial_graph(A, 22, single_plot_from_pickle(dim, l, dv, gtol, perc), l)


def print_convergence(dim, dv, gtol=1.e-10, perc=0):
    obj = m.import_pickle(dim, dv, gtol, perc)
    # print(gtol, obj.fun, obj.message)
    print(obj)

# do_plot(15, 5.0, 1.e-07, 5)
for i in [0.1, 0.5, 1.0, 2.5, 5.0]: print_convergence(15, i, 1.e-07, 5)
