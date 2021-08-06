import measurements as m
import hexagonal_lattice as hl
import numpy as np
import plot


def single_plot_from_pickle(dim, lattice, displace_value=1, d=1, k=2, method='CG'):
    pos = {}
    list_mobile_coords = m.import_pickle(dim, displace_value).x

    for i in range(0, len(lattice)):
        if lattice[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not lattice[i].return_mobility():
            vector = lattice[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])
    return pos


def do_plot(dim, dv, d=1, k=2):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=dv)
    matrices = hl.adjacency_matrix(l)
    A = np.add(matrices[0], matrices[1])

    plot.draw_initial_graph(A, 22, single_plot_from_pickle(dim, l, dv), l)


do_plot(31, 0.1)
