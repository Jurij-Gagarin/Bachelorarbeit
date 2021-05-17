import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import hexagonal_lattice as hl


def generate_initial_plot_positions(lattice):
    pos = {i: (lattice[i].return_coordinates()[0],
               lattice[i].return_coordinates()[1],
               lattice[i].return_coordinates()[2])
           for i in range(0, len(lattice))}
    return pos


def generate_manipulated_plot_positions(dim, lattice):
    pos = {}
    list_mobile_coords = hl.run(dim, plot=False).x

    for i in range(0, len(lattice)):
        if lattice[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not lattice[i].return_mobility():
            vector = lattice[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])
    return pos


def draw_initial_graph(A, angle, pos):
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    with plt.style.context('ggplot'):
        fig = plt.figure(figsize=(12, 12))
        ax = Axes3D(fig)

        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            ax.scatter(xi, yi, zi, c='red', edgecolors='k')

        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(13, angle)
    # Hide the axes
    #ax.set_axis_off()
    plt.show()


dim = 18
d = 1
stretch_factor = 5

ls = hl.create_lattice(dim, d)
l = ls[0]
l = hl.manipulate_lattice(l, d, dim, ls[1], stretch_factor)
matrices = hl.adjacency_matrix(l)
A = np.add(matrices[0], matrices[1])

draw_initial_graph(A, 22, generate_manipulated_plot_positions(dim, l))



