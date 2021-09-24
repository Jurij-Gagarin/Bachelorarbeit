# This module is mostly for plotting.

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
import numpy as np
import hexagonal_lattice as hl
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import interp1d
from math import floor, log10
import pickle


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def generate_initial_plot_positions(lattice):
    pos = {i: (lattice[i].return_coordinates()[0],
               lattice[i].return_coordinates()[1],
               lattice[i].return_coordinates()[2])
           for i in range(0, len(lattice))}
    return pos


def generate_manipulated_plot_positions(dim, lattice, opt, r2=1, displace_value=1, factor=False, d=1, k=2, method='CG',
                                        percentile=0, tol=1.e-06, x0=None):
    pos = {}
    if factor:
        list_mobile_coords = hl.run_sphere(dim, r2, plot=False).x
    else:
        res = hl.run_absolute_displacement(dim, displace_value, d=d, k=k, method=method, percentile=percentile, opt=opt
                                           , tol=tol, x0=x0)
        list_mobile_coords = res.x
        print(res)

    for i in range(0, len(lattice)):
        if lattice[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not lattice[i].return_mobility():
            vector = lattice[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])
    return pos


def draw_initial_graph(A, angle, pos, lattice, nodes=False, vectors=False):
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    with plt.style.context('classic'):
        fig = plt.figure(figsize=(20, 20))
        ax = Axes3D(fig)

        if nodes:
            for key, value in pos.items():
                xi = value[0]
                yi = value[1]
                zi = value[2]
                name = lattice[key].return_name()
                if name[2] == 1:
                    ax.scatter(xi, yi, zi, c='cornflowerblue', edgecolors='k')
                    ax.text(xi + .05, yi + .05, zi, f'({name[0]}{name[1]})')
                else:
                    ax.scatter(xi, yi, zi, c='red', edgecolors='k')
                    ax.text(xi + .05, yi + .05, zi, f'({name[0]}{name[1]})')
                if vectors:
                    ax.plot((0, 1), (0, 0), (0, 0), lw=5, c='cyan')
                    ax.text(.4, -.25, 0, 'd', size=20, c='cyan')
                    ax.text(-.4, -.25, 0, '\u03B4', size=20, c='pink')
                    ax.text(-.85, .7, 0, 'a2', c='gold', size=20)
                    ax.text(.75, .7, 0, 'a1', c='gold', size=20)
                    delta = Arrow3D([0, 0], [0, -1 / 3 ** .5],
                                    [0, 0], mutation_scale=20,
                                    lw=3, arrowstyle="-|>", color="pink")
                    a1 = Arrow3D([0, .5], [0, .5 * 3 ** .5],
                                 [0, 0], mutation_scale=20,
                                 lw=3, arrowstyle="-|>", color="gold")
                    a2 = Arrow3D([0, -.5], [0, .5 * 3 ** .5],
                                 [0, 0], mutation_scale=20,
                                 lw=3, arrowstyle="-|>", color="gold")
                    ax.add_artist(delta)
                    ax.add_artist(a1)
                    ax.add_artist(a2)
        # ax.set_zlim3d(0, d*dim/2)
        # ax.set_xlim3d(-4.5, 4.5)
        # ax.set_ylim3d(-4.5, 4.5)

        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    # 90
    ax.view_init(13, angle)
    # Hide the axes
    # ax.set_axis_off()
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)

    plt.show()


def plot_graph(dim, r2=1, displace_value=1, factor=False, d=1, k=2, nodes=False, method='CG', percentile=0, opt=None,
               tol=1.e-03, x0=None):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=displace_value)
    matrices = hl.dilute_lattice(hl.adjacency_matrix(l), percentile)
    A = np.add(matrices[0], matrices[1])

    draw_initial_graph(A, 22, generate_manipulated_plot_positions(dim, l,
                                                                  r2=r2,
                                                                  displace_value=displace_value, factor=factor,
                                                                  d=d, k=k, method=method, percentile=percentile,
                                                                  opt=opt, tol=tol, x0=x0),
                       l, nodes=nodes)


def import_pickle(path):
    pickle_in = open(path, 'rb')
    return pickle.load(pickle_in)


def contour_coords(dim, displace_value, path, tol=.1):
    lattice = hl.create_lattice(dim)
    l = hl.manipulate_lattice_absolute_value(lattice[0], lattice[1], displace_value=displace_value)
    res = import_pickle(path)
    values = hl.assemble_result(res.x, hl.list_of_coordinates(l)[0], plot=False)
    x = []
    z = []

    for i in range(len(values[0])):
        if abs(values[1][i]) <= tol:
            x.append(values[0][i])
            z.append(values[2][i])

    return x, z, res.fun


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def fit_contour(min_dim, max_dim, disp_value):
    print(f'max dim = {max_dim}')

    for i in range(min_dim, max_dim + 1):
        if i % 2 == 0:
            print(f'working on dim = {i}')
            path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{disp_value}/dim={i}_dv={disp_value}_perc=0.pickle'
            coords = contour_coords(i, disp_value, path=path)
            plt.plot(coords[0], coords[1], marker='o', linestyle='None', c='blue', markersize=1)

            m1 = max(coords[0])
            m2 = min(coords[0])
            x_new = np.linspace(m2, m1, num=1000, endpoint=True)
            f2 = interp1d(coords[0], coords[1])

            plt.plot(x_new, f2(x_new), label=f'dim = {i}, E={round_sig(coords[2])}')
    plt.ylabel('z-Achse', size=16)
    plt.xlabel('x-Achse', size=16)
    plt.legend()
    plt.title(f'Profil der minimierten, geraden Gitter dim 6-50 bei dv={disp_value}', size=20)
    plt.show()


