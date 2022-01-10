# This module is mostly for plotting.

import networkx as nx
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.pyplot as plt
import numpy as np
import hexagonal_lattice as hl
import random as rn
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import interp1d
from math import floor, log10
import pickle
from colour import Color
import helpful_functions as hf
from matplotlib import gridspec


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plot_lattice(lattice):
    # Rather old function. Used to create simple plots
    x = []
    y = []
    z = []
    xb = []
    yb = []
    zb = []
    xg = []
    yg = []
    zg = []

    for i in range(0, len(lattice)):
        node = lattice[i]
        if i in [233, 17]:
            xg.append(node.return_coordinates()[0])
            yg.append(node.return_coordinates()[1])
            zg.append(node.return_coordinates()[2])
        elif node.return_mobility():
            x.append(node.return_coordinates()[0])
            y.append(node.return_coordinates()[1])
            z.append(node.return_coordinates()[2])
        elif not node.return_mobility():
            xb.append(node.return_coordinates()[0])
            yb.append(node.return_coordinates()[1])
            zb.append(node.return_coordinates()[2])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.scatter(x, y, z, c='blue')
    ax.scatter(xb, yb, zb, c='red')
    ax.scatter(xg, yg, zg, c='green')
    plt.show()


def generate_initial_plot_positions(lattice):
    pos = {i: (lattice[i].return_coordinates()[0],
               lattice[i].return_coordinates()[1],
               lattice[i].return_coordinates()[2])
           for i in range(0, len(lattice))}
    return pos


def generate_manipulated_plot_positions(dim, lattice, opt, r=1, displace_value=1, sphere=False, d=1, k=2, method='CG',
                                        percentile=0, tol=1.e-06, x0=None, tg=True, seed=None):
    pos = {}
    if sphere:
        list_mobile_coords = hl.run_sphere(dim, r, dv=displace_value, percentile=percentile, plot=False, seed=seed, d=d,
                                           true_convergence=tg, tol=tol, k=k, x0=x0).x
    else:
        res = hl.run_absolute_displacement(dim, displace_value, d=d, k=k, method=method, percentile=percentile, opt=opt
                                           , tol=tol, x0=x0, true_convergence=tg, seed=seed)
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


def draw_initial_graph(A, angle, pos, lattice, nodes=False, vectors=False, dv=0, rad=0, max_dist=None, d=1,
                       draw_sphere=False):
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    with plt.style.context('classic'):
        fig = plt.figure(figsize=(15, 15), facecolor='white')
        ax = fig.add_subplot(projection='3d')
        ax.set_title('', size=30)
        max_dis = []

        if nodes:
            for key, value in pos.items():
                xi = value[0]
                yi = value[1]
                zi = value[2]
                name = lattice[key].return_name()
                if name[2] == 1:
                    ax.scatter(xi, yi, zi, c='cornflowerblue', edgecolors='k')
                    #ax.text(xi + .05, yi + .05, zi, f'({name[0]}{name[1]})')
                else:
                    ax.scatter(xi, yi, zi, c='red', edgecolors='k')
                    #ax.text(xi + .05, yi + .05, zi, f'({name[0]}{name[1]})')
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

        if draw_sphere:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = -rad * np.cos(u) * np.sin(v)
            y = -rad * np.sin(u) * np.sin(v)
            z = -rad * np.cos(v) - dv
            print(ax.get_xlim())
            print(ax.get_xlim()[0], ax.get_xlim()[1])
            #ax.set_zlim(-300, 0)
            ax.plot_wireframe(x, y, z, color="green", alpha=.2)

        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            max_dis.append(((x[0]-x[1])**2+(y[0]-y[1])**2+(z[0]-z[1])**2)**.5 - d/3**.5)

        # Plot the connecting lines #1f77b4
        if max_dist is None:
            max_dist = max(max_dis)
        print(f'max dist: {max_dist}')
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
            #print(hf.round_sig(max_dis[i]/max_dist), i)
            ax.plot(x, y, z, c=hf.color_fade('#1f77b4', 'red', abs(max_dis[i]/max_dist)), alpha=.75)

    # Set the initial view
    # ax.set_aspect('auto')
    ax.view_init(10, 30)
    # Hide the axes
    # ax.set_axis_off()
    hf.set_axes_equal(ax)
    ax.set_xlabel('x', fontsize=25)
    ax.set_ylabel('y', fontsize=25)
    ax.set_zlabel('z', fontsize=25)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="z", labelsize=20)
    #ax.set_title('dim=20, dv=5, p=10%', size=20)

    plt.show()


def plot_graph(dim, r=1, displace_value=1, sphere=False, d=1, k=2, nodes=False, method='CG', percentile=0, opt=None,
               tol=1.e-03, x0=None, tg=True, seed=None):
    if seed is None:
        seed = rn.random()
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=displace_value)
    if sphere:
        ls = hl.create_lattice_sphere(dim, r ** 2, displace_value, d)
        l = ls[0]
    matrices = hl.dilute_lattice_point(hl.adjacency_matrix(l), percentile, l, seed)
    A = np.add(matrices[0], matrices[1])

    draw_initial_graph(A, 22, generate_manipulated_plot_positions(dim, l,
                                                                  r=r,
                                                                  displace_value=displace_value, sphere=sphere,
                                                                  d=d, k=k, method=method, percentile=percentile,
                                                                  opt=opt, tol=tol, x0=x0, tg=tg, seed=seed),
                       l, nodes=nodes, dv=displace_value, rad=r, draw_sphere=sphere, d=d)


def import_pickle(path):
    pickle_in = open(path, 'rb')
    return pickle.load(pickle_in)


def contour_coords(dim, displace_value, path, tol=.1):
    lattice = hl.create_lattice(dim, 1)
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


def fit_contour(min_dim, max_dim, disp_value):
    print(f'max dim = {max_dim}')
    disp_value = float(disp_value)
    for i in range(min_dim, max_dim + 1):
        if i % 2 == 0:
            print(f'working on dim = {i}')
            path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{disp_value}_0/dim={i}_dv={disp_value}_perc=0.pickle'
            coords = contour_coords(i, disp_value, path=path)
            plt.plot(coords[0], coords[1], marker='o', linestyle='None', c='blue', markersize=1)

            m1 = max(coords[0])
            m2 = min(coords[0])
            x_new = np.linspace(m2, m1, num=1000, endpoint=True)
            f2 = interp1d(coords[0], coords[1])

            plt.plot(x_new, f2(x_new), label=f'dim = {i}, $U={hf.round_sig(coords[2])}$ in J')
    plt.ylabel('z in d', size=20)
    plt.xlabel('x in d', size=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=13.5)
    plt.title(rf'Profil der minimierten Gitter mit gerader dim 6-50 bei $\delta={disp_value}d$', size=20)
    plt.show()


if __name__ == '__main__':
    # d = 3**.5*40*10**(0)
    d = 70
    seed = None
    plot_graph(25, r=d*4, displace_value=10*d, percentile=0, x0=True, tg=True, sphere=True, seed=seed, d=d, k=1000/d**2)
    # fit_contour(5, 50, 10.0)
