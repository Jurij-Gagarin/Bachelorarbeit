import measurements as m
import hexagonal_lattice as hl
import numpy as np
import plot
import pickle
import matplotlib.pyplot as plt
from math import floor, log10, sqrt


def single_plot_from_pickle(dim, dv, path, perc=0, d=1, max_dist=None, max_dist_dim=None):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=dv)
    matrices = hl.dilute_lattice_point(hl.adjacency_matrix(l), perc)
    A = np.add(matrices[0], matrices[1])
    if max_dist is None:
        path2 = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}/dim={max_dist_dim}_dv={dv}_perc=0.pickle'
        max_dist = max(calculate_distance(max_dist_dim, dv, path2, perc, d)) + d/sqrt(3)

    pos = {}
    list_mobile_coords = pickle.load(open(path, 'rb')).x

    for i in range(0, len(l)):
        if l[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not l[i].return_mobility():
            vector = l[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])

    plot.draw_initial_graph(A, 22, pos, l, max_dist)


def print_convergence(dim, dv, gtol=1.e-10, perc=0):
    obj = m.import_pickle(dim, dv, gtol, perc)
    # print(gtol, obj.fun, obj.message)
    print(obj)


def plot_energy_convergence(dv, min_d, max_d):
    energy = np.zeros(max_d - min_d + 1)
    dims = list(range(min_d, max_d + 1))
    for i in range(min_d, max_d + 1):
        path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}/dim={i}_dv={dv}_perc=0.pickle'
        pic = pickle.load(open(path, 'rb'))
        energy[i - min_d] = pic.fun
        if not pic.success:
            print(pic.message, f'dim={i}')
    x_min = 5
    plt.scatter(dims[x_min - min_d:], energy[x_min - min_d:], marker='o', s=15, label=f'dv={dv}')
    plt.xlabel('dim', size=16)
    plt.ylabel('energy', size=16)
    plt.legend()
    plt.title('Minimum energy for lattices of different scales displaced by different dv')
    plt.show()


def calculate_distance(dim, dv, path, perc=0, d=1):
    lattice = hl.create_lattice(dim, d)
    lattice = hl.manipulate_lattice_absolute_value(lattice[0], lattice[1], dv)
    pic = pickle.load(open(path, 'rb'))
    distance = []
    x = pic.x
    adj = hl.dilute_lattice(hl.adjacency_matrix(lattice), perc)
    mrows, mcols, imrows, imcols, e = hl.energy_func_prep(np.triu(adj[0]), np.triu(adj[1]), d)
    xs, no_use, xsdict, xdict = hl.list_of_coordinates(lattice)

    for i in range(len(mrows)):
        rpos = xdict[mrows[i]]
        cpos = xdict[mcols[i]]
        distance.append(((x[rpos] - x[cpos]) ** 2 + (x[rpos + 1] - x[cpos + 1]) ** 2 + (
                x[rpos + 2] - x[cpos + 2]) ** 2) ** .5 - e)

    for i in range(len(imrows)):
        if (lattice[imrows[i]].return_mobility() is True) and (lattice[imcols[i]].return_mobility() is False):
            rpos = xdict[imrows[i]]
            cpos = xsdict[imcols[i]]
            distance.append(((x[rpos] - xs[cpos]) ** 2 + (x[rpos + 1] - xs[cpos + 1]) ** 2 + (
                    x[rpos + 2] - xs[cpos + 2]) ** 2) ** .5 - e)

        elif (lattice[imrows[i]].return_mobility() is False) and (lattice[imcols[i]].return_mobility() is True):
            rpos = xsdict[imrows[i]]
            cpos = xdict[imcols[i]]
            distance.append(((xs[rpos] - x[cpos]) ** 2 + (xs[rpos + 1] - x[cpos + 1]) ** 2 + (
                    xs[rpos + 2] - x[cpos + 2]) ** 2) ** .5 - e)

    return distance


def plot_histograms(dims, dvs):
    for d in dims:
        for dv in dvs:
            paths = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}/dim={d}_dv={dv}_perc=0.pickle'
            arr = calculate_distance(d, dv, paths)
            maximum = max(arr)
            plt.hist(arr, density=True, bins=200, label=f'dim={d}, dv={dv}, max elongation = '
                                                        f'{round(maximum, 2 - int(floor(log10(abs(maximum)))) - 1)}')
        dvs_str = ', '.join(str(e) for e in dvs)
        plt.title(f'Elongation-histogram-plot for lattices with different dims, displaced by dv={dvs_str}')
    plt.legend()
    plt.show()


def plot_links_mean_value(dims, dvs):
    means = []
    links = []
    for d in dims:
        paths = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dvs[0]}/dim={d}_dv={dvs[0]}_perc=0.pickle'
        arr = calculate_distance(d, dvs[0], paths)
        mean = np.mean(arr)
        lattice = hl.create_lattice(d, 1)[0]
        r = hl.list_of_coordinates(lattice)
        links.append(len(r[0])+len(r[1]))
        means.append(mean)
        print(d, dvs[0])
    plt.plot(links, means, label=f'dv={dvs[0]}')

    dvs.pop(0)
    if len(dvs) > 0:
        plot_links_mean_value(dims, dvs)

    plt.legend()
    plt.xlabel('Number of links', size=16)
    plt.ylabel('Mean value of all elongations', size=16)
    plt.title('Mean elongation Value plotted against the number of links', size=20)
    plt.show()


def plot_max_elongation2_vs_energy(dims, dv):
    elong_max2 = []
    energy = []
    for d in dims:
        paths = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}/dim={d}_dv={dv}_perc=0.pickle'
        arr = calculate_distance(d, dv, paths)
        elong_max2.append(max(arr) ** 2)
        pic = pickle.load(open(paths, 'rb'))
        print(d, ':', pic.message)
        energy.append(pic.fun)

    plt.scatter(dims, elong_max2, label='max_elong^2 over dim')
    plt.scatter(dims, energy, label='energy over dim')
    plt.title('Maximum Elongation squared vs. energy')
    plt.legend()
    plt.show()


dim = 30
dv = 10.0
path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}/dim={dim}_dv={dv}_perc=0.pickle'
single_plot_from_pickle(dim, dv, path, max_dist_dim=15)
