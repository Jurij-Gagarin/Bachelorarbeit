import measurements as m
import hexagonal_lattice as hl
import numpy as np
import plot
import pickle
import matplotlib.pyplot as plt
from math import floor, log10, sqrt
from scipy.optimize import curve_fit
import helpful_functions as hf
import os


def single_plot_from_pickle(dim, dv, path, perc=0, seed=None, d=1, max_dist=None):
    ls = hl.create_lattice(dim, d)
    l = ls[0]
    l = hl.manipulate_lattice_absolute_value(l, ls[1], displace_value=dv)
    matrices = hl.dilute_lattice_point(hl.adjacency_matrix(l), perc, l, seed)
    A = np.add(matrices[0], matrices[1])
    pos = {}
    pic = pickle.load(open(path, 'rb'))
    list_mobile_coords = pic.x
    print('Energy: ', pic.fun)

    for i in range(0, len(l)):
        if l[i].return_mobility():
            pos[i] = (list_mobile_coords[0], list_mobile_coords[1], list_mobile_coords[2])

            list_mobile_coords = list_mobile_coords[3:]
        elif not l[i].return_mobility():
            vector = l[i].return_coordinates()
            pos[i] = (vector[0], vector[1], vector[2])

    plot.draw_initial_graph(A, 22, pos, l, max_dist=max_dist)


def print_convergence(dim, dv, gtol=1.e-10, perc=0):
    obj = m.import_pickle(dim, dv, gtol, perc)
    # print(gtol, obj.fun, obj.message)
    print(obj)


def plot_energy_convergence(dv, min_d, max_d):
    energy = np.zeros(max_d - min_d + 1)
    dims = list(range(min_d, max_d + 1))
    for i in range(min_d, max_d + 1):
        path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}_0/dim={i}_dv={dv}_perc=0.pickle'
        pic = pickle.load(open(path, 'rb'))
        energy[i - min_d] = pic.fun
        if not pic.success:
            print(pic.message, f'dim={i}')

    fig, ax = plt.subplots(figsize=[15, 10])
    ax.scatter(dims, energy, label='Daten')
    ax.set_title('Minimale Energie aufgetragen gegen dim', size=20)
    ax.set_ylabel('Minimale Energie in J', size=20)
    ax.set_xlabel('dim', size=20)
    ax.legend(fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_aspect(.2)
    plt.show()


def calculate_distance(dim, dv, path, perc=0, d=1):
    lattice = hl.create_lattice(dim, d)
    lattice = hl.manipulate_lattice_absolute_value(lattice[0], lattice[1], dv)
    pic = pickle.load(open(path, 'rb'))
    distance = []
    x = pic.x
    adj = hl.dilute_lattice_point(hl.adjacency_matrix(lattice), perc)
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
            paths = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dv}_0/dim={d}_dv={dv}_perc=0.pickle'
            arr = calculate_distance(d, dv, paths)
            maximum = max(arr)
            plt.hist(arr, density=True, bins=200, label=f'dim={d}, dv={dv}, max elongation = '
                                                        f'{round(maximum, 2 - int(floor(log10(abs(maximum)))) - 1)}')
        dvs_str = ', '.join(str(e) for e in dvs)
        plt.title(f'Elongation-histogram-plot for lattices with different dims, displaced by dv={dvs_str}', size=20)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Anzahl', fontsize=20)
    plt.xlabel('Abstand - Gleichgewichtsabstand', fontsize=20)
    plt.show()


def plot_links_mean_value(dims, dvs):
    means = []
    links = []
    for d in dims:
        paths = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_5-50_{dvs[0]}_0/dim={d}_dv={dvs[0]}_perc=0.pickle'
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


def make_x4(b):
    x0 = b

    def x4(x, a):
        return a*(x-x0)**4
    return x4


def x4(x, a):
    return a*(x)**4


def plot_energy_vs_dv(dim, min_dv, max_dv, dv_step=.5, perc=0):
    energy = []
    dvs = []
    fit = []
    for i in range(min_dv, max_dv):
        path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_{dim}_0-30_0/dim={dim}_dv={float(i)}_perc={perc}_1.pickle'
        pic = pickle.load(open(path, 'rb'))
        dvs.append(i)
        energy.append(pic.fun)

    fig, ax = plt.subplots(figsize=[15, 10])
    pars, cov = curve_fit(x4, dvs, energy)
    x = np.linspace(min_dv, max_dv, num=100)
    for i in x:
        fit.append(x4(i, pars[0]))
    ax.plot(dvs, energy, marker='o', linestyle='none', label='Daten')
    ax.plot(x, fit, label=f'${hf.round_sig(pars[0])}x^4$, Parameterfehler {hf.round_sig(sqrt(cov), 1)}')
    ax.set_title('Minimale Energie aufgetragen gegen dv', size=20)
    ax.set_ylabel('Minimale Energie in J', size=20)
    ax.set_xlabel('dv in Vielfachen von d', size=20)
    ax.legend(fontsize=15)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_aspect(.2)

    plt.show()


def energy_dil_lattice(path, dim, dv, perc):
    f = open('./seed_list.txt', 'r')
    seed = list(map(int, f.readlines()))
    n = len(seed)
    energy = np.zeros(n)

    for i in range(int(n)):
        paths = path + f'dim={dim}_dv={dv}_perc={perc}_{seed[i]}.pickle'
        energy[i] = pickle.load(open(paths, 'rb')).fun
    print(np.mean(energy), np.std(energy)/sqrt(n))
    f.close()
    plt.hist(energy, bins=20)
    plt.show()
    return energy


def diluted_lattice(dvs, ps, path, plot_energy=False, plot_e_module=False):
    file_names = os.listdir(path)
    fig, ax = plt.subplots(figsize=[15, 10])
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'blue', 'orange',
             'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    e_module = []
    e_module_error = []

    for p in ps:
        mean_energy = []
        standard_deviation = []
        del_index = []
        for d in range(len(dvs)):
            energy = []
            count = 0
            for i in file_names:
                if f'dv={dvs[d]}' in i and f'perc={p}' in i:
                    energy.append(pickle.load(open(path + '/' + i, 'rb')).fun)
                    count += 1

            if not energy:
                ener = 0
            else:
                ener = np.mean(energy)

            if ener > 0:
                mean_energy.append(ener)
                standard_deviation.append(1*np.std(energy)/len(energy)**.5)
            else:
                del_index.append(d)
            print(f'p={p}, d={d}, number of measurements {count}')
        dvsc = np.delete(dvs, del_index)
        if len(del_index) == 0:
            del_index.append(-1)

        pars, cov = curve_fit(x4, dvsc, mean_energy)#, sigma=standard_deviation)
        x0 = dvs[del_index[0]]
        if p==0: x0=0
        #pars, cov = curve_fit(make_x4(x0), dvsc, mean_energy)  # , sigma=standard_deviation)
        e_module.append(pars[0])
        e_module_error.append(np.sqrt(np.diag(cov))[0])
        if plot_energy:
            ax.scatter(dvsc, mean_energy, c=color[0])
            plt.errorbar(dvsc, mean_energy, yerr=standard_deviation, xerr=0, fmt='none', ecolor=color[0], capsize=5)
            color.pop(0)
            x = np.linspace(dvsc[0], dvsc[-1], num=100)
            fit = []
            for i in x:
                fit.append(x4(i, pars[0]))
            ax.plot(x, fit, label=rf'p={2*p}%: a={hf.round_sig(pars[0])}, $\Delta a={hf.round_sig(np.sqrt(np.diag(cov))[0])}$ ')
                                  #rf'b={hf.round_sig(pars[1])}') #label=rf'p={2*p/100}%: $a={hf.round_sig(pars[0])}, b={hf.round_sig(pars[1])}$,'
                                  #rf' $\Delta a={hf.round_sig(np.sqrt(np.diag(cov))[0])}, \Delta b={hf.round_sig(np.sqrt(np.diag(cov))[1])}$')

    if plot_energy:
        ax.set_title('Mittlere minimale Energie aufgetragen gegen dv, für verschiedene p', size=20)
        ax.set_ylabel('Mittlere minimale Energie in J', size=20)
        ax.set_xlabel('dv in Vielfachen von d', size=20)
        ax.legend(fontsize=20)
        ax.tick_params(axis="x", labelsize=15)
        ax.tick_params(axis="y", labelsize=15)
        plt.show()

    if plot_e_module:
        ax.set_title('E-Module eines Gitters für verschiedene Verdünnungen', size=20)
        ax.set_ylabel('E-Modul in J/m', size=20)
        ax.set_xlabel('p in Prozent', size=20)
        for i in range(len(ps)):
            ps[i] = ps[i]*2
        ax.scatter(ps, e_module, c='black')
        plt.errorbar(ps, e_module, yerr=e_module_error, xerr=0, fmt='none', ecolor='black', capsize=5)
        plt.show()

    return e_module, e_module_error







dvs = list(np.arange(15.0, 2.5-.5, -.5))
#ps = [0.0, 0.5, 1.5, 5.0, 10.0]
ps = [0.0, 0.5, 1.0, 1.5, 2.5, 3.75, 5.0, 6.0, 7.5, 9.0, 10.0]
seed = 122775
n = 0
path = '/home/jurij/Python/Physik/Bachelorarbeit-Daten/punktuell'
print(diluted_lattice(dvs, ps, path, plot_e_module=False, plot_energy=True))


# path = f'/home/jurij/Python/Physik/Bachelorarbeit/current_measurements/dim={dim}_dv={dv}_perc={perc}_{seed}.pickle'
# path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim_20_5.0_5/dim=20_dv=5.0_perc=5_{seed}.pickle'
# single_plot_from_pickle(dim, dv, path, perc, seed, max_dist=0.11055606682194574)
# print(energy_dil_lattice(path, dim, dv, perc))


