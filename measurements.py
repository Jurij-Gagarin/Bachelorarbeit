# In this module we perform our simulations.

import hexagonal_lattice as hl
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
from math import sqrt
from networksx_tests import round_sig


def measure_time(n0, n_max, digit=2):
    times = np.zeros(n_max - n0 + 1)

    for n in range(n0, n_max+1):
        start_time = time.time()
        res = hl.run(n, stretch_factor=5, plot=False).fun
        end_time = time.time()
        delta_t = round((end_time - start_time)/60, digit)
        print(f'With dim = {n}, i.e. {len(hl.create_lattice(n)[0])} nodes,'
              f' it took {delta_t}min to calculate the minimal energy {res}J.')
        times[n - n0] = delta_t
    return times


def energy_func_speedtest(dim, num, d=1, k=2):
    lattice = hl.create_lattice(dim)
    l = hl.manipulate_lattice_absolute_value(lattice[0], lattice[1], 0.3)
    t1 = 0
    t2 = 0
    for i in range(num):
        start_time = time.time()
        energy1 = hl.minimize_energy(l).fun
        end_time = time.time()
        t1 += end_time - start_time

        start_time = time.time()
        energy2 = hl.minimize_energy_opt(l).fun
        end_time = time.time()
        t2 += end_time - start_time

    return energy1, energy2, t1, t2


def energy_continuous_stretching(dim, max_stretch, min_stretch=0, export=False):
    results = []

    for i in range(min_stretch, max_stretch+1):
        results.append([i, hl.run(dim, stretch_factor=i, plot=False).fun])
        print(results[i-min_stretch])

    if export:
        df = pd.DataFrame(data=results, columns=['i', 'min energy'])
        time_now = time.localtime()
        time_now = time.strftime('%H:%M:%S', time_now)
        path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim={dim}_min={min_stretch}_max={max_stretch}_{time_now}.csv'
        df.to_csv(path)

    return results


def x2(x, a):
    return a*x**2


def x4(x, a):
    return a*x**4


def plot_from_csv(path, fit=False):
    df = pd.read_csv(path)
    df.i = df.i*0.25
    df.plot(x='i', y='min energy', marker='o')
    plt.xlabel('Auslenkung in % der Gitterbreite')
    plt.ylabel('berechnete minimale Energie')

    if fit:
        x = df['i']
        y = df['min energy']
        pars4, cov4 = curve_fit(x4, x, y)

        residuals = y - x4(x, pars4)
        plt.plot(x, pars4[0]*x**4, color='red', label='x^4')
    plt.legend()
    plt.grid()
    plt.show()


def absolute_stretching(dim, displace_value, d=1, k=2):
    # dims is a list that contains dimensions to create multiple lattices.
    res = hl.run_absolute_displacement(dim, displace_value, d, k, plot=False)

    return res.fun


def absolute_stretching_multi_lattice(dims, displace_value, num, d=1, k=2):
    displace_values = np.linspace(0, displace_value, num=num)
    all_y = []

    for i in range(0, len(dims)):
        y = []
        for j in range(0, len(displace_values)):
            print(dims[i], displace_values[j])
            y.append(absolute_stretching(dims[i], displace_values[j], d, k))
        all_y.append(y)

    return displace_values, all_y


def plot_multiple_absolute_stretching(values, dims, fit=True):
    x = values[0]
    ys = values[1]

    for i in range(0, len(ys)):
        plt.plot(x, ys[i], label=f'dim={dims[i]}', marker='o', linestyle='none')
        pars, cov = curve_fit(x4, x, ys[i])
        ss_res = np.sum((ys[i] - x4(x, pars[0]))**2)
        ss_tot = np.sum((ys[i]-np.mean(ys[i]))**2)
        r2 = round(1 - (ss_res / ss_tot), 2)

        plt.plot(x, x4(x, pars[0]), label=f'fit dim={dims[i]} with a*x^4, a={round_sig(pars[0])}, error of a={round_sig(sqrt(cov), 1)}')
    plt.legend()
    plt.xlabel('\u03B4')
    plt.ylabel('minimale Energie')
    plt.show()


def energy_convergence(min_dim, max_dim, dv, method='CG', gtol=1.e-06):
    x = list(range(min_dim, max_dim+1))
    y = np.zeros(max_dim - min_dim+1)
    for i in x:
        y[i-min_dim] = hl.run_absolute_displacement(i, dv, plot=False, method=method, gtol=gtol).fun
        print(f'current dim={i}')

    plt.plot(x, y, label=f'{gtol}')


# 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'
# 'trust-ncg', 'trust-krylov', 'trust-exact'
# 'Newton-CG'

# energy_convergence(5, 22, .5, gtol=1.e-08)
# energy_convergence(5, 22, .5, gtol=1.e-05)
# plt.legend()
# plt.show()


def export_pickle(dim, dv, gtol=1.e-10):
    path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim={dim}_dv={dv}_gtol={gtol}.pickle'
    result = hl.run_absolute_displacement(dim, dv, plot=False, gtol=gtol)
    pickle_out = open(path, 'wb')
    pickle.dump(result, pickle_out)
    pickle_out.close()


def import_pickle(dim, dv, gtol=1.e-10):
    path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim={dim}_dv={dv}_gtol={gtol}.pickle'
    pickle_in = open(path, 'rb')
    return pickle.load(pickle_in)


for i in range(11, 50):
    export_pickle(i, 0.1)
    print(f'pickle with dim={i} and dv=.1 successfully exported')


# ssh jurijrudos99@twoheaded.physik.fu-berlin.de