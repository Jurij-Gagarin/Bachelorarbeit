import hexagonal_lattice as hl
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def measure_time(n0, n_max, digit=2):
    times = np.zeros(n_max - n0 + 1)

    for n in range(n0, n_max+1):
        start_time = time.time()
        res = hl.run(n, plot=False)
        end_time = time.time()
        delta_t = round((end_time - start_time)/60, digit)
        print(f'With dim = {n}, i.e. {len(hl.create_lattice(n))} nodes,'
              f' it took {delta_t}min to calculate the minimal energy {res}J.')
        times[n - n0] = delta_t
    return times


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


def plot_from_csv(path, fit=False, a0=1):
    df = pd.read_csv(path)
    df.i = df.i*0.25
    df.plot(x='i', y='min energy', marker='o')
    plt.xlabel('Auslenkung in % der Gitterbreite')
    plt.ylabel('berechnete minimale Energie')

    if fit:
        x = df['i']
        y = df['min energy']
        pars2, cov2 = curve_fit(x2, x, y)
        pars4, cov4 = curve_fit(x4, x, y)

    plt.plot(x, pars2[0]*x**2, label='x^2')
    plt.plot(x, pars4[0]*x**4, color='red', label='x^4')
    plt.legend()
    plt.grid()
    plt.show()
    # path = '/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim=9_min=0_max=40_19:23:44.csv'


def absolute_stretching(dims):
    # dims is a list that contains dimensions to create multiple lattices
    pass

# TODO: create a function that minimises lattices with different number of nodes that are manipulated in the same way
#  (same point, some absolute value)
# TODO: test lattice with very large displacements

