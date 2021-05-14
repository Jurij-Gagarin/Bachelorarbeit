import hexagonal_lattice as hl
import time
import numpy as np
import pandas as pd


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
        results.append([i, hl.run(dim, stretch_factor=i, plot=False)])
        print(results[i-min_stretch])

    if export:
        df = pd.DataFrame(data=results, columns=['i', 'min energy'])
        time_now = time.localtime()
        time_now = time.strftime('%H:%M:%S', time_now)
        path = f'/home/jurij/Python/Physik/Bachelorarbeit/measurements/dim={dim}_min={min_stretch}_max={max_stretch}_{time_now}.csv'
        df.to_csv(path)

    return results
