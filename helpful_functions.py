import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from math import floor, log10



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def color_fade(c1, c2, mix=0.0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def plot_colorbar(max_dist, d):
    c1 = '#1f77b4'
    c2 = 'red'
    n = 500

    fig, ax = plt.subplots(figsize=(15, 5))
    for x in range(n + 1):
        ax.axvline(x/n, color=color_fade(c1, c2, x / n), linewidth=4)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(.01)
    eps = d/3**.5
    ratio = max_dist/eps**2
    labels = [1, 1 + .2*ratio, 1 + .4*ratio, 1 + .6*ratio, 1 + .8*ratio, 1 + ratio]
    for i in range(len(labels)):
        labels[i] = str(round_sig(labels[i], 3)) + '$ \epsilon$'
    print(labels)
    ax.set_xticklabels(labels)
    plt.show()


def create_seed_list():
    f = open('./seed_list.txt', 'a')
    seeds = rn.sample(list(range(100000, 1000000)), 100)
    for i in seeds:
        f.writelines(['\n', str(i)])
    f.close()



