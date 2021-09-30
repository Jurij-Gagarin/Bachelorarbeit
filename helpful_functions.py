import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from math import floor, log10


def color_fade(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1 - mix) * c1 + mix * c2)


def round_sig(x, sig=2):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


if __name__ == '__maine__':
    c1 = '#80d8ff'
    c2 = '#D21404'
    n=500

    fig, ax = plt.subplots(figsize=(8, 5))
    for x in range(n + 1):
        ax.axvline(x, color=color_fade(c1, c2, x / n), linewidth=4)
    plt.show()
