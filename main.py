import numpy as np
import matplotlib.pyplot as plt
import random as rn


# returns a lattice K with n x n knots
def create_lattice(n, box_length=100):
    # calculate equilibrium distance
    equilibrium = box_length/(n-1)

    # calculate coordinates of lattice points
    n2 = n**2
    shape_coordinates = (n2, 3)
    K = np.zeros(shape_coordinates)
    I = 0
    J = 0

    for i in range(0, n):
        for j in range(0, n):
            K[I+J] = [j*equilibrium, i*equilibrium, 0]
            J += 1
        J -= 1
        I += 1
    return K

# takes a lattice and returns a randomly (default) displaced lattice
def displace_lattice(lattice, n, m=0, random=True):
    D = np.zeros((n**2, 3))
    if random:
        D[rn.randint(0, n**2)] = [0, 0, 1]
    else:
        D[m] = [0, 0, 1]

    return lattice + D

# plots the lattice
def plot_lattice(lattice):
    x = lattice[:, 0]
    y = lattice[:, 1]
    z = lattice[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.show()


n = 5
L = displace_lattice(create_lattice(n), n)
plot_lattice(L)
