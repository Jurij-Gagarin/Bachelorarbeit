# In this module we will create our lattice, as well as some basic functions.

import math
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from scipy import optimize as opt
import pickle
import helpful_functions as hf
import os


class Node:
    # objects from this class are the nodes of the honeycomb lattice
    # they have the attributes coordinates [x, y, z] and name [n1, n2, b]
    # n1 and n2 are the factors of the primitive lattice vectors b = 0,1 denotes the basis vector
    x = 0
    y = 0
    z = 0
    name = [0, 0, 0]
    movable = True

    def __init__(self, n1, n2, d, b, z=0):
        self.x = n1 * d / 2 - n2 * d / 2
        self.y = (n1 + n2) * math.sqrt(3) * d / 2 - b * d / math.sqrt(3)
        # self.z = 0
        self.name = [n1, n2, b]

    def change_coordinates(self, r):
        self.x = self.x + r[0]
        self.y = self.y + r[1]
        self.z = self.z + r[2]

        return self

    def new_coordinates(self, r):
        self.x = r[0]
        self.y = r[1]
        self.z = r[2]

        return self

    def change_mobility(self, bo):
        self.movable = bo

        return self

    def return_mobility(self):
        return self.movable

    def return_coordinates(self):
        return [self.x, self.y, self.z]

    def return_name(self):
        return self.name


def create_lattice(dim, d):
    # d is distance between two nodes on the x-axis
    # dim is the maximal summed factor for the lattice vectors
    lattice = []
    box_length = d * dim
    skip = False

    for k in range(0, 2):
        for i in range(-dim, dim + 1):
            for j in range(-dim, dim + 1):
                if (abs(i) + abs(j)) <= dim:
                    lattice.append(Node(i, j, d, k))
                    x = lattice[-1].return_coordinates()[0]
                    y = lattice[-1].return_coordinates()[1]
                    if (abs(i) + abs(j)) == dim:
                        lattice[-1] = lattice[-1].change_mobility(False)
                    if skip:
                        lattice[-1] = lattice[-1].change_mobility(False)
                        skip = False
                    if (abs(x)-abs(x)/10000 > box_length / 2) or (abs(y)-abs(y)/10000 > box_length / 2):
                        lattice.pop(-1)
                        skip = True
                        if len(lattice) > 0:
                            lattice[-1] = lattice[-1].change_mobility(False)
                    if (j == 0) and (i == 0) and (k == 0):
                        mid_point = len(lattice) - 1

    return lattice, mid_point


def create_lattice_sphere(dim, r2, dv, d):
    # d is distance between two nodes on the x-axis
    # dim is the maximal summed factor for the lattice vectors
    lattice, mid_point = create_lattice(dim, d)
    r = math.sqrt(r2)
    for i in range(len(lattice)):
        x, y = lattice[i].return_coordinates()[0], lattice[i].return_coordinates()[1]
        mobility = lattice[i].return_mobility()
        if x**2 + y**2 < r2 and mobility:
            lattice[i].change_coordinates([0, 0, -math.sqrt(r2-x**2-y**2)-dv])
        elif mobility:
            lattice[i].change_coordinates([0, 0, -dv*r/(math.sqrt(x**2+y**2))])
    return lattice, mid_point


def manipulate_lattice_random(lattice, n=1):
    # changes z-component from n random nodes. Not used in simulation.
    for k in range(0, n):
        i = rn.randint(0, len(lattice) - 1)
        j = rn.randint(0, 10) / 1
        lattice[i] = lattice[i].change_coordinates([0, 0, j])
        lattice[i] = lattice[i].change_mobility(False)

    return lattice


def manipulate_lattice(lattice, d, dim, point, stretch_factor=5):
    # stretch with factor = 1: 1% of lattice length
    j = stretch_factor * d * dim / 100
    lattice[point] = lattice[point].change_coordinates([0, 0, j])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def manipulate_lattice_absolute_value(lattice, point, displace_value):
    # Displaces middle point by an absolute value
    lattice[point] = lattice[point].new_coordinates([0, 0, displace_value])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def adjacency_matrix(lattice, plot=False):
    # For n particles this creates a n x n matrix that shows the connections between the nodes
    # A represents only connections of movable particles
    # As represents only connections of immovable particles
    dim = len(lattice)
    As = np.zeros((dim, dim))
    A = np.zeros((dim, dim))
    for i in range(0, dim):
        nz = lattice[i].return_name()
        for j in range(0, dim):
            ns = lattice[j].return_name()
            mobilityi = lattice[i].return_mobility()
            mobilityj = lattice[j].return_mobility()
            if nz[2] != ns[2]:
                if ns[2] == 0 and ((nz[1] == ns[1] and nz[0] == ns[0]) or
                                   (nz[1] == ns[1] and nz[0] == ns[0] + 1) or
                                   (nz[1] == ns[1] + 1 and nz[0] == ns[0])):
                    if mobilityi is True and mobilityj is True:
                        A[i][j] = 1
                    else:
                        As[i][j] = 1

                if ns[2] == 1 and ((nz[1] == ns[1] and nz[0] == ns[0]) or
                                   (nz[1] == ns[1] and nz[0] == ns[0] - 1) or
                                   (nz[1] == ns[1] - 1 and nz[0] == ns[0])):
                    if mobilityi is True and mobilityj is True:
                        A[i][j] = 1
                    else:
                        As[i][j] = 1

    if plot:
        # Creates a color plot of the adjacency matrix.
        plt.imshow(np.add(A, As))
        plt.axis('off')
        plt.show()

    return A, As


def clear_lattice(A, As):
    # Clears the lattice from floating nodes, that are connected to only one edge
    s = len(A[0])
    changes = True
    while changes:
        changes = False
        single_rows = np.where(np.sum(np.add(A, As), 1) == 1)[0]
        single_cols = np.where(np.sum(np.add(A, As), 0) == 1)[0]
        if len(single_rows) > 0 or len(single_cols) > 0:
            changes = True
            for i in single_rows:
                A[i] = np.zeros(s)
                As[i] = np.zeros(s)
            for i in single_cols:
                A[:, i] = np.zeros(s)
                As[:, i] = np.zeros(s)

    return A + np.transpose(A), As + np.transpose(As)


def dilute_lattice_point(adjacency_matrix, percentile, lattice, seed):
    # dilutes blue nodes from graph and the corresponding edges
    A = np.tril(adjacency_matrix[0])
    As = np.tril(adjacency_matrix[1])
    if percentile == 0:
        return clear_lattice(A, As)
    n = len(A)
    m = len(A[0])
    blue_nodes = []

    for i in range(len(lattice)):
        if lattice[i].return_name()[2] == 1:
            blue_nodes.append(i)

    rn.seed(seed)
    rows_to_delete = rn.sample(blue_nodes, int(hf.round_sig(n * percentile / 100)))
    for j in rows_to_delete:
        A[j] = np.zeros(m)

    return clear_lattice(A, As)


def dilute_lattice(adjacency_matrix, percentile):
    # Dilutes random edges from the graph. Not used in simulation.
    A = np.triu(adjacency_matrix[0])
    As = np.triu(adjacency_matrix[1])
    rows, cols = np.where(A == 1)
    dil = round(len(rows) * percentile / 100)
    dilution = rn.sample(list(range(len(rows))), dil)

    for i in dilution:
        A[rows[i]][cols[i]] = 0

    return clear_lattice(A, As)


def list_of_coordinates(lattice):
    '''
    Takes coordinates from the lattice and sorts them in two np-arrays. One for mobile coordinates and one
    for immobile coordinates. Two dictionary's are created as well connect the position in lattice-lists and
    the new coordinate lists.
    '''
    list_of_mobile_coords = np.array([])
    list_of_immobile_coords = np.array([])
    mobile_dict = {}
    immobile_dict = {}
    for i in range(0, len(lattice)):
        r = lattice[i]
        if r.return_mobility():
            list_of_mobile_coords = np.append(list_of_mobile_coords, r.return_coordinates())
            mobile_dict[i] = len(list_of_mobile_coords) - 3

        else:
            list_of_immobile_coords = np.append(list_of_immobile_coords, r.return_coordinates())
            immobile_dict[i] = len(list_of_immobile_coords) - 3

    return list_of_immobile_coords, list_of_mobile_coords, immobile_dict, mobile_dict


def energy_func_prep(A, As, d):
    # This does some basic preparation for the optimized energy functions
    mrows, mcols = np.where(A == 1)
    imrows, imcols = np.where(As == 1)
    e = d / math.sqrt(3)
    return mrows, mcols, imrows, imcols, e


def energy_func_opt(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, k):
    # calculates lattice energy (optimized).
    total_energy = 0

    for i in range(len(mrows)):
        rpos = xdict[mrows[i]]
        cpos = xdict[mcols[i]]
        total_energy += (((x[rpos] - x[cpos]) ** 2 + (x[rpos + 1] - x[cpos + 1]) ** 2 + (
                x[rpos + 2] - x[cpos + 2]) ** 2) ** .5 - e) ** 2

    for i in range(len(imrows)):
        if (lattice[imrows[i]].return_mobility() is True) and (lattice[imcols[i]].return_mobility() is False):
            rpos = xdict[imrows[i]]
            cpos = xsdict[imcols[i]]
            total_energy += (((x[rpos] - xs[cpos]) ** 2 + (x[rpos + 1] - xs[cpos + 1]) ** 2 + (
                    x[rpos + 2] - xs[cpos + 2]) ** 2) ** .5 - e) ** 2

        elif (lattice[imrows[i]].return_mobility() is False) and (lattice[imcols[i]].return_mobility() is True):
            rpos = xsdict[imrows[i]]
            cpos = xdict[imcols[i]]
            total_energy += (((xs[rpos] - x[cpos]) ** 2 + (xs[rpos + 1] - x[cpos + 1]) ** 2 + (
                    xs[rpos + 2] - x[cpos + 2]) ** 2) ** .5 - e) ** 2

    return .5 * k * total_energy


def energy_func_sphere(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, r, dv, k):
    total_energy = 0
    total_energy += energy_func_opt(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, k)
    for i in range(int(len(x)/3)):
        rad_node = (r / (x[3 * i] ** 2 + x[3 * i + 1] ** 2 + (x[3 * i + 2] + dv) ** 2) ** .5)**6
        total_energy += rad_node ** 2 #- rad_node

    return total_energy


def energy_func_jac_opt(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, k=2):
    # Calculates jacobian i.e. the gradient for the energy function.
    len_x = len(x)
    grad = np.zeros(len_x)

    for i in range(int(len_x / 3)):
        index = np.where(A[xdict_reverse[3 * i]] == 1)[0]
        for j in range(len(index)):
            # cn = corresponding_node
            if lattice[index[j]].return_mobility():
                cn = xdict[index[j]]
                root = ((x[3 * i] - x[cn]) ** 2 + (x[3 * i + 1] - x[cn + 1]) ** 2 + (
                        x[3 * i + 2] - x[cn + 2]) ** 2) ** .5
                factor = k * (root - e) / root
                grad[3 * i] += factor * (x[3 * i] - x[cn])
                grad[3 * i + 1] += factor * (x[3 * i + 1] - x[cn + 1])
                grad[3 * i + 2] += factor * (x[3 * i + 2] - x[cn + 2])
            else:
                cn = xsdict[index[j]]
                root = ((x[3 * i] - xs[cn]) ** 2 + (x[3 * i + 1] - xs[cn + 1]) ** 2 + (
                        x[3 * i + 2] - xs[cn + 2]) ** 2) ** .5
                factor = k * (root - e) / root
                grad[3 * i] += factor * (x[3 * i] - xs[cn])
                grad[3 * i + 1] += factor * (x[3 * i + 1] - xs[cn + 1])
                grad[3 * i + 2] += factor * (x[3 * i + 2] - xs[cn + 2])
    return grad


def energy_func_jac_sphere(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, r, dv, k):
    # Calculates same jacobian as in energy_func_jac_opt() with an added
    # Lennard-Jones-like potential
    grad = energy_func_jac_opt(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, xdict_reverse, k)

    for i in range(int(len(grad)/3)):
        abs = x[3*i]**2+x[3*i+1]**2+(x[3*i+2]+dv)**2
        factor = -12 * r**12/abs**7 #+ 6*r**6/abs**4
        grad[3 * i] += factor * x[3 * i]
        grad[3 * i + 1] += factor * x[3 * i + 1]
        grad[3 * i + 2] += factor * (x[3 * i + 2] + dv)
    return grad


def minimize_energy_opt(lattice, method, tol, d, k, option, x0, A, jac_func):
    # Structures the act of energy minimization.
    r = list_of_coordinates(lattice)
    preps = energy_func_prep(np.triu(A[0]), np.triu(A[1]), d)
    args = (r[3], r[0], r[2], preps[0], preps[1], preps[2], preps[3], lattice, preps[4], np.add(A[0], A[1]),
            {v: k for k, v in r[3].items()}, k)

    if x0 is None:
        minimum = opt.minimize(energy_func_opt, r[1], method=method, jac=jac_func, tol=tol,
                               args=args, options=option)
    else:
        minimum = opt.minimize(energy_func_opt, x0, method=method, jac=jac_func, tol=tol,
                               args=args, options=option)

    return minimum


def minimize_energy_sphere(lattice, method, tol, d, k, option, x0, A, jac_func, rad, dv):
    # Structures the act of energy minimization. For spherical indenter.
    r = list_of_coordinates(lattice)
    preps = energy_func_prep(np.triu(A[0]), np.triu(A[1]), d)
    args = (r[3], r[0], r[2], preps[0], preps[1], preps[2], preps[3], lattice, preps[4], np.add(A[0], A[1]),
            {v: k for k, v in r[3].items()}, rad, dv, k)

    if x0 is None:
        minimum = opt.minimize(energy_func_sphere, r[1], method=method, jac=jac_func, tol=tol,
                               args=args, options=option)
    else:
        minimum = opt.minimize(energy_func_sphere, x0, method=method, jac=jac_func, tol=tol,
                               args=args, options=option)

    return minimum


def assemble_result(result, fixed_values, plot=False):
    # Takes the list of coordinates from the minimized lattice and returns them as vectors.
    x = []
    y = []
    z = []
    xf = []
    yf = []
    zf = []

    for i in range(0, len(result)):
        if i % 3 == 0:
            x.append(result[i])
        elif i % 3 == 1:
            y.append(result[i])
        else:
            z.append(result[i])

    for i in range(0, len(fixed_values)):
        if i % 3 == 0:
            xf.append(fixed_values[i])
        elif i % 3 == 1:
            yf.append(fixed_values[i])
        else:
            zf.append(fixed_values[i])

    # Very basic plot.
    if plot:
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim([-20, 20])
        # ax.set_ylim([-20, 20])
        ax.set_xlabel('x-Achse')
        ax.set_ylabel('y-Achse')
        ax.scatter(x, y, z, c='blue')
        ax.scatter(xf, yf, zf, c='red')
        plt.show()

    return np.concatenate([x, xf]), np.concatenate([y, yf]), np.concatenate([z, zf])


def run_absolute_displacement(dim, displace_value, d=1, k=2, plot=False, method='CG', tol=1.e-3, percentile=0,
                              opt=None, true_convergence=True, x0=None, jac_func=energy_func_jac_opt, seed=None):
    ls = create_lattice(dim, d)
    l = ls[0]
    l = manipulate_lattice_absolute_value(l, ls[1], displace_value)
    A = dilute_lattice_point(adjacency_matrix(l), percentile, l, seed)

    if x0:
        path = f'./measurements/dim_25_0-30_0/dim={dim}_dv={displace_value}_perc=0_1.pickle'
        x0 = pickle.load(open(path, 'rb')).x

    res = minimize_energy_opt(lattice=l, d=d, k=k, method=method, tol=tol, option=opt, x0=x0, A=A, jac_func=jac_func)
    # print(res.fun, res.message, f'tol={tol}')
    j = 0

    if true_convergence and res.success:
        j = 1

        res2 = minimize_energy_opt(lattice=l, d=d, k=k, method=method, tol=tol / 10 ** j, option=opt, x0=res.x,
                                   A=A, jac_func=jac_func)
        # print(res2.fun, res2.message, f'tol={tol / 10 ** j}')
        while abs(1 - res2.fun / res.fun) > .05:

            if not res2.success:
                # print('Minimization failed, try to increase k')
                print(res2.message)
                break
            j += 1
            res = res2
            res2 = minimize_energy_opt(lattice=l, d=d, k=k, method=method, tol=tol / 10 ** j, option=opt, x0=res.x, A=A,
                                       jac_func=jac_func)
            # print(res2.fun, res2.message, f'tol={tol / 10 ** j}')
        res = res2

    print(res.fun, res.message, f'tol={tol / 10 ** j}')

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0], plot)

    return res


def run_sphere(dim, rad, dv=0, d=1, k=2, plot=False, method='CG', tol=1.e-3, percentile=0,
               opt=None, true_convergence=True, x0=None, jac_func=energy_func_jac_sphere, seed=None):
    ls = create_lattice_sphere(dim, rad ** 2, dv, d)
    l = ls[0]
    #l = manipulate_lattice_absolute_value(l, ls[1], -rad-dv)
    A = dilute_lattice_point(adjacency_matrix(l), percentile, l, seed)

    if x0:
        path = f'./measurements/x0_sphere'
        file_names = os.listdir(path)
        for i in file_names:
            if f'dv={dv}' in i and f'dim={dim}' in i:
                x0 = pickle.load(open(path + '/' + i, 'rb')).x
                print('Found x0 at \n ', path + '/' + i)
                break
        if isinstance(x0, bool):
            print('Did not found x0. Proceed with x0=None')
            x0 = None

    res = minimize_energy_sphere(lattice=l, d=d, k=k, method=method, tol=tol, option=opt, x0=x0, A=A, jac_func=jac_func,
                                 rad=rad, dv=dv)
    # print(res.fun, res.message, f'tol={tol}')
    j = 0

    if true_convergence and res.success:
        j = 1

        res2 = minimize_energy_sphere(lattice=l, d=d, k=k, method=method, tol=tol / 10 ** j, option=opt, x0=res.x,
                                      A=A, jac_func=jac_func, rad=rad, dv=dv)
        # print(res2.fun, res2.message, f'tol={tol / 10 ** j}')
        while abs(1 - res2.fun / res.fun) > .01:

            if not res2.success:
                # print('Minimization failed, try to increase k')
                print(res2.message)
                break
            j += 1
            res = res2
            res2 = minimize_energy_sphere(lattice=l, d=d, k=k, method=method, tol=tol / 10 ** j, option=opt, x0=res.x,
                                          A=A, jac_func=jac_func, rad=rad, dv=dv)
            # print(res2.fun, res2.message, f'tol={tol / 10 ** j}')
        res = res2

    print(res.fun, res.message, f'tol={tol / 10 ** j}')

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0], plot=plot)

    return res


if __name__ == '__main__':
    pass
    # In here you can run this module
    # Used mainly for testing
