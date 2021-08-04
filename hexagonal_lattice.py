# In this module we will create our lattice, as well as some basic functions.

import math
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from scipy import optimize as opt


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
        self.x = n1*d/2 - n2*d/2
        self.y = (n1 + n2)*math.sqrt(3)*d/2 - b*d/math.sqrt(3)
        self.z = z
        self.name = [n1, n2, b]

    def change_coordinates(self, r):
        self.x = self.x + r[0]
        self.y = self.y + r[1]
        self.z = self.z + r[2]

        return self

    def new_coordinates(self, r):
        self.x = r[0]
        self.x = r[1]
        self.x = r[2]

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


def create_lattice(dim, d=1):
    # d is distance between two nodes on the x-axis
    # dim is the maximal summed factor for the lattice vectors
    lattice = []
    box_length = d*dim
    next = False

    for k in range(0, 2):
        for i in range(-dim, dim + 1):
            for j in range(-dim, dim + 1):
                if (abs(i) + abs(j)) <= dim:
                    lattice.append(Node(i, j, d, k))
                    x = lattice[-1].return_coordinates()[0]
                    y = lattice[-1].return_coordinates()[1]
                    if (abs(i) + abs(j)) == dim:
                        lattice[-1] = lattice[-1].change_mobility(False)
                    if next:
                        lattice[-1] = lattice[-1].change_mobility(False)
                        next = False
                    if (abs(x) > box_length/2) or (abs(y) > box_length/2):
                        lattice.pop(-1)
                        next = True
                        if len(lattice) > 0:
                            lattice[-1] = lattice[-1].change_mobility(False)
                    if (j == 0) and (i == 0) and (k == 0):
                        mid_point = len(lattice)-1

    return lattice, mid_point


def create_lattice_sphere(dim, r2, d=1):
    # d is distance between two nodes on the x-axis
    # dim is the maximal summed factor for the lattice vectors
    lattice = []
    box_length = d*dim
    next = False
    r = math.sqrt(r2)

    for k in range(0, 2):
        for i in range(-dim, dim + 1):
            for j in range(-dim, dim + 1):
                if (abs(i) + abs(j)) <= dim:
                    lattice.append(Node(i, j, d, k))
                    x = lattice[-1].return_coordinates()[0]
                    y = lattice[-1].return_coordinates()[1]
                    if (abs(i) + abs(j)) == dim:
                        lattice[-1] = lattice[-1].change_mobility(False)
                    if next:
                        lattice[-1] = lattice[-1].change_mobility(False)
                        next = False
                    if (abs(x) > box_length/2) or (abs(y) > box_length/2):
                        lattice.pop(-1)
                        next = True
                        if len(lattice) > 0:
                            lattice[-1] = lattice[-1].change_mobility(False)
                    elif x**2 + y**2 < r2:
                        lattice[-1] = lattice[-1].change_coordinates([0, 0, -math.sqrt(r2-x**2-y**2)])

    return [lattice, None]


def manipulate_lattice_random(lattice, n=1):
    # changes z-component from n random nodes
    for k in range(0, n):
        i = rn.randint(0, len(lattice) - 1)
        j = rn.randint(0, 10)/1
        lattice[i] = lattice[i].change_coordinates([0, 0, j])
        lattice[i] = lattice[i].change_mobility(False)

    return lattice


def manipulate_lattice(lattice, d, dim, point, stretch_factor=5):
    # stretch with factor = 1: 1% of lattice length
    j = stretch_factor*d*dim/100
    lattice[point] = lattice[point].change_coordinates([0, 0, j])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def manipulate_lattice_absolute_value(lattice, point, displace_value):
    # Displaces middle point by an absolute value
    lattice[point] = lattice[point].change_coordinates([0, 0, displace_value])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def plot_lattice(lattice):
    # Rather old function. Might be removed in the future.
    x = []
    y = []
    z = []
    xb = []
    yb = []
    zb = []

    for i in range(0, len(lattice)):
        node = lattice[i]
        if node.return_mobility():
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
    plt.show()


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
                                   (nz[1] == ns[1] and nz[0] == ns[0]+1) or
                                   (nz[1] == ns[1]+1 and nz[0] == ns[0])):
                    if mobilityi is True and mobilityj is True:
                        A[i][j] = 1
                    else:
                        As[i][j] = 1

                if ns[2] == 1 and((nz[1] == ns[1] and nz[0] == ns[0]) or
                                  (nz[1] == ns[1] and nz[0] == ns[0]-1) or
                                  (nz[1] == ns[1]-1 and nz[0] == ns[0])):
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
            mobile_dict[i] = len(list_of_mobile_coords)-3

        else:
            list_of_immobile_coords = np.append(list_of_immobile_coords, r.return_coordinates())
            immobile_dict[i] = len(list_of_immobile_coords) - 3

    return list_of_immobile_coords, list_of_mobile_coords, immobile_dict, mobile_dict


def energy_func_prep(A, As, d):
    # This does some basic preparation for the optimized energy function
    mrows, mcols = np.where(A == 1)
    imrows, imcols = np.where(As == 1)
    e = d / math.sqrt(3)
    return mrows, mcols, imrows, imcols, e


def energy_func_opt(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, d=1, k=2):
    # calculates lattice energy (optimized).
    total_energy = 0

    for i in range(len(mrows)):
        rpos = xdict[mrows[i]]
        cpos = xdict[mcols[i]]
        total_energy += (math.sqrt((x[rpos]-x[cpos])**2+(x[rpos+1]-x[cpos+1])**2+(x[rpos+2]-x[cpos+2])**2) - e)**2

    for i in range(len(imrows)):
        if (lattice[imrows[i]].return_mobility() is True) and (lattice[imcols[i]].return_mobility() is False):
            rpos = xdict[imrows[i]]
            cpos = xsdict[imcols[i]]
            total_energy += (math.sqrt((x[rpos]-xs[cpos])**2+(x[rpos+1]-xs[cpos+1])**2+(x[rpos+2]-xs[cpos+2])**2)-e)**2
        elif (lattice[imrows[i]].return_mobility() is False) and (lattice[imcols[i]].return_mobility() is True):
            rpos = xsdict[imrows[i]]
            cpos = xdict[imcols[i]]
            total_energy += (math.sqrt((xs[rpos]-x[cpos])**2+(xs[rpos+1]-x[cpos+1])**2+(xs[rpos+2]-x[cpos+2])**2)-e)**2

    return .5 * k * total_energy


def energy_func_jac(x, xdict, xs, xsdict, mrows, mcols, imrows, imcols, lattice, e, A, d=1, k=2):
    # Calculates jacobian i.e. the gradient for the energy function. Might be optimized in the future.
    len_x = len(x)
    grad = np.zeros(len_x)

    # Key is the number of x-coord in x. Gives corresponding particle in adjacency matrix
    xdict_reverse = {v: k for k, v in xdict.items()}
    for i in range(int(len_x/3)):
        index = np.where(A[xdict_reverse[3*i]] == 1)[0]
        for j in range(len(index)):
            # cn = corresponding_node
            if lattice[index[j]].return_mobility():
                cn = xdict[index[j]]
                root = ((x[3*i]-x[cn])**2+(x[3*i+1]-x[cn+1])**2+(x[3*i+2]-x[cn+2])**2)**.5
                factor = k*(root-e)/root**.5
                grad[3*i] += factor*(x[3*i]-x[cn])
                grad[3*i+1] += factor*(x[3*i+1]-x[cn+1])
                grad[3*i+2] += factor*(x[3*i+2]-x[cn+2])
            else:
                cn = xsdict[index[j]]
                root = ((x[3*i]-xs[cn])**2+(x[3*i+1]-xs[cn+1])**2+(x[3*i+2]-xs[cn+2])**2)**.5
                factor = k*(root-e)/root**.5
                grad[3*i] += factor*(x[3*i]-xs[cn])
                grad[3*i+1] += factor*(x[3*i+1]-xs[cn+1])
                grad[3*i+2] += factor*(x[3*i+2]-xs[cn+2])

    return grad


def constraint_sphere(x, r2):
    constraint = []
    for i in range(0, len(x)-2):
        if i % 3 == 0:
            constraint.append({'type': 'ineq', 'fun': lambda x: x[i]**2 + x[i+1]**2 + x[i+2]**2 - r2})

    return constraint


def minimize_energy_opt(lattice, d=1, k=2, method='CG', gtol=1.e-10):
    # Structures the act of energy minimization.
    r = list_of_coordinates(lattice)
    A = adjacency_matrix(lattice)
    preps = energy_func_prep(np.triu(A[0]), np.triu(A[1]), d)

    return opt.minimize(energy_func_opt, r[1], method=method, jac=energy_func_jac, args=(r[3], r[0], r[2], preps[0], preps[1], preps[2], preps[3],
                                                     lattice, preps[4], np.add(A[0], A[1]), d, k), tol=0, options={'gtol':gtol})


def minimize_energy_sphere(lattice, r2, d=1, k=2):
    r = list_of_coordinates(lattice)
    A = adjacency_matrix(lattice)
    preps = energy_func_prep(np.triu(A[0]), np.triu(A[1]), d)
    constraint = constraint_sphere(r[1], r2)

    return opt.minimize(energy_func_opt, r[1], method='SLSQP', jac=energy_func_jac, constraints=constraint,
                        args=(r[3], r[0], r[2], preps[0], preps[1], preps[2], preps[3],
                              lattice, preps[4], np.add(A[0], A[1]), d, k))


def assemble_result(result, fixed_values, plot=True):
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

    # Very basic plot. Might be removed in the future.
    if plot:
        print('h')
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


def run(dim, d=1, k=2, stretch_factor=5, plot=True):
    ls = create_lattice(dim, d)
    l = ls[0]
    l = manipulate_lattice(l, d, dim, ls[1], stretch_factor)
    res = minimize_energy_opt(l, d, k)

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0])

    return res


def run_absolute_displacement(dim, displace_value, d=1, k=2, plot=True, method='CG', gtol=1.e-10):
    ls = create_lattice(dim, d)
    l = ls[0]
    l = manipulate_lattice_absolute_value(l, ls[1], displace_value)
    res = minimize_energy_opt(l, d, k, method=method, gtol=gtol)

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0])

    return res


def run_sphere(dim, r2, d=1, k=2, plot=True):
    l = create_lattice_sphere(dim, r2)[0]
    res = minimize_energy_sphere(l, r2, d, k)

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0])

    return res


if __name__ == '__main__':
    # In here you can run this module
    dim = 10
    displace_value = .5

    '''
    The following will perform a simple lattice minimization. You can create a simple plot with setting 
    plot to True. If you are not interest in the entire minimization message, you can print res.x for 
    the coordinates and res.fun for the minimal energy.
    '''
    # res = run_absolute_displacement(dim, displace_value, plot=False)
    # print(res)

    res = []
    for dim in [5, 10, 15, 20]:
        for i in range(5, 11):
            res.append(run_absolute_displacement(dim, displace_value, plot=False, gtol=10**(-i)))
            print(f'dim={dim}, gtol: e-{i}:', res[-1].fun, res[-1].success)







