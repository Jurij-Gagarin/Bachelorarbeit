import math
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
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
    # 2dim = d
    # 2d + 1 = number of nodes on the x-axis
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


def manipulate_lattice_random(lattice, n=1):
    # changes z-component from n random nodes
    for k in range(0, n):
        i = rn.randint(0, len(lattice) - 1)
        j = rn.randint(0, 10)/1
        lattice[i] = lattice[i].change_coordinates([0, 0, j])
        lattice[i] = lattice[i].change_mobility(False)

    return lattice


def manipulate_lattice(lattice, d, dim, point, stretch_factor=5):
    # stretch with factor = 1: 5% of half lattice length
    # stretch with factor = 5 (default): 25% of half lattice length
    j = stretch_factor*d*dim/400
    lattice[point] = lattice[point].change_coordinates([0, 0, j])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def manipulate_lattice_absolute_value(lattice, point, displace_value):
    lattice[point] = lattice[point].change_coordinates([0, 0, displace_value])
    lattice[point] = lattice[point].change_mobility(False)

    return lattice


def plot_lattice(lattice):
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


def adjacency_matrix(lattice, plot=False, show_graph=False):
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
    A = np.triu(A)
    As = np.triu(As)
    if plot:
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(A)
        plt.show()

    if show_graph:
        rows, cols = np.where(A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=50)
        plt.show()

    return A, As


def list_of_coordinates(lattice):
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


# TODO: optimize this function
def energy_func(x, xdict, A, xs, xsdict, As, lattice, d=1, k=2):
    mrows, mcols = np.where(A == 1)
    imrows, imcols = np.where(As == 1)
    e = d / math.sqrt(3)
    total_energy = 0

    for i in range(0, len(mrows)):
        rpos = xdict[mrows[i]]
        cpos = xdict[mcols[i]]
        total_energy += (math.sqrt((x[rpos]-x[cpos])**2+(x[rpos+1]-x[cpos+1])**2+(x[rpos+2]-x[cpos+2])**2) - e)**2

    for i in range(0, len(imrows)):
        if (lattice[imrows[i]].return_mobility() is True) and (lattice[imcols[i]].return_mobility() is False):
            rpos = xdict[imrows[i]]
            cpos = xsdict[imcols[i]]
            total_energy += (math.sqrt((x[rpos]-xs[cpos])**2+(x[rpos+1]-xs[cpos+1])**2+(x[rpos+2]-xs[cpos+2])**2)-e)**2
        elif (lattice[imrows[i]].return_mobility() is False) and (lattice[imcols[i]].return_mobility() is True):
            rpos = xsdict[imrows[i]]
            cpos = xdict[imcols[i]]
            total_energy += (math.sqrt((xs[rpos]-x[cpos])**2+(xs[rpos+1]-x[cpos+1])**2+(xs[rpos+2]-x[cpos+2])**2)-e)**2
        else:
            rpos = xsdict[imrows[i]]
            cpos = xsdict[imcols[i]]
            total_energy += (math.sqrt((xs[rpos]-xs[cpos])**2+(xs[rpos+1]-xs[cpos+1])**2+(xs[rpos+2]-xs[cpos+2])**2)-e)**2

    return 0.5*k*total_energy


def minimize_energy(lattice, d=1, k=2):
    func = energy_func
    r = list_of_coordinates(lattice)
    x0 = r[1]
    A = adjacency_matrix(lattice)

    return opt.minimize(func, x0, args=(r[3], A[0], r[0], r[2], A[1], lattice, d, k))


def assemble_result(result, fixed_values):
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

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlim([-20, 20])
    # ax.set_ylim([-20, 20])
    ax.set_xlabel('x-Achse')
    ax.set_ylabel('y-Achse')
    ax.scatter(x, y, z, c='blue')
    ax.scatter(xf, yf, zf, c='red')
    plt.show()


def run(dim, d=1, k=2, stretch_factor=5, plot=True):
    ls = create_lattice(dim, d)
    l = ls[0]
    l = manipulate_lattice(l, d, dim, ls[1], stretch_factor)
    res = minimize_energy(l, d, k)

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0])

    return res


def run_absolute_displacement(dim, displace_value, d=1, k=2, plot=True):
    ls = create_lattice(dim, d)
    l = ls[0]
    l = manipulate_lattice_absolute_value(l, ls[1], displace_value)
    res = minimize_energy(l, d, k)

    if plot:
        assemble_result(res.x, list_of_coordinates(l)[0])

    return res


if __name__ == '__main__':
    run_absolute_displacement(5, 1)
