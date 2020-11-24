import numpy as np

# global parameters ------------------------------------------------------------ #
num_graphs = 1000     # number of samples to be generated (per geometry)
N = 100               # number of nodes (per graph)
radius = 1            # radius of the various circles in each geometry
area_sphere = 4 * np.pi * radius**2  # area of a sphere to generate point_density
point_density = N/area_sphere  # density of points
threshold_dist = 0.4  # nodes within this distance will be connected
# ------------------------------------------------------------------------------ #


""" spherical graphs """

# generates positions for each node
def sample_spherical(N, radius, ndim=3):
    vec = np.random.randn(ndim, N)
    vec = vec * radius / np.linalg.norm(vec, axis=0)
    return vec

def sphere_generator():
    data_sphere = []
    # np.random.seed(2020)
    for r in range(num_graphs):
        coords = sample_spherical(N, radius, 3)
        # computes the adjacency matrix
        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                a = coords[:, i]
                b = coords[:, j]
                theta = np.arccos(np.dot(a, b)/radius**2)  # gets the angle between a and b (in radians)
                # ij_dist = np.linalg.norm(a-b) # calculate euclidean distance
                ij_dist = radius * theta  # arclength distance
                if ij_dist < threshold_dist:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_sphere.append(Adj_Matrix)

    return data_sphere


""" planar graphs """

def plane_generator(pt_density):
    radius = (N / (np.pi * pt_density)) ** 0.5  # <-- convert pt_density into radius

    # distance function (law of cosines)
    def dist(rTheta1, rTheta2):  # rThetai is a coordinate tuple: (r, theta)
        a, b = rTheta1[0], rTheta2[0]
        theta1, theta2 = rTheta1[1], rTheta2[1]
        return np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(theta1 - theta2))  # <-- law of cosines

    # computes the adjacency matrices
    data_plane = []
    for r in range(num_graphs):

        # generates dictionary of positions for each node: node_pos = {node_i: (radius, theta)}
        node_pos = {}
        for i in range(N):
            rnd_angle = np.random.random() * 2 * np.pi
            rnd_radii = np.random.random() * radius
            node_pos.update({i: (rnd_radii, rnd_angle)})

        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ij_dist = dist(node_pos[i], node_pos[j])
                if ij_dist < threshold_dist:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_plane.append(Adj_Matrix)

    return data_plane


""" hyperbolic graphs """

# hyperbolic distance function (Wikipedia: "Poincare Disk Model/Distance")
# can also look at https://bjlkeng.github.io/posts/hyperbolic-geometry-and-poincare-embeddings/ for more on hyperbolic distance
def hyp_dist(rTheta1, rTheta2):  # rThetai is a coordinate tuple: (r, theta)  r is the Euclidean distance!
    a, b = rTheta1[0], rTheta2[0]
    theta1, theta2 = rTheta1[1], rTheta2[1]
    ab_dist = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(theta1 - theta2))  # Euclidean distance
    return np.arccosh(1 + 2 * (ab_dist * radius) ** 2 / ((radius ** 2 - a ** 2) * (radius ** 2 - b ** 2)))


def hyperbolic_generator():
    data_hyperbolic = []
    for r in range(num_graphs):
        # generates dictionary of positions (in a circle of radius) for each node: node_pos = {node_i: (radius, theta)}  <-- uses polar coordinates
        # uses the inversion sampling idea to give Euclidean radii supposedly sampled uniformly across a hyperbolic sheet
        node_pos = {}
        for i in range(N):
            rnd_angle = np.random.random() * 2 * np.pi
            p = np.random.random()  # random float between 0 and 1
            rnd_radii = np.arccosh(1 + p * (np.cosh(radius) - 1))
            node_pos.update({i: (rnd_radii, rnd_angle)})

        # computes the adjacency matrix
        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ij_dist = hyp_dist(node_pos[i], node_pos[j])
                if ij_dist < threshold_dist:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_hyperbolic.append(Adj_Matrix)

    return data_hyperbolic


""" generates all graphs """

def generate_all_graphs():
    sphere_graphs = sphere_generator()
    planar_graphs = plane_generator(point_density)
    hyperbolic_graphs = hyperbolic_generator()

    # define the sphere labels  (will use sphere=0, plane=1, hyperbolic=2, for one-hot encoding)
    sphere_labels = [0] * num_graphs
    planar_labels = [1] * num_graphs
    hyperbolic_labels = [2] * num_graphs

    all_graphs = sphere_graphs + planar_graphs + hyperbolic_graphs
    all_labels = sphere_labels + planar_labels + hyperbolic_labels

    return (all_graphs, all_labels)




