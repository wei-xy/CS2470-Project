import numpy as np

# global parameters -------------------------------------------------------------------------------- #
num_graphs = 1000                       # number of samples to be generated (per geometry)
N = 50                                  # number of nodes (per graph)
Radius = 1                              # radius to set the point density
pointDensity = N / (np.pi * Radius**2)  # density of points (to be kept fixed across geometries)
thresholdFrac = 0.4  # used to compute the fraction of the total area within which to connect nodes
# -------------------------------------------------------------------------------------------------- #
""" 
=========================================================================================
NOTE ON POINT DENSITY:
----------------------
In each case (spherical, planar, hyperbolic) the appropriate radius is chosen to keep the 
ratio of the number of nodes (N) to the sampled area fixed. We want the model to learn the 
difference between different geometries, not just artifacts related to the point density, 
such as the average degree. We fix the point density by choosing a radius (declared as 'Radius') 
of a circle in the Euclidean plane. 

Each radius below comes from the condition that (N / Area) = pointDensity: 

sphericalRadius = sqrt(N / 4 * pi * pointDensity)             <-- Area = 4 pi radius^2
planarRadius = sqrt(N / pi * pointDensity)                    <-- Area = pi radius^2
hyperbolicRadius = arccosh(1 + N / (2 * pi * pointDensity))   <-- Area = 2 pi (cosh radius - 1)

=========================================================================================
"""


""" spherical graphs """

# generates positions for each node
def sample_spherical(N, radius, ndim=3):
    vec = np.random.randn(ndim, N)
    vec = vec * radius / np.linalg.norm(vec, axis=0)
    return vec

def sphere_generator():

    """
    We compute sphericalThreshold by regarding thresholdFrac as the ratio of area around a node (where
    other nodes can connect to it) to the total area of the surface.
    In this case, the area around a node forms a circular sector, and we solve for the arclength "radius"
    of that circular sector using:
    thresholdFrac = Area_sector / Total_area = (1 - cos(s / sphericalRadius)) / 2 where s is the arclength
    "radius" of the circular sector

    :return: None
    """

    sphericalRadius = np.sqrt(N / (4 * np.pi * pointDensity))
    sphericalThreshold = sphericalRadius * np.arccos(1 - 2 * thresholdFrac)

    data_sphere = []
    # np.random.seed(2020)
    for r in range(num_graphs):
        coords = sample_spherical(N, sphericalRadius, 3)
        # computes the adjacency matrix
        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                a = coords[:, i]
                b = coords[:, j]
                dot_prod = np.dot(a, b)/sphericalRadius**2
                dot_prod = min(dot_prod, 1)  # <-- sometimes np.dot returns 1.00000000002, messing up np.arccos()

                """ note that when np.arrcos gets 1, it returns a nan """
                theta = np.arccos(dot_prod)  # gets the angle between a and b (in radians)

                # ij_dist = np.linalg.norm(a-b) # calculate euclidean distance
                ij_dist = sphericalRadius * theta  # arclength distance
                if ij_dist < sphericalThreshold:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_sphere.append(Adj_Matrix)

    return data_sphere


""" planar graphs """

def plane_generator():
    """
    We compute planarThreshold by regarding thresholdFrac as the ratio of area around a node (where
    other nodes can connect to it) to the total area of the surface.
    In this case, the area around a node forms a circle, and we solve for the radius of that circle using:
    thresholdFrac = Area_circle / Total_area = s ** 2 / planarRadius ** 2 where s is the radius of the
    surrounding circle

    :return: None
    """

    planarRadius = np.sqrt(N / (np.pi * pointDensity))  # <-- convert pointDensity into radius
    planarThreshold = planarRadius * np.sqrt(thresholdFrac)

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
            rnd_radii = np.random.random() * planarRadius
            node_pos.update({i: (rnd_radii, rnd_angle)})

        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ij_dist = dist(node_pos[i], node_pos[j])
                if ij_dist < planarThreshold:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_plane.append(Adj_Matrix)

    return data_plane


""" hyperbolic graphs """

def hyp_dist(rTheta1, rTheta2):
    """
    Takes in Hyperbolic polar (native) coordinates and returns the corresponding hyperbolic distance

    We compute hyperbolic distance using the "hyperbolic law of cosines" (see "Hyperbolic
    Geometry of Complex Networks" by Krioukov et al)

    :param rTheta1: tuple, (radius1, theta1) (note that the radius is a Hyperbolic distance)
    :param rTheta2: tuple, (radius2, theta2)
    :return: hyperbolic distance between two points in H^2, the hyperbolic plane with curvature -1
    """
    # Euclidean polar coordinates:
    a, b = rTheta1[0], rTheta2[0]
    theta1, theta2 = rTheta1[1], rTheta2[1]

    # hyperbolic distance according to "Hyperbolic Geometry of Complex Networks" by Krioukov et al
    cosh1, cosh2 = np.cosh(a), np.cosh(b)
    sinh1, sinh2 = np.sinh(a), np.sinh(b)
    input = cosh1 * cosh2 - sinh1 * sinh2 * np.cos(theta1 - theta2)
    input = max(1, input)  # sometimes input = 0.99999.. and this messes up np.arccosh()
    h_dist = np.arccosh(input)  # hyperbolic law of cosines

    return h_dist


def hyperbolic_generator():
    """
    Note that we use 'inversion sampling' below, taken from "Gradient Descent in Hyperbolic Space" by
    B. Wilson & M. Leimeister (arxiv: 1805.08207)

    Inversion sampling will sample radii in a circle specified by 'Radius' above with a density consistent
    with a uniformly sampling of points in the infinite Hyperbolic plane H^2

    We generate a dict. of the form: node_pos = {node_i: (radius, theta)}, where radius is a Euclidean
    distance in a circle with radius specified by the parameter named 'Radius' above

    We compute hyperbolicThreshold by regarding thresholdFrac as the ratio of area around a node (where
    other nodes can connect to it) to the total area of the surface.
    In this case, the area around a node forms a hyperbolic circle, and we solve for the radius of that
    circle using:
    thresholdFrac = Area_sector / Total_area = (cosh(s) - 1) / (cosh(hyperbolicRadius) - 1)

    :return: list of length num_graphs of adjacency matrices, each matrix has size N x N
    """

    hyperbolicRadius = np.arccosh(1 + N / (2 * np.pi * pointDensity))
    hyperbolicThreshold = np.arccosh(1 + thresholdFrac * (np.cosh(hyperbolicRadius) - 1))

    data_hyperbolic = []
    for r in range(num_graphs):
        # generates dictionary of positions (in a circle of radius) for each node: node_pos = {node_i: (radius, theta)}  <-- uses polar coordinates
        # uses the inversion sampling idea to give Euclidean radii sampled uniformly across a hyperbolic sheet
        node_pos = {}
        for i in range(N):
            rnd_angle = np.random.random() * 2 * np.pi
            p = np.random.random()  # random float between 0 and 1
            rnd_radii = np.arccosh(1 + p * (np.cosh(hyperbolicRadius) - 1))  # <-- inversion sampling
            node_pos.update({i: (rnd_radii, rnd_angle)})

        # computes the adjacency matrix
        Adj_Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                ij_dist = hyp_dist(node_pos[i], node_pos[j])
                if ij_dist < hyperbolicThreshold:
                    Adj_Matrix[i, j] = 1  # nodes that are connected are assigned a 1 in the matrix

        data_hyperbolic.append(Adj_Matrix)

    return data_hyperbolic


""" generates all graphs """

def generate_all_graphs():

    sphericalRadius = np.sqrt(N / (4 * np.pi * pointDensity))
    planarRadius = np.sqrt(N / (np.pi * pointDensity))
    hyperbolicRadius = np.arccosh(1 + N / (2 * np.pi * pointDensity))

    sphericalThreshold = sphericalRadius * np.arccos(1 - 2 * thresholdFrac)
    planarThreshold = planarRadius * np.sqrt(thresholdFrac)
    hyperbolicThreshold = np.arccosh(1 + thresholdFrac * (np.cosh(hyperbolicRadius) - 1))

    sphereArea = lambda s: 2 * np.pi * (1 - np.cos(s / sphericalRadius)) * sphericalRadius ** 2
    planeArea = lambda s: np.pi * s ** 2
    hypArea = lambda s: 2 * np.pi * (np.cosh(s) - 1)

    # print('spherical:', sphereArea(sphericalThreshold) / sphereArea(np.pi * sphericalRadius))
    # print('planar:', planeArea(planarThreshold) / planeArea(planarRadius))
    # print('hyperbolic:', hypArea(hyperbolicThreshold) / hypArea(hyperbolicRadius))

    sphere_graphs = sphere_generator()
    planar_graphs = plane_generator()
    hyperbolic_graphs = hyperbolic_generator()

    # define the sphere labels  (will use sphere=0, plane=1, hyperbolic=2, for one-hot encoding)
    sphere_labels = [0] * num_graphs
    planar_labels = [1] * num_graphs
    hyperbolic_labels = [2] * num_graphs

    all_graphs = sphere_graphs + planar_graphs + hyperbolic_graphs
    all_labels = sphere_labels + planar_labels + hyperbolic_labels

    return (all_graphs, all_labels)




