import numpy as np
from ..utils import find_borders

import numpy as np



def normalized_zero_mean(im):
    # Zero mean normalized standard deviation
    return (im-np.average(im))/np.std(im)


def find_borders(coord):
    return int(np.min(np.floor(coord))), int(np.max(np.ceil(coord)))





def find_elm_borders_mesh(mesh, n_elms):
    # [Xmin_Xmax,Ymin,Ymax,elm_nr]
    borders = np.zeros((4, n_elms))

    for el in range(n_elms):
        borders[:2, el] = find_borders(mesh.xnodes[mesh.ele[:, el]])
        borders[2:, el] = find_borders(mesh.ynodes[mesh.ele[:, el]])

    return borders







def elm_coords_from_global_q4(corners, global_x, global_y):
    xi1 = lambda ai, bi, ci: 2. * ci / (-bi - (bi ** 2. - 4. * ai * ci) ** 0.5)

    a1 = np.array(
        corners[2] * corners[4] - corners[0] * corners[6] + corners[0] * corners[7] + corners[1] * corners[6] - corners[2] * corners[5] - corners[3] * corners[4] - corners[1] * corners[7] + corners[3] * corners[5])
    b1 = np.array(
        corners[0] * corners[6] - corners[2] * corners[4] - 2. * corners[0] * corners[7] + 2. * corners[3] * corners[4] + corners[1] * corners[7] - corners[3] * corners[5] + corners[0] * global_y - corners[
            1] * global_y + corners[2] * global_y - corners[3] * global_y - corners[4] * global_x + corners[5] * global_x - corners[6] * global_x + corners[7] * global_x)
    c1 = np.array(corners[0] * corners[7] - corners[3] * corners[4] - corners[0] * global_y + corners[3] * global_y + corners[4] * global_x - corners[7] * global_x)

    a2 = np.array(
        corners[1] * corners[4] - corners[0] * corners[5] + corners[0] * corners[6] - corners[2] * corners[4] - corners[1] * corners[7] + corners[3] * corners[5] + corners[2] * corners[7] - corners[3] * corners[6])
    b2 = np.array(
        2. * corners[0] * corners[5] - 2. * corners[1] * corners[4] - corners[0] * corners[6] + corners[2] * corners[4] + corners[1] * corners[7] - corners[3] * corners[5] - corners[0] * global_y + corners[
            1] * global_y - corners[2] * global_y + corners[3] * global_y + corners[4] * global_x - corners[5] * global_x + corners[6] * global_x - corners[7] * global_x)
    c2 = np.array(-corners[0] * corners[5] + corners[1] * corners[4] + corners[0] * global_y - corners[1] * global_y - corners[4] * global_x + corners[5] * global_x)

    if a1 == 0:
        nn1 = -c1 / b1
    else:
        nn1 = xi1(a1, b1, c1)

    if a2 == 0:
        nn2 = -c2 / b2
    else:
        nn2 = xi1(a2, b2, c2)

    return nn1, nn2


def find_element_coordinates_q4(xnod, ynod, elm):
    # Find element borders
    xmin, xmax = find_borders(xnod)
    ymin, ymax = find_borders(ynod)

    # Make grid inside mesh borders
    Xx, Yy = np.meshgrid(range(xmin, xmax + 1), range(ymin, ymax + 1))
    X = Xx.flatten()
    Y = Yy.flatten()

    # Find element coordinates by inverting the shape functions. Works with Q4 only
    ep, ny = elm_coords_from_global_q4(np.array([xnod[elm.corner_nodes], ynod[elm.corner_nodes]]).flatten(), X, Y)

    return [ep, ny, X, Y]


def generate_reference_Q4(mesh, im, elm, norm=False):

    nNodes, nEl = np.shape(mesh.ele)

    # Declare empty lists
    I0grad = [[[], []] for _ in range(nEl)]

    epE = [[] for _ in range(nEl)]
    nyE = [[] for _ in range(nEl)]
    Nref = [[] for _ in range(nEl)]

    Xe = [[] for _ in range(nEl)]
    Ye = [[] for _ in range(nEl)]
    I0 = [[] for _ in range(nEl)]



    for el in range(nEl):
        epE[el], nyE[el], Xe[el], Ye[el] = find_element_coordinates_q4(mesh.xnodes[mesh.ele[:, el]], mesh.ynodes[mesh.ele[:, el]], elm)

        Nref[el] = elm.Nn(epE[el], nyE[el])

        # Normalization:
        if norm:
            im_norm = normalized_zero_mean(im)
        else:
            im_norm = im

        # Image within element borders
        I0[el] = im_norm[Ye[el], Xe[el]]

        # Image gradient within element borders
        I0grad[el][0] = (im_norm[Ye[el], Xe[el] + 1] - im_norm[Ye[el], Xe[el] - 1]) / 2.
        I0grad[el][1] = (im_norm[Ye[el] + 1, Xe[el]] - im_norm[Ye[el] - 1, Xe[el]]) / 2.


    # Declare empty matrices
    K = np.zeros((np.size(mesh.xnodes) * 2, np.size(mesh.xnodes) * 2), dtype=np.float64)
    B = [np.zeros((2 * nNodes, np.shape(Xe[i])[0]), dtype=np.float64) for i in range(nEl)]




    for el in range(nEl):

        for i in range(elm.n_nodes):
            B[el][i, :] = np.array([I0grad[el][0] * Nref[el][:, i]])
            B[el][i + elm.n_nodes, :] = np.array([I0grad[el][1] * Nref[el][:, i]])
        # Calculate B^T * B
        A = np.dot(B[el], B[el].transpose())
        # Assemble K matrix
        K[np.ix_((mesh.ele[:, el] + 1) * 2 - 2, (mesh.ele[:, el] + 1) * 2 - 2)] += A[:elm.n_nodes, :elm.n_nodes]  # OK
        K[np.ix_((mesh.ele[:, el] + 1) * 2 - 2, (mesh.ele[:, el] + 1) * 2 - 1)] += A[:elm.n_nodes, elm.n_nodes:]
        K[np.ix_((mesh.ele[:, el] + 1) * 2 - 1, (mesh.ele[:, el] + 1) * 2 - 2)] += A[elm.n_nodes:, :elm.n_nodes]
        K[np.ix_((mesh.ele[:, el] + 1) * 2 - 1, (mesh.ele[:, el] + 1) * 2 - 1)] += A[elm.n_nodes:, elm.n_nodes:]


    K = np.linalg.inv(K)


    return Nref, I0, K, B
