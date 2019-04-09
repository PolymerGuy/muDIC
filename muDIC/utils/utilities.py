from functools import reduce
import scipy.ndimage as nd
import numpy as np


def convert_to_img_frame(img, node_position, mesh, borders, settings):
    local_node_pos = np.zeros((2, mesh.element_def.n_nodes), dtype=settings.precision)

    # Partition image
    image_frame = extract_subframe(img, borders, settings.pad)

    # Determine nodal positions in image frame coordinates
    local_node_pos[0, :] = node_position[0] + settings.pad - borders[0, :]
    local_node_pos[1, :] = node_position[1] + settings.pad - borders[2, :]

    return image_frame, local_node_pos


def generate_edge_coordinates(seed):
    seeding = np.linspace(0., 1., seed)
    es, ns = np.meshgrid(seeding, seeding)
    mask = np.ones_like(es, dtype=np.bool)
    mask[1:-1, 1:-1] = 0
    return es[mask], ns[mask]


def find_element_borders(node_position, mesh, seed=20):
    e, n = generate_edge_coordinates(seed)
    N_at_borders = mesh.element_def.Nn(e.flatten(), n.flatten())

    # Find global coordinates of elements
    pixel_x = np.einsum("jk,k->j", N_at_borders, node_position[0])
    pixel_y = np.einsum("jk,k->j", N_at_borders, node_position[1])

    axis = None
    # [Xmin_Xmax,Ymin,Ymax,elm_nr]
    borders = np.zeros((4, mesh.n_elms), dtype=np.int)
    borders[0, :] = np.min(pixel_x, axis=axis)
    borders[1, :] = np.max(pixel_x, axis=axis)
    borders[2, :] = np.min(pixel_y, axis=axis)
    borders[3, :] = np.max(pixel_y, axis=axis)

    return borders


def extract_subframe(img, borders, pad):
    return img[borders[2, 0] - pad:borders[3, 0] + pad, borders[0, 0] - pad:borders[1, 0] + pad]


def find_borders(coord):
    return int(np.min(np.floor(coord))), int(np.max(np.ceil(coord)))


def find_inconsistent(ep, ny):
    rem1 = np.where(ep > 1.)
    rem2 = np.where(ep < 0.)
    rem3 = np.where(ny > 1.)
    rem4 = np.where(ny < 0.)
    return reduce(np.union1d, [rem1[0], rem2[0], rem3[0], rem4[0]])

def extract_points_from_image(image, coordinates):
    return nd.map_coordinates(image, coordinates, order=3, prefilter=True)

def image_coordinates(image):
    xs, ys = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    return xs,ys