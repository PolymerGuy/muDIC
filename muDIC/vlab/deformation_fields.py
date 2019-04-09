import numpy as np


def harmonic_x(xs, _, amp=1.1, omega=2.0 * np.pi, img_shape=(500, 500)):
    """
    Displacement field being sinusoidal along the x-axis


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    omega : float
        The angular frequency
    img_shape : tuple
        The shape of the field


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    xs = xs.astype(np.float)

    xs_scaled = xs / np.float(img_shape[0])

    xs_mapped = amp * np.sin(omega * xs_scaled)
    return xs_mapped, np.zeros_like(xs_mapped)


def harmonic_bilat(xs, ys, amp=1.1, omega=np.pi * 2., img_shape=(200, 200)):
    """
    Displacement field being sinusoidal along the both axes


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    omega : float
        The angular frequency
    img_shape : tuple
        The shape of the field


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    xs_scaled = xs / np.float(img_shape[0])
    ys_scaled = ys / np.float(img_shape[1])

    xs_mapped = amp * np.sin(omega * xs_scaled) * np.sin(omega * ys_scaled)
    ys_mapped = amp * np.sin(omega * xs_scaled) * np.sin(omega * ys_scaled)
    return xs_mapped, ys_mapped
