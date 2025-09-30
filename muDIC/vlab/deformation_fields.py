import numpy as np


def harmonic_x(xs, _, amp=1.1, omega=0.05 * np.pi, frame=1):
    """
    Displacement field being sinusoidal along the x-axis


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    omega : float
        The angular frequency in radians/pixel


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    xs = xs.astype(float)

    xs_mapped = amp * np.sin(omega * xs) * float(frame)
    return xs_mapped, np.zeros_like(xs_mapped)


def harmonic_y(_, ys, amp=1.1, omega=0.05 * np.pi, frame=1):
    """
    Displacement field being sinusoidal along the y-axis


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    omega : float
        The angular frequency in radians/pixel


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    ys = ys.astype(float)

    ys_mapped = amp * np.sin(omega * ys) * float(frame)
    return np.zeros_like(ys_mapped), ys_mapped


def linear_x(xs, _, slope=0.001, frame=1):
    """
    Displacement field being linearly increasig along x with zero in the center


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    slope : float
        Displacement increment per pixel along X


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    xs = xs.astype(float)
    center = (xs.max() - xs.min()) / 2.

    xs_mapped = float(frame) * slope * xs - float(frame) * slope * center
    return xs_mapped, np.zeros_like(xs_mapped)


def linear_y(_, ys, slope=0.001, frame=1):
    """
    Displacement field being linearly increasig along y with zero in the center


    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    slope : float
        Displacement increment per pixel along X


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    ys = ys.astype(float)
    center = (ys.max() - ys.min()) / 2.

    xs_mapped = float(frame) * slope * ys - float(frame) * slope * center
    return xs_mapped, np.zeros_like(xs_mapped)


def harmonic_bilat(xs, ys, amp=1.1, omega=0.05 * np.pi, frame=1):
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

    xs_mapped = amp * np.sin(omega * xs) * np.sin(omega * ys) * float(frame)
    ys_mapped = amp * np.sin(omega * xs) * np.sin(omega * ys) * float(frame)
    return xs_mapped, ys_mapped
