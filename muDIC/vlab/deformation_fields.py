import numpy as np


def harmonic_x(xs, _, amp=1.1, omega=0.05 * np.pi, Cx=0, Cy=0, frame=1):
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
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    xs = xs.astype(float)

    xs_mapped = amp * np.sin(omega * (xs-Cx)) * float(frame)
    return xs_mapped, np.zeros_like(xs_mapped)


def harmonic_y(_, ys, amp=1.1, omega=0.05 * np.pi, Cx=0, Cy=0, frame=1):
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
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation    


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    ys = ys.astype(float)

    ys_mapped = amp * np.sin(omega * (ys-Cy)) * float(frame)
    return np.zeros_like(ys_mapped), ys_mapped


def linear_x(xs, _, slope=0.001, Cx=np.nan, Cy=np.nan, frame=1):
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
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation
    

    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    xs = xs.astype(float)
    if np.isnan(Cx):
        center = (xs.max() - xs.min()) / 2.
    else:
        center = Cx

    xs_mapped = float(frame) * slope * (xs - center)
    return xs_mapped, np.zeros_like(xs_mapped)


def linear_y(_, ys, slope=0.001, Cx=np.nan, Cy=np.nan, frame=1):
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
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation    

    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    # Center around x and y
    ys = ys.astype(float)
    if np.isnan(Cy):
        center = (ys.max() - ys.min()) / 2.
    else:
        center = Cy

    ys_mapped = float(frame) * slope * (ys - center)
    return np.zeros_like(ys_mapped), ys_mapped


def harmonic_bilat(xs, ys, amp=1.1, omega=0.05 * np.pi, Cx=0, Cy=0, frame=1):
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
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation
    

    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    xs_mapped = amp * np.sin(omega * (xs-Cx)) * np.sin(omega * (ys-Cy)) * \
        float(frame)
    ys_mapped = amp * np.sin(omega * (xs-Cx)) * np.sin(omega * (ys-Cy)) * \
        float(frame)
    return xs_mapped, ys_mapped


def rigid_rotation(xs, ys, theta=np.pi/50, Cx=0, Cy=0, frame=1):
    """
    Displacement field from rigid rotation about the out of plane axis.

    Parameters
    ----------
    xs : array
        The x-coodinates
    ys : array
        The y-coordinates
    theta : float
        (default np.pi/50) Rotation angle
    Cx : float
        X Location of the center for the deformation
    Cy : float
        Y Location of the center for the deformation


    Returns
    -------
    u_x,u_y : array
        The displacement field values in each direction

    """

    xs_mapped = np.cos(theta * float(frame))*(xs-Cx) + \
        np.sin(theta * float(frame))*(ys-Cy) - (xs-Cx)
    ys_mapped = -np.sin(theta * float(frame))*(xs-Cx) + \
        np.cos(theta * float(frame))*(ys-Cy) - (ys-Cy)

    # Warning: inversion starts zooming out for large rotations.
    # xs_mapped = -(ys-Cy)*(theta*float(frame))
    # ys_mapped = (xs-Cx)*(theta*float(frame))
    return xs_mapped, ys_mapped
