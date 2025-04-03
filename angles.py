import numpy as np


def _compass_angle_degrees(angle):
    a = angle - 90
    neg = 1 - np.minimum(np.maximum(np.ceil(a), 0), 1).astype(int)
    return a + (360 * neg)


def calc_angles(p1, p2, compass=True):
    """ Calculate angle of p2 from p1.

    p1, p2 : Point, or vector of Points
    compass : bool. Return value in compass degrees (i.e. range 0-360)

    Returns:
    Point or vector of Points.

    Note:
    Point: shapely.geometry.Point

    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    theta = np.arctan2(dy, dx)
    theta = np.degrees(theta)
    if compass:
        compass = _compass_angle_degrees(theta)
        return compass
    else:
        return theta
