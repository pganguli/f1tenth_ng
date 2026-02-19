"""
Vehicle parameter definitions for the F1TENTH Gym environment.
"""

from typing import NamedTuple


class VehicleParams(NamedTuple):
    """
    Physical parameters of a vehicle.
    """

    mu: float
    """Surface friction coefficient"""

    C_Sf: float
    """Cornering stiffness coefficient, front"""

    C_Sr: float
    """Cornering stiffness coefficient, rear"""

    lf: float
    """Distance from center of gravity to front axle"""

    lr: float
    """Distance from center of gravity to rear axle"""

    h: float
    """Height of center of gravity"""

    m: float
    """Total mass of the vehicle"""

    MoI: float
    """Moment of inertia of the entire vehicle about the z axis"""

    s_min: float
    """Minimum steering angle constraint"""

    s_max: float
    """Maximum steering angle constraint"""

    sv_min: float
    """Minimum steering velocity constraint"""

    sv_max: float
    """Maximum steering velocity constraint"""

    v_switch: float
    """Switching velocity (velocity at which the acceleration is no longer able to
    create wheel spin)"""

    a_max: float
    """Maximum longitudinal acceleration"""

    v_min: float
    """Minimum longitudinal velocity"""

    v_max: float
    """Maximum longitudinal velocity"""
