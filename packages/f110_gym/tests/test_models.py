"""
Unit tests for F1TENTH vehicle dynamics models.
"""

from f110_gym.envs.dynamic_models import accl_constraints
from f110_gym.envs.vehicle_params import VehicleParams


def test_accl_constraints():
    """Test that acceleration constraints return expected values."""
    params = VehicleParams(
        mu=1.0,
        C_Sf=4.7,
        C_Sr=5.4,
        lf=0.15,
        lr=0.17,
        h=0.07,
        m=3.7,
        MoI=0.04,
        s_min=-0.4,
        s_max=0.4,
        sv_min=-3.2,
        sv_max=3.2,
        v_switch=7.0,
        a_max=9.5,
        v_min=-5.0,
        v_max=20.0,
    )

    # Below switch velocity
    accl = accl_constraints(5.0, 10.0, params)
    assert accl == 9.5

    # Above max velocity
    accl = accl_constraints(21.0, 1.0, params)
    assert accl == 0.0

    # Above switch velocity
    accl = accl_constraints(14.0, 10.0, params)
    assert accl < 9.5
