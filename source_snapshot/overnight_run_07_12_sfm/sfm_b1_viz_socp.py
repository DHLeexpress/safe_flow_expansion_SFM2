"""Compatibility name for the canonical runtime verifier.

All functions are aliases, not a second implementation.  This guarantees that
the certificate drawn in paper diagnostics is the certificate used to label
the B1 expansion query.
"""
from sfm_metrics2 import (  # noqa: F401
    ANGLE_TOL,
    ARTIFICIAL_FACES,
    _angular_constraint,
    _is_feasible,
    _wrap,
    certify_moving_window,
    solve_static_face,
    solve_moving_face,
    verify_in_worker,
    verify_query,
)
