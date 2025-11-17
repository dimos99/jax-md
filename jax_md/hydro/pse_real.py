"""Compatibility wrapper exposing deterministic and stochastic real-space utilities."""

from jax_md.hydro.pse_real_det import (
    REAL_DTYPE,
    F1F2_closed_form,
    Mr_pair_block,
    Mr_self,
    RealSpaceState,
    _current_box_matrix,
    build_Mr_apply,
    mr_matvec,
)
from jax_md.hydro.pse_real_stoch import (
    Preconditioner,
    identity_preconditioner,
    scalar_preconditioner,
    diagonal_preconditioner,
    jacobi_from_self,
    lanczos_sqrt_mv,
    lanczos_sqrt_mv_test,
    sample_mr_sqrt_precond,
    sample_mr_sqrt,
)

__all__ = [
    # Deterministic utilities
    'REAL_DTYPE',
    'F1F2_closed_form',
    'Mr_pair_block',
    'Mr_self',
    'RealSpaceState',
    '_current_box_matrix',
    'build_Mr_apply',
    'mr_matvec',
    # Stochastic utilities
    'Preconditioner',
    'identity_preconditioner',
    'scalar_preconditioner',
    'diagonal_preconditioner',
    'jacobi_from_self',
    'lanczos_sqrt_mv',
    'lanczos_sqrt_mv_test',
    'sample_mr_sqrt_precond',
    'sample_mr_sqrt',
]
