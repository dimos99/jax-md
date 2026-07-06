"""Deterministic and stochastic real-space RPY utilities."""

from jax_md.hydro.rpy_real_det import (
    REAL_DTYPE,
    F1F2_closed_form,
    Mr_pair_block,
    Mr_self,
    RealSpaceState,
    current_box_matrix,
    build_Mr_apply,
    mr_matvec,
)
from jax_md.hydro.rpy_real_det_dipole_helpers import (
    G1G2_closed_form,
    K1K2K3_closed_form,
    Mr_self_dipole,
)
from jax_md.hydro.rpy_real_det_dipole import (
    PreparedGrandReal,
    build_Mr_grand_apply,
    mr_grand_apply_blocks,
    mr_grand_matvec,
    mr_grand_prepare,
)
from jax_md.hydro.rpy_real_stoch import (
    Preconditioner,
    identity_preconditioner,
    scalar_preconditioner,
    diagonal_preconditioner,
    jacobi_from_self,
    lanczos_sqrt_mv,
    sample_mr_sqrt_precond,
    sample_mr_sqrt,
)

__all__ = [
    # Deterministic utilities
    'REAL_DTYPE',
    'F1F2_closed_form',
    'Mr_pair_block',
    'Mr_self',
    'G1G2_closed_form',
    'K1K2K3_closed_form',
    'Mr_self_dipole',
    'RealSpaceState',
    'current_box_matrix',
    'build_Mr_apply',
    'mr_matvec',
    'build_Mr_grand_apply',
    'mr_grand_matvec',
    'PreparedGrandReal',
    'mr_grand_prepare',
    'mr_grand_apply_blocks',
    # Stochastic utilities
    'Preconditioner',
    'identity_preconditioner',
    'scalar_preconditioner',
    'diagonal_preconditioner',
    'jacobi_from_self',
    'lanczos_sqrt_mv',
    'sample_mr_sqrt_precond',
    'sample_mr_sqrt',
]
