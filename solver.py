"""Equilibrium solver and welfare-decomposition utilities.

The solver uses the closed-form risk-sensitive pseudogradient available for
the Gaussian-linear Cournot losses (via ``risk.cournot_pseudograd``).  This
keeps the inner risk computation deterministic and exact.  A backup
envelope-identity path is also provided for cross-checking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from games import Cournot


# ----------------------------------------------------------------------
@dataclass
class SolverResult:
    q: np.ndarray
    iters: int
    residual: float
    converged: bool


def _envelope_pseudograd(game: Cournot, risk, q: np.ndarray, omegas, probs):
    losses = game.loss(q, omegas)                  # (N, K)
    grads = game.grad_loss_wrt_qi(q, omegas)       # (N, K)
    qs = risk.worst_case_measure(losses, probs)
    return np.sum(qs * grads, axis=-1)


def solve_risk_eq(
    game: Cournot,
    risk,
    n_omega: int = 128,
    tol: float = 1e-9,
    max_iter: int = 5000,
    use_closed_form: bool = True,
) -> SolverResult:
    """Projected pseudogradient method for the risk-sensitive Cournot eq.

    By default uses the analytic closed-form risk pseudogradient available
    for both ``EntropicRisk`` and ``Chi2DRO`` on Gaussian-linear losses.
    Set ``use_closed_form=False`` to instead use the envelope-identity
    estimator over Gauss-Hermite quadrature; results should agree to within
    quadrature error.
    """
    omegas, probs = game.quadrature(n_omega)
    J = game.expected_jacobian()
    op_norm = float(np.linalg.norm(J, ord=2))
    eta = 0.5 / op_norm

    q = game.risk_neutral_equilibrium().copy()

    for t in range(max_iter):
        if use_closed_form:
            F = risk.cournot_pseudograd(game, q)
        else:
            F = _envelope_pseudograd(game, risk, q, omegas, probs)
        q_new = np.maximum(q - eta * F, 0.0)
        diff = float(np.max(np.abs(q_new - q)))
        q = q_new
        if diff < tol:
            return SolverResult(q=q, iters=t + 1, residual=diff, converged=True)
    return SolverResult(q=q, iters=max_iter, residual=diff, converged=False)


# ----------------------------------------------------------------------
def welfare_decomposition(game: Cournot, q_star: np.ndarray, q_rho: np.ndarray,
                          n_quad: int = 16):
    """Compute (Delta W, I_F, I_E) along the segment q* -> q_rho.

    With Delta W = W(q_rho) - W(q*), F = expected_grad and E = welfare-
    correct externality (E^W_i = b q_i for Cournot), the identity
        Delta W = -int <F, dq> + int <E^W, dq>
    holds because the welfare gradient satisfies dW/dq_i = -F_i + E^W_i.
    """
    nodes, weights = leggauss(n_quad)
    t_nodes = 0.5 * (nodes + 1.0)
    t_weights = 0.5 * weights

    dq = q_rho - q_star
    I_F = 0.0
    I_E = 0.0
    for t, w in zip(t_nodes, t_weights):
        q_t = q_star + t * dq
        F = game.expected_grad(q_t)
        E = game.externality_welfare(q_t)
        I_F += -w * float(np.dot(F, dq))
        I_E += w * float(np.dot(E, dq))
    Delta_W = game.welfare(q_rho) - game.welfare(q_star)
    return Delta_W, I_F, I_E


# ----------------------------------------------------------------------
def G_Risk(game: Cournot, risk, q: np.ndarray) -> np.ndarray:
    """Perturbation operator G(q) = F^Risk(q) - F(q) (closed form)."""
    return risk.cournot_pseudograd(game, q) - game.expected_grad(q)


def perturbation_jacobian_finite_diff(game: Cournot, risk, q: np.ndarray,
                                      h: float = 1e-4) -> np.ndarray:
    N = game.N
    J = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = h
        J[:, j] = (G_Risk(game, risk, q + e) - G_Risk(game, risk, q - e)) / (2.0 * h)
    return J


def lambda_R_eigenvalue(game: Cournot, risk, q: np.ndarray) -> float:
    """lambda_R(q) = -(1/2) lambda_min( sym(J_GR(q)) )."""
    J = perturbation_jacobian_finite_diff(game, risk, q)
    sym = 0.5 * (J + J.T)
    return -0.5 * float(np.linalg.eigvalsh(sym).min())
