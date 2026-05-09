"""Standard Cournot competition with random demand intercept.

N symmetric firms produce q_i >= 0.  Inverse demand P(Q, w) = a(w) - b Q with
Q = sum_i q_i and a(w) ~ N(a_mean, a_std^2).  Constant marginal cost c.

Loss (= negative profit):
    ell_i(q, w) = -(a(w) - b Q - c) q_i.

Total welfare follows the textbook Marshallian convention:
    W(q) = E[ CS(Q) + PS(q, w) ]
         = E[ (1/2) b Q^2 + (a(w) - b Q - c) Q ]
         = a_mean Q - c Q - (1/2) b Q^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


def gauss_hermite_normal(n_atoms: int, mean: float, std: float):
    """(atoms, probs) approximating N(mean, std^2) by Gauss-Hermite quadrature.

    Uses the probabilist's HermiteE family so that the underlying weight
    function is exp(-x^2 / 2) (matching N(0, 1)).
    """
    x, w = hermegauss(n_atoms)
    probs = w / w.sum()
    atoms = mean + std * x
    return atoms, probs


@dataclass
class Cournot:
    """Symmetric N-firm Cournot game with random demand intercept."""

    N: int = 4
    b: float = 1.0
    a_mean: float = 10.0
    a_std: float = 1.0
    c: float = 2.0

    # ---- losses & gradients ------------------------------------------------
    def loss(self, q: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """Per-player loss matrix.  ``q`` shape (N,), ``omegas`` shape (K,).

        Returns shape (N, K): ell_{i, k} = -(omega_k - b Q - c) q_i.
        """
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        margin = omegas - self.b * Q - self.c          # (K,)
        return -np.outer(q, margin)                    # (N, K)

    def grad_loss_wrt_qi(self, q: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """d ell_i / d q_i evaluated at each (i, k).  Shape (N, K)."""
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        margin = omegas - self.b * Q - self.c          # (K,)
        return -margin[None, :] + self.b * q[:, None]

    def expected_grad(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        return -(self.a_mean - self.b * Q - self.c) + self.b * q

    def expected_jacobian(self, q: np.ndarray | None = None) -> np.ndarray:
        # F_i = -a_mean + b Q + c + b q_i  =>  d F_i / d q_j = b (1 + delta_ij)
        return self.b * (np.ones((self.N, self.N)) + np.eye(self.N))

    # ---- welfare components ------------------------------------------------
    def consumer_surplus(self, q: np.ndarray) -> float:
        Q = float(np.sum(q))
        return 0.5 * self.b * Q * Q

    def producer_surplus(self, q: np.ndarray) -> float:
        # E_omega[(a - bQ - c) Q] = (a_mean - b Q - c) Q.
        Q = float(np.sum(q))
        return (self.a_mean - self.b * Q - self.c) * Q

    def welfare(self, q: np.ndarray) -> float:
        return self.consumer_surplus(q) + self.producer_surplus(q)

    def welfare_grad(self, q: np.ndarray) -> np.ndarray:
        """dW/dq_i = a_mean - b Q - c  (constant across i)."""
        Q = float(np.sum(q))
        return np.full(self.N, self.a_mean - self.b * Q - self.c)

    # ---- externality operators --------------------------------------------
    def externality_firm(self, q: np.ndarray) -> np.ndarray:
        """Firm-firm externality E_i = -sum_{j != i} d E[ell_j] / d q_i.

        E[ell_j] = -(a_mean - bQ - c) q_j  =>  d/dq_i = b q_j (j != i).
        E_i = -b sum_{j != i} q_j = -b (Q - q_i).
        """
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        return -self.b * (Q - q)

    def externality_welfare(self, q: np.ndarray) -> np.ndarray:
        """Welfare-correct externality E^W_i := dW/dq_i + F_i(q).

        For TOTAL welfare W = CS + PS, the textbook firm-firm externality
        does not exhaust the cross effects (consumer surplus contributes a
        bQ term).  E^W is the operator that makes the line-integral identity
            Delta W = -int <F, dq> + int <E^W, dq>
        hold exactly along the path q* -> q^rho.

        For Cournot:  E^W_i = b q_i.
        """
        q = np.asarray(q, dtype=float)
        return self.b * q

    # ---- equilibria & moduli ---------------------------------------------
    def risk_neutral_equilibrium(self) -> np.ndarray:
        return np.full(self.N, (self.a_mean - self.c) / ((self.N + 1) * self.b))

    def social_optimum(self) -> np.ndarray:
        return np.full(self.N, (self.a_mean - self.c) / (self.N * self.b))

    def equilibrium_price(self) -> float:
        return (self.a_mean + self.N * self.c) / (self.N + 1)

    def mu_base(self) -> float:
        """Smallest eigenvalue of the symmetrised base Jacobian.

        sym(J) = b (I + 11^T).  Eigenvalues are {b, ..., b, b(N+1)} so the
        smallest is b.
        """
        return float(self.b)

    # ---- noise discretisation --------------------------------------------
    def quadrature(self, n_atoms: int = 1000):
        atoms, probs = gauss_hermite_normal(n_atoms, self.a_mean, self.a_std)
        return atoms, probs
