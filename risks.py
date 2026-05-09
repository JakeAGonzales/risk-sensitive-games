"""Two risk measures used in the welfare experiment.

Both are specialised to losses that are linear in a Gaussian shock; closed
forms are available for both the certainty equivalent and the risk-sensitive
pseudogradient under the Cournot game.

* ``EntropicRisk(tau)``  -- generic entropic risk (paper Example 2.1).
* ``CVaR(beta)``         -- conditional value at risk (paper Example 2.2).

Each class exposes
    * ``intensity()``                          -> canonical scalar (tau, beta)
    * ``evaluate(losses, probs)``              -> Risk[Z]
    * ``worst_case_measure(losses, probs)``    -> q^* probabilities (closed form)
    * ``risk_grad(loss_grads, losses, probs)`` -> envelope identity gradient
    * ``cournot_pseudograd(game, q)``          -> closed-form risk pseudogradient
                                                  for the Cournot game
    * ``cournot_closed_form_eq(game)``         -> closed-form risk-sensitive eq
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ----------------------------------------------------------------------
class EntropicRisk:
    """Risk_tau[Z] = (1/tau) log E[exp(tau Z)]; tau -> 0 is risk-neutral.

    For Z linear in a Gaussian shock,
        Z = alpha * a + beta,  a ~ N(a_mean, sigma^2)
        Risk_tau[Z] = beta + alpha a_mean + (tau/2) alpha^2 sigma^2.
    """

    name = "entropic"

    def __init__(self, tau: float):
        self.tau = float(tau)

    def intensity(self) -> float:
        return self.tau

    # --- generic envelope-identity interface -------------------------------
    def evaluate(self, losses, probs):
        losses = np.asarray(losses, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if self.tau <= 1e-14:
            return np.sum(probs * losses, axis=-1)
        x = self.tau * losses
        m = np.max(x, axis=-1, keepdims=True)
        return (m.squeeze(-1) + np.log(np.sum(probs * np.exp(x - m), axis=-1))) / self.tau

    def worst_case_measure(self, losses, probs):
        losses = np.asarray(losses, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if self.tau <= 1e-14:
            shape = np.broadcast_shapes(losses.shape, probs.shape)
            return np.broadcast_to(probs, shape).copy()
        x = self.tau * losses
        m = np.max(x, axis=-1, keepdims=True)
        w = probs * np.exp(x - m)
        return w / w.sum(axis=-1, keepdims=True)

    def risk_grad(self, loss_grads, losses, probs):
        qs = self.worst_case_measure(losses, probs)
        return np.sum(qs * loss_grads, axis=-1)

    # --- specialised closed forms for Cournot+Gaussian ---------------------
    def cournot_pseudograd(self, game, q):
        """F^Risk_i = -(a_mean - b Q - c) + b q_i + tau sigma^2 q_i."""
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        base = -(game.a_mean - game.b * Q - game.c) + game.b * q
        return base + self.tau * (game.a_std ** 2) * q

    def cournot_closed_form_eq(self, game):
        denom = (game.N + 1) * game.b + self.tau * (game.a_std ** 2)
        return np.full(game.N, (game.a_mean - game.c) / denom)


# ----------------------------------------------------------------------
def cvar_kappa(beta: float) -> float:
    """CVaR coefficient kappa(beta) = phi(Phi^{-1}(beta)) / (1 - beta).

    Appears in the closed-form CVaR of a Gaussian-linear loss
        Risk_beta[ell] = E[ell] + kappa(beta) * sigma * |q_i|.

    By symmetry of the standard normal, phi(Phi^{-1}(beta)) ==
    phi(Phi^{-1}(1 - beta)), so the equivalent form
    phi(Phi^{-1}(1 - beta)) / (1 - beta) gives the same value.
    """
    if beta <= 0.0:
        return 0.0
    if beta >= 1.0:
        return float("inf")
    z = norm.ppf(beta)
    return float(norm.pdf(z) / (1.0 - beta))


class CVaR:
    """Conditional value at risk at level beta in (0, 1):
        CVaR_beta[Z] = E[Z | Z >= VaR_beta(Z)],
        worst-case measure  dq*/dP = (1/(1-beta)) 1{Z >= VaR_beta(Z)}.

    For Z = alpha * a + b with a ~ N(a_mean, sigma^2),
        CVaR_beta[Z] = alpha a_mean + b + |alpha| * sigma * kappa(beta),
    where kappa(beta) = phi(Phi^{-1}(beta)) / (1 - beta).  In particular for
    the Cournot loss ell_i = -(a - bQ - c) q_i (alpha = -q_i, b = (bQ+c)q_i),
        Risk_beta[ell_i] = -(a_mean - bQ - c) q_i + kappa(beta) * sigma * |q_i|.
    """

    name = "cvar"

    def __init__(self, beta: float):
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"CVaR beta must lie in [0, 1); got {beta}")
        self.beta = float(beta)

    def intensity(self) -> float:
        return self.beta

    # --- generic worst-case measure on discrete atoms ----------------------
    def _wcm_one(self, losses, probs):
        if self.beta <= 1e-14:
            return probs.copy()
        # Sort by descending loss; assign mass 1/(1-beta) to the heaviest
        # losses until cumulative source probability reaches (1 - beta);
        # the boundary atom takes the residual mass.
        order = np.argsort(-losses)
        sorted_p = probs[order]
        cum = np.cumsum(sorted_p)
        tail = 1.0 - self.beta
        q_sorted = np.zeros_like(probs)
        # full atoms strictly inside the tail
        full = cum <= tail + 1e-15
        q_sorted[full] = sorted_p[full] / tail
        # boundary atom (first index where cum > tail)
        if not full.all():
            j = int(np.argmax(~full))
            prev = cum[j - 1] if j > 0 else 0.0
            q_sorted[j] = (tail - prev) / tail
        out = np.zeros_like(probs)
        out[order] = q_sorted
        # numerical guard
        s = out.sum()
        return out / s if s > 0 else probs.copy()

    def worst_case_measure(self, losses, probs):
        losses = np.asarray(losses, dtype=float)
        probs = np.asarray(probs, dtype=float)
        if losses.ndim == 1:
            return self._wcm_one(losses, probs)
        return np.stack([self._wcm_one(losses[i], probs) for i in range(losses.shape[0])])

    def evaluate(self, losses, probs):
        qs = self.worst_case_measure(losses, probs)
        return np.sum(qs * losses, axis=-1)

    def risk_grad(self, loss_grads, losses, probs):
        qs = self.worst_case_measure(losses, probs)
        return np.sum(qs * loss_grads, axis=-1)

    # --- specialised closed forms for Cournot+Gaussian ---------------------
    def cournot_pseudograd(self, game, q):
        """F^Risk_i = -(a_mean - b Q - c) + b q_i + kappa(beta) * sigma * sign(q_i)."""
        q = np.asarray(q, dtype=float)
        Q = float(q.sum())
        base = -(game.a_mean - game.b * Q - game.c) + game.b * q
        return base + cvar_kappa(self.beta) * game.a_std * np.sign(np.maximum(q, 0.0) + 1e-30)

    def cournot_closed_form_eq(self, game):
        num = game.a_mean - game.c - cvar_kappa(self.beta) * game.a_std
        return np.full(game.N, num / ((game.N + 1) * game.b))


# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def _cvar_beta_for_kappa(target_kappa: float) -> float:
    """Invert kappa(beta) = target via brentq.  Cached."""
    if target_kappa <= 0.0:
        return 0.0
    f = lambda b: cvar_kappa(b) - target_kappa
    # kappa(beta) is continuous and increasing on (0, 1) with range (0, infty).
    return float(brentq(f, 1e-12, 1.0 - 1e-12, xtol=1e-14))


def cvar_beta_for_tau(game, tau: float) -> float:
    """Path scaling: choose beta(tau) so that
        ||G_Risk^CVaR(q*; beta(tau))||_2 == ||G_Risk^ent(q*; tau)||_2.

    Working at q* (symmetric, q_i = q*):
        ||G^ent(q*; tau)|| = tau * sigma^2 * q* * sqrt(N)
        ||G^CVaR(q*; beta)|| = kappa(beta) * sigma * sqrt(N)
    matching gives kappa(beta(tau)) = tau * sigma * q*.
    """
    if tau <= 0.0:
        return 0.0
    q_star = float(game.risk_neutral_equilibrium()[0])
    target = tau * game.a_std * q_star
    return _cvar_beta_for_kappa(target)
