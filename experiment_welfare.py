"""Welfare experiment: standard Cournot with random demand.

Game: 4-firm Cournot, linear demand with Gaussian intercept.  Welfare =
consumer surplus + producer surplus.  Two risks: EntropicRisk and CVaR,
calibrated so that ||G_Risk(q*; tau)|| matches between them at every tau.

Outputs: Figures A-E (PDF + PNG) and `results.json` in `results/exp_welfare/`.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from games import Cournot
from risks import CVaR, EntropicRisk, cvar_beta_for_tau, cvar_kappa
from solver import (
    G_Risk,
    lambda_R_eigenvalue,
    solve_risk_eq,
    welfare_decomposition,
)

SEED = 20251109
OUT_DIR = pathlib.Path("results/exp_welfare")
FIG_DIR = OUT_DIR / "figures"


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def validate(game: Cournot) -> None:
    """Closed-form / decomposition checks; aborts on failure."""
    q_star = game.risk_neutral_equilibrium()
    q_opt = game.social_optimum()
    assert np.allclose(q_star, 1.6, atol=1e-12)
    assert np.allclose(q_opt, 2.0, atol=1e-12)
    assert game.welfare(q_opt) > game.welfare(q_star)
    assert abs(game.mu_base() - game.b) < 1e-12

    # Solver matches risk-neutral and risk-sensitive closed forms.
    assert np.allclose(solve_risk_eq(game, EntropicRisk(0.0), tol=1e-12).q,
                       q_star, atol=1e-7)
    for tau in (0.1, 0.2, 0.3):
        for risk in (EntropicRisk(tau), CVaR(cvar_beta_for_tau(game, tau))):
            err = float(np.max(np.abs(
                solve_risk_eq(game, risk, tol=1e-12).q
                - risk.cournot_closed_form_eq(game))))
            assert err < 1e-6, (risk.name, tau, err)

    # Norm-matching at q* along the calibrated path.
    sqrtN = np.sqrt(game.N)
    for tau in np.linspace(0.0, 0.4, 11):
        beta = cvar_beta_for_tau(game, float(tau))
        ent = float(tau) * (game.a_std ** 2) * float(q_star[0]) * sqrtN
        cv = cvar_kappa(beta) * game.a_std * sqrtN
        assert abs(ent - cv) < 1e-8, (tau, ent, cv)
    assert cvar_beta_for_tau(game, 0.4) < 0.5

    # Welfare-decomposition identity.
    for tau in (0.05, 0.2, 0.4):
        q_rho = solve_risk_eq(game, EntropicRisk(tau), tol=1e-12).q
        DW, IF, IE = welfare_decomposition(game, q_star, q_rho)
        assert abs(IF + IE - DW) < 1e-10


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
def run_sweep(game: Cournot, taus: np.ndarray) -> dict:
    q_star = game.risk_neutral_equilibrium()
    keys = ("shift", "DW", "IF", "IE", "CS", "PS", "lambda", "G_norm")
    out: dict = {f"{k}_{r}": [] for k in keys for r in ("ent", "cvar")}
    out["taus"] = taus
    out["beta_path"] = []

    for tau in taus:
        risk_e = EntropicRisk(float(tau))
        beta = cvar_beta_for_tau(game, float(tau))
        risk_c = CVaR(beta)
        out["beta_path"].append(beta)

        for risk, suffix in ((risk_e, "ent"), (risk_c, "cvar")):
            q = solve_risk_eq(game, risk, tol=1e-12).q
            DW, IF, IE = welfare_decomposition(game, q_star, q)
            assert abs(IF + IE - DW) < 1e-10
            out[f"shift_{suffix}"].append(float(np.linalg.norm(q - q_star)))
            out[f"DW_{suffix}"].append(DW)
            out[f"IF_{suffix}"].append(IF)
            out[f"IE_{suffix}"].append(IE)
            out[f"CS_{suffix}"].append(game.consumer_surplus(q))
            out[f"PS_{suffix}"].append(game.producer_surplus(q))
            out[f"G_norm_{suffix}"].append(
                float(np.linalg.norm(G_Risk(game, risk, q_star))))
            out[f"lambda_{suffix}"].append(
                lambda_R_eigenvalue(game, risk, q) if tau > 1e-8 else 0.0)

    for k, v in list(out.items()):
        if k != "taus":
            out[k] = np.asarray(v, dtype=float)
    return out


# ----------------------------------------------------------------------
# Theory
# ----------------------------------------------------------------------
def theory_curve(game: Cournot, q_star: np.ndarray, shifts: np.ndarray,
                 v_R: np.ndarray) -> np.ndarray:
    """Delta W ~ <E, v_R> ||dq|| - (1/2) mu_base ||dq||^2."""
    inner = float(np.dot(game.externality_welfare(q_star), v_R))
    return inner * shifts - 0.5 * game.mu_base() * shifts ** 2


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def make_figures(game: Cournot, sweep: dict) -> dict:
    taus = sweep["taus"]
    q_star = game.risk_neutral_equilibrium()

    # Risk-direction unit vector from a small-tau entropic shift.
    q_dir = solve_risk_eq(game, EntropicRisk(0.05), tol=1e-12).q
    dq_dir = q_dir - q_star
    v_R = dq_dir / max(np.linalg.norm(dq_dir), 1e-30)

    # Slopes for Figure C reference lines.
    kappa_R = float(np.linalg.norm(G_Risk(game, EntropicRisk(1e-3), q_star)) / 1e-3)
    bound_slope = kappa_R / game.mu_base()
    tight_slope = kappa_R / ((game.N + 1) * game.b)

    DW_theory_ent = theory_curve(game, q_star, sweep["shift_ent"], v_R)

    # Figure A -- welfare gap vs tau.
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(taus, sweep["DW_ent"], "-", color="C0", lw=2, label="entropic")
    ax.plot(taus, sweep["DW_cvar"], "-", color="C1", lw=2, label="CVaR")
    ax.plot(taus, DW_theory_ent, "--", color="gray", lw=1.4, label="theory (entropic)")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel(r"risk intensity $\tau$")
    ax.set_ylabel(r"$\Delta\mathcal{W}(\tau)$")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "figA_welfare_gap")

    # Figure B -- decomposition (entropic + CVaR).
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, suffix, title in [(axs[0], "ent", "entropic"), (axs[1], "cvar", "CVaR")]:
        ax.plot(taus, sweep[f"IF_{suffix}"], "-", color="C0", lw=2,
                label=r"$I_F$ (game incentive)")
        ax.plot(taus, sweep[f"IE_{suffix}"], "-", color="C1", lw=2,
                label=r"$I_\mathcal{E}$ (externality)")
        ax.plot(taus, sweep[f"DW_{suffix}"], "--", color="black", lw=1.4,
                label=r"$\Delta\mathcal{W} = I_F + I_\mathcal{E}$")
        ax.axhline(0, color="grey", lw=0.4)
        ax.set_xlabel(r"$\tau$")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("welfare-gap component")
    fig.tight_layout()
    _save(fig, "figB_decomposition")

    # Figure C -- equilibrium movement.
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(taus, sweep["shift_ent"], "-", color="C0", lw=2, label="entropic")
    ax.plot(taus, sweep["shift_cvar"], "-", color="C1", lw=2, label="CVaR")
    ax.plot(taus, bound_slope * taus, "--", color="gray", lw=1.0,
            label=fr"bound $\kappa_R/\mu_{{\rm base}}={bound_slope:.2f}$")
    ax.plot(taus, tight_slope * taus, ":", color="black", lw=1.0,
            label=fr"tight slope ${tight_slope:.3f}$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\|\delta q(\tau)\|_2$")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "figC_eq_movement")

    # Figure D -- welfare gap vs movement (headline).
    s_grid = np.linspace(
        0.0, max(sweep["shift_ent"].max(), sweep["shift_cvar"].max()) * 1.1, 200)
    DW_curve = theory_curve(game, q_star, s_grid, v_R)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(s_grid, DW_curve, "--", color="gray", lw=1.4, label="theory (parabola)")
    sc = ax.scatter(sweep["shift_ent"], sweep["DW_ent"], c=taus, cmap="Blues",
                    marker="o", s=22, edgecolors="C0", linewidths=0.6, label="entropic")
    ax.scatter(sweep["shift_cvar"], sweep["DW_cvar"], c=taus, cmap="Oranges",
               marker="s", s=22, edgecolors="C1", linewidths=0.6, label="CVaR")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel(r"$\|\delta q(\tau)\|_2$")
    ax.set_ylabel(r"$\Delta\mathcal{W}(\tau)$")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04, label=r"$\tau$ (entropic)")
    fig.tight_layout()
    _save(fig, "figD_welfare_vs_movement")

    # Figure E -- conditioning.
    mu_b = game.mu_base()
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for ax, ydata_ent, ydata_cv, ylabel in [
        (axs[0], sweep["lambda_ent"], sweep["lambda_cvar"], r"$\lambda_R(\tau)$"),
        (axs[1], mu_b - 2.0 * sweep["lambda_ent"], mu_b - 2.0 * sweep["lambda_cvar"],
         r"$\mu_R(\tau)$"),
    ]:
        ax.plot(taus, ydata_ent, "-", color="C0", lw=2, label="entropic")
        ax.plot(taus, ydata_cv, "-", color="C1", lw=2, label="CVaR")
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axs[0].axhline(0, color="grey", lw=0.4)
    axs[1].axhline(mu_b, color="black", lw=0.6, ls="--",
                   label=fr"$\mu_{{\rm base}}={mu_b}$")
    for ax in axs:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "figE_conditioning")

    return {
        "kappa_R": kappa_R,
        "bound_slope": bound_slope,
        "tight_slope": tight_slope,
        "v_R": v_R.tolist(),
        "DW_theory_ent": DW_theory_ent.tolist(),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    game = Cournot(N=4, b=1.0, a_mean=10.0, a_std=1.0, c=2.0)

    validate(game)
    sweep = run_sweep(game, np.linspace(0.0, 0.4, 25))
    extras = make_figures(game, sweep)

    results = {
        "seed": SEED,
        "game": dict(N=game.N, b=game.b, a_mean=game.a_mean,
                     a_std=game.a_std, c=game.c),
        "q_star": game.risk_neutral_equilibrium().tolist(),
        "q_social_opt": game.social_optimum().tolist(),
        "mu_base": game.mu_base(),
        "extras": extras,
        "data": {k: v.tolist() for k, v in sweep.items()},
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))

    print(f"[exp_welfare] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
