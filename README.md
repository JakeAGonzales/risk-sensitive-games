# risk-sensitive-games

Welfare experiment for *Risk-Sensitive Games: Monotonicity, Learning, and
Welfare Effects.*  We simulate a textbook 4-firm Cournot competition with
random demand intercept and compare two risk classes (entropic and
CVaR -- two of the paper's three running examples) on three quantities:
equilibrium movement, total welfare
(consumer + producer surplus), and the SMR conditioning constants
`lambda_R`, `mu_R`.

`tau` denotes the risk-aversion intensity (replacing the manuscript's
`s`).  `tau -> 0` is the risk-neutral limit.

## Layout

- `games.py` -- `Cournot` class with closed-form equilibria, total welfare
  (CS + PS), and a welfare-correct externality operator that makes the
  decomposition identity hold.
- `risks.py` -- `EntropicRisk` and `CVaR` with both the generic
  envelope-identity interface and closed-form Cournot+Gaussian
  pseudogradients, plus the calibrated path scaling
  `cvar_beta_for_tau(game, tau)` (numerical inversion of the CVaR
  coefficient ``kappa(beta) = phi(Phi^{-1}(beta)) / (1 - beta)``).
- `solver.py` -- projected-pseudogradient `solve_risk_eq`,
  `welfare_decomposition`, and finite-difference utilities for `lambda_R`.
- `experiment_welfare.py` -- the experiment script: runs all preflight
  closed-form checks, sweeps `tau`, and produces Figures A--E plus
  `results.json` and `summary.md` in `results/exp_welfare/`.

## Reproduction

```
pip install -r requirements.txt
python experiment_welfare.py
```

Runs end-to-end in well under a minute on a laptop.  All NumPy randomness
is seeded; the seed is recorded in `results/exp_welfare/results.json`.

## Pre-flight checks (asserted at every run)

- `q_star = 1.6` and `q_social_opt = 2.0` for the chosen parameters
  `(N=4, b=1, a_mean=10, a_std=1, c=2)`.
- `W(q_social_opt) > W(q_star)` -- free-riding gap exists.
- `solve_risk_eq(EntropicRisk(0)).q == q_star` to `1e-7`.
- Solver matches the closed-form risk-sensitive equilibrium for both
  risks at `tau in {0.1, 0.2, 0.3}` to `1e-6`.
- Welfare decomposition identity `Delta W = I_F + I_E` holds at every
  swept `tau` to `1e-10`.

The script aborts with a clear assertion error if any check fails; no
figures are produced from an unvalidated solver.
