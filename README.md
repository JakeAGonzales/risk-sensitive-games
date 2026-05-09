# risk-sensitive-games

Welfare experiment for *Risk-Sensitive Games: Monotonicity, Learning, and
Welfare Effects.*  Standard 4-firm Cournot competition with random demand,
comparing entropic risk and CVaR.

## Run

```
pip install -r requirements.txt
python experiment_welfare.py
```

Outputs go to `results/exp_welfare/` (figures, `results.json`, `summary.md`).

## Files

- `games.py` -- `Cournot` game.
- `risks.py` -- `EntropicRisk`, `CVaR`, and the calibrated path scaling.
- `solver.py` -- equilibrium solver and welfare decomposition.
- `experiment_welfare.py` -- the experiment script.
