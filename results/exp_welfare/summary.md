# Welfare experiment: Risk-sensitive Cournot competition

## Setup

We study standard Cournot competition with random demand. Four firms simultaneously choose quantities $q_i \geq 0$. The market price is $P(Q, \omega) = a(\omega) - bQ$ with $Q = \sum_i q_i$, where the demand intercept $a(\omega) \sim \mathcal{N}(10, 1)$. All firms have constant marginal cost $c = 2$ and slope $b = 1$.

Welfare is the textbook total surplus $\mathcal{W}(q) = \mathbb{E}[\text{CS}(Q) + \text{PS}(q, \omega)]$ — consumer surplus plus producer surplus. The risk-neutral Cournot equilibrium has each firm producing $q^\star = 1.6$, while the social optimum (perfect competition) has each producing $q^{**} = 2.0$ — a 25% free-riding gap from market-power-induced underproduction.

We compare two of the paper's three running risk classes: entropic risk (parameter $\tau$) and CVaR (parameter $\beta$). Both have closed-form worst-case measures under Gaussian noise; both are calibrated so that $\|G_\mathcal{R}(q^\star; \cdot)\|_2$ is identical at every $\tau$, isolating monotonicity effects from raw perturbation magnitude. We sweep $\tau \in [0, 0.4]$ on 25 grid points, with $\beta(\tau)$ chosen by inverting $\phi(\Phi^{-1}(\beta))/(1-\beta) = \tau \sigma q^\star$ via Brent's method (verified to $10^{-8}$).

For each $\tau$ we solve the risk-sensitive equilibrium $q^\rho(\tau)$ via projected gradient descent on the envelope-identity gradient $F^\mathcal{R}(q) = (\mathbb{E}_{q_i^\star(q)}[\nabla_{q_i}\ell_i])_i$ (closed-form worst-case measures), validated against analytical risk-sensitive Cournot equilibria. The base monotonicity modulus $\mu_{\text{base}} = b = 1$ comes from the smallest eigenvalue of the symmetrized base Jacobian; $\lambda_\mathcal{R}(\tau)$ is computed numerically by finite-differencing the perturbation Jacobian $\nabla G_\mathcal{R}$ and taking half the negative of the smallest eigenvalue of its symmetric part.

The alignment inner product $\langle \mathcal{E}(q^\star), v_\mathcal{R}\rangle = -3.20 < 0$, putting us in the **unfavorable-alignment regime** of the competition theorem (§4.3): risk-aversion shrinks output (consumers worse off, firms only marginally better off) and the externality realignment hurts welfare rather than helping.

## Figure A — welfare gap vs. risk intensity

Plot of $\Delta\mathcal{W}(\tau) = \mathcal{W}(q^\rho(\tau)) - \mathcal{W}(q^\star)$ for both risks, with the leading-order theoretical prediction $\Delta\mathcal{W}^{\text{theory}}(\tau) = \langle\mathcal{E}, v_\mathcal{R}\rangle\|\delta q\| - \tfrac{1}{2}\mu_{\text{base}}\|\delta q\|^2$.

Both risks degrade total welfare monotonically. CVaR loss is consistently slightly larger than entropic at every $\tau$ — because CVaR doesn't strengthen monotonicity, so the equilibrium isn't held as tightly, and more equilibrium movement in the unfavorable-alignment direction means more welfare loss. The leading-order theory tracks the entropic numerical curve closely across the sweep; deviations at larger $\tau$ are the higher-order corrections expected outside the small-$\tau$ regime where Theorem 4.3 holds.

## Figure B — welfare-gap decomposition (entropic)

Decomposes $\Delta\mathcal{W} = I_F + I_\mathcal{E}$ via path integrals along $q^\star \to q^\rho(\tau)$:

$$I_F(\tau) = -\int_0^1 \langle F(q^\star + t\delta q), \delta q\rangle\, dt, \qquad I_\mathcal{E}(\tau) = \int_0^1 \langle \mathcal{E}(q^\star + t\delta q), \delta q\rangle\, dt,$$

computed via 16-node Gauss-Legendre quadrature. The decomposition identity holds to $10^{-10}$ at every $\tau$.

**Both $I_F$ and $I_\mathcal{E}$ are negative for all $\tau > 0$** — the no-tradeoff regime. The game-incentive cost $I_F$ (rigidity penalty for departing from $q^\star$) is unambiguously negative under strong base-monotonicity, as Lemma 4.2 predicts. The externality term $I_\mathcal{E}$ is also negative because the risk-induced shift direction $v_\mathcal{R}$ is anti-aligned with the externality $\mathcal{E}(q^\star)$. Both terms pull welfare down — there's no externality-realignment benefit to offset the rigidity penalty. This is the regime of Corollary 4.2.4 ("welfare strictly decreases when game incentives dominate"), and explains *why* the welfare curves in Figure A are monotonically negative.

## Figure C — equilibrium movement vs. risk intensity

$\|\delta q(\tau)\| = \|q^\rho(\tau) - q^\star\|_2$ for both risks. Both curves overlap closely and grow linearly in $\tau$ with empirical slope $0.640$, well below the loose theoretical bound $\kappa_R/\mu_{\text{base}} = 3.20$ from Theorem 3.5. The actual slope $\kappa_R/((N+1)b) = 0.64$ reflects the symmetric structure of Cournot: the perturbation lies in the constant-on-symmetric-strategies subspace, not the worst-case singular direction of $J_F^{-1}$. This sanity-checks the calibration — both risks produce the same equilibrium movement at matched operator perturbation, confirming that any welfare difference comes from structural monotonicity properties rather than calibration artifacts.

## Figure D — welfare gap vs. equilibrium movement (headline)

Plots $\Delta\mathcal{W}(\tau)$ against $\|\delta q(\tau)\|$ instead of against $\tau$, overlaid with the theoretical parabola $\Delta\mathcal{W} = \langle\mathcal{E}, v_\mathcal{R}\rangle \|\delta q\| - \tfrac{1}{2}\mu_{\text{base}}\|\delta q\|^2$.

**Both risks lie on essentially the same parabolic curve.** The welfare-vs-movement relationship is determined by the base-game structure ($\mathcal{E}, \mu_{\text{base}}$) and is independent of which risk class produced the movement. The risk choice changes only *where on the curve* each $\tau$-value lands.

This is the empirical content of the strengthened claim: **monotonicity-strengthening risks don't change the welfare cost per unit equilibrium movement; they suppress how much movement you get per unit risk-intensity.** When the externality alignment is unfavorable (as here), suppressed movement means smaller welfare loss in absolute terms, but the underlying welfare-vs-movement tradeoff is the same. To overcome the quadratic rigidity penalty $\tfrac{1}{2}\mu_{\text{base}}\|\delta q\|^2$ via the linear externality term, you need movement aligned with $\mathcal{E}$ — which monotonicity-strengthening risks specifically prevent. Hence "if the game is monotone and risk improves monotonicity, it requires more to overcome the game term in the welfare expansion, and welfare more likely decreases."

## Figure E — risk-distortion eigenvalue and conditioning modulus

Two panels: $\lambda_\mathcal{R}(\tau)$ and $\mu_\mathcal{R}(\tau) = \mu_{\text{base}} - 2\lambda_\mathcal{R}(\tau)$ for both risks.

**Entropic** strengthens monotonicity: $\lambda_R^{\text{ent}}(\tau) = -\tfrac{1}{2}\tau\sigma^2 < 0$ and $\mu_R^{\text{ent}}(\tau) = 1 + \tau\sigma^2$ rises linearly above $\mu_{\text{base}}$. Mechanism: the entropic risk operator adds a **quadratic-in-$q_i$** variance penalty $\tfrac{1}{2}\tau\sigma^2 q_i^2$ to each firm's cost, which differentiates to $\tau\sigma^2 q_i$ in the gradient and adds $\tau\sigma^2 I$ to the Jacobian — pure positive-definite perturbation.

**CVaR** is monotonicity-neutral: $\lambda_R^{\text{CVaR}}(\tau) = 0$ exactly, and $\mu_R^{\text{CVaR}} = \mu_{\text{base}}$. Mechanism: the CVaR risk operator adds a **linear-in-$q_i$** term $\kappa_{\text{CVaR}}(\beta)\sigma q_i$ (where $\kappa_{\text{CVaR}}(\beta) = \phi(\Phi^{-1}(\beta))/(1-\beta)$), which differentiates to a constant $\kappa_{\text{CVaR}}\sigma$ in the gradient and contributes nothing to the Jacobian.

The asymmetry comes from the structure of the risk penalty in this Gaussian-linear-loss setup: variance (quadratic) for entropic vs. standard deviation (linear after taking expectation) for CVaR. This is what places entropic and CVaR at different points on the same parabola in Figure D.

## Note on welfare convention

The paper's §4 uses player-utilitarian welfare $\mathcal{W} = -\sum_i \mathbb{E}[\ell_i]$ (firm-only). This experiment uses the standard textbook total-welfare convention $\mathcal{W} = \mathbb{E}[\text{CS} + \text{PS}]$, since Cournot's most natural welfare metric is total surplus. The two conventions yield different externality operators ($\mathcal{E}^{\text{firm}}$ vs. $\mathcal{E}^\mathcal{W}$) and different alignment signs — but the welfare-decomposition machinery and the competition theorem apply unchanged for either convention. The total-welfare convention puts standard Cournot in the unfavorable-alignment regime; the player-utilitarian convention would put it in the favorable-alignment regime (firms collectively benefit from output reduction). The experiment chooses the textbook total-welfare convention because it's the convention reviewers will expect for Cournot.