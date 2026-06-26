# Nonlinear Dynamics on Networks

A simulation repository that builds up in six steps: from a single nonlinear unit, through chaos, game dynamics, and network structure, to dynamics on networks and higher-order (simplicial) interactions. Every script checks a closed-form prediction numerically and prints the recovered number, and every figure uses one shared colour palette.

The work uses theory and synthetic simulation only, with no real-world datasets. It is the companion to the [Erdos-Renyi-Contagion](../Erdos-Renyi-Contagion) repo: Module 6c (simplicial contagion) is the higher-order version of that repo's pairwise SIR and percolation work, and reuses its Batagelj-Brandes generator and colour scheme.

---

## How the modules fit together

The six modules form one progression rather than six separate projects:

```
single unit  ->  chaos in one unit  ->  interacting units  ->  network structure
                                                                      |
                                          dynamics on networks (sync, explosive sync, MSF)
                                                                      |
                                          higher-order interactions (simplicial complexes)
                                                                      |
                                          endpoint: explosive sync and simplicial contagion
```

The design goal is composability. The oscillator units in [`src/oscillators.py`](src/oscillators.py) (FitzHugh-Nagumo, Stuart-Landau, van der Pol) are written as network-ready classes with a single `rhs(state, t, coupling_input)` interface. Module 5 places N of them on a Module 4 graph with one diffusive coupling term ([`coupled_network_rhs`](src/oscillators.py)), without rewriting any unit dynamics. Module 5c exercises this end to end.

---

## Repository structure

```text
nonlinear-dynamics-networks/
├── src/                          # 13 modules
│   ├── integrators.py            # fixed-step RK4 and a solve_ivp wrapper (Lyapunov needs fixed step)
│   ├── continuation.py           # fixed-point tracking and stability for bifurcation diagrams
│   ├── maps.py                   # 1D maps (logistic) and a cobweb helper
│   ├── flows.py                  # Lorenz, Rossler, fixed points, Hopf thresholds
│   ├── lyapunov.py               # map exponent, Benettin largest and spectrum (QR), MSF, Kaplan-Yorke
│   ├── oscillators.py            # FHN, Stuart-Landau, van der Pol classes and coupled_network_rhs
│   ├── games.py                  # replicator RPS, spatial RPS, network RPS, spatial structure
│   ├── networks.py               # ER (Batagelj-Brandes), BA, configuration, WS, Laplacians, centralities
│   ├── powerlaw_fit.py           # Clauset-Shalizi-Newman MLE, KS x_min, bootstrap p-value, OLS (for bias)
│   ├── kuramoto.py               # phase oscillators (mean-field and network), order parameter, sweep
│   ├── simplicial.py             # clique complex, triangles, higher-order Kuramoto, simplicial SIS
│   └── viz.py                    # palette, light and dark themes, despine (ported from the ER repo)
├── scripts/                      # one script per figure
│   ├── m1*  Local dynamics       (bifurcations, phase plane, Hopf, FHN, van der Pol)
│   ├── m2*  Routes to chaos      (logistic cobweb/bifurcation/Lyapunov, Lorenz, Rossler)
│   ├── m3*  Game dynamics        (replicator RPS, spatial RPS, RPS on networks)
│   ├── m4*  Network structure    (centralities, small-world, degree distributions, power-law fitting)
│   ├── m5*  Dynamics on networks (Kuramoto, on networks, coupled units, explosive sync, MSF)
│   └── m6*  Higher-order         (simplicial complex, HO Kuramoto, simplicial contagion, summary)
├── media/
│   ├── figures/                  # one PNG per script
│   └── interactive/              # browser labs and animation clips (see below)
├── requirements.txt              # numpy, scipy, matplotlib, networkx, manim, plotly
└── .github/workflows/ci.yml      # flake8 (E9,F63,F7,F82)
```

Every script resolves `sys.path` at runtime, writes to `media/`, carries a docstring stating the result it checks, prints a theory-vs-measured line with the recovered number, and uses fixed seeds.

---

## Interactive simulations

Open [`media/interactive/index.html`](media/interactive/index.html) in a browser. The labs are self-contained (one local copy of Plotly, no network needed), and each page now carries a written explanation of what it shows, the equations behind it, and a few things to try.

The Module 1 labs cover the topics on bifurcations and limit cycles directly:

- **1D bifurcations** ([`sim_1d_bifurcations.html`](media/interactive/sim_1d_bifurcations.html)): the four normal forms of a one-dimensional flow (saddle-node, transcritical, supercritical and subcritical pitchfork). It draws the bifurcation diagram together with the flow on the line, so you can see fixed points appear, collide and swap stability as the parameter moves.
- **Hopf bifurcation in 2D** ([`sim_2d_bifurcations.html`](media/interactive/sim_2d_bifurcations.html)): the birth of a limit cycle, in both the soft (supercritical) and hard (subcritical) cases, with the amplitude diagram, the unstable cycle, and the hysteresis window of the subcritical case.
- **Van der Pol limit cycle** ([`sim_van_der_pol.html`](media/interactive/sim_van_der_pol.html)): one limit cycle up close, with the phase loop and the waveform side by side, sweeping from a near-sinusoidal orbit to stiff relaxation spikes.
- **Pitchfork potential**, **Stuart-Landau limit cycle**, and the **2D phase portrait** lab round out the local-dynamics picture.

Other labs cover the logistic map, the Lorenz attractor, the double pendulum, Rock-Paper-Scissors replicator dynamics, and Kuramoto synchronisation. The page also lists auto-playing animation clips of the same systems.

---

## The modules

### Module 1: local dynamics (the single unit)
Normal forms and their bifurcations, the trace-determinant plane, the Hopf bifurcation as the birth of a limit cycle, FitzHugh-Nagumo excitability, and van der Pol relaxation oscillations.

Checked: supercritical pitchfork amplitude exponent of 0.5; Stuart-Landau limit-cycle radius proportional to sqrt(mu) (exponent 0.5); FHN Hopf onset; van der Pol period scaling.

### Module 2: routes to chaos (the single unit, pushed)
The logistic map's period-doubling cascade, Lyapunov exponents, and the Lorenz and Rossler attractors.

Checked: Feigenbaum constants delta = 4.6692 and alpha = 2.5029; the Lyapunov exponent crosses zero at each period-doubling; Lorenz subcritical Hopf at rho_H = 24.74; Lorenz lambda_max = 0.906 and Kaplan-Yorke dimension = 2.06; Rossler period-doubling in the Poincare map. Rotatable 3D attractors are exported to `media/interactive/`.

### Module 3: game dynamics (interacting units)
Replicator dynamics of Rock-Paper-Scissors, on the simplex, on a lattice, and on networks.

Checked: neutral RPS conserves V = x1 x2 x3; spatial RPS shows a homogenisation transition, where the nearest-neighbour structure order parameter S(M) falls monotonically as mobility M rises (spiral domains give way to a well-mixed state); on networks, imitation dynamics fixates on every run (loss of biodiversity) while the ensemble mean stays at the symmetric Nash point of 1/3 by cyclic symmetry.

### Module 4: network structure (the substrate)
Centralities, the Watts-Strogatz small-world transition, degree distributions, and careful power-law fitting.

Checked: hub localisation tightens the correlation between degree and eigenvector centrality on BA (r = 0.93) against ER (r = 0.89); a small-world window where the path length L(p) has collapsed but the clustering C(p) is retained; BA degree exponent gamma_MLE = 2.87 (theory 3); the Clauset MLE recovers a synthetic exponent to error near 0.00 with bootstrap p = 0.82, while a naive log-log OLS fit is visibly biased.

### Module 5: dynamics on networks
Kuramoto synchronisation, mean-field and on networks; Module 1 units coupled on graphs; explosive synchronisation; the Master Stability Function.

Checked:
- K_c = 2 / (pi g(0)). A Lorentzian fit recovers K_c = 2.01 against 2.00 (0.6 percent).
- Scale-free topology lowers the sync onset: K50(BA) = 0.30 is below K50(ER) = 0.34.
- Composability: coupled FHN units support a travelling wave; coupled Stuart-Landau units show amplitude death (mean amplitude falls to 0 in a coupling window of [1.5, 3.0]).
- Explosive synchronisation (m5d): with omega_i = k_i the transition is first-order with a hysteresis loop (area 0.12, jump 0.61); shuffling the frequencies restores a continuous transition (area 0.03).
- Master Stability Function (m5e): the eigenratio lambda_N / lambda_2 ranks synchronisability. Predicted and simulated onsets agree with a Spearman rank correlation of 1.00, and the complete graph (eigenratio 1) synchronises first.

### Module 6: higher-order interactions
Simplicial complexes, higher-order Kuramoto, and simplicial contagion, connecting back to the ER repo.

Checked:
- Clique-complex triangle enumeration matches an independent networkx count exactly.
- Higher-order Kuramoto (m6b): triadic (2-simplex) coupling makes synchronisation explosive (hysteresis area 0.08 to 1.60) with random frequencies, without any degree-frequency correlation.
- Simplicial contagion (m6c): the triadic infection channel turns the continuous pairwise epidemic transition first-order, opening a bistable region (forward jump 0.52, hysteresis area 0.0004 to 0.004), the higher-order version of the ER repo's pairwise SIR baseline.
- Capstone (m6d): a 2x2 dashboard shows pairwise (continuous) against higher-order (explosive), for both synchronisation and contagion.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# any single figure
python scripts/m5d_explosive_synchronization.py     # writes to media/figures/, prints theory vs measured

# every figure
for f in scripts/m*.py; do python "$f"; done
```

Each run prints a `VERIFY` block ending in `PASS` or `FAIL` against the stated tolerance, and saves a PNG to `media/figures/`.

---

## Acceptance bar (every script)

A script is done only when it (1) states the result it checks in its docstring, (2) prints theory vs measured with the recovered number, (3) saves a figure from the shared palette, (4) recovers its target within the stated tolerance, and (5) is reproducible (fixed seeds) and passes `flake8 --select=E9,F63,F7,F82`.

---

## Reuse from the Erdos-Renyi repo

- The `viz.py` palette and theme helpers are ported for visual consistency.
- The Batagelj-Brandes O(n + M) ER generator is reused in `networks.py`.
- Module 6c (simplicial contagion) continues that repo's pairwise contagion work into the higher-order setting, so the two repos read together.
