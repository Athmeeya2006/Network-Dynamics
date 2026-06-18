# Nonlinear Dynamics on Networks

A research-grade simulation repository tracing one coherent ladder - from a **single nonlinear unit**, through **chaos**, **game dynamics**, and **network structure**, up to the frontier of **dynamics on networks** and **higher-order (simplicial) interactions**. Every script verifies a closed-form prediction numerically and prints the recovered number; every figure is built from a shared publication palette.

Theory and synthetic simulation **only** - no real-world datasets. This repo is the companion to, and deliberately closes the loop with, the [Erdős-Rényi-Contagion](../Erdos-Renyi-Contagion) repo: Module 6c (simplicial contagion) is the higher-order analogue of that repo's pairwise SIR/percolation work, reusing its Batagelj-Brandes generator and visual identity.

---

## The spine

The six modules are rungs of **one** ladder, not six projects:

```
single unit  →  chaos in one unit  →  interacting units  →  network structure
      \________________________________________________________/
                              │
                  dynamics ON networks (sync, explosive sync, MSF)
                              │
                  higher-order interactions (simplicial complexes)
                              │
              ENDPOINT: explosive sync + simplicial contagion
```

**Prime directive - composability.** The oscillator units in [`src/oscillators.py`](src/oscillators.py) (FitzHugh-Nagumo, Stuart-Landau, van der Pol) are written as network-ready classes with a single `rhs(state, t, coupling_input)` interface. Module 5 instantiates *N* of them on a Module-4 graph with one diffusive coupling term ([`coupled_network_rhs`](src/oscillators.py)) - no unit dynamics are ever rewritten. Module 5c exercises this end to end.

---

## Repository structure

```text
nonlinear-dynamics-networks/
├── src/                          # 13 modules, ~2,500 lines
│   ├── integrators.py            # fixed-step RK4 + solve_ivp wrapper (Lyapunov needs fixed step)
│   ├── continuation.py           # fixed-point tracking + stability for bifurcation diagrams
│   ├── maps.py                   # 1-D maps (logistic) + cobweb helper
│   ├── flows.py                  # Lorenz, Rössler, fixed points, Hopf thresholds
│   ├── lyapunov.py               # map exponent, Benettin largest + spectrum (QR), MSF, Kaplan-Yorke
│   ├── oscillators.py            # FHN / Stuart-Landau / van der Pol classes + coupled_network_rhs
│   ├── games.py                  # replicator RPS, spatial RPS, network RPS, spatial structure
│   ├── networks.py               # ER (Batagelj-Brandes) / BA / config / WS, Laplacians, centralities, eigenratio
│   ├── powerlaw_fit.py           # Clauset-Shalizi-Newman MLE + KS x_min + bootstrap p-value + OLS (for bias)
│   ├── kuramoto.py               # phase oscillators (mean-field & network), order parameter, adiabatic sweep
│   ├── simplicial.py             # clique complex, triangles, higher-order Kuramoto, simplicial SIS
│   └── viz.py                    # palette + light/dark themes + despine (ported from the ER repo)
├── scripts/                      # 27 scripts, one per figure, ~3,600 lines
│   ├── m1*  Local dynamics       (bifurcations, phase plane, Hopf, FHN, van der Pol)
│   ├── m2*  Routes to chaos      (logistic cobweb/bifurcation/Lyapunov, Lorenz, Rössler)
│   ├── m3*  Game dynamics        (replicator RPS, spatial RPS, RPS on networks)
│   ├── m4*  Network structure    (centralities, small-world, degree distributions, power-law fitting)
│   ├── m5*  Dynamics ON networks (Kuramoto, on networks, coupled units, explosive sync, MSF)
│   └── m6*  Higher-order         (simplicial complex, HO Kuramoto, simplicial contagion, summary)
├── media/
│   ├── figures/                  # 27 publication PNGs
│   └── interactive/              # rotatable Plotly attractors (Lorenz, Rössler)
├── requirements.txt              # numpy, scipy, matplotlib, networkx, manim, plotly
└── .github/workflows/ci.yml      # flake8 (E9,F63,F7,F82)
```

Every script: resolves `sys.path` dynamically, writes to `media/`, carries a docstring stating the proof it verifies, prints a **theory-vs-measured** line with the recovered number, and uses fixed seeds.

---

## The ladder, rung by rung

### Module 1 - Local dynamics (the single unit)
Normal forms and their bifurcations; the trace-determinant plane; the Hopf bifurcation as the birth of a limit cycle; FitzHugh-Nagumo excitability; van der Pol relaxation oscillations.
- **Verified:** supercritical pitchfork amplitude exponent → **0.5**; Stuart-Landau limit-cycle radius ∝ √μ (exponent **0.5**); FHN Hopf onset; van der Pol period scaling.

### Module 2 - Routes to chaos (the single unit, pushed)
The logistic map's period-doubling cascade; Lyapunov exponents; the Lorenz and Rössler attractors.
- **Verified:** Feigenbaum **δ → 4.6692** and **α → 2.5029**; Lyapunov exponent crosses zero at each period-doubling; Lorenz subcritical Hopf at **ρ_H ≈ 24.74**; Lorenz **λ_max ≈ 0.906**, **Kaplan-Yorke dimension ≈ 2.06**; Rössler period-doubling in the Poincaré map. Rotatable 3-D attractors exported to `media/interactive/`.

### Module 3 - Game dynamics (interacting units, the bridge)
Replicator dynamics of Rock-Paper-Scissors, on the simplex, on a lattice, and on networks.
- **Verified:** neutral RPS conserves **V = x₁x₂x₃**; spatial RPS shows a **homogenisation transition** - the nearest-neighbour structure order parameter *S(M)* collapses monotonically as mobility *M* rises (spiral domains → well-mixed); on networks, imitation dynamics **fixates every run** (biodiversity loss) while the **ensemble mean stays at the symmetric Nash 1/3** by cyclic symmetry.

### Module 4 - Network structure (the substrate)
Centralities, the Watts-Strogatz small-world transition, degree distributions, and rigorous power-law fitting.
- **Verified:** hub localisation tightens the degree↔eigenvector-centrality correlation on BA (**r ≈ 0.93**) vs ER (**r ≈ 0.89**); a small-world window where *L*(p) has collapsed but *C*(p) is retained; BA degree exponent **γ_MLE ≈ 2.87** (theory 3); **Clauset MLE recovers a synthetic exponent to err ≈ 0.00** with bootstrap **p ≈ 0.82**, while naive log-log **OLS is visibly biased** - the methodological upgrade over the ER repo's OLS-only fits.

### Module 5 - Dynamics ON networks (the synthesis, the payoff)
Kuramoto synchronisation, mean-field and on networks; Module-1 units coupled on graphs; explosive synchronisation; the Master Stability Function.
- **Verified:**
  - **K_c = 2/(πg(0))** - Lorentzian fit recovers **K_c = 2.01** vs 2.00 (0.6 %).
  - Scale-free topology lowers the sync onset: **K₅₀(BA) = 0.30 < K₅₀(ER) = 0.34**.
  - **Composability:** coupled FHN units support a travelling wave; coupled Stuart-Landau units exhibit **amplitude death** (mean amplitude → 0 in a coupling window [1.5, 3.0]) - the interface works end to end.
  - **Explosive synchronisation (m5d, lab payoff):** with **ωᵢ = kᵢ** the transition is **first-order with a hysteresis loop** (area 0.12, jump 0.61); shuffling the frequencies restores a continuous transition (area 0.03).
  - **Master Stability Function (m5e):** the eigenratio **λ_N/λ₂** ranks synchronisability - predicted and simulated onsets agree with **Spearman rank correlation = 1.00**, and the complete graph (eigenratio 1) synchronises first.

### Module 6 - Higher-order interactions (the frontier)
Simplicial complexes, higher-order Kuramoto, and simplicial contagion - closing the loop with the ER repo.
- **Verified:**
  - Clique-complex triangle enumeration matches an independent networkx count **exactly**.
  - **Higher-order Kuramoto (m6b):** triadic (2-simplex) coupling makes synchronisation **explosive** (hysteresis area 0.08 → 1.60) **with random frequencies - no degree-frequency correlation needed**.
  - **Simplicial contagion (m6c):** the triadic infection channel turns the continuous pairwise epidemic transition **first-order**, opening a **bistable region** (forward jump 0.52; hysteresis area 0.0004 → 0.004) - the higher-order analogue of the ER repo's pairwise SIR baseline.
  - **Capstone (m6d):** a 2×2 dashboard shows pairwise→continuous vs higher-order→explosive, for **both** synchronisation and contagion.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# any single figure
python scripts/m5d_explosive_synchronization.py     # → media/figures/, prints theory-vs-measured

# the whole ladder
for f in scripts/m*.py; do python "$f"; done
```

Each run prints a `VERIFY` block ending in `PASS`/`FAIL` against the stated tolerance, and saves a PNG to `media/figures/`.

---

## Acceptance bar (every script)

A script is *done* only when it (1) states the theorem it verifies in its docstring, (2) prints theory-vs-measured with the recovered number, (3) saves a publication-quality figure from the shared palette, (4) recovers its target within the stated tolerance, and (5) is reproducible (fixed seeds) and passes `flake8 --select=E9,F63,F7,F82`.

---

## Reuse from the Erdős-Rényi repo

- The `viz.py` palette/theme helpers are ported verbatim for visual consistency.
- The Batagelj-Brandes O(n + M) ER generator is reused in `networks.py`.
- Module 6c (simplicial contagion) is intentionally the higher-order continuation of that repo's pairwise contagion work, so the two repos read as one research arc.
