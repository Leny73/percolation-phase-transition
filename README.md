# Percolation Phase Transition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A Python project for **2D site-percolation** and **phase transition** analysis on a square lattice,
motivated by stochastic models of large-scale star formation in molecular clouds.

---

## Scientific context — Stochastic Star Formation

Interstellar molecular clouds can be mapped onto a 2-D square lattice in which each site is
*occupied* (activated) with independent probability *p*, representing the local likelihood of
reaching the critical density for star formation.

At the **percolation threshold** $p_c \approx 0.5927$ a *giant connected component* spanning the
entire lattice first appears — the onset of a large-scale, galaxy-wide star-formation episode.

The ratio of the second-largest to the largest cluster size, $L_2 / L_1$, exhibits a sharp peak
near $p_c$ and is used here as a finite-size estimator of the threshold:

$$p_c^{(L)} = \arg\max_p \frac{L_2(p,L)}{L_1(p,L)} \xrightarrow{L \to \infty} 0.5927$$

---

## Project structure

```
percolation-phase-transition/
├── src/
│   ├── __init__.py
│   ├── rng_engine.py      # PCG32 RNG, JIT-compiled with Numba
│   └── simulation.py      # Grid generation, cluster labelling, L1/L2
├── tests/
│   ├── __init__.py
│   ├── test_rng_engine.py
│   └── test_simulation.py
├── notebooks/
│   └── percolation_analysis.ipynb   # Full analysis and visualisation
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Modules

### `src/rng_engine.py`

Implements the **PCG32** (Permuted Congruential Generator, 32-bit output) algorithm, a fast,
statistically high-quality pseudo-random number generator.  All hot-path functions are decorated
with Numba's `@njit` for near-C performance.

| Function | Description |
|---|---|
| `pcg32_seed(state, inc)` | Initialise the generator state |
| `pcg32_next(state, inc)` | Advance state, return next `uint32` |
| `pcg32_float(state, inc)` | Return next float in `[0, 1)` |
| `pcg32_fill_float(n, seed, inc)` | Fill a length-*n* array with uniform floats |

### `src/simulation.py`

| Function | Description |
|---|---|
| `generate_grid(L, p, seed, inc)` | Numba-accelerated *L × L* binary lattice |
| `label_clusters(grid)` | 4-connected cluster labelling via `scipy.ndimage.label` |
| `cluster_sizes(labeled, n)` | Sorted cluster-size array |
| `get_l1_l2(labeled, n)` | Return *(L1, L2)* — largest and second-largest sizes |
| `run_simulation(L, p, seed, inc)` | End-to-end helper returning *(L1, L2, grid, labeled)* |

---

## Quick start

```bash
# 1. Clone and install dependencies
git clone https://github.com/Leny73/percolation-phase-transition.git
cd percolation-phase-transition
pip install -r requirements.txt

# 2. Run the test suite
python -m pytest tests/ -v

# 3. Launch the notebook
jupyter notebook notebooks/percolation_analysis.ipynb
```

### Minimal Python example

```python
from src.simulation import run_simulation

L1, L2, grid, labeled = run_simulation(L=100, p=0.5927, seed=42)
print(f"L1={L1}, L2={L2}, L2/L1={L2/L1:.4f}")
```

---

## Requirements

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `numba` | JIT compilation of RNG and grid generation |
| `scipy` | Connected-component labelling (`ndimage.label`) |
| `matplotlib` | Visualisation |
| `jupyter` | Interactive notebook |

Install with:

```bash
pip install -r requirements.txt
```

---

## License

This project is released under the [MIT License](LICENSE).
