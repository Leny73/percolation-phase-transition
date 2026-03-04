# 2D Percolation & Phase Transition Analysis
### Computational Physics Project | Stochastic SSPSF Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## 1. Overview

This repository presents a high-performance numerical simulation of **Site Percolation** on a 2D
square lattice. The project investigates the phase transition from isolated clusters to a spanning
(percolating) cluster, identifying the critical threshold $p_c$ and validating the results against
theoretical scaling laws and fractal geometry.

## 2. Key Technical Features

* **High Performance:** Implements `Numba` (`@njit`) for LLVM-based JIT compilation, ensuring
  near-native execution speeds for large-scale Monte Carlo simulations.
* **Advanced RNG:** Utilizes the **PCG32** (Permuted Congruential Generator) for high-quality,
  statistically independent random streams, crucial for scientific reproducibility.  A pure-Python
  class-based implementation with `clock_seed()` (SplitMix64 mixing) is also provided for
  portability.
* **Modular Design:** Professional software architecture with clear separation between simulation
  logic, RNG engines, and statistical analysis modules.
* **Quality Assurance:** Includes automated unit tests for RNG uniformity and grid boundary
  conditions.

## 3. Scientific Methodology & Indicators

### Phase Transition Analysis

The simulation tracks the order parameter $L_1$ (normalized size of the largest cluster) and the
susceptibility proxy $L_2$ (second-largest cluster size).

* **Critical Threshold ($p_c$):** For a 2D square lattice, the theoretical threshold is
  $p_c \approx 0.5927$.
* **Finite-Size Scaling:** Experiments are conducted across multiple lattice sizes
  $L \in \{100, 200, 400\}$ to extrapolate thermodynamic behaviour.

### Fractal Dimension ($d_f$)

At the critical point $p = p_c$, the percolating cluster behaves as a fractal object. This project
calculates the fractal dimension by fitting the mass–length scaling law:

$$M(L) \propto L^{d_f}$$

The expected theoretical value for 2D percolation is $d_f = 91/48 \approx 1.896$.

## 4. Project Structure

```text
├── src/                        # Core implementation logic
│   ├── rng_engine.py           # PCG32 JIT-optimised with Numba (@njit)
│   ├── pcg32_personal.py       # Pure-Python PCG32 class + clock_seed()
│   └── simulation.py           # Grid generation, cluster labelling & L1/L2
├── notebooks/                  # Colab/Jupyter research & visualisation
├── tests/                      # Automated unit tests
├── README.md                   # Documentation
└── requirements.txt            # Dependency manifest
```

## 5. Installation & Usage

**Clone the repository:**

```bash
git clone https://github.com/Leny73/percolation-phase-transition.git
cd percolation-phase-transition
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Minimal Python example:**

```python
from src.simulation import run_simulation

L1, L2, grid, labeled = run_simulation(L=100, p=0.5927, seed=42)
print(f"L1={L1}, L2={L2}, L2/L1={L2/L1:.4f}")
```

**Using the pure-Python PCG32 with a clock-based seed:**

```python
from src.pcg32_personal import clock_seed, PCG32

rng = PCG32(clock_seed())
print([rng.random() for _ in range(5)])
```

**Launch the notebook:**

```bash
jupyter notebook notebooks/percolation_analysis.ipynb
```

**Run Tests:**

```bash
pytest tests/
```

## 6. License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

*Developed as part of the Computational Physics (PIF) curriculum, 2023.*
