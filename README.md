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
  logic, RNG engines, theory, and statistical analysis modules.
* **Input Validation:** Early edge-case checking via `validate_percolation_params` provides
  descriptive errors before any expensive computation begins.
* **Theory Module (`theory.py`):** Estimates the fractal dimension $d_f$ from finite-size
  simulation data via log-log linear regression and compares against the exact RG value
  $d_f = 91/48 \approx 1.8958$.
* **Validation Module (`validation.py`):** Extracts the correlation-length exponent $\nu$ via
  finite-size scaling of the susceptibility proxy and benchmarks it against the exact RG prediction
  $\nu = 4/3$.
* **Power-Law Visualisation:** `main.py` generates a log-log cluster-size distribution plot at
  $p_c$ and fits the Fisher exponent $\tau = 187/91 \approx 2.055$.
* **Quality Assurance:** Includes automated unit tests for RNG uniformity, grid boundary
  conditions, fractal-dimension estimation, and critical-exponent validation.

## 3. Scientific Methodology & Indicators

### Phase Transition Analysis

The simulation tracks the order parameter $L_1$ (normalized size of the largest cluster) and the
susceptibility proxy $L_2$ (second-largest cluster size).

* **Critical Threshold ($p_c$):** For a 2D square lattice, the theoretical threshold is
  $p_c \approx 0.5927$ (best estimate: $p_c = 0.59274605$, Ziff 2021).
* **Finite-Size Scaling:** Experiments are conducted across multiple lattice sizes
  $L \in \{100, 200, 400\}$ to extrapolate thermodynamic behaviour.

### Fractal Dimension ($d_f$)

At the critical point $p = p_c$, the percolating cluster behaves as a fractal object. This project
calculates the fractal dimension by fitting the mass–length scaling law:

$$M(L) \propto L^{d_f}$$

The expected theoretical value for 2D percolation is $d_f = 91/48 \approx 1.896$.

### Correlation-Length Exponent ($\nu$)

The module `validation.py` estimates $\nu$ via finite-size scaling of the susceptibility proxy at
$p_c$:

$$\langle L_2 \rangle(p_c, L) \sim L^{\gamma/\nu}, \quad \gamma/\nu = 43/24 \approx 1.792$$

Fitting $\log\langle L_2\rangle$ vs $\log L$ yields $\gamma/\nu$, from which
$\nu = \gamma / (\gamma/\nu)$ is inferred ($\gamma = 43/18$, exact RG).  The theoretical value is
$\nu = 4/3$.

### Cluster-Size Distribution ($\tau$)

At the critical point the number density of clusters of size $s$ follows a power law:

$$n(s) \sim s^{-\tau}, \quad \tau = 187/91 \approx 2.055$$

`main.py` generates a log-log histogram of $n(s)$ at $p_c$ and fits the slope, providing the
strongest visual evidence that the simulation sits exactly at the phase transition.

## 4. Project Structure

```text
├── main.py                     # Entry point: phase-transition plot & power-law viz
├── src/                        # Core implementation logic
│   ├── rng_engine.py           # PCG32 JIT-optimised with Numba (@njit)
│   ├── pcg32_personal.py       # Pure-Python PCG32 class + clock_seed()
│   ├── simulation.py           # Grid generation, cluster labelling & L1/L2
│   ├── theory.py               # Fractal-dimension estimation via log-log regression
│   └── validation.py           # Critical-exponent (ν) validation via FSS
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

**Run the full analysis (phase-transition plot + power-law distribution):**

```bash
python main.py
```

This saves `phase_transition.png` (order-parameter sweep) and `power_law.png` (log-log cluster-size
distribution at $p_c$ with fitted Fisher exponent $\tau$).

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

**Estimate the fractal dimension from simulation data:**

```python
import numpy as np
from src.theory import calculate_fractal_dimension

Ls = np.array([100, 200, 400], dtype=float)
L1s = np.array([500, 1800, 6500], dtype=float)  # largest-cluster sizes at p_c
df = calculate_fractal_dimension(L1s, Ls)
```

**Validate the correlation-length exponent ν via finite-size scaling:**

```python
from src.validation import check_critical_exponents

nu = check_critical_exponents(L_values=(50, 100, 200), n_samples=10)
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
