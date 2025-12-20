# `sangkuriang`: Data Analysis Scripts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/OSF-10.17605/OSF.IO/MS5EJ-blue)](https://doi.org/10.17605/OSF.IO/MS5EJ)

Supplementary analysis scripts for the [`sangkuriang`](https://pypi.org/project/sangkuriang-ideal-solver/) KdV soliton solver.

## Related Resources

| Resource | Link |
|----------|------|
| **Solver (PyPI)** | https://pypi.org/project/sangkuriang-ideal-solver/ |
| **Solver (GitHub)** | https://github.com/sandyherho/sangkuriang-ideal-solver |
| **Data Repository (OSF)** | https://doi.org/10.17605/OSF.IO/MS5EJ |

## Requirements

```bash
pip install numpy scipy matplotlib netCDF4
```

## Usage

```bash
cd scripts/
python conservation_analysis.py
python spatiotemporal_evolution.py
python soliton_dynamics.py
python spectral_information.py
python phase_space_analysis.py
```

Each script reads NetCDF data from `../data/` and outputs figures to `../figs/` and statistics to `../stats/`.

## Analysis Scripts

### 1. Conservation Analysis (`conservation_analysis.py`)

Validates the three KdV invariants:

$$M = \int u \, dx, \quad P = \int u^2 \, dx, \quad E = \int \left[\frac{\varepsilon}{2} u^3 - \frac{3\mu}{2} \left(\frac{\partial u}{\partial x}\right)^2\right] dx$$

### 2. Spatiotemporal Evolution (`spatiotemporal_evolution.py`)

3D surface plots of $u(x,t)$ for all four test cases with unified axis limits and colorbar.

### 3. Soliton Dynamics (`soliton_dynamics.py`)

Peak tracking and velocity verification against the theoretical relation:

$$v = \frac{\varepsilon A}{3}$$

where $A$ is the soliton amplitude.

### 4. Spectral Information (`spectral_information.py`)

Information-theoretic measures from Fourier analysis:

**Spectral Entropy:**
$$S_k = -\frac{\sum_k p_k \ln p_k}{\ln N}, \quad p_k = \frac{P_k}{\sum_k P_k}$$

**LMC Statistical Complexity:**
$$C = H \times D$$

where $H$ is normalized entropy and $D = \sum_k (p_k - 1/N)^2$ is disequilibrium.

**Fisher Information:**
$$F = \int \frac{1}{p} \left(\frac{\partial p}{\partial x}\right)^2 dx$$

### 5. Phase Space Analysis (`phase_space_analysis.py`)

Recurrence Quantification Analysis (RQA) for dynamical characterization:

**Recurrence Matrix:**
$$R_{ij} = \Theta(\epsilon - \|\mathbf{x}_i - \mathbf{x}_j\|)$$

**Recurrence Rate:**
$$RR = \frac{1}{N^2} \sum_{i,j} R_{ij}$$

**Determinism:**
$$DET = \frac{\text{points in diagonal lines}}{\text{total recurrence points}}$$

High $DET$ (>0.99) confirms integrable (non-chaotic) dynamics.

## Directory Structure

```
sangkuriang-analysis/
├── scripts/
│   ├── conservation_analysis.py
│   ├── spatiotemporal_evolution.py
│   ├── soliton_dynamics.py
│   ├── spectral_information.py
│   └── phase_space_analysis.py
├── data/           # NetCDF files (from sangkuriang solver)
├── figs/           # Output figures (eps, pdf, png)
├── stats/          # Output statistics (txt)
├── LICENSE
└── README.md
```

## Authors

- Sandy H. S. Herho
- Faruq Khadami
- Iwan P. Anwar
- Dasapta E. Irawan

## License

MIT License
