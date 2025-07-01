# Nonlinear Newton Systems

A Python implementation and analysis of classical and modified Newton-Raphson methods for solving systems of nonlinear equations.

## 📖 Overview

This project explores:
- **Full Newton–Raphson:** Jacobian recomputed every iteration (quadratic convergence).
- **Modified Newton:** Jacobian fixed at initial guess (linear convergence).
- **Interval analysis:** Determine sane initial guess intervals.
- **Convergence plots:** Semilog and log–log charts, plus reference quadratic-order line.

## 🚀 Features

- Modular functions:
  - `f(x)` and `J(x)` definitions.
  - `newton_system` and `modified_newton` solvers.
- Automatic detection of divergence (NaN/∞) without crashing.
- Toggleable warning suppression for overflow.
- Generation of separate convergence plots per initial guess.
- Script prints iteration counts and stops gracefully on divergence.

## 🛠️ Requirements

- Python 3.8+
- NumPy
- Matplotlib

Install dependencies via pip:
```bash
pip install numpy matplotlib

