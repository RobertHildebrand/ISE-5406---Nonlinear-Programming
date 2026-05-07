# Final Exam — Datasets

Three `.npz` files used by the take-home Final Exam. Load with `numpy.load`. Problem 4 generates its own random data (no file needed).

| File | Used by | Contents |
|---|---|---|
| `portfolio30.npz` | Problem 1 | `Sigma` (30×30 covariance, PSD, $\kappa \approx 1300$), `mu` (30-vector of expected returns), plus the underlying factor structure (`B`, `sector`) for reference. |
| `wdbc.npz` | Problem 2 | UCI Wisconsin Diagnostic Breast Cancer. `A` (569 × 30 features, pre-standardized), `y` ({−1, +1} labels for malignant/benign), plus `feature_names` and `target_names`. |
| `case9.npz` | Problem 3 | IEEE 9-bus AC OPF benchmark. Per-unit `G`, `B` (admittance), generator bounds `P_g_min/max`, `Q_g_min/max`, loads `P_d`, `Q_d`, quadratic cost coefficients `c2`, `c1`, `c0`, voltage bounds `V_min`, `V_max`, `gen_bus`, `baseMVA`. |

## Loading code

```python
import numpy as np

# Problem 1
d = np.load('data/portfolio30.npz')
Sigma, mu = d['Sigma'], d['mu']

# Problem 2
d = np.load('data/wdbc.npz')
A, y = d['A'], d['y']

# Problem 3
d = np.load('data/case9.npz')
G, B   = d['G'], d['B']
P_d, Q_d = d['P_d'], d['Q_d']
P_g_min, P_g_max = d['P_g_min'], d['P_g_max']
Q_g_min, Q_g_max = d['Q_g_min'], d['Q_g_max']
c2, c1, c0       = d['c2'], d['c1'], d['c0']
V_min, V_max     = d['V_min'], d['V_max']
```

## Reproducibility

All three datasets are reproducible from `generate_data.py` (fixed numpy seeds, sklearn-bundled WDBC, pypower-bundled IEEE-9). Re-running `python3 generate_data.py` recreates byte-identical files.

## Dependencies

Generating the files requires:

- `numpy`
- `scikit-learn` (for `wdbc`)
- `pypower` (for `case9`)

Loading the files only requires `numpy`.
