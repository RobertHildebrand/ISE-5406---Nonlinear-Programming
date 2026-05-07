"""Regenerate the three exam data files from scratch.

Outputs (in this directory):
  portfolio30.npz   -- 30-asset Markowitz problem (Sigma, mu, factor structure)
  wdbc.npz          -- UCI Wisconsin Diagnostic Breast Cancer (sklearn)
  case9.npz         -- IEEE 9-bus AC OPF benchmark (pypower)

Problem 4 (saddle-point dynamics) generates its own random data and needs no file.

Run:
    python3 generate_data.py
"""
from __future__ import annotations
import os
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def make_portfolio() -> None:
    """30 assets with a 1-market + 5-sector factor model."""
    rng = np.random.default_rng(42)
    n = 30
    n_factors = 6
    B = np.zeros((n, n_factors))
    B[:, 0] = rng.uniform(0.6, 1.4, n)             # market beta
    sector = rng.integers(1, n_factors, n)
    for i in range(n):
        B[i, sector[i]] = rng.uniform(0.5, 1.0)
    factor_var = np.array([0.04**2, 0.03**2, 0.025**2, 0.02**2, 0.018**2, 0.015**2])
    F = np.diag(factor_var)
    idio = rng.uniform(0.005, 0.025, n) ** 2
    Sigma = B @ F @ B.T + np.diag(idio)
    Sigma = 0.5 * (Sigma + Sigma.T)
    mu = 0.008 + 0.004 * B[:, 0] + 0.0015 * rng.standard_normal(n)
    np.savez(os.path.join(DATA_DIR, "portfolio30.npz"),
             Sigma=Sigma, mu=mu, B=B, sector=sector)
    eigs = np.linalg.eigvalsh(Sigma)
    print(f"[OK] portfolio30  n={n}  kappa(Sigma)={eigs[-1]/eigs[0]:.0f}  mu in [{mu.min():.4f}, {mu.max():.4f}]")


def make_wdbc() -> None:
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer()
    A = bc.data.astype(np.float64)
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    y = np.where(bc.target == 1, 1, -1).astype(np.int64)
    np.savez(os.path.join(DATA_DIR, "wdbc.npz"),
             A=A, y=y,
             feature_names=np.array(bc.feature_names),
             target_names=np.array(bc.target_names))
    print(f"[OK] wdbc  N={A.shape[0]} p={A.shape[1]}  pos rate={float((y==1).mean()):.3f}")


def make_case9() -> None:
    from pypower.api import case9
    case = case9()
    bus, gen, branch, gencost = case["bus"], case["gen"], case["branch"], case["gencost"]
    baseMVA = float(case["baseMVA"])
    nb = bus.shape[0]
    P_d = bus[:, 2] / baseMVA
    Q_d = bus[:, 3] / baseMVA
    V_min = bus[:, 12]; V_max = bus[:, 11]
    P_g_min = np.zeros(nb); P_g_max = np.zeros(nb)
    Q_g_min = np.zeros(nb); Q_g_max = np.zeros(nb)
    c2 = np.zeros(nb); c1 = np.zeros(nb); c0 = np.zeros(nb)
    for i, b in enumerate(gen[:, 0].astype(int)):
        idx = b - 1
        P_g_min[idx] = gen[i, 9] / baseMVA
        P_g_max[idx] = gen[i, 8] / baseMVA
        Q_g_min[idx] = gen[i, 4] / baseMVA
        Q_g_max[idx] = gen[i, 3] / baseMVA
        c2[idx] = gencost[i, 4]
        c1[idx] = gencost[i, 5]
        c0[idx] = gencost[i, 6]
    Y = np.zeros((nb, nb), dtype=complex)
    for k in range(branch.shape[0]):
        f = int(branch[k, 0]) - 1; t = int(branch[k, 1]) - 1
        z = branch[k, 2] + 1j * branch[k, 3]
        y = 1.0 / z
        b_sh = branch[k, 4]
        Y[f, f] += y + 1j * b_sh / 2
        Y[t, t] += y + 1j * b_sh / 2
        Y[f, t] -= y; Y[t, f] -= y
    np.savez(os.path.join(DATA_DIR, "case9.npz"),
             G=Y.real, B=Y.imag,
             P_g_min=P_g_min, P_g_max=P_g_max,
             Q_g_min=Q_g_min, Q_g_max=Q_g_max,
             P_d=P_d, Q_d=Q_d,
             c2=c2, c1=c1, c0=c0,
             V_min=V_min, V_max=V_max,
             gen_bus=gen[:, 0].astype(np.int64),
             baseMVA=baseMVA)
    print(f"[OK] case9  IEEE 9-bus, total load = {P_d.sum() * baseMVA:.0f} MW")


if __name__ == "__main__":
    make_portfolio()
    make_wdbc()
    make_case9()
    print(f"\nAll files in {DATA_DIR}:")
    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            print(f"  {f:>20s}  ({os.path.getsize(path):,} bytes)")
