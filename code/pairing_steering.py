"""Verify the pairing-as-class-steering-lever result (SM Sec. S4.3).

Shows that per-context pairing efficiencies phi_m reshape the effective gain spectrum
P^T diag(phi) P, converting a spiked (avalanche-prone) portfolio into a geometric
(monitorable) one, and quantifies the coordination-efficiency cost.
"""
import numpy as np
from scipy.optimize import minimize

rng = np.random.default_rng(0)
d, M = 6, 6
common = np.zeros(d); common[0] = 1.0
P = np.array([common + 0.25 * rng.standard_normal(d) for _ in range(M)])

def spectrum(f):
    ev = np.sort(np.linalg.eigvalsh(P.T @ np.diag(f) @ P))[::-1]
    return ev[ev > 1e-9]

def logspacing_var(f):
    ev = spectrum(f)
    if len(ev) < 3:
        return 1e3
    return np.var(np.log(ev[:-1] / ev[1:]))

f_min = 0.15
res = minimize(logspacing_var, np.ones(M) * 0.7,
               bounds=[(f_min, 1)] * M, method='L-BFGS-B')
f_opt = res.x

print(f"log-spacing variance: uniform={logspacing_var(np.ones(M)):.2f} "
      f"-> optimized={logspacing_var(f_opt):.2f}")
print(f"coordination proxy sum(f): {M} -> {f_opt.sum():.2f} "
      f"({100*(1-f_opt.sum()/M):.0f}% sacrificed)")
print("optimized schedule f_m:", f_opt.round(2))
