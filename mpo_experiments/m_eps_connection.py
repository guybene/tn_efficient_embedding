import numpy as np
import matplotlib.pyplot as plt
from tensor_sketcher import TensorSketcher

# ---- params ----
d, N, D, K = 2, 4, 2, 1
delta      = 0.2
n_samples  = 200
epsilons   = np.array([0.4, 0.3,0.2, 0.1, 0.05])
T_true     = d ** (N)

# ---- empirical failure rate ----
def fail_rate(m, eps):
    bad = 0
    print("Checking m: ", m)
    for i in range(n_samples):
        est = float(TensorSketcher(m=int(m), N=N, d=d, D=D, K=K).sketch(mpos=None))
        bad += abs(est - T_true) > eps * abs(T_true)
    return bad / n_samples

# ---- bisect on log2(m): smallest m with fail_rate <= delta ----
def min_m(eps):
    print("Working on epsilon: ", eps)
    m_lo, m_hi = 1, 4
    while fail_rate(m_hi, eps) > delta:
        m_hi *= 2
    while m_hi - m_lo > 1:
        m_mid = int(round(np.sqrt(m_lo * m_hi)))
        m_mid = min(max(m_mid, m_lo + 1), m_hi - 1)
        if fail_rate(m_mid, eps) <= delta: m_hi = m_mid
        else:                              m_lo = m_mid
    return m_hi

# ---- sweep ----
m_star = np.array([min_m(e) for e in epsilons])
print("eps:   ", epsilons)
print("m*:    ", m_star)

# ---- fit log m = a - beta log eps ----
slope, intercept = np.polyfit(np.log(epsilons), np.log(m_star), 1)
beta = -slope
print(f"fitted slope = {slope:.3f}  =>  m* ~ eps^(-{beta:.3f})  (theory: -2)")

# ---- plot ----
plt.figure(figsize=(7, 5))
plt.loglog(epsilons, m_star, 'o', ms=9, label=r'empirical $m^*$')
plt.loglog(epsilons, np.exp(intercept) * epsilons**slope, '-',
           label=fr'fit: slope $= {slope:.2f}$')
plt.loglog(epsilons, m_star[0] * (epsilons[0]/epsilons)**2, '--',
           label=r'reference slope $-2$')
plt.xlabel(r'$\epsilon$'); plt.ylabel(r'$m^*$')
plt.title(fr'$d={d},\ N={N},\ D={D},\ K={K},\ \delta={delta}$')
plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.tight_layout()
plt.savefig('m_eps_scaling.png', dpi=150)
