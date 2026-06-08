"""
Numerical check of chain-value asymptotics.
Imports build_W, build_u_init, build_u_end from the user's file.
"""
import sys, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '/mnt/user-data/uploads')
from original_tested_create_M import (
    build_W, build_u_init, build_u_end, closest_dividers
)


def _params(m, dist):
    m1, m2 = closest_dividers(m)
    s2 = 1.0/m
    s4 = 3.0*s2**2 if dist == 'gaussian' else s2**2
    return m1, m2, s2, s4


def _pieces(chain, m, dist):
    m1, m2, s2, s4 = _params(m, dist)
    u_i = build_u_init(chain[0], chain[1], m, m1, m2, s2, s4)
    u_e = build_u_end(chain[-1], m, m1, m2, s2, s4)
    Ws  = [build_W(P, m, m1, m2, s2, s4) for P in chain[2:-1]]
    return u_i, Ws, u_e


def spec_radius(chain, m, dist='gaussian'):
    _, Ws, _ = _pieces(chain, m, dist)
    P = np.eye(4)
    for W in Ws:
        P = W @ P
    return float(np.max(np.abs(np.linalg.eigvals(P))))


def post_init_norm(chain, m, dist='gaussian'):
    u_i, Ws, _ = _pieces(chain, m, dist)
    v = u_i.copy()
    for W in Ws:
        v = W @ v
    return float(np.linalg.norm(v))


def full_value(chain, m, dist='gaussian'):
    u_i, Ws, u_e = _pieces(chain, m, dist)
    v = u_i.copy()
    for W in Ws:
        v = W @ v
    return float(u_e @ v)


import os

def plot_chain_analysis(chains, m_values, savedir='chain_analysis',
                        dist='gaussian'):
    os.makedirs(savedir, exist_ok=True)
    m_arr = np.asarray(m_values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    for chain in chains:
        y = np.abs([full_value(chain, m, dist) for m in m_values])
        ax.plot(m_arr, y, 'o-', label=' '.join(chain), markersize=4)
    ax.plot(m_arr, 1.0/m_arr,         'k--', alpha=0.7, label='1/m')
    ax.plot(m_arr, 1.0 + 1.0/m_arr,   'k:',  alpha=0.7, label='1 + 1/m')
    ax.set_xlabel('m')
    ax.set_title(r'$u_{\rm end}^{\top}\,(\prod W^{P})\,u_{\rm init}$')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig( 'per_element_analysis.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    chains = [
        ['4K','4K','4K','4K'],
        ['4K','4K','A','4K'],
        ['4K','4K','U1','4K'],
        ['4K','4K','U1','U1','4K'],
        ['4K','4K','U1','4K','U1','4K'],
        ['4K','4K','U1','A','U1','4K'],
        ['4K','4K','U1','U2','4K'],
    ]
    m_vals = np.arange(50,500,10)
    plot_chain_analysis(chains, m_vals)
