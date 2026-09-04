"""Six-panel walkthrough of alg:trace on a 7-node star tree (single internal
node A_4, row leaves A_1..A_3, column leaves A_5..A_7, one dangling edge per
leaf).  House style (mps_mpo_mpop.png) plus three marker colors:
green = operands of the next contraction, yellow = sketch cores,
red = contracted intermediates (unlabeled except the final SAS^T)."""
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'cm'

BLUE, GREEN, YELLOW, RED = '#5494EC', '#7EC87E', '#F2C94C', '#E06666'
R = 0.34

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for ax in axes:
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.45, 3.1)


def node(ax, x, y, label, color=BLUE, r=R, fs=11):
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor='black',
                        lw=1.2, zorder=3))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, zorder=4)


def bond(ax, p, q, label=None, rp=R, rq=R, loff=(0.18, 0.05), fs=11):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    x1, y1 = p[0] + ux * rp, p[1] + uy * rp
    x2, y2 = q[0] - ux * rq, q[1] - uy * rq
    ax.plot([x1, x2], [y1, y2], color='black', lw=1.2, zorder=1)
    if label:
        ax.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label,
                ha='left', va='center', fontsize=fs, zorder=4)


def leg(ax, p, ang_deg, label=None, r0=R, length=0.62, fs=11):
    a = math.radians(ang_deg)
    x1, y1 = p[0] + r0 * math.cos(a), p[1] + r0 * math.sin(a)
    x2, y2 = p[0] + (r0 + length) * math.cos(a), p[1] + (r0 + length) * math.sin(a)
    ax.plot([x1, x2], [y1, y2], color='black', lw=1.2, zorder=1)
    if label:
        ax.text(p[0] + (r0 + length + 0.28) * math.cos(a),
                p[1] + (r0 + length + 0.28) * math.sin(a), label,
                ha='center', va='center', fontsize=fs, zorder=4)


def panel_label(ax, s):
    ax.text(0, -3.2, s, ha='center', va='center', fontsize=13)


# shared positions: star with center A4, rows up, columns down
P = {1: (-1.5, 1.35), 2: (0.0, 1.7), 3: (1.5, 1.35), 4: (0.0, 0.0),
     5: (-1.5, -1.35), 6: (0.0, -1.7), 7: (1.5, -1.35)}
LEG_ANG = {1: 135, 2: 90, 3: 45, 5: 225, 6: 270, 7: 315}


def draw_leaves(ax, ids, colors=None):
    for i in ids:
        node(ax, *P[i], rf'$\mathcal{{A}}_{i}$',
             color=(colors or {}).get(i, BLUE))
        leg(ax, P[i], LEG_ANG[i], r'$d$')


# ---------------- (a) the 7-node tree --------------------------------------
ax = axes[0]
draw_leaves(ax, [1, 2, 3, 5, 6, 7], colors={1: GREEN})
node(ax, *P[4], r'$\mathcal{A}_4$', color=GREEN)
for i in [1, 2, 3, 5, 6, 7]:
    bond(ax, P[4], P[i], r'$D$')
panel_label(ax, r'$\mathrm{(a)}$')

# ---------------- (b) contract (A_1, A_4), reshape -------------------------
ax = axes[1]
draw_leaves(ax, [2, 3, 5, 6, 7])
node(ax, *P[4], r'$\{\mathcal{A}_1\mathcal{A}_4\}$', r=0.52, fs=8)
leg(ax, P[4], 160, r'$d$', r0=0.46, length=0.85)
for i in [2, 3, 5, 6, 7]:
    bond(ax, P[4], P[i], r'$D$', rp=0.46)
panel_label(ax, r'$\mathrm{(b)}$')

# ---------------- (c) mark the operands, add the sketch core S_1 -----------
ax = axes[2]
draw_leaves(ax, [3, 5, 6, 7])
node(ax, *P[4], r'$\{\mathcal{A}_1\mathcal{A}_4\}$', color=GREEN, r=0.52, fs=8)
node(ax, *P[2], r'$\mathcal{A}_2$', color=GREEN)
S1 = (-1.4, 1.5)
node(ax, *S1, r'$\mathcal{S}_1$', color=YELLOW)
bond(ax, P[4], S1, r'$d$', rp=0.46, loff=(-0.42, 0.05))
bond(ax, P[2], S1, r'$d$', loff=(0.02, 0.26))
leg(ax, S1, 125, r'$m$')
bond(ax, P[4], P[2], r'$D$', rp=0.46)
for i in [3, 5, 6, 7]:
    bond(ax, P[4], P[i], r'$D$', rp=0.46)
panel_label(ax, r'$\mathrm{(c)}$')

# ---------------- (d) the contracted intermediate (red, unlabeled) ---------
ax = axes[3]
draw_leaves(ax, [5, 6, 7])
node(ax, *P[3], r'$\mathcal{A}_3$')
node(ax, *P[4], None, color=RED, r=0.46)
S2 = (0.0, 1.7)
node(ax, *S2, r'$\mathcal{S}_2$', color=YELLOW)
bond(ax, P[4], S2, r'$m$', rp=0.46, loff=(-0.45, 0.05))
bond(ax, P[3], S2, r'$d$', loff=(0.02, 0.28))
leg(ax, S2, 125, r'$m$')
for i in [3, 5, 6, 7]:
    bond(ax, P[4], P[i], r'$D$', rp=0.46)
panel_label(ax, r'$\mathrm{(d)}$')

# ---------------- (e) one step before the end ------------------------------
ax = axes[4]
node(ax, *P[4], None, color=RED, r=0.46)
leg(ax, P[4], 90, r'$m$', r0=0.46, length=0.85)          # row output
node(ax, *P[7], r'$\mathcal{A}_7$')
bond(ax, P[4], P[7], r'$D$', rp=0.46)
S2c = (-1.3, -1.5)
node(ax, *S2c, r"$\mathcal{S}_2'$", color=YELLOW)
bond(ax, P[4], S2c, r'$m$', rp=0.46, loff=(-0.42, 0.05))
bond(ax, P[7], S2c, r'$d$', loff=(0.02, -0.30))
leg(ax, S2c, 235, r'$m$')
panel_label(ax, r'$\mathrm{(e)}$')

# ---------------- (f) the final matrix -------------------------------------
ax = axes[5]
node(ax, *P[4], r'$\mathbf{S}\mathbf{A}\mathbf{S}^{\top}$', color=RED,
     r=0.55, fs=11)
leg(ax, P[4], 90, r'$m$', r0=0.55, length=0.85)
leg(ax, P[4], 270, r'$m$', r0=0.55, length=0.85)
panel_label(ax, r'$\mathrm{(f)}$')

plt.tight_layout()
plt.savefig('alg1_walkthrough.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print('saved alg1_walkthrough.png')
