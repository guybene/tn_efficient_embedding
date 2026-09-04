"""Figure in the style of mps_mpo_mpop.png:
(a) the general tree operator A: the 5-node tree A_1..A_5 (A_1,A_2,A_3 leaves,
    A_4,A_5 internal), spectrum Lambda on the glue leg, glued to the conjugate
    copy -- drawn as an actual branching tree,
(b) the dimension tree of the (reshaped) dangling legs with one sketch core
    per internal node (no dimension labels, per def:dimtree)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(10.5, 8.0))
ax.set_aspect('equal')
ax.axis('off')

BLUE = '#5494EC'          # sampled from the user's mpo_mpop.png
R = 0.36                  # node radius
LEG = 0.62                # dangling-leg length

def node(x, y, label, fs=11, r=R):
    ax.add_patch(Circle((x, y), r, facecolor=BLUE, edgecolor='black',
                        lw=1.2, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, zorder=4)

def bond(p, q, label=None, loff=(0.20, 0.0), fs=11, rp=R, rq=R):
    """Edge between node centers p, q, trimmed at the circles."""
    import math
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    x1, y1 = p[0] + ux * rp, p[1] + uy * rp
    x2, y2 = q[0] - ux * rq, q[1] - uy * rq
    ax.plot([x1, x2], [y1, y2], color='black', lw=1.2, zorder=1)
    if label:
        ax.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label,
                ha='left', va='center', fontsize=fs, zorder=4)

def leg(x, y, dx, dy, label=None, fs=11):
    ax.plot([x, x + dx], [y, y + dy], color='black', lw=1.2, zorder=1)
    if label:
        ax.text(x + dx * 1.35, y + dy * 1.35, label, ha='center',
                va='center', fontsize=fs, zorder=4)

def caption(x, y, s):
    ax.text(x, y, s, ha='center', va='center', fontsize=12, zorder=4)

# ---------------- (a) the operator A, drawn as a tree ----------------------
# leaves A1 (two legs), A2, A3 on top; internal A4; the CENTER node A5 = the
# two copies' internal cores contracted with diag(lambda) on the glue (built
# at data creation, discussed in the text); mirrored conjugate copy below.
P = {'A1': (0.8, 4.2), 'A2': (2.4, 4.2), 'A3': (4.0, 4.2),
     'A4': (1.6, 2.85), 'A5': (2.8, 1.5)}
Pc = {k: (v[0], -v[1] + 3.0) for k, v in P.items() if k != 'A5'}  # mirror

node(*P['A1'], r'$\mathcal{A}_1$')
node(*P['A2'], r'$\mathcal{A}_2$')
node(*P['A3'], r'$\mathcal{A}_3$')
node(*P['A4'], r'$\mathcal{A}_4$')
node(*P['A5'], r'$\mathcal{A}_5$')
node(*Pc['A4'], r'$\mathcal{A}_4^{*}$')
node(*Pc['A1'], r'$\mathcal{A}_1^{*}$')
node(*Pc['A2'], r'$\mathcal{A}_2^{*}$')
node(*Pc['A3'], r'$\mathcal{A}_3^{*}$')

bond(P['A1'], P['A4'], r'$\chi$', loff=(-0.48, 0.02))
bond(P['A2'], P['A4'], r'$\chi$', loff=(0.16, 0.02))
bond(P['A4'], P['A5'], r'$\chi$', loff=(-0.48, 0.02))
bond(P['A3'], P['A5'], r'$\chi$', loff=(0.16, 0.02))
bond(Pc['A4'], P['A5'], r'$\chi$', loff=(-0.48, 0.02))
bond(Pc['A3'], P['A5'], r'$\chi$', loff=(0.16, 0.02))
bond(Pc['A1'], Pc['A4'], r'$\chi$', loff=(-0.48, 0.02))
bond(Pc['A2'], Pc['A4'], r'$\chi$', loff=(0.16, 0.02))

# dangling legs: two on A1, one on A2, one on A3 (mirrored below)
leg(P['A1'][0] - 0.22, P['A1'][1] + R - 0.06, -0.42, 0.50, r'$d$')
leg(P['A1'][0] + 0.22, P['A1'][1] + R - 0.06, 0.42, 0.50, r'$d$')
leg(P['A2'][0], P['A2'][1] + R, 0, LEG, r'$d$')
leg(P['A3'][0], P['A3'][1] + R, 0, LEG, r'$d$')
leg(Pc['A1'][0] - 0.22, Pc['A1'][1] - R + 0.06, -0.42, -0.50, r'$d$')
leg(Pc['A1'][0] + 0.22, Pc['A1'][1] - R + 0.06, 0.42, -0.50, r'$d$')
leg(Pc['A2'][0], Pc['A2'][1] - R, 0, -LEG, r'$d$')
leg(Pc['A3'][0], Pc['A3'][1] - R, 0, -LEG, r'$d$')

caption(2.4, -4.0, r'$\mathrm{(a)}$')

# ---------------- (b) the dimension tree (no dimension labels) -------------
DX0 = 6.9
yl, ym, yr = 0.2, 1.7, 3.2          # leaves / internal / root rows
xe12, xe3, xe4 = DX0, DX0 + 1.5, DX0 + 3.0
xs1 = (xe12 + xe3) / 2
xs2 = xs1 + 1.125

node(xe12, yl, r'$e_{1}e_{2}$', fs=10)
node(xe3, yl, r'$e_{3}$')
node(xe4, yl, r'$e_{4}$')
node(xs1, ym, r'$\mathcal{S}_{t_1}$', fs=10)
node(xs2, yr, r'$\mathcal{S}_{t_2}$', fs=10)

bond((xe12, yl), (xs1, ym))
bond((xe3, yl), (xs1, ym))
bond((xs1, ym), (xs2, yr))
bond((xe4, yl), (xs2, yr))

caption(DX0 + 1.5, -4.0, r'$\mathrm{(b)}$')

ax.set_xlim(-0.4, 10.6)
ax.set_ylim(-4.45, 5.5)
plt.tight_layout()
plt.savefig('general_tree_op.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print('saved general_tree_op.png')
