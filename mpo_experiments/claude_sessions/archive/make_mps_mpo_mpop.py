"""Figure in the style of mpo_mpop.png, with an added left panel:
(a) psi psi^* as two hovering MPS layers (B_i on top, B_i^* below),
(b) an MPO, (c) the MPO power (MPOP)."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams['font.size'] = 12
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(15, 7.0))
ax.set_aspect('equal')
ax.axis('off')

BLUE = '#5494EC'          # sampled from the user's mpo_mpop.png
R = 0.36                  # node radius
DX = 1.30                 # node spacing
LEG = 0.72                # dangling-leg length

def node(x, y, label, fs=11):
    ax.add_patch(Circle((x, y), R, facecolor=BLUE, edgecolor='black',
                        lw=1.2, zorder=3))
    ax.text(x, y, label, ha='center', va='center', fontsize=fs, zorder=4)

def hbond(x1, x2, y, label, loff=0.20):
    ax.plot([x1, x2], [y, y], color='black', lw=1.2, zorder=1)
    ax.text((x1 + x2) / 2, y + loff, label, ha='center', va='bottom',
            fontsize=11, zorder=4)

def vleg(x, y0, y1, label=None, side=-0.16, lpos='end'):
    ax.plot([x, x], [y0, y1], color='black', lw=1.2, zorder=1)
    if label:
        ly = y1 if lpos == 'end' else (y0 + y1) / 2
        va = 'bottom' if y1 > y0 else 'top'
        if lpos == 'mid':
            va = 'center'
        ax.text(x + side, ly, label, ha='right', va=va, fontsize=11, zorder=4)

def dots(x, y, s=r'$\cdots$', fs=13):
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, zorder=4)

def row(x0, y, labels, bond=r'$D$'):
    """Nodes labels[0], labels[1], cdots, labels[2]; every segment labeled.
    Returns the x positions [x1, x2, xdots, xN]."""
    xs = [x0, x0 + DX, x0 + 2 * DX, x0 + 3 * DX]
    node(xs[0], y, labels[0])
    node(xs[1], y, labels[1])
    dots(xs[2], y)
    node(xs[3], y, labels[2])
    hbond(xs[0] + R, xs[1] - R, y, bond)
    hbond(xs[1] + R, xs[2] - 0.28, y, bond)
    hbond(xs[2] + 0.28, xs[3] - R, y, bond)
    return xs

def caption(x, y, s):
    ax.text(x, y, s, ha='center', va='center', fontsize=12, zorder=4)

# ---------------- (a) psi psi^*: two hovering MPS layers -------------------
ax0 = 0.55
ya_top, ya_bot = 3.55, 1.75
xs = row(ax0, ya_top, [r'$\mathcal{B}_1$', r'$\mathcal{B}_2$', r'$\mathcal{B}_N$'],
         bond=r'$\chi$')
for x in (xs[0], xs[1], xs[3]):
    vleg(x, ya_top + R, ya_top + R + LEG, r'$d$')
xs = row(ax0, ya_bot, [r'$\mathcal{B}_1^{*}$', r'$\mathcal{B}_2^{*}$', r'$\mathcal{B}_N^{*}$'],
         bond=r'$\chi$')
for x in (xs[0], xs[1], xs[3]):
    vleg(x, ya_bot - R, ya_bot - R - LEG, r'$d$')
caption(ax0 + 1.5 * DX, -1.05, r'$\mathrm{(a)}$')

# ---------------- (b) MPO --------------------------------------------------
bx0 = 6.05
yb = 2.65
xs = row(bx0, yb, [r'$\mathcal{A}_1$', r'$\mathcal{A}_2$', r'$\mathcal{A}_N$'])
for x in (xs[0], xs[1], xs[3]):
    vleg(x, yb + R, yb + R + LEG, r'$d$')
    vleg(x, yb - R, yb - R - LEG, r'$d$')
caption(bx0 + 1.5 * DX, -1.05, r'$\mathrm{(b)}$')

# ---------------- (c) MPOP -------------------------------------------------
cx0 = 11.55
y1, y2, yk = 4.75, 3.20, 0.45   # rows (1), (2), vdots, (k)
xs = row(cx0, y1, [r'$\mathcal{A}_1^{(1)}$', r'$\mathcal{A}_2^{(1)}$', r'$\mathcal{A}_N^{(1)}$'])
for x in (xs[0], xs[1], xs[3]):
    vleg(x, y1 + R, y1 + R + LEG, r'$d$')                      # top dangling
row(cx0, y2, [r'$\mathcal{A}_1^{(2)}$', r'$\mathcal{A}_2^{(2)}$', r'$\mathcal{A}_N^{(2)}$'])
for x in (xs[0], xs[1], xs[3]):
    vleg(x, y1 - R, y2 + R, r'$d$', lpos='mid')                # rows 1-2 contracted
    vleg(x, y2 - R, y2 - R - 0.55, r'$d$')                     # stubs below row 2
    dots(x, (y2 - R - 0.55 + yk + R + 0.55) / 2, r'$\vdots$', fs=13)
    vleg(x, yk + R, yk + R + 0.55, r'$d$')                     # stubs above row k
row(cx0, yk, [r'$\mathcal{A}_1^{(k)}$', r'$\mathcal{A}_2^{(k)}$', r'$\mathcal{A}_N^{(k)}$'])
for x in (xs[0], xs[1], xs[3]):
    vleg(x, yk - R, yk - R - LEG, r'$d$')                      # bottom dangling
caption(cx0 + 1.5 * DX, -1.05, r'$\mathrm{(c)}$')

ax.set_xlim(-0.3, 16.1)
ax.set_ylim(-1.5, 6.25)
plt.tight_layout()
plt.savefig('mps_mpo_mpop.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print('saved mps_mpo_mpop.png')
