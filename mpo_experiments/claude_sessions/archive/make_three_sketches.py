"""Three sketching tensor networks side by side: TT (ours), Solomonik, Kronecker."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_aspect('equal')
ax.axis('off')

BLUE = '#bcd6f0'    # dense core
YEL = '#ffd24d'    # split-gadget core

def core(x, y, r=0.20, fc=BLUE, ec='black', lw=1.6, z=3):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

def leg(x1, y1, x2, y2, lw=1.6, z=1):
    ax.plot([x1, x2], [y1, y2], color='black', lw=lw, zorder=z)

def lbl(x, y, s, fs=12, **kw):
    ax.text(x, y, s, ha='center', va='center', fontsize=fs, zorder=4, **kw)

# ================= Panel 1: TT (ours) -- caterpillar tree =================
# data legs (bottom) feed a left-to-right cascade of dense merge nodes.
d = [(0.25, 0.4), (0.95, 0.4), (1.5, 0.4), (2.4, 0.4)]
n1, n2, n3 = (0.6, 1.5), (1.5, 2.5), (2.4, 3.5)
for (x, y) in d:
    leg(x, y, x, 0.15); lbl(x, -0.05, r'$d$', fs=11)
# merges: n1 = merge(d1,d2); n2 = merge(n1,d3); n3 = merge(n2,d4)
leg(*d[0], *n1); leg(*d[1], *n1)
leg(*d[2], *n2)
leg(*d[3], *n3)
leg(*n1, *n2); leg(*n2, *n3)
for p in (n1, n2, n3):
    core(*p, fc=BLUE)
leg(n3[0], n3[1], n3[0], 4.3); lbl(n3[0], 4.5, r'$m$', fs=11)   # output leg
lbl(1.4, -0.7, 'TT (ours)', fs=15, fontweight='bold')

# ================= Panel 2: Solomonik -- Kronecker layer + split-gadget tree =
OFF = 5.0
kx = [OFF+0.3, OFF+1.1, OFF+1.9, OFF+2.7]; ky = 1.2
for x in kx:
    leg(x, ky, x, 0.4); lbl(x, -0.05, r'$d$', fs=11)   # data legs
    core(x, ky, fc=BLUE)                               # Kronecker cores (dense)

def gadget(x, y):
    """split gadget = two yellow cores connected by a bond; returns (in, out) points."""
    lx, rx = x-0.18, x+0.18
    core(lx, y, r=0.15, fc=YEL); core(rx, y, r=0.15, fc=YEL)
    leg(lx, y, rx, y)
    return (lx, y), (rx, y)

gL_in, gL_out = gadget(OFF+0.7, 2.4)
gR_in, gR_out = gadget(OFF+2.3, 2.4)
leg(kx[0], ky, *gL_in); leg(kx[1], ky, *gL_in)
leg(kx[2], ky, *gR_in); leg(kx[3], ky, *gR_in)
r_in, r_out = gadget(OFF+1.5, 3.6)
leg(*gL_out, *r_in); leg(*gR_out, *r_in)
leg(r_out[0], r_out[1], r_out[0], 4.4); lbl(r_out[0], 4.6, r'$m$', fs=11)  # output
lbl(OFF+1.5, -0.7, 'Solomonik', fs=15, fontweight='bold')

# ================= Panel 3: Kronecker -- independent cores ================
OFF2 = 10.0
kkx = [OFF2+0.2, OFF2+1.0, OFF2+1.8, OFF2+2.6]; kkc = 2.2
for x in kkx:
    core(x, kkc, fc=BLUE)
    leg(x, kkc, x, 1.3); lbl(x, 1.05, r'$d$', fs=11)     # data leg
    leg(x, kkc, x, 3.1); lbl(x, 3.35, r'$m$', fs=11)     # output leg
lbl(OFF2+1.4, -0.7, 'Kronecker', fs=15, fontweight='bold')

# ================= legend =================
core(0.1, -1.35, r=0.13, fc=BLUE); ax.text(0.35, -1.35, 'dense core', fontsize=10, va='center', zorder=4)
core(3.4, -1.35, r=0.13, fc=YEL);  ax.text(3.65, -1.35, 'split-gadget core', fontsize=10, va='center', zorder=4)

ax.set_xlim(-0.6, 13.2)
ax.set_ylim(-1.8, 5.0)
plt.tight_layout()
plt.savefig('three_sketches.png', dpi=150, bbox_inches='tight')
print('saved three_sketches.png')
