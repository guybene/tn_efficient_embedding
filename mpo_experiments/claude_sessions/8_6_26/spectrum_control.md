# Controlling the spectrum of the tree operator

Correction first: you do **not** need a large $\chi$ — the condition is
$\chi \le d$ (bond dimension at most the physical dimension). If you do not
care about $\chi$, set $\chi = d$ and forget it.

## Step 1: why $\lambda$ on the glue leg = the spectrum

The half-tree is a map $H:\mathbb{C}^{g}\to\mathbb{C}^{d^{4}}$ (glue leg in,
physical legs out), and the operator we build is

$$
A \;=\; H\,\Lambda\,H^{H},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_g).
$$

Suppose the columns $h_1,\dots,h_g$ of $H$ are **orthonormal**, i.e.
$H^{H}H=I_g$. Then for each column:

$$
A\,h_i \;=\; H\Lambda\,(H^{H}h_i) \;=\; H\Lambda\,e_i \;=\; \lambda_i\,h_i .
$$

So each $h_i$ is literally an eigenvector with eigenvalue $\lambda_i$, and
every direction orthogonal to all of them is killed (eigenvalue $0$). In other
words, $A=H\Lambda H^{H}$ **is** an eigendecomposition — the numbers you typed
are the eigenvalues, exactly:

- rank $=$ number of nonzero $\lambda_i$;
- $\rho = \sum_i \lambda_i / \max_i \lambda_i$, chosen directly.

If instead $H^{H}H\neq I$, the eigenvalues of $A$ are those of
$\Lambda^{1/2}\,(H^{H}H)\,\Lambda^{1/2}$ — distorted by the Gram matrix of
$H$'s columns, i.e. random junk again.

## Step 2: why $\chi \le d$ decides whether $H^{H}H = I$

$H$ is not one matrix we QR — it is a **chain** of small cores. Each core is
QR-orthonormalized at build, and a chain of inner-product-preserving maps
preserves inner products overall. But a linear map can only preserve inner
products when it goes from a *smaller* space into a *bigger* one: you cannot
place $\chi$ orthonormal vectors inside a $d$-dimensional space when
$\chi > d$.

Where each core maps (input $\to$ output):

| core | map | condition |
|------|-----------------|-----------|
| I2 | $g \to \chi^2$ | $g \le \chi^2$ |
| I1 | $\chi \to \chi^2$ | always fine |
| T1 | $\chi \to d^2$ | $\chi \le d^2$ |
| **T2, T3** | $\chi \to d$ | $\chi \le d$ — **the bottleneck** |

With $d=2$, $\chi=3$ (the old default), the cores T2/T3 squeeze $3$
directions into a $2$-dimensional space — orthonormality is impossible there,
the chain distorts, and the $\lambda$'s are only approximate. With
$\chi = d = 2$ every link preserves inner products, $H^{H}H=I$ exactly, and
Step 1 applies.

Numerical check (already run): requested spectrum $(1,\,0.5,\,0.25,\,0)$ at
$d=2,\ \chi=2,\ g=4$ — measured eigenvalues $(1,\,0.5,\,0.25,\,0)$ to
$10^{-12}$; rank $=3$, $\rho = 1.75 = \sum\lambda/\lambda_{\max}$.

## How $H$ gets orthonormal columns without ever QR-ing $H$

We never orthonormalize $H$ itself — each small core is QR-orthonormalized at
build, and orthonormality survives contraction automatically, by two one-line
facts:

1. **Composition.** If $U^{H}U=I$ and $V^{H}V=I$, then
   $(UV)^{H}(UV) = V^{H}U^{H}UV = V^{H}V = I$.
2. **Tensor product.** If $A^{H}A=I$ and $B^{H}B=I$, then
   $(A\otimes B)^{H}(A\otimes B) = (A^{H}A)\otimes(B^{H}B) = I$.

Naming note: $I1$ and $I2$ are **not** identities — they are the two
*Internal* nodes of the tree (random QR'd cores with no dangling physical
leg): $I1$ maps $z\to x\otimes y$ ($\chi^2\times\chi$, orthonormal columns)
and $I2$ maps $g\to z\otimes w$ ($\chi^2\times g$). The only identity below is
$I_d$, the $d\times d$ identity on an untouched leg.

Why does $H$ factor into these operations at all? Two rules turn any network
contraction into matrix algebra:

- **Rule A (parallel = Kronecker).** Tensors acting on *different* legs — not
  connected to each other — combine as a Kronecker product. This is the
  definition of $\otimes$ read element-wise,
  $(A\otimes C)_{(x,e),(z,w)} = A_{x,z}\,C_{e,w}$:

$$
\sum_{z,w} A_{x,z}\,C_{e,w}\,B_{(z,w),g}
= \sum_{z,w} (A\otimes C)_{(x,e),(z,w)}\,B_{(z,w),g}
= \big[(A\otimes C)\,B\big]_{(x,e),g}.
$$

- **Rule B (series = matrix product).** Connecting an output leg to an input
  leg (= summing the shared index) is matrix multiplication. An untouched leg
  carries the identity matrix.

Reading the tree from the glue leg outward,

$$
H_{(e_1e_2e_3e_4),\,g}
=\sum_{x,y,z,w} T_{1,(e_1e_2),x}\,T_{2,e_3,y}\,I1_{(x,y),z}\;T_{3,e_4,w}\;I2_{(z,w),g},
$$

layer 1 is $I2:g\to(z,w)$; layer 2 applies $I1$ on leg $z$ and $T_3$ on leg
$w$ in parallel (Rule A) onto $I2$'s outputs (Rule B); layer 3 applies $T_1$
on $x$, $T_2$ on $y$, identity on the untouched $e_4$. Hence:

$$
H \;=\; \underbrace{(T_1\otimes T_2\otimes I_d)}_{\text{layer 3:}\ x\to(e_1,e_2),\ y\to e_3}
\;\underbrace{(I1\otimes T_3)}_{\text{layer 2:}\ z\to(x,y),\ w\to e_4}
\;\underbrace{I2}_{\text{layer 1:}\ g\to(z,w)} .
$$

Legs not touched at a stage carry an identity factor, itself trivially an
isometry. Each core individually is a small matrix: `draw()` reshapes it to
(all output dims) $\times$ (input dim) and QRs it, giving orthonormal columns
whenever rows $\ge$ columns — which is precisely where $\chi\le d$ enters for
$T_2,T_3$. By facts 1+2 every stage is an isometry, so $H^{H}H=I$ without
anyone ever forming the $d^{4}\times g$ matrix.

Check it yourself with `TreeOp.dense_half()`, which contracts one copy of the
tree with the glue leg open and returns the full $d^{4}\times g$ matrix $H$:

```python
op = TreeOp(d=2, chi=2, g=4, seed=1)
H = op.dense_half()
(H.conj().T @ H - torch.eye(4, dtype=H.dtype)).abs().max()
```

Measured: $\max|H^{H}H-I| = 1.3\cdot10^{-15}$ at $\chi=2\le d$ (orthonormal to
machine precision), versus $0.70$ at $\chi=3>d$ — the bottleneck condition in
action.

## Summary

$$
\boxed{\;\lambda \text{ on the glue leg } = \text{candidate eigenvalues};
\qquad
\chi \le d \;(\text{and } g \le \chi^2) = \text{the condition under which the
tree does not distort them.}\;}
$$
