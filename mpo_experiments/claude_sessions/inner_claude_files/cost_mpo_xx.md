# Per-sample contraction cost of the six traced MPS (MPO-case) sketches

Exact cost of ONE estimate of `Tr(rho_A^K)` when the operator is a **traced MPS**
(`XX_MPS` in `sketches.py`): `rho_A = C E C^H`, with `C` = the ket A-part of the MPS
(`NA` cores `B_l` of shape `(D_{l-1}, d, D_l)`) and `E` = the `chi x chi` environment
Gram at the cut.  Companion to `cost_per_algorithm.md` (the bond-1 chain case); this file
is the **MPO case with a real bond dimension** — every formula is a function of the bond
profile

```
bonds = [D_0 = 1, D_1, ..., D_NA = chi]     (D_l = right bond of site l)
```

Every formula below is the literal `cost(K, bonds)` of the corresponding class in
`sketches.py`, **validated integer-exact against the `_ec` op counter**
(`factored_cost`) by `test_sketches.check_cost_traced MPS`, on

* a **uniform D=2** traced MPS (random state, `chi_max=2`; bonds `[1,2,...,2]`), and
* the **real XX operator**, whose bonds vary site to site (e.g. `[1,2,4,8,16]` at NA=4),

for every sketch, `m in {4,6}`, `K in {1,3}`.

## Setup

Every sketch computes `V = S_bra C in F^{rows x chi}` by ONE left-to-right sweep over
the A-part cores (the environment leg is carried through), then the shared estimator
(`estimate_factored`) chains the copies:

| # | contraction | what it does | cost |
|---|---|---|---|
| E1 | `nre,ef->nrf` | apply the environment Gram `E` to `V_r` | `rows.chi^2` |
| E2 | `nrf,nsf->nrs` | `M_r = (V_r E) V_{r+1}^H` | `rows^2.chi` |
| E3 | `nij,njk->nik` (x K-1) | chain the `M_r` | `(K-1).rows^3` |
| E4 | `nii->n` | final trace | `rows` |

> **overhead(K) = `K.(rows.chi^2 + rows^2.chi) + (K-1).rows^3 + rows`**
> **total = `K.(one V)` + overhead(K)**  — scalar sketches (`rows=1`): overhead
> `= K.(chi^2 + chi) + (K-1) + 1`.

`d` = physical dim (2), `m` = sketch parameter, `K` = power.  Sums run over sites in
1-based labels; `D_l = bonds[l]`.

## 1. `kron` / `kron_pauli` — product probe (rows = 1)

One d-vector per site (`kron_pauli`: the complex partial-basis table; identical op
count — the counter is dtype-blind).

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `nd,dy->ny` | site 1 (left bond 1) | `d.D_1` |
| 2 | `nx,xdy->ndy` | apply core `B_l`, l = 2..NA | `D_{l-1}.d.D_l` |
| 3 | `nd,ndy->ny` | dot `g_l` | `d.D_l` |

> **one V = `d.D_1 + sum_{l=2}^{NA} (D_{l-1}.d.D_l + d.D_l)`**
> uniform D: `O(K NA d D^2)` — `O(1)` in `m`.

## 2. `tt` / `tt_capped` — rMPS probe, bond R (rows = 1)

`tt`: `R = m` (unbounded); `tt_capped`: `R = min(m, NA)` (Schmidt limit).

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `ndr,dy->nry` | core 1 (left bond 1) | `d.R.D_1` |
| 2 | `nrx,xdy->nrdy` | apply core `B_l`, l = 2..NA-1 | `R.D_{l-1}.d.D_l` |
| 3 | `nrdy,nrds->nsy` | absorb probe core `c_l` | `R^2.d.D_l` |
| 4 | `nrx,xdy->nrdy` | apply last core | `R.D_{NA-1}.d.D_NA` |
| 5 | `nrdy,nrd->ny` | absorb last probe core | `R.d.D_NA` |

> **one V = `d.R.D_1 + sum_{l=2}^{NA-1}(R.D_{l-1}.d.D_l + R^2.d.D_l) + R.D_{NA-1}.d.D_NA + R.d.D_NA`**
> uniform D: `O(K NA d R (R + D) D)` — `O(m^2)` in `m` for `tt`, flat for `tt_capped`
> once `m >= NA`.

## 3. `ttm` — alg:mpop's sketch (rows = m), DIRECTION-ADAPTIVE sweep

The sweep direction matters (found by `check_optimal_factored`): **left-to-right** drags
the size-`m` output leg `mu` through every site (`~m^3.d.D` per site); **right-to-left**
instead carries the open cut leg `chi` (`~m^2.d.D.chi` per site).  `V` executes
whichever closed form is cheaper for the given bond profile — for the experiments
(`m >= chi`) that is right-to-left.  `cost = K.min(LR, RL) + overhead`.

LR (output leg carried):

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `nudr,dy->nury` | core 1 | `m^2.d.D_1` |
| 2 | `nurx,xdy->nurdy` | apply core `B_l`, l = 2..NA-1 | `m^2.D_{l-1}.d.D_l` |
| 3 | `nurdy,nrds->nusy` | absorb sketch core `c_l` | `m^3.d.D_l` |
| 4 | `nurx,xdy->nurdy` | apply last core | `m^2.D_{NA-1}.d.D_NA` |
| 5 | `nurdy,nrd->nuy` | absorb last sketch core | `m^2.d.D_NA` |

> **LR = `m^2.d.D_1 + sum_{l=2}^{NA-1}(m^2.D_{l-1}.d.D_l + m^3.d.D_l) + m^2.D_{NA-1}.d.D_NA + m^2.d.D_NA`** — `O(m^3)` in `m`.

RL (cut leg carried; `chi = D_NA`):

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `nrd,xdy->nrxy` | last core first | `m.d.D_{NA-1}.chi` |
| 2 | `nrxy,wdx->nrwdy` | apply core `B_l`, l = NA-1..2 | `m.D_{l-1}.d.D_l.chi` |
| 3 | `nrwdy,nsdr->nswy` | absorb sketch core `c_l` | `m^2.D_{l-1}.d.chi` |
| 4 | `nrxy,dx->nrdy` | core 1 | `m.d.D_1.chi` |
| 5 | `nrdy,nudr->nuy` | absorb sketch core 1 (the `mu` leg appears only here) | `m^2.d.chi` |

> **RL = `m.d.D_{NA-1}.chi + sum_{l=2}^{NA-1}(m.D_{l-1}.d.D_l.chi + m^2.D_{l-1}.d.chi) + m.d.D_1.chi + m^2.d.chi`** — `O(m^2.chi)` in `m`.

## 4. `ma` — alg:mps's sketch = Ma Alg. 1 (rows = m)

Kronecker layer + gadget pairs `g_a in R^{m x s1 x m}`, `g_b in R^{m x s2 x m}`,
`s1.s2 = m` (`s1` = the divisor of `m` nearest `sqrt(m/chi)` — Ma's cost-optimal split,
feasible for every `m`; `E[S^T S] = I` holds for ANY split).

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `ndu,dy->nuy` | Kron-sketch site 1 | `d.m.D_1` |
| 2 | `ndu,xdy->nxuy` | Kron-sketch site i, i = 2..NA | `d.m.D_{i-1}.D_i` |
| 3 | `nux,nual->nalx` | absorb `g_a` | `m^2.s1.D_{i-1}` |
| 4 | `nalx,nxaby->nlby` | tie the (s1,s2)-split site | `m^2.D_{i-1}.D_i` |
| 5 | `nlby,nlbv->nvy` | absorb `g_b` | `m^2.s2.D_i` |

> **one V = `d.m.D_1 + sum_{i=2}^{NA}(d.m.D_{i-1}.D_i + m^2.s1.D_{i-1} + m^2.D_{i-1}.D_i + m^2.s2.D_i)`**
> uniform D, balanced split: `O(K NA (m^2.5 sqrt(D) + m^2 D^2) D/sqrt(D)... )` — leading
> `m`-order `O(m^2 (s1 + s2)) = O(m^2.5)` at the balanced split (plus the `K m^3` chain
> from the overhead, which dominates for `K >= 2` at large `m`).

## Verification

Two independent tests (both in `test_sketches.py`):

1. **`check_cost_traced MPS`** — formula == `_ec` counter, exact integers, uniform D=2 and
   XX bonds `[1,2,4,8,16]`, all sketches, `m in {4,6}`, `K in {1,3}`.
2. **`check_optimal_factored`** — the measured count of `V` vs the opt_einsum-optimal
   path of the same network (same cost convention), in the dominant-ambient regime
   (`d^NA >> ` sketch intermediates).  Ratios: `kron`/`kron_pauli`/`ma`/tree-`tt`
   **1.000** (exactly optimal), xx-`tt` 1.031, `ttm` 1.036, tree-ours 1.020 (boundary
   constants, not scaling).  A ratio < 1 is impossible unless the tally UNDER-counts —
   so this test also proves no contraction escapes `_ec`.  This test caught and fixed
   two genuinely suboptimal first-draft orders: `ttm`'s output-leg-dragging LR sweep
   (now adaptive, above) and tree-`tt`'s mid-sweep hit of the 4-leg internal block.

Notes:
- The chain-side (identity/all_ones, bond-1 MPO) formulas remain in
  `cost_per_algorithm.md`, validated by `check_cost`; this file is their bond-D
  counterpart for the traced MPS representation.  A bond-D MPO chain sweep (merged
  `chi^2` cores) is deliberately NOT implemented — the traced MPS is the cheaper, exact
  representation of the same operator.
- Tree-sketch (exp_general_tn) costs are the measured `_ec` counts (`factored_cost`);
  they have no hand formulas — the topology is a single fixed 5-node tree.
