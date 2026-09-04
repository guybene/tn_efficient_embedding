# Per-iteration contraction cost of the five sketches

Cost of estimating `Tr(A^K)` for one draw (one "iteration") with each sketch, contracting `A`
as a **tensor network**. This is the cost plotted on the cost/error x-axis; it is computed
**exactly** (every pairwise contraction summed, `free·contracted·free`) by the `_ec` op counter
in `sketches.py` (`cost_per_iter`), not asymptotically. Every formula below is the literal
`Sketch.cost(K, D)` of `sketches.py`, validated against the counter by `check_cost`.

## Setup

- `A` is a **general MPO** with one `D×d×d×D` core per site, MPO **bond dimension `D`** (the
  full operator is the contraction of the `N` cores). The cost depends only on the contraction
  **shapes**, so it is **A-agnostic** (identical for any MPO of the same bond `D`).
- The **bond `D` threads the contraction**: a term picks up
  - **`D²`** where an MPO **core is applied** (the core carries two `D`-bonds), and
  - **`D¹`** where the bond is merely **carried through** the running environment.
- **`D` is kept symbolic below.** Our two test operators — `identity` (`I`) and `all_ones`
  (`J/d^N`) — are **bond-1 MPOs, so `D = 1`** and all `D`-powers collapse to 1. That `D=1` is a
  property of *our* data only; the formulas hold for any MPO, and `D>1` is read off directly.
- `d` = physical dim, `N` = sites, `m` = sketch parameter, `K` = power (copies of `A`).
  For `solomonik`, `m = m₁·m₂` is the gadget split (balanced `m₁≈m₂≈√m`).
- `Mᵢ = Sᵢ A Sᵢ₊₁ᵀ`; the estimate is `Tr(M₁ M₂ ⋯ M_K)`.
- **Two output regimes:**
  - **scalar** (`rows = 1`): `kron`, `tt`. Each `Mᵢ` is `1×1`; the chain `∏ᵢ Mᵢ` is a product of
    `K` scalars — the chain costs `(K−1)·1`, negligible.
  - **matrix** (`rows = m`): `ttm`, `solomonik`, `dense`. Each `Mᵢ` is an `m×m` matrix, so
    `Tr(M₁⋯M_K)` is `K−1` matmuls of `m×m` ⇒ a **`(K−1)·m³`** chain term.

So **per-iteration cost = `K · (one Mᵢ)` + chain**, chain `= (K−1)m³` (matrix) or `(K−1)` (scalar).

Each section gives the **EXACT total** (literal closed form of `Sketch.cost`, no approximation)
and, separately, the **asymptotic** — which drops *only numeric constants and additive
dominated terms*, **keeping every parameter** (e.g. `(m²+1)(D+2K) = O(m²(D+K))`).

---

## 1. `kron` — single product vector (rows = 1)

`v = ⊗_l g_l`, `g_l ∈ ℝ^d`; `Mᵢ = ∏_{l=1}^{N} g_{i,l}ᵀ A_l g_{j,l}` (a scalar), swept site by site.

**Per site `l`** (2 contractions):

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `ab,nb->na` | apply core `A_l` to `g_{j,l}` (contract phys-in) | `d²·D²` |
| 2 | `na,na->n`  | dot `g_{i,l}` with it (contract phys-out)        | `d·D²`  |

Per site `d²D² + dD²`, over `N` sites; chain = `K−1` scalar (1×1) products.

> **one `Mᵢ` = `N·d²D² + N·dD²` = `N·d(d+1)·D²`**
> **EXACT total = `K·N·d(d+1)·D² + (K−1)`**   →   **`O(K N d² D²)`**  (`O(1)` in `m`)

(The `N−1` inter-site bonds of size `D` fold into the scalar product, uncounted; the `D²` is the
bulk-core rate — exact at `D=1`, a mild over-count for `D>1`.)

## 2. `tt` — single TT vector, scalar (rows = 1), bond `R = min(m, N)`

`Mᵢ = ⟨vᵢ|A|vⱼ⟩` swept with environment `(Rᵢ, D, Rⱼ)`, internal **bond `R = min(m, N)`** — capped
at `N` because beyond the TT's Schmidt limit the cores are rank-deficient (extra bond gives no
variance reduction, only `O(R³)` cost).  **The cost formula is unchanged**; the bond `m` is simply
replaced by `R = min(m, N)` (so the per-contraction `m`-factors below are this `R`).

**Core `0`** (left bond size 1) — 2 contractions:

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `nidx,de->niex`  | apply MPO core `A_0` (left bond =1 ⇒ single D bond) | `d²D·m` |
| 2 | `niex,niey->nxy` | absorb core `c_0` (phys + size-1 left bond) | `d·D·m²` |

**Each middle core `l = 1 … N−2`** (both MPS bonds `m`) — 3 contractions:

| # | contraction | what it does | cost |
|---|---|---|---|
| 3 | `nij,nidx->njdx` | contract incoming bond `R_i` | `d·D·m³`  |
| 4 | `njdx,de->njex`  | apply MPO core `A_l`          | `d²D²·m²` |
| 5 | `njex,njey->nxy` | contract `R_j` + phys        | `d·D·m³`  |

**Core `N−1`** (right bond size 1) — 3 contractions, cheaper:

| # | contraction | what it does | cost |
|---|---|---|---|
| 6 | `nij,nidx->njdx` | contract `R_i` (right bond =1) | `d·D·m²` |
| 7 | `njdx,de->njex`  | apply MPO core `A_{N−1}` (right bond =1 ⇒ single D bond) | `d²D·m` |
| 8 | `njex,njey->nxy` | contract bond + phys (out =1)  | `d·D·m`  |

Chain = `K−1` scalar products. Summing all sites:

> **one `Mᵢ` = `(d²D·R + dD·R²)` + `(N−2)(2dD·R³ + d²D²·R²)` + `(dD·R² + d²D·R + dD·R)`,  `R = min(m, N)`**
> **EXACT total = `K·[ (N−2)(2dD·R³ + d²D²·R²) + 2d²D·R + 2dD·R² + dD·R ] + (K−1)`,  `R = min(m, N)`**
> →   asymptotic **`O(K N dD·R²(R + dD))`**; since `R ≤ N`, **bounded by `O(K N dD·N²(N + dD))`** — i.e. constant in `m` once `m ≥ N` (`O(K N d·N³)` at `D=1`).

## 3. `ttm` — TT on physical legs + dangling output leg `m` (rows = m)

Same physical-TT sweep, every MPS bond `= m`; the output legs `(μ,ν)` ride core 0 and, contracted
**pairwise**, keep the sweep at `m³` (not `m⁴`).

**Each "full" core** (core 0 and the `N−2` middles — `N−1` cores total) — 3 contractions:

| # | contraction (core 0 variant in parens) | what it does | cost |
|---|---|---|---|
| a | `nij,naxi->najx`  (core 0: `nij,nuxi->nujx`) | contract incoming bond | `d·D·m³`  |
| b | `najx,xy->najy`                              | apply MPO core `A_l`   | `d²D²·m²` |
| c | `najy,ncyj->nac`  (core 0: `nujy,nvyj->nuv`) | contract bond + phys   | `d·D·m³`  |

**Boundary core `N−1`** (right bond size 1) — 2 contractions:

| # | contraction | what it does | cost |
|---|---|---|---|
| d | `naxi,xy->nayi`  | apply MPO core `A_{N−1}` (right bond =1) | `d²D²·m` |
| e | `nayi,ncyi->nac` | absorb core `c_{N−1}`                    | `d·D·m²` |

Chain = `K−1` matmuls of `m×m`. Summing:

> **one `Mᵢ` = `(N−1)(2dD·m³ + d²D²·m²)` + `(d²D²·m + dD·m²)`**
> **EXACT total = `K·[ (N−1)(2dD·m³ + d²D²·m²) + d²D²·m + dD·m² ] + (K−1)·m³`**
> →   **`O(K N dD·m²(m + dD))`**  (`O(K N d·m³)` at `D=1`)

## 4. `solomonik` — gadget k-space chain, explicit halves (rows = m)

Each gadget `g = G₁·G₂` is **never formed**; its two random halves are absorbed one at a time into
the running `m×m` environment `E` (the opt_einsum-optimal order, verified by `check_optimal`).

**Each Kronecker leaf `Y_l = Xᵢ[l]ᵀ A_l Xⱼ[l]`** (`N` of them; bonds split for `l≥1`, `m₁·m₂=m`) —
2 contractions:

| # | contraction (split-bond variant in parens) | what it does | cost |
|---|---|---|---|
| L1 | `nak,ab->nbk`  (split: `npab,pq->nqab`)    | `Xᵢ[l]·A_l` (phys-in)  | `d²·D²·m`  |
| L2 | `nbk,nbj->nkj` (split: `nqab,nqcd->nabcd`) | `·Xⱼ[l]` (phys-out)    | `d·D²·m²`  |

**Each gadget block `p = 0 … N−2`** (`N−1` blocks), 5 contractions on `E`:

| # | contraction | what it does | cost |
|---|---|---|---|
| 1 | `nij,njal->nial`     | absorb ket half `G^{(p,1)}` (contract env bond)   | `m³·m₁·D` |
| 2 | `nial,nlbo->niabo`   | absorb ket half `G^{(p,2)}` (contract dummy `l`)  | `m⁴·D`    |
| 3 | `niabo,ncdab->niocd` | tie the split leaf `Y_{p+1}` (contract `a,b`)     | `m⁴·D`    |
| 4 | `niocd,nice->node`   | absorb bra half `G^{(p,1)}` (contract bond + `a`) | `m⁴·D`    |
| 5 | `node,nedf->nfo`     | absorb bra half `G^{(p,2)}` (contract dummy + `b`)| `m³·m₂·D` |

Per block `3m⁴·D + m³(m₁+m₂)·D`. Chain = `K−1` matmuls of `m×m`. Summing:

> **one `Mᵢ` = `N(d²D²·m + dD²·m²)` + `(N−1)·D·(3m⁴ + m³(m₁+m₂))`**
> **EXACT total = `K·[ N·d²·m·D² + N·d·m²·D² + (N−1)·D·(3m⁴ + m³(m₁+m₂)) ] + (K−1)·m³`**
> →   **`O(K N D·m(m³ + dD(d+m)))`**  (`O(K N m⁴)` at `D=1`)

(`m₁·m₂=m` ⇒ `m³(m₁+m₂) ≤ m³(m+1) = O(m⁴)`, same order as the `3m⁴`, so Phase 2 is `O(N·D·m⁴)` for
any factorization; `m₁,m₂` survive only in the **exact** total.)

## 5. `dense` — same chain, single random `(m,m,m)` core (rows = m)

Gadget ablation: each gadget is one **drawn `(m,m,m)` core** (std `1/√m`, so `E[core²]=1/m` ⇒
`E[SᵀS]=I`). The `N` Kronecker leaves are identical to solomonik's (`L1`: `d²D²m`, `L2`: `dD²m²`).
Being a primitive (no halves), **each core block** (`N−1` of them) is 3 contractions:

| # | contraction (first block / bulk block) | what it does | cost |
|---|---|---|---|
| 1 | `nikz,nil->nlkz` / `nxy,nxkb->nykb`   | env meets the core (contract a bond)  | `m⁴·D` |
| 2 | `nlkz,nkp->nlpz` / `nykb,nkl->nylb`   | tie the leaf (contract a bond)        | `m⁴·D` |
| 3 | `nlpz,nlpw->nzw` / `nylb,nylc->nbc`   | close the bra core (contract 2 bonds) | `m⁴·D` |

Per block `3m⁴·D` (no boundary half-absorptions, no gadget formation). Chain = `K−1` `m×m` matmuls. Summing:

> **one `Mᵢ` = `N(d²D²·m + dD²·m²)` + `3(N−1)·m⁴·D`**
> **EXACT total = `K·[ N·d²·m·D² + N·d·m²·D² + 3(N−1)·m⁴·D ] + (K−1)·m³`**
> →   **`O(K N D·m(m³ + dD(d+m)))`**  (`O(K N m⁴)` at `D=1`)

**`dense` < `solomonik`:** identical except `dense` lacks solomonik's `(N−1)·D·m³(m₁+m₂)` boundary
term (3 vs 5 contractions/block) — lower order, so the same asymptotic but strictly cheaper at
finite `m`. They're **no longer identical** (they were, in a pre-halves version); comparing them
isolates the gadget's 4th-moment (variance) benefit net of its extra (lower-order) cost.

---

## Summary  (EXACT totals; asymptotics keep every parameter)

| sketch | out | EXACT total | asymptotic | in `m` (D=1) |
|---|---|---|---|---|
| `kron` | `1×1` | `K·N·d(d+1)·D² + (K−1)` | `O(K N d² D²)` | `O(1)` |
| `tt` | `1×1` | `K[(N−2)(2dD·R³+d²D²R²)+2d²D·R+2dD·R²+dD·R] + (K−1)`, `R=min(m,N)` | `O(K N dD·R²(R+dD))` ≤ `O(K N dD·N²(N+dD))` | `O(K N d·N³)` |
| `ttm` | `m×m` | `K[(N−1)(2dD·m³+d²D²m²)+d²D²m+dD·m²] + (K−1)m³` | `O(K N dD·m²(m+dD))` | `O(K N d·m³)` |
| `solomonik` | `m×m` | `K[N d²m D²+N d m²D²+(N−1)D(3m⁴+m³(m₁+m₂))] + (K−1)m³` | `O(K N D·m(m³+dD(d+m)))` | `O(K N m⁴)` |
| `dense` | `m×m` | `K[N d²m D²+N d m²D²+3(N−1)m⁴D] + (K−1)m³` | `O(K N D·m(m³+dD(d+m)))` | `O(K N m⁴)` |

**Verification** (exact `_ec` counter == these formulas, **our data `D=1`**, `d=2, N=4, K=3`):

| sketch | m=8 | m=16 | m=32 | ratio/doubling | matches |
|---|---|---|---|---|---|
| `kron` | 74 | 74 | 74 | `1.0×` | `O(1)` ✓ |
| `tt` | 2 234 | 2 234 | 2 234 | `1.0×` | bond capped at `min(m,N)=4`: flat in `m` ✓ |
| `ttm` | 22 240 | 166 592 | 1 288 576 | `~7.6×` | `O(m³)` ✓ |
| `solomonik` | 141 184 | 2 079 488 | 31 942 144 | `~15×` | `O(m⁴)` ✓ |
| `dense` | 113 536 | 1 784 576 | 28 403 200 | `~15.9×` | `O(m⁴)` ✓ |

Notes:
- **`solomonik ≠ dense`** now (e.g. 2.08M vs 1.78M at m=16): solomonik pays the extra
  `(N−1)D·m³(m₁+m₂)` for feeding gadget halves; the gadget is a variance device that here also
  costs a (lower-order) amount. Comparing the two isolates the gadget's 4th-moment benefit *net
  of* its extra cost.
- The matrix sketches (`ttm`, `solomonik`, `dense`) all pay the `(K−1)m³` chain. For `ttm` that
  is the same order as its `m³` sweep; for `solomonik`/`dense` it is dominated by the `N m⁴` sweep.
- `D` is symbolic; the table fixes `D=1` because identity and all-ones are bond-1 MPOs. For a
  general MPO substitute the `D`-powers shown (D² on applied cores, D¹ on threaded bonds).
- Costs are exact (summed per contraction), not leading-order only — the table's raw numbers
  include every sub-term.
