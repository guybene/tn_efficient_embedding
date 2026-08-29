"""CODE ATTIC (2026-08-11) -- pieces retired from sketches.py / test_sketches.py when
the experiment files were consolidated.  NOT runnable standalone: methods belong to the
classes named in the comments; to revive, paste them back and re-add
`import string` (sketches.py) / `import math` (test_sketches.py).

Contents:
 1. DenseTree            -- the gadget ABLATION sketch (single random (m,m,m) core per
                            chain block instead of Solomonik's structured halves).
                            Benched June 2026: cheaper but worse variance, dropped from
                            every experiment's algo list.
 2. mm_einsum methods    -- per-sketch builders of the M_i = S A S^T network for
                            opt_einsum (Sketch base stub + Kron/TT/Solomonik/TTm).
 3. check_optimal et al. -- the optimality check: our M contraction order == the
                            opt_einsum optimum (verified June 2026 for every chain
                            sketch at d^N >> m^4; results recorded in
                            inner_claude_files/cost_per_algorithm.md).  Needs (2).
"""
import math
import string

import torch


# ============================================================================
# 1. DenseTree (was in sketches.py REGISTRY as "dense_tree")
# ============================================================================
class DenseTree:  # (Sketch)
    """Leaf-less chain: N-1 primitive cores, no Kronecker leaves.  Core 0 = (d, d, m)
    carries phys sites 0 AND 1; core l (1 <= l <= N-2) = (m, d, m) carries phys site l+1,
    exactly a TTm middle core.  The LAST core's out-leg is the open row leg mu -> rows = m.
    S_{mu,x} = sum over N-2 internal bonds of the product of N-1 cores, so
    E[S^T S] = sigma^{2(N-1)} * m^{N-2} (internal) * m (mu) * I  =>  sigma^2 = 1/m.
    Leading cost O(K N d m^3) -- same order as TTm; the gadget's m^4 is gone with the leaves."""
    name = "dense_tree"

    def rows(self):
        return self.m

    def draw(self, B):
        from sketches import _rand
        d, N, m = self.d, self.N, self.m
        sc = 1.0 / math.sqrt(m)
        cores = [_rand((B, d, d, m), self.dist, self.device) * sc]        # (x0, x1, z1)
        cores += [_rand((B, m, d, m), self.dist, self.device) * sc
                  for _ in range(N - 2)]                                  # (z_l, x_{l+1}, z_{l+1})
        return cores

    def M(self, ci, A, cj):
        from sketches import _ec
        # env E = (B, z_i, z_j); sweep front -> back; final out-legs = the open (mu_i, mu_j).
        t = _ec("nabz,ax->nxbz", ci[0], A[0])        # apply A_0            d^3 m
        t = _ec("nxbz,by->nxyz", t, A[1])            # apply A_1            d^3 m
        E = _ec("nxyz,nxyw->nzw", t, cj[0])          # absorb cj[0]         d^2 m^2
        for l in range(1, self.N - 1):               # core l = site l+1
            t = _ec("nij,nixz->njxz", E, ci[l])      # contract z_i         d m^3
            t = _ec("njxz,xy->njyz", t, A[l + 1])    # apply A_{l+1}        d^2 m^2
            E = _ec("njyz,njyw->nzw", t, cj[l])      # contract z_j + phys  d m^3
        return E                                     # (B, m, m) = (mu_i, mu_j)

    def cost(self, K, D=1):
        d, N, m = self.d, self.N, self.m
        cost = 0
        # core-0 block (two physical legs, no env yet):
        cost += d ** 3 * m * D           # apply A_0 to (x0, x1, z)
        cost += d ** 3 * m * D ** 2      # apply A_1
        cost += d ** 2 * m ** 2 * D      # absorb cj[0] -> (z_i, z_j) env
        # N-2 middle cores (TTm-rate blocks):
        cost += (N - 2) * (d * m ** 3 * D)        # absorb ci[l]
        cost += (N - 2) * (d ** 2 * m ** 2 * D ** 2)  # apply A_{l+1}
        cost += (N - 2) * (d * m ** 3 * D)        # absorb cj[l]
        cost *= K                        # K independent copies M_1, ..., M_K
        cost += (K - 1) * m ** 3         # chain Tr(M_1 ... M_K): K-1 (m x m) matmuls
        cost += m                        # final trace of the (m x m) product
        return cost

    def mm_einsum(self, fac, A):                     # M_i = S A S^T (m x m, row legs open)
        c, N, sy = fac, self.N, iter(string.ascii_letters)
        mb, mk = next(sy), next(sy)                  # bra/ket row legs
        bb = [next(sy) for _ in range(N - 2)]        # bra internal bonds
        kb = [next(sy) for _ in range(N - 2)]        # ket internal bonds
        p = [next(sy) for _ in range(N)]             # bra phys
        q = [next(sy) for _ in range(N)]             # ket phys
        ops, t = [], []

        def side(mu, phys, bonds):
            ops.append(c[0][0])
            t.append(phys[0] + phys[1] + (bonds[0] if N > 2 else mu))
            for l in range(1, N - 1):                # core l = (z_{l-1}, x_{l+1}, z_l)
                ops.append(c[l][0])
                t.append(bonds[l - 1] + phys[l + 1] + (bonds[l] if l < N - 2 else mu))

        side(mb, p, bb); side(mk, q, kb)
        for l in range(N):
            ops.append(A[l]); t.append(p[l] + q[l])
        return ",".join(t) + "->" + mb + mk, ops

    def dense(self, cores):
        B, d, N, m = cores[0].shape[0], self.d, self.N, self.m
        t = cores[0].reshape(B, d * d, m)
        for l in range(1, N - 1):
            t = torch.einsum("bPr,brds->bPds", t, cores[l]).reshape(B, -1, m)
        return t.transpose(1, 2)                     # (B, m, d^N)

    def chunk(self, K):
        return max(32, min(16384, 2_000_000 // max(1, self.m ** 2 * self.d)))


# ============================================================================
# 2. mm_einsum methods (were on Sketch / Kron / TT / Solomonik / TTm)
# ============================================================================
# --- Sketch (base stub) ---
def mm_einsum(self, fac, A):
    """einsum for ONE M_i = S A S^T from the raw factors: returns (eq, operands) for
    opt_einsum.  Same S (factors) on both sides, A in the middle; output = the row legs
    (scalar for rows=1).  `check_optimal` hands this to opt_einsum to confirm our `M`
    contraction order is optimal.  Override per sketch."""
    raise NotImplementedError


# --- Kron ---
def kron_mm_einsum(self, fac, A):                     # M_i = prod_l g_l^T A_l g_l  (scalar)
    g, sy = fac[0], iter(string.ascii_letters)
    ops, t = [], []
    for l in range(self.N):
        p, q = next(sy), next(sy)                # bra phys, ket phys
        ops += [g[l], A[l], g[l]]
        t += [p, p + q, q]
    return ",".join(t) + "->", ops


# --- TT ---
def tt_mm_einsum(self, fac, A):                     # M_i = <v|A|v>  (scalar)
    c, N, sy = fac, self.N, iter(string.ascii_letters)
    bb = [next(sy) for _ in range(N - 1)]        # bra internal bonds
    kb = [next(sy) for _ in range(N - 1)]        # ket internal bonds
    p = [next(sy) for _ in range(N)]             # bra phys ; q = ket phys
    q = [next(sy) for _ in range(N)]
    ops, t = [], []

    def side(phys, bonds):                       # bra/ket cores (size-1 boundary bonds squeezed)
        for l in range(N):
            core = c[l][0]
            sub = (bonds[l - 1] if l else "") + phys[l] + (bonds[l] if l < N - 1 else "")
            if l == 0:
                core = core.reshape(core.shape[1], core.shape[2])
            elif l == N - 1:
                core = core.reshape(core.shape[0], core.shape[1])
            ops.append(core); t.append(sub)

    side(p, bb); side(q, kb)
    for l in range(N):
        ops.append(A[l]); t.append(p[l] + q[l])  # A_l ties bra phys p_l to ket phys q_l
    return ",".join(t) + "->", ops


# --- Solomonik ---
def solomonik_mm_einsum(self, fac, A):           # M_i = S A S^T (m x m, gadget HALVES explicit)
    from sketches import _m1m2
    halves, X = fac
    d, N, m = self.d, self.N, self.m
    m1, m2 = _m1m2(m)
    sy = iter(string.ascii_letters + string.digits)
    new = lambda: next(sy)
    ops, t = [], []
    pb = [new() for _ in range(N)]               # bra phys ; pk = ket phys
    pk = [new() for _ in range(N)]

    def side(phys):                              # leaves + gadget halves; returns row leg n
        in_sym = None
        for p in range(N - 1):
            G1, G2 = halves[p][0][0], halves[p][1][0]
            a, b, l = new(), new(), new()
            inp = new() if p == 0 else in_sym
            out = new()
            if p == 0:
                ops.append(X[0, 0]); t.append(phys[0] + inp)                  # X^0 (d,m)
            ops.append(X[0, p + 1].reshape(d, m1, m2)); t.append(phys[p + 1] + a + b)
            ops.append(G1); t.append(inp + a + l)                             # G^(p,1)
            ops.append(G2); t.append(l + b + out)                             # G^(p,2)
            in_sym = out
        return in_sym

    n_bra, n_ket = side(pb), side(pk)
    for l in range(N):
        ops.append(A[l]); t.append(pb[l] + pk[l])
    return ",".join(t) + "->" + n_bra + n_ket, ops


# --- TTm ---
def ttm_mm_einsum(self, fac, A):                     # M_i = S A S^T  (m x m), output legs on core0
    c, N, sy = fac, self.N, iter(string.ascii_letters)
    mb, mk = next(sy), next(sy)                  # bra/ket output (row) legs
    bb = [next(sy) for _ in range(N - 1)]
    kb = [next(sy) for _ in range(N - 1)]
    p = [next(sy) for _ in range(N)]
    q = [next(sy) for _ in range(N)]
    ops, t = [], []

    def side(mu, phys, bonds):
        for l in range(N):
            core = c[l][0]
            if l == 0:                           # (mu, d, R_out)
                sub = mu + phys[0] + bonds[0]
            else:
                sub = bonds[l - 1] + phys[l] + (bonds[l] if l < N - 1 else "")
                if l == N - 1:                   # squeeze size-1 right bond
                    core = core.reshape(core.shape[0], core.shape[1])
            ops.append(core); t.append(sub)

    side(mb, p, bb); side(mk, q, kb)
    for l in range(N):
        ops.append(A[l]); t.append(p[l] + q[l])
    return ",".join(t) + "->" + mb + mk, ops


# ============================================================================
# 3. check_optimal + _path_cost (were in test_sketches.py)
# ============================================================================
def _path_cost(eq, operands, path):
    """Cost of an opt_einsum `path` over network `eq` in OUR convention (product of all
    distinct index dims per pairwise step -- identical to `_ec`), so it's comparable to
    `cost_per_iter`.  opt_einsum's own `opt_cost` is FLOPs (~2x this), hence the replay."""
    ins, out = eq.split("->")[0].split(","), eq.split("->")[1]
    sizes = {ch: dim for s, op in zip(ins, operands) for ch, dim in zip(s, op.shape)}
    terms, total = [set(s) for s in ins], 0
    for step in path:
        inv = set().union(*(terms[i] for i in step))
        for i in sorted(step, reverse=True):
            terms.pop(i)
        total += math.prod(sizes[ch] for ch in inv)
        keep = set(out).union(*terms) if terms else set(out)
        terms.append(inv & keep)
    return total


@torch.no_grad()
def check_optimal(d=6, N=4, ms=(4,), tol=2e-2):
    """Is our `M` contraction ORDER optimal?  Each sketch's `mm_einsum` builds the M_i = S A S^T
    network from the real factors; opt_einsum finds the optimal contraction, and we compare its
    cost to our `cost_per_iter` (both in the `_ec` convention).  Run with d^N >> m^4 (default
    d=6) so opt_einsum can't beat the sweep by densifying the small d^N-space -- the order is
    d-independent, so this validates the same order used at d=2.  1% slack absorbs tt's
    vanishing meet-in-the-middle boundary constant; kron sits below 1 (uncounted 1x1 mults)."""
    import opt_einsum as oe
    from sketches import REGISTRY, make_A
    from test_sketches import cost_per_iter
    assert d ** N > max(ms) ** 4, f"need d^N >> m^4 (else opt_einsum densifies); raise d or N"
    A = make_A("identity", d, N)                                 # D=1
    print(f"optimality check (opt_einsum):  measured vs optimal order   (d={d} N={N} D=1)")
    ok = True
    for name, cls in REGISTRY.items():
        for m in ms:
            try:
                sk = cls(d, N, m)
            except ValueError:          # sketch doesn't support this d (kron_pauli: d=2 only)
                continue
            fac = sk.draw(1)
            eq, operands = sk.mm_einsum(fac, A)
            got = oe.contract(eq, *operands).reshape(-1)          # guard: mm_einsum builds M_i
            assert torch.allclose(got, sk.M(fac, A, fac)[0].reshape(-1), atol=1e-6), \
                f"{name}: mm_einsum network != sk.M"
            path, _ = oe.contract_path(eq, *operands, optimize="dp")
            opt, meas = _path_cost(eq, operands, path), cost_per_iter(sk, 1, A)
            ok &= 1 - tol <= meas / opt <= 1 + tol
            print(f"  [{name:9s}] m={m}: meas={meas} opt={opt} "
                  f"[{'ok' if ok else 'SUBOPTIMAL'}]")
    print(f"  -> {'ALL OPTIMAL' if ok else 'CHECK FLAGGED'}")
    assert ok, "check_optimal FAILED: M contraction order is not optimal"
    return ok
