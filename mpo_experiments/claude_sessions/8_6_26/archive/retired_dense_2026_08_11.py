"""CODE ATTIC (2026-08-11, second pass) -- the dense-matrix test layer retired from
sketches.py / test_sketches.py.  NOT runnable standalone: methods belong to the classes
named in the comments.

Why retired: the dense methods existed ONLY as bit-for-bit test references (self_test /
check_isometry / _factored_checks).  They were replaced by the cheaper-but-complete
battery: unbiasedness at K=1 AND K=2 (on the identity for the chain sketches, where the
pair pins E[S^H S] = I exactly via Cauchy-Schwarz), plus integer-exact hand cost
formulas for the hairpin sketches (check_cost_hairpin, cost_mpo_hairpin.md).
To revive a dense check, paste the method back into its class and the test back into
test_sketches.py.

Contents:
 1. dense() methods of the chain sketches (Sketch stub, Kron, TT, Solomonik, TTm)
 2. dense_S() methods of the tree sketches (TreeOurs, TreeKron, TreeTT)
 3. dense_S() methods of the hairpin sketches (HpKron, HpTT, HpTTm, HpMa)
 4. the retired tests: _A_dense, _dense_estimate, self_test, check_isometry,
    _factored_dense_estimate, _factored_checks, check_tree, check_xx
"""
import math

import torch


# ============================================================================
# 1. chain dense() methods
# ============================================================================
# --- Sketch (base stub) ---
def dense(self, fac):
    """Dense (B, rows, d^N) sketch -- TEST REFERENCE ONLY."""
    raise NotImplementedError


# --- Kron ---
def kron_dense(self, g):
    B = g.shape[0]
    v = g[:, 0]
    for l in range(1, self.N):
        v = (v[:, :, None] * g[:, l][:, None, :]).reshape(B, -1)
    return v.reshape(B, 1, self.d ** self.N)


# --- TT ---
def tt_dense(self, cores):
    B = cores[0].shape[0]
    t = cores[0].reshape(B, self.d, cores[0].shape[-1])
    for l in range(1, self.N):
        t = torch.einsum("bpr,brds->bpds", t, cores[l])
        t = t.reshape(B, -1, cores[l].shape[-1])
    return t.reshape(B, 1, self.d ** self.N)


# --- Solomonik ---
def solomonik_dense(self, fac):
    from sketches import _m1m2
    halves, X = fac
    B, d, N, m = X.shape[0], self.d, self.N, self.m
    m1, m2 = _m1m2(m)
    gad = [torch.einsum("bpil,bljq->bpijq", G1, G2).reshape(B, m, m1 * m2, m)
           for G1, G2 in halves]                 # form gadget -- TEST REFERENCE ONLY
    t = torch.einsum("bkjz,bdk->bdjz", gad[0], X[:, 0])
    t = torch.einsum("bikz,bek->biez", t, X[:, 1])
    run = t.reshape(B, d * d, m)
    for idx, p in enumerate(range(2, N)):
        gp = torch.einsum("bakz,bdk->badz", gad[1 + idx], X[:, p])
        run = torch.einsum("bPa,badz->bPdz", run, gp)
        run = run.reshape(B, -1, m)
    return run.transpose(1, 2)


# --- TTm ---
def ttm_dense(self, cores):
    B, m, d, N = cores[0].shape[0], self.m, self.d, self.N
    t = cores[0]                                     # (B, mu, d, r)
    for l in range(1, N):
        t = torch.einsum("bmpr,brds->bmpds", t, cores[l])
        t = t.reshape(B, m, -1, cores[l].shape[-1])
    return t.reshape(B, m, d ** N)


# ============================================================================
# 2. tree dense_S() methods
# ============================================================================
def treeours_dense_S(self, fac, op):
    S1, S2 = fac
    S = torch.einsum("nabcu,nuev->nvabce", S1, S2)             # (n, m, e1,e2,e3,e4)
    return S.reshape(S.shape[0], self.m, self.d ** 4)


def treekron_dense_S(self, fac, op):
    g, B = fac, fac.shape[0]
    v = g[:, 0]
    for l in range(1, 4):
        v = (v[:, :, None] * g[:, l][:, None, :]).reshape(B, -1)
    return v.reshape(B, 1, self.d ** 4)


def treett_dense_S(self, fac, op):
    c1, c2, c3, c4 = fac
    t = torch.einsum("nar,nrbs->nabs", c1, c2)
    t = torch.einsum("nabs,nsct->nabct", t, c3)
    t = torch.einsum("nabct,nte->nabce", t, c4)
    return t.reshape(t.shape[0], 1, self.d ** 4)


# ============================================================================
# 3. hairpin dense_S() methods
# ============================================================================
def hpkron_dense_S(self, fac, op):
    g, B = fac, fac.shape[0]
    v = g[:, 0]
    for l in range(1, self.NA):
        v = (v[:, :, None] * g[:, l][:, None, :]).reshape(B, -1)
    return v.reshape(B, 1, self.d ** self.NA)


def hptt_dense_S(self, fac, op):
    t = fac[0]                                                     # (n, d, R)
    for l in range(1, self.NA - 1):
        t = torch.einsum("npr,nrds->npds", t, fac[l])
        t = t.reshape(t.shape[0], -1, t.shape[-1])
    t = torch.einsum("npr,nrd->npd", t, fac[-1])
    return t.reshape(t.shape[0], 1, self.d ** self.NA)


def hpttm_dense_S(self, fac, op):
    t = fac[0]                                                     # (n, mu, d, r)
    for l in range(1, self.NA - 1):
        t = torch.einsum("nupr,nrds->nupds", t, fac[l])
        t = t.reshape(t.shape[0], self.m, -1, t.shape[-1])
    t = torch.einsum("nupr,nrd->nupd", t, fac[-1])
    return t.reshape(t.shape[0], self.m, self.d ** self.NA)


def hpma_dense_S(self, fac, op):
    X, gads = fac
    n, d, m = X.shape[0], self.d, self.m
    S = X[:, 0].transpose(1, 2)                                    # (n, m, d^1)
    for i in range(1, self.NA):
        xi = X[:, i].reshape(n, d, self.s1, self.s2)
        ga, gb = gads[i - 1]
        Mi = torch.einsum("nual,ndab,nlbv->nudv", ga, xi, gb)      # (n, m_in, d, m_out)
        S = torch.einsum("nup,nudv->nvpd", S, Mi).reshape(n, m, -1)
    return S                                                       # (n, m, d^NA)


# ============================================================================
# 4. retired tests (were in test_sketches.py; need the methods above pasted back)
# ============================================================================
def _A_dense(A):
    """Dense (D, D) operator from the (N, d, d) MPO -- self-test reference only."""
    full = A[0]
    for l in range(1, A.shape[0]):
        full = torch.kron(full, A[l])
    return full


@torch.no_grad()
def _dense_estimate(sketch, facs, A):
    Ad = _A_dense(A)
    S = torch.stack([sketch.dense(f) for f in facs], 1)             # (B, K, rows, D)
    SA = torch.matmul(S.conj(), Ad.to(S.dtype))    # bra side conjugated (no-op for real sketches)
    K = len(facs)
    P = None
    for i in range(K):
        Mi = torch.einsum("crd,cnd->crn", SA[:, i], S[:, (i + 1) % K])
        P = Mi if P is None else torch.matmul(P, Mi)
    return torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)


@torch.no_grad()
def self_test():
    """Bit-for-bit: TN  M / estimate == dense, on the SAME factors, both operators."""
    from sketches import REGISTRY, estimate, make_A
    print("self-test: TN  ==  dense   (same random factors)")
    worst = 0.0
    for d, N in [(3, 6), (2, 6)]:
        for name, cls in REGISTRY.items():
            try:
                sk = cls(d, N, 9, dist="rademacher")
            except ValueError:          # sketch doesn't support this d (kron_pauli: d=2 only)
                continue
            B = 24
            for kind in ("identity", "all_ones"):
                A = make_A(kind, d, N)
                fi, fj = sk.draw(B), sk.draw(B)
                Si = sk.dense(fi).conj()             # bra side (no-op for real sketches)
                e_M = (sk.M(fi, A, fj)
                       - torch.matmul(torch.matmul(Si, _A_dense(A).to(Si.dtype)),
                                      sk.dense(fj).transpose(-1, -2))).abs().max().item()
                facs = [sk.draw(B) for _ in range(3)]
                e_e = (estimate(sk, facs, A) - _dense_estimate(sk, facs, A)).abs().max().item()
                worst = max(worst, e_M, e_e)
                print(f"  d={d} N={N} {name:9s} {kind:8s}: M={e_M:.1e}  est[K=3]={e_e:.1e}")
    ok = worst < 1e-7
    print(f"worst |TN - dense| = {worst:.2e}  ->  {'ALL PASS' if ok else 'FAIL'}")
    assert ok, f"self_test FAILED: TN != dense (worst |TN - dense| = {worst:.2e})"
    return ok


@torch.no_grad()
def check_isometry(d=2, N=4, m=9, R=1_000_000, tol=1e-2):
    """E[S^T S] = I_{d^N}?  Monte-Carlo estimate: average S^H S over R draws of the dense
    d^N sketch, then take max|E[S^H S] - I|.  Seeded for a deterministic assert."""
    from sketches import REGISTRY, DTYPE
    torch.manual_seed(0)
    Dn = d ** N
    Id = torch.eye(Dn, dtype=DTYPE)
    print(f"isometry: max|E[S^T S] - I|   (d={d} N={N} m={m} R={R} D={Dn})")
    worst = 0.0
    for name, cls in REGISTRY.items():
        try:
            sk = cls(d, N, m)
        except ValueError:              # sketch doesn't support this d (kron_pauli: d=2 only)
            continue
        acc = None
        done = 0
        chunk = max(1, min(8192, 4_000_000 // (sk.sketch_size() * Dn)))
        while done < R:
            c = min(chunk, R - done)
            S = sk.dense(sk.draw(c))                   # (c, rows, D) -- same dense() as self_test
            t = torch.einsum("crd,cre->de", S.conj(), S)  # sum S^H S (conj = no-op for real)
            acc = t if acc is None else acc + t
            done += c
        err = (acc / R - Id).abs().max().item()
        worst = max(worst, err)
        print(f"  [{name:9s}] max|E[S^T S] - I| = {err:.2e}")
    ok = worst < tol
    print(f"worst = {worst:.2e}  ->  {'ALL PASS' if ok else 'FAIL'}")
    assert ok, f"check_isometry FAILED: max|E[S^T S] - I| = {worst:.2e} >= {tol}"
    return ok


@torch.no_grad()
def _factored_dense_estimate(sk, facs, op):
    """Dense reference for estimate_factored: V_r from dense_S (bra-conjugated for the
    pauli sketches) times the dense factor C, chained in plain torch."""
    from sketches import CDTYPE
    C = op.C if hasattr(op, "C") else op.dense_C().to(CDTYPE)
    Vs = []
    for f in facs:
        S = sk.dense_S(f, op)
        S = S.conj() if "pauli" in sk.name else S
        Vs.append(S.to(CDTYPE) @ C.to(CDTYPE))
    E = None if op.E is None else op.E.to(CDTYPE)
    P = None
    for r in range(len(facs)):
        Vr = Vs[r] if E is None else Vs[r] @ E
        Mi = torch.einsum("nrf,nsf->nrs", Vr, Vs[(r + 1) % len(facs)].conj())
        P = Mi if P is None else torch.matmul(P, Mi)
    return torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)


@torch.no_grad()
def _factored_checks(op, registry, make, label, R_iso=150_000, B_unb=200_000, tol_z=5.0):
    """Shared battery: (1) V == dense_S @ C bit-for-bit, (2) estimate_factored == dense
    chain, (3) E[S^H S] ~ I Monte-Carlo, (4) quick unbiasedness z-test at K=2."""
    from sketches import estimate_factored, CDTYPE
    C = (op.C if hasattr(op, "C") else op.dense_C().to(CDTYPE)).to(CDTYPE)
    Dn = C.shape[0]
    worst_v, worst_e = 0.0, 0.0
    print(f"{label}: V == dense, estimator == dense, isometry, unbiasedness")
    for name, sk in [(n, make(n, cls)) for n, cls in registry.items()]:
        fac = sk.draw(7)
        Sb = sk.dense_S(fac, op)
        Sb = Sb.conj() if "pauli" in name else Sb
        ref = Sb.to(CDTYPE) @ C
        got = sk.V(fac, op).to(CDTYPE)
        e_v = (got - ref).abs().max().item()
        facs = [sk.draw(5) for _ in range(3)]
        got_e = estimate_factored(sk, facs, op)
        ref_e = _factored_dense_estimate(sk, facs, op)
        e_e = ((got_e - ref_e).abs() / ref_e.abs().max().clamp(min=1e-30)).max().item()
        worst_v, worst_e = max(worst_v, e_v), max(worst_e, e_e)
        acc, done, chunk = None, 0, max(1, 2_000_000 // (sk.sketch_size() * Dn))
        while done < R_iso:
            c = min(chunk, R_iso - done)
            S = sk.dense_S(sk.draw(c), op)
            t = torch.einsum("crd,cre->de", S.conj(), S)
            acc = t if acc is None else acc + t
            done += c
        e_iso = (acc / R_iso - torch.eye(Dn, dtype=acc.dtype)).abs().max().item()
        tv, ests, done = op.true_pk(2), [], 0
        chunk = sk.chunk(2)
        while done < B_unb:
            c = min(chunk, B_unb - done)
            ests.append(estimate_factored(sk, [sk.draw(c) for _ in range(2)], op))
            done += c
        est = torch.cat(ests)
        z = abs(est.mean().item() - tv) / max(1e-30, est.std().item() / B_unb ** 0.5)
        print(f"  [{name:10s}] V={e_v:.1e}  est={e_e:.1e}  iso={e_iso:.2e}  "
              f"unbiased z={abs(z):.2f}")
        assert e_iso < 3e-2, f"{label}/{name}: isometry off ({e_iso:.2e})"
        assert abs(z) < tol_z, f"{label}/{name}: biased (z={z:.2f})"
    ok = worst_v < 1e-8 and worst_e < 1e-8
    print(f"worst |TN - dense|: V={worst_v:.2e} est={worst_e:.2e} -> "
          f"{'ALL PASS' if ok else 'FAIL'}")
    assert ok, f"{label}: TN != dense"
    return True


@torch.no_grad()
def check_tree():
    from sketches import TreeOp, TREE_REGISTRY
    op = TreeOp(d=2, chi=3, g=3, seed=1)
    print(f"tree op: spectrum={[round(v, 4) for v in op.lam.tolist()]}  rho={op.rho():.2f}")
    return _factored_checks(op, TREE_REGISTRY,
                            lambda n, cls: cls(4, d=2), "tree")


@torch.no_grad()
def check_xx():
    from sketches import hairpin_from_xx, xx_pk_exact, HP_REGISTRY, CDTYPE
    NA = 4
    hp = hairpin_from_xx(NA)
    ks = (1, 2, 3, 4)
    ff = xx_pk_exact(2 * NA, NA, ks)
    C = hp.dense_C().to(CDTYPE)
    rho = C @ hp.E.to(CDTYPE) @ C.conj().T                     # dense rho_A (16 x 16)
    print(f"xx op: NA={NA} chi={hp.chi}  rho(intrinsic dim)={hp.rho():.2f}")
    for k in ks:
        dense_pk = torch.linalg.matrix_power(rho, k).diagonal().sum().real.item()
        a, b, c = hp.true_pk(k), dense_pk, ff[k]
        print(f"  p_{k}: hairpin={a:.10f}  dense={b:.10f}  free-fermion={c:.10f}")
        assert abs(a - b) < 1e-8 and abs(a - c) < 1e-7, f"p_{k} mismatch"
    assert abs(hp.true_pk(1) - 1.0) < 1e-8, "Tr(rho_A) != 1"
    return _factored_checks(hp, HP_REGISTRY,
                            lambda n, cls: cls(NA, hp.chi, 4), "xx-hairpin")
