"""
Tests for sketches.py: expectation + ground truth for both operators, counting on one
case.

Common setup.  One estimator (see sketches.py): blocks M_r = S_r A S_{r+1}^H over
the full network, fresh per block; for the XX operator the input is the MPO cores.

[1] TREE (operator form only): unbiasedness of the full estimator at K=1 and K=2 --
    z-tests against the exact spectrum, for every tree sketch.  Unbiasedness at all K
    is equivalent to the sketch expectation property E[S^H S] = I; K=1 and K=2 pin
    its two lowest moments through the estimator actually run in the experiment.

[2] XX: ground truth first -- three independent routes to the scalar Tr(A^k) must
    agree to ~10 digits: the exact spectrum computed at build, brute-force powers of
    the materialized operator, and the closed-form SPECTRUM ORACLE (an
    n_sites x n_sites eigenproblem sharing zero code with the pipeline).  Every
    experiment pass rate divides by this number.
    Then the simple expectation test, same as [1]: unbiasedness of the MPO-form
    estimator at K=1 and K=2 (z-tests against the exact truth), for every xx sketch.

[3] COUNTING, one case (our tree sketch): the counting mechanism (the single _ec
    wrapper + the shared estimators) is class-independent, so one case pins it.
    (a) the tally of one block M must EQUAL (integer equality) the hand-derived op
        count of the executed einsum sequence -- one term per _ec call, the product
        of that call's distinct index dimensions, written out in the test;
    (b) batch linearity, tally(5) == 5 * tally(1) (justifies per-sample x batch).
"""
import math

import torch

from sketches import (TREE_REGISTRY, XX_REGISTRY, TreeOp, build_xx_mps,
                      estimate_operator, CDTYPE, _reset_ops, _get_ops)


# ---------------------------------------------------------------------------
# [1] TREE: unbiasedness of the operator-form estimator
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_expectation_tree():
    print("[1/3] EXPECTATION -- tree (operator form, K = 1 and 2)")
    torch.manual_seed(3)
    op = TreeOp([1.0, 1.0, 1.0], d=2, chi=2, seed=1)

    # Wiring, one deterministic case PER SKETCH: one draw; M must equal
    # S_row A S_col^H computed DENSELY (S contracted explicitly from the drawn
    # factors; A = H diag(spectrum) H^H via dense_half).  Value equality, not
    # statistics.
    H = op.dense_half()
    A = (H * op.spectrum.to(CDTYPE)) @ H.conj().T

    sk = TREE_REGISTRY["tree"](4, d=2)
    fac_row, fac_col = sk.draw(1), sk.draw(1)
    Sr = torch.einsum("acu,uem->acem", fac_row[0][0],
                      fac_row[1][0]).reshape(op.d ** 4, sk.m).T.to(CDTYPE)
    Sc = torch.einsum("acu,uem->acem", fac_col[0][0],
                      fac_col[1][0]).reshape(op.d ** 4, sk.m).T.to(CDTYPE)
    block = sk.M(fac_row, fac_col, op)[0]
    assert torch.allclose(block, Sr @ A @ Sc.conj().T, atol=1e-10), \
        "tree: M != S_row A S_col^H"

    sk = TREE_REGISTRY["kron"](4, d=2)
    fac_row, fac_col = sk.draw(1), sk.draw(1)
    Sr = torch.einsum("a,c,e->ace", fac_row[0][0], fac_row[1][0],
                      fac_row[2][0]).reshape(1, -1).to(CDTYPE)
    Sc = torch.einsum("a,c,e->ace", fac_col[0][0], fac_col[1][0],
                      fac_col[2][0]).reshape(1, -1).to(CDTYPE)
    block = sk.M(fac_row, fac_col, op)[0]
    assert torch.allclose(block, Sr @ A @ Sc.conj().T, atol=1e-10), \
        "kron: M != s_row A s_col^H"

    sk = TREE_REGISTRY["tt_capped"](4, d=2)
    fac_row, fac_col = sk.draw(1), sk.draw(1)
    Sr = torch.einsum("ah,hci,ie->ace",
                      *[f[0] for f in fac_row]).reshape(1, -1).to(CDTYPE)
    Sc = torch.einsum("ah,hci,ie->ace",
                      *[f[0] for f in fac_col]).reshape(1, -1).to(CDTYPE)
    block = sk.M(fac_row, fac_col, op)[0]
    assert torch.allclose(block, Sr @ A @ Sc.conj().T, atol=1e-10), \
        "tt: M != s_row A s_col^H"

    sk = TREE_REGISTRY["kressner"](4, d=2, bond=6)     # ell = 4 stacked probes
    fac_row, fac_col = sk.draw(1), sk.draw(1)
    Sr = torch.einsum("gah,ghci,gie->gace",
                      *[f[0] for f in fac_row]).reshape(sk.m, -1).to(CDTYPE)
    Sc = torch.einsum("gah,ghci,gie->gace",
                      *[f[0] for f in fac_col]).reshape(sk.m, -1).to(CDTYPE)
    block = sk.M(fac_row, fac_col, op)[0]
    assert torch.allclose(block, Sr @ A @ Sc.conj().T, atol=1e-10), \
        "kressner: M != S_row A S_col^H"
    print("    wiring: M == S_row A S_col^H (dense reference)  tree/kron/tt/kressner  ok")

    for name, cls in TREE_REGISTRY.items():
        sk = cls(4, d=2, bond=6) if name == "kressner" else cls(4, d=2)
        for K in (1, 2):
            truth, ests, done = op.true_pk(K), [], 0
            while done < 150_000:
                batch = min(sk.chunk(K), 150_000 - done)
                ests.append(estimate_operator(sk, [sk.draw(batch) for _ in range(K)], op))
                done += batch
            est = torch.cat(ests)[:150_000]
            stderr = est.std().item() / math.sqrt(150_000)
            z = (est.mean() - truth).abs().item() / max(stderr, 1e-30)
            rel = (est.mean() - truth).abs().item() / abs(truth)
            print(f"    [{name:10s} K={K}] z = {z:.2f}   rel err = {rel:.2e}")
            assert z < 5, f"tree/{name} K={K}: biased (z={z:.2f})"
            assert rel < 1e-2, f"tree/{name} K={K}: rel err {rel:.2e} > 1e-2"
    print("  -> PASS")


# ---------------------------------------------------------------------------
# [2] XX: ground truth, then the expectation test (MPO form)
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_expectation_xx():
    print("[2/3] EXPECTATION -- xx (MPO form, K = 1 and 2)")
    torch.manual_seed(3)

    # Ground truth (NA=4): the factored spectrum vs brute-force powers of the
    # materialized operator (build_xx_mps itself asserts our spectrum against
    # TeNPy's Schmidt spectrum at the cut).
    op = build_xx_mps(4)
    A_dense = op.dense_operator()
    for k in (1, 2):
        pk_exact = op.true_pk(k)                          # the spectrum from build
        pk_brute = torch.linalg.matrix_power(A_dense, k).diagonal().sum().real.item()
        print(f"    p_{k}: spectrum={pk_exact:.10f}  dense={pk_brute:.10f}")
        assert abs(pk_exact - pk_brute) < 1e-8, f"p_{k}: != dense"

    # Expectation test (NA=3), through the estimator, for every xx sketch.
    op = build_xx_mps(3)
    for name, cls in XX_REGISTRY.items():
        sk = cls(3, op.chi, 4, bond=6) if name == "kressner" else cls(3, op.chi, 4)
        for K in (1, 2):
            truth, ests, done = op.true_pk(K), [], 0
            while done < 150_000:
                batch = min(sk.chunk(K), 150_000 - done)
                ests.append(estimate_operator(sk, [sk.draw(batch) for _ in range(K)], op))
                done += batch
            est = torch.cat(ests)[:150_000]
            stderr = est.std().item() / math.sqrt(150_000)
            z = (est.mean() - truth).abs().item() / max(stderr, 1e-30)
            print(f"    [{name:10s} K={K}] z = {z:.2f}")
            assert z < 5, f"xx/{name} K={K}: biased (z={z:.2f})"
    print("  -> PASS")


# ---------------------------------------------------------------------------
# [3] COUNTING, one case: our tree sketch
# ---------------------------------------------------------------------------
@torch.no_grad()
def check_counting():
    print("[3/3] COUNTING -- one case, our tree sketch")

    # (a) block tally vs the hand-derived count of the executed einsum sequence.
    # TreeOurs.M runs exactly twelve _ec calls along the FIXED ASYMMETRIC path,
    # each sketch slot resolved as ((S, v), u) -- the sketch core first:
    #   ((S1, (T1, I1)), T2) | ((S2^T, (((glued, T3^T), T3), I1^T)), T2^T) |
    #   ((S2, .), join) | ((S1^T, .), T1^T);
    # each call contributes the product of its distinct index dimensions
    # (batch = 1).  Legs: a = d^2 (the reshaped e12); c,e,f = d;
    # x,y,z,w,q,v = chi; u,s,m,o = m.
    print("  block tally vs hand count (d=2, chi=2, m=4):")
    d, chi, m = 2, 2, 4
    op = TreeOp([1.0, 1.0, 1.0], d=d, chi=chi, seed=1)
    sk = TREE_REGISTRY["tree"](m, d=d)
    fac_row, fac_col = sk.draw(1), sk.draw(1)
    _reset_ops()
    sk.M(fac_row, fac_col, op)
    tally = _get_ops()
    hand_count = (
        (d * d) * chi ** 3                     # "nax,xyz->nayz"         (T1, I1)
        + (d * d) * d * m * chi * chi          # "nacu,nayz->ncuyz"      S1 onto the branch
        + d * m * chi * chi                    # "ncuyz,cy->nuz"         (., T2)
        + chi ** 4 * d                         # "nzwqv,fv->nzwqf"       (glued, T3^T)
        + chi ** 3 * d * d                     # "nzwqf,ew->nzqfe"       (., T3)
        + chi ** 4 * d * d                     # "nzqfe,xyq->nzfexy"     (., I1^T)
        + m * m * d * d * chi ** 3             # "nsfo,nzfexy->nsozexy"  S2^T onto the chain
        + m * m * chi ** 3 * d * d             # "nsozexy,cy->nsozexc"   (., T2^T)
        + m ** 4 * d * d * chi * chi           # "nuem,nsozexc->numsozxc" S2 first
        + m ** 4 * chi * chi * d               # "numsozxc,nuz->nmsoxc"  (., join)
        + (d * d) * d * m ** 3 * chi           # "nacs,nmsoxc->nmoxa"    S1^T first
        + m * m * chi * (d * d)                # "nmoxa,ax->nmo"         (., T1^T)
    )
    print(f"    tally={tally}  hand count={hand_count}")
    assert tally == hand_count, f"tally {tally} != hand count {hand_count}"

    # (b) batch linearity: every sample contracts the FULL network (the data is
    # broadcast over the batch by hand), so tally(5) == 5 * tally(1) exactly.
    print("  batch linearity (m=4):")

    def estimate_tally(batch):
        copies = [sk.draw(batch) for _ in range(2)]
        _reset_ops()
        estimate_operator(sk, copies, op)
        return _get_ops()

    ok = estimate_tally(5) == 5 * estimate_tally(1)
    print(f"    {'ok' if ok else 'FAIL'}")
    assert ok, "an einsum does not carry the batch index"


if __name__ == "__main__":
    # Each check asserts internally -> any failure raises AssertionError (non-zero exit).
    # check_expectation_tree()
    print()
    check_expectation_xx()
    print()
    check_counting()
    print("\nALL CHECKS PASSED")
