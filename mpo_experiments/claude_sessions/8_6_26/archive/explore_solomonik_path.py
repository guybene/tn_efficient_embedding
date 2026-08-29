"""
EXPLORATION (scratch): the EXPLICIT solomonik M_i = S A S^T network, with every gadget
broken into its two random halves G^(p,1), G^(p,2) (calculate_M.tex Eq. g_def), so the
gadget FORMATION (~m^4 each, currently free in draw()) is visible to opt_einsum.  We then
ask opt_einsum for the optimal contraction and compare its cost to "form-the-gadget-first".

Index layout (sites l=0..N-1, gadgets p=0..N-2), one side:
  X^0          : (phys_0, k0)                 k0 size m       -> G^(0,1).in
  X^l (l>=1)   : (phys_l, a_{l-1}, b_{l-1})    sizes (m1,m2)  -> G^(l-1,1).a , G^(l-1,2).b
  G^(p,1)      : (in_p, a_p, l_p)              (m, m1, m)     in_0=k0 ; in_p=z_{p-1}
  G^(p,2)      : (l_p,  b_p, out_p)            (m, m2, m)     out_{N-2}=n ; else z_p
  A^l          : (phys_l_bra, phys_l_ket)      (d, d)
Output legs: n_bra, n_ket  (both size m).
"""
import math, itertools, torch
import opt_einsum as oe

from sketches import _m1m2, _ec, _reset_ops, _get_ops


def _path_cost(eq, operands, path):
    """opt_einsum path cost in OUR _ec convention (product of all distinct dims per step)."""
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


def build_explicit(d, N, m, seed=0):
    """Return (eq, operands) for M_i = S A S^T with gadgets as separate halves."""
    import string
    g = torch.Generator().manual_seed(seed)
    m1, m2 = _m1m2(m)
    syms = iter(string.ascii_letters + string.digits)   # 62 unique single chars (enough)
    new = lambda: next(syms)

    ops, eqs = [], []

    # physical legs (shared bra/ket through A), and the A cores
    phys_bra = [new() for _ in range(N)]
    phys_ket = [new() for _ in range(N)]

    def side(phys):
        """append one S's operands; return its output leg symbol."""
        # gadget chain bonds z_0..z_{N-3}; out of last gadget = n (returned)
        outsym = None
        in_sym = None
        for p in range(N - 1):
            a_p, b_p, l_p = new(), new(), new()
            in_p = new() if p == 0 else in_sym       # in_0 = k0 (leaf0 bond); else prev z
            out_p = new()                             # z_p (or n for last)
            # leaf X^0 attaches to in_0 ; leaf X^{p+1} attaches to (a_p,b_p)
            if p == 0:
                ops.append(torch.randn(d, m, generator=g)); eqs.append(phys[0] + in_p)  # X^0
            ops.append(torch.randn(d, m1, m2, generator=g)); eqs.append(phys[p + 1] + a_p + b_p)  # X^{p+1}
            ops.append(torch.randn(m, m1, m, generator=g)); eqs.append(in_p + a_p + l_p)   # G^(p,1)
            ops.append(torch.randn(m, m2, m, generator=g)); eqs.append(l_p + b_p + out_p)  # G^(p,2)
            in_sym = out_p
        return in_sym  # = n

    n_bra = side(phys_bra)
    n_ket = side(phys_ket)
    for l in range(N):
        ops.append(torch.randn(d, d, generator=g)); eqs.append(phys_bra[l] + phys_ket[l])

    return ",".join(eqs) + "->" + n_bra + n_ket, ops


def main():
    d, N, m = 6, 4, 4
    assert d ** N > m ** 4, "want d^N > m^4 so opt_einsum can't densify the d^N space"
    m1, m2 = _m1m2(m)
    eq, ops = build_explicit(d, N, m)
    n_ops = len(ops)
    print(f"d={d} N={N} m={m} (m1={m1} m2={m2});  {n_ops} primitive tensors in M_i")
    print(f"first-block tensors (A_0,A_1, bra g0 halves, ket g0 halves) = 6, as expected\n")

    for algo in ("dp", "greedy", "auto-hq"):
        path, info = oe.contract_path(eq, *ops, optimize=algo)
        c = _path_cost(eq, ops, path)
        print(f"  optimize={algo:8s}: _ec-cost = {c:>10d}   (oe flops log10={info.opt_cost:.3g})")

    # reference: form each gadget first (m^4 each), then contract the formed-gadget chain.
    # formation: 2 sides * (N-1) gadgets * (p*a*l*b*out=m*m1*m*m2*m = m^4)
    form = 2 * (N - 1) * (m ** 3 * m1 * m2)
    print(f"\n  forming all gadgets first costs {form} (= 2*(N-1)*m^4) BEFORE any chaining")


if __name__ == "__main__":
    main()
