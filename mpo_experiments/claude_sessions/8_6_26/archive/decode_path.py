"""Decode the opt_einsum 'dp' path for the explicit half-network into a human-readable
sequence of pairwise contractions (which named tensors merge, over which bonds)."""
import opt_einsum as oe
from explore_solomonik_path import build_explicit
from sketches import _m1m2


def names_for(N):
    """Human names aligned with build_explicit's operand order."""
    nm = []
    for tag in ("bra", "ket"):
        for p in range(N - 1):
            if p == 0:
                nm.append(f"X0[{tag}]")
            nm.append(f"X{p+1}[{tag}]")
            nm.append(f"G{p}.1[{tag}]")
            nm.append(f"G{p}.2[{tag}]")
    nm += [f"A{l}" for l in range(N)]
    return nm


def decode(d=6, N=4, m=4):
    m1, m2 = _m1m2(m)
    eq, ops = build_explicit(d, N, m)
    ins = eq.split("->")[0].split(",")
    sizes = {ch: dim for s, op in zip(ins, ops) for ch, dim in zip(s, op.shape)}
    path, info = oe.contract_path(eq, *ops, optimize="dp")

    terms = list(ins)                       # index strings
    names = names_for(N)                    # parallel human names
    print(f"d={d} N={N} m={m} (m1={m1} m2={m2});  {len(ops)} primitives\n")
    print("pairwise contraction sequence (dp optimal):")
    for step in info.contraction_list:
        idxs, _, sub, _, _ = step
        # idxs = positions (in current `terms`) being contracted
        merged_names = [names[i] for i in idxs]
        ineq, outeq = sub.split("->")
        contracted = sorted((set(ineq.replace(",", "")) - set(outeq)))
        cdim = "x".join(f"{sizes[c]}" for c in contracted) or "-"
        odim = "x".join(f"{sizes[c]}" for c in outeq) or "scalar"
        for i in sorted(idxs, reverse=True):
            terms.pop(i); names.pop(i)
        terms.append(outeq)
        names.append("(" + " * ".join(merged_names) + ")")
        print(f"  {' * '.join(merged_names):55s}  over[{','.join(contracted)}={cdim}] -> {outeq}({odim})")


if __name__ == "__main__":
    decode()
