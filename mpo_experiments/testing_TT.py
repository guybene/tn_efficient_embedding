import torch

@torch.no_grad()
def sample_cores(R, N, d, distribution, device="cpu"):
    rand = ((lambda *s: torch.randint(0,2,s,dtype=torch.float32,device=device)*2-1)
            if distribution=="rademacher"
            else (lambda *s: torch.randn(*s, device=device)))
    return [rand(d, R)] + [rand(R, d, R) for _ in range(N-2)] + [rand(R, d)]

@torch.no_grad()
def tt_inner(a, b):
    """<S_a, S_b> as TT-TT contraction. Both are lists of N cores, bond R, phys d."""
    L = torch.einsum('dr,dR->rR', a[0], b[0])                # (R,R)
    for Ga, Gb in zip(a[1:-1], b[1:-1]):
        L = torch.einsum('rR,rds,RdS->sS', L, Ga, Gb)
    return torch.einsum('rR,rd,Rd->', L, a[-1], b[-1])       # scalar

@torch.no_grad()
def trace_sample(d, N, R, k, distribution, device="cpu"):
    """One sample of Tr(prod_l S_l^T S_l) at A=I, with each S_l a single normalized TT tensor."""
    S = [sample_cores(R, N, d, distribution, device) for _ in range(k)]
    scale = 1.0 / R**(N-1)        # |S_l|^2 has E = d^N after this normalization
    val = torch.tensor(1.0)
    for l in range(k):
        val = val * scale * tt_inner(S[l], S[(l+1) % k])
    return val

trace_sample(2,3,4,2,distribution="rademacher")