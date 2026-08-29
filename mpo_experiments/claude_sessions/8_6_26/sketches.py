# @title
"""
Tensor-network sketches for the trace estimators of the two experiments.

The DATA is always the operator A itself, given as a symmetric tensor network:
  * TreeOp (exp_general_tn): A = the 10-node network of the 5-node tree T glued to
    its conjugate copy along the glue leg g (PSD by construction, rank <= g).
  * XX_MPS (exp_xx): A = rho_A of the XX ground state as an MPO: mpo_cores =
    per-site Kronecker MPS.MPS^* cores (bond chi^2, cut closed at build).
    mps_cores (the factor W, traced-out B side absorbed into the last core) are
    kept for GROUND TRUTH only.  Never expanded.

ONE ESTIMATOR (`estimate_operator`; the MPS experiment was REMOVED 2026-08-28:
given the factor W, Tr((W W^H)^k) is exactly computable from the chi x chi Gram
W^H W, so estimating it is unnecessary): the input is A's network.  Every block
    M_r = S_r A S_{r+1}^H
is contracted over the FULL network, fresh per block, and E_k = Tr(prod_r M_r),
S_{K+1} = S_1.  For the XX operator the input is the MPO (op.mpo_cores).
Families: TREE_REGISTRY (tree, kron, tt_capped) and XX_REGISTRY (ours, tt_capped,
kron, kron_pauli -- the last is Noa's partial basis, XX-only).

Operation counter: every contraction in an estimator goes through `_ec`, which adds
the EXACT cost of that pairwise contraction (product of all its distinct index
dimensions) to a global tally exposed via `_reset_ops`/`_get_ops`.  (Drawing the
factors is NOT counted: sketch construction, not estimation.)  Per-sample cost =
`estimate_cost` (the measured tally).  All correctness and counting checks live in
test_sketches.py.
"""
import math
import torch

from tenpy.models.xxz_chain import XXZChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

DTYPE = torch.float64
CDTYPE = torch.complex128     # complex twin of DTYPE (kron_pauli draws complex Pauli eigenvectors)

# ----------------------------------------------------------------------------
# exact operation counter (sum of free*contracted*free over the estimator's contractions)
# ----------------------------------------------------------------------------
_OPS = 0


def _reset_ops():
    global _OPS
    _OPS = 0


def _get_ops():
    return _OPS


def _ec(eq, *ops):
    """torch.einsum that also adds the exact cost (product of all distinct index dims) of
    this pairwise contraction to the global op tally.  Index labels are single chars."""
    global _OPS
    sizes = {}
    for spec, op in zip(eq.split("->")[0].split(","), ops):
        for ch, dim in zip(spec, op.shape):
            sizes[ch] = dim
    f = 1
    for v in sizes.values():
        f *= v
    _OPS += int(f)
    return torch.einsum(eq, *ops)


# ----------------------------------------------------------------------------
def _rand(shape, dist, device):
    """Mean-0, var-1 core entries.  rademacher: sigma4=1 ; gaussian: sigma4=3."""
    if dist == "rademacher":
        return torch.randint(0, 2, shape, device=device, dtype=DTYPE).mul_(2).sub_(1)
    if dist == "gaussian":
        return torch.randn(shape, dtype=DTYPE, device=device)
    raise ValueError(dist)


def _pauli_table(device="cpu"):
    """The 4-vector table of Noa's d=2 PARTIAL BASIS (feldman2022entanglement Eq. 29) --
    sqrt(2) * the eigenvectors of sigma_x and sigma_y -- rotated by her basis unitary
    at the XX best case (phi, theta) = (pi/2, pi/2) (her Sec. IV C, Eq. 55, Table I),
    U = exp(i (pi/2) sx/2) exp(i (pi/2) sy/2); SU(2) HALF-ANGLE convention.  Eq. 55
    prints the exponents WITHOUT the 1/2, but under that reading (pi/2, pi/2) gives
    U = (i sx)(i sy) = -i sz, which maps the table to itself up to a global phase --
    provably a no-op on the estimator -- contradicting Table I's best != worst.  With
    half angles (pi/2, pi/2) rotates the sampled pair of bases from {sx, sy} to
    {sy, sz}.  Isotropy is rotation-invariant: E[|v><v|] = U I U^H = I, so the
    estimator stays unbiased."""
    angle = math.pi / 4                       # (pi/2)/2, the half angle in the exponent
    c, s = math.cos(angle), math.sin(angle)
    Ux = torch.tensor([[c, 1j * s], [1j * s, c]], dtype=CDTYPE)   # exp(i angle sx)
    Uy = torch.tensor([[c, s], [-s, c]], dtype=CDTYPE)            # exp(i angle sy)
    tbl = torch.tensor([[1, 1], [1, -1], [1, 1j], [1, -1j]],
                       dtype=CDTYPE)                       # sqrt(2) * eigvecs of sx, sy
    return (tbl @ (Ux @ Uy).T).to(device)                  # rows v -> U v


# ============================================================================
# ============================================================================
#  THE TWO OPERATORS AND THEIR SKETCHES
#
#  Operator form: blocks M_r = S_r A S_{r+1}^H over the FULL network.
#  State form (XX only): v_r = S_r B once per copy, chained through env_gram.
# ============================================================================
# ============================================================================

# ----------------------------------------------------------------------------
# The general tree operator (exp_general_tn)
# ----------------------------------------------------------------------------
class TreeOp:
    """The general symmetric tensor-network operator of experiment 1.

    A is the 10-node network: the 5-node TREE below glued to its conjugate copy
    along the glue leg g -- so A is PSD with rank(A) <= g.  `spectrum` (MANDATORY,
    nonnegative; its length IS the glue dimension g) is placed on the glue leg,
    A = H diag(spectrum) H^H.  Every core is a random ISOMETRY -- chi <= d and
    g <= chi^2 are ASSERTED at build, never worked around -- so H has orthonormal
    columns and spectrum's entries ARE A's nonzero eigenvalues EXACTLY:
    rank = #nonzero, rho = sum/max, chosen directly.  Pass all ones for the flat
    case.  Ground truth (true_pk, intrinsic_dim) is computed from the input
    spectrum itself -- nothing is materialized; dense_half() is the test hook.

        e1   e2
         \\   /            dangling (physical) legs: {e1, e2} on T1 -- RESHAPED at
        T1[d^2,x]          build into ONE leg e12 of size d^2 (alg:trace step 3),
           |x              e3 on T2, e4 on T3 -- so each side presents THREE
        I1[x,y,z] --y-- T2[d,y]      dangling legs (d^2, d, d);
           |z                        I1, I2: INTERNAL nodes (no dangling legs)
        I2[z,w,g] --w-- T3[d,w]
           |g  <-- glued to the conjugate copy

    A itself is a SINGLE tree of 2N-1 = 9 nodes: the two copies' I2 cores merge
    into one center node when the glue edge is contracted -- which happens HERE, at
    build.  Every OTHER node stays separate: the intermediate tensors are part of
    the input, and each algorithm block contracts the full 9-node network.  Stored:
        T1 (e12, x)   T2 (e3, y)   I1 (x, y, z)   T3 (e4, w)           row side
        center (z, w, q, v)  the MERGED node: both I2's, spectrum on glue, summed
        T1_col (f12, xc)  T2_col (f3, yc)  I1_col (xc, yc, q)  T3_col (f4, v)
            column side (stored conjugated -- it is data)

    The FIXED contraction path of every algorithm block (ASYMMETRIC: T1 first on
    the rows, T3^T first on the columns; only the dimension-tree keying of the
    sketches is symmetric):
        ((T1, I1), T2)                          -> S1
        ((((glued, T3^T), T3), I1^T), T2^T)     -> S2^T
        join the two subtensors                 -> S2
        (., T1^T)                               -> S1^T
    How each sketch family resolves its slot (ours ((S, v), u); TT ((Su, u),
    (Sv, v)); kron sketches all dangling edges first) is fixed in the respective
    M docstrings.
    """
    n_legs = 3

    def __init__(self, spectrum, d=2, chi=3, seed=0, device="cpu"):
        lam = torch.as_tensor(spectrum, dtype=DTYPE)
        assert lam.ndim == 1 and (lam >= 0).all(), "spectrum: 1-D, nonnegative"
        g = lam.shape[0]                             # glue dimension = len(spectrum)
        gen = torch.Generator().manual_seed(seed)

        def cplx(*sh):
            re = torch.randn(*sh, generator=gen, dtype=DTYPE)
            im = torch.randn(*sh, generator=gen, dtype=DTYPE)
            return (re + 1j * im).to(CDTYPE)

        def draw(phys_shape, glue_dim):
            """One core as a map glue -> phys legs: a random ISOMETRY (QR, orthonormal
            columns).  Requires the output space to be at least as large as the input;
            asserted, never worked around."""
            n_rows = 1
            for s in phys_shape:
                n_rows *= s
            assert n_rows >= glue_dim, \
                "core cannot be an isometry: need chi <= d and len(spectrum) <= chi^2"
            return torch.linalg.qr(cplx(n_rows, glue_dim))[0].reshape(*phys_shape,
                                                                      glue_dim)

        T1 = draw((d * d,), chi)                     # (e12, x): map x -> the RESHAPED
                                                     # {e1,e2} leg of size d^2
        T2 = draw((d,), chi)                         # (e3, y):      map y -> e3
        I1 = draw((chi, chi), chi)                   # (x, y, z):    map z -> x,y
        I2 = draw((chi, chi), g)                     # (z, w, g):    map g -> z,w
        T3 = draw((d,), chi)                         # (e4, w):      map w -> e4
        self.T1, self.T2, self.I1, self.T3 = (T1.to(device), T2.to(device),
                                              I1.to(device), T3.to(device))
        # the merged center node: the two copies' I2 cores, spectrum on the glue
        # edge, summed AT BUILD
        self.center = torch.einsum("zwl,l,qvl->zwqv", I2, lam.to(CDTYPE),
                                   I2.conj()).to(device)
        self.T1_col = T1.conj().to(device)
        self.T2_col = T2.conj().to(device)
        self.I1_col = I1.conj().to(device)
        self.T3_col = T3.conj().to(device)
        self.d, self.chi, self.g, self.device = d, chi, g, device
        self._I2 = I2.to(device)                     # kept for dense_half (test only)
        # isometric cores (asserted) => H^H H = I, so the input IS A's nonzero
        # spectrum -- nothing is materialized
        self.spectrum = lam.to(device)

    def true_pk(self, k):
        return float((self.spectrum ** k).sum())                   # Tr(A^k), exact

    def intrinsic_dim(self):
        return float(self.spectrum.sum() / self.spectrum.max())    # Tr(A)/||A||_2

    def dense_half(self):
        """The half map H (d^4 x g) -- TEST ONLY, at test sizes: contract ONE copy
        of the 5-node tree (T1, T2, I1, I2, T3), glue leg open.  Its columns are
        orthonormal (isometric cores, asserted at build): H^H H = I_g, so
        A = H diag(spectrum) H^H has spectrum's entries as its exact nonzero
        eigenvalues."""
        half = torch.einsum("px,cy,xyz,zwl,ew->pcel",
                            self.T1, self.T2, self.I1, self._I2, self.T3)
        return half.reshape(self.d ** 4, self.g)


class TreeSketch:
    """Interface for sketches of TreeOp.

    A block is M(fac_row, fac_col, op) = S_row A S_col^H, contracted as ONE sweep
    of A's tree per block, nothing reused across blocks.  A's center node already
    contains the glue contraction (done at build), so no generating-half object
    ever exists anywhere in the algorithms."""

    def __init__(self, m, d=2, dist="rademacher", device="cpu"):
        self.m, self.d, self.dist, self.device = m, d, dist, device

    def sketch_size(self):
        raise NotImplementedError

    def draw(self, B):
        raise NotImplementedError

    def M(self, fac_row, fac_col, op):
        """Sketches a single data tensor A to create M=S_iAS_I^T"""
        raise NotImplementedError

    def chunk(self, K):
        return 4096


class TreeOurs(TreeSketch):
    """alg:trace on the tree: one plain order-3 random core per dimension-tree node
    (std 1/sqrt(m)).  S1 (d^2, d, m) merges the reshaped {e1,e2} leg e12 with e3;
    S2 (m, d, m) merges S1's output with e4 and carries the sketch leg.  rows = m.
    (The {e1,e2} -> d^2 reshape is part of the OPERATOR build, alg:trace step 3.)"""
    name = "tree"

    def sketch_size(self):
        return self.m

    def draw(self, B):
        d, m, sc = self.d, self.m, 1.0 / math.sqrt(self.m)
        S1 = _rand((B, d * d, d, m), self.dist, self.device) * sc  # (e12, e3, mu1)
        S2 = _rand((B, m, d, m), self.dist, self.device) * sc      # (mu1, e4, mu)
        return S1, S2

    def M(self, fac_row, fac_col, op):
        """One block along the FIXED ASYMMETRIC contraction path (brackets = order;
        ^T marks column-side cores; row and column sides are contracted in
        DIFFERENT orders -- T1 first on the rows, T3^T first on the columns; only
        the dimension-tree keying of the sketch cores is symmetric).  Each sketch
        slot resolves as ((S, v), u): the sketch core contracts the running
        operand FIRST, then the incoming one:
            ((S1, (T1, I1)), T2)
            ((S2^T, (((glued, T3^T), T3), I1^T)), T2^T)   (f4 closes; s, nu open)
            ((S2, .), join with the first subtensor)
            ((S1^T, .), T1^T)"""
        S1r, S2r = fac_row[0].to(CDTYPE), fac_row[1].to(CDTYPE)
        S1c, S2c = fac_col[0].conj().to(CDTYPE), fac_col[1].conj().to(CDTYPE)
        n = S1r.shape[0]                # every sample contracts the FULL network:
        T1b = op.T1.unsqueeze(0).expand(n, -1, -1)           # data batched by hand
        Cb = op.center.unsqueeze(0).expand(n, -1, -1, -1, -1)
        a = _ec("nax,xyz->nayz", T1b, op.I1)            # (T1, I1)
        a = _ec("nacu,nayz->ncuyz", S1r, a)             # ((S1, .),  -- S1 first
        a = _ec("ncuyz,cy->nuz", a, op.T2)              #  ., T2)
        t = _ec("nzwqv,fv->nzwqf", Cb, op.T3_col)       # (glued, T3^T)
        t = _ec("nzwqf,ew->nzqfe", t, op.T3)            # (., T3)
        t = _ec("nzqfe,xyq->nzfexy", t, op.I1_col)      # (., I1^T)
        t = _ec("nsfo,nzfexy->nsozexy", S2c, t)         # ((S2^T, .),  -- S2^T first
        t = _ec("nsozexy,cy->nsozexc", t, op.T2_col)    #  ., T2^T)
        t = _ec("nuem,nsozexc->numsozxc", S2r, t)       # ((S2, .),  -- S2 first
        t = _ec("numsozxc,nuz->nmsoxc", t, a)           #  ., join the subtensors)
        t = _ec("nacs,nmsoxc->nmoxa", S1c, t)           # ((S1^T, .),  -- S1^T first
        return _ec("nmoxa,ax->nmo", t, op.T1_col)       #  ., T1^T)  -> (n, mu, nu)


class TreeKron(TreeSketch):
    """Kronecker probe: one independent vector PER DANGLING EDGE -- THREE per side
    after the build reshape: d^2 for e12, d for e3, d for e4.  rows = 1, m ignored."""
    name = "kron"

    def sketch_size(self):
        return 1

    def draw(self, B):
        d = self.d
        return [_rand((B, d * d), self.dist, self.device),   # e12 (the reshaped pair)
                _rand((B, d), self.dist, self.device),       # e3
                _rand((B, d), self.dist, self.device)]       # e4

    def M(self, fac_row, fac_col, op):
        """One block: FIRST sketch every dangling edge (each probe vector folded
        onto its own node), THEN run the same FIXED ASYMMETRIC path as TreeOurs.M
        on the sketched nodes:
        ((T1s, I1), T2s) | ((((glued, T3s^T), T3s), I1^T), T2s^T) | join | T1s^T."""
        gr = [f.to(CDTYPE) for f in fac_row]
        gc = [f.conj().to(CDTYPE) for f in fac_col]
        t1 = _ec("na,ax->nx", gr[0], op.T1)             # sketch ALL dangling edges:
        t2 = _ec("nc,cy->ny", gr[1], op.T2)             # T1s, T2s, T3s
        t3 = _ec("ne,ew->nw", gr[2], op.T3)
        u1 = _ec("na,ax->nx", gc[0], op.T1_col)         # T1s^T
        u2 = _ec("nc,cy->ny", gc[1], op.T2_col)         # T2s^T
        u3 = _ec("nf,fv->nv", gc[2], op.T3_col)         # T3s^T
        a = _ec("nx,xyz->nyz", t1, op.I1)               # then the path: (T1s, I1)
        a = _ec("nyz,ny->nz", a, t2)                    # (., T2s)
        t = _ec("zwqv,nv->nzwq", op.center, u3)         # (glued, T3s^T)
        t = _ec("nzwq,nw->nzq", t, t3)                  # (., T3s)
        t = _ec("nzq,xyq->nzxy", t, op.I1_col)          # (., I1^T)
        t = _ec("nzxy,ny->nzx", t, u2)                  # (., T2s^T)
        t = _ec("nz,nzx->nx", a, t)                     # join the two subtensors
        return _ec("nx,nx->n", u1, t).reshape(-1, 1, 1) # (., T1s^T)


class TreeTT(TreeSketch):
    """TT probe over the tree's THREE dangling legs (d^2, d, d) in their natural
    order e12, e3, e4 (bond R).  The literature defines TT probes for chains only;
    this is the mechanical extension -- a probe needs only a leg ORDERING.
    rows = 1.  Core std R^{-(3-1)/(2*3)}, the TT normalization for E[S^H S] = I.
    Base class of the registered tt_capped; not registered itself."""
    name = "tt"

    def sketch_size(self):
        return 1

    def draw(self, B):
        d, R = self.d, self.m
        sc = R ** (-(3 - 1) / (2 * 3))
        return [_rand((B, d * d, R), self.dist, self.device) * sc,
                _rand((B, R, d, R), self.dist, self.device) * sc,
                _rand((B, R, d), self.dist, self.device) * sc]

    def M(self, fac_row, fac_col, op):
        """One block along the same FIXED ASYMMETRIC path as TreeOurs.M; each
        sketch slot resolves as ((Su, u), (Sv, v)): every probe core first folds
        onto the operand carrying ITS OWN edge, then the sketched operands join.
        Row chain c1..c3 (bonds h, i), column chain d1..d3 (bonds p, k)."""
        c1, c2, c3 = [f.to(CDTYPE) for f in fac_row]
        d1, d2, d3 = [f.conj().to(CDTYPE) for f in fac_col]
        n = c1.shape[0]                 # every sample contracts the FULL network:
        T1b = op.T1.unsqueeze(0).expand(n, -1, -1)           # data batched by hand
        Cb = op.center.unsqueeze(0).expand(n, -1, -1, -1, -1)
        a = _ec("nax,xyz->nayz", T1b, op.I1)            # (T1, I1)
        a = _ec("nah,nayz->nhyz", c1, a)                # ((c1, branch),
        b = _ec("nhci,cy->nhiy", c2, op.T2)             #  (c2, T2)) -- then join:
        a = _ec("nhyz,nhiy->niz", a, b)
        t = _ec("nzwqv,fv->nzwqf", Cb, op.T3_col)       # (glued, T3^T)
        t = _ec("nzwqf,ew->nzqfe", t, op.T3)            # (., T3)
        t = _ec("nzqfe,xyq->nzfexy", t, op.I1_col)      # (., I1^T)
        t = _ec("nkf,nzfexy->nkzexy", d3, t)            # ((d3, chain),
        b = _ec("npck,cy->npky", d2, op.T2_col)         #  (d2, T2^T)) -- then join:
        t = _ec("nkzexy,npky->nzexp", t, b)
        t = _ec("nie,nzexp->nizxp", c3, t)              # ((c3, .),
        t = _ec("nizxp,niz->nxp", t, a)                 #  ., join the subtensors)
        b = _ec("nap,ax->npx", d1, op.T1_col)           # ((d1, T1^T),
        return _ec("nxp,npx->n", t, b).reshape(-1, 1, 1)  # ., final join)


class TreeTTCapped(TreeTT):
    """TreeTT with the bond capped at min(m, 4).  The experiments' rMPS baseline."""
    name = "tt_capped"

    def __init__(self, m, d=2, dist="rademacher", device="cpu"):
        super().__init__(min(m, 4), d, dist, device)


TREE_REGISTRY = {"tree": TreeOurs, "kron": TreeKron, "tt_capped": TreeTTCapped}


# ----------------------------------------------------------------------------
# The XX operator (exp_xx)
# ----------------------------------------------------------------------------
class XX_MPS:
    """The operator A = rho_A = Tr_B |psi><psi| of an MPS, as an MPO:
      * mpo_cores -- THE INPUT: NA order-4 cores (chi_l^2, d, d, chi_r^2), the
        per-site Kronecker product of the factor W's core with its conjugate
        (MPS.MPS^* = MPO, bond chi^2); the last core's right-bond pair is
        contracted at build (closing rho_A = W W^H), leaving a trailing bond 1.
      * mps_cores -- GROUND TRUTH ONLY (never an algorithm input): NA cores of
        the factor W, with the traced-out B side E = F F^T absorbed into the
        last core.  Used for the spectrum and the test-only dense_operator.
    Formed ONCE at build -- forming the operator, uncounted.  Never expanded.
    Ground truth: rho_A's nonzero spectrum = eig(W^H W), a chi x chi Gram
    eigenproblem."""

    def __init__(self, mps_cores, NA, device="cpu"):
        cores = [torch.as_tensor(c, dtype=DTYPE) for c in mps_cores]
        L = len(cores)
        env = torch.ones(1, 1, dtype=DTYPE)
        for n in range(L - 1, NA - 1, -1):           # contract the whole B side
            env = torch.einsum("xdy,yz,wdz->xw", cores[n], env, cores[n].conj())
        e, U = torch.linalg.eigh(env)                # E = F F^T at the cut ...
        F = U @ torch.diag(e.clamp(min=0).sqrt()) @ U.T # Take the sqrt and divide between MPS's
        cores = cores[:NA]
        cores[-1] = torch.einsum("xdy,yz->xdz", cores[-1], F)   # ... absorbed
        self.mps_cores = [c.to(device) for c in cores]
        self.mpo_cores = []
        for c in cores[:-1]:
            k = torch.einsum("xdy,wez->xwdeyz", c, c.conj())
            self.mpo_cores.append(k.reshape(c.shape[0] ** 2, c.shape[1], c.shape[1], c.shape[2] ** 2).to(device)) # Reshape the chi^2=D
        c = cores[-1]
        k = torch.einsum("xdj,wej->xwde", c, c.conj())          # cut closed at build
        self.mpo_cores.append(k.reshape(c.shape[0] ** 2, c.shape[1],
                                        c.shape[1], 1).to(device))
        gram = torch.ones(1, 1, dtype=DTYPE, device=device)
        for c in self.mps_cores:
            gram = torch.einsum("xw,xdy,wdz->yz", gram, c.conj(), c)
        self.spectrum = torch.linalg.eigvalsh(gram).clamp(min=0)
        self.n_sites, self.d = NA, cores[0].shape[1]
        self.chi = self.mps_cores[-1].shape[2]
        self.device = device

    def true_pk(self, k):
        return float((self.spectrum ** k).sum())                   # Tr(A^k), exact

    def intrinsic_dim(self):
        return float(self.spectrum.sum() / self.spectrum.max())    # Tr(A)/||A||_2

    def dense_operator(self):
        """A materialized (d^n_sites square) -- TEST ONLY, at test sizes."""
        ket = self.mps_cores[0].reshape(-1, self.mps_cores[0].shape[2])
        for c in self.mps_cores[1:]:
            ket = torch.einsum("px,xdy->pdy", ket, c).reshape(-1, c.shape[2])
        return ket @ ket.conj().T


def build_xx_mps(NA, device="cpu"):
    L = 2 * NA
    model = XXZChain({"L": L, "Jxx": 2.0, "Jz": 0.0, "hz": 0.0,
                      "bc_MPS": "finite"})
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up", "down"] * (L // 2),
                                 bc="finite")
    dmrg.run(psi, model, {"trunc_params": {"svd_min": 1e-10},
                          "max_E_err": 1e-12, "mixer": True})
    op = XX_MPS([psi.get_B(i, form="B").to_ndarray() for i in range(L)], NA, device)
    lam = torch.as_tensor(psi.get_SL(NA), dtype=DTYPE) ** 2   # rho_A's eigenvalues
    assert torch.allclose(op.spectrum.sort().values, lam.sort().values,
                          atol=1e-8), "XX_MPS spectrum != TeNPy Schmidt spectrum"
    return op


# ----------------------------------------------------------------------------
# XX sketches.  M(fac_row, fac_col, op): the input is op.mpo_cores; one L->R
# sweep per block, per site ((W, Srow_n), A_n) then Scol_n (alg:mpop).  Returns
# the block S_row A S_col^H, (batch, rows, rows).  kron sketches EVERYTHING
# beforehand (each vector onto its own core), then runs the pure sweep.
# ----------------------------------------------------------------------------
def _dt(c, dt):
    return c if c.dtype == dt else c.to(dt)


class XXSketch:
    def __init__(self, n_sites, chi, m, dist="rademacher", device="cpu"):
        self.n_sites, self.chi, self.m = n_sites, chi, m
        self.dist, self.device, self.d = dist, device, 2

    def rows(self):
        raise NotImplementedError

    def draw(self, B):
        raise NotImplementedError

    def M(self, fac_row, fac_col, op):
        """The block S_row A S_col^H swept over op.mpo_cores ->
        (batch, rows, rows)."""
        raise NotImplementedError

    def chunk(self, K):
        return 2048


class XXKron(XXSketch):
    """Product probe: one Rademacher/Gaussian d-vector per site.  rows = 1.
    Both cases: sketch EVERYTHING beforehand (each site's vector(s) folded onto
    that site's core), then run the pure L->R sweep."""
    name = "kron"

    def rows(self):
        return 1

    def draw(self, B):
        return _rand((B, self.n_sites, self.d), self.dist, self.device)

    def M(self, fac_row, fac_col, op):
        gr, gc = fac_row, fac_col.conj()
        dt = gr.dtype
        ts = []
        for l in range(self.n_sites):                  # sketch ALL sites first:
            t = _ec("nd,bdez->nbez", gr[:, l], _dt(op.mpo_cores[l], dt))
            ts.append(_ec("ne,nbez->nbz", gc[:, l], t))
        W = ts[0]
        for t in ts[1:]:                               # then the pure L->R sweep
            W = _ec("nxy,nyz->nxz", W, t)
        return W                                       # (n, 1, 1)

    def chunk(self, K):
        return 16384


class XXKronPauli(XXKron):
    """Noa's partial-basis probe (d=2) in her XX best-case basis (phi, theta) =
    (pi/2, pi/2) (feldman2022entanglement Table I) -- see _pauli_table.
    Complex, BRA convention: v = S_bar B, so the MPO-case block comes out as
    S_bar_r A S_{r+1}^T -- Noa's bra-ket product."""
    name = "kron_pauli"

    def __init__(self, n_sites, chi, m, dist="rademacher", device="cpu"):
        super().__init__(n_sites, chi, m, dist, device)
        self._tbl = _pauli_table(device)

    def draw(self, B):
        idx = torch.randint(0, 4, (B, self.n_sites), device=self.device)
        return self._tbl[idx]

    def M(self, fac_row, fac_col, op):
        return super().M(fac_row.conj(), fac_col.conj(), op)   # BRA convention


class XXTT(XXSketch):
    """TT (rMPS) probe over the sites, internal bond R = m, scalar output.  rows = 1.
    Core std R^{-(n-1)/(2n)} so E[S^T S] = I.  Base class of the registered
    tt_capped; not registered itself."""
    name = "tt"

    def __init__(self, n_sites, chi, m, dist="rademacher", device="cpu"):
        super().__init__(n_sites, chi, m, dist, device)
        self.bond = m

    def rows(self):
        return 1

    def draw(self, B):
        d, R, N = self.d, self.bond, self.n_sites
        sc = R ** (-(N - 1) / (2 * N))
        cores = [_rand((B, d, R), self.dist, self.device) * sc]
        cores += [_rand((B, R, d, R), self.dist, self.device) * sc for _ in range(N - 2)]
        cores += [_rand((B, R, d), self.dist, self.device) * sc]
        return cores

    def M(self, fac_row, fac_col, op):
        """One L->R sweep over op.mpo_cores, per site ((W, crow_n), A_n) then
        ccol_n -- the two probe chains take the row and column legs."""
        dt = fac_row[0].dtype
        cc = [f.conj() for f in fac_col]
        t = _ec("ndr,xdez->nrez", fac_row[0], _dt(op.mpo_cores[0], dt))
        t = _ec("nes,nrez->nrsz", cc[0], t)
        for l in range(1, self.n_sites - 1):
            t = _ec("nrsz,nrdw->nszdw", t, fac_row[l])
            t = _ec("nszdw,zdeq->nsweq", t, _dt(op.mpo_cores[l], dt))
            t = _ec("nsep,nsweq->nwpq", cc[l], t)
        t = _ec("nrsz,nrd->nszd", t, fac_row[-1])
        t = _ec("nszd,zdeq->nseq", t, _dt(op.mpo_cores[-1], dt))
        return _ec("nse,nseq->nq", cc[-1], t).reshape(-1, 1, 1)        # (n, 1, 1)


class XXTTCapped(XXTT):
    """XXTT with the bond capped at min(m, n_sites) (the Schmidt limit -- beyond it
    extra bond is rank-deficient cost)."""
    name = "tt_capped"

    def __init__(self, n_sites, chi, m, dist="rademacher", device="cpu"):
        super().__init__(n_sites, chi, min(m, n_sites), dist, device)
        self.m = m


class XXOurs(XXSketch):
    """OUR sketch (alg:mpop, VERBATIM): N-1 order-3 cores per sample -- S_1 of
    shape (d, d, m) merges the first two sites' physical legs; S_n (m, d, m) for
    n = 2..N-1 merges the running sketch leg with site n+1's leg; the block's
    output legs are S_{N-1}'s.  rows = m.  Core std 1/sqrt(m)."""
    name = "ours"

    def rows(self):
        return self.m

    def draw(self, B):
        d, m, N = self.d, self.m, self.n_sites
        sc = 1.0 / math.sqrt(m)
        cores = [_rand((B, d, d, m), self.dist, self.device) * sc]     # S_1 (e1, e2, mu)
        cores += [_rand((B, m, d, m), self.dist, self.device) * sc
                  for _ in range(N - 2)]                # S_n (mu_{n-1}, e_{n+1}, mu_n)
        return cores

    def M(self, fac_row, fac_col, op):
        """alg:mpop's sweep, verbatim from the paper:
            W <- ((S_1^{(r)} . A_1) . A_2) . S_1^{(r+1)}
            W <- ((W . S_n^{(r)}) . A_{n+1}) . S_n^{(r+1)}   for n = 2..N-1."""
        dt = fac_row[0].dtype
        cc = [f.conj() for f in fac_col]
        t = _ec("nabu,xaez->nbuez", fac_row[0], _dt(op.mpo_cores[0], dt))
        t = _ec("nbuez,zbfw->nuefw", t, _dt(op.mpo_cores[1], dt))
        t = _ec("nuefw,nefv->nuvw", t, cc[0])
        for l in range(1, self.n_sites - 1):
            t = _ec("nuvw,nudp->nvwdp", t, fac_row[l])
            t = _ec("nvwdp,wdeq->nvpeq", t, _dt(op.mpo_cores[l + 1], dt))
            t = _ec("nvpeq,nves->npsq", t, cc[l])
        return t.reshape(t.shape[0], self.m, self.m)    # trailing bond 1 -> (n, m, m)

    def chunk(self, K):
        return max(32, min(8192, 1_000_000 // max(1, self.m ** 2 * self.chi)))


XX_REGISTRY = {"ours": XXOurs, "tt_capped": XXTTCapped,
               "kron": XXKron, "kron_pauli": XXKronPauli}


# ----------------------------------------------------------------------------
# the estimator + the measured per-sample cost
# ----------------------------------------------------------------------------
@torch.no_grad()
def estimate_operator(sk, facs, op):
    """The input is A's network.  E_k = Tr(prod_r M_r) with
    M_r = S_r A S_{r+1}^H contracted over the full network, fresh per block
    (S_{K+1} = S_1).  Works for every sketch of either operator."""
    K, P = len(facs), None
    for r in range(K):
        Mi = sk.M(facs[r], facs[(r + 1) % K], op)
        P = Mi if P is None else _ec("nij,njk->nik", P, Mi)
    return _ec("nii->n", P)


@torch.no_grad()
def estimate_cost(sk, op, K):
    """EXACT per-sample op count of one Tr(A^K) estimate, measured by the _ec
    tally at batch 1 (the tally is batch-linear -- asserted in test_sketches)."""
    facs = [sk.draw(1) for _ in range(K)]
    _reset_ops()
    estimate_operator(sk, facs, op)
    return _get_ops()
