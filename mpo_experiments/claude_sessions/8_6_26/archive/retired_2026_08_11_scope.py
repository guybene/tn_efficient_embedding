"""CODE ATTIC (2026-08-11, third pass) -- scope narrowing per user directive:
  * kron_pauli (Noa's partial basis) is used ONLY on the XX operator, where it has its
    physics meaning (HpKronPauli stays live).  The chain and tree variants below were
    retired from sketches.py.
  * the unbounded-bond `tt` was UNREGISTERED everywhere (the experiments start from the
    bounded tt_capped); the TT / HpTT / TreeTT classes remain in sketches.py as base
    classes -- nothing was deleted for those, so nothing is archived here for them.
NOT runnable standalone: paste back into sketches.py to revive.
"""
import torch


# --- was: chain KronPauli(Kron) ---
class KronPauli:  # (Kron)
    """Noa's PARTIAL-BASIS product vectors (feldman2022entanglement, Sec. III C, Eq. 29),
    d = 2 only.  Each site vector is drawn uniformly from the sqrt(d)-scaled eigenvectors of
    d of the d+1 Pauli matrices -- for d = 2 the eigenbases of sigma_x and sigma_y, i.e. the
    four COMPLEX vectors (1, s) with s in {1, -1, i, -i} (E-matrix of their Eq. 41/53).
    E[|v><v|] = I per site, so the estimator stays unbiased; the estimate is Noa's Eq. 23
    cyclic bra-ket product, so M_i = <V_i| A |V_j> CONJUGATES the bra factor (for the real
    sketches conj is a no-op, so the harness convention S_i A S_j^T is unchanged).
    `dist` is ignored (the distribution IS the fixed partial basis).  Known behaviour, used
    as validation targets: zero variance on computational-basis product states (their best
    case); worst case = equal-phase product state, per-site factor 3/2 (their Eq. 48) --
    which is exactly our `all_ones` operator."""
    name = "kron_pauli"

    def __init__(self, d, N, m, dist="rademacher", device="cpu", phi=0.0, theta=0.0):
        from sketches import _pauli_table
        if d != 2:
            raise ValueError("kron_pauli implements the d = 2 Pauli partial basis only")
        super().__init__(d, N, m, dist, device)
        self.phi, self.theta = phi, theta
        self._tbl = _pauli_table(phi, theta, device)

    def draw(self, B):
        idx = torch.randint(0, 4, (B, self.N), device=self.device)
        return self._tbl[idx]                                  # (B, N, 2) complex

    def M(self, gi, A, gj):
        from sketches import CDTYPE
        # <V_i| A |V_j>: conjugate the bra factors, lift A to complex; sweep is inherited.
        return super().M(gi.conj(), A.to(CDTYPE), gj)


# --- was: TreeKronPauli(TreeKron) ---
class TreeKronPauli:  # (TreeKron)
    """Noa's partial-basis probe on the tree (d=2 only), optionally basis-optimized by
    (phi, theta) -- see _pauli_table.  BRA convention: V applies conj(S)."""
    name = "kron_pauli"

    def __init__(self, m, d=2, dist="rademacher", device="cpu", phi=0.0, theta=0.0):
        from sketches import _pauli_table
        if d != 2:
            raise ValueError("kron_pauli implements the d = 2 Pauli partial basis only")
        super().__init__(m, d, dist, device)
        self.phi, self.theta = phi, theta
        self._tbl = _pauli_table(phi, theta, device)

    def draw(self, B):
        idx = torch.randint(0, 4, (B, 4), device=self.device)
        return self._tbl[idx]                                  # (B, 4, 2) complex

    def V(self, fac, op):
        return super().V(fac.conj(), op)
