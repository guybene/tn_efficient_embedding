"""CODE ATTIC (2026-08-12) -- the identity / all_ones synthetic spectra were PARKED
from exp_xx.py per user directive ("not as interesting as the XX or general case;
remove for now").  To restore: paste MpoHandle + algos_for back into exp_xx.py,
re-add the imports (REGISTRY, make_A, true_value, estimate), restore the CONFIG
entries below, and re-branch get_handle / the sites_list lines on spectrum.

The chain sketch classes themselves (Kron/TT/TTCapped/TTm/Solomonik) are still LIVE in
sketches.py -- they remain the test fixtures for the exact identity isometry argument
(check_unbiased) and the chain cost checks; only their experiment consumer is parked.

CONFIG entries that were removed:
    "spectra": ["xx", "identity", "all_ones"],
    "N_list": [4, 5, 6],       # identity/all_ones sites                (Colab: [4,6,8])
    "base_sites": {"xx": 4, "identity": 5, "all_ones": 5},
"""


class MpoHandle:
    """identity / all_ones as bond-1 MPOs -- the validated chain classes and their
    closed-form costs.  'ma' maps to the chain gadget sketch 'solomonik'."""
    _NAME = {"ma": "solomonik"}

    def __init__(self, cfg, kind, sites):
        from sketches import make_A
        self.cfg, self.kind, self.sites = cfg, kind, sites
        self.A = make_A(kind, cfg["d"], sites, "cpu")

    def true(self, K):
        from sketches import true_value
        return true_value(self.kind, self.cfg["d"], self.sites, K)

    def make(self, name, m, angles=None):
        from sketches import REGISTRY
        cfg, name = self.cfg, self._NAME.get(name, name)
        if name == "kron_pauli":
            raise ValueError("kron_pauli is XX-only (see algos_for)")
        return REGISTRY[name](cfg["d"], self.sites, m, cfg["dist"], "cpu")

    def estimate(self, sk, facs):
        from sketches import estimate
        return estimate(sk, facs, self.A)

    def cost(self, sk, K):
        return sk.cost(K)


def algos_for(cfg, spectrum):
    """kron_pauli (Noa's partial basis) runs ONLY on the xx spectrum, where the Pauli
    bases carry the physics meaning; every other method runs everywhere."""
    return [a for a in cfg["algos"] if spectrum == "xx" or a != "kron_pauli"]
