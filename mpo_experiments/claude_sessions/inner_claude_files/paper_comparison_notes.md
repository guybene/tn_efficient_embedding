# Experiments-section prep: what the three comparison papers actually do

Read 2026-08-06 (full PDFs) for designing our Experiments section. Sources:
`papers_for_claude/Noa_paper.pdf`, `papers_for_claude/TT Stack.pdf`,
`8_6_26/additional materials/Linear algebra at exponential scale.pdf`.

---

## 1. Noa (feldman2022entanglement) — the direct competitor

**Same estimator topology as ours.** Their p̂_n (Eq. 21–23) is exactly our E_k with
S_i = rank-1 random product vector (complex, Pauli/Weyl-eigenvector frames, norm² = d per
site, cyclic closure reusing |V^(1)⟩ at both ends). Differences are sketch rank and
distribution only, not topology.

**Their protocol (mirror this):**
- Fig. 7: relative error |p_n − est|/p_n vs **M/ξ_n^{N_A}** (log-log); curves for different
  N_A collapse; ~M^{−1/2} decay; smoothing = average over 20 random permutations of the M
  sample stream.
- Fig. 8 + Table I: Var/p_n² vs N_A on log-linear; straight line ⇒ Var/p² = ξ_n^{N_A};
  base ξ_n extracted by linear fit. Their measured ξ_n ∈ [1.05, 2.9] (toric code ξ_2≈1.23,
  ξ_3≈1.80, ξ_4≈2.75; XX chain ξ_2≈1.54, ξ_3≈1.65, ξ_4≈2.09, basis-optimized).
- Required samples: M = ξ_n^{N_A}/ε² (Chebyshev only).

**Fairness requirements:**
- Compare against their **basis-optimized partial-basis** variant (Sec. IV C: local rotation
  e^{iφσx}e^{iθσy} scanned to minimize Var(p̂_1)), not a naive version.
- Their best case: weakly entangled states + good basis — partial-basis variance is
  **exactly 0** for computational-basis product states. Their worst cases: equal-phase
  product states ξ_n = (3/2)^n (d=2, partial), full-basis product states (4/3)^n,
  maximally mixed also exponential (Eqs. 43–50).
- Their benchmarks: toric-code iPEPS (D=2, checkerboard N_A = 8/18/32, exact
  p_n = 2^{−(w²/2−1)(n−1)}) and critical XX chain MPS (half-chain N_A = 4…20, CFT formula).
- Exact variance is computable: Var(p̂_n) = Tr((ρ_A^{⊗2}E)^n) − p_n², E = ⊗ per-site E^[α]
  (Eqs. 34–41; partial-basis d=2: E^[α] = I⊗I + ½(σx⊗σx+σy⊗σy)). We can plot their exact
  variance curve, not just sampled.

## 2. TT Stack (cazeaux2026linear) — sketch family, no trace experiments

**No trace estimation anywhere** (experiments = TT-rounding, QTT Hadamard products, one
sketched Krylov eigensolver). Confirms OSE/OSI-wrong-primitive stance: all their guarantees
scale in subspace dimension r (= d^N for full-rank A).

**Their sketch = a (P,R) family at fixed rows m = PR** (their notation: d=order, n=mode;
translate d→N, n→d):
- R=1 → Khatri–Rao; P=1 → Gaussian TT (Ma–Solomonik family, their ref [27]). So TTStack is
  not a new topology for our E_k — it parameterizes between two sketches we already have.
- **Orthogonal TTStack** (Stiefel-manifold cores, Def. 3.4): empirically beat plain TTS in
  every test, **no published theory**. Beating/matching it empirically is meaningful.
- Apply cost to TT of rank χ: O(N d P R χ(R+χ)); MPO-vector variant Sec 5.2.2.

**Portable theory (the right primitive, buried in their Sec 6–7):**
- Strong JLM: single-vector ε-distortion needs only R = O(N log(1/δ)), P = O(1/ε²) — no r.
- **Prop 7.4 fourth moment**: for one Gaussian-TT block, E[Tr(ΩSΩ*)²] ≤ Σ_I γ_I‖tr_I(S)‖_F²
  with Σ_I γ_I = (1+2/R)^N (real), γ_{[N]}=1. ⇒ k=1 Hutchinson-type variance in
  partial-trace norms, /P for the stack. Same structural object as our variance functionals;
  worth citing next to meyer2026kronecker (which they import as their Def. 7.1).
- Failure mechanism = "overwhelming orthogonality" on Kronecker/rank-1 vectors — exactly the
  axis where our kron sketch fails at high K.

**Their protocol:** fix m = PR, sweep R ∈ {1,4,8,16,(32)}; median + 25/75 IQR over
100/10/1000 trials depending on experiment.

## 3. Camaño–Epperly–Meyer–Tropp (camano2026exp) — closest in spirit, k=1 only

**Estimators** (all with rMPS probes: Gaussian MPS, bond χ, cores N(0,1/χ) bulk /
N(0,1/√χ) boundary, isotropic; central thesis: works iff **χ ≳ n**):
- MPS Girard–Hutchinson (Alg. 1), MPS GramNyström (Alg. 2, eigenvalues-only path,
  ~8-digit cap from stability shift), MPS Nyström++ (Alg. 3), MPS XNysTrace (Alg. 4 —
  their empirical winner, **explicitly no theory**), funNyström entropy from GramNyström
  eigenvalues.
- Variance (Thm 1.2/4.6, real): Var(ωᵀAω) ≤ 2(1+1/χ)^{n−1}‖A‖_F² +
  [3(1+2/χ)^{n−1} − 2(1+1/χ)^{n−1} − 1]‖A‖_*². **Tight at A = all-ones** — our `all_ones`
  test operator is literally their tightness witness. Nuclear-norm term = their structural
  weakness on flat spectra (identity!): ‖A‖_* can be d^{n/2}‖A‖_F.
- Fractured-Gaussian picture (Thm 4.4): rMPS first four moments = Kronecker product with
  Bernoulli(1/χ) cuts per bond; P(no cut) = (1−1/χ)^n explains the χ ≍ n threshold.

**They never estimate Tr(A^k) or Rényi entropies.** §3.4.3/Fig. 4 does von Neumann entropy
of a 200-site PTFIM half-chain RDM via GramNyström eigenvalues (ρ_A represented as an
order-2n_A MPS of bond ξ, "not an MPO" — App. A). **Rényi-k via cyclic sketching vs their
methods is an open, unclaimed comparison.** To make them do Tr(A^k): GH on A^k needs k
successive MPO–MPS products with bond growth D^k·χ (or truncation bias they never analyze)
— this is precisely our D^k blow-up narrative; or funNyström-style Σλ̂_i^k (tail-biased).

**Reusable test systems with exact ground truth:**
- Classical 1D Ising e^{−βH}: exact MPO, D=2, n=50, β ∈ {0.1, 1, 10} = flat/medium/decaying
  spectrum (Fig. 3). Flat spectrum is their bad regime; fast decay their good one.
- Synthetic 2^50 psd MPOs (Fig. 5): exponential diag α^i, α=0.7 (D=1); inverse Laplacian
  (D=5); staircase (D=7).
- PTFIM n=200, h=1.1 ground state via DMRG2 (tol 1e−8); half-chain RDM (Fig. 4).
- Code: https://github.com/chriscamano/TNrandNLA

**Their protocol:** 60 trials, median + 10%/90% quantiles, relative trace error vs number
of matrix–MPS accesses (t up to 200); probe χ ∈ {1,4,8,16} in practice (theory says χ≳n but
they call that pessimistic; Fig. 1 uses χ=4 at n=70).

---

## 4. Implications for our Experiments section

**Positioning.** Three distinct comparisons, three distinct claims:
1. **vs Noa (kron)**: same topology, rank-1 sketch → exponential ξ_n^{N_A}; ours polynomial.
   Plot Var/est² vs N with fitted base (their Fig. 8 style) and relative error vs samples.
   Use their basis-optimized partial-basis as the strong baseline; include a state where
   their method excels (weakly entangled / computational-basis-like) for honesty.
2. **vs Camaño (rMPS GH / XNysTrace on A^k)**: k=1-native methods forced to k>1 pay D^k
   bond growth per query or truncation bias. Count A-applications AND FLOPs (our exact op
   counter in `sketches.py` / `cost_per_algorithm.md` gives the x-axis). Reuse their test
   MPOs for ground truth; identity/flat spectrum is their weak spot (nuclear-norm term),
   fast-decay their strength — show both.
3. **vs TT Stack**: no trace experiments to compare against; treat as sketch-family
   ablation — at fixed m, sweep between KR-like and TT-like structure (we already have
   kron/tt/ttm/solomonik/dense); optionally add their Orthogonal-core variant as an
   empirical curve. Cite Prop 7.4 in theory discussion, not experiments.

**Protocol conventions to adopt (common to all three papers):** median + quantile bands
(10/90 or 25/75) over ≥60 trials; log-log relative-error plots; error vs both #samples and
#A-accesses/FLOPs; fitted exponential-base plots for variance vs N.

**Assets already in place (8_6_26/):** five sketches with exact per-iteration cost formulas
validated against op counter (`cost_per_algorithm.md`, `sketches.py`); drivers
`theorem_assert.py` (m* vs N/K/C) and `embedding_compare.py` (cost-to-target); Colab
harness. Current test operators: identity (flat spectrum — Camaño-hard), all_ones
(their variance-bound tightness witness), both bond-1 MPOs.
