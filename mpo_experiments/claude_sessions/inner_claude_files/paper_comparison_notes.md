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


---

## 5. 2026-10-09 -- THE comparison set (user's decision; all five re-read in full)

The user states these FIVE are the most relevant papers and THE actual papers to
compare against. Everything else in `papers_for_claude/` is supporting material
(Ma-Solomonik = engine, Meyer-Avron = Kronecker lower bound, Rakhshan-Rabusseau =
TT-Rademacher, Ahle et al. = JL-moment machinery).

1. **Kressner** = Bujanovic-Kressner-Olic, *Stochastic trace estimation with tensor
   train random vectors* (arXiv:2606.15679, `bujanovic2026tt`). Girard-Hutchinson
   with Gaussian TT probes, k=1 only. Thm 5 moment bound E[(u^T w)^{2p}] <=
   (2p-1)!! [(1+2/r)...(1+2(p-1)/r)]^{d-1}; Thm 7 Chebyshev l >= 22 eps^-2 delta^-1 at
   rank r >= d-1; Cor 8 median-of-means O(eps^-2 log 1/delta); Thm 9 rank r = d*p,
   p = 2 ceil(1/2 log(1/delta)) gives l >= 16 e^7 eps^-2 p WITHOUT median-of-means;
   Thm 13 OSI with P = O(eps^-2 max{k, log(2k/delta)}) (factor d better than TTStack);
   Sec 4 Nystrom++ needs the spectral-tail mass tau_k(A); A = I shows NO uniform
   Frobenius-scale variance bound (Var/||I||_F^2 >= 2 n^{d-1}/r). No experiments.
2. **Noa** = Feldman-Kshetrimayum-Eisert-Goldstein (PRX Quantum 3, 030312, 2022,
   `feldman2022entanglement`). SAME cyclic estimator as ours with rank-1 product
   probes (Eq. 23); exact variance Tr((rho_A^{(x)2} E)^n) - p_n^2 with E = (x) E^[alpha]
   (Eqs. 34-41); scaling factor xi_n^{N_A} (Table I: XX xi_2 1.54 / xi_3 1.65 / xi_4
   2.09 best basis); partial-basis + basis optimization (Sec IV C, Eq. 55); benchmarks
   toric-code iPEPS checkerboard and XX half-chain; protocol Fig. 7 rel. error vs
   M/xi^{N_A}, Fig. 8 Var/p^2 vs N_A with fitted base.
3. **TT Stack** = Cazeaux-Dupuy-Figueroa Justiniano, *Linear-scaling Tensor Train
   Sketching* (arXiv:2603.11009 v3, `cazeaux2026linear`). TTStack (P stacked
   Gaussian-TT row blocks of rank R) interpolates Khatri-Rao (R=1) and Gaussian TT
   (P=1); Thm 3.6 OSE R = O(d(r+log 1/delta)), P = O(eps^-2); Thm 3.9 OSI R >= p_F d,
   P = O(eps^-2 (r + log r/delta)) via subspace entanglement C_Q(R) (maximal on
   Kronecker vectors = overwhelming orthogonality); Prop 7.4 fourth-moment bound
   Mom(S) <= sum_I gamma_I ||tr_I(S)||_F^2, sum gamma_I = (1+p_F/R)^d (equality over C);
   Orthogonal TTStack (Stiefel cores) empirically best, no theory. NO trace
   experiments (TT rounding, Hadamard products, LiH Rayleigh-Ritz only).
4. **Khan** = Khan-Pratap-Sharma, *Accurate Trace Estimation with Fewer Random Bits
   via Recursive TensorSketch* (arXiv:2609.18577, Sep 2026). Estimator
   tr(Pi_p A Pi_p^T), Pi_p = Ahle et al. Recursive TensorSketch (p CountSketch leaves
   m x d, p-1 degree-2 TensorSketch nodes m^2 -> m, p a power of 2). Thm 11 unbiased,
   Var <= (10p/m + 100p^2/m^2) tr(A)^2 (layer constants c_i = 2 leaves, 8 nodes);
   Thm 14 (eps,delta) via Chebyshev when m >= 20p/(eps^2 delta); complex variant
   (fourth roots of unity) 4p/m + 16p^2/m^2, m >= 8p/(eps^2 delta); random bits
   O(p(d+m) log m). k=1 only, matvec model, no tensor networks, no experiments.
   Baselines in their Table 1: Hutchinson, Meyer-Avron Kronecker-Hutchinson.
5. **Camano** = Camano-Epperly-Meyer-Tropp, *Linear algebra at exponential scale via
   tensor network dimension reduction* (arXiv:2606.15350, `camano2026exp`). rMPS
   probes (cores N(0,1/chi) bulk, N(0,1/sqrt chi) boundary); Thm 1.2 variance
   Var <= 2(1+1/chi)^{n-1}||A||_F^2 + [3(1+2/chi)^{n-1} - 2(1+1/chi)^{n-1} - 1]||A||_*^2
   (real), tight at A = all-ones; Thm 4.4 fractured Gaussian (Bernoulli(1/chi) cuts)
   -> thesis chi >~ n; Alg 1 MPS Girard-Hutchinson, Alg 2 GramNystrom (8-digit cap
   from the stability shift), Alg 3 Nystrom++ (Thm 3.3/3.4: O(1/eps) matvecs when
   chi >~ n/eps), Alg 4 XNysTrace (empirical winner, NO theory); Sec 3.4.3 von Neumann
   entropy of the 200-site PTFIM half-chain RDM via funNystrom eigenvalues (rho_A as an
   order-2n_A MPS, "not an MPO!"). k=1 only. Test MPOs: classical Ising e^{-beta H}
   (D=2, n=50, beta 0.1/1/10), exp-diag alpha^i D=1, inverse Laplacian D=5, staircase
   D=7. Protocol: 60 trials, median + 10/90 quantiles. Code github.com/chriscamano/TNrandNLA.

What that means for the Experiments section: the k>=2 cyclic estimator has NO
competitor in this set except Noa (rank-1); Kressner/Camano/TT Stack/Khan are all k=1
and enter as (a) the rMPS/TT-probe baseline we run as `tt_capped`, (b) variance
bounds to quote, (c) protocol conventions. Khan has no tensor-network cost model at
all (random bits is its axis).
