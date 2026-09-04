# Handoff — Tensor Tree Sketching paper (session ending 2026-08-02)

**Read this first in the new session.** Ground truth for the paper is the user's
**Overleaf `main.tex`** — every local copy in this folder is stale. Ask for the
current upload before editing anything.

---

## 1. The project in three sentences

We write a rigorous math paper (SIAP/JMLR register): a **general framework —
theory and algorithms — for estimating $\mathrm{Tr}(A^k)$ of a PSD matrix $A$
given in tensor-network form**, via the sketched cyclic estimator
$E_k=\mathrm{Tr}(S_1AS_2^\top S_2A\cdots S_kAS_1^\top)$. The central theorems are
**agnostic to what $A$ is** (only PSD + the $(\varepsilon,\delta)$-embedding
property of the sketch); sample complexity is polynomial in $N,k$ — versus
exponential for Kronecker/product-state probing. The MPO/MPS cases are the
explicit worked examples (motivated by Noa's entanglement problem), never the
headline.

## 2. Papers (all in `papers_for_claude/`) — where to look for what

- **`Noa_paper.pdf`** — Feldman–Kshetrimayum–Eisert–Goldstein, *Entanglement
  estimation in tensor network states via sampling* (PRX Quantum 3, 030312,
  2022). **The physics problem we solve.** RDM moments $p_n=\mathrm{Tr}(\rho_A^n)$
  (Rényi moments), von Neumann entropy as the $n\to1$ limit; their estimator =
  cyclic products of Kronecker/product random vectors, variance $\xi^{N_A}$ —
  exponential. **Fig. 1** is the canonical picture: RDM of an MPS = two copies,
  conjugated copy with physical legs facing **down**, bond $D^2$. Cited
  `feldman2022entanglement`; source for the Rényi→vN limit and the MPS·MPS\*
  stacking narrative.
- **`Linjian Ma.pdf`** — Ma–Solomonik, *Cost-efficient Gaussian tensor network
  embeddings* (NeurIPS 2022, arXiv:2205.13163). **Our engine.** Thm 3.1 =
  $(\varepsilon,\delta)$-accurate sufficient condition (each linearized core's
  row size $\Omega(N_E\log(1/\delta)/\varepsilon^2)$); Cor 3.2 = tree/all-$m$
  special case (we cite Thm 3.1, multi-output, as our `prop:ma`); **Algorithm 1**
  = per-node Kronecker layer + binary-tree merge gadgets — App. C.1 + Fig. 11 +
  cost (C.3) $y_i=a_ib_ic_id_im^2+m^2d_i\sqrt{a_ib_ic_im}\min(\sqrt{a_i},\sqrt{c_i})$;
  Kronecker-data cost $\Theta(\sum s_jm+Nm^{2.5})$. Used for the MPS-case
  algorithm. ⚠ See §5: gadget wiring details.
- **`Kressner.pdf`** — Bujanović–Kressner–Olić, *Stochastic trace estimation
  with tensor train random vectors* (arXiv:2606.15679). **Mathematical
  sibling, different method**: TT structure on the Girard–Hutchinson *probe*
  (not on a sketch $S$); rank $r\ge d{-}1$ ⇒ dimension-independent
  $\ell=O(\varepsilon^{-2}\delta^{-1})$, median-of-means ⇒ $\log(1/\delta)$;
  OSI result; proves no uniform Frobenius-scale variance bound for TT probes.
  $k{=}1$ only — our cyclic $k\ge2$ machinery has no analogue there. Cited
  `bujanovic2026tt`. (Their $d$ = our $N$, their $n$ = our $d$.)
- **`TT Stack.pdf`** — Cazeaux–Dupuy–Figueroa Justiniano, *Linear-scaling
  Tensor Train Sketching* (arXiv:2603.11009). **Style/math resemblance model**
  (their register is close to ours). TTStack sketch (P stacked blocks × rank R),
  OSE/OSI linear in tensor order. **Settled conclusion (do not relitigate):**
  OSE/OSI is the *wrong primitive* for full-rank trace estimation — their $r$
  is the target-subspace dimension, which for us is $d^N$; our element-wise
  union bound pays only the log. Cited `cazeaux2026linear`; appears in related
  work + `rem:osi`.
- **`linear algebra at exponential scale.pdf`** — Camaño–Epperly–Meyer–Tropp
  (arXiv:2606.15350), "Raphael's paper". **Style model + RDM framing
  template.** §3.4.3 (quantum entropy estimation) is the template the user's
  physics paragraph follows: $\psi\in\C^{N_A}\otimes\C^{N_B}$, subsystems
  $A,B$, $\rho_A=\mathrm{Tr}_B(\psi\psi^*)$ via the partial trace operator. At
  that spot they cite Noa [36], Brydges [12], Huang–Kueng–Preskill [52] — we
  copied that citation set. Also: MPS Girard–Hutchinson, GramNyström/Nyström++;
  a remark that $\rho_A$ of an MPS is representable as an MPS of order $2n_A$,
  bond $\chi$ (not an MPO!) — their implementation trick. Cited `camano2026exp`.
- **`mpo mps contraction.pdf`** — Camaño–Epperly–Tropp, *Successive randomized
  compression* (SRC, arXiv:2504.06475). Math-level register for MPS/MPO
  definitions ("MPOs represent operators… matrices or operators on functions").
  Cited `camano2025src` in the MPO definition.
- **`bad_at_kronecker.pdf`** — Meyer–Avron, *Hutchinson's Estimator is Bad at
  Kronecker-Trace-Estimation* (SIMAX 2026). **The exponential wall we beat**:
  Kronecker/rank-1 probes need $\Theta(3^k/\varepsilon^2)$ (real) /
  $\Theta(2^k/\varepsilon^2)$ (complex) worst case; exact variance via partial
  traces/transposes. Cited `meyer2026kronecker`; the "polynomial vs
  exponential" contrast in our intro.
- **`TT_paper.pdf`** — Rakhshan–Rabusseau, *Rademacher Random Projections with
  Tensor Networks* (2021). TT-Rademacher matches TT-Gaussian variance
  ($\frac1k(3(1+2/R)^{N}-1)\|X\|^4$) — justifies Rademacher cores; negative
  result: MPO-shaped RP is not a JLT. Cited `rakhshan2021rademacher`. Also a
  model for "one-sentence physics taste."
- **`lemma_1.lyx`** — the user's own Hebrew k=1 lemma. **Proof-style template**:
  $A=BB^\top$, $\mathrm{Tr}(SAS^\top)=\|SB\|_F^2$, column-by-column, explicit
  steps. All proofs should read like this (but probabilistic — keep the union
  bound).
- (Not in folder, in bib: `camano2025osi` = arXiv:2508.21189, OSI framework —
  one-sided, cited in `rem:osi` as also-wrong-primitive.)

## 3. What has been done (section map of main.tex)

- **Intro**: user's voice. Arc: trace at network level → MPO can't be stored →
  Hutchinson → physics needs $\mathrm{Tr}(A^k)$, $D^k$ blow-up → Noa's cyclic
  Kronecker estimator, exponential (Avron) → we want polynomial in all params →
  Ma linear-in-N precedent → contributions (general theory / general algorithm /
  explicit cases). Some `[#cite]` placeholders remain.
- **Related work**: deferred stub (user will do later, "way more complicated").
- **Preliminaries**: user's definitions. An audit found unused ones
  (`def:unfolding`, `rem:path-cost`, `rem:symorder`, `def:constrained`+remark,
  `cor:trace-power`; trim `def:intermediate`; `def:sketching-tn` has a forward
  dependency on `def:symtn`/`def:ctree` — should move after them). Approved in
  principle; **check whether the user executed it**.
- **Problem statement**: estimator `eq:estimator`, goal `eq:goal` ((ε,δ) +
  efficiency).
- **Theory**: `prop:ma` = Ma Thm 3.1, edge-language, multi-output (no
  single-output restriction — deliberate, discussed); k=1 lemma in lemma_1
  style ($B B^\top$ + Frobenius + union bound over rank, $\delta'=\delta/r$);
  k≥2 `thm:tracek` + `lem:gram` (polarization, $k d_\Pi^2$ events) +
  `lem:master` (multilinear deterministic bound, $\rho=\mathrm{Tr}/\|\cdot\|_2$);
  Gaussian corollaries `cor:k1-gaussian`, `cor:k-gaussian` with $d_\Pi$.
  User was reworking parts of theory by hand — verify current state.
- **Algorithms**: `alg:trace` (general; sample order-3 tensors per node of the
  common dimension tree $T_D$, reuse row/column; `lem:order-free`,
  `lem:correct`, `cor:chain`). **MPO case** (`sec:alg-mpo`), physics-first:
  RDM à la camano2026exp §3.4.3 (user rewrote in own voice: subsystems $A$/$B$,
  $N_A/N_B$, $\rho_A$, $R_k:=\mathrm{Tr}(\rho_A^k)$) → Rényi/vN via Noa →
  `eq:mps-def` (MPS) + `eq:mpo-def` (MPO) + MPS·MPS\* is an MPO (**cited, not
  proved** — two copies, conjugate legs down, bond $\chi^2$) → MPOP cores
  $D^k$ (`eq:mpop-def`, merged-core formula with $\tenA^{\langle k\rangle}$) →
  "estimate $E_k$". `alg:mpop`: left-to-right, per-site $O(dDm^3+d^2D^2m^2)$,
  total $O(kN(dDm^3+d^2D^2m^2)+km^3)$; correctness via corollaries.
  **MPS case** (`sec:alg-mpo-2`): input MPS $\tenB$/vector $b$,
  $A=bb^\top$, sketch the factor: $v_r=S^{(r)}b$,
  $M_r=v_rv_{r+1}^\top$, $E_k=\prod_r\langle v_r,v_r\rangle$ ($k$ dot
  products); sketch = Ma Alg 1 (per-node Kronecker $(d,m)$ tensors →
  App-C.1 gadgets per contraction, reshape splitting the sketch leg into
  $\sqrt{m/\chi}\times\sqrt{m\chi}$); short correctness approved (cite Ma for
  embedding + $(Sb)(Sb)^\top=SAS^\top$ + the two $m$-formulas). ⚠ See §5.
- **Figures**: `mps_mpo_mpop.png` — (a) two hovering MPS layers
  $\mathcal B_i$ / $\mathcal B_i^*$ (legs up/down), (b) MPO, (c) MPOP; user's
  style: blue **#5494EC**, CM fonts, $d/\chi/D$ on every leg/bond, copy
  superscripts in (c), bare panel labels (a)(b)(c). Script:
  `make_mps_mpo_mpop.py`. Also `three_sketches.png` (TT caterpillar /
  Solomonik split-gadget tree / Kronecker) + script.
- **Bibliography**: `citations_added.md` lists everything (8 new:
  vidal2003, eisert2010area, schollwock2011, orus2014, camano2026exp,
  brydges2019probing, huang2020predicting, headrick2010renyi; +18 baseline).
  User's own keys: `camano2025src`, `hubig2017mpo`. Overleaf bib = `refs.bib`.
- **Still stubs**: Experiments (user has them, goes last), Conclusions,
  Related work, second half of 4.2.2 per user's edits.

## 4. Style rules (non-negotiable, learned this session)

- **Macros**: only `tensor_macros_v0.tex` (`\matA`, `\tenB`, `\vv`, `\vb`,
  `\trace(...)` with parentheses not braces, `\TnormS/\FnormS/\Tnorm`,
  `\inner{}{}`, `\kronprod`, `\vectorize`, `\matLambda`). No `\eps`, `\Tr`,
  `\norm`, `\E`, `\SymMat` (use `\operatorname{SymMat}`), no new macro files.
- **$d^N$ explicit** — never introduce $n:=d^N$. (Paper now also uses
  $d_\Pi$ in places — follow whatever the current main.tex does.)
- **Framing**: general framework first; MPO/MPS are examples. Don't repeat
  the general-vs-example line constantly either.
- **Proof style**: like `lemma_1.lyx` — explicit, step-by-step, factor
  $BB^\top$, Frobenius identities; probabilistic results keep the union bound.
- **Cite > prove for background** ("the less you prove the better") — e.g.,
  MPS·MPS\* = MPO is cited (Verstraete/Schollwöck/Noa), not derived.
- **Notation bookkeeping**: subsystem $A$/$B$ à la Noa/camano2026exp; the
  environment was renamed $\overline{A}$→ then user settled on $B$… **check
  current main.tex** — at one point $B$ was freed for MPS cores $\mathcal B_i$.
  Sketch copies $S^{(r)}$, blocks $M_r$, gadgets $g_i^{(r,a)},g_i^{(r,b)}$.
- **Figures**: match the existing style exactly (color #5494EC, CM math fonts,
  label every leg/bond, bare (a)(b)(c) panel labels).
- **Deliverables**: paste-ready LaTeX in MD files (user copies into Overleaf).
  Don't touch files without asking; user often has newer versions.
- **File scope**: only the session folder and one level up.

## 5. Warnings / open issues

- **⚠ Ma gadget wiring (App. C.1)**: my reconstructed contraction order and
  gadget tensor dimensions were **WRONG** (twice). The user fixed the MPS-case
  algorithm **by hand** — their main.tex version is ground truth; do not
  regenerate from `mps_algorithm.md` (stale) or from any reconstruction.
  General rule: never derive-and-assert wiring/shapes not verifiable from text
  actually read — flag uncertainty and ask.
- $\R$-vs-$\C$ seam: theory over $\R$ (transposes), MPO/MPS defs over $\C$;
  physics paragraph carries one "(over $\C$ … conjugate transpose)"
  parenthetical. User aware, unresolved.
- Preliminaries cleanup (removals/reorder) — approved but execution unverified.
- Intro still has `[#cite ...]` placeholders; $N$ vs $N_A$ drift flagged in the
  physics paragraph; `R_k` vs Noa's `p_n` naming flagged.
- `prop:ma` all-$m$ edge condition vs Ma's row-size condition: the MPS-case
  gadget has internal edges smaller than $m$, so its correctness cites Ma
  Thm 3.1 directly, not `prop:ma`.
- Rank-one bonus fact (cut from correctness, available if wanted): for
  $A=bb^\top$, $\rho=1$, so the $\rho^2$ branch of the $k\ge2$ bound is vacuous.

## 6. Local file inventory (this folder; all possibly stale vs Overleaf)

`main_draft.tex` (old), `tensor_macros_v0.tex` (canonical macros),
`refs_draft.bib` (superset mirror of Overleaf `refs.bib`),
`citations_added.md` (bib ledger), `mps_mpo_mpop.png` + `make_mps_mpo_mpop.py`,
`three_sketches.png` + `make_three_sketches.py`, `figures_tikz.tex`,
`mps_algorithm.md` (⚠ stale/wrong gadget details), `mps_to_mpo_continuation.md`,
`prop_ma_theorem31.md`, `eps_delta_proof.tex` (original theory source),
`papers_for_claude/` (all papers above).

## 7. Likely next steps

1. Get current `main.tex`; treat as ground truth; diff against §3 map.
2. Finish MPS-case section per user's corrected algorithm.
3. Experiments section (user has results; goes last).
4. Related work (deferred; threads listed in the current stub).
5. Preliminaries cleanup execution; intro citation placeholders; notation
   seams ($\R/\C$, $N/N_A$).
