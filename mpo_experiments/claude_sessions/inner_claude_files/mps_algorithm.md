# Paragraph before Algorithm 3

```latex
\paragraph{MPS Algorithm}
We now treat the factored form: the input is an MPS $\tenB$ on $N$ sites
\eqref{eq:mps-def}, representing the vector $\vb\in\R^{d^{N}}$, and the
operator is the rank-one projector $\matA=\vb\vb^{\top}$ of
\Cref{sec:alg-mpo}. Rather than forming the MPO of $\matA$ and running
\Cref{alg:mpop}, we sketch the factor instead of the operator: each
replica block factorizes as
\[
  \matM_{r}
  =\matS^{(r)}\matA\,\matS^{(r+1)\top}
  =\bigl(\matS^{(r)}\vb\bigr)\bigl(\matS^{(r+1)}\vb\bigr)^{\top}
  =\vv_{r}\vv_{r+1}^{\top},
  \qquad
  \vv_{r}:=\matS^{(r)}\vb\in\R^{m},
\]
so the estimator collapses to $k$ dot products,
\[
  E_{k}
  =\trace\bigl(\vv_{1}\vv_{2}^{\top}\,\vv_{2}\vv_{3}^{\top}\cdots
    \vv_{k}\vv_{1}^{\top}\bigr)
  =\prod_{r=1}^{k}\inner{\vv_{r}}{\vv_{r}}.
\]
To compute the sketched vectors $\vv_{r}$ we use the sketching algorithm
of \cite{masolomonik2022}, which attains a lower contraction cost than
the tree sketch of \Cref{alg:mpop}. Their sketch is applied in two
parts. First, a per-node Kronecker step: the physical leg of each site
is sketched by an independent $(d,m)$ tensor, leaving $N$ sketched nodes
$\hat{\tenB}_{n}$. Then, along the left-to-right contraction path, each
contraction $i$ of copy $r$ receives two gadget tensors
$g_{i}^{(r,a)}\in\R^{m\times\sqrt{m/\chi}\times\sqrt{m}}$ and
$g_{i}^{(r,b)}\in\R^{\sqrt{m\chi}\times\sqrt{m}\times m}$, sampled from
$\mu$, which together merge the two sketch legs of size $m$ into a
single one \cite[App.~C.1]{masolomonik2022}. Explicitly, each iteration
performs three contractions (\Cref{rem:contraction-cost}):
\begin{itemize}
  \item \emph{Absorb:} the running tensor $\ten{W}\in\R^{m\times\chi}$
  is contracted with $\hat{\tenB}_{i+1}\in\R^{\chi\times m\times\chi}$
  over the bond, at cost $\chi^{2}m^{2}$, giving a tensor in
  $\R^{m\times m\times\chi}$, which we \emph{reshape} into
  $\R^{m\times\sqrt{m/\chi}\times\sqrt{m\chi}\times\chi}$ by splitting
  the new sketch leg (a free operation).
  \item \emph{First gadget:} contract with $g_{i}^{(r,a)}$ over the leg
  pair $(m,\sqrt{m/\chi})$, at cost $\chi m^{2.5}$, giving a tensor in
  $\R^{\sqrt{m\chi}\times\sqrt{m}\times\chi}$.
  \item \emph{Second gadget:} contract with $g_{i}^{(r,b)}$ over the
  leg pair $(\sqrt{m\chi},\sqrt{m})$, at cost $\chi^{1.5}m^{2}$,
  returning the new $\ten{W}\in\R^{m\times\chi}$.
\end{itemize}
Each iteration therefore costs $\Theta(\chi^{2}m^{2}+\chi m^{2.5})$,
matching \cite[Eq.~(C.3)]{masolomonik2022}, and each copy contracts to
the sketched vector $\vv_{r}\in\R^{m}$.
```

# Final cost paragraph

```latex
\paragraph{Cost of \Cref{alg:mps}.}
For each copy, the Kronecker step contracts $\ten{K}_n^{(r)}$ with the
physical leg of $\tenB_n$ at cost $\chi^{2}dm$ per site
(\Cref{rem:contraction-cost}), and each of the $N-1$ iterations costs
$\Theta(\chi^{2}m^{2}+\chi m^{2.5})$, as itemized above. Summing, a
single sketched copy costs
$O\!\bigl(N(\chi^{2}dm+\chi^{2}m^{2}+\chi m^{2.5})\bigr)$; we run this
$k$ times and finish with $k$ dot products at cost $km$. All in all,
\[
  C_{\mathrm{tot}}
  =O\!\bigl(kN\chi^{2}dm+kN\chi^{2}m^{2}+kN\chi m^{2.5}\bigr),
\]
polynomial in all system parameters, fulfilling the efficiency
requirement of \Cref{sec:problem}. Compared with running \Cref{alg:mpop}
on the MPO form of $\matA=\vb\vb^{\top}$, whose bond dimension is
$D=\chi^{2}$, the factored route replaces the leading term
$kN\,d\chi^{2}m^{3}$ by $kN\chi m^{2.5}$ (for $m\ge\chi^{2}$): sketching
the factor is cheaper by a factor of order $d\chi\sqrt{m}$.
```
