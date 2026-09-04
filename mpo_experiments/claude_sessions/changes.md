# changes — contraction path → contraction tree → constrained contraction tree

Three definitions plus a few words in the algorithm lemmas. Input stays a contraction
path; the tree is derived from it; the constrained notion lives at the tree level, like
Solomonik's.

## 1. Contraction path and contraction tree

**Replace** the whole `\begin{definition}[Contraction tree and contraction path]...\end{definition}`
block (`def:ctree`) with:

```latex
\begin{definition}[Contraction path]\label{def:cpath}
A \emph{contraction path} of a tensor network $G=(V,E)$ is a sequence of
$|V|-1$ contractions
$T_0=\bigl((U_1,W_1),\dots,(U_{|V|-1},W_{|V|-1})\bigr)$
in which, starting from each vertex $v\in V$ as its own intermediate tensor,
each step $(U_i,W_i)$ contracts two distinct intermediate tensors, on the
vertex sets $U_i$ and $W_i$, into a single intermediate on $U_i\cup W_i$
(\Cref{def:contraction}), so that after the last step a single tensor remains.
\end{definition}

\begin{definition}[Contraction tree of a path]\label{def:ctree}
The \emph{contraction tree} $T$ of a contraction path $T_0$ of $G=(V,E)$ is
the rooted binary tree whose leaves are the singletons $\{v\}$, $v\in V$, and
in which, for every contraction $(U,W)$ of $T_0$, the vertex $U\cup W$ has
children $U$ and $W$. It is a full binary tree with $|V|-1$ internal vertices
and root $V$, recording the contractions of $T_0$ and forgetting their order;
conversely, every topological sort (children before parents) of a contraction
tree is a contraction path.
\end{definition}
```

(Label `def:ctree` stays on the tree so existing `\Cref{def:ctree}` references compile.)

## 2. Constrained contraction tree

1. **Delete** the current `\begin{definition}[Constrained contraction path]...\end{definition}`
   and keep its trailing `\begin{remark}...\end{remark}` aside (step 3 reuses it verbatim).
2. **Insert** right after `\end{definition}` of `def:sketching-tn` (it references that
   definition, so it must come after):

```latex
\begin{definition}[Constrained contraction tree]\label{def:constrained}
Let $G_{\tenA}=(V_{\tenA},E_{\tenA})$ be a data tensor network with
contraction path $T_0$ (\Cref{def:cpath}) and contraction tree $T$
(\Cref{def:ctree}), and let $G=(V_{\tenA}\cup V_S,E_{\tenA}\cup E_S)$ be the
joint network obtained by attaching an embedding network $G_S=(V_S,E_S)$ to
dangling edges of $G_{\tenA}$ --- e.g.\ samples of a sketching tensor network
(\Cref{def:sketching-tn}) attached to the row and column edges. A contraction
tree $T_B$ of $G$ is \emph{constrained on} $T_0$, written $T_0\subset T_B$,
if for every contraction $(U,W)\in T$ there exists a contraction
$(\hat U,\hat W)\in T_B$ with
\[
  \hat U\cap V_{\tenA}=U
  \qquad\text{and}\qquad
  \hat W\cap V_{\tenA}=W
\]
\cite[Def.~1]{masolomonik2022}.
\end{definition}
```

3. Paste your old remark (verbatim, untouched) right below it.

## 3. A few words in the algorithm lemmas

**Algorithm 1** (`rem:correct`): in the statement, after
`...the $(\varepsilon,\delta)$ guarantee \eqref{eq:goal}.`, add:

```latex
Moreover, the contraction tree executed by \Cref{alg:trace} is constrained on
$T_0$ (\Cref{def:constrained}).
```

In the proof, after `...is exactly the estimator \eqref{eq:estimator}.`, add:

```latex
Finally, each local contraction of $2$--$4$ nodes contains exactly one step
merging the operand containing $U_i$ with the operand containing $W_i$, and
at that step $\hat U\cap V_{\tenA}=U_i$ and $\hat W\cap V_{\tenA}=W_i$; hence
the executed contraction tree is constrained on $T_0$.
```

**Algorithm 2** (`lem:mpop-correct`): in the proof, after
`...the $(\varepsilon,\delta)$ accuracy therefore transfers unchanged.`
(before the `\leanref` lines), add:

```latex
By the same lemma, the executed contraction tree is constrained on the
left-to-right path $T_0$ (\Cref{def:constrained}).
```

**Algorithm 3** (`lem:mps-correct`): in the proof, after
`...the estimator is exactly $E_k=\prod_{r=1}^{k}\inner{\vv_r}{\vv_r}$.`
(before the `\leanref` lines), add:

```latex
Finally, the sketching contraction tree produced by \cite[Alg.~1]{masolomonik2022}
is, by construction, constrained on the given data contraction tree
(\Cref{def:constrained}).
```

(Algorithm 3 cites Ma rather than our Algorithm-1 lemma because it does not instantiate
\Cref{alg:trace} — the constrained property there comes from Ma's construction.)

## ⚠ Knock-on (one wording fix) and options

- `def:dimtree` opens with "Let $T$ be a contraction tree of $G_{\tenA}$" — should now read
  "the contraction tree of a contraction path $T_0$ of $G_{\tenA}$ (\Cref{def:ctree})".
- `rem:path-cost` and the `rem:correct` proof's "$T_0$ consists of $|V_{\tenA}|-1$
  contractions" are exactly right under the merge-only path — no change.
- Optional, one phrase: in the problem statement, "Construct a contraction path on
  $G=(V_{\tenA}\cup V_{S},E_{\tenA}\cup E_{S})$" → append "whose contraction tree is
  constrained on $T_0$ (\Cref{def:constrained})" — makes non-interruption a stated
  requirement that the lemmas then discharge.
