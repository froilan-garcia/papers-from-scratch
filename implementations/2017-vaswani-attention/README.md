# The Transformer from scratch — Vaswani et al. (2017)

Implementation of *Attention Is All You Need* (NIPS 2017). See the
[review](../../reviews/2017-vaswani-attention.md) for the paper's context and
results.

Built incrementally, same as the [Lasso](../1996-tibshirani-lasso/). Current status:

| # | Piece | Status |
|---|-------|--------|
| 1 | Scaled dot-product attention (Eq. 1) + why the $\sqrt{d_k}$ | ⬜ pending |
| 2 | Multi-head attention — projections, reshape, parameter count | ⬜ pending |
| 3 | Causal mask — the decoder's autoregressive property | ⬜ pending |
| 4 | Sinusoidal positional encoding + the linear-shift property | ⬜ pending |
| 5 | Full encoder block (attention + FFN + residual + LayerNorm) | ⬜ pending |
| 6 | Minimal trainable Transformer on a toy task + LR schedule (Eq. 3) | ⬜ pending |
| 7 | Own ablation — Table 3 row (A): vary `h` at constant compute | ⬜ pending |
| 8 | Cross-check against `torch.nn.MultiheadAttention` | ⬜ pending |

Pieces 1–4 are **numpy only** — deliberately, because the whole point is to see
the algebra. `torch` enters at Piece 5.

## Fundamentals — where each piece comes from

Notes to work through before writing code. The idea is not to treat attention as a
black box of three letters (Q, K, V) but to see **what operation it actually is**.

### 1. Attention is a weighted average, and a dot product decides the weights

Stripping away all the notation, attention does this: for each position, it produces
a **convex combination of the values**. The only non-trivial part is where the
weights come from.

    output_i = sum_j  alpha_ij * v_j        with  sum_j alpha_ij = 1,  alpha_ij >= 0

The $\alpha_{ij}$ come from a softmax over the **compatibilities** between query $i$
and key $j$. And the compatibility chosen is the simplest possible: the **dot
product** $q_i \cdot k_j$, which measures alignment. In matrix form (Eq. 1 of the
paper):

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Dimensions, which is where one gets lost: $Q \in \mathbb R^{n_q \times d_k}$,
$K \in \mathbb R^{n_k \times d_k}$, $V \in \mathbb R^{n_k \times d_v}$. Then
$QK^\top \in \mathbb R^{n_q \times n_k}$ (the **attention matrix**, one row per
query), the softmax is **row-wise**, and the output lands in
$\mathbb R^{n_q \times d_v}$. Note that $n_q$ and $n_k$ may differ — which is exactly
what makes encoder-decoder attention possible.

### 2. The $\sqrt{d_k}$ is not cosmetic (footnote 4 of the paper)

This is the detail worth *deriving* rather than memorising. Suppose $q, k$ have
independent components of mean 0 and variance 1. Then:

$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i
\quad\Longrightarrow\quad
E[q\cdot k] = 0, \qquad \mathrm{Var}(q\cdot k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i) = d_k,$$

using independence and $\mathrm{Var}(q_ik_i) = E[q_i^2k_i^2] = E[q_i^2]E[k_i^2] = 1$.

So the **standard deviation of the logits grows like $\sqrt{d_k}$**. With $d_k = 64$
that means logits of typical magnitude $\pm 8$, and differences of that order between
them. A softmax with widely separated logits **saturates**: it approaches a one-hot,
and its Jacobian

$$\frac{\partial\,\mathrm{softmax}(z)_i}{\partial z_j} = \mathrm{softmax}(z)_i(\delta_{ij} - \mathrm{softmax}(z)_j)$$

tends to **zero** when one component $\to 1$ and the rest $\to 0$. Dead gradient, no
learning. Dividing by $\sqrt{d_k}$ returns the variance to 1 and keeps the softmax in
its sensitive regime. **Piece 1 checks this numerically.**

### 3. Multi-head: why it does not cost more

With a single head at dimension $d_{\text{model}}$, the projections would be
$W^Q, W^K, W^V \in \mathbb R^{d_{\text{model}}\times d_{\text{model}}}$. With $h$
heads, each projects to $d_k = d_v = d_{\text{model}}/h$, so each
$W_i^Q \in \mathbb R^{d_{\text{model}}\times d_{\text{model}}/h}$. Stacking the $h$
heads:

    h * (d_model x d_model/h)  =  d_model x d_model

**The same number of parameters.** That is why the paper says the total cost is
similar to that of single-head attention at full dimension. What is gained is that
each head can specialise in a different subspace; what is lost is resolution per
head — hence Table 3(A) showing that **too many heads also hurts**. With
$d_{\text{model}}=512$ and $h=8$: $d_k=d_v=64$.

### 4. The causal mask, and why it is $-\infty$ and not 0

For the decoder to be autoregressive, position $i$ must not see position $j > i$. The
temptation is to zero the weights after the softmax, but that **breaks the
normalisation** (they would stop summing to 1). The paper's solution is to mask
**before**, setting the illegal logits to $-\infty$:

    softmax(...,-inf, ...)  ->  exp(-inf) = 0

That way the zero comes out *of the softmax itself* and the rows still sum to 1. In
code a very negative number (`-1e9`) is used for numerical stability.

### 5. Positional encoding: why sinusoids

Without recurrence or convolution, attention is **permutation-equivariant**: if you
shuffle the input positions, the output shuffles the same way. That is, the model
**does not know the order**. It has to be injected, and the paper adds it to the
embeddings:

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

The property that motivates it: for each pair of dimensions $(2i, 2i+1)$ and with
$\omega_i = 10000^{-2i/d_{\text{model}}}$, a fixed shift $k$ acts as a **rotation**:

$$\begin{pmatrix}\sin(\omega_i(pos+k))\\ \cos(\omega_i(pos+k))\end{pmatrix}
=\begin{pmatrix}\cos\omega_i k & \sin\omega_i k\\ -\sin\omega_i k & \cos\omega_i k\end{pmatrix}
\begin{pmatrix}\sin(\omega_i\,pos)\\ \cos(\omega_i\,pos)\end{pmatrix}$$

The matrix **does not depend on $pos$**, only on the shift $k$ — which is exactly
what the paper means by *"$PE_{pos+k}$ can be represented as a linear function of
$PE_{pos}$"*. That observation is the direct seed of **RoPE** (Track D of the
ROADMAP), which instead of *adding* the encoding **rotates** the queries and keys.
Piece 4 verifies the identity numerically.

## Stack

`numpy` (Pieces 1–4), `torch` (5–8), `matplotlib` for the figures. Nothing outside
the repo's base stack.
