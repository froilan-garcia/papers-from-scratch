# Attention Is All You Need

**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (Google Brain / Google Research / U. Toronto) · **Year:** 2017 · **Venue:** NIPS 2017 (31st Conference on Neural Information Processing Systems), Long Beach, CA · **Link/DOI:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
**Field:** ML / deep learning · **Read:** 2026-07-29

## TL;DR

The authors propose the **Transformer**: an encoder-decoder architecture that **removes recurrence and convolutions entirely** and relies solely on attention mechanisms. The central piece is **scaled dot-product attention**, $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, replicated across $h$ parallel heads. The argument is not only about quality but about **computational complexity**: a recurrent layer needs $O(n)$ sequential operations and connects distant positions through paths of length $O(n)$; self-attention needs $O(1)$ sequential operations and a maximum path of $O(1)$ — everything parallelises and any pair of positions "sees" each other directly. The result: **28.4 BLEU** on WMT14 English→German (more than 2 points above the state of the art, ensembles included) training for 3.5 days on 8 GPUs, a fraction of the competitors' cost. It is the paper on which the whole current LLM ecosystem is built.

## Context and motivation

Around 2017 the state of the art in machine translation consisted of recurrent encoder-decoder architectures (LSTM, GRU) **with attention added** — the line Bahdanau (2014) → Luong (2015) → GNMT (2016). The problem is structural: an RNN produces hidden states $h_t = f(h_{t-1}, x_t)$, and that dependence on $h_{t-1}$ **prevents parallelisation within an example**. With long sequences this is fatal, all the more so because memory constraints limit *batching* across examples.

The convolutional alternatives (ByteNet, ConvS2S, Extended Neural GPU) do parallelise, but the number of operations needed to relate two positions **grows with the distance**: linearly in ConvS2S, logarithmically in ByteNet. That makes long dependencies hard to learn.

The key observation: in those models attention was already doing the heavy lifting of connecting arbitrary positions, but **always alongside an RNN**. The authors ask what happens if the RNN is removed and only the attention is left. The answer — and the title — is that it suffices.

## Methodology

### Scaled dot-product attention (Eq. 1)

Attention maps a **query** and a set of **key-value** pairs to an output, which is a weighted sum of the values; the weight of each value is given by a compatibility function between the query and its key. Packing the queries into $Q$, the keys into $K$ and the values into $V$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Why $\sqrt{d_k}$?** It is the most-cited detail and the paper justifies it in footnote 4: if the components of $q$ and $k$ are independent with mean 0 and variance 1, then $q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ has mean 0 and **variance $d_k$**. With large $d_k$ the dot products blow up in magnitude, pushing the softmax into regions of **tiny gradient** (saturation). Dividing by $\sqrt{d_k}$ renormalises the variance to 1 and avoids this.

Compared with Bahdanau's **additive attention** (which uses a one-layer feed-forward network as the compatibility function), the multiplicative one has similar theoretical complexity but is **much faster and more memory-efficient** in practice, because it reduces to highly optimised matrix multiplication.

### Multi-head attention

Instead of a single attention in dimension $d_{\text{model}}$, $Q$, $K$ and $V$ are linearly projected **$h$ times** with different learned projections, attended to in parallel, and concatenated:

$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,
\qquad \mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Motivation: it allows jointly attending to information from **different representation subspaces** at different positions; with a single head, averaging prevents this. Configuration: $h=8$, $d_k = d_v = d_{\text{model}}/h = 64$. Since each head operates in reduced dimension, **the total cost is similar to that of a single-head attention at full dimension**.

### The three uses of attention in the model

1. **Encoder-decoder attention:** the queries come from the previous decoder layer, the keys and values from the encoder output. Each decoder position attends to the whole input (this is the classical seq2seq attention).
2. **Encoder self-attention:** $Q$, $K$ and $V$ all come from the same place (the previous layer). Each position attends to all of them.
3. **Masked decoder self-attention:** the same, but each position may only attend **up to and including itself**. It is implemented by setting the illegal entries to $-\infty$ *before* the softmax. It preserves the autoregressive property.

### The rest of the architecture

- **Stacks:** $N=6$ identical layers in encoder and decoder. The decoder adds a third sub-layer (the attention to the encoder).
- **Residual + norm:** each sub-layer is wrapped as $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. All sub-layers and embeddings produce $d_{\text{model}} = 512$ so that the residuals fit.
- **Position-wise FFN (Eq. 2):** $\mathrm{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$, applied identically at each position, with inner layer $d_{ff}=2048$. Equivalent to two convolutions of kernel 1.
- **Embeddings:** shared between the two embedding layers and the pre-softmax transformation; multiplied by $\sqrt{d_{\text{model}}}$.
- **Positional encoding:** since there is neither recurrence nor convolution, order has to be injected. They use sinusoids with frequencies in geometric progression from $2\pi$ to $10000\cdot 2\pi$:

  $$
  PE_{(pos,2i)} = \sin\!\left(pos/10000^{2i/d_{\text{model}}}\right), \qquad
  PE_{(pos,2i+1)} = \cos\!\left(pos/10000^{2i/d_{\text{model}}}\right)
  $$

  The hypothesis: for any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$, which would make it easy to learn to attend by relative position. They chose the sinusoidal version (over learned embeddings, which give nearly identical results) because it **might extrapolate** to sequences longer than those seen in training.

### The complexity argument (Sec. 4, Table 1)

This is the theoretical justification of the design, and it deserves careful reading. With $n$ = sequence length, $d$ = representation dimension, $k$ = kernel, $r$ = neighbourhood:

| Layer type | Complexity per layer | Sequential ops. | Maximum path |
|---|---|---|---|
| **Self-attention** | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k\cdot n\cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (restricted) | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |

Three criteria: cost per layer, parallelisability (sequential ops.) and **path length** between long-range dependencies — the shorter the path signals travel forwards and backwards, the easier it is to learn distant dependencies. Self-attention wins on all three except cost per layer, and there it is **cheaper than the recurrent one when $n < d$**, which is the usual case with word-piece or BPE representations.

> 💡 That $O(n^2 \cdot d)$ is exactly the limitation that would define the following decade: Longformer, FlashAttention, Mamba... The paper itself already points to *restricted self-attention* as a way out.

### Training (Sec. 5)

- **Data:** WMT 2014 EN-DE (4.5M pairs, BPE with a shared vocabulary of ~37000 tokens) and EN-FR (36M sentences, word-piece of 32000). Batches of ~25000 source and 25000 target tokens.
- **Hardware:** 8 NVIDIA P100 GPUs. Base: 100K steps ≈ **12 hours**. Big: 300K steps ≈ **3.5 days**.
- **Optimiser (Eq. 3):** Adam with $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$, and the celebrated **warmup schedule**:
$$lrate = d_{\text{model}}^{-0.5}\cdot\min\!\left(step\_num^{-0.5},\ step\_num\cdot warmup\_steps^{-1.5}\right)$$
It rises linearly for the first $warmup\_steps = 4000$ steps and then decays as $1/\sqrt{step}$.
- **Regularisation:** residual dropout $P_{drop}=0.1$ (on the output of each sub-layer and on the embeddings+PE sum) and **label smoothing** $\epsilon_{ls}=0.1$ — which *hurts* perplexity (the model learns to be less certain) but *improves* accuracy and BLEU.
- **Inference:** beam search with beam 4 and length penalty $\alpha=0.6$; averaging of the last 5 checkpoints (base) or 20 (big).

## Main results

- **WMT14 EN-DE (Table 2):** Transformer (big) **28.4 BLEU**, >2.0 above the best previous result *including ensembles*. The **base** model (27.3) already beats everything published, with a training cost of $3.3\times10^{18}$ FLOPs against $\sim10^{20}$ for the competitors — **one or two orders of magnitude cheaper**.
- **WMT14 EN-FR:** **41.8 BLEU** (big), a new single-model state of the art, at less than 1/4 of the previous cost. *(Note: the text of Sec. 6.1 says 41.0 while the abstract and Table 2 say 41.8 — a known inconsistency in the paper.)*
- **Ablations (Table 3)** — the most instructive part:
  - **(A) Number of heads:** a single head is **0.9 BLEU worse** than the best configuration, but **too many heads also hurts**. There is an optimum (8–16).
  - **(B) Reducing $d_k$ hurts**, suggesting that "determining compatibility is not easy" and that a more sophisticated function than the dot product might help.
  - **(C)** Bigger models, better. **(D)** Dropout is very useful against overfitting.
  - **(E) Learned positional embeddings ≈ sinusoids** (25.7 vs 25.8 BLEU dev). The sinusoidal choice was for extrapolation, not performance.
- **Generalisation (Sec. 6.3, Table 4):** a 4-layer Transformer on Penn Treebank *constituency parsing* gives **91.3 F1** with WSJ alone (40K sentences) and **92.7** semi-supervised — beating all previous results except the RNNG of Dyer et al., and **with hardly any task-specific tuning**. Proof that the architecture is not a translation trick.

## Strengths and limitations

**Strengths:** radical simplicity — removing components (recurrence, convolution) and improving results is the best kind of result; the complexity argument of Table 1 is an *a priori* design justification, not a *post hoc* rationalisation; the ablations are honest and rich (including that more heads hurt and that sinusoids add nothing over learned embeddings); efficiency is the real headline (1–2 orders of magnitude fewer FLOPs); and it generalises beyond translation.

**Limitations (some the paper's, others visible only in hindsight):**
- **The $O(n^2)$ bottleneck** in memory and compute with respect to sequence length. The paper acknowledges it and proposes restricted attention as future work; a whole line of later research (Longformer, FlashAttention, S4/Mamba) is born here.
- **Post-norm.** The paper uses $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$, which turned out to be **unstable to train without careful warmup** — hence the schedule of Eq. (3). Modern models use **pre-norm** ($x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$), which is far more stable (Xiong et al. 2020).
- **The title overstates slightly.** Attention is not "all you need": the position-wise FFNs are ~2/3 of the parameters and do essential work, and the residuals, the normalisation and the learning-rate schedule are equally necessary for it to train at all.
- **Superseded components:** the sinusoids have given way to **RoPE**; the FFN's ReLU to **GELU/SwiGLU**; Adam to **AdamW**. None of this detracts from the core.
- **Framed as translation.** The paper does not anticipate that the real application would be **language modelling at scale** (BERT, GPT). The largest scale here is 213M parameters.
- **No theory of why it works:** the justification is empirical and complexity-based; interpretability is dispatched with "the heads seem to learn different tasks" and a few examples in the appendix.

## Implementation ideas

The Transformer is probably **the highest-yield exercise on the whole deep learning track**: implementing it from scratch forces one to understand batched linear algebra, masking and normalisation. A plan in pieces (lasso/Markowitz style):

1. **Scaled dot-product attention (Eq. 1)** in pure numpy, without batching: $QK^\top$, scale, softmax, $\times V$. ~10 lines. **Validate the reason for the $\sqrt{d_k}$**: generate Gaussian $q,k$ and check empirically that $\mathrm{Var}(q\cdot k)= d_k$, and plot how the softmax saturates (and the gradient dies) without the scaling. It is footnote 4 of the paper turned into a figure.
2. **Multi-head attention** with the projections $W^Q, W^K, W^V, W^O$ and the *reshape* into $h$ heads. Verify that the parameter cost is **equal** to that of one head at full dimension (the claim of Sec. 3.2.2).
3. **Causal mask** of the decoder: a triangular matrix with $-\infty$, checking that row $i$ places mass only on columns $\le i$. Visualise the attention matrix before and after.
4. **Sinusoidal positional encoding:** implement it and **draw the heat map** $PE(pos, i)$ (a beautiful figure). Verify numerically the key property: that $PE_{pos+k}$ is a linear combination of $PE_{pos}$ with a matrix independent of $pos$ (it is a 2×2 rotation for each pair of dimensions — a direct connection with RoPE).
5. **A full encoder block:** multi-head + FFN + residual + LayerNorm, in PyTorch. Count parameters and **verify the ~1/3 attention, ~2/3 FFN split**.
6. **A minimal trainable Transformer** on a toy task (copying/reversing sequences, or translating numbers into words). Reproduce the **learning-rate schedule of Eq. (3)** and plot it.
7. **An ablation of my own:** repeat row (A) of Table 3 at toy scale — vary $h \in \{1,2,4,8\}$ at constant compute and check that a single head loses and that too many do too.
8. **Validate against `torch.nn.MultiheadAttention`** with the same weights, checking numerical equality.

## Connections

- **Track A of the [ROADMAP](../ROADMAP.md):** this paper is the destination of the historical line RNN → attention. Reading **Bahdanau (2014)** first — the real origin of attention, where the RNN is still present — makes the leap intelligible as "remove the RNN", which is exactly the thesis. **Luong (2015)** contributes the multiplicative attention that is scaled here.
- **Track D (engineering):** almost all of that track consists of patches to limitations of *this* paper — **RoPE** (replaces the sinusoids), **FlashAttention** (attacks the $O(n^2)$ in memory), **AdamW** (replaces Adam+warmup), **Layer Normalization** (the pre/post-norm question), **multi-query attention** (cheapens inference).
- **Track B (tokenisation):** the paper uses **BPE** (Sennrich 2015) and **word-piece** (GNMT 2016) without discussing them; they are prerequisites for understanding the input layer.
- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** a lateral but real connection — the attention softmax produces **dense** weights (all > 0), and there is a whole line of *sparse attention* seeking what $L_1$ does in regression: exact zeros.
