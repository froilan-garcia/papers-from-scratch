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

## Fundamentos — de dónde sale cada pieza

Notas de repaso antes de picar código. La idea es no tratar la atención como una
caja negra de tres letras (Q, K, V) sino ver **qué operación es realmente**.

### 1. La atención es una media ponderada, y los pesos los decide un producto escalar

Quitando toda la notación, la atención hace esto: para cada posición, produce una
**combinación convexa de los values**. Lo único no trivial es de dónde salen los
pesos.

    salida_i = sum_j  alpha_ij * v_j        con  sum_j alpha_ij = 1,  alpha_ij >= 0

Los $\alpha_{ij}$ salen de un softmax sobre las **compatibilidades** entre la
query $i$ y la key $j$. Y la compatibilidad elegida es la más simple posible: el
**producto escalar** $q_i \cdot k_j$, que mide alineamiento. En forma matricial
(Eq. 1 del paper):

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Dimensiones, que es donde se pierde uno: $Q \in \mathbb R^{n_q \times d_k}$,
$K \in \mathbb R^{n_k \times d_k}$, $V \in \mathbb R^{n_k \times d_v}$. Entonces
$QK^\top \in \mathbb R^{n_q \times n_k}$ (la **matriz de atención**, una fila por
query), el softmax es **por filas**, y la salida queda en $\mathbb R^{n_q \times d_v}$.
Nótese que $n_q$ y $n_k$ pueden diferir — eso es justo lo que permite la atención
encoder-decoder.

### 2. El $\sqrt{d_k}$ no es cosmético (nota 4 del paper)

Este es el detalle que conviene *derivar*, no memorizar. Supongamos $q, k$ con
componentes independientes de media 0 y varianza 1. Entonces:

$$q\cdot k = \sum_{i=1}^{d_k} q_i k_i
\quad\Longrightarrow\quad
E[q\cdot k] = 0, \qquad \mathrm{Var}(q\cdot k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_ik_i) = d_k,$$

usando independencia y $\mathrm{Var}(q_ik_i) = E[q_i^2k_i^2] = E[q_i^2]E[k_i^2] = 1$.

Así que la **desviación típica de los logits crece como $\sqrt{d_k}$**. Con
$d_k = 64$ eso son logits de magnitud típica $\pm 8$, y diferencias de ese orden
entre ellos. El softmax con logits muy separados se **satura**: se acerca a un
one-hot, y su jacobiano

$$\frac{\partial\,\mathrm{softmax}(z)_i}{\partial z_j} = \mathrm{softmax}(z)_i(\delta_{ij} - \mathrm{softmax}(z)_j)$$

tiende a **cero** cuando alguna componente $\to 1$ y el resto $\to 0$. Gradiente
muerto, no aprende. Dividir por $\sqrt{d_k}$ devuelve la varianza a 1 y mantiene
el softmax en su régimen sensible. **Pieza 1 lo comprueba numéricamente.**

### 3. Multi-head: por qué no sale más caro

Con una sola cabeza a dimensión $d_{\text{model}}$, las proyecciones serían
$W^Q, W^K, W^V \in \mathbb R^{d_{\text{model}}\times d_{\text{model}}}$. Con $h$
cabezas, cada una proyecta a $d_k = d_v = d_{\text{model}}/h$, así que cada
$W_i^Q \in \mathbb R^{d_{\text{model}}\times d_{\text{model}}/h}$. Apilando las
$h$ cabezas:

    h * (d_model x d_model/h)  =  d_model x d_model

**El mismo número de parámetros.** Por eso el paper dice que el coste total es
similar al de la atención de cabeza única a dimensión completa. Lo que se gana es
que cada cabeza puede especializarse en un subespacio distinto; lo que se pierde
es resolución por cabeza — de ahí que la Tabla 3(A) muestre que **demasiadas
cabezas también empeora**. Con $d_{\text{model}}=512$ y $h=8$: $d_k=d_v=64$.

### 4. La máscara causal, y por qué es $-\infty$ y no 0

Para que el decoder sea autorregresivo, la posición $i$ no puede ver la $j > i$.
La tentación es poner a cero los pesos después del softmax, pero eso **rompe la
normalización** (dejarían de sumar 1). La solución del paper es enmascarar
**antes**, poniendo los logits ilegales a $-\infty$:

    softmax(...,-inf, ...)  ->  exp(-inf) = 0

Así el cero sale *del propio softmax* y las filas siguen sumando 1. En código se
usa un número muy negativo (`-1e9`) por estabilidad numérica.

### 5. Positional encoding: por qué sinusoides

Sin recurrencia ni convolución, la atención es **permutación-equivariante**: si
barajas las posiciones de entrada, la salida se baraja igual. Es decir, el modelo
**no sabe el orden**. Hay que inyectarlo, y el paper lo suma a los embeddings:

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

La propiedad que lo motiva: para cada par de dimensiones $(2i, 2i+1)$ y con
$\omega_i = 10000^{-2i/d_{\text{model}}}$, un desplazamiento fijo $k$ actúa como una
**rotación**:

$$\begin{pmatrix}\sin(\omega_i(pos+k))\\ \cos(\omega_i(pos+k))\end{pmatrix}
=\begin{pmatrix}\cos\omega_i k & \sin\omega_i k\\ -\sin\omega_i k & \cos\omega_i k\end{pmatrix}
\begin{pmatrix}\sin(\omega_i\,pos)\\ \cos(\omega_i\,pos)\end{pmatrix}$$

La matriz **no depende de $pos$**, solo del desplazamiento $k$ — que es
exactamente lo que el paper quiere decir con *"$PE_{pos+k}$ puede representarse
como función lineal de $PE_{pos}$"*. Esa observación es la semilla directa de
**RoPE** (Ruta D del ROADMAP), que en vez de *sumar* la codificación **rota** las
queries y keys. La Pieza 4 verifica la identidad numéricamente.

## Stack

`numpy` (Piezas 1–4), `torch` (5–8), `matplotlib` para las figuras. Nada fuera
del stack base del repo.
