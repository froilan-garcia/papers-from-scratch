# Mean-variance portfolio selection — Markowitz (1952)

Implementation of the E-V rule from *Portfolio Selection* (Markowitz, 1952).
See the [review](../../reviews/1952-markowitz-portfolio-selection.md) for the
paper's context and results.

Built incrementally, same as the [Lasso](../1996-tibshirani-lasso/). Current status:

| # | Piece | Status |
|---|-------|--------|
| 0 | Fundamentals — deriving the frontier by Lagrange (below) | ✅ notes |
| 1 | Closed-form efficient frontier (no sign constraint) | ⬜ pending |
| 2 | Fig. 2 — simplex triangle, isomean lines, isovariance ellipses, critical line | ⬜ pending |
| 3 | Fig. 6 — QP frontier with `w >= 0`; verify the connected parabola segments | ⬜ pending |
| 4 | The diversification floor — refuting the law of large numbers (p. 79) | ⬜ pending |
| 5 | Real data + bootstrap-resampled frontier (links to Efron 1979) | ⬜ pending |

## Fundamentos — la frontera eficiente por multiplicadores de Lagrange

El paper resuelve los casos de 3 y 4 activos **geométricamente** y dice
explícitamente (p. 79) que no deriva el caso de $n$ activos analíticamente. Esa
derivación —estándar hoy— es la Pieza 1. Conviene tenerla a mano antes de picar
código, porque el resultado (una **parábola** en $(E,V)$) es justo lo que la
Fig. 6 del paper dibuja a trozos.

**Planteamiento.** Con $\Sigma$ la matriz de covarianzas ($p \times p$, simétrica
definida positiva), $\boldsymbol\mu$ el vector de retornos esperados y $\mathbf 1$
el vector de unos, buscamos los pesos $\mathbf w$ que minimizan la varianza para
un retorno objetivo $E$:

$$\min_{\mathbf w} \ \tfrac{1}{2}\mathbf w^\top \Sigma \mathbf w
\quad \text{s.a.} \quad \boldsymbol\mu^\top \mathbf w = E, \quad \mathbf 1^\top \mathbf w = 1.$$

> **Nota:** aquí se **relaja** la restricción $w_i \ge 0$ del paper. Es lo que
> permite la forma cerrada; con la restricción de signo hay que ir a un QP
> (Pieza 3) y aparecen los quiebros de la poligonal. El $\tfrac12$ es cosmético,
> para que la derivada salga limpia.

**Lagrangiano** (ahora sí, restricciones de *igualdad*, así que es el Lagrange de
toda la vida — a diferencia del lasso, donde $\lambda$ se fija a mano):

$$\mathcal L = \tfrac12 \mathbf w^\top\Sigma\mathbf w
- \lambda(\boldsymbol\mu^\top\mathbf w - E) - \gamma(\mathbf 1^\top\mathbf w - 1).$$

Derivando en $\mathbf w$ e igualando a cero:

$$\Sigma\mathbf w - \lambda\boldsymbol\mu - \gamma\mathbf 1 = \mathbf 0
\quad\Longrightarrow\quad
\boxed{\ \mathbf w = \Sigma^{-1}(\lambda\boldsymbol\mu + \gamma\mathbf 1)\ }$$

Los pesos óptimos son **combinación lineal de dos carteras fijas**, $\Sigma^{-1}\boldsymbol\mu$
y $\Sigma^{-1}\mathbf 1$. Esto ya es, en germen, el **teorema de separación en dos
fondos** de Tobin (1958): toda cartera eficiente se obtiene mezclando dos.

**Los cuatro escalares.** Sustituyendo en las restricciones aparecen siempre las
mismas tres cantidades:

$$A = \mathbf 1^\top\Sigma^{-1}\mathbf 1, \qquad
B = \mathbf 1^\top\Sigma^{-1}\boldsymbol\mu, \qquad
C = \boldsymbol\mu^\top\Sigma^{-1}\boldsymbol\mu, \qquad D = AC - B^2.$$

Las dos restricciones quedan como un sistema $2\times 2$ en $(\lambda, \gamma)$:

$$\begin{pmatrix} C & B \\ B & A \end{pmatrix}
\begin{pmatrix} \lambda \\ \gamma \end{pmatrix} =
\begin{pmatrix} E \\ 1 \end{pmatrix}
\quad\Longrightarrow\quad
\lambda = \frac{AE - B}{D}, \qquad \gamma = \frac{C - BE}{D}.$$

**La frontera.** El truco para la varianza es no expandir nada: usando la
condición de primer orden $\Sigma\mathbf w = \lambda\boldsymbol\mu + \gamma\mathbf 1$,

$$V = \mathbf w^\top\Sigma\mathbf w = \mathbf w^\top(\lambda\boldsymbol\mu + \gamma\mathbf 1)
= \lambda\underbrace{\boldsymbol\mu^\top\mathbf w}_{=\,E} + \gamma\underbrace{\mathbf 1^\top\mathbf w}_{=\,1}
= \lambda E + \gamma.$$

Sustituyendo:

$$\boxed{\ V(E) = \frac{AE^2 - 2BE + C}{D}\ }$$

**Una parábola en $(E,V)$** — y por tanto una **hipérbola** en $(\sigma, E)$, que
es como se dibuja hoy. Coherente con la Fig. 6 del paper: allí sale a trozos
*porque* Markowitz sí impone $w_i \ge 0$.

**Cartera de mínima varianza.** Derivando: $V'(E) = (2AE - 2B)/D = 0$, luego

$$E_{\min} = \frac{B}{A}, \qquad V_{\min} = \frac{1}{A}, \qquad
\mathbf w_{\min} = \frac{\Sigma^{-1}\mathbf 1}{A}.$$

Nótese que $\mathbf w_{\min}$ **no depende de $\boldsymbol\mu$**: solo de $\Sigma$.
Esa es la razón práctica de que la cartera de mínima varianza sea mucho más
robusta que la tangente — los retornos esperados son justo lo peor estimado
(ver la crítica de Michaud en el ROADMAP, bloque Q2).

**Signos.** Con $\Sigma$ definida positiva, $\Sigma^{-1}$ también lo es, así que
$A > 0$ y $C > 0$. Y $D = AC - B^2 > 0$ por Cauchy–Schwarz en el producto escalar
$\langle \mathbf x, \mathbf y\rangle = \mathbf x^\top\Sigma^{-1}\mathbf y$, con
igualdad solo si $\boldsymbol\mu \propto \mathbf 1$ (todos los activos con el
mismo retorno esperado — el caso degenerado de la nota 9 del paper, donde las
isomedias dejan de estar definidas). Por tanto la parábola **abre hacia arriba**
y el mínimo es genuino.

### Por qué la covarianza y no la varianza

La tesis conceptual del paper (p. 89), en una línea de álgebra. Para la cartera
equiponderada $w_i = 1/N$:

$$V = \frac{1}{N^2}\sum_{i}\sigma_{ii} + \frac{1}{N^2}\sum_{i\neq j}\sigma_{ij}
= \frac{1}{N}\overline{\sigma^2} + \frac{N-1}{N}\overline{\sigma_{ij}}
\ \xrightarrow[N\to\infty]{}\ \overline{\sigma_{ij}}.$$

El primer término (varianzas propias) se **diluye** como $1/N$; el segundo
(covarianza media) **no**. Ese límite $\overline{\sigma_{ij}}$ es el suelo del
que habla Markowitz al rechazar la ley de los grandes números, y es el riesgo que
más tarde se llamará *sistemático*. Con activos incorrelados $\overline{\sigma_{ij}} = 0$
y sí se llega a cero; con activos de un mismo sector, no. **Es la Pieza 4 en una
fórmula.**

## Stack

`numpy`, `scipy` (QP en la Pieza 3), `matplotlib`. `pandas` para los datos reales
de la Pieza 5. Nada fuera del stack base del repo.
