# Portfolio Selection

**Autores:** Harry Markowitz (The Rand Corporation) · **Año:** 1952 · **Venue:** The Journal of Finance, 7(1), 77–91 · **Enlace/DOI:** [10.1111/j.1540-6261.1952.tb01525.x](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x) · [JSTOR 2975974](https://www.jstor.org/stable/2975974) · [PDF (curso HKUST)](https://www.math.hkust.edu.hk/~maykwok/courses/ma362/07F/markowitz_JF.pdf)
**Campo:** economía financiera / optimización · **Leído:** 2026-07-29

## TL;DR

Markowitz funda la **teoría moderna de carteras (MPT)** convirtiendo el refrán "no pongas todos los huevos en la misma cesta" en un problema de optimización. Rechaza la regla de *maximizar el retorno esperado descontado* (que nunca implica diversificación: pone todo en el activo de mayor valor) y propone la **regla media-varianza (E-V)**: retorno esperado $E$ deseable, varianza $V$ indeseable, y el inversor debe elegir una cartera **eficiente** —mínima $V$ para cada $E$, máxima $E$ para cada $V$—. La clave: el riesgo de una cartera lo gobiernan las **covarianzas**, no las varianzas individuales; por eso funciona la diversificación, y por eso solo funciona "la buena" (entre activos poco correlados). El paper es deliberadamente **geométrico**: resuelve los casos de 3 y 4 activos con isomedias e isovarianzas, sin dar el algoritmo general. Nobel de Economía 1990.

## Contexto y motivación

Antes de 1952 no había teoría formal de *cómo combinar* activos. Markowitz separa el problema en **dos etapas** (p. 77):

1. De la observación y la experiencia se forman **creencias** sobre el comportamiento futuro de los activos ($\mu_i$, $\sigma_{ij}$).
2. De esas creencias se elige la cartera.

**Este paper trata solo la segunda etapa**; lo dice al abrir y lo repite al cerrar (p. 91). La primera —cómo estimar $\mu_i$ y $\sigma_{ij}$— queda explícitamente fuera.

**El ataque a la regla del valor descontado (pp. 77–78).** Si $R = \sum_i X_i R_i$ con $R_i$ el retorno descontado del activo $i$, entonces $R$ es una media ponderada de los $R_i$ con pesos no negativos que suman 1. Maximizarla exige $X_i = 1$ para el activo de mayor $R_i$ (y si hay empates, cualquier reparto entre ellos sirve igual). Conclusión de Markowitz: *"In no case is a diversified portfolio preferred to all non-diversified portfolios"*. Como la diversificación es observada y sensata, la regla debe rechazarse **a la vez como hipótesis descriptiva y como máxima normativa**.

**El ataque a "diversifica y confía en la ley de los grandes números" (p. 79).** Existe una regla intermedia —repartir entre los activos de máximo retorno esperado, confiando en que la LGN haga que el rendimiento real se acerque al esperado— que también falla:

> *"This presumption, that the law of large numbers applies to a portfolio of securities, cannot be accepted. The returns from securities are too intercorrelated. Diversification cannot eliminate all variance."*

La correlación pone un **suelo** a la reducción de riesgo por diversificación. Esta es la grieta por la que entra toda la teoría posterior del riesgo sistemático.

## Metodología

**El objeto (p. 81).** Con $N$ activos, $X_i$ la fracción de riqueza en el activo $i$, $\mu_i = E(R_i)$, y $\sigma_{ij} = E[(R_i - \mu_i)(R_j - \mu_j)] = \rho_{ij}\sigma_i\sigma_j$ la covarianza (con $\sigma_{ii}$ la varianza). El retorno de la cartera $R = \sum_i R_i X_i$ es una suma ponderada de variables aleatorias, con:

$$E = \sum_{i=1}^N X_i \,\mu_i, \qquad V = \sum_{i=1}^N \sum_{j=1}^N \sigma_{ij}\, X_i X_j.$$

Los $R_i$ son aleatorios; los $X_i$ **no** — los fija el inversor. Restricciones: $\sum_i X_i = 1$ y **$X_i \ge 0$** (el paper excluye explícitamente las ventas en corto).

**La regla E-V (p. 82, Fig. 1).** De todas las combinaciones $(E,V)$ alcanzables, el inversor elige una **eficiente**: mínima $V$ para $E$ dada, o máxima $E$ para $V$ dada. La Fig. 1 dibuja el conjunto alcanzable como una región y marca su frontera eficiente.

> ⚠️ **Ojo con los ejes:** el paper dibuja $V$ en la vertical y $E$ en la horizontal (Figs. 1 y 6), al revés de la convención moderna ($\sigma$ horizontal, $E$ vertical). Al reproducir las figuras hay que decidir si se es fiel al original o se traduce.

**El caso de 3 activos (p. 83).** El modelo se reduce a las ecuaciones numeradas del paper:

$$\text{1)}\ E = \sum_{i=1}^{3} X_i\mu_i \qquad \text{2)}\ V = \sum_{i=1}^{3}\sum_{j=1}^{3} X_iX_j\sigma_{ij} \qquad \text{3)}\ \sum_{i=1}^{3}X_i = 1 \qquad \text{4)}\ X_i \ge 0$$

Sustituyendo 3′) $X_3 = 1 - X_1 - X_2$ se pasa a **geometría bidimensional** en $(X_1, X_2)$. En particular (Eq. 1′):

$$E = \mu_3 + X_1(\mu_1 - \mu_3) + X_2(\mu_2 - \mu_3).$$

El conjunto alcanzable es el **triángulo $abc$** (el símplex) de la Fig. 2.

**Isomedias e isovarianzas (p. 84).** Markowitz define la *isomean curve* como el lugar de carteras con $E$ dada, y la *isovariance line* como el de carteras con $V$ dada. Del análisis de las fórmulas:

- Las **isomedias son rectas paralelas** ($E$ es lineal en $X$). Despejando de la Eq. (1′):
$$X_2 = \frac{E - \mu_3}{\mu_2 - \mu_3} - \frac{\mu_1 - \mu_3}{\mu_2 - \mu_3}X_1,$$
cuya **pendiente no depende de $E$** (solo cambia la ordenada en el origen) — de ahí el paralelismo.
- Las **isovarianzas son elipses concéntricas**, centradas en el punto $\hat{X}$ que **minimiza $V$**. La varianza crece al alejarse de $\hat{X}$.

> Detalle técnico fino (nota 12, p. 89): para que las isovarianzas sean **elipses** es *necesario y suficiente* que no haya dos carteras distintas con retornos perfectamente correlados. Si las hay, la forma degenera.

**El conjunto eficiente y la *critical line* (p. 85, Figs. 2–3).** Para una $E$ dada, la mejor cartera es el punto donde la **recta isomedia es tangente a una elipse isovarianza**; Markowitz lo llama $\hat{X}(E)$. Al variar $E$, esos puntos trazan una curva que —afirma, omitiendo el álgebra— **es una recta**: la ***critical line* $l$**, que pasa por $\hat{X}$.

El conjunto eficiente se construye recorriendo esa lógica **dentro del triángulo**:

- Si $\hat{X}$ **cae dentro** del conjunto alcanzable (Fig. 2), $\hat{X}$ es eficiente y el conjunto eficiente arranca ahí, avanza por la critical line hasta topar con un borde, y sigue por el borde hasta el punto de máxima $E$.
- Si $\hat{X}$ **cae fuera** (Fig. 3), se empieza en el punto alcanzable de mínima varianza (sobre un borde), se avanza hasta cortar la critical line, se sigue por ella hasta otro borde, y se termina en el vértice de máxima $E$.

El resultado general (p. 87, Fig. 4 para 4 activos en el tetraedro): **el conjunto eficiente es siempre una poligonal — una serie de segmentos conectados**, con un extremo en la cartera de mínima varianza y el otro en la de máximo retorno esperado.

> **Matiz histórico importante:** el término *critical line* y su papel geométrico **sí están en este paper de 1952** (incluida la nota 10, que esboza cómo recorrer las líneas críticas de los subespacios $X_i = 0$ en el caso general). Lo que **no** hay es el algoritmo sistemático: eso llega en **Markowitz (1956)**, *"The optimization of a quadratic function subject to linear constraints"*. Hoy el problema se resuelve trivialmente como un **QP**.

**La frontera en el espacio $(E,V)$ (p. 87, Figs. 5–6).** Sobre el plano de las $X$, $E$ es un **plano** y $V$ un **paraboloide**. Restringidos al conjunto eficiente (que es poligonal), la sección del plano da segmentos rectos y la del paraboloide da **arcos de parábola**. Por tanto, al graficar $V$ frente a $E$ para carteras eficientes se obtiene **una serie de segmentos de parábola conectados** (Fig. 6) — no una única parábola, precisamente por las restricciones $X_i \ge 0$ que van activándose.

**Por qué la diversificación funciona, y el "tipo correcto" (p. 89).** Aquí está la aportación conceptual más citada:

> *"Not only does the E-V hypothesis imply diversification, it implies the 'right kind' of diversification for the 'right reason'."*

Con dos activos, $V = X_1^2\sigma_1^2 + X_2^2\sigma_2^2 + 2X_1X_2\rho_{12}\sigma_1\sigma_2$: si $\rho_{12} < 1$, la varianza de la cartera cae por debajo de la media ponderada de las individuales. De ahí el ejemplo célebre: una cartera de **sesenta valores ferroviarios no está tan bien diversificada** como una del mismo tamaño repartida entre ferrocarril, utilities, minería y manufactura, porque las empresas de un mismo sector tienden a ir mal a la vez. La adecuación de la diversificación **no depende del número de activos**, sino de evitar activos con **covarianzas altas entre sí**.

**Robustez ante la medida de riesgo (p. 89).** Si en vez de $V$ el inversor usara la desviación típica $\sigma = \sqrt{V}$ o el coeficiente de dispersión $\sigma/E$, **su elección seguiría estando en el mismo conjunto eficiente** (son transformaciones monótonas de $V$ a $E$ fija).

**Justificación de la regla (pp. 90–91).** Markowitz **no** deriva E-V de axiomas de utilidad esperada (nada de utilidad cuadrática aquí — eso es posterior, en su libro de 1959). La defiende como *"a working hypothesis and a working maxim"* para instituciones que consideran el rendimiento bueno, el riesgo malo y el juego evitable. Y acota su alcance con el **tercer momento** $M_3$ (nota 13): si la utilidad fuese $U(E,V,M_3)$ con $\partial U/\partial M_3 \neq 0$, el inversor aceptaría algunas apuestas justas. Por eso E-V describe la conducta de **"inversión"** y no la de **"especulación"**.

## Resultados principales

- **La regla de máximo retorno descontado queda descartada**: nunca implica diversificación, ni como descripción ni como norma. La regla E-V sí la implica, y por la razón correcta (las covarianzas).
- **Caracterización geométrica completa** del conjunto eficiente para 3 y 4 activos vía isomedias (rectas paralelas) e isovarianzas (elipses concéntricas), con la *critical line* como eje del argumento.
- **El conjunto eficiente es una poligonal** (segmentos conectados) en el espacio de carteras, y una **serie de arcos de parábola** en el plano $(E,V)$, para cualquier número de activos.
- **La ley de los grandes números no elimina el riesgo**: los retornos están demasiado intercorrelados; la diversificación tiene un suelo.
- **El riesgo relevante es la covarianza**, no la varianza individual — la semilla directa del $\beta$ del CAPM una década después.

## Puntos fuertes y limitaciones

**Fuertes:** convierte una intuición cualitativa en un problema de optimización convexa bien definido; identifica la **covarianza** como el objeto central del riesgo (de ahí derivan CAPM, APT y el *factor investing*); la exposición geométrica es de una claridad excepcional y directamente reproducible; honestidad sobre el alcance —dice qué no hace (la etapa 1, el caso $n$ general, la dinámica) y anuncia el tratamiento general futuro—; funda un campo entero.

**Limitaciones (unas reconocidas por el propio autor, otras visibles en retrospectiva):**

- **Autoimpuestas y declaradas (p. 79):** *(1)* no deriva resultados analíticamente para $N$ activos, solo geométricamente para 3 y 4; *(2)* asume **creencias probabilísticas estáticas** — modelo de un solo periodo, sin rebalanceo ni costes de transacción.
- **Depende de $\mu_i$ y $\sigma_{ij}$ como inputs.** Markowitz **es consciente** y lo señala al final (p. 91): sugiere *tentativamente* usar los momentos observados del pasado, pero añade que *"better methods, which take into account more information, can be found"* y que hace falta una reformulación probabilística del análisis de valores — *"another story"*. La crítica moderna es que la optimización **amplifica** el error de estimación (*error maximization*, Michaud 1989): pesos extremos, inestables, malos fuera de muestra.
- **La varianza es simétrica**: penaliza igual el lado bueno y el malo. El paper acota esto vía $M_3$ (asimetría), pero la alternativa que el propio Markowitz preferiría después —la **semivarianza**— **no aparece aquí** (es de 1959).
- **Sin activo libre de riesgo.** Añadirlo (Tobin, 1958) da el **teorema de separación en dos fondos** y la *capital market line*; sobre eso Sharpe (1964) construye el CAPM. Este paper es el cimiento, no el edificio.
- **Sin ventas en corto** ($X_i \ge 0$), lo que es una elección de modelado razonable pero limita el espacio de carteras y es justo lo que hace que la frontera sea poligonal a trozos.
- **Colas pesadas y momentos superiores**: la estructura $(E,V)$ ignora curtosis y (salvo la nota sobre $M_3$) asimetría, que Mandelbrot y Fama documentarían como esenciales en mercados reales.

## Ideas de implementación

Todo el núcleo es álgebra lineal reproducible con numpy/scipy. Propuesta por piezas (al estilo del Lasso):

1. **Frontera eficiente en forma cerrada** (caso relajado, solo $\sum X_i = 1$, permitiendo cortos) vía multiplicadores de Lagrange. Con $A = \mathbf 1^\top\Sigma^{-1}\mathbf 1$, $B = \mathbf 1^\top\Sigma^{-1}\boldsymbol\mu$, $C = \boldsymbol\mu^\top\Sigma^{-1}\boldsymbol\mu$, $D = AC - B^2$:
$$V(E) = \frac{AE^2 - 2BE + C}{D}, \qquad \mathbf w_{\min} = \frac{\Sigma^{-1}\mathbf 1}{A}.$$
Es una **parábola** en $(E,V)$ — coherente con la Fig. 6 del paper, que sin restricciones de signo sería un único arco. ~20 líneas.
2. **Frontera con $X_i \ge 0$** como **QP** (`scipy.optimize` o `cvxpy`): min $\mathbf w^\top\Sigma\mathbf w$ s.a. $\boldsymbol\mu^\top\mathbf w = E^\*$, $\sum w_i = 1$, $w_i \ge 0$; barrer $E^\*$. **Verificar la tesis del paper**: que los pesos óptimos trazan una **poligonal** y que $(E,V)$ da **arcos de parábola conectados**, detectando los quiebros donde un $w_i$ toca 0.
3. **Reproducir la Fig. 2** (la joya pedagógica): triángulo del símplex en $(X_1,X_2)$, **rectas isomedia**, **elipses isovarianza**, el centro $\hat X$, la ***critical line*** y el conjunto eficiente en negrita. Y la **Fig. 3** como segundo caso (con $\hat X$ fuera del triángulo).
4. **Demo del suelo de la diversificación**: $V$ de la cartera equiponderada frente a $N$, comparando activos incorrelados ($V \to 0$) vs correlados ($V \to$ covarianza media). Es la refutación de la LGN de la p. 79, en una figura.
5. **Ejemplo "sesenta ferroviarias" (p. 89)**: dos carteras del mismo tamaño, una intra-sector (correlación alta) y otra multi-sector, mostrando que la segunda domina. La tesis del "*right kind of diversification*" hecha código.
6. **Datos reales**: unos pocos tickers, estimar $\boldsymbol\mu$ y $\Sigma$, trazar la frontera, marcar mínima varianza y tangente.
7. **Frontera remuestreada** *(extensión — enlaza con el bootstrap)*: remuestrear retornos con bootstrap, recalcular la frontera $B$ veces y visualizar su **inestabilidad**. Es exactamente la preocupación que Markowitz deja abierta en la p. 91, con la herramienta de Efron.
8. **Activo libre de riesgo → capital market line** *(extensión, Tobin 1958)*: cartera tangente y recta de asignación de capital; antesala del CAPM.

Validar contra `PyPortfolioOpt` o contra la solución analítica de mínima varianza.

## Conexiones

- **[Efron (1979), Bootstrap](1979-efron-bootstrap.md):** el bootstrap es la vía estándar para meter incertidumbre en la frontera eficiente —notoriamente inestable— remuestreando retornos y obteniendo una distribución de carteras óptimas (Michaud). El puente es exacto: Markowitz cierra el paper (p. 91) diciendo que estimar $\mu_i$ y $\sigma_{ij}$ del pasado es solo una sugerencia tentativa y que hacen falta métodos mejores; el bootstrap es una de las respuestas.
- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** penalizar los pesos con $L_1$ (Brodie et al. 2009, *"Sparse and stable Markowitz portfolios"*) produce carteras **esparsas y estables**, atacando el problema de pesos extremos. La misma geometría del rombo $L_1$ que anula coeficientes anula posiciones. Nota curiosa: ambos problemas son **QP con restricciones**, y ambos tienen soluciones **lineales a trozos** en su parámetro (la *critical line* de Markowitz ↔ el *regularization path* del Lasso vía LARS).
- **Futuras del [ROADMAP](../ROADMAP.md), pista quant:** Tobin (1958, separación en dos fondos), Sharpe (1964, CAPM — el equilibrio construido sobre esta frontera), Black & Scholes (1973), Kalman (1960, estimación secuencial de $\boldsymbol\mu$/$\Sigma$), y López de Prado (2016, *Hierarchical Risk Parity* — evita invertir $\Sigma$, atacando el mismo talón de Aquiles).
