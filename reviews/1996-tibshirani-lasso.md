# Regression Shrinkage and Selection via the Lasso

**Autores:** Robert Tibshirani · **Año:** 1996 · **Venue:** Journal of the Royal Statistical Society, Series B, 58(1), 267–288 · **Enlace/DOI:** [10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) · [JSTOR 2346178](https://www.jstor.org/stable/2346178)
**Campo:** estadística / ML · **Leído:** 2026-07-11

## TL;DR

Tibshirani propone el *lasso* (least absolute shrinkage and selection operator): mínimos cuadrados con la restricción $\sum_j |\beta_j| \le t$. La geometría de la bola $L_1$ (un rombo con esquinas en los ejes) hace que algunos coeficientes sean **exactamente cero**, combinando lo mejor de la selección de subconjuntos (interpretabilidad) y de ridge (estabilidad por ser un proceso continuo). Es el paper fundacional de la regularización $L_1$, hoy omnipresente en estadística en alta dimensión.

## Contexto y motivación

Los estimadores OLS tienen poco sesgo pero mucha varianza, y con muchos predictores no son interpretables. Las dos soluciones clásicas fallan por lados opuestos: la selección de subconjuntos es interpretable pero inestable (proceso discreto: cambios pequeños en los datos cambian el modelo elegido), y ridge es estable pero nunca anula coeficientes. El precedente directo es el *non-negative garotte* de Breiman (1993), que reescala los OLS con factores no negativos de suma acotada; su defecto es depender explícitamente de los OLS, que se comportan mal con colinealidad. El lasso evita ese uso explícito.

## Metodología

**Definición (Eq. 1).** Con predictores estandarizados ($\sum_i x_{ij}/N = 0$, $\sum_i x_{ij}^2/N = 1$):

$$\hat{\beta}^{lasso} = \arg\min_\beta \sum_{i=1}^N \Big(y_i - \alpha - \sum_j \beta_j x_{ij}\Big)^2 \quad \text{sujeto a} \quad \sum_j |\beta_j| \le t$$

Equivalente en forma lagrangiana a penalizar con $\lambda \sum_j |\beta_j|$. El parámetro se suele normalizar como $s = t / \sum_j |\hat{\beta}_j^{OLS}| \in [0, 1]$.

**Caso ortonormal (Eq. 3).** Si $X^T X = I$, la solución es *soft thresholding*:

$$\hat{\beta}_j = \mathrm{sign}(\hat{\beta}_j^{OLS})\,\big(|\hat{\beta}_j^{OLS}| - \gamma\big)^+$$

frente a ridge (encoge proporcionalmente, $\hat{\beta}_j^{OLS}/(1+\gamma)$) y subset selection (*hard thresholding*). Conexión directa con el soft shrinkage de ondículas de Donoho & Johnstone (1994): el lasso alcanza asintóticamente el riesgo del selector de subconjuntos ideal salvo un factor $2\log p + 1$ (Sección 10).

**Geometría (Sección 2.3).** Los contornos elípticos del RSS tocan primero la región factible; con $L_1$ la región es un rombo y el contacto suele ocurrir en una esquina ($\beta_j = 0$). Con la bola $L_2$ de ridge no hay esquinas. Es la figura más famosa del paper (Fig. 2). Nota: para $p > 2$ con correlación, el lasso puede incluso cambiar el signo respecto al OLS.

**Interpretación bayesiana (Sección 5).** El lasso es la moda a posteriori con priors independientes doble-exponenciales (Laplace): $f(\beta_j) \propto \exp(-|\beta_j|/\tau)$ — más masa en 0 y en las colas que la prior normal implícita de ridge.

**Elección de $t$ (Sección 4).** Tres métodos: cross-validation quíntuple sobre una rejilla de $s$, GCV usando el número efectivo de parámetros $p(t) = \mathrm{tr}\{X(X^TX + \lambda W^-)^{-1}X^T\}$ con $W = \mathrm{diag}(|\hat{\beta}_j|)$, y el estimador insesgado del riesgo de Stein (mucho más barato: una sola optimización).

**Algoritmo (Sección 6).** Programación cuadrática introduciendo secuencialmente las restricciones de signo $\delta_i^T \beta \le t$ violadas (de las $2^p$ posibles; en la práctica converge en $0.5p$–$0.75p$ iteraciones). Alternativa de David Gay: escribir $\beta_j = \beta_j^+ - \beta_j^-$ con $2p$ variables y $2p+1$ restricciones. **Nota histórica:** hoy nadie lo resuelve así — LARS (Efron et al. 2004) y sobre todo el descenso por coordenadas (Friedman et al. 2007, base de `glmnet`) lo hicieron trivial.

## Resultados principales

- **Datos de cáncer de próstata** (Stamey et al. 1989; $N=97$, 8 predictores, respuesta `lpsa`): con $\hat{s} = 0.44$ elegido por GCV, el lasso retiene `lcavol`, `lweight` y `svi` — el mismo subconjunto que best subset, pero con coeficientes encogidos (Tabla 1, Fig. 5 con las trayectorias de coeficientes).
- **Simulaciones (Sección 7), tres regímenes** que estructuran la conclusión:
  1. Pocos efectos grandes → gana subset selection (y el garotte); el lasso queda cerca.
  2. Número moderado de efectos moderados → **gana el lasso** (ej. 1: MSE mediano 1.93 con GCV vs 2.79 del OLS y 3.21 de ridge).
  3. Muchos efectos pequeños → gana ridge con claridad.
- GCV es el mejor selector de $t$ de forma consistente en todos los ejemplos.
- **Extensiones (Secciones 8–9):** a GLM vía IRLS (demo con regresión logística en los datos de cifosis), a árboles (shrinkage en vez de poda) y a MARS.

## Puntos fuertes y limitaciones

**Fuertes:** resuelve simultáneamente predicción e interpretabilidad con una sola idea convexa; la explicación geométrica es cristalina; honestidad empírica poco común — el paper delimita él mismo los regímenes donde el lasso pierde; conexiones profundas (Bayes/Laplace, soft thresholding de ondículas, la familia bridge $L_q$ de Frank & Friedman donde $q=1$ es el menor exponente convexo).

**Limitaciones (algunas visibles solo en retrospectiva):** el algoritmo original de QP es tosco comparado con LARS/descenso por coordenadas; el estimador de error estándar (Eq. 7) da varianza 0 justo para los coeficientes anulados; no hay teoría de consistencia en selección (llegaría con Zhao & Yu 2006, Zou 2006 — lasso adaptativo); con predictores muy correlados el lasso elige uno arbitrariamente (motivó el elastic net, Zou & Hastie 2005); no trata el caso $p \gg N$ que luego sería su gran nicho.

## Ideas de implementación

> **Nota posterior (2026-08-01).** La [implementación](../implementations/1996-tibshirani-lasso/)
> está hecha, y se apartó de la idea 1 de esta lista: se usa el algoritmo de
> programación cuadrática de la Sección 6 del propio paper, no el descenso por
> coordenadas de Friedman et al. (2007), que es once años posterior. La idea 4
> (Tabla 3) corre pero no se publica: el ordenamiento de métodos sale y los
> niveles no, y el diagnóstico depende de detalles del montaje de simulación
> todavía sin cerrar. Lo demás está hecho y validado.

Todo lo central es reproducible con numpy puro:

1. ~~**Solver por descenso por coordenadas**~~ → se hizo el de la Sec. 6, que es el del paper. ✅
2. **Fig. 1**: las cuatro funciones de shrinkage en diseño ortonormal (subset, ridge, lasso, garotte). ✅
3. **Fig. 5 + Tabla 1**: trayectorias de coeficientes y modelo en $\hat{s}=0.44$ con los datos reales de próstata (disponibles en la web de *Elements of Statistical Learning*). ✅
4. **Ejemplo 1 de simulación (Tabla 3)**: comparar MSE de OLS, lasso-CV y ridge sobre 50 réplicas del modelo $\beta = (3, 1.5, 0, 0, 2, 0, 0, 0)$, correlación $\rho^{|i-j|}$ con $\rho = 0.5$, $\sigma = 3$. ⚠️ corre, no se publica.
5. Validar el solver propio contra `sklearn.linear_model.Lasso`. ✅ a $8\times10^{-13}$ en toda la trayectoria, y contra LARS sin convertir convenciones.

## Balance tras implementarlo

Lo que queda claro solo después de haberlo hecho, y que la lectura no daba:

**El encogimiento es el peaje, no el objetivo.** La bola $L_1$ es exactamente la envolvente convexa de los puntos 1-dispersos $\{\pm t\,e_j\}$: se toman los modelos que uno quiere y se convexifica para poder resolver. Los vértices sobreviven —de ahí los ceros— pero el óptimo puede caer en el interior de una cara, y ese es todo el sesgo. Encoger es lo que se paga por la convexidad, no lo que se busca; el acrónimo pone *Shrinkage* delante de *Selection* y engaña. La prueba es que la literatura posterior (garotte, lasso adaptativo, SCAD) se dedica a quitarle ese encogimiento sin perder la selección.

**"Shrinkage" ni siquiera está bien definido fuera del diseño ortonormal.** La Eq. 3 y las cuatro curvas de la Fig. 1 son el retrato de un caso particular. Con predictores correlados no existe ninguna función $h$ con $\hat\beta_j = h(\hat\beta_j^{OLS})$: sobre 140 diseños al azar, para $\hat\beta_j^{OLS}\approx 2$ el lasso reparte valores entre 0 y 2.84. Y con $p\ge3$ un coeficiente puede **crecer** al apretar el presupuesto (comprobado: de 0.95 a 1.81), el análogo lasso del repunte de ridge en $\rho>1/2$. Lo único que encoge es el escalar $\sum_j|\beta_j|$.

**Lo que sí sobrevive es una forma cerrada condicionada.** Sobre el conjunto activo $A$ con signos $s_A$, KKT da $\hat\beta_A = \hat\beta^{\mathrm{ols}(A)} - \lambda (X_A^\top X_A)^{-1}s_A$: OLS reajustado sobre $A$, desplazado. No falta álgebra, falta **combinatoria** — qué $A$ y qué signos. De ahí sale además, demostrada, la linealidad a trozos de las trayectorias, que es lo que LARS explota. Está en la sección 14 de las [deducciones](../implementations/1996-tibshirani-lasso/DEDUCCIONES.md).

**Qué demuestra el paper y qué solo afirma.** Selección y convexidad son construcción pura, y se cumplen. La **estabilidad**, en cambio, la deja a las simulaciones — pudiendo demostrarla: el lasso es la proyección de $\hat\beta^{OLS}$ sobre un convexo en la métrica $X^\top X$, y las proyecciones sobre convexos son no expansivas, luego es *demostrablemente* al menos tan estable como mínimos cuadrados. Selección de subconjuntos proyecta sobre una unión de subespacios, que no es convexa, y por eso salta. La estabilidad no viene de encoger: viene de que el conjunto factible sea convexo, que es exactamente lo que comparte con ridge.

**La mitad estructural aguanta; la empírica se agrieta.** Reproducimos la Tabla 1 clavada y las figuras, pero el GCV de la Eq. 10 no elige el $\hat s = 0.44$ del paper bajo ninguna lectura razonable (nos da 0.69), y en la Tabla 3 no hay una $\sigma$ que dé a la vez los niveles de error y la estructura de los modelos. Más dos erratas encontradas al deducir: la Eq. 6 necesita un límite inferior que el paper no da, y la fórmula del riesgo de Stein imprime `max` donde va `min`. Nada de eso toca la conclusión que el paper de verdad extrae, que es un **ordenamiento** de métodos, y ese sí se sostiene.

## Conexiones

- Primera review del repo — sin conexiones internas todavía.
- Futuras naturales del [ROADMAP](../ROADMAP.md): Efron (1979, bootstrap — usado aquí para los errores estándar), Bottou et al. (2018, optimización — el lasso como problema convexo no diferenciable), Blei et al. (2017, inferencia variacional — la lectura bayesiana con prior de Laplace), Friedman (2001, gradient boosting — la otra vía de selección implícita de variables).
