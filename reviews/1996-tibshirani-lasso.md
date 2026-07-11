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

Todo lo central es reproducible con numpy puro:

1. **Solver por descenso por coordenadas** con soft thresholding — el estándar moderno, ~40 líneas.
2. **Fig. 1**: las cuatro funciones de shrinkage en diseño ortonormal (subset, ridge, lasso, garotte).
3. **Fig. 5 + Tabla 1**: trayectorias de coeficientes y modelo en $\hat{s}=0.44$ con los datos reales de próstata (disponibles en la web de *Elements of Statistical Learning*).
4. **Ejemplo 1 de simulación (Tabla 3)**: comparar MSE de OLS, lasso-CV y ridge sobre 50 réplicas del modelo $\beta = (3, 1.5, 0, 0, 2, 0, 0, 0)$, correlación $\rho^{|i-j|}$ con $\rho = 0.5$, $\sigma = 3$.
5. Validar el solver propio contra `sklearn.linear_model.Lasso`.

## Conexiones

- Primera review del repo — sin conexiones internas todavía.
- Futuras naturales del [ROADMAP](../ROADMAP.md): Efron (1979, bootstrap — usado aquí para los errores estándar), Bottou et al. (2018, optimización — el lasso como problema convexo no diferenciable), Blei et al. (2017, inferencia variacional — la lectura bayesiana con prior de Laplace), Friedman (2001, gradient boosting — la otra vía de selección implícita de variables).
