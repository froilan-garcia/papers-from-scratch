# Bootstrap Methods: Another Look at the Jackknife

**Autores:** Bradley Efron · **Año:** 1979 · **Venue:** The Annals of Statistics, 7(1), 1–26 (1977 Rietz Lecture) · **Enlace/DOI:** [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) · [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full)
**Campo:** estadística / ML · **Leído:** 2026-07-23

## TL;DR

Efron introduce el **bootstrap**: para estimar la distribución muestral de un estadístico $R(\mathbf{X}, F)$ sin fórmulas analíticas, sustituye la distribución desconocida $F$ por la **distribución empírica** $\hat{F}$ (masa $1/n$ en cada dato observado) y estudia $R(\mathbf{X}^*, \hat{F})$, donde $\mathbf{X}^*$ se remuestrea **con reemplazo** de los datos. El paper lo presenta como generalización del jackknife —de hecho demuestra que el jackknife es la **aproximación lineal (delta method)** del bootstrap— y lo valida en media, varianza, mediana, tasas de error en análisis discriminante, estadístico de Wilcoxon y regresión. Es el paper fundacional de toda la estadística computacional basada en remuestreo.

## Contexto y motivación

El jackknife de Quenouille–Tukey estima sesgo y varianza de un estadístico recomputándolo $n$ veces, cada vez quitando una observación. Funciona bien para estadísticos "suaves" (media, correlación) pero **falla para la mediana** (su estimador de varianza no es ni siquiera consistente) y es confuso en situaciones no equilibradas (dos muestras, regresión). Efron busca un método más primitivo y general del que el jackknife sea un caso particular. La respuesta: en lugar de aproximar linealmente quitando puntos, **remuestrear directamente** de $\hat{F}$ y dejar que el ordenador calcule la distribución por fuerza bruta.

## Metodología

**Problema general.** Dada una muestra $\mathbf{X} = (X_1,\dots,X_n)$ con $X_i \sim_{\text{iid}} F$ (desconocida) y un estadístico $R(\mathbf{X}, F)$ —típicamente $R = t(\mathbf{X}) - \theta(F)$, el error de un estimador respecto al parámetro verdadero—, estimar la **distribución muestral** de $R$ a partir de los datos.

**La receta del bootstrap (Sec. 2).** Tres pasos:

1. Construir la distribución empírica $\hat{F}$: masa $1/n$ en cada $x_i$.
2. Extraer una **muestra bootstrap** remuestreando con reemplazo (Eq. 2.4):
$$X_i^* \sim_{\text{iid}} \hat{F}, \qquad \mathbf{X}^* = (X_1^*,\dots,X_n^*).$$
3. Aproximar la distribución de $R(\mathbf{X}, F)$ por la **distribución bootstrap** de (Eq. 2.5):
$$R^* = R(\mathbf{X}^*, \hat{F}).$$

La justificación es la **consistencia de Fisher**: cualquier estimador razonable de la distribución de $R$ debe acertar cuando $F = \hat{F}$, y $\hat{F}$ es el centro de la clase de $F$ plausibles dado que hemos observado $\mathbf{X} = \mathbf{x}$.

**Ejemplo mínimo — la media (Eqs. 2.6–2.8).** Con $R = \bar{X} - \theta(F)$, la varianza bootstrap reproduce la fórmula clásica:
$$\mathrm{Var}_*(\bar{X}^* - \bar{x}) = \frac{\bar{x}(1-\bar{x})}{n}.$$

**Tres formas de calcular la distribución bootstrap (Sec. 2, clave del paper):**

- **Método 1 — cálculo teórico directo.** Cuando se puede resolver a mano (media, varianza, mediana).
- **Método 2 — Monte Carlo.** Generar $N$ muestras bootstrap $\mathbf{x}^{*1},\dots,\mathbf{x}^{*N}$ y usar el histograma de $R(\mathbf{x}^{*j}, \hat{F})$ como aproximación. **Es el que se usa hoy universalmente**; el resto del bootstrap moderno es esto.
- **Método 3 — expansión de Taylor** de $R$ en torno a $\hat{F}$. Resulta ser **exactamente el jackknife infinitesimal** (Sec. 5).

**Variantes de $\hat{F}$ (Sec. 3).** El bootstrap no obliga a usar la empírica:
- **Bootstrap suavizado (smoothed, Eq. 3.11):** en vez de remuestrear puntos discretos, añadir ruido: $X_i^* = \bar{x} + c\,[\,x_{I_i} - \bar{x} + \hat{\sigma} Z_i\,]$ con $Z_i$ de media 0 y varianza fijada. Equivale a remuestrear de una $\hat{F}$ con ventana kernel.
- **Bootstrap simétrico (Eq. 3.8):** si se asume $F$ simétrica, reflejar $\hat{F}$ respecto a la mediana ($\hat{F}_{\text{SYM}}$).

En un experimento Monte Carlo con la mediana ($n=13$, $\mathcal{N}(0,1)$), la conclusión es sobria: **el bootstrap más simple (3.6) va casi tan bien como las versiones suavizada y simétrica** (Tabla 1). El bootstrap estima $E_F R = 0.95$ razonablemente con solo $n=13$.

**Relación con el jackknife (Sec. 5).** Escribiendo $P_i^* = N_i^*/n$ (proporción de veces que $x_i$ aparece en la muestra bootstrap) y expandiendo $R(\mathbf{P}^*)$ en serie de Taylor en torno a $\mathbf{P}^* = \mathbf{e}/n$:
$$R(\mathbf{P}^*) \doteq R(\mathbf{e}/n) + (\mathbf{P}^* - \mathbf{e}/n)\mathbf{U} + \tfrac{1}{2}(\mathbf{P}^* - \mathbf{e}/n)\mathbf{V}(\mathbf{P}^* - \mathbf{e}/n)'$$
se obtienen (Eqs. 5.8–5.11) las expresiones de sesgo y varianza:
$$\mathrm{Bias}_F\,\theta(\hat{F}) \approx \frac{1}{2n}\bar{V}, \qquad \mathrm{Var}_F\,\theta(\hat{F}) \approx \sum_{i=1}^n U_i^2 / n^2.$$
Estas coinciden con el **jackknife infinitesimal** de Jaeckel; el jackknife ordinario reemplaza las derivadas $U_i = \partial R/\partial P_i$ por diferencias finitas (Eq. 5.12): $\tilde{U}_i = (n-1)(\bar{R}^* - R^*_{(i)})$. **Moraleja: jackknife = bootstrap linealizado.** Por eso el jackknife falla con la mediana (no es suave: las fórmulas de extrapolación cuadrática no valen, Remark J).

**Bootstrap paramétrico (Remark K).** Si se conoce la familia de $F$ (p. ej. normal), usar como $\hat{F}$ el **MLE paramétrico** en vez de la empírica. Para la normal, el bootstrap paramétrico de una probabilidad $\mathrm{Prob}\{\bar{X} \in [a,b]\}$ coincide con la aproximación de Edgeworth cuando $n \gtrsim 20$.

**Transformaciones y cantidades pivotales (Remarks B, D–F).** Semilla de los intervalos de confianza modernos. Cualquier cuantil de la distribución bootstrap de $R^*$ se mapea al cuantil correspondiente de $S^* = g(R^* + \hat\theta) - g(\hat\theta)$ (Eq. 8.1): **el bootstrap es equivariante bajo transformaciones monótonas**. La Fig. 1 lo ilustra con la correlación de 9 pares de datos ($\hat\rho = 0.945$): la distribución bootstrap de $\hat\rho^* - \hat\rho$ es asimétrica, la de $\tanh^{-1}\hat\rho^* - \tanh^{-1}\hat\rho$ (transformación de Fisher) es casi simétrica y pivotal. **Aviso importante (Remark E):** el bootstrap da enunciados de *frecuencia*, no de *verosimilitud*; quedan problemas de inferencia que ninguna precisión del bootstrap resuelve. Los métodos de IC (percentil, BCa) **no están en este paper** — llegan en Efron (1981, 1987).

## Resultados principales

- **Mediana (Sec. 3):** el bootstrap estima correctamente (asintóticamente) la varianza $n E_*(R^*)^2 \to 1/4f^2(\theta)$, **caso en que el jackknife es inconsistente** ($n\,\mathrm{Var}(R) \to \tfrac{1}{4f^2}\cdot[\chi_2^2/2]^2$, con media 2 y varianza 20). Es el argumento estrella a favor del bootstrap.
- **Tasas de error en análisis discriminante (Sec. 4, Tabla 2):** el bootstrap estima a la vez el sesgo ($E_* R^*$) y la desviación estándar ($SD_*(R^*)$) del error de clasificación, y su estimador de $R$ tiene **~3× menos variabilidad que cross-validation / leave-one-out** (SD 0.078 vs 0.026 para el mismo sesgo).
- **Wilcoxon (Sec. 6):** el bootstrap reproduce la fórmula clásica de la varianza del estadístico de Wilcoxon (Eq. 6.7).
- **Regresión (Sec. 7):** remuestrear los **residuos** $\hat\epsilon_i$ da $\mathrm{Cov}_*\hat\beta^* = \hat\sigma^2 G^{-1}$ (Eq. 7.7), la fórmula clásica; el bootstrap "simetriza" automáticamente los datos, algo que el jackknife necesita hacer a mano.
- **Coste (Remark A):** Método 2 cuesta ~$N$ veces el cálculo original. En 1977, $N=1000$ para $n=20$ costaba \$4 en el 370/168 de Stanford. Hoy es gratis — de ahí que el bootstrap explotara en los 80.

## Puntos fuertes y limitaciones

**Fuertes:** una idea simple y universal (sustituir $F$ por $\hat{F}$) que unifica jackknife, delta method y remuestreo bajo un mismo marco; honestidad ejemplar (delimita dónde el bootstrap simple basta y dónde apenas gana la versión suavizada); visión anticipada de casi todo lo que vendría (paramétrico, suavizado, pivotalidad, transformaciones); el argumento asintótico vía multinomial (Remarks G–H) es elegante y general.

**Limitaciones (algunas por diseño, otras por época):**
- Asume **datos iid**. No hay bootstrap para datos dependientes (series temporales): el *block bootstrap* llega con Künsch (1989) y Hall. Nada en el paper cubre ese caso.
- **No desarrolla intervalos de confianza.** Los Remarks D–F muestran que usar $\hat\theta - \theta$ como pivote es problemático, pero la solución (percentil, $BC_a$) es posterior (Efron 1981, 1987; Efron & Tibshirani 1993).
- El bootstrap **falla en los bordes**: máximo de una uniforme, parámetros en la frontera, distribuciones de colas pesadas sin varianza. El paper no lo discute (se sabría después).
- Método 3 (Taylor) "parece sospechoso" porque la dimensión crece con $n$ (Remark H) — Efron lo justifica pero queda técnicamente delicado.
- Elección de $N$ (número de réplicas Monte Carlo): el paper nota que subir de 100 a 10000 mejora poco el sesgo, pero no da guía sistemática (llegaría con la teoría de $BC_a$).

## Ideas de implementación

El bootstrap es de los papers más agradecidos de implementar: todo es Monte Carlo con numpy. Lo que el usuario pidió (varios tipos de bootstrap) encaja como una familia, **marcando qué está en el paper y qué es extensión moderna**:

1. **Bootstrap no paramétrico (Método 2)** — el núcleo del paper. Remuestreo con reemplazo + histograma de $R^*$. ~15 líneas. Validar reproduciendo Eq. (2.8) para la media y el caso de la mediana (Sec. 3) donde el jackknife falla.
2. **Bootstrap paramétrico (Remark K)** — ajustar una familia (p. ej. normal) y remuestrear de ella. Comparar con el no paramétrico en datos normales.
3. **Bootstrap suavizado (Eq. 3.11)** y **simétrico (Eq. 3.8)** — las dos variantes de $\hat{F}$ del paper; reproducir cualitativamente la Tabla 1 (los tres van casi igual para la mediana).
4. **Jackknife vs bootstrap (Sec. 5)** — implementar el jackknife de la varianza y **verificar numéricamente que es la aproximación lineal del bootstrap**, y que diverge para la mediana. Es la tesis central del paper hecha código.
5. **Intervalos de confianza** *(extensión post-1979, señalar en el README)*: **percentil** (los cuantiles empíricos de $R^*$; esbozado en Remark D), **básico/pivotal** (reflejar el percentil), y **$BC_a$** (bias-corrected and accelerated, Efron 1987). Reproducir la Fig. 1 (correlación de 9 pares, transformación de Fisher) es el vehículo natural.
6. **Block bootstrap** *(extensión, Künsch 1989 — no está en el paper)*: remuestrear bloques contiguos para datos dependientes. Útil de cara a la asignatura de series temporales del máster y a quant. Dejar claro en el README que es un añadido posterior.

Datos: la correlación de Fig. 1 viene en el propio paper (9 pares, Remark B). Para el resto, simulaciones sintéticas como en el paper.

## Conexiones

- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** el paper del Lasso usa el bootstrap para los errores estándar de los coeficientes (su Eq. 7), y señala su fallo — da varianza 0 justo para los coeficientes anulados. Conexión directa: el bootstrap es la herramienta de incertidumbre que al Lasso se le resiste.
- **[Markowitz (1952), Portfolio Selection](1952-markowitz-portfolio-selection.md):** el bootstrap es la vía estándar para meter incertidumbre en la frontera eficiente (remuestrear retornos → distribución de las carteras óptimas), que es notoriamente inestable. Buen puente entre ambos.
- **Futuras del [ROADMAP](../ROADMAP.md):** Breiman (2001, Random Forests — el *bagging* es bootstrap + agregación), Bottou et al. (2018, la conexión remuestreo/optimización), y la asignatura de series temporales (block bootstrap).
