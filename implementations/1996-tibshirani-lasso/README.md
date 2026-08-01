# Lasso — Tibshirani (1996)

Implementación de *Regression Shrinkage and Selection via the Lasso*. Contexto y
resultados del paper en la [review](../../reviews/1996-tibshirani-lasso.md).

> 📐 **La matemática está en [DEDUCCIONES.md](DEDUCCIONES.md)**, desarrollada de
> principio a fin: el problema, su geometría, los casos que se resuelven a mano, el
> algoritmo general, la forma cerrada que sobrevive sobre el conjunto activo y la
> elección del presupuesto. Este README solo recoge lo que se ejecuta y lo que sale.

> **Estado: cerrada.** El solver, el caso ortonormal, el caso $p=2$, los datos de
> próstata, las figuras del paper y el cotejo contra `scikit-learn`. Hubo una
> versión anterior que se descartó por usar convenciones y algoritmos que no son
> los del paper; queda archivada en la rama `lasso-v1-archive` y no es la
> referencia.
>
> **Lo que no está aquí:** la reproducción de la Tabla 3 (simulaciones de la
> Sec. 7). Corre y el ordenamiento de métodos sale, pero los niveles de error no
> cuadran con ninguna $\sigma$, y el diagnóstico depende de detalles del montaje
> de simulación que todavía no domino. Se queda en local hasta poder decir de
> quién es el desajuste; afirmarlo ahora sería afirmar de más.

## Qué cuadra y qué no

| Objetivo del paper | Resultado |
|---|---|
| Eq. 3 (caso ortonormal) | ✅ coincide con el solver a $1.8\times10^{-15}$ |
| Sec. 6, iteraciones entre $0.5p$ y $0.75p$ | ✅ media $\approx 0.5p$, nunca por encima de $0.75p$ |
| Trayectoria: $s=1\Rightarrow$ OLS, $s=0\Rightarrow 0$ | ✅ a $3\times10^{-15}$ |
| Tabla 1, columna mínimos cuadrados | ✅ a 0.01 — **pero solo con el fichero de datos sin corregir** |
| Tabla 1, columna lasso (en $s=0.44$) | ✅ a 0.01, y selecciona exactamente `lcavol`, `lweight`, `svi` |
| Fig. 5 (trayectorias de próstata) | ✅ |
| Fig. 1 y Fig. 2 | ✅ el lasso cae en la esquina ($\beta_1 = 2.8\times10^{-17}$), ridge no |
| Eq. 5 y 6 "valid even if the predictors are correlated" | ✅ el mismo $\gamma$ en las dos coordenadas y en los cinco $\rho$, a 9 decimales |
| Fig. 4 (dos predictores) | ✅ incluido el repunte de ridge para $\rho>1/2$, que sale deducido |
| Contra `sklearn.linear_model.Lasso` y LARS | ✅ a $8\times10^{-13}$ en toda la trayectoria |
| $\hat s = 0.44$ por GCV (Eq. 10) | ❌ **nuestro GCV minimiza en 0.69** |

La discrepancia del GCV está diagnosticada abajo, y no es un fallo del solver:
en $s=0.44$ reproducimos la Tabla 1 clavada, lo que aísla el problema en el
selector y no en el lasso.

## Reglas de esta implementación

Decisiones tomadas por adelantado, para que no aparezcan escondidas en un
comentario a mitad del código.

**1. El objetivo es el del paper, sin reescalados modernos.** Eq. 1:

$$\min_\beta \ \sum_{i=1}^N\Big(y_i - \alpha - \sum_j \beta_j x_{ij}\Big)^2
\qquad \text{sujeto a} \quad \sum_j |\beta_j| \le t$$

Sin el $\frac{1}{2N}$ de `glmnet`/`sklearn`, que no está en el paper: el $\frac12$ es
para que la derivada salga limpia y el $\frac1N$ para que $\lambda$ no dependa del
tamaño de muestra. La v1 lo usaba sin avisar, de modo que su $\lambda$ era
$\lambda_{\text{paper}}/2N$.

**2. El parámetro se indexa como lo indexa el paper**, $s = t/\sum_j|\hat\beta_j^{OLS}| \in [0,1]$,
porque la Sección 4 dice literalmente que la CV se hace *"over a grid of values of
$s$ from 0 to 1 inclusive"*. La v1 barría $\lambda$ en escala logarítmica.

**3. El algoritmo es el de la Sección 6**, programación cuadrática con las
restricciones de signo introducidas secuencialmente. **No** descenso por
coordenadas: eso es de Friedman, Hastie, Höfling & Tibshirani (2007).

### Desviaciones respecto al paper

- **El QP interior.** La Sec. 6 delega cada subproblema en Lawson & Hansen (1974).
  Aquí está escrito el método de conjunto activo primal estándar que eso significa
  (`constrained_ls` en [lasso.py](lasso.py)). Mismo QP, misma solución, álgebra
  exacta; lo que cambia es de quién es el código.
- **La interpretación de $W^-$ en la Eq. 9.** El paper solo dice "generalized
  inverse". Leída como pseudoinversa de Moore–Penrose, los coeficientes nulos
  quedarían *sin* penalizar, que es al revés de lo que hace falta. Se usa la
  lectura $1/|\beta_j|\to\infty$ (los nulos salen del ajuste), que es la única
  compatible con que la Eq. 7 dé varianza 0 para ellos, como reporta la Tabla 2.
  Están implementadas y comparadas las dos.
- **CV por pliegue.** Dentro de cada pliegue, $t = s\sum_j|\hat\beta_j^{OLS}(\text{train})|$,
  con el OLS *de ese pliegue*. El paper no lo especifica.
- **No implementado:** el estimador insesgado del riesgo de Stein (Eq. 11), que
  solo está derivado para el diseño ortogonal. La fórmula sí se comprueba —y se le
  encuentra una errata— en [orthonormal.py](orthonormal.py), pero no se usa como
  selector.
- **No publicado:** las simulaciones de la Sec. 7. Ver el recuadro de estado.

## Por qué este orden y no el del paper

El paper va: definición → caso ortonormal → geometría → datos de próstata →
elección de $t$ → Bayes → algoritmo → simulaciones. Ese orden es **expositivo**:
sirve para convencer a un lector.

Un orden de **implementación** responde en cada paso a *¿qué necesita existir para
que el siguiente funcione?* Sale distinto:

| Cambia | El paper lo pone | Aquí | Por qué |
|---|---|---|---|
| El algoritmo (Sec. 6) | casi al final | paso 4 | Sin solver no hay nada que validar. |
| Los datos de próstata (Sec. 3) | al principio | paso 8 | Necesita un selector de $s$, que es el paso 7. |
| Las figuras 1 y 2 (Sec. 2) | al principio | paso 9 | Ilustran, no habilitan. |
| El caso $p=1$ | no aparece | paso 2 | Es el átomo del que sale todo lo demás. |

## Los pasos

**1. Evaluar antes de resolver** ✅ — `rss`, `l1_norm`, `is_feasible`.
$\text{RSS} = \beta^\top(X^\top X)\beta - 2y^\top X\beta + y^\top y$ tiene Hessiana
$2X^\top X \succeq 0$, luego es convexo y el mínimo sobre un convexo es único si
$X^\top X\succ0$ — que es por qué cualquier solver correcto da la misma respuesta.

**2. El caso de un solo predictor** ✅ — El óptimo sin restringir es
$\hat b = x^\top y/x^\top x$, y como la parábola es simétrica en torno a él, la
solución restringida es el punto de $[-t,t]$ más cercano:
$b^\star = \mathrm{sign}(\hat b)\min(|\hat b|,t)$. Ojo al matiz: con $p=1$ eso es
**recorte**, no *soft thresholding*. El soft thresholding necesita $p\ge2$
compartiendo un mismo presupuesto — es el multiplicador común el que traslada
todos los coeficientes por la misma constante.

**3. El caso ortonormal, y de ahí la Eq. 3** ✅ — Si $X^\top X = I$ entonces
$\hat\beta^{o}=X^\top y$ y

$$\|y-X\beta\|^2 = \|y\|^2 - 2\beta^\top\hat\beta^o + \|\beta\|^2 = \|\beta-\hat\beta^o\|^2 + \text{cte}$$

que **separa** el problema en $p$ problemas de una dimensión atados por un único
$\gamma\ge0$. La estacionariedad da $\beta_j = \hat\beta^o_j - \gamma\,\mathrm{sign}(\beta_j)$,
o sea la Eq. 3. *Validado:* coincide con el solver a $1.8\times10^{-15}$
([orthonormal.py](orthonormal.py)).

**4. El algoritmo del paper (Sec. 6)** ✅ — $\sum_j|\beta_j|\le t$ equivale a las
$2^p$ restricciones $\delta_i^\top\beta\le t$ porque $\max_\delta \delta^\top\beta$
se alcanza en $\delta=\mathrm{sign}(\beta)$ y vale $\sum_j|\beta_j|$: la bola $L_1$
es un politopo de $2^p$ caras. El algoritmo nunca las construye todas; añade las
violadas de una en una hasta que se cumple Kuhn–Tucker. *Validado:* reproduce el
paso 3 en diseño ortonormal, y la media de iteraciones sale $\approx0.5p$, en el
extremo bajo del rango $[0.5p, 0.75p]$ que afirma el paper.

*De dónde salen los ceros exactos:* si dos vectores de signos activos difieren
solo en la coordenada $j$, restar sus igualdades $\delta^\top\beta=t$ y
$\delta'^\top\beta=t$ da $2\delta_j\beta_j=0$, luego $\beta_j=0$. Es estructural,
no un umbral numérico.

**5. La trayectoria completa** ✅ — *Validado:* en $s=1$ da OLS a $3\times10^{-15}$,
en $s=0$ da ceros exactos, $\sum|\beta_j|$ es monótona y nunca excede el
presupuesto.

**6. Elegir $s$ (I): validación cruzada** ✅ — La Eq. 8, $PE = ME + \sigma^2$, dice
que error de predicción y error de modelo difieren en una constante y se minimizan
en el mismo sitio. Lo que **no** sirve es el RSS de entrenamiento: $\beta$ se
eligió para hacerlo pequeño en esos mismos puntos, así que decrece con $s$ y
siempre elegiría $s=1$.

**7. Elegir $s$ (II): GCV** ⚠️ — El lasso no es un *linear smoother*; el puente de
la Sec. 2.5 es escribir $\sum_j|\beta_j| = \sum_j\beta_j^2/|\beta_j|$, lo que
convierte el ajuste en el ridge de la Eq. 9 y permite la traza
$p(t) = \mathrm{tr}\{X(X^\top X+\lambda W^-)^{-1}X^\top\}$. El $\lambda$ se saca de
Kuhn–Tucker: $|x_j^\top(y-X\hat\beta)| = \lambda$ para toda coordenada activa —
comprobado, la dispersión entre coordenadas es $10^{-13}$. **No reproduce el
0.44 del paper**; ver abajo.

**8. Datos reales: próstata** ✅ — Tabla 1 y Fig. 5 reproducidas a 0.01, con
$\hat s$ derivado y no metido a mano.

![Fig. 5 — trayectorias de próstata](fig5_prostate_paths.png)

El orden de entrada de los predictores es el del paper: `lcavol` desde el
principio, `svi` sobre 0.23, `lweight` sobre 0.32, y `lcp` y `age` entrando en
negativo al final. La línea gris marca el $\hat s = 0.44$ del paper y la roja el
0.69 de nuestro GCV.

**9. Las figuras 1 y 2** ✅ — Las cuatro funciones de shrinkage y la geometría del
rombo contra el círculo. El contorno de la Fig. 2 se dibuja pasando por la
solución que devuelve el solver, así que la figura es consecuencia del código y no
un dibujo de lo que debería salir.

**10. Contra `scikit-learn`** ✅ — Lo primero no es comparar sino **alinear las
convenciones**, porque los dos objetivos no son el mismo: el del paper es
$\|y-X\beta\|^2$ con restricción, el de la librería es
$\frac{1}{2N}\|y-X\beta\|^2 + \alpha\|\beta\|_1$. Igualando las dos formas
lagrangianas sale $\alpha = \lambda/N$, con el $\lambda$ de KKT de la
[sección 14](DEDUCCIONES.md) — que es la única conversión que se usa.

Tres comprobaciones de fuerza creciente ([sklearn_check.py](sklearn_check.py)):

| | Resultado |
|---|---|
| Un punto: $s=0.44$, el modelo de la Tabla 1 | máx. dif. $1.4\times10^{-13}$, mismo soporte, mismo RSS a 10 decimales |
| Toda la trayectoria, 41 valores de $s$ | máx. dif. $8.1\times10^{-13}$ |
| Contra LARS, **sin convertir nada**, cotejando a igual $\|\beta\|_1$ | máx. dif. $4.9\times10^{-14}$ en los 8 codos |

La tercera es la que vale, porque no usa $\alpha$ ni $\lambda$: un error en la
conversión no puede tapar un error en el solver.

Y un detalle que sale al revés de lo que uno esperaría: **los ceros de `sklearn`
imprimen `0.0` exacto y los nuestros $10^{-14}$**. Los nuestros son exactos por
álgebra (paso 4) pero llegan a la salida por un sistema lineal, que redondea; el
descenso por coordenadas asigna el 0 literalmente —el umbral blando *es* su
actualización— aunque el soporte que elige dependa de su tolerancia. Cada uno es
exacto en un sitio distinto.

## Las discrepancias con el paper

Una es numérica y salió al ejecutar; las otras dos salieron al deducir y están en
su sitio dentro de [DEDUCCIONES.md](DEDUCCIONES.md):

- **La Eq. 6 necesita un límite inferior que el paper no da.** Vale para
  $\hat\beta_1^o-\hat\beta_2^o\le t\le\hat\beta_1^o+\hat\beta_2^o$; el paper solo
  enuncia el superior. Por debajo, una coordenada ya se anuló y la solución es
  $(t,0)$. Verificado en [two_predictors.py](two_predictors.py).
- **La fórmula del riesgo de Stein (Sec. 4) tiene una errata:** imprime
  $\max(|\hat\beta_j^o/\hat\tau|,\gamma^2)$ donde toca
  $\min(|\hat\beta_j^o/\hat\tau|,\gamma)^2$. La versión correcta sigue al riesgo
  verdadero en todos los $\gamma$ (15.25 contra 15.33 en $\gamma$ grande); la
  impresa se dispara a 1144. Comprobado en [orthonormal.py](orthonormal.py).

### 1. El fichero de datos de próstata cambió después de 1996

La fila 32 tiene hoy `lweight` = 3.8044 (44.9 g). El fichero que usó Tibshirani
tenía 6.1076 (449 g), un error de coma decimal corregido más tarde en la web de
*Elements of Statistical Learning*.

| | máx. desviación vs Tabla 1 (columna LS) |
|---|---|
| con `lweight[32] = 6.1076` (1996) | **0.01** |
| con `lweight[32] = 3.8044` (actual) | 0.04 |

[prostate.py](prostate.py) usa por defecto el valor de 1996 e imprime las dos
columnas, porque el objetivo es reproducir el paper.

### 2. El GCV no elige $\hat s = 0.44$

Sobre los datos del paper, la Eq. 10 minimiza en $s = 0.69$; la CV quíntuple, en
0.63. En $s=0.44$ el GCV vale 0.578 frente a 0.516 en su mínimo, así que no es
cuestión de resolución de rejilla ni de una curva plana.

No es el solver: **en $s=0.44$ la columna lasso de la Tabla 1 sale clavada**
(0.56 / 0.10 / 0.16 y exactamente `lcavol`, `lweight`, `svi`, el resto en cero).
El problema está en el selector. Comprobado además:

- lectura de conjunto activo de $W^-$ (la nuestra) → 0.69;
- lectura Moore–Penrose → 0.75, aún más lejos;
- $p(t)$ tendría que inflarse unas **6 veces** para llevar el mínimo a 0.48;
- $p(t) := $ número de coeficientes no nulos → 0.70.

Ninguna lectura razonable de la Eq. 10 da 0.44. Es coherente con que la literatura
posterior abandonara este GCV: los grados de libertad por aproximación ridge son
poco fiables para el lasso.

## Cómo ejecutar

Desde este directorio, con el entorno `papers`:

```bash
python orthonormal.py && python lasso.py && python two_predictors.py && python prostate.py && python figures.py && python derivation_figures.py && python sklearn_check.py
```

| Fichero | Pasos | Qué hace |
|---|---|---|
| [DEDUCCIONES.md](DEDUCCIONES.md) | — | El desarrollo matemático completo, con las figuras |
| [derivation_figures.py](derivation_figures.py) | — | Las figuras de ese documento (prefijo `ded_`) |
| [lasso.py](lasso.py) | 1, 4, 5 | Objetivo, QP de conjunto activo, algoritmo de la Sec. 6, trayectoria |
| [orthonormal.py](orthonormal.py) | 2, 3 | Eq. 3 en forma cerrada, las cuatro funciones de shrinkage y la comprobación de la fórmula de Stein |
| [two_predictors.py](two_predictors.py) | 3 | Eq. 5, Eq. 6 y Fig. 4 — valida la deducción del caso $p=2$ |
| [selection.py](selection.py) | 6, 7 | CV quíntuple y GCV (Eq. 10) |
| [prostate.py](prostate.py) | 8 | Tabla 1 y Fig. 5 |
| [figures.py](figures.py) | 9 | Fig. 1 y Fig. 2 |
| [sklearn_check.py](sklearn_check.py) | 10 | Conversión de convenciones y cotejo contra `sklearn` y LARS |

`data/prostate.data` no se versiona; se descarga de la web de *Elements of
Statistical Learning*.

### Después, si apetece, y etiquetado como posterior a 1996

- Descenso por coordenadas (Friedman et al. 2007), escrito y comparado contra el
  de la Sec. 6 — no solo llamado a través de `sklearn`.
- LARS (Efron et al. 2004) implementado, aprovechando que la linealidad a trozos
  ya está demostrada en la sección 14 de [DEDUCCIONES.md](DEDUCCIONES.md).
- Cerrar la Tabla 3 y publicarla, cuando el montaje de la simulación esté
  entendido a fondo.

## Stack

`numpy` y `matplotlib`; `pandas` para leer los datos de próstata; `scikit-learn`
**solo** en [sklearn_check.py](sklearn_check.py), como referencia externa contra la
que cotejar — no lo usa nada del solver. La programación cuadrática está escrita a
mano, así que no hace falta `scipy.optimize`.
