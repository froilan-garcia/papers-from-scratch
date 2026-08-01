# El lasso — desarrollo matemático

De la definición al algoritmo y a la elección del parámetro. El orden es el de la
construcción: se plantea el problema, se le mira la forma, se resuelve entero en
los casos que se dejan, se ve qué sobrevive al levantar las hipótesis, y solo
entonces se ataca el caso general.

**Notación.** $X$ es $N\times p$ con predictores estandarizados,
$\sum_i x_{ij}/N = 0$ y $\sum_i x_{ij}^2/N = 1$; $y$ está centrada;
$S = X^\top X$ (con esa normalización $S$ tiene **diagonal $N$**, no 1);
$\hat\beta^{o}$ es el estimador de mínimos cuadrados. Las referencias al paper de
Tibshirani (1996) van al pasar, como puntos de anclaje.

Las figuras están **calculadas con el solver** de esta implementación, no
dibujadas: cada una comprueba lo que acompaña. Las genera
[derivation_figures.py](derivation_figures.py) y los scripts de las figuras del
paper.

---

### Parte I — El problema y su forma
&nbsp;&nbsp;1. [Planteamiento, y por qué el intercepto se va](#s1)
&nbsp;&nbsp;2. [El error cuadrático es un elipsoide centrado en el OLS](#s2)
&nbsp;&nbsp;3. [La región factible es un politopo con esquinas en los ejes](#s3)
&nbsp;&nbsp;4. [El rango útil del presupuesto](#s4)

### Parte II — Resolverlo donde se deja
&nbsp;&nbsp;5. [Un solo predictor: aparece la no-derivabilidad](#s5)
&nbsp;&nbsp;6. [Diseño ortonormal: el problema se separa](#s6)
&nbsp;&nbsp;7. [El multiplicador, exacto](#s7)
&nbsp;&nbsp;8. [Las otras tres reglas de encogimiento, gratis](#s8)

### Parte III — Qué sobrevive con predictores correlados
&nbsp;&nbsp;9. [Dos predictores: el autovector que borra la correlación](#s9)
&nbsp;&nbsp;10. [Hasta dónde vale, y ridge en el mismo espejo](#s10)
&nbsp;&nbsp;11. [Por qué $p>2$ rompe la simetría](#s11)

### Parte IV — El caso general
&nbsp;&nbsp;12. [De la geometría al algoritmo](#s12)
&nbsp;&nbsp;13. [Por qué para, por qué es óptimo y por qué los ceros son exactos](#s13)
&nbsp;&nbsp;14. [OLS sobre el conjunto activo, menos un sesgo](#s14)

### Parte V — Elegir el presupuesto
&nbsp;&nbsp;15. [La correspondencia entre restringir y penalizar](#s15)
&nbsp;&nbsp;16. [Qué querríamos minimizar](#s16)
&nbsp;&nbsp;17. [Una linealización con dos consecuencias](#s17)
&nbsp;&nbsp;18. [De la validación cruzada dejando uno fuera al GCV](#s18)
&nbsp;&nbsp;19. [El riesgo de Stein, que el caso ortonormal ya permite](#s19)

### Parte VI — La misma cosa desde otro sitio
&nbsp;&nbsp;20. [La prior de Laplace](#s20)

---

# Parte I — El problema y su forma

<a name="s1"></a>
## 1. Planteamiento, y por qué el intercepto se va

El objeto es

$$\min_{\alpha,\beta}\ \sum_{i=1}^N\Big(y_i-\alpha-\sum_j\beta_jx_{ij}\Big)^2
\qquad\text{sujeto a}\qquad \sum_j|\beta_j|\le t .$$

Antes de nada conviene quitarnos $\alpha$ de encima, y se puede: **la restricción
no lo toca**. Para $\beta$ fijo se minimiza en $\alpha$ sin restricción alguna,

$$\frac{\partial}{\partial\alpha}\sum_i\Big(y_i-\alpha-\sum_j\beta_jx_{ij}\Big)^2=0
\quad\Longrightarrow\quad
\alpha=\bar y-\sum_j\beta_j\bar x_j,$$

y al estar los predictores centrados, $\bar x_j=0$, luego $\hat\alpha=\bar y$
**sea cual sea $\beta$**, y por tanto sea cual sea $t$.

Así que centrando $y$ el intercepto desaparece y queda un problema limpio en
$\beta$:

$$\min_\beta\ \|y-X\beta\|^2\qquad\text{sujeto a}\qquad \sum_j|\beta_j|\le t .$$

Todo lo que sigue trabaja con esta forma. Dos objetos que mirar por separado: la
**función objetivo** y la **región factible**.

---

<a name="s2"></a>
## 2. El error cuadrático es un elipsoide centrado en el OLS

Empecemos por el objetivo, y hagamos primero la única propiedad del ajuste por
mínimos cuadrados que vamos a necesitar.

**El residuo del OLS es ortogonal a todas las columnas.** Derivando el error
cuadrático respecto a un coeficiente,

$$f(\beta)=\sum_i\Big(y_i-\sum_j x_{ij}\beta_j\Big)^2,
\qquad
\frac{\partial f}{\partial\beta_k}=-2\sum_i x_{ik}\Big(y_i-\sum_j x_{ij}\beta_j\Big)
=-2\,(X^\top r)_k ,$$

de modo que anular las $p$ derivadas en el mínimo es exactamente

$$X^\top r^o=0,\qquad r^o:=y-X\hat\beta^o,$$

o, reordenando, $X^\top X\hat\beta^o=X^\top y$. Son las **ecuaciones normales**, y
no son un hecho independiente: son la condición de optimalidad del OLS escrita de
otra forma.

Geométricamente dicen lo evidente. El conjunto $\{X\beta:\beta\in\mathbb R^p\}$ es
el espacio columna de $X$, y minimizar $\|y-X\beta\|^2$ es buscar en él el punto más
cercano a $y$, o sea la **proyección ortogonal** de $y$. El vector que une un punto
con su proyección es perpendicular al subespacio, y por tanto a cada uno de sus
generadores — las columnas. De ahí el nombre: *normal* = perpendicular.

Hay un tercer argumento, sin cálculo ni geometría, que explica por qué tiene que
ser así. Si sobrara correlación, $x_k^\top r=c\ne0$, moviendo $\beta_k$ en
$\varepsilon$ el residuo pasa a $r-\varepsilon x_k$ y

$$\|r-\varepsilon x_k\|^2=\|r\|^2-2\varepsilon c+\varepsilon^2\|x_k\|^2,$$

que en $\varepsilon=c/\|x_k\|^2$ vale $\|r\|^2-c^2/\|x_k\|^2<\|r\|^2$: se podría
haber hecho mejor. Es decir, **la correlación que queda en el residuo es error que
todavía se podía quitar**, y en el mínimo no puede quedar ninguna. Guarda esta
lectura, porque en la sección 14 veremos que el lasso **sí deja correlación sin
extraer**, a propósito, y cuánta exactamente.

**Ahora sí, el objetivo.** Separando del residuo la parte que explica el OLS,

$$y - X\beta = r^o + X(\hat\beta^o-\beta),$$

el término cruzado se anula por lo anterior, y además **idénticamente en $\beta$**,
que es lo que hace de esto una identidad global y no una aproximación local:

$$2\,r^{o\top}X(\hat\beta^o-\beta)=2\big[X^\top r^o\big]^\top(\hat\beta^o-\beta)=0 .$$

Queda

$$\boxed{\ \|y-X\beta\|^2=\underbrace{\|y-X\hat\beta^o\|^2}_{\text{constante}}+(\beta-\hat\beta^o)^\top S\,(\beta-\hat\beta^o).\ }$$

Esto **cambia por completo lo que es el problema**. El objetivo no es "una suma de
cuadrados sobre los datos": es la **distancia al OLS medida en la métrica $S$**,
más una constante que no depende de $\beta$. Es decir:

> El lasso es la **proyección de $\hat\beta^{o}$ sobre la región factible**, en la
> métrica que define $X^\top X$.

Tres consecuencias inmediatas que usaremos sin volver a demostrarlas:

- Las curvas de nivel son **elipsoides concéntricos centrados en $\hat\beta^o$**,
  con forma dada por $S$. Minimizar es inflar el elipsoide desde $\hat\beta^o$
  hasta que toca la región factible.
- La Hessiana es $2S\succeq0$, luego el problema es **convexo**; y si $S\succ0$ es
  estrictamente convexo y **el mínimo es único**. Por eso cualquier método
  correcto da la misma respuesta, y podemos comparar implementaciones.
  (Si $X$ no tiene rango completo, $\hat\beta^o$ no es único, pero la proyección
  $X\hat\beta^o$ sí lo es y por tanto $r^o$ también: la identidad sigue en pie. El
  paper lo señala de pasada — *"the design matrix need not be of full rank"*.)
- Toda la información de los datos entra por solo dos objetos, $\hat\beta^o$ y $S$.

---

<a name="s3"></a>
## 3. La región factible es un politopo con esquinas en los ejes

Ahora el otro objeto. ¿Qué es $\{\beta:\sum_j|\beta_j|\le t\}$?

La clave es una identidad de una línea. Para cualquier vector de signos
$\delta\in\{-1,1\}^p$,

$$\delta^\top\beta=\sum_j\delta_j\beta_j\ \le\ \sum_j|\beta_j|,$$

con igualdad si y solo si $\delta_j=\mathrm{sign}(\beta_j)$ en toda coordenada no
nula. Como ese $\delta$ siempre existe, el máximo se alcanza:

$$\max_{\delta\in\{-1,1\}^p}\delta^\top\beta=\sum_j|\beta_j|
\qquad\Longrightarrow\qquad
\Big(\sum_j|\beta_j|\le t\ \Longleftrightarrow\ \delta^\top\beta\le t\ \ \forall\delta\Big).$$

O sea: la región es la **intersección de $2^p$ semiespacios**, un politopo con
$2^p$ caras (una por vector de signos) y $2p$ vértices, que son $\pm t\,e_j$.

![el politopo L1 en p=3](ded_polytope.png)

Con $p=3$ se ve el conteo entero: 8 caras y 6 vértices, todos **sobre los ejes**.
Y ahí está el hecho central de todo el método: **un vértice es un punto con $p-1$
coordenadas exactamente 0**. La bola $L_2$ de ridge es la esfera inscrita — sin
caras y sin esquinas.

Juntando esto con la sección 2 ya tenemos el problema entero en una imagen: un
elipsoide que crece desde $\hat\beta^o$ hasta tocar un politopo. Si toca por una
cara, ninguna coordenada se anula; si toca por una esquina, varias sí.

![Fig. 2 — la geometría](fig2_geometry.png)

Los contornos están centrados en $\hat\beta^o$ por la identidad de la sección 2, y
el contorno rojo es el que pasa por la solución que devuelve el solver. A la
izquierda toca en una esquina y sale $\beta_1=2.8\times10^{-17}$; a la derecha, con
la bola $L_2$, el contacto es tangencial y no cae en ningún eje.

Esto ya explica *cualitativamente* por qué el lasso selecciona variables. Lo que
falta —y es lo que ocupa el resto del documento— es **cuánto** encoge, **dónde**
está exactamente el punto de contacto, y **cómo** encontrarlo.

---

<a name="s4"></a>
## 4. El rango útil del presupuesto

Antes de calcular nada, acotemos el problema. Sea $t_0=\sum_j|\hat\beta_j^o|$.

- Si $t\ge t_0$, el punto $\hat\beta^o$ es factible; y siendo el mínimo global sin
  restringir, es la solución. El problema es **constante** en $[t_0,\infty)$.
- Si $t=0$ la región factible es $\{0\}$.

Luego toda la variación ocurre en $t\in[0,t_0]$, y conviene reparametrizar

$$s=\frac{t}{t_0}=\frac{t}{\sum_j|\hat\beta_j^{o}|}\in[0,1],$$

con $s=1$ el OLS y $s=0$ el vector nulo. Es la indexación que usa el paper y la que
usamos en todo el código.

---

# Parte II — Resolverlo donde se deja

Tenemos el problema planteado y su geometría. Toca calcular. La estrategia es la
de siempre: resolverlo entero en el caso más simple posible y ver qué pieza de esa
solución sobrevive al complicarlo.

<a name="s5"></a>
## 5. Un solo predictor: aparece la no-derivabilidad

Con $p=1$: minimizar $\|y-xb\|^2$ con $|b|\le t$. Por la sección 2 el objetivo es
una parábola centrada en $\hat b=x^\top y/x^\top x$, y la región factible es el
segmento $[-t,t]$. Una parábola es simétrica y decreciente hacia su vértice, luego
el mínimo sobre un segmento está en el punto del segmento **más cercano al
vértice**:

$$b^\star=\mathrm{sign}(\hat b)\,\min(|\hat b|,t).$$

Esto es **recorte**: si el OLS cabe en el presupuesto, no se toca; si no cabe, se
pone en el borde. Nada más.

Merece la pena parar aquí un momento, porque en este caso trivial ya asoma la única
dificultad técnica de todo el problema. El valor absoluto **no es derivable en 0**,
así que no se puede igualar una derivada a cero sin más. Con $p=1$ hemos podido
esquivarlo con un argumento geométrico (parábola contra segmento). Con más
coordenadas no se podrá, y habrá que tratarlo de frente. Esa no-derivabilidad va a
reaparecer tres veces más: en la sección 6, en el algoritmo de la sección 12 y en
la lectura bayesiana de la sección 20. **Siempre es la misma.**

---

<a name="s6"></a>
## 6. Diseño ortonormal: el problema se separa

El siguiente caso más simple no es "pocos predictores", sino **predictores
ortonormales**: $X^\top X=I$. Es la hipótesis que hace desaparecer la métrica.

Con $X^\top X=I$ se tiene $\hat\beta^o=X^\top y$, y la identidad de la sección 2
queda

$$\|y-X\beta\|^2=\|\beta-\hat\beta^o\|^2+\text{cte},$$

o sea distancia **euclídea**. El objetivo pasa a ser $\sum_j(\beta_j-\hat\beta_j^o)^2$:
una suma de términos que **no se mezclan**. El problema se separa en $p$ problemas
de una dimensión... salvo que **todos comparten el mismo presupuesto $t$**. Esa
atadura es lo único que queda de acoplamiento, y es lo que va a producir la fórmula.

Ahora sí hay que enfrentar la no-derivabilidad. La herramienta es el
**subdiferencial**:

$$\partial|\beta_j|=\begin{cases}\{\mathrm{sign}(\beta_j)\}&\beta_j\ne0\\[2pt] [-1,1]&\beta_j=0\end{cases}$$

El problema es convexo y $\beta=0$ es estrictamente factible para $t>0$ (condición
de Slater), así que KKT es **necesario y suficiente**. Con multiplicador
$\gamma\ge0$ para la restricción, la condición estacionaria es

$$0\in 2(\beta_j-\hat\beta_j^o)+2\gamma\,\partial|\beta_j| .$$

Tres casos, según el signo de la solución:

| | condición | despejando | consistente si |
|---|---|---|---|
| $\beta_j>0$ | $\beta_j-\hat\beta_j^o+\gamma=0$ | $\beta_j=\hat\beta_j^o-\gamma$ | $\hat\beta_j^o>\gamma$ |
| $\beta_j<0$ | $\beta_j-\hat\beta_j^o-\gamma=0$ | $\beta_j=\hat\beta_j^o+\gamma$ | $\hat\beta_j^o<-\gamma$ |
| $\beta_j=0$ | $0\in-2\hat\beta_j^o+2\gamma[-1,1]$ | — | $\lvert\hat\beta_j^o\rvert\le\gamma$ |

Son excluyentes y cubren todo $\mathbb{R}$, y se resumen en una línea:

$$\boxed{\ \hat\beta_j=\mathrm{sign}(\hat\beta_j^o)\big(|\hat\beta_j^o|-\gamma\big)^{+}\ }$$

que es la Eq. 3 del paper. El valor de $\gamma$ lo fija la holgura complementaria
$\gamma\big(\sum_j|\beta_j|-t\big)=0$: o bien $\gamma=0$ y estamos en el OLS (lo
que exige $t\ge t_0$, coherente con la sección 4), o bien la restricción está
activa y $\sum_j|\hat\beta_j|=t$.

**Y aquí está la diferencia con la sección 5.** Con $p=1$ salía recorte; ahora sale
una **traslación**: todos los coeficientes bajan la *misma* cantidad $\gamma$ y los
que se pasan de cero se quedan en cero. Lo que ha cambiado es que hay **un
multiplicador único compartido por todas las coordenadas** — precisamente la
atadura que quedaba. El recorte del caso $p=1$ es lo que se ve cuando no hay nada
con quien compartir.

![recorte contra soft thresholding](ded_clip_vs_soft.png)

A la izquierda el recorte: los coeficientes grandes quedan intactos salvo el tope.
A la derecha la traslación: **toda** la recta baja lo mismo. Solo la segunda anula
un intervalo entero alrededor del origen, y por eso solo la segunda selecciona
variables.

---

<a name="s7"></a>
## 7. El multiplicador, exacto

La fórmula anterior deja $\gamma$ definido implícitamente por
$\sum_j(|\hat\beta_j^o|-\gamma)^+=t$. No hace falta resolverlo por bisección:
la ecuación es explícita.

Sea $a_j=|\hat\beta_j^o|$ y sean $a_{(1)}\ge\dots\ge a_{(p)}$ ordenados. La función

$$\varphi(\gamma)=\sum_j(a_j-\gamma)^+$$

es continua, **lineal a trozos** con nodos en los $a_{(k)}$, y decreciente, con
$\varphi(0)=\sum_j a_j=t_0$ y $\varphi(a_{(1)})=0$. Para $t\in[0,t_0]$ hay **raíz
única**. En el tramo donde sobreviven exactamente los $k$ mayores la ecuación es
lineal,

$$\sum_{i\le k}\big(a_{(i)}-\gamma\big)=t
\qquad\Longrightarrow\qquad
\gamma=\frac{\sum_{i\le k}a_{(i)}-t}{k},$$

y el $k$ correcto es el mayor cuyo $\gamma$ resultante cumple $\gamma<a_{(k)}$.

![la raíz de phi](ded_gamma_root.png)

Las líneas de puntos son los nodos: cada vez que un coeficiente más se anula, la
pendiente sube en 1. Entre nodos es una recta, así que basta identificar el tramo y
despejar. Es `gamma_for_budget` en [orthonormal.py](orthonormal.py).

---

<a name="s8"></a>
## 8. Las otras tres reglas de encogimiento, gratis

Con la maquinaria montada, las tres alternativas clásicas salen en pocas líneas en
el mismo diseño ortonormal, y conviene tenerlas porque el resto del documento las
usa como término de comparación.

**Ridge.** Minimizar $\|y-X\beta\|^2+\gamma\|\beta\|^2$; por la sección 2 y con
$X^\top X=I$ es $\|\beta-\hat\beta^o\|^2+\gamma\|\beta\|^2$, **derivable en todas
partes** — no hay valor absoluto, no hay subdiferencial:

$$2(\beta_j-\hat\beta_j^o)+2\gamma\beta_j=0
\quad\Longrightarrow\quad
\hat\beta_j=\frac{\hat\beta_j^o}{1+\gamma}.$$

Encogimiento **proporcional**. Un factor multiplicativo nunca lleva a cero un
número no nulo: por eso ridge no selecciona.

**Mejor subconjunto.** Quedarse con los $k$ mayores en valor absoluto equivale a
fijar un umbral $\lambda$ y hacer $\hat\beta_j=\hat\beta_j^o$ si
$|\hat\beta_j^o|>\lambda$ y 0 si no. Es **discontinuo**, y esa discontinuidad es
exactamente la inestabilidad que se le reprocha: un cambio infinitesimal en los
datos puede saltar el umbral y cambiar el modelo.

**Garotte.** Minimizar $\|y-\sum_j c_j\hat\beta_j^ox_j\|^2$ con $c_j\ge0$ y
$\sum_jc_j\le t$. En diseño ortonormal el objetivo es
$\sum_j\hat\beta_j^{o2}(c_j-1)^2+\text{cte}$, y KKT con $\gamma\ge0$ para el
presupuesto y $\mu_j\ge0$ para $c_j\ge0$ da
$2\hat\beta_j^{o2}(c_j-1)+\gamma-\mu_j=0$. Si $c_j>0$ entonces $\mu_j=0$ y
$c_j=1-\gamma/2\hat\beta_j^{o2}$; si eso sale negativo, $c_j=0$. Luego

$$\hat\beta_j=\Big(1-\frac{\gamma}{2\hat\beta_j^{o2}}\Big)^{\!+}\hat\beta_j^o .$$

Corta —su umbral está en $|\hat\beta_j^o|=\sqrt{\gamma/2}$— pero el factor tiende a
1 para coeficientes grandes, o sea **encoge menos los grandes** que el lasso.

![Fig. 1 — las cuatro funciones de shrinkage](fig1_shrinkage_functions.png)

Las cuatro juntas, y la lectura es la distancia a la diagonal. La comparación que
importa es (b) contra (c): **ridge escala y el lasso traslada**, y esa es toda la
diferencia entre encoger y seleccionar. (a) corta pero a saltos, (d) corta y se
pega a la diagonal en la cola.

---

# Parte III — Qué sobrevive con predictores correlados

La ortonormalidad fue una hipótesis fuerte: hizo desaparecer la métrica $S$ y con
ella el acoplamiento entre coordenadas. Levantarla del todo lleva directamente al
caso general, que necesita algoritmo. Pero hay un paso intermedio que se resuelve a
mano y que enseña exactamente **qué parte del resultado dependía de la
ortonormalidad y cuál no**: dos predictores, con la correlación que sea.

<a name="s9"></a>
## 9. Dos predictores: el autovector que borra la correlación

Sea $p=2$ y, sin pérdida de generalidad, $\hat\beta_1^o,\hat\beta_2^o>0$.
Supongamos que la solución tiene las dos coordenadas positivas y la restricción
activa.

En el cuadrante positivo $|\beta_1|+|\beta_2|=\beta_1+\beta_2$ es **lineal**: no
hay no-derivabilidad y KKT es el de toda la vida. Con multiplicador $\gamma'$,

$$2S(\beta-\hat\beta^o)+\gamma'\mathbf 1=0
\qquad\Longrightarrow\qquad
\beta-\hat\beta^o=-\frac{\gamma'}{2}\,S^{-1}\mathbf 1,
\qquad \mathbf 1=(1,1)^\top .$$

El desplazamiento desde el OLS va en la dirección $S^{-1}\mathbf1$, que **en
principio depende de la correlación**. Aquí es donde entra la estandarización de la
sección 1, que hasta ahora solo habíamos usado para quitar el intercepto: al tener
todas las columnas la misma norma, $S$ tiene **diagonal constante**,

$$S=\begin{pmatrix}a&b\\ b&a\end{pmatrix},\qquad a=N,\quad b=N\rho .$$

Y una matriz así tiene a $\mathbf 1$ como **autovector**: $S\mathbf 1=(a+b)\mathbf 1$,
luego $S^{-1}\mathbf 1=\frac{1}{a+b}\mathbf 1$. Sustituyendo, con
$\gamma:=\gamma'/2(a+b)$,

$$\boxed{\ \beta_j=\hat\beta_j^o-\gamma\quad\text{en ambas coordenadas.}\ }$$

**La correlación entra solo por el escalar $a+b$, y queda absorbida en $\gamma$.**
La *dirección* del desplazamiento la fija la estandarización, no $\rho$. Por eso la
fórmula "vale aunque los predictores estén correlados", que es la afirmación de la
Eq. 5 del paper.

Y es exactamente la misma forma que en el caso ortonormal de la sección 6: restar
una constante común. Lo que la ortonormalidad daba de más era la *separabilidad*;
la **traslación uniforme** aguanta con solo la estandarización, siempre que $p=2$.

Imponiendo ahora $\beta_1+\beta_2=t$ se despeja el multiplicador,

$$\hat\beta_1^o+\hat\beta_2^o-2\gamma=t
\quad\Longrightarrow\quad
\gamma=\frac{\hat\beta_1^o+\hat\beta_2^o-t}{2},$$

y sustituyendo queda la forma cerrada (Eq. 6):

$$\hat\beta_1=\Big(\frac t2+\frac{\hat\beta_1^o-\hat\beta_2^o}{2}\Big)^{\!+},
\qquad
\hat\beta_2=\Big(\frac t2-\frac{\hat\beta_1^o-\hat\beta_2^o}{2}\Big)^{\!+}.$$

---

<a name="s10"></a>
## 10. Hasta dónde vale, y ridge en el mismo espejo

**El rango de validez.** La deducción supuso *ambas coordenadas positivas*. Al
apretar el presupuesto, $\gamma$ crece y la coordenada pequeña se anula cuando
$\gamma=\hat\beta_2^o$, es decir cuando $t=\hat\beta_1^o-\hat\beta_2^o$. Por debajo
de ahí el problema es de una dimensión y la solución es $(t,0)$ — no lo que da la
fórmula. Con lo cual la forma cerrada vale en

$$\hat\beta_1^o-\hat\beta_2^o\ \le\ t\ \le\ \hat\beta_1^o+\hat\beta_2^o,$$

y no solo con la cota superior, que es la única que el paper menciona. Con su
ejemplo $\hat\beta^o=(6,3)$ el corte está en $t=3$, y el solver lo confirma: en
$t=2.5$ da $(2.5,\,0)$ mientras la fórmula daría $(2.75,\,0)$.

**Ridge en el mismo caso.** Vale la pena hacerlo porque el argumento de autovectores
se reutiliza tal cual y explica un fenómeno que si no parece caprichoso. Ridge
resuelve $(S+\lambda I)\beta=S\hat\beta^o$; como $S$ e $I$ conmutan comparten
autovectores, y con diagonal constante esos autovectores son **fijos**:
$v_+=(1,1)$ con autovalor $a+b$, y $v_-=(1,-1)$ con $a-b$. Cada componente se
encoge por su propio factor:

$$\beta=\frac{a+b}{a+b+\lambda}\,P_+\hat\beta^o+\frac{a-b}{a-b+\lambda}\,P_-\hat\beta^o .$$

Si $\rho>0$ entonces $a-b<a+b$ y por tanto
$\frac{a-b}{a-b+\lambda}<\frac{a+b}{a+b+\lambda}$: **la componente antisimétrica —lo
que diferencia a los dos coeficientes— se encoge más que la simétrica, que es su
media.** Ridge, literalmente, tira a igualar los coeficientes.

Eso tiene una consecuencia contraintuitiva que se puede cuantificar. Con
$\hat\beta^o=(6,3)$ es $P_+\hat\beta^o=(4.5,4.5)$ y $P_-\hat\beta^o=(1.5,-1.5)$,
luego

$$\frac{d\beta_2}{d\lambda}\Big|_{\lambda=0}=-\frac{4.5}{a+b}+\frac{1.5}{a-b}\ >\ 0
\iff 1.5(a+b)>4.5(a-b) \iff \boxed{\ \rho>\tfrac12\ }$$

o sea: **el coeficiente pequeño sube al apretar la restricción exactamente cuando
$\rho>1/2$**, porque la atracción hacia la media le gana al encogimiento global.

![Fig. 4 — dos predictores](fig4_two_predictors.png)

Toda la Parte III en una figura. La curva negra del lasso es **una sola**: una recta
de pendiente 1 —el desplazamiento a lo largo de $-\mathbf 1$ de la sección 9— desde
$(6,3)$ hasta $(3,0)$, idéntica para los cinco $\rho$; a partir de $(3,0)$ sigue por
el eje, que es donde la forma cerrada deja de valer. Las de ridge dependen de
$\rho$ y se abren en abanico, y las de $\rho=0.68$ y $\rho=0.90$ **suben por encima
de $\beta_2=3$** antes de bajar, que es el umbral $\rho>1/2$ recién deducido.

---

<a name="s11"></a>
## 11. Por qué $p>2$ rompe la simetría

Todo lo anterior descansa en que $\mathbf 1$ sea autovector de $S$, y eso pasa
porque una matriz $2\times2$ simétrica con diagonal constante **solo tiene un grado
de libertad fuera de la diagonal**. Con $p>2$ hay $\binom p2$ correlaciones
distintas y $\mathbf1$ ya no es autovector en general: $S^{-1}\mathbf 1$ deja de ser
proporcional a $\mathbf1$ y **el desplazamiento deja de ser el mismo en todas las
coordenadas**.

Las consecuencias son concretas:

- Se acabaron las formas cerradas **incondicionales**: ya no hay una regla que
  lleve $\hat\beta^o$ a $\hat\beta$ sin más. Hay que resolver el problema de
  verdad. (Lo que sí sobrevive es una forma cerrada *condicionada* a saber qué
  coordenadas sobreviven y con qué signo; la deduciremos en la sección 14, una vez
  tengamos con qué comprobarla.)
- Los signos pueden cambiar. Con $p=2$ el movimiento es a lo largo de $-\mathbf1$ y
  las coordenadas se **paran** en 0 en vez de cruzarlo, así que la solución vive en
  el mismo cuadrante que el OLS. Con $p>2$ no hay tal garantía, y el paper enseña un
  ejemplo en el que el lasso cae en un octante distinto (su Fig. 3).

Se acabó lo que se podía hacer a mano. Toca algoritmo.

---

# Parte IV — El caso general

<a name="s12"></a>
## 12. De la geometría al algoritmo

La sección 3 ya dio el ingrediente: la región factible es la intersección de
$2^p$ semiespacios $\delta^\top\beta\le t$. Y por la sección 2 el objetivo es
cuadrático. Luego el lasso, sin aproximación ninguna, es un

> **programa cuadrático con $2^p$ restricciones lineales.**

El problema es el $2^p$: con $p=10$ ya son 1024 restricciones, y con $p=20$, un
millón. Pero de todas ellas **casi ninguna está activa** en el óptimo. La idea del
algoritmo (Sec. 6 del paper) es no construirlas: mantener un conjunto $E$ de
restricciones candidatas, resolver el QP pequeño que solo las incluye a ellas, y si
la solución viola el presupuesto, añadir la restricción violada —que es
$\delta=\mathrm{sign}(\hat\beta)$, por la identidad de la sección 3— y repetir.

Y fíjate que esa enumeración de vectores de signos es, otra vez, **la
no-derivabilidad de la sección 5**: el valor absoluto no es una función suave, sino
el máximo de $2^p$ funciones lineales, y el precio de tratarlo con herramientas
lineales es tener que ir descubriendo cuál de esas piezas manda.

---

<a name="s13"></a>
## 13. Por qué para, por qué es óptimo y por qué los ceros son exactos

Tres cosas hay que comprobar, y las tres son cortas.

**Termina.** Cada pasada añade un elemento a $E$, y hay a lo sumo $2^p$ vectores de
signos. (En la práctica bastan muchas menos: midiendo sobre diseños aleatorios,
la media está en torno a $0.5p$.)

**Lo que devuelve es óptimo del problema completo.** Sea $P$ el problema con las
$2^p$ restricciones y $P_E$ el relajado que solo tiene las de $E$. Al tener menos
restricciones, $P_E$ tiene región factible mayor, luego $\min P_E\le\min P$. Al
salir del bucle, $\hat\beta$ (i) **alcanza** $\min P_E$ y (ii) cumple
$\sum_j|\hat\beta_j|\le t$, o sea es **factible para $P$**. Entonces

$$\min P\ \le\ g(\hat\beta)\ =\ \min P_E\ \le\ \min P,$$

y todo son igualdades. Un punto factible que además resuelve una relajación resuelve
el original; no hay que verificar nada más.

**Los ceros son exactos.** Si en el óptimo hay dos vectores de signos activos
$\delta$ y $\delta'$ que difieren **solo** en la coordenada $j$, restando sus
igualdades $\delta^\top\hat\beta=t$ y $\delta'^\top\hat\beta=t$:

$$(\delta-\delta')^\top\hat\beta=0\ \Longrightarrow\ 2\delta_j\hat\beta_j=0
\ \Longrightarrow\ \hat\beta_j=0 .$$

Es una consecuencia **algebraica** de que dos caras del politopo se corten en una
arista o un vértice —la imagen de la sección 3, ahora en cuentas—, no un umbral
numérico. Por eso en la implementación los ceros salen a $10^{-17}$: lo único que
queda por debajo es el redondeo del sistema lineal.

**Un apunte sobre el $2^p$.** Ese exponente se puede cambiar por variables extra,
en vez de gestionarlo con un conjunto activo. Escribiendo
$\beta_j=\beta_j^+-\beta_j^-$ con $\beta^\pm\ge0$ y
$\sum_j\beta_j^++\sum_j\beta_j^-\le t$, se pasa de $p$ variables con $2^p$
restricciones a $2p$ variables con solo $2p+1$. Que sea el mismo problema se ve
yendo en las dos direcciones: dado $\beta$ factible, $\beta_j^\pm$ son sus partes
positiva y negativa y $\sum_j(\beta_j^++\beta_j^-)=\sum_j|\beta_j|\le t$; y dado
$(\beta^+,\beta^-)$ factible, la desigualdad triangular da
$\sum_j|\beta_j^+-\beta_j^-|\le\sum_j(\beta_j^++\beta_j^-)\le t$. Las dos
aplicaciones conservan el objetivo, luego los mínimos coinciden. (Esa segunda
desigualdad es estricta si $\beta_j^+$ y $\beta_j^-$ son ambos positivos, pero en un
óptimo eso no puede pasar: bajar los dos lo mismo deja $\beta$ igual y **libera
presupuesto**.) Es la variante que el paper atribuye a David Gay.

Con esto ya se puede recorrer la trayectoria entera barriendo $s$ de 0 a 1:

![Fig. 5 — trayectorias de próstata](fig5_prostate_paths.png)

Cada curva es un coeficiente. Los tramos rectos son los intervalos en que el
conjunto activo no cambia, y los codos son los $s$ en que una variable entra o sale
— los mismos codos que reaparecerán en la sección 15. En $s=1$ se recupera el OLS
y en $s=0$ todo se anula, como exigía la sección 4.

Que esos tramos salgan **rectos** es un hecho que la figura enseña y que todavía no
hemos demostrado. Sale de la sección siguiente.

---

<a name="s14"></a>
## 14. OLS sobre el conjunto activo, menos un sesgo

Ya sabemos calcular $\hat\beta(t)$. Pero calcular no es saber qué forma tiene la
respuesta: en la Parte II teníamos una fórmula, y por eso pudimos comparar cuatro
reglas de encogimiento en dos páginas; ahora tenemos un número que sale de un
bucle. La pregunta que cierra esta parte es si queda algo de la Eq. 3 con $S$
general.

**Las condiciones de optimalidad, sin ortonormalidad.** En la sección 6 escribimos
la estacionariedad con el subdiferencial bajo $X^\top X=I$. Nada de aquel argumento
usaba la hipótesis salvo para separar el problema, así que lo repetimos con la
métrica puesta. Con la forma penalizada —lícita por la sección 15, que aún no
necesitamos más que como cambio de nombre—

$$0\in-2X^\top(y-X\beta)+2\lambda\,\partial\|\beta\|_1 ,$$

y como $\partial\|\beta\|_1$ se descompone coordenada a coordenada, esto es

$$x_j^\top(y-X\hat\beta)=\lambda\,\mathrm{sign}(\hat\beta_j)\ \ (\hat\beta_j\ne0),
\qquad
\big|x_j^\top(y-X\hat\beta)\big|\le\lambda\ \ (\hat\beta_j=0).$$

La convexidad y la condición de Slater son las mismas de la sección 6, luego esto
sigue siendo **necesario y suficiente**. Conviene subrayar que es **exacto**: no
hemos aproximado nada, y en particular no hemos linealizado el valor absoluto.

Ya se puede leer algo. Que el residuo del lasso tenga la *misma* correlación
$\lambda$ con todas las columnas activas, y no más de $\lambda$ con las nulas, es
una condición fuerte y comprobable: sobre los datos de próstata en $s=0.44$ las
tres coordenadas activas dan $17.985$ con una dispersión entre ellas de
$7.5\times10^{-14}$, y las nulas se quedan en $16.31$ o menos.

Y aquí se cierra lo que quedó apuntado en la sección 2. Allí vimos que el OLS deja
el residuo **ortogonal a todas las columnas**, porque cualquier correlación
sobrante sería error que aún se podía quitar. El lasso **deja correlación sin
extraer, a propósito y en una cantidad exacta**: sacarla costaría más presupuesto
$L_1$ del que vale. Donde el OLS da $\max_j|x_j^\top r|=3.3\times10^{-14}$, el
lasso da $\lambda$. Las ecuaciones normales de la sección 2 son el caso
$\lambda=0$ de esto, que es justo el extremo $t\ge t_0$ de la sección 4.

**Despejar.** Sea $A=\{j:\hat\beta_j\ne0\}$ el conjunto activo y $s_A$ su vector de
signos. Como las coordenadas de fuera valen 0, se tiene $X\hat\beta=X_A\hat\beta_A$
y las ecuaciones activas son un sistema lineal ordinario:

$$X_A^\top\big(y-X_A\hat\beta_A\big)=\lambda s_A
\qquad\Longrightarrow\qquad
\boxed{\ \hat\beta_A=\hat\beta^{\,\mathrm{ols}(A)}-\lambda\,(X_A^\top X_A)^{-1}s_A\ }$$

Es decir: **el OLS reajustado sobre las variables activas, desplazado en la
dirección $(X_A^\top X_A)^{-1}s_A$.** El matiz de "reajustado" no es cosmético — no
es el OLS completo restringido a $A$, sino el que sale de correr mínimos cuadrados
solo con esas columnas. En próstata a $s=0.44$ el reajuste da
$(0.6468,\,0.2512,\,0.2744)$ y el OLS completo $(0.6883,\,0.2245,\,0.3155)$ sobre
las mismas tres. Con el $\lambda$ de arriba, la fórmula reproduce lo que devuelve
el solver con un error máximo de $4.1\times10^{-15}$.

**Las tres cosas que ya sabíamos eran esta.** Especializando:

| poniendo | queda | que es |
|---|---|---|
| $S=I$ y $A$ todo | $\hat\beta=\hat\beta^o-\lambda s$ | la Eq. 3, sección 6 |
| $p=2$ estandarizado, $s_A=\mathbf1$ | $S^{-1}\mathbf1=\mathbf1/(a+b)$, traslación uniforme | la Eq. 5, sección 9 |
| $\lambda=0$ | $X^\top r=0$ | las ecuaciones normales, sección 2 |

El autovector de la sección 9 no era una casualidad de $p=2$: era el caso en que
$(X_A^\top X_A)^{-1}s_A$ resulta proporcional a $s_A$, que es exactamente cuando el
desplazamiento se ve como un encogimiento.

**Primera consecuencia: los tramos son rectos, y ahora está demostrado.** En
cualquier intervalo de $\lambda$ donde $A$ y $s_A$ no cambian, la fórmula es
**afín en $\lambda$**, con pendiente constante $-(X_A^\top X_A)^{-1}s_A$. Eso
convierte en teorema lo que la Fig. 5 enseñaba y lo que la sección 15 dará por
bueno al hablar de $\lambda(t)$: los codos son precisamente los $\lambda$ donde una
coordenada entra en $A$ o sale de él. Es la estructura que explota LARS.

**Segunda consecuencia: el desplazamiento no es un encogimiento por coordenadas.**
La dirección $(X_A^\top X_A)^{-1}s_A$ mezcla coordenadas, y nada obliga a que su
componente $j$ tenga el signo de $s_j$. Cuando no lo tiene, ese coeficiente se
mueve **alejándose de cero** al apretar el presupuesto.

![el desplazamiento no es encogimiento](ded_active_set.png)

A la izquierda, un diseño de $p=3$ con $r_{12}=0$ y $r_{13}=r_{23}=0.65$, elegido
porque ahí $R^{-1}\mathbf1=(2.26,\,2.26,\,-1.94)$ tiene la tercera componente
negativa. El solver da $\hat\beta^o=(1.03,\,0.98,\,0.97)$, y sin embargo
$\hat\beta_3$ sube hasta $1.81$ —casi el doble de su OLS— mientras
$\sum_j|\beta_j|$, la línea discontinua, baja monótona como tiene que bajar. Lo
único que encoge es el escalar del presupuesto.

Es el mismo fenómeno que dedujimos para ridge en la sección 10, donde el
coeficiente pequeño subía si $\rho>1/2$ porque la atracción hacia la media ganaba
al encogimiento. Aquí gana la geometría de $S_A^{-1}$ sobre la del signo. Y hace
falta $p\ge3$: con $p=2$ el movimiento va por $-\mathbf1$ y no puede darse la
vuelta, que es justo lo que probamos en la sección 11.

A la derecha está la afirmación general. Sobre 140 diseños correlados al azar, a
presupuesto fijo $s=0.5$, se dibuja $\hat\beta_j$ contra $\hat\beta_j^o$. La curva
roja es lo que promete la Eq. 3 con el $\gamma$ mediano; lo que sale es una nube.
En $\hat\beta_j^o\approx2$ el lasso reparte valores entre $0$ y $2.84$: si
existiera una función $h$ con $\hat\beta_j=h(\hat\beta_j^o)$, ese segmento azul
tendría longitud cero. Con lo cual, dicho sin rodeos:

> Fuera del diseño ortonormal **no hay ninguna regla de encogimiento por
> coordenadas**. Las cuatro curvas de la Fig. 1 son el retrato de un caso
> particular, no la definición del método.

**Por qué esto no es un algoritmo.** La fórmula presupone $A$ y $s_A$, que es
tanto como presuponer la solución. Y la combinatoria no ha desaparecido: donde la
sección 12 tenía $2^p$ vectores de signos, aquí hay hasta $3^p$ configuraciones
—cada coordenada nula, positiva o negativa—, así que enumerar sigue sin ser una
opción. La fórmula **caracteriza**; el algoritmo de la sección 12 **encuentra**.
Por eso el objeto natural del lasso es la trayectoria entera y no el punto suelto,
y por eso la sección siguiente empieza por ordenar cómo se la indexa.

---

# Parte V — Elegir el presupuesto

Ya sabemos calcular $\hat\beta(t)$ para cada $t$. Falta lo que en la práctica
decide el resultado: **qué $t$**. Antes de poder discutirlo hace falta una
herramienta más.

<a name="s15"></a>
## 15. La correspondencia entre restringir y penalizar

Todo el desarrollo ha usado la forma **restringida**, $\sum_j|\beta_j|\le t$. Casi
toda la literatura posterior usa la **penalizada**,
$\|y-X\beta\|^2+\lambda\sum_j|\beta_j|$. Son el mismo problema, y conviene saber
exactamente en qué sentido.

Sea la función valor

$$V(t)=\min\Big\{\|y-X\beta\|^2:\ \sum_j|\beta_j|\le t\Big\}.$$

$V$ es **convexa** —función valor de un programa convexo respecto al término
independiente de la restricción— y **no creciente**, porque más presupuesto no
puede empeorar el mínimo. El multiplicador cumple $\lambda(t)\in-\partial V(t)$
(salvo un factor 2 según cómo se escriba el lagrangiano); y al ser $V$ convexa,
$\partial V$ es no decreciente, luego

$$\lambda(t)\ \text{es no creciente en } t .$$

La correspondencia es **monótona**, y por eso da igual indexar por $t$, por
$\lambda$ o por $s$. Lo que sí importa es no mezclar convenciones sin decirlo: un
$\lambda$ de `glmnet` no es el de aquí.

![la función valor](ded_value_function.png)

A la izquierda $V(t)$, convexa y no creciente, y plana a partir de $t_0$ como pedía
la sección 4. A la derecha, dos cálculos independientes de $\lambda$ superpuestos:
el que sale de KKT en la sección 14 y la derivada numérica
$-V'(t)/2$. Coinciden. Y se ve que $\lambda(t)$ es **lineal a trozos**, con los
codos donde cambia el conjunto activo: los mismos de la Fig. 5. Esa estructura
lineal a trozos —demostrada en la sección 14— es lo que explota LARS (Efron et al.,
2004) para recorrer la trayectoria entera de un tirón.

---

<a name="s16"></a>
## 16. Qué querríamos minimizar

Lo que interesa de un ajuste es cuánto se parece a la verdad, no cuánto se parece a
los datos con los que se hizo. Con $Y=\eta(X)+\epsilon$, $E[\epsilon]=0$,
$\mathrm{var}(\epsilon)=\sigma^2$ y $\epsilon$ independiente de $X$, se definen el
**error de modelo** y el **error de predicción**

$$\mathrm{ME}=E\{\hat\eta(X)-\eta(X)\}^2,
\qquad
\mathrm{PE}=E\{Y-\hat\eta(X)\}^2 .$$

Están relacionados de forma trivial. Desarrollando con $\hat\eta$ fijo,

$$\mathrm{PE}=E\{\eta(X)+\epsilon-\hat\eta(X)\}^2
=\underbrace{E\{\eta-\hat\eta\}^2}_{\mathrm{ME}}
+2\underbrace{E[\epsilon(\eta-\hat\eta)]}_{=0\ \text{por independencia}}
+\underbrace{E[\epsilon^2]}_{\sigma^2},$$

o sea $\mathrm{PE}=\mathrm{ME}+\sigma^2$. **Difieren en una constante que no
depende de $t$**, luego se minimizan en el mismo sitio: podemos elegir $t$
minimizando el error de predicción —que sí se puede estimar— aunque lo que nos
importe sea el error de modelo, que no.

En el caso lineal $\eta(x)=x^\top\beta$ el error de modelo tiene forma cerrada:

$$\mathrm{ME}=E_X\big[(\hat\beta-\beta)^\top xx^\top(\hat\beta-\beta)\big]
=(\hat\beta-\beta)^\top V(\hat\beta-\beta),$$

con $V=E[xx^\top]$ la covarianza poblacional. En una simulación se conocen $\beta$ y
$V$, así que **el error de modelo se calcula exacto**, sin conjunto de test y sin el
ruido de muestreo que este acarrearía. Es la métrica con la que el paper compara
métodos en su Tabla 3.

**Lo que no vale.** El RSS de entrenamiento **no** estima $\mathrm{PE}$, y no es
cuestión de sesgo pequeño: es monótono. El conjunto factible $\{\sum_j|\beta_j|\le
s\,t_0\}$ **crece** con $s$, y el mínimo sobre un conjunto mayor solo puede bajar,
luego $\mathrm{RSS}(s)$ es no creciente y siempre elegiría $s=1$. Hacen falta datos
que el ajuste no haya visto — o un sustituto analítico de esos datos, que es lo que
construyen las tres secciones siguientes.

---

<a name="s17"></a>
## 17. Una linealización con dos consecuencias

La vía directa es **validación cruzada**: partir la muestra, ajustar en unos
pliegues y medir en el que queda fuera. No necesita más teoría, pero cuesta un
ajuste por pliegue y por punto de la rejilla.

La alternativa barata necesita tratar el lasso como si fuera lineal, y hay una
manera de hacerlo. Parte de una identidad tonta:

$$\sum_j|\beta_j|=\sum_j\frac{\beta_j^2}{|\beta_j|}.$$

En sí no dice nada. Pero si **congelamos** $W=\mathrm{diag}(|\hat\beta_j|)$ en la
solución ya calculada, el lado derecho pasa a ser una forma **cuadrática**
$\beta^\top W^-\beta$, y el problema penalizado se convierte en un ridge, que sí es
derivable y sí es lineal:

$$-2X^\top(y-X\beta)+2\lambda W^-\beta=0
\quad\Longrightarrow\quad
\tilde\beta=(X^\top X+\lambda W^-)^{-1}X^\top y .$$

Conviene decir qué se paga y qué no. El $\lambda$ **no** se paga: ya salió exacto
de KKT en la sección 14. Lo que sí exige esta aproximación son las **dos cosas** de
abajo, que necesitan una matriz sombrero y por tanto un ajuste lineal de verdad; y
como $W$ se ha congelado, ninguna de las dos es exacta.

**(a) Errores estándar.** $\tilde\beta=My$ con
$M=(X^\top X+\lambda W^-)^{-1}X^\top$ es **lineal en $y$**, así que su covarianza es
inmediata con $\mathrm{Cov}(y)=\sigma^2I$:

$$\mathrm{Cov}(\tilde\beta)=\sigma^2MM^\top
=\sigma^2(X^\top X+\lambda W^-)^{-1}X^\top X(X^\top X+\lambda W^-)^{-1},$$

que es la Eq. 7 del paper. Tiene un defecto que el propio paper señala: si
$\hat\beta_j=0$ entonces $1/|\hat\beta_j|\to\infty$, la fila $j$ de $M$ se anula y
la varianza estimada sale **exactamente 0** — un intervalo de confianza degenerado
justo para los coeficientes sobre los que menos seguros estamos.

> Ese defecto resuelve de paso una ambigüedad. La Eq. 9 solo dice que $W^-$ es "una
> inversa generalizada", y leída como pseudoinversa de Moore–Penrose sería
> $W^-_{jj}=0$ sobre los coeficientes nulos, o sea penalización **cero** sobre
> ellos — al revés de lo que hace falta. Que las varianzas salgan 0 obliga a la
> lectura $1/|\beta_j|\to\infty$: los coeficientes nulos **salen del ajuste**. Es la
> que usa [selection.py](selection.py).

**(b) Parámetros efectivos.** Los valores ajustados son
$\hat y=X\tilde\beta=Hy$ con $H=X(X^\top X+\lambda W^-)^{-1}X^\top$: un **linear
smoother**. Para un ajuste lineal ordinario sobre $q$ regresores $H$ es una
proyección y $\mathrm{tr}(H)=q$ = número de parámetros; la traza generaliza esa
cuenta a suavizadores que no son proyecciones. Se define entonces

$$p(t)=\mathrm{tr}\{X(X^\top X+\lambda W^-)^{-1}X^\top\},$$

que vale $p$ en $\lambda=0$ y tiende a 0 cuando todo se anula.

---

<a name="s18"></a>
## 18. De la validación cruzada dejando uno fuera al GCV

Ahora que el lasso tiene una matriz sombrero $H$ —aunque sea por aproximación—
podemos usar la maquinaria de los suavizadores lineales, que permite hacer
validación cruzada **sin volver a ajustar**.

Sea $\hat y=Hy$ y sea $\hat y_i^{(-i)}$ la predicción en $i$ ajustando sin la
observación $i$. Sea $\tilde y$ igual a $y$ pero con la entrada $i$ sustituida por
$\hat y_i^{(-i)}$. Si el suavizador es autoconsistente —ajustar a $\tilde y$
reproduce el mismo ajuste en $i$—, entonces $(H\tilde y)_i=\hat y_i^{(-i)}$, y por
linealidad

$$\hat y_i-\hat y_i^{(-i)}=(Hy)_i-(H\tilde y)_i=h_{ii}\big(y_i-\tilde y_i\big)
=h_{ii}\big(y_i-\hat y_i^{(-i)}\big).$$

Sumando y restando $\hat y_i$ y despejando:

$$\boxed{\ y_i-\hat y_i^{(-i)}=\frac{y_i-\hat y_i}{1-h_{ii}}\ }$$

**Los $N$ ajustes salen de uno solo**, dividiendo cada residuo por $1-h_{ii}$. El
GCV da un paso más y sustituye cada $h_{ii}$ por su media $\mathrm{tr}(H)/N$, lo
que evita calcular la diagonal de $H$ y hace el criterio invariante por rotaciones:

$$\mathrm{GCV}(t)=\frac1N\,\frac{\mathrm{rss}(t)}{\{1-p(t)/N\}^2}.$$

El denominador es el precio por complejidad: cuantos más parámetros efectivos,
más se infla el RSS antes de compararlo.

![GCV y CV sobre próstata](ded_gcv_cv.png)

Conviene saber que **este puente no es gratis**. El lasso no es un linear smoother
y $H$ salió de congelar $W$; en los datos de próstata, ni el GCV ni la CV quíntuple
caen donde el paper reporta ($\hat s=0.44$), y la curva del GCV no es plana en esa
zona: vale 0.578 en 0.44 contra 0.516 en su mínimo. La CV sí es bastante plana
—la banda es $\pm1$ error estándar entre pliegues— así que su 0.63 discrimina poco;
el GCV no tiene esa excusa. En el [README](README.md) está el diagnóstico completo.

---

<a name="s19"></a>
## 19. El riesgo de Stein, que el caso ortonormal ya permite

Queda una tercera vía, y es la más barata de todas: **una sola optimización**, sin
pliegues y sin trazas. Solo es posible porque la Parte II nos dio la forma cerrada.

El lema de Stein dice que si $z\sim N_p(\mu,I)$ y $\hat\mu=z+g(z)$ con $g$ casi
diferenciable, entonces

$$E\|\hat\mu-\mu\|^2=p+E\Big(\|g(z)\|^2+2\sum_{i}\frac{\partial g_i}{\partial z_i}\Big),$$

y lo notable es que el lado derecho **se puede evaluar con los datos**, sin conocer
$\mu$: es un estimador insesgado del riesgo. Aplicándolo al *soft thresholding* de
la sección 6, con $g_i=\hat\mu_i-z_i$:

| región | $\hat\mu_i$ | $g_i$ | $g_i^2$ | $\partial g_i/\partial z_i$ |
|---|---|---|---|---|
| $\lvert z_i\rvert>\gamma$ | $z_i-\gamma\,\mathrm{sign}(z_i)$ | $-\gamma\,\mathrm{sign}(z_i)$ | $\gamma^2$ | $0$ |
| $\lvert z_i\rvert<\gamma$ | $0$ | $-z_i$ | $z_i^2$ | $-1$ |

En la primera región $z_i^2>\gamma^2$ y se queda $\gamma^2$; en la segunda
$z_i^2<\gamma^2$ y se queda $z_i^2$. En los dos casos, **el menor de los dos**:

$$\|g\|^2=\sum_i\min(|z_i|,\gamma)^2,
\qquad 2\sum_i\frac{\partial g_i}{\partial z_i}=-2\,\#\{i:|z_i|<\gamma\},$$

y por tanto

$$\boxed{\ E\|\hat\mu-\mu\|^2=p-2\,\#\{i:|z_i|<\gamma\}+\sum_{i}\min(|z_i|,\gamma)^2 .\ }$$

Minimizando en $\gamma$ se obtiene $\hat\gamma$, y de ahí el presupuesto
correspondiente, $\hat t=\sum_j(|\hat\beta_j^o|-\hat\gamma)^+$, que no es más que
evaluar la fórmula de la sección 6 y sumar.

**El paper imprime esta fórmula con `max` donde va `min`**, y con el cuadrado sobre
$\gamma$ en vez de fuera del paréntesis. Es falsable sin discusión, porque un
estimador insesgado tiene que promediar al riesgo verdadero, y basta mirar los dos
extremos: en $\gamma\to0$ no se umbraliza nada, $\hat\mu=z$ y el riesgo es $p$; en
$\gamma\to\infty$ se anula todo, $\hat\mu=0$ y el riesgo es $\|\mu\|^2$.

| | riesgo verdadero | con `min` | como se imprime |
|---|---|---|---|
| $\gamma\to0$ | $p$ | $p-0+0=p$ ✔ | $p+\sum_i\lvert z_i\rvert$ ✘ |
| $\gamma\to\infty$ | $\lVert\mu\rVert^2$ | $p-2p+\sum z_i^2\to\lVert\mu\rVert^2$ ✔ | $p-2p+p\gamma^2\to\infty$ ✘ |

(en el segundo caso se usa $E\sum_iz_i^2=p+\|\mu\|^2$). La versión con `min` es
además la de Donoho & Johnstone (1994), que el propio paper cita dos líneas más
abajo. Comprobado sobre el escaneo del JSTOR, no es un fallo de OCR.

![la errata de Stein](ded_stein.png)

Contrastado con Monte Carlo: la curva roja se superpone a la negra en todo el rango
y la azul se dispara. En $\gamma$ grande el riesgo verdadero tiende a
$\|\mu\|^2=15.25$, la versión con `min` da 15.33 y la impresa, 1144.

---

# Parte VI — La misma cosa desde otro sitio

<a name="s20"></a>
## 20. La prior de Laplace

Todo lo anterior ha sido optimización. El mismo estimador sale de un planteamiento
completamente distinto, y la traducción explica de otra manera por qué hay ceros.

Con verosimilitud $y\mid\beta\sim N(X\beta,\sigma^2I)$ y prior independiente de
Laplace $f(\beta_j)=\frac{1}{2\tau}e^{-|\beta_j|/\tau}$, la posterior es

$$f(\beta\mid y)\ \propto\ \exp\Big\{-\frac{1}{2\sigma^2}\|y-X\beta\|^2\Big\}\prod_j\exp\Big(-\frac{|\beta_j|}{\tau}\Big),$$

y tomando $-\log$ y multiplicando por $2\sigma^2$ —que no mueve el mínimo—

$$-2\sigma^2\log f(\beta\mid y)=\|y-X\beta\|^2+\frac{2\sigma^2}{\tau}\sum_j|\beta_j|+\text{cte}.$$

La **moda a posteriori** es exactamente el lasso penalizado con
$\lambda=2\sigma^2/\tau$. Ridge es lo mismo con prior normal.

![Fig. 7 — las dos priors](ded_priors.png)

Dibujadas con la **misma varianza**, para que la comparación sea de forma. La doble
exponencial pone más masa cerca de 0 *y* en las colas, a costa de la zona
intermedia: es el sesgo del lasso a producir coeficientes o bien nulos o bien
grandes, ahora como afirmación sobre lo que creemos *a priori* del mundo.

Y el pico de esa densidad en 0 es el mismo $|\beta_j|$ no derivable con el que
tropezamos en la sección 5, que obligó al subdiferencial en la sección 6 y a
enumerar vectores de signos en la sección 12. Visto así: **lo que hace difícil
optimizar el lasso y lo que le hace seleccionar variables son el mismo hecho.** Una
prior suave da una moda interior y un problema fácil; una prior con pico da ceros y
un problema no derivable. No se puede tener una cosa sin la otra.
