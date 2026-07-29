# Attention Is All You Need

**Autores:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (Google Brain / Google Research / U. Toronto) · **Año:** 2017 · **Venue:** NIPS 2017 (31st Conference on Neural Information Processing Systems), Long Beach, CA · **Enlace/DOI:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
**Campo:** ML / deep learning · **Leído:** 2026-07-29

## TL;DR

Los autores proponen el **Transformer**: una arquitectura encoder-decoder que **elimina por completo la recurrencia y las convoluciones** y se apoya únicamente en mecanismos de atención. La pieza central es la **scaled dot-product attention**, $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$, replicada en $h$ cabezas paralelas. El argumento no es solo de calidad sino de **complejidad computacional**: una capa recurrente necesita $O(n)$ operaciones secuenciales y conecta posiciones distantes con caminos de longitud $O(n)$; la self-attention necesita $O(1)$ secuenciales y camino máximo $O(1)$ — todo se paraleliza y cualquier par de posiciones se "ve" directamente. Resultado: **28.4 BLEU** en WMT14 inglés→alemán (más de 2 puntos sobre el estado del arte, incluidos ensembles) entrenando 3.5 días en 8 GPUs, una fracción del coste de los competidores. Es el paper sobre el que se construye todo el ecosistema actual de LLMs.

## Contexto y motivación

Hacia 2017 el estado del arte en traducción automática eran arquitecturas encoder-decoder recurrentes (LSTM, GRU) **con atención añadida** — la línea Bahdanau (2014) → Luong (2015) → GNMT (2016). El problema es estructural: una RNN genera estados ocultos $h_t = f(h_{t-1}, x_t)$, y esa dependencia de $h_{t-1}$ **impide paralelizar dentro de un ejemplo**. Con secuencias largas es fatal, porque además las restricciones de memoria limitan el *batching* entre ejemplos.

Las alternativas convolucionales (ByteNet, ConvS2S, Extended Neural GPU) sí paralelizan, pero el número de operaciones para relacionar dos posiciones **crece con la distancia**: lineal en ConvS2S, logarítmico en ByteNet. Eso dificulta aprender dependencias largas.

La observación clave: en esos modelos la atención ya hacía el trabajo pesado de conectar posiciones arbitrarias, pero **siempre acompañada de una RNN**. Los autores se preguntan qué pasa si se quita la RNN y se deja solo la atención. La respuesta —y el título— es que basta.

## Metodología

### Scaled dot-product attention (Eq. 1)

La atención mapea una **query** y un conjunto de pares **key-value** a una salida, que es una suma ponderada de los values; el peso de cada value lo da una función de compatibilidad entre la query y su key. Empaquetando las queries en $Q$, las keys en $K$ y los values en $V$:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**¿Por qué $\sqrt{d_k}$?** Es el detalle más citado y el paper lo justifica en la nota 4: si las componentes de $q$ y $k$ son independientes con media 0 y varianza 1, entonces $q\cdot k = \sum_{i=1}^{d_k} q_ik_i$ tiene media 0 y **varianza $d_k$**. Con $d_k$ grande los productos escalares se disparan en magnitud, empujando al softmax a regiones de **gradiente minúsculo** (saturación). Dividir por $\sqrt{d_k}$ renormaliza la varianza a 1 y lo evita.

Frente a la **atención aditiva** de Bahdanau (que usa una red feed-forward de una capa como función de compatibilidad), la multiplicativa tiene complejidad teórica similar pero es **mucho más rápida y eficiente en memoria** en la práctica, porque se reduce a multiplicación de matrices altamente optimizada.

### Multi-head attention

En vez de una sola atención en dimensión $d_{\text{model}}$, se proyectan linealmente $Q$, $K$, $V$ **$h$ veces** con proyecciones aprendidas distintas, se atiende en paralelo, y se concatena:

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,
\qquad \mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Motivación: permite atender conjuntamente a información de **distintos subespacios de representación** en distintas posiciones; con una sola cabeza el promediado lo impide. Configuración: $h=8$, $d_k = d_v = d_{\text{model}}/h = 64$. Como cada cabeza opera en dimensión reducida, **el coste total es similar al de una atención de cabeza única a dimensión completa**.

### Los tres usos de la atención en el modelo

1. **Encoder-decoder attention:** las queries vienen de la capa anterior del decoder, las keys y values del *output* del encoder. Cada posición del decoder atiende a toda la entrada (es la atención clásica de seq2seq).
2. **Self-attention del encoder:** $Q$, $K$, $V$ vienen todas del mismo sitio (la capa anterior). Cada posición atiende a todas.
3. **Self-attention enmascarada del decoder:** igual, pero cada posición solo puede atender **hasta e incluyendo ella misma**. Se implementa poniendo a $-\infty$ las entradas ilegales *antes* del softmax. Preserva la propiedad autorregresiva.

### El resto de la arquitectura

- **Pilas:** $N=6$ capas idénticas en encoder y decoder. El decoder añade un tercer sub-layer (la atención al encoder).
- **Residual + norm:** cada sub-layer se envuelve como $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Todos los sub-layers y embeddings producen $d_{\text{model}} = 512$ para que los residuales encajen.
- **FFN por posición (Eq. 2):** $\mathrm{FFN}(x) = \max(0, xW_1+b_1)W_2+b_2$, aplicada idénticamente a cada posición, con capa interna $d_{ff}=2048$. Equivale a dos convoluciones de kernel 1.
- **Embeddings:** compartidos entre las dos capas de embedding y la transformación pre-softmax; multiplicados por $\sqrt{d_{\text{model}}}$.
- **Positional encoding:** como no hay recurrencia ni convolución, hay que inyectar el orden. Usan sinusoides de frecuencias en progresión geométrica de $2\pi$ a $10000\cdot 2\pi$:
$$PE_{(pos,2i)} = \sin\!\left(pos/10000^{2i/d_{\text{model}}}\right), \qquad
PE_{(pos,2i+1)} = \cos\!\left(pos/10000^{2i/d_{\text{model}}}\right)$$
La hipótesis: para cualquier desplazamiento fijo $k$, $PE_{pos+k}$ es **función lineal** de $PE_{pos}$, lo que facilitaría aprender a atender por posiciones relativas. Eligieron la versión sinusoidal (frente a embeddings aprendidos, que dan resultados casi idénticos) porque **podría extrapolar** a secuencias más largas que las vistas en entrenamiento.

### El argumento de complejidad (Sec. 4, Tabla 1)

Es la justificación teórica del diseño, y merece leerse con atención. Con $n$ = longitud de secuencia, $d$ = dimensión de representación, $k$ = kernel, $r$ = vecindario:

| Tipo de capa | Complejidad por capa | Ops. secuenciales | Camino máximo |
|---|---|---|---|
| **Self-attention** | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrente | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional | $O(k\cdot n\cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (restringida) | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |

Tres criterios: coste por capa, paralelizable (ops. secuenciales) y **longitud del camino** entre dependencias largas — cuanto más corto el camino que recorren las señales hacia delante y hacia atrás, más fácil aprender dependencias lejanas. La self-attention gana en los tres salvo en coste por capa, y ahí es **más barata que la recurrente cuando $n < d$**, que es el caso habitual con representaciones tipo word-piece o BPE.

> 💡 Ese $O(n^2 \cdot d)$ es exactamente la limitación que definiría la década siguiente: Longformer, FlashAttention, Mamba... El propio paper ya apunta la *restricted self-attention* como salida.

### Entrenamiento (Sec. 5)

- **Datos:** WMT 2014 EN-DE (4.5M pares, BPE con vocabulario compartido de ~37000 tokens) y EN-FR (36M frases, word-piece de 32000). Batches de ~25000 tokens fuente y 25000 destino.
- **Hardware:** 8 GPUs NVIDIA P100. Base: 100K pasos ≈ **12 horas**. Big: 300K pasos ≈ **3.5 días**.
- **Optimizador (Eq. 3):** Adam con $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$, y el célebre **schedule con warmup**:
$$lrate = d_{\text{model}}^{-0.5}\cdot\min\!\left(step\_num^{-0.5},\ step\_num\cdot warmup\_steps^{-1.5}\right)$$
Sube linealmente durante los primeros $warmup\_steps = 4000$ pasos y luego decae como $1/\sqrt{step}$.
- **Regularización:** dropout residual $P_{drop}=0.1$ (a la salida de cada sub-layer y a la suma embeddings+PE) y **label smoothing** $\epsilon_{ls}=0.1$ — que *empeora* la perplejidad (el modelo aprende a estar más inseguro) pero *mejora* accuracy y BLEU.
- **Inferencia:** beam search con haz 4 y penalización de longitud $\alpha=0.6$; promediado de los últimos 5 checkpoints (base) o 20 (big).

## Resultados principales

- **WMT14 EN-DE (Tabla 2):** Transformer (big) **28.4 BLEU**, >2.0 por encima del mejor resultado previo *incluidos ensembles*. El modelo **base** (27.3) ya supera a todo lo publicado, con coste de entrenamiento $3.3\times10^{18}$ FLOPs frente a $\sim10^{20}$ de los competidores — **un orden de magnitud o dos más barato**.
- **WMT14 EN-FR:** **41.8 BLEU** (big), nuevo estado del arte de modelo único, con menos de 1/4 del coste del anterior. *(Nota: el texto de la Sec. 6.1 dice 41.0 mientras el abstract y la Tabla 2 dicen 41.8 — inconsistencia conocida del paper.)*
- **Ablaciones (Tabla 3)** — la parte más instructiva:
  - **(A) Número de cabezas:** una sola cabeza es **0.9 BLEU peor** que la mejor configuración, pero **demasiadas cabezas también empeora**. Hay un óptimo (8–16).
  - **(B) Reducir $d_k$ perjudica**, lo que sugiere que "determinar compatibilidad no es fácil" y que una función más sofisticada que el producto escalar podría ayudar.
  - **(C)** Modelos más grandes, mejores. **(D)** El dropout es muy útil contra el sobreajuste.
  - **(E) Embeddings posicionales aprendidos ≈ sinusoides** (25.7 vs 25.8 BLEU dev). La elección sinusoidal fue por extrapolación, no por rendimiento.
- **Generalización (Sec. 6.3, Tabla 4):** un Transformer de 4 capas en *constituency parsing* del Penn Treebank da **91.3 F1** solo con WSJ (40K frases) y **92.7** semi-supervisado — superando a todos los previos salvo el RNNG de Dyer et al., y **sin apenas tuning específico**. Prueba de que la arquitectura no es un truco de traducción.

## Puntos fuertes y limitaciones

**Fuertes:** simplicidad radical — quitar componentes (recurrencia, convolución) y mejorar resultados es el mejor tipo de resultado; el argumento de complejidad de la Tabla 1 es una justificación de diseño *a priori*, no un ajuste *post hoc*; las ablaciones son honestas y ricas (incluyendo que más cabezas empeoran y que las sinusoides no aportan sobre lo aprendido); la eficiencia es el verdadero titular (1–2 órdenes de magnitud menos FLOPs); y generaliza fuera de traducción.

**Limitaciones (unas del paper, otras solo visibles en retrospectiva):**
- **El cuello de botella $O(n^2)$** en memoria y cómputo respecto a la longitud de secuencia. El paper lo reconoce y propone atención restringida como trabajo futuro; toda una línea de investigación posterior (Longformer, FlashAttention, S4/Mamba) nace de aquí.
- **Post-norm.** El paper usa $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$, que resultó ser **inestable de entrenar sin warmup cuidadoso** — de ahí el schedule de la Eq. (3). Los modelos modernos usan **pre-norm** ($x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$), mucho más estable (Xiong et al. 2020).
- **El título exagera un poco.** La atención no es "todo lo que necesitas": las FFN por posición son ~2/3 de los parámetros y hacen trabajo esencial, y los residuales, la normalización y el schedule de learning rate son igual de necesarios para que entrene.
- **Componentes superados:** las sinusoides han cedido ante **RoPE**; la ReLU del FFN ante **GELU/SwiGLU**; Adam ante **AdamW**. Nada de esto resta al núcleo.
- **Enmarcado como traducción.** El paper no anticipa que la verdadera aplicación sería el **modelado de lenguaje a escala** (BERT, GPT). La escala máxima aquí son 213M parámetros.
- **Sin teoría de por qué funciona:** la justificación es empírica y de complejidad; la interpretabilidad se despacha con "las cabezas parecen aprender tareas distintas" y unos ejemplos en el apéndice.

## Ideas de implementación

El Transformer es probablemente **el ejercicio más rentable de toda la pista de deep learning**: implementarlo desde cero fuerza a entender álgebra lineal por lotes, enmascaramiento y normalización. Plan por piezas (estilo Lasso/Markowitz):

1. **Scaled dot-product attention (Eq. 1)** en numpy puro, sin batching: $QK^\top$, escala, softmax, $\times V$. ~10 líneas. **Validar el porqué del $\sqrt{d_k}$**: generar $q,k$ gaussianos y comprobar empíricamente que $\mathrm{Var}(q\cdot k)= d_k$, y graficar cómo se satura el softmax (y se muere el gradiente) sin la escala. Es la nota 4 del paper hecha figura.
2. **Multi-head attention** con las proyecciones $W^Q, W^K, W^V, W^O$ y el *reshape* a $h$ cabezas. Verificar que el coste en parámetros es **igual** al de una cabeza a dimensión completa (la afirmación de la Sec. 3.2.2).
3. **Máscara causal** del decoder: matriz triangular con $-\infty$, comprobando que la fila $i$ solo pone masa en columnas $\le i$. Visualizar la matriz de atención antes/después.
4. **Positional encoding sinusoidal:** implementar y **dibujar el mapa de calor** $PE(pos, i)$ (una figura preciosa). Verificar numéricamente la propiedad clave: que $PE_{pos+k}$ es combinación lineal de $PE_{pos}$ con matriz independiente de $pos$ (es una rotación 2×2 por cada par de dimensiones — conexión directa con RoPE).
5. **Bloque encoder completo:** multi-head + FFN + residual + LayerNorm, en PyTorch. Contar parámetros y **verificar el reparto ~1/3 atención, ~2/3 FFN**.
6. **Transformer mínimo entrenable** sobre una tarea de juguete (copiar/invertir secuencias, o traducción de números a texto). Reproducir el **schedule de learning rate de la Eq. (3)** y graficarlo.
7. **Ablación propia:** repetir la fila (A) de la Tabla 3 a escala de juguete — variar $h \in \{1,2,4,8\}$ a cómputo constante y comprobar que una sola cabeza pierde y que demasiadas también.
8. **Validar contra `torch.nn.MultiheadAttention`** con los mismos pesos, comprobando igualdad numérica.

## Conexiones

- **Ruta A del [ROADMAP](../ROADMAP.md):** este paper es el destino de la línea histórica RNN → atención. Leer antes **Bahdanau (2014)** —el origen real de la atención, donde aún hay RNN— hace que el salto se entienda como "quitar la RNN", que es exactamente la tesis. **Luong (2015)** aporta la atención multiplicativa que aquí se escala.
- **Ruta D (ingeniería):** casi toda esa ruta son parches a limitaciones de *este* paper — **RoPE** (sustituye las sinusoides), **FlashAttention** (ataca el $O(n^2)$ en memoria), **AdamW** (sustituye el Adam+warmup), **Layer Normalization** (el pre/post-norm), **multi-query attention** (abarata la inferencia).
- **Ruta B (tokenización):** el paper usa **BPE** (Sennrich 2015) y **word-piece** (GNMT 2016) sin discutirlos; son prerrequisito para entender la capa de entrada.
- **[Tibshirani (1996), Lasso](1996-tibshirani-lasso.md):** conexión lateral pero real — el softmax de la atención produce pesos **densos** (todos > 0), y hay toda una línea de *sparse attention* que busca lo que el $L_1$ hace en regresión: poner ceros exactos.
