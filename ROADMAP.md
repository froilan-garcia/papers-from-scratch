# Hoja de ruta: papers por asignatura del máster

Máster en Métodos Analíticos para Datos Masivos: Big Data (UC3M, inicio septiembre 2026).
60 ECTS · 1 año · en inglés · [plan de estudios](https://www.uc3m.es/master/big-data)

Cada asignatura lleva asociados papers candidatos para el flujo `/paper` (review + implementación en Python). Marcar con `[x]` los ya revisados y enlazar su review.

## Cuatrimestre 1 · Semicuatrimestre I (sept–oct)

### Matemáticas para el análisis de datos
- [ ] Halko, Martinsson & Tropp (2011) — *Finding Structure with Randomness* (SVD aleatorizada) — arXiv:0909.4061
- [ ] Candès & Wakin (2008) — *An Introduction to Compressive Sampling*

### Estadística para el análisis de datos
- [ ] Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso* — [review](reviews/1996-tibshirani-lasso.md) ✔ · [implementación](implementations/1996-tibshirani-lasso/) en curso (pieza 1/5: solver ✔)
- [x] Efron (1979) — *Bootstrap Methods: Another Look at the Jackknife* — [review](reviews/1979-efron-bootstrap.md) ✔ · implementación pendiente
- [ ] Hoerl & Kennard (1970) — *Ridge Regression: Biased Estimation for Nonorthogonal Problems* (solo review, sin código — cierra el díptico de regularización con el Lasso; ridge = *weight decay* en NNs)

### Fundamentos tecnológicos en el mundo Big Data
- [ ] Dean & Ghemawat (2004) — *MapReduce: Simplified Data Processing on Large Clusters*
- [ ] Zaharia et al. (2012) — *Resilient Distributed Datasets* (Spark)

### Computación de altas prestaciones para Big Data
- [ ] Blelloch (1990) — *Prefix Sums and Their Applications* (paralelismo básico)

### Back-end para análisis de Big Data
- [ ] Chang et al. (2006) — *Bigtable: A Distributed Storage System*
- [ ] Lakshman & Malik (2010) — *Cassandra: A Decentralized Structured Storage System*

## Cuatrimestre 1 · Semicuatrimestre II (nov–dic)

### Distribución de contenidos en Internet
- [ ] Karger et al. (1997) — *Consistent Hashing and Random Trees* (base de los CDN)

### Modelos de predicción
- [ ] Friedman (2001) — *Greedy Function Approximation: A Gradient Boosting Machine*
- [ ] Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System* — arXiv:1603.02754

### Aprendizaje estadístico
- [ ] Breiman (2001) — *Random Forests*
- [ ] Breiman (2001) — *Statistical Modeling: The Two Cultures* (solo review, sin código)

### Optimización para grandes volúmenes de datos
- [ ] Bottou, Curtis & Nocedal (2018) — *Optimization Methods for Large-Scale Machine Learning* — arXiv:1606.04838
- [ ] Boyd et al. (2011) — *Distributed Optimization via ADMM*
- [ ] Kingma & Ba (2014) — *Adam: A Method for Stochastic Optimization* — arXiv:1412.6980

### Inteligencia para Big Data: métodos y tecnologías
- [ ] Mikolov et al. (2013) — *Efficient Estimation of Word Representations* (word2vec) — arXiv:1301.3781

## Cuatrimestre 2 · Semicuatrimestre III (feb–mar)

### Aprendizaje Bayesiano
- [ ] Blei, Ng & Jordan (2003) — *Latent Dirichlet Allocation*
- [ ] Blei, Kucukelbir & McAuliffe (2017) — *Variational Inference: A Review for Statisticians* — arXiv:1601.00670
- [ ] Hoffman & Gelman (2014) — *The No-U-Turn Sampler* — arXiv:1111.4246

### Análisis de series temporales y predicción
- [ ] Taylor & Letham (2017) — *Forecasting at Scale* (Prophet)
- [ ] Salinas et al. (2020) — *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* — arXiv:1704.04110

### Aprendizaje automático
- [ ] Vaswani et al. (2017) — *Attention Is All You Need* — arXiv:1706.03762
- [ ] Srivastava et al. (2014) — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
- [ ] He et al. (2015) — *Deep Residual Learning for Image Recognition* — arXiv:1512.03385

### Aplicaciones del análisis de Big Data a los negocios
- [ ] Koren, Bell & Volinsky (2009) — *Matrix Factorization Techniques for Recommender Systems* (premio Netflix)

### Análisis y explotación de datos de la Web
- [ ] Brin & Page (1998) — *The Anatomy of a Large-Scale Hypertextual Web Search Engine* (PageRank)
- [ ] Kleinberg (1999) — *Authoritative Sources in a Hyperlinked Environment* (HITS)

## Cuatrimestre 2 · Semicuatrimestre IV (abr–may) — optativas

### Análisis de redes y visualización de datos
- [ ] Blondel et al. (2008) — *Fast Unfolding of Communities in Large Networks* (Louvain) — arXiv:0803.0476
- [ ] Grover & Leskovec (2016) — *node2vec* — arXiv:1607.00653
- [ ] Kipf & Welling (2016) — *Semi-Supervised Classification with Graph Convolutional Networks* (GCN) — arXiv:1609.02907
- [ ] Veličković et al. (2017) — *Graph Attention Networks* — arXiv:1710.10903 — atención aplicada a grafos
- [ ] Chen et al. (2018) — *Neural Ordinary Differential Equations* — arXiv:1806.07366 — EDOs y redes profundas; terreno directo de tu grado

### Análisis de datos para la sociedad inteligente
- [ ] (pendiente de elegir optativa)

### Seguridad de la información
- [ ] (pendiente de elegir optativa)

## Pista extra: deep learning, LLMs y modelos generativos

Cola de lectura **exploratoria** (sin compromiso de implementar todo) para después del Lasso y el bootstrap. Los PDFs están descargados en `papers/` salvo donde se indique — ver [papers/INDEX.md](papers/INDEX.md) para el inventario completo.

Marcados con ⭐ los que son lectura obligada; con 🔧 los que merecen implementación desde cero.

### Ruta A — De las RNN al Transformer (la línea histórica)

El orden importa: cada paper responde a una limitación del anterior. Es la mejor forma de entender *por qué* existe el Transformer.

- [ ] Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory* — el problema del gradiente que se desvanece
- [ ] Pascanu, Mikolov & Bengio (2012) — *On the Difficulty of Training RNNs* — arXiv:1211.5063 — la teoría del gradiente que explota/desvanece
- [ ] Cho et al. (2014) — *Learning Phrase Representations* (GRU + encoder-decoder) — arXiv:1406.1078
- [ ] Chung et al. (2014) — *Empirical Evaluation of Gated RNNs* — arXiv:1412.3555 — LSTM vs GRU cara a cara
- [ ] Sutskever, Vinyals & Le (2014) — *Sequence to Sequence Learning with Neural Networks* — arXiv:1409.3215
- [ ] ⭐ Bahdanau, Cho & Bengio (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate* — arXiv:1409.0473 — **el origen real de la atención**
- [ ] Luong, Pham & Manning (2015) — *Effective Approaches to Attention-based NMT* — arXiv:1508.04025 — atención multiplicativa
- [ ] Xu et al. (2015) — *Show, Attend and Tell* — arXiv:1502.03044 — atención visual
- [ ] Graves, Wayne & Danihelka (2014) — *Neural Turing Machines* — arXiv:1410.5401 — memoria externa direccionable
- [ ] van den Oord et al. (2016) — *WaveNet* — arXiv:1609.03499 — convoluciones dilatadas causales
- [ ] ⭐🔧 Vaswani et al. (2017) — *Attention Is All You Need* — arXiv:1706.03762 — implementar self-attention y multi-head desde cero es el ejercicio más rentable de toda la pista

### Ruta B — Tokenización y embeddings

- [ ] Mikolov et al. (2013) — *Efficient Estimation of Word Representations* (word2vec) — arXiv:1301.3781
- [ ] Mikolov et al. (2013) — *Distributed Representations of Words and Phrases* (negative sampling) — arXiv:1310.4546
- [ ] Pennington, Socher & Manning (2014) — *GloVe: Global Vectors for Word Representation*
- [ ] ⭐🔧 Sennrich, Haddow & Birch (2015) — *Neural Machine Translation of Rare Words with Subword Units* (**BPE**) — arXiv:1508.07909 — el algoritmo de tokenización que usan GPT y casi todos; implementable en ~50 líneas
- [ ] Wu et al. (2016) — *Google's Neural Machine Translation System* — arXiv:1609.08144 — **WordPiece**, el tokenizador de BERT
- [ ] Kudo (2018) — *Subword Regularization* (tokenizador unigram) — arXiv:1804.10959
- [ ] Kudo & Richardson (2018) — *SentencePiece* — arXiv:1808.06226
- [ ] Yu et al. (2023) — *MEGABYTE: Predicting Million-byte Sequences* — arXiv:2305.07185 — modelado a nivel de byte
- [ ] Pagnoni et al. (2024) — *Byte Latent Transformer* — arXiv:2412.09871 — **sin tokenizador**; lo más actual del área

### Ruta C — LLMs modernos

- [ ] Peters et al. (2018) — *Deep Contextualized Word Representations* (ELMo) — arXiv:1802.05365
- [ ] ⭐ Devlin et al. (2018) — *BERT* — arXiv:1810.04805
- [ ] Raffel et al. (2019) — *Exploring the Limits of Transfer Learning* (T5) — arXiv:1910.10683
- [ ] Liu et al. (2019) — *RoBERTa* — arXiv:1907.11692
- [ ] ⭐ Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3) — arXiv:2005.14165
- [ ] Kaplan et al. (2020) — *Scaling Laws for Neural Language Models* — arXiv:2001.08361
- [ ] ⭐ Hoffmann et al. (2022) — *Training Compute-Optimal LLMs* (Chinchilla) — arXiv:2203.15556
- [ ] Ouyang et al. (2022) — *Training LMs to Follow Instructions with Human Feedback* (InstructGPT/RLHF) — arXiv:2203.02155
- [ ] Wei et al. (2022) — *Chain-of-Thought Prompting* — arXiv:2201.11903
- [ ] Lewis et al. (2020) — *Retrieval-Augmented Generation* (RAG) — arXiv:2005.11401
- [ ] Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2) — no está en arXiv; PDF en [cdn.openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] Dai et al. (2019) — *Transformer-XL* — arXiv:1901.02860
- [ ] Sanh et al. (2019) — *DistilBERT* — arXiv:1910.01108 — destilación de modelos
- [ ] Clark et al. (2020) — *ELECTRA* — arXiv:2003.10555
- [ ] Beltagy et al. (2020) — *Longformer* — arXiv:2004.05150 — atención dispersa para documentos largos
- [ ] Wei et al. (2022) — *Emergent Abilities of Large Language Models* — arXiv:2206.07682
- [ ] Bai et al. (2022) — *Constitutional AI* — arXiv:2212.08073 — alineamiento sin etiquetado humano masivo
- [ ] Touvron et al. (2023) — *LLaMA* — arXiv:2302.13971
- [ ] Rafailov et al. (2023) — *Direct Preference Optimization* — arXiv:2305.18290 — RLHF sin RL
- [ ] Radford et al. (2022) — *Whisper* — arXiv:2212.04356 — reconocimiento de voz a escala

### Ruta D — Ingeniería del Transformer (por qué los LLMs actuales son viables)

- [ ] Su et al. (2021) — *RoFormer* (**RoPE**, embeddings posicionales rotatorios) — arXiv:2104.09864
- [ ] Shazeer (2019) — *Fast Transformer Decoding* (multi-query attention) — arXiv:1911.02150
- [ ] 🔧 Dao et al. (2022) — *FlashAttention* — arXiv:2205.14135 — conecta con la asignatura de computación de altas prestaciones
- [ ] Shazeer et al. (2017) — *Outrageously Large Neural Networks* (mixture of experts) — arXiv:1701.06538
- [ ] Fedus, Zoph & Shazeer (2021) — *Switch Transformers* — arXiv:2101.03961 — MoE llevado al billón de parámetros
- [ ] Hu et al. (2021) — *LoRA: Low-Rank Adaptation* — arXiv:2106.09685 — álgebra lineal pura, muy tu perfil
- [ ] Gu, Goel & Ré (2021) — *Structured State Spaces* (S4) — arXiv:2111.00396 — el precursor de Mamba
- [ ] Gu & Dao (2023) — *Mamba: Linear-Time Sequence Modeling* — arXiv:2312.00752 — la alternativa post-Transformer
- [ ] Ba, Kiros & Hinton (2016) — *Layer Normalization* — arXiv:1607.06450
- [ ] Loshchilov & Hutter (2017) — *Decoupled Weight Decay* (AdamW) — arXiv:1711.05101 — el *weight decay* bien hecho (ridge, no lasso)
- [ ] Hendrycks & Gimpel (2016) — *Gaussian Error Linear Units* (GELU) — arXiv:1606.08415

### Ruta E — Modelos generativos (GAN, VAE, difusión)

**GAN (redes adversarias):**
- [ ] ⭐🔧 Goodfellow et al. (2014) — *Generative Adversarial Nets* — arXiv:1406.2661 — teoría de juegos + estadística; el equilibrio minimax se deriva a mano
- [ ] Mirza & Osindero (2014) — *Conditional GAN* — arXiv:1411.1784
- [ ] Radford, Metz & Chintala (2015) — *DCGAN* — arXiv:1511.06434
- [ ] Salimans et al. (2016) — *Improved Techniques for Training GANs* — arXiv:1606.03498 — por qué las GAN no entrenan
- [ ] 🔧 Arjovsky, Chintala & Bottou (2017) — *Wasserstein GAN* — arXiv:1701.07875 — transporte óptimo; matemáticamente el más bonito
- [ ] Isola et al. (2016) — *pix2pix* — arXiv:1611.07004 · Zhu et al. (2017) — *CycleGAN* — arXiv:1703.10593
- [ ] Karras et al. (2017) — *Progressive GAN* — arXiv:1710.10196 · Karras et al. (2018) — *StyleGAN* — arXiv:1812.04948
- [ ] Brock et al. (2018) — *BigGAN* — arXiv:1809.11096

**Variacionales y flujos:**
- [ ] ⭐🔧 Kingma & Welling (2013) — *Auto-Encoding Variational Bayes* (VAE) — arXiv:1312.6114 — enlaza directamente con Aprendizaje Bayesiano (ELBO, reparametrización)
- [ ] Rezende & Mohamed (2015) — *Variational Inference with Normalizing Flows* — arXiv:1505.05770
- [ ] van den Oord et al. (2017) — *VQ-VAE* — arXiv:1711.00937 — latentes discretos
- [ ] Kingma & Dhariwal (2018) — *Glow* — arXiv:1807.03039

**Difusión:**
- [ ] ⭐ Ho, Jain & Abbeel (2020) — *Denoising Diffusion Probabilistic Models* — arXiv:2006.11239 — procesos de difusión: física estadística aplicada
- [ ] 🔧 Song et al. (2020) — *Score-Based Generative Modeling through SDEs* — arXiv:2011.13456 — ecuaciones diferenciales estocásticas, puro perfil de físico
- [ ] Song, Meng & Ermon (2020) — *DDIM* — arXiv:2010.02502 — muestreo rápido
- [ ] Ho & Salimans (2022) — *Classifier-Free Guidance* — arXiv:2207.12598
- [ ] Rombach et al. (2021) — *Latent Diffusion Models* (Stable Diffusion) — arXiv:2112.10752
- [ ] Radford et al. (2021) — *CLIP* — arXiv:2103.00020

**Ejemplos adversarios (robustez, el otro sentido de "adversario"):**
- [ ] Szegedy et al. (2013) — *Intriguing Properties of Neural Networks* — arXiv:1312.6199 — el descubrimiento
- [ ] Goodfellow, Shlens & Szegedy (2014) — *Explaining and Harnessing Adversarial Examples* — arXiv:1412.6572 — FGSM

### Ruta F — Fundamentos de visión y entrenamiento

- [ ] LeCun et al. (1998) — *Gradient-Based Learning Applied to Document Recognition* (LeNet)
- [ ] Krizhevsky, Sutskever & Hinton (2012) — *ImageNet Classification with Deep CNNs* (AlexNet)
- [ ] He et al. (2015) — *Deep Residual Learning* (ResNet) — arXiv:1512.03385
- [ ] Ioffe & Szegedy (2015) — *Batch Normalization* — arXiv:1502.03167
- [ ] Srivastava et al. (2014) — *Dropout*
- [ ] Ronneberger et al. (2015) — *U-Net* — arXiv:1505.04597
- [ ] Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words* (ViT) — arXiv:2010.11929

### Ruta G — Aprendizaje por refuerzo (opcional)

- [ ] Mnih et al. (2013) — *Playing Atari with Deep Reinforcement Learning* (DQN) — arXiv:1312.5602
- [ ] Schulman et al. (2017) — *Proximal Policy Optimization* — arXiv:1707.06347 — la base del RLHF de los LLMs

## Preparación quant (julio–octubre 2026) — prácticas verano 2027

Objetivo: solicitar prácticas de quant research / quant trading para el verano de 2027. Los procesos son **rolling** (~70% de plazas firmadas para mediados de octubre).

**Estrategia (revisada 23 jul 2026):** no correr con Citadel en agosto sin estar afilado. Suspender el OA (online assessment) tiene *cooldown* de 6–12 meses — ese es el coste real, no ninguna "blacklist" (aplicar y ser rechazado no quema para el futuro). Por eso: **agosto = 100% preparación + GitHub; 1ª semana de septiembre = aplicar en bloque** ya preparado (el OA llega ~5–7 días después). Rolling sigue abierto en septiembre; aplicar pronto, no en octubre. Plan detallado en [quant/plan-agosto-septiembre.md](quant/plan-agosto-septiembre.md).

### Calendario de solicitudes — aplicar en bloque la 1ª semana de septiembre

| Firma | Oficinas (Europa) | Nota | Enlace |
|---|---|---|---|
| Citadel / Citadel Securities | Londres | Abierta desde julio — se aplica algo tarde, pero se aplica | [citadel.com/careers](https://www.citadel.com/careers/open-opportunities/internships/) |
| Jane Street | Londres | Abre agosto 2026 | [janestreet.com](https://www.janestreet.com/join-jane-street/internships/) |
| Optiver | Ámsterdam, Londres | Abre agosto 2026 | [optiver.com](https://www.optiver.com/join-us/) |
| IMC Trading | Ámsterdam | ~agosto | [imc.com](https://careers.imc.com/) |
| G-Research | Londres | ~sept | [gresearch.com](https://www.gresearch.com/vacancies/) |
| Qube Research & Technologies | Londres | ~sept | [qube-rt.com](https://www.qube-rt.com/careers/) |
| Squarepoint Capital | Londres | ~sept | [squarepoint-capital.com](https://www.squarepoint-capital.com/open-positions) |
| Flow Traders | Ámsterdam | ~sept | [flowtraders.com](https://www.flowtraders.com/careers) |

### Plan B — máster especializado (ETH/UvA), decisión en noviembre

**No es necesario** para ser competitivo: el perfil (doble grado Mates+Física + máster Big Data + portfolio de papers) ya encaja en quant research/trading. Es una **palanca opcional** con tres usos: (a) plan B si el ciclo de septiembre no cuaja, (b) arreglar el timing de prácticas (el máster de Big Data es de 1 año → verano de prácticas justo al graduarse; el MFE da verano intermedio), (c) enchufarse a la pipeline de Ámsterdam.

- **ETH/UZH — MSc Quantitative Finance:** 3 semestres, ~€2.700 todo el máster. Fall 2026 cerrado; **solicitudes para otoño 2027 abren en noviembre 2026** (cae en tu 1er semestre de la UC3M → opción de bajo riesgo). [math.ethz.ch](https://math.ethz.ch/studies/master-programmes/master-quantitative-finance.html)
- **UvA (Ámsterdam):** MSc Quantitative Finance o MSc Actuarial Science & Mathematical Finance (track QRM). ~1 año, tasa UE ~€2.600. Optiver/IMC/Flow reclutan en el campus. [abs.uva.nl](https://abs.uva.nl/content/masters/finance-quantitative-finance/study-programme/study-programme.html)

Criterio: aplicar a quant este ciclo con el perfil actual; dejar la solicitud de ETH (nov 2026) como opción viva; **decidir en firme a principios de 2027**, ya con resultados en la mano. No matricularse en un 2º máster "por si acaso" antes de intentarlo.

### Preparación de entrevistas (en paralelo, ritmo diario corto)

- [ ] Probabilidad y brainteasers: Xinfeng Zhou, *A Practical Guide to Quantitative Finance Interviews* (el "libro verde") — 1 sesión corta al día.
- [ ] Cálculo mental rápido: entrenar con juegos tipo Optiver 80-in-8 / arithmetic-game.
- [ ] Algoritmia en Python: LeetCode nivel medium, 3-4 problemas/semana.
- [ ] CV en inglés de una página, orientado a quant (perfil matemático + este repo como portfolio).

### Papers quant para el flujo `/paper` (orden sugerido)

- [ ] Markowitz (1952) — *Portfolio Selection* — el punto de partida: optimización media-varianza, álgebra lineal pura.
- [ ] Black & Scholes (1973) — *The Pricing of Options and Corporate Liabilities* — conexión directa con física (ecuación del calor).
- [ ] Kalman (1960) — *A New Approach to Linear Filtering and Prediction Problems* — filtrado secuencial, base de muchas señales quant.
- [ ] Gatev, Goetzmann & Rouwenhorst (2006) — *Pairs Trading: Performance of a Relative-Value Arbitrage Rule* — primera estrategia completa con backtest.
- [ ] Moskowitz, Ooi & Pedersen (2012) — *Time Series Momentum* — factor investing, series temporales financieras.
- [ ] Avellaneda & Stoikov (2008) — *High-frequency Trading in a Limit Order Book* — market making, control estocástico.
- [ ] López de Prado (2016) — *Building Diversified Portfolios that Outperform Out-of-Sample* (Hierarchical Risk Parity) — ML aplicado a carteras, enlaza con el máster.

Sinergias con el resto del roadmap: el Lasso ya revisado, Friedman (boosting) y Bottou (optimización estocástica) son directamente relevantes para quant research — citarlos en entrevistas.

## Preparación previa (julio–agosto 2026, antes de empezar)

Orden sugerido para llegar a septiembre con base sólida — de menos a más exigente:

1. Tibshirani (1996), Lasso — estadística + optimización, implementación asequible (coordinate descent).
2. Brin & Page (1998), PageRank — álgebra lineal pura (autovectores), conecta con tu perfil matemático.
3. Friedman (2001), Gradient Boosting — el caballo de batalla del análisis de datos tabular.
4. Kingma & Ba (2014), Adam — puerta de entrada a la optimización estocástica.
5. Blei et al. (2003), LDA — introducción perfecta al mundo bayesiano del segundo cuatrimestre.
