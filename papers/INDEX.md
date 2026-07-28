# Índice de papers descargados

Inventario de `papers/`. Convención de nombre: `<año>-<primer-autor>-<titulo-corto>.pdf`;
ese mismo slug se reutiliza en `reviews/` e `implementations/`.

Rutas de lectura y orden sugerido en el [ROADMAP](../ROADMAP.md).
⭐ = lectura obligada · 🔧 = merece implementación desde cero.

## Estadística

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Bootstrap Methods: Another Look at the Jackknife](1979-efron-bootstrap.pdf) | Efron | 1979 | ✅ review · implementación pendiente | 2283 KB |
| [Regression Shrinkage and Selection via the Lasso](1996-tibshirani-lasso.pdf) | Tibshirani | 1996 | ✅ review + implementación (piezas 1–2) | 2020 KB |

## A · RNN→Transformer

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Long Short-Term Memory](1997-hochreiter-lstm.pdf) | Hochreiter & Schmidhuber | 1997 | El gradiente que se desvanece | 237 KB |
| [Neural Machine Translation by Jointly Learning to Align and Translate](2014-bahdanau-attention-nmt.pdf) | Bahdanau, Cho & Bengio | 2014 | **El origen de la atención** | 434 KB |
| [Learning Phrase Representations using RNN Encoder-Decoder](2014-cho-gru-encoder-decoder.pdf) | Cho et al. | 2014 | GRU + encoder-decoder | 1115 KB |
| [Sequence to Sequence Learning with Neural Networks](2014-sutskever-seq2seq.pdf) | Sutskever, Vinyals & Le | 2014 |  | 109 KB |
| [Attention Is All You Need](2017-vaswani-attention.pdf) | Vaswani et al. | 2017 | ⭐🔧 El paper clave de la pista | 2163 KB |

## B · Tokenización

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Distributed Representations of Words and Phrases](2013-mikolov-negative-sampling.pdf) | Mikolov et al. | 2013 | negative sampling | 122 KB |
| [Efficient Estimation of Word Representations in Vector Space](2013-mikolov-word2vec.pdf) | Mikolov et al. | 2013 | word2vec | 223 KB |
| [GloVe: Global Vectors for Word Representation](2014-pennington-glove.pdf) | Pennington, Socher & Manning | 2014 |  | 2557 KB |
| [Neural Machine Translation of Rare Words with Subword Units](2015-sennrich-bpe-subword.pdf) | Sennrich, Haddow & Birch | 2015 | ⭐🔧 **BPE** — implementable en ~50 líneas | 188 KB |
| [SentencePiece: A Simple and Language Independent Subword Tokenizer](2018-kudo-sentencepiece.pdf) | Kudo & Richardson | 2018 |  | 206 KB |
| [Subword Regularization](2018-kudo-subword-regularization.pdf) | Kudo | 2018 | Tokenizador unigram | 321 KB |

## C · LLMs

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [BERT: Pre-training of Deep Bidirectional Transformers](2018-devlin-bert.pdf) | Devlin et al. | 2018 | ⭐ | 756 KB |
| [Deep Contextualized Word Representations](2018-peters-elmo.pdf) | Peters et al. | 2018 | ELMo | 415 KB |
| [RoBERTa: A Robustly Optimized BERT Pretraining Approach](2019-liu-roberta.pdf) | Liu et al. | 2019 |  | 204 KB |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](2019-raffel-t5.pdf) | Raffel et al. | 2019 | T5 | 1162 KB |
| [Language Models are Few-Shot Learners](2020-brown-gpt3.pdf) | Brown et al. | 2020 | ⭐ GPT-3 | 6609 KB |
| [Scaling Laws for Neural Language Models](2020-kaplan-scaling-laws.pdf) | Kaplan et al. | 2020 |  | 2434 KB |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](2020-lewis-rag.pdf) | Lewis et al. | 2020 | RAG | 864 KB |
| [Training Compute-Optimal Large Language Models](2022-hoffmann-chinchilla.pdf) | Hoffmann et al. | 2022 | ⭐ Chinchilla | 5863 KB |
| [Training Language Models to Follow Instructions with Human Feedback](2022-ouyang-instructgpt-rlhf.pdf) | Ouyang et al. | 2022 | InstructGPT / RLHF | 1755 KB |
| [Chain-of-Thought Prompting Elicits Reasoning in LLMs](2022-wei-chain-of-thought.pdf) | Wei et al. | 2022 |  | 870 KB |

## D · Ingeniería

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](2017-shazeer-moe-sparse.pdf) | Shazeer et al. | 2017 | Mixture of experts | 531 KB |
| [Fast Transformer Decoding: One Write-Head is All You Need](2019-shazeer-multi-query-attention.pdf) | Shazeer | 2019 | Multi-query attention | 139 KB |
| [LoRA: Low-Rank Adaptation of Large Language Models](2021-hu-lora.pdf) | Hu et al. | 2021 | Álgebra lineal pura | 1571 KB |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](2021-su-rope-roformer.pdf) | Su et al. | 2021 | RoPE | 585 KB |
| [FlashAttention: Fast and Memory-Efficient Exact Attention](2022-dao-flashattention.pdf) | Dao et al. | 2022 | 🔧 Enlaza con HPC | 2569 KB |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](2023-gu-mamba.pdf) | Gu & Dao | 2023 | La alternativa post-Transformer | 1141 KB |

## E · Generativos

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Auto-Encoding Variational Bayes](2013-kingma-vae.pdf) | Kingma & Welling | 2013 | ⭐🔧 VAE — enlaza con Bayesiano | 3834 KB |
| [Generative Adversarial Nets](2014-goodfellow-gan.pdf) | Goodfellow et al. | 2014 | ⭐🔧 Minimax derivable a mano | 518 KB |
| [Unsupervised Representation Learning with DCGANs](2015-radford-dcgan.pdf) | Radford, Metz & Chintala | 2015 |  | 7282 KB |
| [Wasserstein GAN](2017-arjovsky-wasserstein-gan.pdf) | Arjovsky, Chintala & Bottou | 2017 | 🔧 Transporte óptimo | 8627 KB |
| [Denoising Diffusion Probabilistic Models](2020-ho-diffusion-ddpm.pdf) | Ho, Jain & Abbeel | 2020 | ⭐ Física estadística aplicada | 10026 KB |
| [Learning Transferable Visual Models From Natural Language Supervision](2021-radford-clip.pdf) | Radford et al. | 2021 | CLIP | 6653 KB |
| [High-Resolution Image Synthesis with Latent Diffusion Models](2021-rombach-latent-diffusion.pdf) | Rombach et al. | 2021 | Stable Diffusion | 39885 KB |

## F · Visión

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Gradient-Based Learning Applied to Document Recognition](1998-lecun-gradient-based-learning.pdf) | LeCun et al. | 1998 | LeNet | 932 KB |
| [ImageNet Classification with Deep Convolutional Neural Networks](2012-krizhevsky-alexnet.pdf) | Krizhevsky, Sutskever & Hinton | 2012 | AlexNet | 1385 KB |
| [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](2014-srivastava-dropout.pdf) | Srivastava et al. | 2014 |  | 2801 KB |
| [Deep Residual Learning for Image Recognition](2015-he-resnet.pdf) | He et al. | 2015 | ResNet | 800 KB |
| [Batch Normalization](2015-ioffe-batchnorm.pdf) | Ioffe & Szegedy | 2015 |  | 169 KB |
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](2015-ronneberger-unet.pdf) | Ronneberger et al. | 2015 |  | 1610 KB |
| [An Image is Worth 16x16 Words](2020-dosovitskiy-vit.pdf) | Dosovitskiy et al. | 2020 | ViT | 3656 KB |

## G · RL

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Playing Atari with Deep Reinforcement Learning](2013-mnih-dqn-atari.pdf) | Mnih et al. | 2013 | DQN | 472 KB |
| [Proximal Policy Optimization Algorithms](2017-schulman-ppo.pdf) | Schulman et al. | 2017 | Base del RLHF | 2855 KB |

## Máster · Matemáticas

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Stable Signal Recovery from Incomplete and Inaccurate Measurements](2005-candes-stable-signal-recovery.pdf) | Candès, Romberg & Tao | 2005 | Minimización L1 — conecta con el Lasso | 462 KB |
| [Finding Structure with Randomness](2011-halko-randomized-svd.pdf) | Halko, Martinsson & Tropp | 2011 | SVD aleatorizada | 1257 KB |

## Máster · Fundamentos Big Data

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [MapReduce: Simplified Data Processing on Large Clusters](2004-dean-mapreduce.pdf) | Dean & Ghemawat | 2004 |  | 186 KB |
| [Resilient Distributed Datasets](2012-zaharia-spark-rdd.pdf) | Zaharia et al. | 2012 | Spark | 865 KB |

## Máster · Back-end

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Bigtable: A Distributed Storage System for Structured Data](2006-chang-bigtable.pdf) | Chang et al. | 2006 |  | 216 KB |
| [Cassandra: A Decentralized Structured Storage System](2010-lakshman-cassandra.pdf) | Lakshman & Malik | 2010 |  | 130 KB |

## Máster · CDN

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Consistent Hashing and Random Trees](1997-karger-consistent-hashing.pdf) | Karger et al. | 1997 |  | 201 KB |

## Máster · Modelos de predicción

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Greedy Function Approximation: A Gradient Boosting Machine](2001-friedman-gradient-boosting.pdf) | Friedman | 2001 |  | 949 KB |
| [XGBoost: A Scalable Tree Boosting System](2016-chen-xgboost.pdf) | Chen & Guestrin | 2016 |  | 922 KB |

## Máster · Aprendizaje estadístico

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Random Forests](2001-breiman-random-forests.pdf) | Breiman | 2001 |  | 120 KB |
| [Statistical Modeling: The Two Cultures](2001-breiman-two-cultures.pdf) | Breiman | 2001 | Solo review, sin código | 300 KB |

## Máster · Optimización

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Distributed Optimization via ADMM](2011-boyd-admm.pdf) | Boyd et al. | 2011 |  | 775 KB |
| [Adam: A Method for Stochastic Optimization](2014-kingma-adam.pdf) | Kingma & Ba | 2014 |  | 570 KB |
| [Optimization Methods for Large-Scale Machine Learning](2018-bottou-optimization-ml.pdf) | Bottou, Curtis & Nocedal | 2018 |  | 1908 KB |

## Máster · Bayesiano

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [Latent Dirichlet Allocation](2003-blei-lda.pdf) | Blei, Ng & Jordan | 2003 |  | 408 KB |
| [The No-U-Turn Sampler](2014-hoffman-nuts.pdf) | Hoffman & Gelman | 2014 |  | 1003 KB |
| [Variational Inference: A Review for Statisticians](2017-blei-variational-inference.pdf) | Blei, Kucukelbir & McAuliffe | 2017 |  | 1780 KB |

## Máster · Series temporales

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [DeepAR: Probabilistic Forecasting with Autoregressive RNNs](2020-salinas-deepar.pdf) | Salinas et al. | 2020 |  | 555 KB |

## Quant

| Paper | Autores | Año | Nota | Tamaño |
|---|---|---|---|---|
| [A New Approach to Linear Filtering and Prediction Problems](1960-kalman-filtering.pdf) | Kalman | 1960 |  | 173 KB |
| [High-frequency Trading in a Limit Order Book](2008-avellaneda-market-making.pdf) | Avellaneda & Stoikov | 2008 | Market making | 423 KB |

## Pendientes de descargar (no accesibles automáticamente)

Tres se resistieron a la descarga automática (paywall o bloqueo del servidor). Guardar en `papers/` con el slug indicado:

- `1952-markowitz-portfolio-selection` — *Portfolio Selection* (Markowitz, 1952). **Es el siguiente paper quant**, así que conviene tenerlo. Vía [Wiley (DOI)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x) o [JSTOR 2975974](https://www.jstor.org/stable/2975974) — con el acceso institucional de la UC3M/Oviedo debería abrirse.
- `2008-candes-compressive-sampling` — *An Introduction to Compressive Sampling* (Candès & Wakin, 2008), IEEE Signal Processing Magazine. Acceso vía IEEE Xplore con credenciales de la universidad. *(Nota: sí está descargado `2005-candes-stable-signal-recovery`, del mismo grupo y muy relacionado.)*
- `2017-taylor-prophet` — *Forecasting at Scale* (Taylor & Letham, 2017). Preprint en [PeerJ](https://peerj.com/preprints/3190/) — el servidor rechaza la descarga automática, pero desde el navegador se baja sin problema.

---

**Total en `papers/`: 65 PDFs.**
