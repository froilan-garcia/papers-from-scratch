# Index of downloaded papers

Inventory of `papers/`. Naming convention: `<year>-<first-author>-<short-title>.pdf`;
that same slug is reused in `reviews/` and `implementations/`.

> ⚠️ **This is a shelf, not a to-do list.** These are papers downloaded to have at
> hand; they do not have to be read.

Reading tracks and suggested order in the [ROADMAP](../ROADMAP.md).
⭐ = the essential one of the block · 🔧 = worth implementing from scratch.

## Statistics

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Bootstrap Methods: Another Look at the Jackknife](1979-efron-bootstrap.pdf) | Efron | 1979 | ✅ review · implementation pending | 2283 KB |
| [Regression Shrinkage and Selection via the Lasso](1996-tibshirani-lasso.pdf) | Tibshirani | 1996 | ✅ review + implementation ✔ | 2020 KB |

## A · RNN→Transformer

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Long Short-Term Memory](1997-hochreiter-lstm.pdf) | Hochreiter & Schmidhuber | 1997 | The vanishing gradient | 237 KB |
| [On the Difficulty of Training Recurrent Neural Networks](2012-pascanu-difficulty-training-rnn.pdf) | Pascanu, Mikolov & Bengio | 2012 | The theory of exploding/vanishing gradients | 610 KB |
| [Neural Machine Translation by Jointly Learning to Align and Translate](2014-bahdanau-attention-nmt.pdf) | Bahdanau, Cho & Bengio | 2014 | **The origin of attention** | 434 KB |
| [Learning Phrase Representations using RNN Encoder-Decoder](2014-cho-gru-encoder-decoder.pdf) | Cho et al. | 2014 | GRU + encoder-decoder | 1115 KB |
| [Empirical Evaluation of Gated Recurrent Neural Networks](2014-chung-gated-rnn-evaluation.pdf) | Chung et al. | 2014 | LSTM vs GRU, head to head | 667 KB |
| [Neural Turing Machines](2014-graves-neural-turing-machines.pdf) | Graves, Wayne & Danihelka | 2014 | Addressable external memory | 1325 KB |
| [Sequence to Sequence Learning with Neural Networks](2014-sutskever-seq2seq.pdf) | Sutskever, Vinyals & Le | 2014 |  | 109 KB |
| [Effective Approaches to Attention-based NMT](2015-luong-attention.pdf) | Luong, Pham & Manning | 2015 | Multiplicative attention | 243 KB |
| [Show, Attend and Tell](2015-xu-show-attend-tell.pdf) | Xu et al. | 2015 | Visual attention | 9358 KB |
| [WaveNet: A Generative Model for Raw Audio](2016-oord-wavenet.pdf) | van den Oord et al. | 2016 | Causal dilated convolutions | 2786 KB |
| [Attention Is All You Need](2017-vaswani-attention.pdf) | Vaswani et al. | 2017 | ⭐🔧 ✅ review · implementation in progress | 2163 KB |

## B · Tokenisation

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Distributed Representations of Words and Phrases](2013-mikolov-negative-sampling.pdf) | Mikolov et al. | 2013 | negative sampling | 122 KB |
| [Efficient Estimation of Word Representations in Vector Space](2013-mikolov-word2vec.pdf) | Mikolov et al. | 2013 | word2vec | 223 KB |
| [GloVe: Global Vectors for Word Representation](2014-pennington-glove.pdf) | Pennington, Socher & Manning | 2014 |  | 2557 KB |
| [Neural Machine Translation of Rare Words with Subword Units](2015-sennrich-bpe-subword.pdf) | Sennrich, Haddow & Birch | 2015 | ⭐🔧 **BPE** — implementable in ~50 lines | 188 KB |
| [Google's Neural Machine Translation System](2016-wu-gnmt-wordpiece.pdf) | Wu et al. | 2016 | **WordPiece** (BERT's) | 1648 KB |
| [SentencePiece: A Simple and Language Independent Subword Tokenizer](2018-kudo-sentencepiece.pdf) | Kudo & Richardson | 2018 |  | 206 KB |
| [Subword Regularization](2018-kudo-subword-regularization.pdf) | Kudo | 2018 | Unigram tokeniser | 321 KB |
| [MEGABYTE: Predicting Million-byte Sequences](2023-yu-megabyte.pdf) | Yu et al. | 2023 | Byte-level modelling | 895 KB |
| [Byte Latent Transformer: Patches Scale Better Than Tokens](2024-pagnoni-byte-latent-transformer.pdf) | Pagnoni et al. | 2024 | No tokeniser — the most current | 2332 KB |

## C · LLMs

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [BERT: Pre-training of Deep Bidirectional Transformers](2018-devlin-bert.pdf) | Devlin et al. | 2018 | ⭐ | 756 KB |
| [Deep Contextualized Word Representations](2018-peters-elmo.pdf) | Peters et al. | 2018 | ELMo | 415 KB |
| [Transformer-XL: Attentive LMs Beyond a Fixed-Length Context](2019-dai-transformer-xl.pdf) | Dai et al. | 2019 |  | 4463 KB |
| [RoBERTa: A Robustly Optimized BERT Pretraining Approach](2019-liu-roberta.pdf) | Liu et al. | 2019 |  | 204 KB |
| [Language Models are Unsupervised Multitask Learners](2019-radford-gpt2.pdf) | Radford et al. | 2019 | GPT-2 | 569 KB |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](2019-raffel-t5.pdf) | Raffel et al. | 2019 | T5 | 1162 KB |
| [DistilBERT, a Distilled Version of BERT](2019-sanh-distilbert.pdf) | Sanh et al. | 2019 | Distillation | 425 KB |
| [Longformer: The Long-Document Transformer](2020-beltagy-longformer.pdf) | Beltagy, Peters & Cohan | 2020 | Sparse attention | 526 KB |
| [Language Models are Few-Shot Learners](2020-brown-gpt3.pdf) | Brown et al. | 2020 | ⭐ GPT-3 | 6609 KB |
| [ELECTRA: Pre-training Text Encoders as Discriminators](2020-clark-electra.pdf) | Clark et al. | 2020 |  | 486 KB |
| [Scaling Laws for Neural Language Models](2020-kaplan-scaling-laws.pdf) | Kaplan et al. | 2020 |  | 2434 KB |
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](2020-lewis-rag.pdf) | Lewis et al. | 2020 | RAG | 864 KB |
| [Constitutional AI: Harmlessness from AI Feedback](2022-bai-constitutional-ai.pdf) | Bai et al. | 2022 | Alignment (Anthropic) | 2039 KB |
| [Training Compute-Optimal Large Language Models](2022-hoffmann-chinchilla.pdf) | Hoffmann et al. | 2022 | ⭐ Chinchilla | 5863 KB |
| [Training Language Models to Follow Instructions with Human Feedback](2022-ouyang-instructgpt-rlhf.pdf) | Ouyang et al. | 2022 | InstructGPT / RLHF | 1755 KB |
| [Robust Speech Recognition via Large-Scale Weak Supervision](2022-radford-whisper.pdf) | Radford et al. | 2022 | Whisper | 985 KB |
| [Chain-of-Thought Prompting Elicits Reasoning in LLMs](2022-wei-chain-of-thought.pdf) | Wei et al. | 2022 |  | 870 KB |
| [Emergent Abilities of Large Language Models](2022-wei-emergent-abilities.pdf) | Wei et al. | 2022 |  | 823 KB |
| [Direct Preference Optimization](2023-rafailov-dpo.pdf) | Rafailov et al. | 2023 | DPO — RLHF without RL | 1268 KB |
| [LLaMA: Open and Efficient Foundation Language Models](2023-touvron-llama.pdf) | Touvron et al. | 2023 |  | 709 KB |

## D · Engineering

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Layer Normalization](2016-ba-layer-normalization.pdf) | Ba, Kiros & Hinton | 2016 | The Transformer's normalisation | 598 KB |
| [Gaussian Error Linear Units (GELUs)](2016-hendrycks-gelu.pdf) | Hendrycks & Gimpel | 2016 | The activation in BERT/GPT | 2702 KB |
| [Decoupled Weight Decay Regularization](2017-loshchilov-adamw.pdf) | Loshchilov & Hutter | 2017 | AdamW — weight decay done properly | 5921 KB |
| [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](2017-shazeer-moe-sparse.pdf) | Shazeer et al. | 2017 | Mixture of experts | 531 KB |
| [Fast Transformer Decoding: One Write-Head is All You Need](2019-shazeer-multi-query-attention.pdf) | Shazeer | 2019 | Multi-query attention | 139 KB |
| [Switch Transformers: Scaling to Trillion Parameter Models](2021-fedus-switch-transformer.pdf) | Fedus, Zoph & Shazeer | 2021 | MoE at scale | 1273 KB |
| [Efficiently Modeling Long Sequences with Structured State Spaces](2021-gu-s4-state-spaces.pdf) | Gu, Goel & Ré | 2021 | S4 — the precursor of Mamba | 3306 KB |
| [LoRA: Low-Rank Adaptation of Large Language Models](2021-hu-lora.pdf) | Hu et al. | 2021 | Pure linear algebra | 1571 KB |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](2021-su-rope-roformer.pdf) | Su et al. | 2021 | RoPE | 585 KB |
| [FlashAttention: Fast and Memory-Efficient Exact Attention](2022-dao-flashattention.pdf) | Dao et al. | 2022 | 🔧 Links with HPC | 2569 KB |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](2023-gu-mamba.pdf) | Gu & Dao | 2023 | The post-Transformer alternative | 1141 KB |

## E · Generative models

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Auto-Encoding Variational Bayes](2013-kingma-vae.pdf) | Kingma & Welling | 2013 | ⭐🔧 VAE — links with Bayesian learning | 3834 KB |
| [Intriguing Properties of Neural Networks](2013-szegedy-intriguing-properties.pdf) | Szegedy et al. | 2013 | Discovers adversarial examples | 6411 KB |
| [Explaining and Harnessing Adversarial Examples](2014-goodfellow-adversarial-examples.pdf) | Goodfellow, Shlens & Szegedy | 2014 | FGSM | 1012 KB |
| [Generative Adversarial Nets](2014-goodfellow-gan.pdf) | Goodfellow et al. | 2014 | ⭐🔧 Minimax derivable by hand | 518 KB |
| [Conditional Generative Adversarial Nets](2014-mirza-conditional-gan.pdf) | Mirza & Osindero | 2014 |  | 785 KB |
| [Unsupervised Representation Learning with DCGANs](2015-radford-dcgan.pdf) | Radford, Metz & Chintala | 2015 |  | 7282 KB |
| [Variational Inference with Normalizing Flows](2015-rezende-normalizing-flows.pdf) | Rezende & Mohamed | 2015 | Links with Bayesian learning | 2833 KB |
| [Image-to-Image Translation with Conditional Adversarial Networks](2016-isola-pix2pix.pdf) | Isola et al. | 2016 | pix2pix | 9105 KB |
| [Improved Techniques for Training GANs](2016-salimans-improved-gan-training.pdf) | Salimans et al. | 2016 | Why GANs do not train | 2291 KB |
| [Wasserstein GAN](2017-arjovsky-wasserstein-gan.pdf) | Arjovsky, Chintala & Bottou | 2017 | 🔧 Optimal transport | 8627 KB |
| [Progressive Growing of GANs](2017-karras-progressive-gan.pdf) | Karras et al. | 2017 |  | 27856 KB |
| [Neural Discrete Representation Learning (VQ-VAE)](2017-oord-vq-vae.pdf) | van den Oord, Vinyals & Kavukcuoglu | 2017 | Discrete latents | 3100 KB |
| [Unpaired Image-to-Image Translation (CycleGAN)](2017-zhu-cyclegan.pdf) | Zhu et al. | 2017 |  | 36671 KB |
| [Large Scale GAN Training (BigGAN)](2018-brock-biggan.pdf) | Brock, Donahue & Simonyan | 2018 |  | 10657 KB |
| [A Style-Based Generator Architecture for GANs](2018-karras-stylegan.pdf) | Karras, Laine & Aila | 2018 | StyleGAN | 22385 KB |
| [Glow: Generative Flow with Invertible 1x1 Convolutions](2018-kingma-glow.pdf) | Kingma & Dhariwal | 2018 |  | 12939 KB |
| [Denoising Diffusion Probabilistic Models](2020-ho-diffusion-ddpm.pdf) | Ho, Jain & Abbeel | 2020 | ⭐ Applied statistical physics | 10026 KB |
| [Denoising Diffusion Implicit Models](2020-song-ddim.pdf) | Song, Meng & Ermon | 2020 | DDIM — fast sampling | 10602 KB |
| [Score-Based Generative Modeling through SDEs](2020-song-score-based-sde.pdf) | Song et al. | 2020 | 🔧 SDEs — squarely a physicist's territory | 26287 KB |
| [Learning Transferable Visual Models From Natural Language Supervision](2021-radford-clip.pdf) | Radford et al. | 2021 | CLIP | 6653 KB |
| [High-Resolution Image Synthesis with Latent Diffusion Models](2021-rombach-latent-diffusion.pdf) | Rombach et al. | 2021 | Stable Diffusion | 39885 KB |
| [Classifier-Free Diffusion Guidance](2022-ho-classifier-free-guidance.pdf) | Ho & Salimans | 2022 |  | 3685 KB |

## F · Vision

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Gradient-Based Learning Applied to Document Recognition](1998-lecun-gradient-based-learning.pdf) | LeCun et al. | 1998 | LeNet | 932 KB |
| [ImageNet Classification with Deep Convolutional Neural Networks](2012-krizhevsky-alexnet.pdf) | Krizhevsky, Sutskever & Hinton | 2012 | AlexNet | 1385 KB |
| [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](2014-srivastava-dropout.pdf) | Srivastava et al. | 2014 |  | 2801 KB |
| [Deep Residual Learning for Image Recognition](2015-he-resnet.pdf) | He et al. | 2015 | ResNet | 800 KB |
| [Batch Normalization](2015-ioffe-batchnorm.pdf) | Ioffe & Szegedy | 2015 |  | 169 KB |
| [U-Net: Convolutional Networks for Biomedical Image Segmentation](2015-ronneberger-unet.pdf) | Ronneberger et al. | 2015 |  | 1610 KB |
| [An Image is Worth 16x16 Words](2020-dosovitskiy-vit.pdf) | Dosovitskiy et al. | 2020 | ViT | 3656 KB |

## G · RL

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Playing Atari with Deep Reinforcement Learning](2013-mnih-dqn-atari.pdf) | Mnih et al. | 2013 | DQN | 472 KB |
| [Proximal Policy Optimization Algorithms](2017-schulman-ppo.pdf) | Schulman et al. | 2017 | The basis of RLHF | 2855 KB |

## Master's · Mathematics

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Stable Signal Recovery from Incomplete and Inaccurate Measurements](2005-candes-stable-signal-recovery.pdf) | Candès, Romberg & Tao | 2005 | L1 minimisation — connects with the lasso | 462 KB |
| [Finding Structure with Randomness](2011-halko-randomized-svd.pdf) | Halko, Martinsson & Tropp | 2011 | Randomized SVD | 1257 KB |

## Master's · Big data foundations

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [MapReduce: Simplified Data Processing on Large Clusters](2004-dean-mapreduce.pdf) | Dean & Ghemawat | 2004 |  | 186 KB |
| [Resilient Distributed Datasets](2012-zaharia-spark-rdd.pdf) | Zaharia et al. | 2012 | Spark | 865 KB |

## Master's · Back-end

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Bigtable: A Distributed Storage System for Structured Data](2006-chang-bigtable.pdf) | Chang et al. | 2006 |  | 216 KB |
| [Cassandra: A Decentralized Structured Storage System](2010-lakshman-cassandra.pdf) | Lakshman & Malik | 2010 |  | 130 KB |

## Master's · CDN

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Consistent Hashing and Random Trees](1997-karger-consistent-hashing.pdf) | Karger et al. | 1997 |  | 201 KB |

## Master's · Prediction models

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Greedy Function Approximation: A Gradient Boosting Machine](2001-friedman-gradient-boosting.pdf) | Friedman | 2001 |  | 949 KB |
| [XGBoost: A Scalable Tree Boosting System](2016-chen-xgboost.pdf) | Chen & Guestrin | 2016 |  | 922 KB |

## Master's · Statistical learning

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Random Forests](2001-breiman-random-forests.pdf) | Breiman | 2001 |  | 120 KB |
| [Statistical Modeling: The Two Cultures](2001-breiman-two-cultures.pdf) | Breiman | 2001 | Review only, no code | 300 KB |

## Master's · Optimisation

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Distributed Optimization via ADMM](2011-boyd-admm.pdf) | Boyd et al. | 2011 |  | 775 KB |
| [Adam: A Method for Stochastic Optimization](2014-kingma-adam.pdf) | Kingma & Ba | 2014 |  | 570 KB |
| [Optimization Methods for Large-Scale Machine Learning](2018-bottou-optimization-ml.pdf) | Bottou, Curtis & Nocedal | 2018 |  | 1908 KB |

## Master's · Bayesian learning

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Latent Dirichlet Allocation](2003-blei-lda.pdf) | Blei, Ng & Jordan | 2003 |  | 408 KB |
| [The No-U-Turn Sampler](2014-hoffman-nuts.pdf) | Hoffman & Gelman | 2014 |  | 1003 KB |
| [Variational Inference: A Review for Statisticians](2017-blei-variational-inference.pdf) | Blei, Kucukelbir & McAuliffe | 2017 |  | 1780 KB |

## Master's · Time series

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [DeepAR: Probabilistic Forecasting with Autoregressive RNNs](2020-salinas-deepar.pdf) | Salinas et al. | 2020 |  | 555 KB |

## Master's · Networks

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [The Anatomy of a Large-Scale Hypertextual Web Search Engine](1998-brin-pagerank.pdf) | Brin & Page | 1998 | PageRank — pure eigenvectors | 120 KB |
| [Fast Unfolding of Communities in Large Networks](2008-blondel-louvain.pdf) | Blondel et al. | 2008 | Louvain | 1557 KB |
| [Matrix Factorization Techniques for Recommender Systems](2009-koren-matrix-factorization.pdf) | Koren, Bell & Volinsky | 2009 | Netflix prize | 1511 KB |
| [node2vec: Scalable Feature Learning for Networks](2016-grover-node2vec.pdf) | Grover & Leskovec | 2016 |  | 781 KB |
| [Semi-Supervised Classification with Graph Convolutional Networks](2016-kipf-gcn.pdf) | Kipf & Welling | 2016 | GCN | 853 KB |
| [Graph Attention Networks](2017-velickovic-graph-attention.pdf) | Veličković et al. | 2017 | GAT | 1599 KB |
| [Neural Ordinary Differential Equations](2018-chen-neural-ode.pdf) | Chen et al. | 2018 | ODEs and deep networks | 3897 KB |

## Quant

| Paper | Authors | Year | Note | Size |
|---|---|---|---|---|
| [Portfolio Selection](1952-markowitz-portfolio-selection.pdf) | Markowitz | 1952 | ⭐🔧 ✅ review · implementation in progress | 1132 KB |
| [A New Approach to Linear Filtering and Prediction Problems](1960-kalman-filtering.pdf) | Kalman | 1960 |  | 173 KB |
| [High-frequency Trading in a Limit Order Book](2008-avellaneda-market-making.pdf) | Avellaneda & Stoikov | 2008 | Market making | 423 KB |

## Still to download

- `2008-candes-compressive-sampling` — *An Introduction to Compressive Sampling* (Candès & Wakin, 2008)
- `2017-taylor-prophet` — *Forecasting at Scale* (Taylor & Letham, 2017)

### Quant — still to download (see the thematic blocks in the [ROADMAP](../ROADMAP.md))

The classics are nearly all in paywalled journals (JSTOR / Wiley / Elsevier); a legitimate
copy is usually available on university course pages. Confirmed useful links:

- `1964-sharpe-capm` — *Capital Asset Prices* (Sharpe, 1964) — [hosted PDF](http://psc.ky.gov/pscecf/2012-00221/rateintervention@ag.ky.gov/10252012f/sharpe_-_CAPM.pdf) · [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1964.tb02865.x)
- `1958-tobin-liquidity-preference` — *Liquidity Preference as Behavior Towards Risk* (Tobin, 1958, *RES* 25)
- `1970-fama-efficient-markets` — *Efficient Capital Markets* (Fama, 1970, *JF* 25)
- `1973-black-scholes-options` — *The Pricing of Options and Corporate Liabilities* (Black & Scholes, 1973, *JPE* 81)
- `1993-fama-french-three-factor` — *Common Risk Factors in the Returns on Stocks and Bonds* (*JFE* 33)
- `2004-ledoit-wolf-shrinkage` — *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* (*JMVA* 88) — the direct antidote to Markowitz's estimation problem
- `2000-almgren-chriss-execution` — *Optimal Execution of Portfolio Transactions* — [PDF (NYU)](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- `2001-cont-stylized-facts` — *Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues* (*Quantitative Finance* 1)

---

**Total in `papers/`: 112 PDFs.**
