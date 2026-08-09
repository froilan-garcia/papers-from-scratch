# Roadmap: papers by master's course

MSc in Analytical Methods for Big Data (UC3M, starting September 2026).
60 ECTS · 1 year · taught in English · [syllabus](https://www.uc3m.es/master/big-data)

Each course carries candidate papers for the `/paper` workflow (review + Python
implementation). Reviewed ones are marked `[x]` with a link to their review.

> 📖 **This document is a catalogue, not a queue.** The papers for each course are read
> *when that course comes round*, not before.

## Term 1 · First half (Sep–Oct)

### Mathematics for data analysis
- [ ] Halko, Martinsson & Tropp (2011) — *Finding Structure with Randomness* (randomized SVD) — arXiv:0909.4061
- [ ] Candès & Wakin (2008) — *An Introduction to Compressive Sampling*

### Statistics for data analysis
- [x] Tibshirani (1996) — *Regression Shrinkage and Selection via the Lasso* — [review](reviews/1996-tibshirani-lasso.md) ✔ · [implementation](implementations/1996-tibshirani-lasso/) ✔ using the quadratic programming algorithm of Sec. 6 (not coordinate descent, which is from 2007). Eq. 3, Eqs. 5–6, Table 1 and the four figures reproduced, and cross-checked against `sklearn` and LARS to $10^{-13}$. The full mathematical development is in [DERIVATIONS.md](implementations/1996-tibshirani-lasso/DERIVATIONS.md), 20 sections with figures computed by the solver. Documented discrepancies: the `lweight` typo, the GCV that does not give 0.44, and two errata in the paper (the lower limit missing from Eq. 6, and `max` for `min` in the Stein formula). The Sec. 7 simulations stay local until the setup is settled
- [x] Efron (1979) — *Bootstrap Methods: Another Look at the Jackknife* — [review](reviews/1979-efron-bootstrap.md) ✔ · [implementation](implementations/1979-efron-bootstrap/) ✔ closed at four of the paper's eight sections and three of its eleven remarks, by choice, with the reason for each omission stated. Eq. (2.8), Eq. (3.5) against enumeration, the six probabilities of Eq. (3.6), column (3.6) of Table 1, Eqs. (5.14)–(5.15), Eq. (7.7) and the symmetrization of Eq. (7.9), Eq. (8.1) and Figure 1 all reproduced. The development is in [DERIVATIONS.md](implementations/1979-efron-bootstrap/DERIVATIONS.md), seven sections from the question the paper answers to the jackknife. Documented discrepancies: Eq. (3.12) needs a factor $\sqrt n$ before Table 1 can be reproduced, and $E_F R$ comes out at 0.982 rather than the stated 0.95. Beyond the paper: the limiting law of the jackknife variance of a median depends on the parity of $n$
- [ ] Hoerl & Kennard (1970) — *Ridge Regression: Biased Estimation for Nonorthogonal Problems* (review only, no code — it closes the regularisation diptych with the lasso; ridge = *weight decay* in neural networks)

### Technological foundations of big data
- [ ] Dean & Ghemawat (2004) — *MapReduce: Simplified Data Processing on Large Clusters*
- [ ] Zaharia et al. (2012) — *Resilient Distributed Datasets* (Spark)

### High-performance computing for big data
- [ ] Blelloch (1990) — *Prefix Sums and Their Applications* (basic parallelism)

### Back-end for big data analysis
- [ ] Chang et al. (2006) — *Bigtable: A Distributed Storage System*
- [ ] Lakshman & Malik (2010) — *Cassandra: A Decentralized Structured Storage System*

## Term 1 · Second half (Nov–Dec)

### Content distribution on the internet
- [ ] Karger et al. (1997) — *Consistent Hashing and Random Trees* (the basis of CDNs)

### Prediction models
- [ ] Friedman (2001) — *Greedy Function Approximation: A Gradient Boosting Machine*
- [ ] Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System* — arXiv:1603.02754

### Statistical learning
- [ ] Breiman (2001) — *Random Forests*
- [ ] Breiman (2001) — *Statistical Modeling: The Two Cultures* (review only, no code)

### Optimisation for large-scale data
- [ ] Bottou, Curtis & Nocedal (2018) — *Optimization Methods for Large-Scale Machine Learning* — arXiv:1606.04838
- [ ] Boyd et al. (2011) — *Distributed Optimization via ADMM*
- [ ] Kingma & Ba (2014) — *Adam: A Method for Stochastic Optimization* — arXiv:1412.6980

### Intelligence for big data: methods and technologies
- [ ] Mikolov et al. (2013) — *Efficient Estimation of Word Representations* (word2vec) — arXiv:1301.3781

## Term 2 · First half (Feb–Mar)

### Bayesian learning
- [ ] Blei, Ng & Jordan (2003) — *Latent Dirichlet Allocation*
- [ ] Blei, Kucukelbir & McAuliffe (2017) — *Variational Inference: A Review for Statisticians* — arXiv:1601.00670
- [ ] Hoffman & Gelman (2014) — *The No-U-Turn Sampler* — arXiv:1111.4246

### Time series analysis and forecasting
- [ ] Taylor & Letham (2017) — *Forecasting at Scale* (Prophet)
- [ ] Salinas et al. (2020) — *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks* — arXiv:1704.04110

### Machine learning
- [x] Vaswani et al. (2017) — *Attention Is All You Need* — [review](reviews/2017-vaswani-attention.md) ✔ · [implementation](implementations/2017-vaswani-attention/) in progress
- [ ] Srivastava et al. (2014) — *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*
- [ ] He et al. (2015) — *Deep Residual Learning for Image Recognition* — arXiv:1512.03385

### Business applications of big data analysis
- [ ] Koren, Bell & Volinsky (2009) — *Matrix Factorization Techniques for Recommender Systems* (Netflix prize)

### Web data analysis and exploitation
- [ ] Brin & Page (1998) — *The Anatomy of a Large-Scale Hypertextual Web Search Engine* (PageRank)
- [ ] Kleinberg (1999) — *Authoritative Sources in a Hyperlinked Environment* (HITS)

## Term 2 · Second half (Apr–May) — electives

### Network analysis and data visualisation
- [ ] Blondel et al. (2008) — *Fast Unfolding of Communities in Large Networks* (Louvain) — arXiv:0803.0476
- [ ] Grover & Leskovec (2016) — *node2vec* — arXiv:1607.00653
- [ ] Kipf & Welling (2016) — *Semi-Supervised Classification with Graph Convolutional Networks* (GCN) — arXiv:1609.02907
- [ ] Veličković et al. (2017) — *Graph Attention Networks* — arXiv:1710.10903 — attention applied to graphs
- [ ] Chen et al. (2018) — *Neural Ordinary Differential Equations* — arXiv:1806.07366 — ODEs and deep networks; direct territory for a physics background

### Data analysis for the smart society
- [ ] (elective still to be chosen)

### Information security
- [ ] (elective still to be chosen)

## Library: deep learning, LLMs and generative models

> ⚠️ **This is NOT a reading queue or a to-do list.** It is a **reference shelf**: papers
> already downloaded so that they are at hand the day they are needed. They do not have
> to be read, neither in order nor at all.

The tracks below indicate **in what order they would make sense** *if* one day I pull on
a topic, because each paper answers the limitation of the previous one. They are for
orienting oneself within the shelf, not for walking end to end.

Full inventory in [papers/INDEX.md](papers/INDEX.md). ⭐ = the essential one of the block ·
🔧 = the one worth implementing.

### Track A — From RNNs to the Transformer (the historical line)

Order matters: each paper answers a limitation of the previous one. It is the best way to
understand *why* the Transformer exists.

- [ ] Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory* — the vanishing gradient problem
- [ ] Pascanu, Mikolov & Bengio (2012) — *On the Difficulty of Training RNNs* — arXiv:1211.5063 — the theory of exploding/vanishing gradients
- [ ] Cho et al. (2014) — *Learning Phrase Representations* (GRU + encoder-decoder) — arXiv:1406.1078
- [ ] Chung et al. (2014) — *Empirical Evaluation of Gated RNNs* — arXiv:1412.3555 — LSTM vs GRU head to head
- [ ] Sutskever, Vinyals & Le (2014) — *Sequence to Sequence Learning with Neural Networks* — arXiv:1409.3215
- [ ] ⭐ Bahdanau, Cho & Bengio (2014) — *Neural Machine Translation by Jointly Learning to Align and Translate* — arXiv:1409.0473 — **the real origin of attention**
- [ ] Luong, Pham & Manning (2015) — *Effective Approaches to Attention-based NMT* — arXiv:1508.04025 — multiplicative attention
- [ ] Xu et al. (2015) — *Show, Attend and Tell* — arXiv:1502.03044 — visual attention
- [ ] Graves, Wayne & Danihelka (2014) — *Neural Turing Machines* — arXiv:1410.5401 — addressable external memory
- [ ] van den Oord et al. (2016) — *WaveNet* — arXiv:1609.03499 — causal dilated convolutions
- [x] ⭐🔧 Vaswani et al. (2017) — *Attention Is All You Need* — [review](reviews/2017-vaswani-attention.md) ✔ · [implementation](implementations/2017-vaswani-attention/) in progress. Implementing self-attention and multi-head from scratch is the highest-yield exercise on the whole track

### Track B — Tokenisation and embeddings

- [ ] Mikolov et al. (2013) — *Efficient Estimation of Word Representations* (word2vec) — arXiv:1301.3781
- [ ] Mikolov et al. (2013) — *Distributed Representations of Words and Phrases* (negative sampling) — arXiv:1310.4546
- [ ] Pennington, Socher & Manning (2014) — *GloVe: Global Vectors for Word Representation*
- [ ] ⭐🔧 Sennrich, Haddow & Birch (2015) — *Neural Machine Translation of Rare Words with Subword Units* (**BPE**) — arXiv:1508.07909 — the tokenisation algorithm used by GPT and almost everyone else; implementable in ~50 lines
- [ ] Wu et al. (2016) — *Google's Neural Machine Translation System* — arXiv:1609.08144 — **WordPiece**, BERT's tokeniser
- [ ] Kudo (2018) — *Subword Regularization* (unigram tokeniser) — arXiv:1804.10959
- [ ] Kudo & Richardson (2018) — *SentencePiece* — arXiv:1808.06226
- [ ] Yu et al. (2023) — *MEGABYTE: Predicting Million-byte Sequences* — arXiv:2305.07185 — byte-level modelling
- [ ] Pagnoni et al. (2024) — *Byte Latent Transformer* — arXiv:2412.09871 — **no tokeniser**; the most current work in the area

### Track C — Modern LLMs

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
- [ ] Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2) — not on arXiv; PDF at [cdn.openai.com](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ ] Dai et al. (2019) — *Transformer-XL* — arXiv:1901.02860
- [ ] Sanh et al. (2019) — *DistilBERT* — arXiv:1910.01108 — model distillation
- [ ] Clark et al. (2020) — *ELECTRA* — arXiv:2003.10555
- [ ] Beltagy et al. (2020) — *Longformer* — arXiv:2004.05150 — sparse attention for long documents
- [ ] Wei et al. (2022) — *Emergent Abilities of Large Language Models* — arXiv:2206.07682
- [ ] Bai et al. (2022) — *Constitutional AI* — arXiv:2212.08073 — alignment without massive human labelling
- [ ] Touvron et al. (2023) — *LLaMA* — arXiv:2302.13971
- [ ] Rafailov et al. (2023) — *Direct Preference Optimization* — arXiv:2305.18290 — RLHF without RL
- [ ] Radford et al. (2022) — *Whisper* — arXiv:2212.04356 — speech recognition at scale

### Track D — Transformer engineering (why today's LLMs are viable)

- [ ] Su et al. (2021) — *RoFormer* (**RoPE**, rotary positional embeddings) — arXiv:2104.09864
- [ ] Shazeer (2019) — *Fast Transformer Decoding* (multi-query attention) — arXiv:1911.02150
- [ ] 🔧 Dao et al. (2022) — *FlashAttention* — arXiv:2205.14135 — connects with the high-performance computing course
- [ ] Shazeer et al. (2017) — *Outrageously Large Neural Networks* (mixture of experts) — arXiv:1701.06538
- [ ] Fedus, Zoph & Shazeer (2021) — *Switch Transformers* — arXiv:2101.03961 — MoE taken to a trillion parameters
- [ ] Hu et al. (2021) — *LoRA: Low-Rank Adaptation* — arXiv:2106.09685 — pure linear algebra
- [ ] Gu, Goel & Ré (2021) — *Structured State Spaces* (S4) — arXiv:2111.00396 — the precursor of Mamba
- [ ] Gu & Dao (2023) — *Mamba: Linear-Time Sequence Modeling* — arXiv:2312.00752 — the post-Transformer alternative
- [ ] Ba, Kiros & Hinton (2016) — *Layer Normalization* — arXiv:1607.06450
- [ ] Loshchilov & Hutter (2017) — *Decoupled Weight Decay* (AdamW) — arXiv:1711.05101 — *weight decay* done properly (ridge, not lasso)
- [ ] Hendrycks & Gimpel (2016) — *Gaussian Error Linear Units* (GELU) — arXiv:1606.08415

### Track E — Generative models (GAN, VAE, diffusion)

**GANs (adversarial networks):**
- [ ] ⭐🔧 Goodfellow et al. (2014) — *Generative Adversarial Nets* — arXiv:1406.2661 — game theory + statistics; the minimax equilibrium can be derived by hand
- [ ] Mirza & Osindero (2014) — *Conditional GAN* — arXiv:1411.1784
- [ ] Radford, Metz & Chintala (2015) — *DCGAN* — arXiv:1511.06434
- [ ] Salimans et al. (2016) — *Improved Techniques for Training GANs* — arXiv:1606.03498 — why GANs do not train
- [ ] 🔧 Arjovsky, Chintala & Bottou (2017) — *Wasserstein GAN* — arXiv:1701.07875 — optimal transport; mathematically the prettiest
- [ ] Isola et al. (2016) — *pix2pix* — arXiv:1611.07004 · Zhu et al. (2017) — *CycleGAN* — arXiv:1703.10593
- [ ] Karras et al. (2017) — *Progressive GAN* — arXiv:1710.10196 · Karras et al. (2018) — *StyleGAN* — arXiv:1812.04948
- [ ] Brock et al. (2018) — *BigGAN* — arXiv:1809.11096

**Variational models and flows:**
- [ ] ⭐🔧 Kingma & Welling (2013) — *Auto-Encoding Variational Bayes* (VAE) — arXiv:1312.6114 — links directly with Bayesian learning (ELBO, reparametrisation)
- [ ] Rezende & Mohamed (2015) — *Variational Inference with Normalizing Flows* — arXiv:1505.05770
- [ ] van den Oord et al. (2017) — *VQ-VAE* — arXiv:1711.00937 — discrete latents
- [ ] Kingma & Dhariwal (2018) — *Glow* — arXiv:1807.03039

**Diffusion:**
- [ ] ⭐ Ho, Jain & Abbeel (2020) — *Denoising Diffusion Probabilistic Models* — arXiv:2006.11239 — diffusion processes: applied statistical physics
- [ ] 🔧 Song et al. (2020) — *Score-Based Generative Modeling through SDEs* — arXiv:2011.13456 — stochastic differential equations, squarely a physicist's territory
- [ ] Song, Meng & Ermon (2020) — *DDIM* — arXiv:2010.02502 — fast sampling
- [ ] Ho & Salimans (2022) — *Classifier-Free Guidance* — arXiv:2207.12598
- [ ] Rombach et al. (2021) — *Latent Diffusion Models* (Stable Diffusion) — arXiv:2112.10752
- [ ] Radford et al. (2021) — *CLIP* — arXiv:2103.00020

**Adversarial examples (robustness, the other sense of "adversarial"):**
- [ ] Szegedy et al. (2013) — *Intriguing Properties of Neural Networks* — arXiv:1312.6199 — the discovery
- [ ] Goodfellow, Shlens & Szegedy (2014) — *Explaining and Harnessing Adversarial Examples* — arXiv:1412.6572 — FGSM

### Track F — Vision and training fundamentals

- [ ] LeCun et al. (1998) — *Gradient-Based Learning Applied to Document Recognition* (LeNet)
- [ ] Krizhevsky, Sutskever & Hinton (2012) — *ImageNet Classification with Deep CNNs* (AlexNet)
- [ ] He et al. (2015) — *Deep Residual Learning* (ResNet) — arXiv:1512.03385
- [ ] Ioffe & Szegedy (2015) — *Batch Normalization* — arXiv:1502.03167
- [ ] Srivastava et al. (2014) — *Dropout*
- [ ] Ronneberger et al. (2015) — *U-Net* — arXiv:1505.04597
- [ ] Dosovitskiy et al. (2020) — *An Image is Worth 16x16 Words* (ViT) — arXiv:2010.11929

### Track G — Reinforcement learning (optional)

- [ ] Mnih et al. (2013) — *Playing Atari with Deep Reinforcement Learning* (DQN) — arXiv:1312.5602
- [ ] Schulman et al. (2017) — *Proximal Policy Optimization* — arXiv:1707.06347 — the basis of RLHF in LLMs

## Quant papers for the `/paper` workflow

A block oriented towards *quantitative research*: portfolio theory, covariance
estimation, financial time series and microstructure. It complements the master's
courses on the applied side, and several of its papers are implementable from scratch
with the base stack.

Organised by theme rather than as a queue: each block answers a different question.
⭐ = the essential one of the block · 🔧 = worth implementing from scratch.

#### Q1 · Portfolio theory and market equilibrium (the canon)

The historical line: each paper answers a limitation of the previous one. It is the block
with the best understanding-per-effort ratio and the one most cited in interviews.

- [x] ⭐🔧 **Markowitz (1952)** — *Portfolio Selection* — [review](reviews/1952-markowitz-portfolio-selection.md) ✔ · [implementation](implementations/1952-markowitz-portfolio-selection/) in progress. Mean-variance optimisation; pure linear algebra. PDF in `papers/`.
- [ ] Tobin (1958) — *Liquidity Preference as Behavior Towards Risk* — adds the **risk-free asset** → the **two-fund separation theorem** and the capital market line.
- [ ] ⭐ Sharpe (1964) — *Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk* — **CAPM**: $\beta$ and systematic risk. It is Markowitz taken to market equilibrium. [PDF](http://psc.ky.gov/pscecf/2012-00221/rateintervention@ag.ky.gov/10252012f/sharpe_-_CAPM.pdf)
- [ ] Fama (1970) — *Efficient Capital Markets: A Review of Theory and Empirical Work* — **EMH**, the three forms of efficiency. The conceptual frame of "no free lunch".
- [ ] 🔧 Fama & French (1993) — *Common Risk Factors in the Returns on Stocks and Bonds* — the **three-factor model**; CAPM is not enough. Implementable as a regression on real factors (free data on French's website).

#### Q2 · The Achilles heel: estimating $\boldsymbol\mu$ and $\Sigma$

Markowitz himself leaves the stage-1 problem open. This block is **the one most directly
connected with the master's** (regularisation, covariance matrices, ML).

- [ ] Michaud (1989) — *The Markowitz Optimization Enigma: Is Optimized Optimal?* — why optimisation **amplifies** estimation error (*error maximization*).
- [ ] ⭐🔧 Ledoit & Wolf (2004) — *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices* — **covariance shrinkage**; it is *ridge* applied to $\Sigma$, a direct connection with the lasso already reviewed.
- [ ] Black & Litterman (1992) — *Global Portfolio Optimization* — injecting Bayesian *views* on $\boldsymbol\mu$; links with Bayesian learning.
- [ ] 🔧 López de Prado (2016) — *Building Diversified Portfolios that Outperform Out-of-Sample* (**Hierarchical Risk Parity**) — clustering instead of inverting $\Sigma$; ML applied to portfolios.
- [ ] Brodie et al. (2009) — *Sparse and Stable Markowitz Portfolios* — an $L_1$ penalty on the weights. **The exact crossing of lasso × Markowitz.**

#### Q3 · Derivative pricing

- [ ] ⭐ Black & Scholes (1973) — *The Pricing of Options and Corporate Liabilities* — a direct connection with physics: the PDE reduces to the **heat equation**.
- [ ] Merton (1973) — *Theory of Rational Option Pricing* — the general and rigorous formulation.
- [ ] 🔧 Cox, Ross & Rubinstein (1979) — *Option Pricing: A Simplified Approach* — the **binomial model**; the pedagogical route to implementing Black–Scholes and watching the convergence to the continuum.

#### Q4 · Financial time series and stylised facts

- [ ] ⭐ Cont (2001) — *Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues* — heavy tails, volatility clustering, absence of autocorrelation. **Read early**: it is the reality check on all the Gaussian assumptions of Q1.
- [ ] 🔧 Engle (1982) — *Autoregressive Conditional Heteroscedasticity* (**ARCH**) — Nobel 2003; volatility is not constant.
- [ ] Bollerslev (1986) — *Generalized ARCH* (**GARCH**) — the real workhorse.
- [ ] 🔧 Kalman (1960) — *A New Approach to Linear Filtering and Prediction Problems* — sequential filtering; the basis of many quant signals and of dynamic estimation of $\boldsymbol\mu$/$\Sigma$. PDF in `papers/`.

#### Q5 · Strategies and microstructure

- [ ] 🔧 Gatev, Goetzmann & Rouwenhorst (2006) — *Pairs Trading: Performance of a Relative-Value Arbitrage Rule* — the first **complete strategy with a backtest**; cointegration.
- [ ] Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers* — the original **momentum** paper.
- [ ] Moskowitz, Ooi & Pedersen (2012) — *Time Series Momentum* — factor investing over time series.
- [ ] Kyle (1985) — *Continuous Auctions and Insider Trading* — the canonical model of **microstructure** and informational impact.
- [ ] 🔧 Almgren & Chriss (2000) — *Optimal Execution of Portfolio Transactions* — **optimal execution**: stochastic control, much asked about in interviews. [PDF](https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf)
- [ ] Avellaneda & Stoikov (2008) — *High-frequency Trading in a Limit Order Book* — **market making**, HJB. PDF in `papers/`.

#### A suggested order, if the thread gets pulled

**Q1 (Markowitz → Sharpe) → Q2 (Ledoit–Wolf) → Q4 (Cont) → Q5 (pairs trading)**. That
route goes from theory to backtest by way of the statistical critique, and leaves a very
presentable GitHub portfolio: efficient frontier, covariance shrinkage, stylised facts
and a strategy with a backtest.

Synergies with the rest of the roadmap: the lasso already reviewed (regularisation → Q2),
Efron/bootstrap (uncertainty of the frontier → Q1–Q2), Friedman (boosting) and Bottou
(stochastic optimisation) are directly relevant to quantitative research.

## Groundwork (July–August 2026, before starting)

A suggested order for reaching September on solid ground — from least to most demanding:

1. ~~Tibshirani (1996), Lasso~~ — done. Statistics + optimisation; it came out via quadratic programming with an active set, which is what the paper uses, not via coordinate descent.
2. Brin & Page (1998), PageRank — pure linear algebra (eigenvectors).
3. Friedman (2001), Gradient Boosting — the workhorse of tabular data analysis.
4. Kingma & Ba (2014), Adam — the gateway to stochastic optimisation.
5. Blei et al. (2003), LDA — a perfect introduction to the Bayesian world of the second term.
