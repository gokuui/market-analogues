# Historical Stock Analogue Representation — Research and Implementation Plan

**Date:** 2026-08-18
**Status:** Proposed for review
**Scope:** Point-in-time historical analogue retrieval for US equities; representation first, outcome analysis second, automatic setup discovery later

**Reader-facing plan:** Open `2026-08-18-historical-analogue-plan.html` in a browser. This Markdown file is the detailed technical and research appendix.

## 1. Executive decision

The system should not search for a named VCP, flag, cup, or EP. It should answer:

> Among episodes that were already observable at the query cutoff, which stocks developed most similarly to this stock—including price action, shocks, volume, relative strength, and market context—and what happened after their cutoffs?

There is no single mathematical representation that simultaneously:

- preserves every OHLCV observation;
- captures daily, weekly, monthly, and yearly structure;
- ignores irrelevant differences such as nominal price;
- preserves relevant differences such as a 7% versus 40% contraction;
- permits formations to unfold at somewhat different speeds;
- retains stock/index/sector interactions;
- remains interpretable; and
- searches tens of millions of historical stock-date episodes quickly.

The recommended solution is therefore a **lossless canonical episode plus a portfolio of complementary derived views**:

1. A canonical, point-in-time episode retains everything needed to reproduce exactly what was observable at the cutoff.
2. A dense multiscale trajectory retains candle mechanics, volume, relative strength, and market paths.
3. A structural skeleton retains swings, shocks, contractions, expansions, gaps, and their relationships.
4. A time-scale view retains localized behavior at different frequencies.
5. A context view retains market, sector, breadth, volatility, and liquidity state.
6. Retrieval uses a fast coarse search followed by transparent, constrained elastic re-ranking.

The first implementation should be transparent. Learned embeddings remain a later benchmark, not the starting point and not a price-prediction model.

## 2. Representation and retrieval are different problems

The project has four separate layers. They must not be optimized as if they were one model.

| Layer | Question | Must not contain |
|---|---|---|
| Canonical episode | What was observable at cutoff `t`? | Any post-`t` observation |
| Representation | How do we express that information for comparison? | Outcomes or setup success labels |
| Similarity | What differences matter, and how much? | A hidden definition of “successful chart” |
| Outcome analysis | What occurred after comparable episodes? | Data used to alter the already-locked representation during the same test |

This separation prevents a subtle form of leakage: defining “similar” in the way that happens to produce the most profitable neighbors.

## 3. The information contract: what the trader could see

### 3.1 Episode identity and time semantics

Every episode is identified by:

```text
security_id + cutoff_session + representation_version + data_snapshot_version
```

`security_id` must be stable across ticker changes and must not merge different companies that reused the same ticker. The cutoff must specify the exact decision moment:

- **after close:** that day's complete OHLCV bar is available;
- **at next open:** the previous close history plus the current opening gap are available;
- **intraday:** requires a separate intraday design and must not be approximated with the completed daily bar.

Version one should use **after-close episodes**. EP/gap-open retrieval can be added as a separately timestamped episode type.

### 3.2 Canonical source data

The canonical episode is a pointer into immutable source tables, not a duplicated 252-day array for every date. It must be able to reconstruct:

#### Stock-native observations

- raw and adjusted open, high, low, close;
- raw share volume and, if available, split-adjusted volume;
- adjustment factor and corporate-action records;
- trading-session calendar, missing bars, suspensions, and zero-volume observations;
- security metadata known at the cutoff: exchange, security type, sector, industry, shares outstanding when available, and listing status;
- data source, validation flags, correction history, and bar confidence.

#### Market and cross-asset observations

- NASDAQ Composite or an explicitly chosen broad NASDAQ benchmark;
- QQQ for tradeable large-cap NASDAQ behavior;
- SPY for the broad US market;
- IWM for smaller-company/risk-appetite context;
- point-in-time sector or industry benchmark;
- breadth series constructed from the eligible point-in-time universe;
- volatility context such as VIX only when the source and history are reliable.

#### Optional non-chart context

- earnings date and whether the report was already public;
- earnings surprise/catalyst classification;
- split, offering, FDA, merger, or other event flags.

These optional event channels are important for EP-like searches, but absence of an event feed must not block the chart-only first version. The interface must say when catalyst information is unknown.

### 3.3 Point-in-time corporate-action policy

This requires an explicit audit. A vendor's currently adjusted historical series may apply a split or dividend that had not occurred at an old simulated cutoff. Although return-based normalization removes much of the scale effect, future adjustments can still change historical price/volume appearance and data-quality decisions.

For each historical cutoff, either:

1. reconstruct adjustment factors known as of that cutoff; or
2. prove that the chosen invariant representation is unchanged by later adjustments and separately correct volume.

The raw source, adjustment factor, and transformed bar must all remain inspectable.

## 4. The invariance contract

“Similar” is impossible to define until we state which transformations should and should not alter similarity.

| Difference between two charts | Desired treatment | Reason |
|---|---|---|
| $20 versus $200 nominal price | Ignore | Nominal level is not chart structure |
| 1M versus 100M normal daily shares | Mostly ignore | Use relative volume, but preserve liquidity separately |
| 8% versus 35% base depth | Preserve | Amplitude materially changes the setup |
| Three-week versus six-week tightening | Partly tolerate, preserve duration | Development speed may vary, but duration contains information |
| One-day shift of an interior pivot | Tolerate slightly | Exact calendar alignment is too rigid |
| Major advance versus flat prior trend | Preserve strongly | Prior momentum is part of the development |
| Stock up while index falls versus both up | Preserve | Relative strength and context differ |
| One isolated bad tick | Be robust, never silently delete | Avoid artifact-driven neighbors |
| Dividend/split unit change | Ignore after correct adjustment | It is not economic shape |
| Reversal of time order | Never ignore | Sequence is essential |
| Rotation of a plotted curve | Never treat as equivalent | An uptrend is not a downtrend |

This immediately rules out using a shape method with automatic rotation invariance as the sole representation. In generic geometry rotation can preserve shape; in finance it changes meaning.

## 5. Canonical causal channel set

All rolling normalization at day `i` must use observations available no later than `i`. A convenient normalized trajectory uses the following channel groups.

### 5.1 Price and candle mechanics

For each session `i`:

```text
close_path_i   = log(adj_close_i / adj_close_start)
cc_return_i    = log(adj_close_i / adj_close_{i-1})
overnight_i    = log(adj_open_i / adj_close_{i-1})
intraday_i     = log(adj_close_i / adj_open_i)
range_i        = log(adj_high_i / adj_low_i)
body_i         = (adj_close_i - adj_open_i) / causal_ATR_i
upper_wick_i   = (adj_high_i - max(adj_open_i, adj_close_i)) / causal_ATR_i
lower_wick_i   = (min(adj_open_i, adj_close_i) - adj_low_i) / causal_ATR_i
close_location = (adj_close_i - adj_low_i) / (adj_high_i - adj_low_i)
```

This preserves distinctions a close-only line loses: gaps, reversals, wide-range accumulation, long upper wicks, and closes near the high or low.

Add deterministic chart-overlay trajectories a professional may actually display:

- distance to and slope of 10/20/21-day exponential averages;
- distance to and slope of 50/100/150/200-day simple averages;
- ordering and separation of those averages over time;
- distance to rolling 20/50/252-day highs and lows;
- distance to ordinary and event-anchored VWAP when the anchor is defined causally.

These overlays never replace the underlying OHLC path. They make visually salient relationships directly comparable and can be ablated to determine whether they add anything beyond raw geometry.

### 5.2 Shock channels

A shock is not a single binary flag. Preserve a causal shock vector:

- robust z-score of close-to-close return;
- robust z-score of overnight gap;
- robust z-score of true range;
- robust z-score of relative volume;
- joint price-volume shock magnitude;
- sign and whether the close held or rejected the move;
- distance to recent high/low when the shock occurred.

Use rolling median and median absolute deviation where practical so an earlier extreme does not dominate the scale estimate.

### 5.3 Volume, participation, and liquidity

Retain multiple meanings of volume instead of one `rvol` scalar:

```text
log_share_volume
log_dollar_volume
volume / causal_median_volume_20
volume / causal_median_volume_50
volume_percentile_252
signed_price_volume = sign(cc_return) * relative_volume
dollar_volume_percentile_in_universe
turnover = volume / point_in_time_shares_outstanding  # when available
```

Also retain price-volume coupling over time: high-volume up days, high-volume down days, dry-up during pullbacks, and expansion after compression.

### 5.4 Volatility, compression, and “spring” state

The spring analogy should be implemented as observable scale relationships, not fictional market physics:

- true range and ATR as percentages of price;
- short/medium/long realized volatility paths;
- rolling high-low envelope widths at several horizons;
- Bollinger or Donchian width as a sequence, not only today's value;
- short-range/long-range ratios;
- successive swing amplitude and duration ratios;
- distance from price to the upper/lower boundary of the current structural range;
- compression persistence and most recent expansion impulse.

This describes coiling, tightening, expansion, and failed compression while remaining interpretable.

### 5.5 Relative movement

For benchmark `b`:

```text
relative_path_i = log(adj_close_stock_i / adj_close_b_i)
excess_return_i = stock_return_i - benchmark_return_i
```

Use at least broad-market and sector-relative paths. A later experiment can add causal rolling-beta residuals:

```text
residual_return_i = stock_return_i - beta_{i-1} * market_return_i
```

Keep raw stock and benchmark paths as well. A residual-only representation would erase the fact that both the stock and market are crashing.

Also retain point-in-time cross-sectional trajectories where the universe supports them:

- stock momentum/relative-strength percentile at 21/63/126/252 sessions;
- industry and sector momentum percentile;
- dollar-volume/liquidity percentile;
- distance of those ranks from their recent highs and their rate of improvement/deterioration.

These ranks answer a different question from the stock/index ratio: not merely whether the stock beat the index, but whether it was becoming a true market leader.

### 5.6 Market context

The market should be represented as a trajectory, not merely `market_above_SMA200 = 1`:

- index OHLC mechanics and normalized price path;
- index short/medium/long volatility and drawdown paths;
- index volume where meaningful;
- breadth: percentage above moving averages, advance/decline balance, new highs/new lows;
- distribution/accumulation day sequence;
- cross-sectional dispersion and fraction of stocks making large moves;
- sector path and sector-relative-to-market path.

Discrete regime labels can be derived later for explanation. They must not replace the underlying context trajectory.

### 5.7 Availability and quality masks

Every channel group carries masks for:

- unavailable;
- stale/forward-filled;
- suspect bar;
- zero volume;
- insufficient lookback;
- non-trading or suspended;
- optional data absent.

Missingness must never be silently converted to a neutral zero because zero is often a meaningful value.

## 6. Time horizons: preserve the chart at several zoom levels

Professional chart reading is multiscale. The proposed episode supplies these synchronized views, all ending at the same cutoff:

| View | Default history | Purpose |
|---|---:|---|
| Micro | 21 sessions | Recent tightening, gap, breakout pressure, failed moves |
| Short | 63 sessions | Current base and recent momentum |
| Setup | 126 sessions | Six-month development |
| Long | 252 sessions | Prior run, stage, 52-week position |
| Structural weekly | 104 weeks | Longer trend and prior major bases |

The raw bar store should retain all available history. These windows are representation views, not destructive truncations. Initial search should compare 126-day and 252-day views jointly; the 21/63-day views increase weight near the current endpoint without throwing away the prior move.

## 7. Research map: candidate representations

### 7.1 Dense raw or resampled trajectories

**Method:** normalized multichannel sequences, optionally downsampled with piecewise aggregate approximation (PAA).

**Captures:** exact order, amplitude, candle mechanics, cross-channel state.
**Loses:** PAA smooths short shocks; fixed alignment dislikes different development speeds.
**Use:** mandatory transparent baseline and fast coarse retrieval.
**Decision:** **P0 — implement first.**

### 7.2 Swing landmarks, PIPs, polygonal chains, and directional events

**Method:** represent important turning points and swings by duration, signed amplitude, slope, range, volume behavior, and relative-strength behavior. Run at multiple prominence thresholds rather than committing to one ZigZag percentage.

**Captures:** the relational grammar a trader describes: advance → pullback → recovery → smaller pullback → shakeout → tight range.
**Loses:** details between landmarks; thresholds can be unstable.
**Use:** interpretable structural distance and explanations.
**Decision:** **P0 — implement as a multithreshold skeleton, always alongside dense data.**

The PIP approach was introduced specifically for flexible pattern matching and subsequently used for financial pattern discovery. Adaptive polygonal methods such as ABBA also preserve ordered peaks/troughs and segment duration/increment.

### 7.3 Euclidean distance, correlation, and k-Shape-style distance

**Method:** z-normalized Euclidean distance or normalized cross-correlation on fixed vectors.

**Captures:** overall morphology cheaply and indexably.
**Loses:** local tempo variation; naïve global z-normalization can make a 5% and 50% formation appear identical.
**Use:** baseline and candidate generation.
**Decision:** **P0. Preserve percent amplitude in separate channels instead of normalizing it away.**

### 7.4 Dynamic Time Warping family

**Method:** DTW, derivative DTW, shapeDTW, multivariate dependent/independent DTW, soft-DTW, or Time Warp Edit Distance.

**Captures:** similar developments that unfold at different local speeds. ShapeDTW improves local-structure matching; derivative DTW emphasizes slopes.
**Risks:** unconstrained DTW can manufacture similarity by excessively stretching a four-day shock into a month or aligning unrelated candles. It is quadratic without pruning and multivariate channel treatment is nontrivial.
**Use:** re-rank a small candidate set, expose the warping path and penalty.
**Decision:** **P0 for constrained DTW; P1 experiments for shapeDTW, TWED, and elastic functional distance.**

Default safeguards:

- Sakoe-Chiba-style warping band around 10–15% of window length;
- explicit penalty for total warp;
- compare dependent and independent multivariate warping;
- never warp stock and market along unrelated time maps in the primary context view;
- retain original duration and warp cost as output fields.

### 7.5 Fourier and symbolic Fourier methods

**Method:** DFT/DCT or Symbolic Fourier Approximation (SFA).

**Captures:** global periodicity and compact indexable summaries.
**Loses:** plain Fourier coefficients localize poorly in time; identical frequency energy can arise in a different event order.
**Use:** scalable coarse-search benchmark or supplementary periodicity descriptor.
**Decision:** **P1 for SFA; do not use Fourier alone as the core.**

### 7.6 SAX, ordinal patterns, and adaptive symbolic strings

**Method:** PAA plus symbols (SAX), order patterns/permutation entropy, SFA words, or adaptive linear-segment symbols such as ABBA.

**Captures:** compact strings, motif indexing, approximate up/down grammar, robustness to noise.
**Loses:** magnitude, exact gaps, candle anatomy, and multichannel coupling unless explicitly extended.
**Use:** coarse retrieval, motif counts, explanation, and a human-readable “chart sentence.”
**Decision:** **P1 supplement, never the only representation.**

### 7.7 Wavelets and scattering transforms

**Method:** discrete/continuous wavelets or wavelet scattering over returns, range, relative volume, and relative-strength paths.

**Captures:** localized shocks and oscillations at several scales; unlike a global Fourier transform, it preserves where scale-specific activity occurred. Scattering adds stability to small deformations.
**Loses:** raw coefficients are difficult to interpret and can become high-dimensional; choices of wavelet and scales matter.
**Use:** supplementary multiscale descriptor, especially for contraction/expansion and shock structure.
**Decision:** **P1 controlled experiment. Do not replace the path or skeleton.**

### 7.8 Matrix Profile and multidimensional motif search

**Method:** exact or approximate all-subsequence similarity joins under z-normalized Euclidean distance; multidimensional extensions such as mSTAMP.

**Captures:** recurring motifs, discords, and candidate subsequences at large scale without named patterns.
**Loses:** its default normalization/distance is not the full trader similarity definition, and choosing fixed subsequence length remains necessary.
**Use:** discovery and fast baseline search within individual or selected channel groups.
**Decision:** **P1 for motif discovery and a retrieval baseline, after the causal channel store exists.**

### 7.9 Path signatures and lead-lag transforms

**Method:** truncated iterated-integral signatures of a time-augmented multichannel path; lead-lag transforms retain discrete variation and order interactions.

**Captures:** ordered nonlinear interactions such as price leading volume, stock leading sector, and compression preceding expansion. It is mathematically well suited to multivariate paths.
**Loses/risks:** truncation grows rapidly with channel count; raw coefficients are hard to explain; standard signatures have invariances that must be deliberately broken with time/basepoint augmentation. Recent financial evidence is mixed and does not establish equity analogue superiority.
**Use:** promising experimental embedding and cross-channel interaction descriptor.
**Decision:** **P2, after transparent baselines.**

### 7.10 Functional/elastic shape analysis

**Method:** curve registration and Fisher-Rao/SRVF distances that separate amplitude variation from phase/timing variation.

**Captures:** a principled distinction between “same move at a different speed” and “different move magnitude.”
**Risks:** generic removal of amplitude, scale, rotation, or reparameterization may remove financially meaningful information. Multichannel extensions and indexing are more complex.
**Use:** research benchmark against constrained DTW.
**Decision:** **P2.**

### 7.11 Change points, state segmentation, and PELT

**Method:** segment mean/variance/trend/volume regimes, producing a sequence of states.

**Captures:** transitions such as advance → consolidation → volatility contraction → shock.
**Loses:** within-state geometry; segmentation is parameter/model dependent.
**Use:** structural skeleton support and explanations.
**Decision:** **P1, compared with multithreshold swing segmentation.**

### 7.12 Image encodings and chart vision

**Method:** rendered candlestick images, Gramian Angular Fields, Markov Transition Fields, or recurrence plots processed by vision methods.

**Captures:** useful spatial textures for classifiers and can mimic visual chart input.
**Loses/risks:** rendering choices can become unintended features; exact OHLCV and cross-channel semantics are less direct; a chart image is downstream of data we already possess.
**Use:** later benchmark only if numeric representations fail to reproduce human judgments.
**Decision:** **P2/defer; not a core representation.**

### 7.13 Visibility graphs and topological data analysis

**Method:** convert series to graphs or delay-embedded point clouds and compare graph/topological summaries.

**Captures:** scale-free/recurrence/global dynamical properties and possibly market-wide transition structure.
**Loses/risks:** indirect relation to a trader's setup judgment; high parameter and interpretation burden; financial studies often focus on crashes/regimes rather than individual setup analogues.
**Use:** exploratory market-regime descriptor, not initial stock-shape retrieval.
**Decision:** **P2/defer.**

### 7.14 Learned representations

**Method:** TS2Vec, temporal-neighborhood coding, triplet/contrastive causal convolutions, transformers, autoencoders, or metric learning from trader pair judgments.

**Captures:** complex multichannel interactions and scalable embeddings.
**Risks:** augmentations silently define invariances; generic benchmark success does not establish trader-perceptual similarity; embeddings can learn ticker, era, volatility, or data-source artifacts; explanations are weaker.
**Use:** later challenger against the locked transparent system. The most relevant eventual use is learning a similarity metric from explicit trader pairwise judgments—not predicting returns.
**Decision:** **P2. It must beat transparent retrieval on held-out human similarity judgments and stability tests.**

Random convolution methods such as ROCKET/MiniROCKET are strong classification transforms but do not automatically provide a meaningful nearest-neighbor geometry. Treat them as classifier/embedding benchmarks, not assumed similarity metrics.

## 8. Recommended representation portfolio

### View A — dense multiscale trajectory

Produce synchronized tensors at 21, 63, 126, and 252 sessions plus 104 weekly bars. Channel groups are independently normalized so price, volume, and market series cannot dominate by units.

Preserve both:

- paths normalized to the episode start, which express development; and
- local causal scale channels, which preserve whether moves were economically large or small.

### View B — structural swing/event skeleton

At several prominence thresholds, extract events with:

```text
event_type
start_offset, end_offset, duration
signed_price_change, max_excursion, retracement
slope, curvature/acceleration proxy
range and volatility before/during/after
relative volume and signed-volume behavior
stock-minus-market and stock-minus-sector movement
gap/shock anatomy
```

Add relations between adjacent/nonadjacent events:

```text
pullback_2_depth / pullback_1_depth
pullback_2_duration / pullback_1_duration
recovery_strength / preceding_decline
volume_during_pullback / volume_during_advance
recent_range / prior_range
distance_to_prior_high
```

This is where VCP-like contraction emerges naturally without a `vcp=true` rule.

### View C — time-scale descriptor

Run wavelet/scattering experiments separately on:

- close-to-close returns;
- true-range percentage;
- relative volume;
- relative-strength returns;
- benchmark returns.

Do not concatenate thousands of coefficients blindly. Compare compact energy/modulation summaries at predeclared scales against the baseline.

### View D — context trajectory

Keep the market/sector/breadth trajectory as its own similarity component. Produce separate output scores for:

- stock morphology;
- participation/volume;
- relative strength;
- broad-market context;
- sector context;
- structural skeleton;
- timing/warp.

Initially use Pareto or rule-based re-ranking rather than hiding these in one learned scalar.

## 9. Retrieval architecture

```text
Query cutoff
    │
    ├── Resolve point-in-time eligible historical episodes
    │       ├── candidate cutoff < query cutoff
    │       ├── sufficient history and quality
    │       ├── liquidity/security-type policy
    │       └── temporal exclusion around same security/query
    │
    ├── Stage 1: fast candidate generation
    │       ├── multiresolution PAA/raw-path vector
    │       ├── normalized Euclidean/correlation
    │       └── approximate-neighbor index or exact batched search
    │
    ├── Stage 2: structural and elastic re-ranking
    │       ├── constrained multivariate DTW/shape distance
    │       ├── swing/event edit distance
    │       ├── shock and volume distance
    │       └── retain per-view scores and alignment path
    │
    ├── Stage 3: context treatment
    │       ├── show stock-shape-only neighbors
    │       ├── show context-matched neighbors
    │       └── optionally filter/re-rank by market and sector similarity
    │
    └── Diversify results
            ├── suppress overlapping windows
            ├── cap same-security duplicates
            └── show successful and failed future paths without selection
```

### 9.1 Why context should remain separable

A trader may ask either:

1. “Has this stock shape occurred before anywhere?” or
2. “Has it occurred in the same kind of market?”

If market context is fused too strongly at candidate generation, the system cannot answer the first question or measure how outcomes change by regime. Therefore retrieve on stock development first, then expose context-matched results and conditional outcome slices.

### 9.2 Candidate exclusion rules

- Candidate cutoff must be strictly earlier than the simulated query cutoff.
- A candidate's outcome at horizon `h` is usable only when `candidate_cutoff + h <= query_cutoff`; otherwise that outcome is censored at the information boundary.
- Exclude windows overlapping the query episode.
- For the same security, use a minimum separation such as one full setup window.
- Collapse overlapping historical matches into one episode family.
- Do not let one long-lived stock dominate the top `k`.
- Keep delisted securities and failed formations.

## 10. Representation benchmark: how “best” will be decided

No paper can tell us which representation best matches this domain. We need a controlled benchmark.

### 10.1 Benchmark query set

Create a stratified set of approximately 300–500 cutoff charts containing:

- known trader examples, with the chart ending before the celebrated outcome;
- random liquid stock-dates;
- failed breakouts and failed EP-like gaps;
- high- and low-volatility eras;
- bull, bear, correction, and sideways market contexts;
- different capitalization/liquidity groups;
- examples with clean and messy swing geometry.

Known examples are sanity checks and query seeds, not the training or validation universe.

### 10.2 Human similarity labels

For a blind pair of episodes, collect separate 0–4 judgments for:

- overall development similarity;
- prior-trend similarity;
- swing/base/contraction similarity;
- recent endpoint/readiness similarity;
- shock/gap similarity;
- volume/participation similarity;
- relative-strength similarity;
- market/sector-context similarity.

Also ask which of two candidates is closer to the query. Pairwise comparisons are easier and more consistent than asking for an absolute universal score.

Outcome charts must be hidden during representation labeling.

### 10.3 Synthetic invariance tests

Starting with real episodes, generate controlled variants:

- multiply nominal prices by a constant;
- multiply ordinary volume by a constant;
- apply a legitimate split transformation;
- move one interior swing by one or two sessions;
- slightly stretch/compress the time axis;
- inject one bad tick;
- deepen a contraction materially;
- reverse market-relative strength while keeping the stock path similar;
- reorder two shocks.

The representation should be invariant only to the first small set and sensitive to the economically meaningful changes.

### 10.4 Metrics

Evaluate each representation on:

- pairwise agreement with human ranking;
- precision@`k` and nDCG on human relevance;
- neighbor stability under harmless perturbations;
- sensitivity under meaningful perturbations;
- overlap/diversity of returned episodes;
- missing-data behavior;
- inspectability of why a match occurred;
- build time, index size, and query latency;
- reconstruction error for compressed views;
- stability across eras and query strata.

Do not choose a representation based on subsequent returns at this stage.

### 10.5 Acceptance gate

A representation advances only if it improves held-out human similarity ranking or materially improves speed while preserving ranking. Added mathematical sophistication without measurable retrieval improvement is rejected.

## 11. Outcome analysis after representation lock

Once representation and similarity are locked, attach a separate future table containing:

- forward close returns at 5, 10, 20, 40, 60, and 120 sessions;
- MFE and MAE over those horizons;
- time to `+1R`, `+2R`, and specified percentage targets;
- time to stop/structural invalidation;
- target-before-stop and stop-before-target;
- gap/overnight contribution;
- future realized volatility and drawdown;
- delisting/acquisition and unavailable-outcome flags.

Compare neighbor outcomes with matched controls selected on cutoff era, market regime, sector, capitalization/liquidity, and starting volatility. Report distributions and uncertainty, not only averages.

For historical simulation, the neighbor index itself must be chronological: a query at time `t` may retrieve only episodes ending before `t`, even though their outcomes are now known to the researcher.

Outcome maturity must also be chronological. For example, at a simulated query on 1 June 2015, a 60-session outcome can be shown or used only for candidates whose 60th subsequent session occurred by 1 June 2015. More recent shape matches may still be returned, but their not-yet-observable outcomes must be marked censored. This prevents the analogue explorer from indirectly knowing the future through a historical neighbor's outcome.

## 12. Phased experimental program

### Phase 0 — data and causality audit

**Deliverables**

- stable security identifier and ticker-history policy;
- active plus delisted point-in-time universe audit;
- adjusted OHLC and volume-adjustment audit;
- index/ETF/sector/breadth inventory;
- bar-quality masks and correction provenance;
- cutoff-time contract and leakage tests.

**Gate:** randomly selected episodes can be reconstructed exactly with no post-cutoff inputs.

### Phase 1 — canonical episode builder

**Deliverables**

- episode specification and source-table schema;
- causal channel calculations;
- 21/63/126/252-session and weekly views;
- availability masks;
- exact chart renderer used for human review.

**Gate:** transformation tests pass for splits, missing bars, gaps, and causal rolling windows.

### Phase 2 — transparent retrieval baseline

Compare:

1. normalized close path + Euclidean distance;
2. OHLC/candle tensor + weighted Euclidean distance;
3. price + volume;
4. price + volume + relative strength;
5. the complete dense stock view;
6. the complete view with context used only for re-ranking.

**Gate:** complete views improve blinded retrieval relevance over close-only without unstable neighbor changes.

### Phase 3 — elastic and structural representations

Implement and compare:

- constrained DTW;
- derivative DTW;
- shapeDTW-style local descriptors;
- dependent versus independent multivariate DTW;
- multithreshold swing/PIP skeleton;
- skeleton edit/relational distance;
- dense + skeleton fusion.

**Gate:** the added method improves held-out human ranking enough to justify latency and complexity. Alignment paths must be reviewable.

### Phase 4 — multiscale and symbolic challengers

Compare:

- wavelet/scattering descriptors;
- SAX/SFA/ABBA strings;
- Matrix Profile/mSTAMP motif baselines;
- change-point state sequences.

**Gate:** keep only components that add retrieval relevance, stability, or scale beyond Phase 3.

### Phase 5 — representation benchmark and weight lock

- complete blind trader annotations;
- estimate inter-rater agreement;
- choose channel normalization and weights without outcomes;
- freeze representation version 1;
- publish ablations and rejected methods.

### Phase 6 — scaled historical index and analogue explorer

**Interface output**

- query chart and exact cutoff;
- top analogues with synchronized overlays;
- per-view similarity scores;
- swing correspondence and DTW alignment;
- stock-shape-only versus context-matched result tabs;
- subsequent-path fan chart beginning after the cutoff;
- outcomes, matched controls, and uncertainty;
- filters for era, sector, liquidity, and market regime.

### Phase 7 — discovery and optional learned similarity

Only after retrieval works:

- cluster episode neighborhoods;
- discover motifs/discords and recurring setup families;
- learn metric weights from held-out trader pair judgments;
- benchmark TS2Vec/triplet/transformer embeddings;
- test whether learned embeddings add value without erasing interpretability.

This phase discovers families such as “VCP-like” or “EP-like” after the fact. It does not begin with those labels.

## 13. Priority experiment table

| ID | Question | Comparison | Priority |
|---|---|---|---|
| E0 | Can every query be reconstructed causally? | canonical data versus rendered chart | P0 |
| E1 | How much does close-only miss? | close versus OHLC candle mechanics | P0 |
| E2 | Does volume improve human similarity? | price versus price+volume | P0 |
| E3 | Does RS improve retrieval? | stock-only versus stock+benchmark/sector relative paths | P0 |
| E4 | How much tempo flexibility is desirable? | Euclidean versus constrained DTW bands | P0 |
| E5 | Does structure add beyond dense paths? | dense versus skeleton versus fusion | P0 |
| E6 | Should market context filter or re-rank? | early fusion versus late fusion versus separate result sets | P0 |
| E7 | Do wavelets capture useful contraction/shock information? | fusion with/without compact wavelet descriptor | P1 |
| E8 | Can symbolic methods accelerate search safely? | PAA/SAX/SFA/ABBA candidate recall | P1 |
| E9 | Does Matrix Profile find meaningful unnamed motifs? | mSTAMP versus retrieval neighborhoods | P1 |
| E10 | Do signatures capture price-volume-market interaction? | transparent fusion versus signature augmentation | P2 |
| E11 | Can learned similarity beat transparent similarity? | held-out human judgments and stability tests | P2 |
| E12 | Do analogue outcomes differ from matched controls? | chronologically valid neighbor cohorts | after lock |

Every trial, parameter set, and rejection must be logged. This research ledger is necessary because repeated representation and weight searches create the same data-snooping problem as repeated trading-rule backtests.

## 14. Recommended version-one boundary

Version one should include:

- after-close daily episodes;
- US common stocks across NASDAQ, NYSE, and NYSE American, active and delisted, if the data entitlement supports them; a NASDAQ-only proof of concept must be labeled as such;
- 2000 onward where the data passes quality checks;
- 21/63/126/252 daily and 104-week views;
- adjusted OHLC candle mechanics;
- causal shock, relative-volume, volatility, and compression trajectories;
- QQQ/IXIC, SPY, IWM, sector, relative-strength, and available breadth context;
- dense fixed-vector candidate search;
- multithreshold swing skeleton;
- constrained multivariate DTW re-ranking;
- per-component similarity explanations;
- future distribution display only after the cutoff.

Version one should not require:

- predicting returns with a trained model;
- naming patterns;
- chart-image CNNs;
- transformers;
- path signatures;
- topological features;
- news/fundamental coverage;
- automatic trade execution.

Those remain testable extensions, not architectural dependencies.

## 15. Repository-specific implications

The local NASDAQ work already contains a useful foundation: approximately 11,871 active and delisted symbols were used in prior experiments, data begins around 2000, and an IXIC series exists on the `nasdaq` branch. The same logs also document repeated data artifacts, delisting-handling bugs, and strategy results inflated by bad symbols. Therefore data quality must be part of the representation, not a preprocessing footnote.

NASDAQ-listed stocks are not the entire US stock market. If the intended question is about all US stocks—as the trader examples imply—the production universe should eventually include NYSE and NYSE American common stocks as well. A NASDAQ-only first benchmark is acceptable for speed, but its neighbors and conclusions must not be described as the complete US historical universe.

Specific required changes before analogue work:

1. Audit ticker reuse and introduce a stable `security_id`.
2. Verify that active/delisted membership and security type are correct at each cutoff.
3. Audit whether historical volume is adjusted consistently with adjusted OHLC.
4. Preserve raw bars and adjustment factors rather than only the validated transformed schema.
5. Add QQQ, SPY, IWM, sector/industry proxies, and point-in-time breadth; IXIC alone is insufficient context.
6. Preserve bar-level quality flags instead of only excluding entire tickers.
7. Separate episode inputs from future outcomes at the storage/API boundary.
8. Reuse current named features as explanation metadata and ablation channels, not as the search definition.

The current scalar feature approach in `src/ml/features/` is still useful for annotations such as “ATR contraction,” “RS near high,” or “large gap.” It cannot replace the ordered path because it loses when and how those states developed.

## 16. Proposed storage contracts

### Source bars

```text
security_id, session, raw_ohlcv, adjusted_ohlcv,
adjustment_factor, source, quality_flags, snapshot_version
```

### Context bars

```text
context_id, session, ohlcv, breadth_fields,
source, quality_flags, snapshot_version
```

### Episode manifest

```text
episode_id, security_id, cutoff_session, cutoff_type,
available_history, universe_flags, representation_version,
source_snapshot_version
```

### Representation registry

```text
episode_id, view_name, view_version, vector_or_object_location,
channel_mask, normalization_parameters, build_hash
```

### Outcomes

```text
episode_id, horizon, forward_return, mfe, mae,
target_first, stop_first, event_flags, outcome_quality
```

The representation builder must not have read access to the outcome table in its normal execution path.

## 17. Review decisions

The defaults recommended for approval are:

1. **Primary cutoff:** after close.
2. **Primary setup horizons:** 126 and 252 sessions, with 21/63-day endpoint emphasis and 104 weekly bars of structural context.
3. **Similarity output:** separate shape, volume, RS, structure, market, sector, and warp scores.
4. **Context policy:** late fusion; show both shape-only and context-matched neighbors.
5. **Initial matcher:** fixed multiscale vector candidate search plus constrained DTW and swing-skeleton re-ranking.
6. **Ground truth for representation:** blinded trader similarity judgments, not future return.
7. **Discovery timing:** only after representation version 1 is locked.

## 18. Primary research map

### Financial chart-pattern computation

- Lo, Mamaysky, and Wang, [Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation](https://www.nber.org/papers/w7613) (2000): nonparametric smoothing, geometric extrema, and conditional return distributions on US equities.
- Wan and Si, [A formal approach to chart patterns classification in financial time series](https://doi.org/10.1016/j.ins.2017.05.028) (2017): formal first-order specifications for 53 named chart patterns; useful as a rule-based contrast to analogue retrieval.
- Tsinaslanidis, [Subsequence dynamic time warping for charting: Bullish and bearish class predictions for NYSE stocks](https://doi.org/10.1016/j.eswa.2017.10.055) (2018): historical price-volume subsequence matching with DTW variants.
- Tsinaslanidis and Guijarro, [What makes trading strategies based on chart pattern recognition profitable?](https://onlinelibrary.wiley.com/doi/10.1111/exsy.12596) (2021): generic historical pattern recognition and trading evaluation, with parameter/cost concerns.
- Han et al., [A pattern representation of stock time series based on DTW](https://doi.org/10.1016/j.physa.2020.124161) (2020): DTW-derived representation intended to preserve stock-series morphology.

### Classical, symbolic, and structural representations

- Chung, Fu, Luk, and Ng, [Flexible time series pattern matching based on perceptually important points](https://research.polyu.edu.hk/en/publications/flexible-time-series-pattern-matching-based-on-perceptually-impor/) (2001): landmark/PIP representation for flexible matching.
- Lin, Keogh, Wei, and Lonardi, [Experiencing SAX: a novel symbolic representation of time series](https://www.cs.ucr.edu/~stelo/papers/DMKD07b.pdf) (2007): indexable symbolic dimensionality reduction.
- Ding et al., [Querying and Mining of Time Series Data: Experimental Comparison of Representations and Distance Measures](https://www.cs.ucr.edu/~eamonn/vldb_08_Experimental_comparison_time_series.pdf) (2008): broad empirical comparison of representation and elastic-distance claims.
- Schäfer and Högqvist, [SFA: A Symbolic Fourier Approximation and Index for Similarity Search](https://www.openproceedings.org/2012/conf/edbt/SchaferH12.pdf) (2012): compact Fourier-derived symbolic indexing.
- Elsworth and Güttel, [ABBA: adaptive Brownian bridge-based symbolic aggregation of time series](https://link.springer.com/article/10.1007/s10618-020-00689-6) (2020): adaptive polygonal compression preserving ordered shape.
- Bandt and Pompe, [Permutation Entropy: A Natural Complexity Measure for Time Series](https://doi.org/10.1103/PhysRevLett.88.174102) (2002): ordinal pattern representation robust to monotonic transformations and noise.
- Batista, Wang, and Keogh, [A Complexity-Invariant Distance Measure for Time Series](https://epubs.siam.org/doi/10.1137/1.9781611972818.60) (2011): correction for the tendency of complex paths to appear farther apart than simple ones.

### Elastic alignment and shape

- Keogh and Pazzani, [Derivative Dynamic Time Warping](https://www.cs.ucr.edu/~eamonn/sdm01.pdf) (2001): alignment based on local derivatives rather than only raw values.
- Marteau, [Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching](https://pubmed.ncbi.nlm.nih.gov/19110495/) (2009): edit-based elastic metric with explicit stiffness/time treatment.
- Shokoohi-Yekta et al., [Generalizing DTW to the multi-dimensional case requires an adaptive approach](https://pmc.ncbi.nlm.nih.gov/articles/PMC5668684/) (2017): dependent and independent multivariate warping are not equivalent.
- Zhao and Itti, [shapeDTW: Shape Dynamic Time Warping](https://doi.org/10.1016/j.patcog.2017.09.020) (2018): local shape descriptors before DTW alignment.
- Cuturi and Blondel, [Soft-DTW: a Differentiable Loss Function for Time-Series](https://arxiv.org/abs/1703.01541) (2017): differentiable softened alignment, most relevant to later metric learning.
- Srivastava et al., [Registration of Functional Data Using Fisher-Rao Metric](https://arxiv.org/abs/1103.3817) (2011): principled separation of amplitude and phase variation via elastic functional geometry.

### Multiscale and motif discovery

- Mallat, [Group Invariant Scattering](https://arxiv.org/abs/1101.2286) (2011): wavelet scattering stable to small deformations.
- Andén and Mallat, [Deep Scattering Spectrum](https://www.di.ens.fr/~mallat/papiers/AudioIEEESP2013Scat.pdf) (2013): higher-order scattering for transient and modulation structure.
- Yeh et al., [Matrix Profile I: All Pairs Similarity Joins for Time Series](https://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf) (2016): scalable exact motif/discord/similarity-join framework.
- Yeh, Kavantzas, and Keogh, [Matrix Profile VI: Meaningful Multidimensional Motif Discovery](https://mcyeh.github.io/paper/2017_icdm_meaningful_multidimensional_motif.pdf) (2017): multidimensional motifs and channel subsets.
- Paparrizos and Gravano, [k-Shape: Efficient and Accurate Clustering of Time Series](https://www.cs.columbia.edu/~gravano/Papers/2015/sigmod2015.pdf) (2015): scalable shape clustering with normalized cross-correlation.
- Killick, Fearnhead, and Eckley, [Optimal Detection of Changepoints With a Linear Computational Cost](https://doi.org/10.1080/01621459.2012.737745) (2012): exact penalized segmentation via PELT.

### Geometric and image alternatives

- Lyons, Ni, and Oberhauser, [A feature set for streams and an application to high-frequency financial tick data](https://ora.ox.ac.uk/objects/uuid%3Acf684258-560a-41b4-bb39-efdb0ecc792e/) (2014): path signatures for financial streams.
- Fermanian, [Embedding and learning with signatures](https://doi.org/10.1016/j.csda.2020.107148) (2021): empirical study of signature embeddings and lead-lag choices.
- Wang and Oates, [Imaging Time-Series to Improve Classification and Imputation](https://www.ijcai.org/Proceedings/15/Papers/553.pdf) (2015): Gramian angular and Markov transition image encodings.
- Eckmann, Kamphorst, and Ruelle, [Recurrence Plots of Dynamical Systems](https://www.ihes.fr/~ruelle/PUBLICATIONS/%5B92%5D.pdf) (1987): recurrence representation of dynamical trajectories.
- Lacasa et al., [From time series to complex networks: The visibility graph](https://arxiv.org/abs/0810.0920) (2008): graph representation retaining selected time-series structure.
- Gidea and Katz, [Topological Data Analysis of Financial Time Series: Landscapes of Crashes](https://arxiv.org/abs/1703.04385) (2018): persistent-homology features for multidimensional financial regimes.

### Learned representation challengers

- Franceschi, Dieuleveut, and Jaggi, [Unsupervised Scalable Representation Learning for Multivariate Time Series](https://papers.nips.cc/paper/8713-unsupervised-scalable-representation-learning-for-multivariate-time-series.pdf) (2019): causal dilated convolutions with triplet loss.
- Tonekaboni, Eytan, and Goldenberg, [Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding](https://arxiv.org/abs/2106.00750) (2021): representations based on local temporal neighborhoods.
- Yue et al., [TS2Vec: Towards Universal Representation of Time Series](https://ojs.aaai.org/index.php/AAAI/article/view/20881) (2022): hierarchical contrastive representations at multiple semantic scales.
- Dempster, Petitjean, and Webb, [ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels](https://arxiv.org/abs/1910.13051) (2020): scalable random convolution transform, included as a classification/embedding challenger rather than an assumed distance.

### Research-selection controls

- Sullivan, Timmermann, and White, [Data-Snooping, Technical Trading Rule Performance, and the Bootstrap](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00163) (1999): technical-rule evaluation must account for the full set of tried rules.
- White, [A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152) (2000): repeated use of one history for model selection can manufacture apparent superiority.

## 19. Final recommendation

Do not choose between “wave,” “DTW,” “string,” or “geometry” as if one must win before implementation. Build a lossless point-in-time episode and benchmark these as complementary views.

The strongest first hypothesis is:

> A dense multiscale OHLCV-relative-strength-market trajectory, augmented by a multithreshold swing/event skeleton and compared with constrained elastic alignment, will reproduce professional chart-similarity judgments better than either hand-engineered scalar features or any single transform alone.

That hypothesis is concrete, falsifiable, interpretable, and compatible with later discovery of unnamed setup families.
