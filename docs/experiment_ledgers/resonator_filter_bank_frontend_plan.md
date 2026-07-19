# Resonator Filter-Bank Frontend — Experiment Plan

## OscNet RFB encoding + our HORN substrate

*OscNet research plan, drafted July 2026; revised after identifying the
source paper and cross-checking claims against our experiment log.
Status: **audio-first — Stages 0–5 + Sprints A–C** — learnable RFB→HORN with
**dense coupling ~0.76 clean** (≫ mel ~0.61; ≫ frozen fractal ~0.65); soft
naug owns noise; HORN≫MLP; high→low coupling **not** recovered. CSVs through
`rfb_plus_tonotopic`.
**Paper:** [`docs/paper_rfb_horn/main.pdf`](paper_rfb_horn/main.pdf)
(LaTeX in [`paper_rfb_horn/`](paper_rfb_horn/)).*

### Settled insights (ledger for later use)

Keep this block current as stages land; detail lives in the stage tables below.

| Claim | Status | Evidence |
| --- | --- | --- |
| Spectral frontend necessary in small regime (H0) | **PASS** | RFB ≫ raw / equal-freq (Stage 0a) |
| Log/cochlear structure matters vs equal-freq | **PASS** | equal-freq ~chance throughout |
| Frozen RFB alone beats mel | **FAIL** | best frozen ~0.44–0.51 vs mel ~0.58–0.60 |
| Time frames + HORN help frozen RFB | **partial** | frames→HORN 0.47 > pooled; still < mel |
| Learnable `{ω,Q}` + frames→HORN beats mel/conv (H3/H1) | **PASS** | dense HORN **~0.76@10k** ≫ mel ~0.61; HORN≫MLP |
| HORN-as-brain (not just RFB features) (H5) | **PASS** | scale: dense HORN ~0.76 vs MLP ~0.36 |
| Transfer beyond digit words | **PASS** | core10 learn→HORN 0.61 ≫ mel 0.47 |
| Additive-noise robustness | **PASS at scale + soft aug** | naug0.4: white 0.63 ≫ mel 0.42; clean still 0.73 |
| Band-collapse under learning | **open polish** | near-dups persist; reg=0.5 trims dups slightly, no acc gain |
| Static-image RFB / spatial DHO frontend | **PARK / fail** | Gabor ~30%; spatial resonator ~16%; keep HORN-as-brain for vision. Experiment package removed; core Gabor/encoder APIs kept. |
| Phase readout load-bearing (Stage 3) | **FAIL — amp wins** | phase-only ~0.18; both 0.55 < amp 0.65 |
| Bank nonlinearity (Stage 4) | **FAIL — linear enough** | drive_tanh ≈ linear; envelope hurts; AGC chance |
| High→low coupling streamlining (Stage 5 mech.) | **FAIL / not seen** | Stage 5 fractal≈0.9; Sprint C dense/tonotopic≈1.0–1.06 |
| Learnable dense HORN coupling (Sprint C) | **PASS (acc)** / **FAIL (mech)** | dense ~0.76 ≫ frozen fractal ~0.65; tonotopic prior ≈ dense, no high→low |
| Fair sequential controls (`rfb_plus_controls`) | **nuance** | mel→GRU 0.88 / mel→HORN 0.86 ≫ RFB→HORN 0.76 on clean; HORN ~7k≈GRU ~19k; RFB wins white-noise |
| Iso-param GRU (`rfb_plus_isoparam`) | **done** | mel: HORN 0.863 ≈ GRU(~7k) 0.873; RFB: GRU(~7k) 0.845 ≫ HORN 0.764; RFB wins white-noise |
| RFB 4096-sample subsample | **disclosed** | kept for compute; paper protocol + limitations state mel is full-rate, RFB is 4096-subsampled |

**Working recipe (audio):** learnable RFB, `feature_mode=frames`, `head_kind=horn`,
`horn_coupling_kind=dense` (or `dense_tonotopic`; same acc), `readout=amplitude`,
`nonlinearity=none`, ~32 bands, Q=4, 16 kHz, `horn_steps=16`,
`collapse_reg_weight≈0.1`, lr≈3e-3.
**Scale default:** ~10k train, `head_hidden_dim≈64`.
**Robust default:** `train_level_aug_db≈12`, `train_noise_aug_prob≈0.4`
(keeps clean ≫ mel while owning white/pink/band).
**Note:** Sprint B’s ~0.77 “fractal” arm had an accidentally trainable W; with
truly frozen fractal structure (Sprint C) clean drops to ~0.65 — dense W is
what recovers ~0.76.

**Naming.** In code and OscNet APIs we call this a **resonator filter bank
(RFB)** / `ResonatorBank`. We cite *AudioPrism* (Pietras et al.) only as
external prior art in this research note — we do not use their product name
in identifiers, modules, or CLI flags.

---

## 0. One-paragraph summary

Every oscillator experiment in this repo so far has used oscillators as the
**recurrent substrate** — a coupled field that settles, denoises, and is read
out (see `docs/what_the_physics_prior_buys.md`). This plan isolates the
complementary role: oscillators as an **input transform**. The concrete
target is the architecture in *AudioPrism* (Pietras, Carvalho, Dubinin,
Ferrand, Singer, Effenberger; ICNCE 2026) — Felix Effenberger's group's
system — which stacks a **PRISM** (bank of heavily damped, log-spaced,
uncoupled resonators performing cochlear-like tonotopic decomposition) in
front of a **HORN** (recurrently coupled oscillators for multi-scale
integration). PRISM parameters are analytic/frozen (cochlear gains), not
learned. On spoken-digit classification in the small (~3k–42k) regime they
report a decisive ablation: without PRISM the baseline sits near chance; with
it, ~60–75% accuracy. We have never built that frontend. This document states
what of our own logs does and does not support the idea, lays out falsifiable
hypotheses with matched controls, and sequences cheap home-domain probes
before CIFAR transfer.

---

## 1. Motivation — what is established, what is not

### 1.1 External lead: AudioPrism (cite, don't reconstruct)

AudioPrism = PRISM + HORN (Pietras, Carvalho, Dubinin, Ferrand, Singer,
Effenberger; ICNCE 2026):

| Stage | Role | Dynamics |
| --- | --- | --- |
| **PRISM** | Cochlear-like frontend | Heavily damped harmonic oscillators, log-spaced freqs spanning the speech band; **parameters configured via analytic gain functions** (frozen, not learned); tonotopic decomposition |
| **HORN** | Recurrent integrator | Coupled oscillators with heterogeneous δ–γ frequencies; trainable coupling matrix for multi-scale temporal integration |

Settled from the paper (upgrade former open questions to knowns):

1. **PRISM is purely analytic / frozen.** Replication of their architecture
   means the Stage 1 frozen-reservoir arm, not a learned bank. Learnable
   `{ω, Q}` (our Stage 2 / H3) is an explicit *extension beyond* AudioPrism;
   a null there does not contradict the paper.
2. **Necessity is scoped to the small / parameter-matched regime.** Their
   figure shows HORN_50 (~3k), HORN100 (~11k), HORN200 (~42k) with PRISM
   reaching ~60–75% test accuracy, while no-PRISM HORNs and a "more raw
   inputs" control stay near chance (~10%). A large unconstrained net on
   Speech Commands digits would not fail — H0 must be read as *in this
   parameter-efficient regime*, which is also their selling point.
3. **Post-training mechanism:** recurrent weights streamline information
   high-frequency → low-frequency nodes — a concrete Stage 5 diagnostic
   target, not just an accuracy number.

Primary claim in-domain: **learnability / parameter-efficient multi-scale
representation**, not robustness.

Source:
`https://iffindico.fz-juelich.de/event/28/contributions/710/attachments/386/455/ICNCE_2026_AudioPrism.pdf`

Related substrate: Effenberger et al., PNAS 122(4), e2412830122, 2025.
DHO unit (their Fig. B): `ẍ + 2γ ẋ + ω² x = α I(t)`.

### 1.2 Our logs — related fragments, not confirmations of the prism principle

Earlier drafts treated our multimode and resonant-readout wins as miniature
PRISMs. That overstated the evidence.

1. **Multimode HORN helped, but not (proven) via frequency tuning.**
   256 sites × 1 mode → generated-class acc 0.53 / attractor 0.50; 256 sites
   × 2 modes → 0.77 / 0.73 at matched total state
   (`### CIFAR RGB Multimode HORN Probe`, seed 23, 20 epochs — a hypothesis
   generator by the project's own rigor norms, not a confirmed result). The
   Multimode Carrier Active-Ingredient Ablation (2026-07-15) then showed that
   for *settling fill-in*, the **equal-frequency** variant (1.0/1.0) was best
   and the widest frequency split (0.5/1.5) was worst — active ingredient =
   **per-site capacity and mode coupling**, not spectral separation
   (`docs/experiment_report.md`). Caveat: that ablation measured fill-in, not
   generation accuracy, so frequency-tuning-for-generation remains open —
   which this plan will settle with an equal-frequency control, not treat as
   settled fact.

2. **Resonant filter-bank readout** improved the semantic/diversity frontier
   at strength 0.05 (~675 params, single seed) but was strength-fragile
   (`resonant010` worse than baseline) and failed to transfer to the 512-site
   field. Useful fragment, not an independent confirmation of input-side
   PRISM.

3. **Frequency objective probe** moved high-freq/edge ratios toward real
   images but traded away class accuracy and mostly added ringing — evidence
   that frequency structure is under-exploited, not that a resonator frontend
   works.

**Honest through-line:** AudioPrism is a strong external lead for a
PRISM→HORN stack in audio. Our logs show that structured oscillator capacity
and resonant *observables* can help, but they do **not** already confirm that
tuned-frequency *input decomposition* is the mechanism. The equal-frequency
bank control below is mandatory precisely because of this gap.

### 1.3 Scientific neighbors (novelty honesty)

- **Scattering transforms** — fixed wavelet/Gabor frontends for vision.
- **SincNet / LEAF** — learnable center-frequency / bandwidth frontends for
  audio (our Stage 2 / H3 is essentially their design — an extension beyond
  AudioPrism's frozen analytic PRISM; parameterize band edges, not raw
  filter taps).
- **Constant-Q / gammatone / cochlear filter banks** — classical DSP that a
  linear PRISM must beat or match.
- **Physical reservoir computing** — frozen dynamical frontend + trained
  readout. AudioPrism's own PRISM is already this; our Stage 1 frozen arm
  mirrors it.
- **Carvalho et al., Phys. Rev. Applied 24, 064055 (2025)** (arXiv:2509.04064;
  AudioPrism ref. [3]) — analog-electronic HORN. Digital-twin readout agreed
  with the analog system at only **~28%** due to precision mismatch; full
  performance recovered when the analog HORN was used as a **reservoir with a
  re-trained linear readout**. Direct external evidence from Felix's group
  for (a) the frozen-frontend / reservoir arm and (b) the detuning /
  component-tolerance axis in Stage 1 — they already published the failure
  mode that arm probes. Load-bearing for H2's analog relevance.

---

## 2. The concept in precise terms

A single driven, damped oscillator (AudioPrism DHO) obeys

```text
x'' + 2 * gamma * x' + omega_k^2 * x = alpha * I(t)
```

Steady-state response amplitude peaks near `omega_k / 2π`, with bandwidth set
by damping (`Q ~ omega_k / (2 gamma)` in this convention). An array
`{omega_k}` driven by the same input yields band amplitudes — a
spectrogram-like feature map.

Key properties (corrected):

- **Uncoupled and parallel.** No oscillator–oscillator interaction in the
  baseline PRISM. (Weak tonotopic coupling is a later arm.)
- **Complete coverage with controlled overlap**, not "non-overlapping ≈
  orthogonal." Truly gapped bands *lose* information between centers. Want a
  frame: full coverage, minimal redundancy. Constant-Q vs linear spacing is a
  real design choice (AudioPrism uses log-spaced / cochlear).
- **Two Q regimes.** PRISM uses **heavy damping** (fast analysis, little
  memory). HORN uses lighter damping / recurrent coupling (integration,
  fading memory). Do not conflate them.
- **Linear closed form available.** A driven linear damped resonator is a
  second-order IIR (biquad). Stage 0–1 should use closed-form / IIR
  evaluation, not unrolled ODE steps. Reserve time-stepped integration for
  nonlinear Stage 4.
- **Feedforward analysis, not recurrent memory.** Encoder role. The full
  AudioPrism claim is **PRISM → HORN**, not PRISM → pixel decoder alone.

Readout math (pin down before coding): band amplitude from quadrature,
`A² = x² + (ẋ/ω)²` (or equivalent analytic envelope); specify integration /
averaging window, transient discard, and the discrete-time stability /
Nyquist bound so log-spaced high-ω resonators remain well-posed.

---

## 3. Central hypotheses (falsifiable)

| ID | Claim | Falsifier |
| --- | --- | --- |
| **H0** | In the **small / parameter-matched regime**, *some* spectral frontend is necessary for an oscillatory stack to learn (AudioPrism-style necessity; their no-PRISM HORNs stay near chance) | Mel/STFT/Gabor also unlocks learning equally at matched capacity → not oscillator-specific. (A large unconstrained net succeeding does *not* falsify H0.) |
| **H1** | A tuned resonator bank matches or beats a **matched classical filter bank** (mel / STFT / Gabor) and a matched conv stem on on-nominal quality | Classical bank or conv wins → ODE/resonance machinery adds nothing over textbook spectral features |
| **H2** | Resonator frontend degrades more gracefully than a matched learned/regularized stem under **held-out input** corruptions — especially structured/severe and band-limited stressors; *not* expected to win on mild distributed noise. Detuning `{ω,Q}` (Carvalho analog mismatch) is in-scope. | Conv/mel/regularized-conv equal or better on the predicted families |
| **H3** | *(Extension beyond AudioPrism.)* Learnable `{ω_k, Q_k}` (band-edge parameterization, SincNet-style) beats a fixed cochlear/log-spaced bank | Learning doesn't help, or bands collapse — a null-vs-paper result, not a contradiction of AudioPrism |
| **H4** | Cochlear-style nonlinearity / gain control beats a linear bank | Linear enough |
| **H5** | Full **PRISM → HORN** beats PRISM → MLP/decoder (frontend value is architectural, not just feature cosmetic); trained HORN shows high→low frequency information routing | Frontend helps any head equally, or no high→low routing → mechanism did not transfer |

**H2 mechanism (sharpened, not overclaimed).** A *fixed linear* resonator bank
is a linear operator — so is a frozen conv. The project's "bounded contractive
dynamics fail soft" result was about *nonlinear recurrent settling*; it does
not transfer to a linear encoder by physics, only by analogy. The testable
version: the bank is a **smooth, low-dimensional, band-structured
parameterization** that cannot express sharp data-specific features, hence
flatter degradation. That makes the regularized / spectrally-constrained conv
stem the direct skeptic control (analog of the regularized StateMLP that made
the OOD crossover credible).

Hybrid Frontier already showed distributed held-out corruptions (Gaussian,
salt-and-pepper) stayed with free-form arms; oscillator wins lived in
structure-destroying corruption at severity. **H2 predicts the same shape on
the encoder side:** help on structured/severe and band-limited stressors;
likely no win on mild distributed noise. A mixed result must be read against
that prediction, not reinterpreted after the fact.

**Analog relevance (Carvalho et al. 2025):** their analog HORN matched the
digital twin's readout at only ~28% until used as a reservoir with a
re-trained linear readout. Detuning / precision stress in Stage 1 is therefore
probing a failure mode Felix's group already measured, not a speculative
hardware story.

---

## 4. Honest caveats

1. **Audio is the home domain; images are a transfer test.** AudioPrism's win
   is about temporal multi-scale structure speech has natively. On static
   images, "frequency-tuned filters" collapses toward a spatial-frequency /
   Gabor / scattering frontend — close to what CNN first layers already learn
   and to a well-studied literature. Treat CIFAR stages as transfer, not
   replication.
2. **Task mismatch.** They classify spoken digits; we mostly generate /
   recover CIFAR. Different fitness functions. Stage 0 uses their task shape.
3. **Win over FFT/mel is conditional** on (a) learnable tuning, (b)
   nonlinearity / cochlear gain, and/or (c) analog efficiency / detuning
   robustness. Fixed linear bank alone is "just a slower filter bank."
4. **Reservoir vs end-to-end** are different models; test both.
5. **"What is the signal" for images is make-or-break**, not a deferrable
   footnote — see Stage 0b.

---

## 5. Staged experiment plan

Default CIFAR budget when used: ~evening of A10G, 4 seeds, seed-paired
comparisons, 40 epochs, full robustness harness
(`compute_generator_robustness_metrics`,
`--robustness-eval-heldout-corruptions`, OOD occlusion). Report **trainable
parameter counts** per arm (frozen frontends change effective capacity).

Quantitative go/no-go template (reuse across stages): within X% of the best
matched baseline on-nominal **and** better on ≥2 predicted held-out families,
4 seeds, seed-paired — matching the repo's rigor norms. Set X at Stage 0
write-up time from the audio baselines (don't invent a fake threshold now).

### Stage 0a — Audio home-domain probe (H0, H1) — cheap, decisive

Test the PRISM claim where AudioPrism stated it. Spoken-digit / Speech
Commands subset; **small / parameter-matched** classifier or HORN head
(target their ~3k–42k regime, not a large unconstrained net).
**Run multi-seed / Speech Commands batteries on Modal** via
`scripts/modal_audio_digit.py` (not localhost). Local: unit tests + tiny
smokes only.

Arms (matched feature dim / parameter budget where applicable):

- raw waveform → head (no frontend; their chance-level baseline);
- "more raw inputs" / wider unstructured input at matched dim (their
  HORN100-with-30-inputs style control — capacity without structure);
- mel / STFT / classical filter bank → head;
- learned 1-D conv frontend → head;
- **fixed linear resonator bank** (heavy damping, log-spaced, **frozen
  analytic cochlear gains** as in the paper; closed-form IIR, not ODE
  unroll) → head — the actual AudioPrism replication arm;
- **equal-frequency resonator bank** (all ω identical, matched everything
  else) — kills the prism interpretation if it ties the tuned bank;
- optional: random-frequency bank.

**Go/no-go:** tuned resonator must beat no-frontend and unstructured-input
controls at matched small capacity (H0) and match/beat mel/STFT (H1). If
equal-frequency ties tuned, retire "prism" and treat as capacity. If the
bank can't match FFT/mel on audio, **stop** — image stages are moot for the
AudioPrism claim.

#### Stage 0a results (Modal, 2026-07-19)

Launch: `OSCNET_MODAL_MAX_CONTAINERS=10 modal run scripts/modal_audio_digit.py
--sweep-preset stage0a_speech_commands` (and `stage0a_synthetic`).
CSVs: `outputs/analysis/modal_audio_digit_stage0a_*.csv`.

Speech Commands digits (8 kHz, 1.0 s, 16 bands, 40 epochs, 4 seeds;
frozen-frontend arms ≈1.3k trainable head params):

| frontend | mean eval acc |
| --- | ---: |
| conv1d (learned) | 0.598 |
| stft | 0.547 |
| mel | 0.529 |
| **resonator (RFB)** | **0.453** |
| resonator_equal | 0.162 |
| raw | 0.134 |
| raw_wide | 0.126 |

- **H0 — PASS.** Tuned RFB ≫ raw / raw_wide / equal-frequency at matched
  small capacity. Frequency structure is load-bearing (not mere capacity).
- **H1 — PARTIAL.** RFB trails mel/STFT by ~7–9 pp and learned conv by
  ~15 pp. Not a hard AudioPrism failure (still far above chance / unstructured
  controls), but we do **not** yet match classical frontends under this
  probe. Proceed to Stage 1 as a transfer test with that gap acknowledged;
  revisit Q / band count / unit-peak schedule if CIFAR attribution is weak.
- Synthetic battery is **inconclusive** (ceiling: most arms ≈1.0; equal-freq
  0.61) — keep Speech Commands as the Stage 0a decision surface.

#### `rfb_tune` results (Modal, 2026-07-19)

Hyperparam battery (Speech Commands, 2 seeds, head=64). CSV:
`outputs/analysis/modal_audio_digit_rfb_tune.csv`.

| arm | mean acc |
| --- | ---: |
| conv1d b32 8 kHz | 0.608 |
| mel b32 **16 kHz** | 0.596 |
| stft b48 8 kHz | 0.541 |
| mel b32/48 8 kHz | ~0.52–0.53 |
| **best RFB** b32 Q4 **16 kHz** | **0.443** |
| RFB variants (24–48 bands, Q 2–12) | 0.38–0.44 |
| resonator_equal b32 | 0.136 |

**Takeaway:** more bands / Q / 16 kHz does **not** close H1. Structure still
matters (≫ equal-freq). Next lever: **stop whole-clip pooling** — feed
time-varying RFB frames into HORN (`horn_stack`).

#### `horn_stack` results (Modal, 2026-07-19)

Framed RFB→HORN vs pooled controls (32 bands, Q=4, 16 kHz, 2 seeds). CSV:
`outputs/analysis/modal_audio_digit_horn_stack.csv`.

| arm | mean acc |
| --- | ---: |
| conv1d → MLP | 0.609 |
| mel → HORN (pooled) | 0.589 |
| stft → MLP | 0.586 |
| mel → MLP | 0.582 |
| **RFB → HORN (frames)** | **0.470** |
| RFB → HORN (pooled) | ~0.42–0.43 |
| RFB → MLP (pooled) | 0.426 |
| equal-freq RFB → HORN (frames) | 0.220 |
| RFB → MLP (mean of frames) | 0.153 |

**Takeaway:** time-varying RFB frames + HORN lifts frozen RFB but still
trailed mel — until Stage 2 learning (below).

#### `rfb_learn` results (Modal, 2026-07-19) — Stage 2 / H3

Learnable `{ω,Q}` via **freq-domain DHO power response** (stable grads; IIR
BPTT NaN'd) + residual tanh parameterization around cochlear init. CSV:
`outputs/analysis/modal_audio_digit_rfb_learn.csv`.

| arm | mean acc |
| --- | ---: |
| **RFB learn → HORN (frames)** | **0.641** |
| mel → MLP | 0.602 |
| conv1d → MLP | 0.588 |
| RFB frozen → HORN (frames) | 0.511 |
| RFB frozen → MLP (pooled) | 0.430 |
| RFB learn → MLP (pooled) | 0.394 |
| equal-freq → HORN (frames) | 0.233 |

**Takeaway:** H3 **PASS** under this probe — learning bands helps a lot, but
only when paired with **frames→HORN** (learn+pooled MLP actually regresses).
Structure still required (≫ equal-freq). Band-collapse diagnostic shows
several near-duplicate frequencies — addressed by `band_spacing_regularizer`
in the confirm sweep. **H1 cleared** vs mel/conv for the learnable
AudioPrism-style stack.

#### `audioprism_confirm` (Modal, 2026-07-19) — full claim package

Preset: `scripts/modal_audio_digit.py --sweep-preset audioprism_confirm`.
CSV: `outputs/analysis/modal_audio_digit_audioprism_confirm.csv` (4 seeds).

| arm | clean | SNR 10 | SNR 0 |
| --- | ---: | ---: | ---: |
| **digits: RFB learn → HORN (frames, reg)** | **0.648** | 0.284 | 0.215 |
| digits: conv1d → MLP | 0.601 | 0.339 | 0.164 |
| digits: mel → MLP | 0.583 | **0.396** | 0.238 |
| digits: RFB frozen → HORN (frames) | 0.515 | 0.362 | 0.252 |
| digits: RFB learn → MLP h=96 (param-matched) | 0.328 | 0.241 | 0.182 |
| digits: RFB learn → MLP h=48 (equal-width) | 0.315 | 0.235 | 0.149 |
| **core10: RFB learn → HORN** | **0.608** | 0.282 | 0.132 |
| core10: mel → MLP | 0.473 | 0.290 | 0.169 |

**Verdicts:**
- **Not a fluke:** 4-seed learn→HORN **0.648±0.022** still clears mel/conv.
- **Full AudioPrism claim (HORN-as-brain):** at matched size, frames→HORN
  ≫ frames→MLP (~0.65 vs ~0.33). Flat readout cannot use the RFB frames.
- **Harder words:** core10 transfer holds — learn→HORN 0.61 ≫ mel 0.47.
- **Noise:** additive white noise hurts the learnable stack more than mel at
  SNR 10; frozen RFB/mel degrade more gracefully. Noise robustness is **not**
  won yet on this stressor.
- **Collapse reg:** accuracy stable, but near-duplicate bands still appear
  (`n_near_duplicate` ~4–9) — reg weight / hinge may need a stronger
  follow-up; not blocking the claim.

### Stage 0b — Image drive definition (local prototype, before Modal) — PARKED

Decide "what is the signal" with a cheap local prototype; do not spend a full
CIFAR sweep on an ill-posed drive.

| Option | Verdict posture |
| --- | --- |
| (a) Raster / row-scan 1-D stream | Cheap audio analogy; known anisotropic pathology — diagnostic only |
| (b) **2-D spatial-frequency / Gabor-like bank** | Honest image analog; default candidate for Stage 1 |
| (c) Per-patch temporal unrolling | Only if 0a suggests temporal structure is load-bearing |

If (b) is chosen, Stage 1's scientific neighbor is explicitly "fixed
structured conv / scattering vs learned conv," and H1 must include a **fixed
Gabor/DCT** arm or the resonator adds nothing over textbook vision frontends.

### Stage 1 — Fixed linear bank vs matched controls (H0–H2) — CIFAR transfer

**1a — classification gate** (mirrors Stage 0a; **package removed** after PARK):
historical Modal preset `stage1_cifar10`. Arms: gabor, gabor_equal, dct,
frozen_random, learned_conv, raw_wide, raw.

#### Stage 1a results (Modal, 2026-07-19)

CIFAR-10 RGB, 4k/1k, 40 epochs, 2 seeds, ~16-band frontends, small MLP head.
CSV: `outputs/analysis/modal_cifar_rfb_stage1.csv`.

| frontend | mean eval acc |
| --- | ---: |
| learned_conv | 0.316 |
| **gabor** | **0.304** |
| raw_wide | 0.291 |
| frozen_random | 0.281 |
| dct | 0.264 |
| gabor_equal | 0.255 |
| raw | 0.127 |

- **H0 — weak PASS.** Tuned Gabor > equal-scale and ≫ raw; but only a thin
  edge over frozen-random / raw_wide — scale structure helps, is not a large
  effect under this small probe.
- **H1 — soft PASS vs DCT.** Gabor beats the radial-DCT classical bank; does
  not beat learned conv (expected).
- Absolute accuracies are low (~30%): intentional small-capacity gate, not an
  SOTA CIFAR classifier. Proceed to **1b** (wire into generative
  `encode_image_state`) with eyes open that image-RFB attribution is thinner
  than audio Stage 0a.

**1a′ — spatially driven resonator bank (oscillator-faithful vision):**
each coarse spatial cell pulse-drives a shared `ResonatorBank` (real DHOs,
not Gabors). Historical Modal bake-off vs Gabor: preset `spatial_vs_gabor`.

#### Stage 1a′ results (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_cifar_rfb_spatial_vs_gabor.csv` (CIFAR-10, 2 seeds).

| frontend | mean eval acc |
| --- | ---: |
| **gabor** | **0.304** |
| frozen_random | 0.281 |
| gabor_equal | 0.255 |
| row_scan (resonators) | 0.201 |
| spatial_resonator | 0.159 |
| spatial_resonator_equal | 0.147 |

**Verdict / PARK:** oscillator-faithful image frontends **lose** to Gabors by
a wide margin. Pulse-driven spatial RFB is barely above chance; row-scan
resonators are better but still well below Gabor / frozen-random. The RFB
win is tied to **native temporal signals (audio)**. For vision, keep
**HORN-as-brain** (coupled substrate); do **not** chase image-PRISM /
image-RFB further on this line. Stage 1b below is optional / deferred.

**1b — generative path (deferred):** winner of 1a/1a′ replacing or preceding
`encode_image_state` (`oscnet/models/generative/horn.py`) → state grid the
decoder/HORN expects.

**Arms (unbundle the axes — do not confound resonator vs frozen vs
structure):**

| Arm | Purpose |
| --- | --- |
| Learned conv anchor encoder (current) | Production baseline |
| Regularized / spectrally-constrained conv stem | Skeptic control for H2 |
| Frozen random conv + trained decoder | Is "frozen/reservoir" the active ingredient? |
| Fixed classical bank (Gabor / DCT), frozen | Is "structured spectral" enough without resonance/ODE? |
| Tuned resonator bank, frozen | **AudioPrism-faithful** PRISM reservoir (primary replication) |
| Tuned resonator bank, end-to-end | Extension: backprop through bank |
| **Equal-frequency resonator bank**, frozen | Capacity vs tuning (mandatory) |

**Eval additions for mechanistic H2** (extend
`_corrupt_images_heldout` in `metrics.py` beyond gaussian / salt_pepper /
stripes):

- Spectrally structured stressors: blur / low-pass, high-pass noise,
  band-limited noise, contrast shift — degradation should be legible *per
  band* if the mechanism is real.
- **Detuning stress:** jitter the bank's `{ω, Q}` (component tolerance) —
  the failure mode Carvalho et al. already measured (~28% digital-twin
  agreement until reservoir re-train); maps onto the existing weight-noise
  harness.

Corruptions applied **before** the frontend. Score on-nominal + predicted
held-out families; concede mild distributed noise unless data surprise us.

**Go/no-go:** tuned resonator beats equal-frequency and frozen-random-conv;
matches/beats fixed Gabor on-nominal (H1); beats regularized conv on ≥2
predicted structured held-out families (H2). Even H2-without-H1 is a
two-regime positive. H1-without-H2 means "decent inductive bias, no
robustness unification."

### Stage 2 — Learnable tuning (H3) — extension beyond AudioPrism

AudioPrism's PRISM is frozen-analytic; this stage asks whether learning the
bands buys anything *on top of* that result. A null (learning doesn't help)
is compatible with the paper — it would mean their analytic cochlear gains
were already sufficient.

Trainable band edges / `{ω, Q}` with **SincNet-style parameterization**
(not raw ω through unrolled dynamics — gradients w.r.t. ω are oscillatory and
local-minima-ridden). Log-spaced / cochlear init. Add **band-collapse
diagnostic** (multiple resonators converging on one ω) to metrics. Arms:
fixed Stage-1 winner vs learned; compare end-to-end learning against the
frozen reservoir baseline that mirrors the paper.

### Stage 3 — Readout content (amplitude / phase / both)

Amplitude envelope vs phase (`sin/cos`) vs both via `readout=` on the RFB
bank (IIR for frozen; complex ``H(jω)`` for phase/both; amplitude keeps the
Stage-2 log-power path). Modal:
`scripts/modal_audio_digit.py --sweep-preset rfb_stage3`.

#### `rfb_stage3` results (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_stage3.csv` (2 seeds).

| arm | clean | SNR 10 |
| --- | ---: | ---: |
| **learn → HORN, amplitude, reg=0.1** | **0.654** | 0.246 |
| learn → HORN, amplitude, reg=0.5 | 0.642 | 0.264 |
| mel → MLP | 0.602 | 0.407 |
| learn → HORN, both | 0.554 | 0.291 |
| frozen → HORN, amplitude | 0.514 | 0.354 |
| frozen → HORN, both | 0.483 | 0.376 |
| learn → HORN, **phase-only** | **0.179** | 0.133 |

**Verdict:** phase is **not** load-bearing on this probe — amplitude wins;
concatenating phase hurts; phase-only is near chance. Stronger collapse reg
(0.5) trims near-duplicates slightly without helping accuracy. Keep
`readout=amplitude`. 

### Stage 4 — Nonlinearity (H4)

Compressive drive (`drive_tanh`), envelope soft-compression (`envelope_soft`),
and time-stepped AGC damping (`agc`) vs linear bank. Modal:
`scripts/modal_audio_digit.py --sweep-preset rfb_stage4`.

#### `rfb_stage4` results (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_stage4.csv` (2 seeds).

| arm | clean | SNR 10 |
| --- | ---: | ---: |
| learn → HORN, **drive_tanh** | **0.650** | 0.254 |
| learn → HORN, **linear** | 0.642 | 0.256 |
| mel → MLP | 0.602 | 0.407 |
| learn → HORN, envelope_soft | 0.583 | 0.263 |
| frozen → HORN, linear | 0.514 | 0.354 |
| frozen → HORN, drive_tanh | 0.502 | 0.347 |
| frozen → HORN, **agc** | **0.100** | 0.100 |

**Verdict (H4 FAIL on this probe):** linear is enough. `drive_tanh` is a
wash vs linear (no SNR win either); `envelope_soft` hurts clean accuracy;
our Euler AGC arm collapsed to chance (implementation/stress failure, not a
biology win). Keep `nonlinearity=none`. Noise robustness still favors mel.

### Robustness follow-up (pre–Stage 5)

Honest cochlear-ish stress **without** rebuilding AGC: random ±dB level
augmentation in training; eval on white / pink / mid-band noise + quiet/loud
gains. Modal: `scripts/modal_audio_digit.py --sweep-preset rfb_robust`.

#### `rfb_robust` results (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_robust.csv` (2 seeds).

| arm | clean | white10 | pink10 | band10 | quiet−20 | loud+6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RFB learn→HORN | **0.642** | 0.256 | 0.443 | 0.226 | 0.201 | 0.562 |
| RFB + ±12 dB level-aug | 0.637 | 0.281 | 0.382 | 0.274 | **0.467** | 0.589 |
| mel | 0.602 | **0.407** | **0.510** | **0.437** | 0.548 | **0.599** |
| mel + ±12 dB level-aug | 0.576 | 0.400 | 0.472 | 0.424 | 0.577 | 0.563 |

**Verdict:** level-aug is the right *kind* of cochlear stress — it massively
rescues RFB under quiet (−20 dB: 0.20→0.47) with almost no clean cost. It does
**not** close the gap to mel on additive noise (white / pink / band). Pink is
easier than white for RFB (0.44 vs 0.26), so the earlier “RFB brittle to noise”
story was partly white-noise-specific. **Default recipe stays linear, no AGC;
optional `train_level_aug_db≈12` if level robustness matters.** Discuss before
Stage 5.

### Stage 5 — Full AudioPrism stack: PRISM → HORN (H5)

Architecture confirmation (RFB→HORN vs RFB→MLP) plus mechanistic diagnostics:
band half-ablation (high vs low necessity) and high→low coupling flow after
mapping HORN units to preferred RFB bands via ``i2h``. Note: fractal coupling
*structure* is fixed; only strength + input routing are learned.
Modal: `scripts/modal_audio_digit.py --sweep-preset rfb_stage5`.

#### `rfb_stage5` results (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_stage5.csv` (2 seeds).

| arm | clean | SNR10 | ablate low-drop | ablate high-drop | h→l ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| **learn RFB→HORN** | **0.642** | 0.256 | **0.526** | 0.409 | 0.943 |
| mel→MLP | 0.602 | 0.407 | 0.481 | 0.367 | — |
| frozen RFB→HORN | 0.514 | 0.354 | 0.411 | 0.409 | 0.902 |
| learn RFB→MLP (matched) | 0.332 | 0.202 | 0.229 | 0.145 | — |
| frozen RFB→MLP | 0.174 | 0.103 | 0.074 | 0.074 | — |

**Verdicts:**
- **H5 architecture PASS:** RFB→HORN ≫ RFB→MLP (learn 0.64 vs 0.33; frozen
  0.51 vs 0.17). Frontend value is not just cosmetic features.
- **High→low coupling FAIL / not seen:** preferred-band coupling ratio
  ≈0.90–0.94 (slightly *low→high* if anything). Fractal ``W`` structure is
  fixed — this stack does not reproduce AudioPrism's reported high→low
  streamlining under our diagnostic.
- **Band necessity:** both halves matter; **low-band ablation hurts more**
  than high (0.53 vs 0.41 drop on the winner) — not a high→low story.

### Stage 5+ — research brief & ranked sprint board (2026-07-19)

**Goal (not “beat Whisper”):** widen the OscNet claim under matched small
capacity — harder audio, clearer HORN advantage, close known holes — then
optionally one external KWS slice. SOTA keyword spotters (~96% Speech
Commands, 60k–few-M params) are a different game; we use them as *context*,
not the success criterion.

#### Expert lenses (what each domain says moves the needle)

| Lens | Consensus lever | Fit to our stack |
| --- | --- | --- |
| Small-footprint KWS | Learnable filterbanks help most when channels are few / noise is present; multi-condition train; SpecAugment-style | We already learn `{ω,Q}`; **noise-aug train** is the missing twin of our eval stress |
| SincNet / LEAF | Constrained learnable fronts beat free conv; LEAF adds learnable pooling/compression (PCEN-like) | Our Stage 4 blunt compress failed; **PCEN / per-band learnable gain** is the lit replacement |
| Filterbank+noise (ICASSP’23) | Learned banks adapt to noise spectra; dropout on bank helps; mel≈learned at K=40, learned wins at K≪10 | Suggests a **low-band RFB** stress (K=8–16) + noise-aug as a distinctive OscNet win surface |
| AudioPrism / HORN papers | Frozen PRISM + HORN; post-train high→low flow; noise tolerance from oscillatory dynamics | We matched arch win; mech high→low needs **learnable tonotopic W**; HORN noise story → test under multi-condition |
| OscNet prior (our ledger) | HORN≫MLP; amp; linear; level-aug helps quiet; white hurts RFB more than pink | Don’t rebuild AGC; do **noise-aug + scale + tonotopic coupling** |

#### Hypotheses ranked by expected value (needle × cost × OscNet-fit)

| ID | Hypothesis | EV | Cost | Decision |
| --- | --- | --- | --- | --- |
| **A** | Multi-condition **noise+level aug in training** closes white/pink/band gap vs mel without killing clean acc | **Highest** | Low | **Do first** |
| **B** | **Scale** (8–16k train, h=64–96, 4 seeds) widens clean gap and stabilizes | High | Med | Second |
| **C** | Learnable **dense tonotopic** HORN coupling recovers high→low *and* may lift acc | Med-High | Med | Third (science + maybe gains) |
| **D** | **PCEN / soft learnable per-band gain** (LEAF-style) beats linear amp under noise | Med | Med | After A (don’t confuse with Stage 4 AGC) |
| **E** | **Low-K bank** (8–16 bands) + learn RFB→HORN beats low-K mel (lit niche) | Med | Low | Nice OscNet-shaped claim |
| **F** | Hybrid router gated on RFB band-energy typicality | Med | Med | Synergy with existing Hybrid work |
| **G** | Full Speech Commands 35-class / external KWS bake-off | Low for science / High for PR | High | Only after A–C |
| — | Rebuild cochlear AGC / phase readout | Low | — | **Parked** (Stages 3–4) |

#### Sprint sequence

1. `rfb_plus_noiseaug` — train-time white/pink/band + level; same eval suite as `rfb_robust`
2. `rfb_plus_scale` — more data / width / seeds on winner ± noiseaug
3. `rfb_plus_tonotopic` — learnable dense W + Stage 5 diagnostics retest
4. Optional: PCEN arm, low-K bake-off, hybrid gate

### Stage 5+ — execution log

#### Sprint A — `rfb_plus_noiseaug` (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_plus_noiseaug.csv` (2 seeds).

| arm | clean | white10 | pink10 | band10 | quiet−20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RFB clean-train | **0.642** | 0.256 | 0.443 | 0.226 | 0.201 |
| RFB + level-aug only | 0.637 | 0.281 | 0.382 | 0.274 | **0.467** |
| mel clean-train | 0.602 | 0.407 | 0.510 | 0.437 | 0.548 |
| mel + level+noise aug | 0.560 | **0.479** | 0.509 | 0.482 | 0.561 |
| RFB + level+noise aug | 0.536 | 0.449 | 0.473 | **0.494** | 0.416 |

**Verdict (A partial PASS):** multi-condition aug **closes the additive-noise
gap** (RFB white 0.26→0.45; band 0.23→0.49, ties/beats mel on band) but
**costs ~0.10 clean accuracy**. Level-only remains best for quiet. Keep both
recipes: clean-train for nominal claim; `laug12+naug0.7` when robustness is
the target. Superseded at scale by Sprint B soft `naug≈0.4`.

#### Sprint B — `rfb_plus_scale` (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_plus_scale.csv` (2 seeds;
10k train / 1.2k eval / 50 ep / h=64).

| arm | clean | white10 | pink10 | band10 | quiet−20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **RFB→HORN clean** | **0.766** | 0.405 | 0.595 | 0.315 | 0.201 |
| **RFB→HORN + soft naug0.4** | **0.730** | **0.629** | **0.660** | **0.635** | 0.568 |
| RFB→HORN + hard naug0.7 | 0.705 | 0.678 | 0.689 | 0.682 | 0.566 |
| mel clean | 0.606 | 0.405 | 0.488 | 0.432 | 0.570 |
| mel + soft naug0.4 | 0.585 | 0.417 | 0.511 | 0.483 | 0.571 |
| RFB→MLP matched | 0.356 | 0.192 | 0.335 | 0.162 | 0.262 |

**Verdict (B strong PASS):** scale is the real needle-mover. Clean RFB→HORN
jumps **0.64→0.77** and opens a large gap over mel (0.61). Soft noise-aug at
scale is the sweet spot — clean still **0.73 ≫ mel**, and additive noise
**dominates mel** (white 0.63 vs 0.42). HORN≫MLP still holds hard. Soft
`naug≈0.4` becomes the robust default; Sprint C (tonotopic coupling) next if
pursuing mechanism / further gains.

#### Sprint C — `rfb_plus_tonotopic` (Modal, 2026-07-19)

CSV: `outputs/analysis/modal_audio_digit_rfb_plus_tonotopic.csv` (2 seeds;
10k / h=64 / 50 ep; Stage 5 diags on).

| arm | clean | white10 | h→l ratio | ablate low / high drop |
| --- | ---: | ---: | ---: | ---: |
| **dense** clean | **0.761** | 0.469 | 1.04 | 0.63 / 0.55 |
| **dense_tonotopic** clean | **0.758** | 0.485 | 1.01 | 0.63 / 0.56 |
| dense + soft naug0.4 | 0.740 | **0.642** | 1.06 | 0.61 / 0.55 |
| dense_tonotopic + soft | 0.745 | 0.639 | 1.02 | 0.62 / 0.55 |
| fractal_fixed clean | 0.649 | 0.247 | 1.00 | 0.54 / 0.45 |
| fractal_fixed + soft | 0.638 | 0.525 | 1.00 | 0.53 / 0.46 |
| mel MLP | 0.608 | 0.398 | — | — |
| RFB→MLP | 0.356 | 0.175 | — | — |

**Verdict (C mixed):** **accuracy PASS** — learnable dense W is load-bearing
(~0.76 vs frozen fractal ~0.65; ties Sprint B once we admit B’s fractal was
accidentally learnable). **Mechanism FAIL** — tonotopic init does not create
high→low streamlining (ratios≈1.0–1.06); dense ≈ dense_tonotopic on acc.
Low-band ablation still hurts more than high. Default coupling → `dense`.
Park AudioPrism high→low claim for this stack; next optional sprints are
low-K niche / soft-aug polish / PCEN / hybrid gate.

---

## 6. Design axes (resolved by stage, not guessed)

| Axis | Baseline | Where decided |
| --- | --- | --- |
| Domain | Audio first (0a), then CIFAR transfer | Stage 0a gate |
| Image drive | Prefer Gabor-like 2-D; raster diagnostic only | Stage 0b |
| Coupling in PRISM | Strictly uncoupled; weak tonotopic later | post–Stage 1 |
| Linear vs nonlinear | Linear IIR until Stage 4 | Stage 4 |
| Fixed vs learned tuning | **Fixed analytic cochlear** (paper); learned = Stage 2 extension | Stage 2 |
| Readout | Quadrature amplitude first | Stage 3 |
| Reservoir vs backprop | Frozen = paper-faithful; backprop = extension | Stage 1 |
| Spacing | Log / constant-Q (AudioPrism); linear as ablation note | Stage 1 |

---

## 7. What each outcome would mean

- **H0+H1+H2 (audio and/or image):** strongest — tuned resonance is a real
  frontend, not just "any spectral features," and the robustness thesis spans
  encoder and substrate.
- **H0+H1, H2 fails:** tuned decomposition is a good inductive bias (Gabor /
  SincNet territory) but the robustness-unification thesis is false. Still
  publishable; narrow the claim.
- **H2 holds, H1 does not:** two-regime positive on the input side — physics
  prior trades nominal accuracy for graceful degradation. Consistent with the
  rest of this repo.
- **Equal-frequency ties or beats tuned:** prism interpretation dies; effect
  was capacity / frozen structure. Retire frequency-tuning story; keep only
  what the controls support.
- **Classical Gabor/mel wins:** ODE resonator implementation adds nothing over
  textbook signal processing at this scale. Valuable negative; points to
  analog/nonlinearity/detuning as the only remaining bets (Stage 4 + detuning
  stress).
- **Audio Stage 0a fails, or only audio works:** image-frontend idea retires
  cleanly; PRISM stays an audio/temporal result (AudioPrism's actual claim).
- **H5 fails (PRISM helps any head equally):** frontend is a feature cosmetic;
  the PRISM→HORN architectural claim does not transfer.

---

## 8. Sequencing and synergy with other open work

1. **Router-fix first** (fixed-statistic vs learned vs oracle gating in
   `HybridImageGenerator`; Hybrid Frontier in `docs/experiment_report.md`).
   Completes a claim already in flight.
2. **Synergy, not competition:** Hybrid Frontier found the learned router
   fails OOD and needs a *fixed, non-learned shift statistic*. PRISM band
   energies are exactly such a statistic (spectral typicality of the input).
   A one-line follow-up after Stage 1: gate the hybrid on resonator band
   residual / spectral typicality. This slightly raises the prism plan's
   priority once the router fix lands.
3. **Then Stage 0a (audio)** — de-risks everything.
4. **Stage 0b (image drive prototype)** — local, cheap.
5. **Stage 1 CIFAR** with the full matched-control battery → branch on
   go/no-go.
6. Stages 2–4 only if Stage 1 clears the attribution bar; Stage 5
   (PRISM→HORN) is the headline architecture once the frontend is real.

Recommended order: **router fix → Stage 0a → Stage 0b → Stage 1 → branch
(including hybrid-gate synergy) → Stages 2–5 as warranted.**

---

## 9. Questions for Felix — and how answers re-prioritize stages

**Already settled by the paper (do not re-ask as unknowns):** PRISM is
frozen analytic cochlear gains; necessity is in the small-HORN regime;
high→low frequency routing is a reported post-training phenomenon; DHO form
is `ẍ + 2γẋ + ω²x = αI(t)`.

Map remaining answers to schedule changes:

| Question | If answer is… | Then… |
| --- | --- | --- |
| Unpublished deltas beyond the ICNCE abstract / figure? | Yes + deltas | Fold into Stage 0a before CIFAR |
| PRISM strictly uncoupled, or weak tonotopic coupling? | Weakly coupled | Add coupled-bank arm immediately after Stage 1 |
| Exact no-PRISM / "30 inputs" baseline construction? | Details | Match Stage 0a controls exactly |
| Readout into HORN: envelope, Hilbert, complex state, rate? | Amplitude-only | Stage 3 collapses to a check, not a sweep |
| Image / spatial PRISM actually run, or speculative? | Speculative | Keep Stage 0a as the real claim; CIFAR labeled transfer |
| Robustness / noise results beyond digit accuracy? | Yes | Align our H2 stressors with theirs |
| Shareable gain formulas, channel count, γ schedule, coupling init? | Yes | Copy into Stage 0a for fair replication |
| Relationship to Carvalho analog-HORN reservoir re-train — any PRISM-on-analog plans? | Yes | Prioritize detuning arm; possible joint experiment |

---

## 10. Implementation checklist (cost and interpretability)

Operational core — keep this section if the prose is ever trimmed.

- [x] Closed-form / biquad IIR for linear bank (Stages 0–1); no ODE unroll
- [x] Analytic cochlear / log-spaced gains matching AudioPrism (frozen
      primary arm); DHO form `ẍ + 2γẋ + ω²x = αI(t)`
- [x] Quadrature / RMS amplitude readout; unit-peak α; dt / Nyquist for
      log-spaced ω
- [x] Stage 0a in **small / parameter-matched** regime; Modal sweeps for
      Speech Commands + synthetic (`scripts/modal_audio_digit.py`)
- [x] Equal-frequency and fixed mel/STFT arms in Stage 0a
- [ ] Frozen-random-conv and regularized-conv skeptic arms (Stage 1)
- [x] Parameter counts reported per arm
- [ ] Extend `_corrupt_images_heldout` with spectral stressors + `{ω,Q}`
      detuning (Carvalho-style precision / component mismatch)
- [ ] Stage 5 high→low frequency information-flow diagnostic
- [x] Band-collapse diagnostic when learning `{ω, Q}` (Stage 2)
- [x] Band-spacing regularizer (`band_spacing_regularizer`,
      `collapse_reg_weight`) — confirm on Modal `audioprism_confirm`
      (accuracy OK; near-duplicates not fully eliminated)
- [x] Learnable `{ω,Q}` via residual schedule + freq-domain DHO response
      (IIR BPTT unstable; Stage 2 landed on Modal `rfb_learn`)
- [x] Stage 0a audio path: `oscnet/experiments/audio_digit/` (dedicated;
      not forced through `audio_wavelet.py`)
- [x] Image RFB **parked** after Stage 1a/1a′ (audio-only for this claim)
- [x] `audioprism_confirm` results ingested (4-seed / noise / matched MLP)
- [x] Stage 3 — amplitude / phase / both (`rfb_stage3`); **amp wins**
- [x] Stage 4 — nonlinearity (`rfb_stage4`); **linear enough** (H4 fail)
- [x] Robustness probe (`rfb_robust`): level-aug helps quiet; mel still
      wins additive noise
- [x] Stage 5 — H5 architecture PASS; high→low coupling **not** replicated
- [x] Stage 5+ research brief + ranked sprint board (in plan)
- [x] Stage 5+ Sprint A — `rfb_plus_noiseaug` (noise gap closes; clean tradeoff)
- [x] Stage 5+ Sprint B — `rfb_plus_scale` (**0.77 clean**; soft naug owns noise)
- [x] Stage 5+ Sprint C — `rfb_plus_tonotopic` (dense wins acc; high→low still absent)
- [ ] Hybrid router follow-up: gate on RFB band-energy typicality
- [ ] After C: soft-aug polish / low-K niche / PCEN / hybrid gate (parked until C lands)
