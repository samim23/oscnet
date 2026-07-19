# Oscillatory Audio Generation — Experiment Plan

## OscNet latent dynamics for speech / music synthesis

*OscNet research plan, drafted July 2026.*  
*Status: **Stage 0 not started** — plan only.*  
*Precedent: spoken-digit RFB→HORN encoding sprint
([`resonator_filter_bank_frontend_plan.md`](resonator_filter_bank_frontend_plan.md),
paper [`docs/paper_rfb_horn/`](../paper_rfb_horn/)).*  
*Related ledgers:*
[`experiment_report.md`](experiment_report.md),
[`../papers/what_the_physics_prior_buys.md`](../papers/what_the_physics_prior_buys.md).*

### Settled insights (keep current as stages land)

| Claim | Status | Evidence |
| --- | --- | --- |
| *(none yet — Stage 0 pending)* | — | — |

**Inherited from digit encoding (do not re-litigate):**

| Insight | Carry into generation? |
| --- | --- |
| Temporal integration is load-bearing (pooled MLP fails) | **yes** — use framed features |
| Fair sequential controls required (GRU / iso-param) | **yes** — same discipline |
| HORN is a compact sequential binder (~7k ≈ larger GRU on mel) | **hypothesis** — test under generative loss |
| RFB helps additive noise more than clean accuracy vs mel | **optional** — Stage 0 G3 (amp), not the thesis |
| Phase readout load-bearing for **digit classification** | **FAIL there** — do not block Stage 1a G6 for **generation** |
| Equal-frequency / structure ablations matter | **yes** if RFB is used |
| Image-RFB / JEPA-lite parked | **stay parked** |
| Old wavelet AE (static coeff + fake time) | **do not restart from** |

### Inherited from image gen / physics-prior (hypotheses, not claims)

Source: `experiment_report.md`, `docs/papers/what_the_physics_prior_buys.md`.
Transfer as **testable design**, not as “CIFAR wins ⇒ music.”

| Lesson | Carry how | Do **not** transfer |
| --- | --- | --- |
| Matched non-oscillatory controls are non-negotiable | GRU + iso-param; optional same-stack residual MLP (`state_mlp`) | Claiming HORN from unpaired baselines |
| Physics prior ≈ finite-window **contraction / noise cleanup**, often with a clean-quality cost | Stage 0b/1: settling-depth + mild latent noise stress | “HORN generates better by default” |
| Settling has a useful window (~8–16 steps; 32+ can hurt) | Depth diagnostic `k ∈ {0,1,2,4,8,16,32}` | Blindly maximizing HORN steps |
| Structured state priors fight collapse | Stage 1 free continuation only; + shuffled-prior controls | Class-prior shortcuts as “diversity” |
| Multimode capacity can help after single-mode baseline | Stage 1+ gate | Leading with failed image “slow global carrier” |
| Hybrid roles beat pure-oscillator generators | HORN as latent **binder / refiner**; decoder can be conventional | Requiring HORN to invent all spectral detail |

### Protocol lessons from the digit sprint

| Lesson | Stage 0 default | Later (only if Stage 0 is interesting) |
| --- | --- | --- |
| **4096 RFB subsample** | Headline = **mel frames at full rate**. RFB arm = full-rate or disclosed + matched. | Revisit RFB once latent story is clear. |
| **Fixed ω/γ, 1 step/frame, tanh** | Start crude (digit-comparable). GRU win ≠ “oscillators can’t generate.” | If HORN ≈ GRU: learnable timescales / multi-step settle / richer drive. |
| **Fair GRU / iso-param** | Mandatory on the same features. | Same. |

### Borrowed from modern ML audio gen (do not reinvent)

Industry stack (borrow the ends, invent the middle):

```text
waveform
  → mel or neural-codec latent     # borrow (mel first)
  → generative / sequential model  # OscNet HORN lives HERE
  → vocoder / codec decoder        # borrow later (Griffin-Lim → HiFi-GAN/Vocos)
```

| Practice | Our use |
| --- | --- |
| Work in **mel / latent**, not raw waveform at Stage 0 | Next-frame on mel frames |
| **Teacher-forced** next-frame / next-latent before open sampling | Stage 0 primary |
| Frozen or simple **invert** for listening (Griffin-Lim; later frozen vocoder) | Qualitative only until Stage 2 |
| Spectral metrics + listening; don’t trust MSE alone | MSE + copy margin + optional mel distance |
| Diffusion / flow / codec-LM scaling | Optional Stage 2+ *wrappers* around an oscillatory latent — **not** Stage 0 |

### The bet — unfair advantage (not a guaranteed win)

There is **no hidden cheat code** that guarantees beating WaveNet /
Transformers on leaderboards. Digits and image nulls already showed that
“physics-shaped” ≠ free SOTA points. What we *do* have is a sharper,
domain-aligned bet:

> ONNs make **multi-timescale continuous state** first-class (frequency,
> damping, beating, entrainment, settling). WaveNet/Transformers
> *approximate* temporal structure; oscillators *are* that structure.
> At small capacity / limited data, a HORN latent may carry **mid-scale**
> audio structure (phones, onsets, motifs) more cheaply and more stably
> than a generic sequence model — especially under settling, noise, and
> short continuation.

That is the same family of win as digit HORN **compactness** + physics-prior
**contraction** — applied where the domain matches the machine.

| Claimed “unfair” edge | Real? | Role in this sprint |
| --- | --- | --- |
| Native multi-timescale dynamics | **best digital bet** | Stage 0/1 latent |
| Param efficiency / inductive bias | **plausible** (digit precedent) | G1/G2 |
| Settling / attractor stability | **plausible** (physics-prior) | G5, continuation |
| Phase / resonance as first-class state | **wildcard** | See § Frontend ladder — digit phase failed for *classification*; gen may differ |
| Analog / neuromorphic cheap temporal compute | **real long-horizon** | Not a Stage 0 success criterion |

**What is *not* the secret sauce:** “audio is oscillatory so we must win”;
more HORN steps / denser \(W\) / RFB mysticism; skipping fair GRU controls;
asking HORN to be WaveNet at 16 kHz.

**What we easily overlook**

1. **Don’t permanently throw away native variables.** Log-mel is Stage-0
   correct but discards phase/resonance. Escalate via the frontend ladder
   (§ below) before parking the whole line.
2. **Wrong game.** Classical stacks win leaderboards. Our unfair game is
   param efficiency + dynamical stability + structure-through-time (±
   robustness) — not MusicLM vibes alone.
3. **Hybrid is the advantage.** Micro → vocoder/codec; mid → HORN; long →
   conditioning. Pure “HORN is WaveNet” is how we hit the wall again.
4. **Hardware** is long-horizon unfairness; it will not make Stage 0 MSE
   look magical.

### Frontend / latent ladder (mel → ONN-native) — assigned by stage

Mel and RFB are the **same job** (spectral frontend → frames), different
basis: triangular mel filters vs log-spaced DHOs. Learnable RFB is already a
trainable oscillatory preprocessor (`{ω,Q}`; digit precedent). It becomes
*more* ONN-native only when we keep what resonators are — state \((x,v)\),
quadrature amplitude/phase — not when we only emit log-amp like mel.

Digit Stage 3: `readout=amplitude` won; phase-only ~chance for
**classification**. That does **not** settle generation: labels often need
envelopes; synthesis may need timing/interference. Give phase/state a reason
to matter (recon / continuation / invert), or it will look “irrelevant” again.

| Rung | Representation | Stage | Role |
| --- | --- | --- | --- |
| **A** | Mel log-amp frames @ full rate | **0 headline** | Fair HORN vs GRU; no RFB confound |
| **B** | RFB amplitude frames @ full rate (learn or frozen) | **0 optional** (G3) | Mel-analog, physics-shaped bands; incremental, not the wildcard |
| **C** | RFB `readout=both` or `phase` (amp + \(\cos\phi,\sin\phi\)) | **1a wildcard** | First true phase test under generative loss |
| **D** | RFB state frames \((x,v)\) or complex envelope | **1b** if C interesting | Full resonator trajectory as features |
| **E** | Complex STFT / analytic features + HORN | **1 control** | Separates “phase helps” from “our DHO helps” |
| **F** | Neural codec latent + HORN | **2+** | SOTA-shaped quality path; not more ONN-native |
| **G** | Waveform-driven bank, frame raw states (no mel) | **park unless D/E sing** | Closest to oscillators-all-the-way; hardest |

**Codec ≠ filterbank.** Codecs compress for bitrate; RFB is inductive-bias
frontend. Stacking (codec→HORN) is allowed in Stage 2+; it does not replace
the phase/state wildcard.

**Escalation rule**

```text
Stage 0: A (required) ± B
  → if G1 fails on A (and B doesn’t save it):
Stage 1a: try C once under gen loss  (don’t park yet)
  → if C fails: park ONN audio-gen OR one shot at D/E
  → if C helps: D, then invert/vocoder (Stage 2), music data (Stage 3)
```

**Definition of sprint success (not “beat Transformers”)**

One clean positive:

> HORN ≈ (or beats) iso-param GRU on next-frame and/or short continuation
> in mel (or richer latent) space, **with a settling or robustness
> signature the GRU lacks**, at fewer or similar params.

If we get that → the sauce was real. If not → dark intuition wins again,
and we know why — layered bet still intact for a phase-richer follow-up.

### Intellectual posture — hope + expected wall

We have hit “traditional ML wins under fair controls” before (digits, image
gen nulls). Common themes:

1. Task already solved by flexible sequence models at matched capacity.
2. Wrong success metric (label acc / pixel MSE) for what oscillators buy.
3. Fairer controls shrink the gap.
4. Oscillators asked to be the *entire* system (frontend + binder + decoder).
5. Representation crippled the prior (fake time, decimation, 1-step crude cell).

**Dark default:** Stage 0 may again show GRU ≥ HORN on clean next-frame.
That is a useful park signal, not a surprise.

**Hopeful difference for audio:** generation asks a dynamical state to *carry
structure through time*, which is closer to reservoirs than digit labeling —
*if* we do not force a small HORN to invent sample-level microstructure.

**Layered ownership (do not collapse these roles):**

| Layer | Who owns it | Notes |
| --- | --- | --- |
| Microstructure (phase, sample-level) | mel / codec / vocoder | Borrow; Stage 0 does not generate raw waveform |
| Mid structure (phones, onsets, motifs) | **HORN / reservoir latent** | The actual bet |
| Long structure (phrase, form) | conditioning / hierarchy / later scale | Not Stage 0 |

**Questions Stage 0/1 answer (instead of philosophizing):**

- Next mel frame: HORN ≈ iso-param GRU?
- Settling: help then hurt (finite window)?
- Mild latent noise / missing frames: gentler degradation?
- Free continuation: any structure, or collapse?

If all “no” → park cleanly (or escalate to phase-rich latent once). If some
“yes” → audio may be the first domain where the physics prior is not just a
sad null.

**Working recipe:** *TBD after Stage 0.*

**Naming.** Package: `audio_gen`. APIs: RFB / HORN / mel / GRU — no external product names.

---

## 0. One-paragraph summary

We want oscillators as a **generative latent**, not as a digit classifier.
The least-wrong “unfair advantage” is native multi-timescale continuous state
for mid-scale audio structure at small capacity — not leaderboard supremacy
over WaveNet/Transformers. Digits and image work already taught that fair
controls shrink gaps; this sprint succeeds only if HORN matches an iso-param
GRU on next-frame / short continuation *with* a settling or robustness
signature, at similar or fewer params. We borrow mel→vocoder practice, put
OscNet in the latent slot, keep microstructure out of the HORN, and treat a
clean park as a valid outcome.

---

## 1. Motivation — what is established, what is not

### 1.1 Established (encoding lane)

- Learnable RFB + frames + dense HORN is a viable small spoken-digit encoder.
- Mel + GRU / mel + HORN lead on clean digits; RFB sequential arms keep more
  white-noise accuracy under clean training.
- Process precedent: ledger + experiment package + Modal presets + paper.

### 1.2 Not established (generative lane)

- Oscillatory latents help **synthesize** audio structure.
- RFB frontend is necessary (or even helpful) for generation.
- Image-generator tricks transfer numbers (only hypotheses transfer).
- The old `audio_wavelet` path is a foundation — it is smoke reconstruction
  with fake temporal structure (`audio_wavelet.py`).

### 1.3 Goal (north star vs Stage 0)

| Horizon | Goal |
| --- | --- |
| **North star** | Speech or music generation with an oscillatory latent (mel / codec space → waveform) |
| **Stage 0** | At matched small capacity, HORN latent dynamics match iso-param GRU on **short next-frame prediction** in mel (full-rate) |

If Stage 0 fails cleanly, park or change the latent/objective — do not scale
failure into a vocoder.

---

## 2. Hypotheses (falsifiable)

| ID | Claim | Falsifier |
| --- | --- | --- |
| **G0** | Framed spectral features beat raw / pooled for next-frame in the small regime | Mel ≢ raw/pooled at matched capacity |
| **G1** | HORN matches iso-param GRU on next-frame mel MSE / spectral distance | GRU ≫ HORN on same features + seeds |
| **G2** | HORN is parameter-efficient under generative loss (~⅓–½ GRU params near-match) | Parity only when HORN ≥ GRU params |
| **G3** | RFB **amplitude** frames ≥ mel under matched heads *(Stage 0 optional)* | Mel ≫ RFB-amp → keep mel; still escalate to G6 before full park |
| **G4** *(Stage 1)* | Short continuation / reconstruction is structured (listening + metrics) | Collapse / silence / copy-last |
| **G5** *(Stage 1)* | HORN shows a finite settling window and/or better degradation under mild latent noise than matched controls | Flat or worse across depths; no robustness crossover |
| **G6** *(Stage 1a wildcard)* | Phase-aware RFB (`both` / state) beats RFB-amp or mel under **generative** loss at matched capacity | No gain vs amp/mel — phase was classification-irrelevant *and* gen-irrelevant |

Never claim “HORN helps” without a sequential control.

---

## 3. Task design (Stage 0)

### 3.1 Objective

**Primary — next-frame prediction (mel @ full rate)**

- Input: frames \(z_{1:t}\) (mel log-amp, \(T\) short, e.g. 16–64).
- Predict: \(z_{t+1}\) (teacher-forced sequence loss over \(t = 1..T-1\)).
- Metrics: frame MSE/L1, copy-last-frame margin (must beat predict \(z_t\)),
  optional mel/spectral distance, **param count**.
- Diagnostic: next-frame error vs settle/unroll depth
  \(k \in \{0,1,2,4,8,16,32\}\) (physics-prior carry).

**Secondary (Stage 0b, only if G1 interesting)**

- Encode short clip → settle HORN → decode frames; matched GRU AE.
- Optional Griffin-Lim sheet (listening only, not a claim metric).

**Out of scope for Stage 0**

- Raw-waveform diffusion, training a SOTA vocoder, EnCodec/SoundStream
  end-to-end, long-form music, lyrics, large SSL teachers.
- Restarting `audio_wavelet` static-coeff sequences.
- RFB-as-headline if it reintroduces 4096-subsample confounds.

### 3.2 Data

| Tier | Data | Role |
| --- | --- | --- |
| **0a** | Synthetic: noise bursts, harmonic stacks, AM/FM | CI + smoke |
| **0b** | Speech Commands short clips (reuse digit tooling) | Scientific gate |
| **0c** | Small music slice (loops / NSynth / `my_audio_samples`) | After 0b is sane |

### 3.3 Arms (matched capacity)

**Stage 0 frontends:** `mel` (headline, full rate); optional `rfb_learn` /
`rfb_frozen` **amplitude** @ full rate (G3); `raw_frames` control.  
**Not in Stage 0:** RFB phase/both/state (that’s G6 / Stage 1a).  

**Latent / predictor:** `horn_dense` | `gru` | `gru_isoparam` | `mlp_pooled`
(must fail) | optional `state_mlp_isoparam`.  
**Decode:** linear or tiny MLP → frame space.

Four seeds on the headline mel comparison. Fixed schedule; report final metric.

### 3.4 Go / no-go

| Gate | Proceed if |
| --- | --- |
| **Enter Stage 1** | G0 pass **and** G1 near-parity (or win) on mel 0b |
| **Enter Stage 1a (G6)** | G1 fails on mel (± RFB-amp) **or** G1 passes and we want the phase upside — run RFB `both`/state once under gen loss before park |
| **Park oscillatory gen** | G1 fails **and** G6 fails (phase/state also dead) after sane tuning |
| **Drop RFB-amp only** | G3 fails — mel default; phase ladder still allowed |
| **Stage 1 focus** | Binder/refiner + G5; priors only if continuation collapses |

---

## 4. Stages (sequence)

| Stage | Question | Frontend rung | Deliverable |
| --- | --- | --- | --- |
| **0** | Next-frame; HORN vs GRU (± iso, ± state_mlp) vs pooled | **A** mel; optional **B** RFB-amp | Package + Modal CSV; G0–G3 |
| **1** | Continuation / recon; settle depth + mild noise; prior if collapse | Stay on winner of A/B | Listening sheet; G4–G5 |
| **1a** | Does phase/state matter under **gen** loss? | **C** then **D**; **E** as control | G6 settle; only then deepen ONN-native frontend |
| **2** | Stronger invert (frozen HiFi-GAN/Vocos); optional **F** codec latent | Winner of 0/1/1a | Waveform samples |
| **3** | Music data + conditioning | Stable rung from above | Only if structure holds |
| **Park** | SOTA waveform chase; **G** without evidence; wavelet-static; image slow-carrier | — | Explicit PARK lines |

---

## 5. Engineering plan (digit-sprint shaped)

### 5.1 New package (do not overload `audio_digit`)

```text
oscnet/experiments/audio_gen/     # config, data, models, metrics, runner, cli
examples/audio_gen.py
scripts/modal_audio_gen.py
tests/test_audio_gen.py
docs/experiment_ledgers/audio_generation_plan.md
```

Reuse: mel/RFB frame patterns, `FractalHORNCell`, SC/Modal helpers.  
Do not reuse: classification-as-metric; wavelet fake-time sequences.

### 5.2 Modal presets (proposed)

| Preset | Intent |
| --- | --- |
| `agen_stage0_smoke` | Synthetic plumbing |
| `agen_stage0_mel_controls` | mel × {HORN, GRU, GRU-iso, MLP-pooled}, 4 seeds, SC |
| `agen_stage0_rfb_amp` | Full-rate RFB amplitude ± same heads (G3) |
| `agen_stage1a_rfb_phase` | RFB `both` / state vs amp; HORN vs GRU (G6) — only after Stage 0 |

→ `outputs/analysis/modal_audio_gen_*.csv`.

### 5.3 Metrics (minimum)

- Next-frame MSE / L1  
- Copy-last-frame margin  
- Param count  
- Depth sweep curve (diagnostic)  
- Optional: Griffin-Lim sheet (Stage 0b/1)

---

## 6. Relation to other OscNet lines

| Line | Relation |
| --- | --- |
| Digit RFB→HORN | Encoding precedent + cells; process template |
| Image HORN generator | Settling / prior / multimode **hypotheses** only |
| Physics-prior note | Contraction vs quality tradeoff — Stage 1 stress tests |
| Modern audio gen (external) | mel / codec + vocoder practice; OscNet = latent slot |
| Audio wavelet AE | Historical smoke; not the research path |

---

## 7. Immediate next actions

1. Plan updated (this doc).  
2. Scaffold `oscnet/experiments/audio_gen/` + synthetic next-frame smoke (**rung A**).  
3. Modal `agen_stage0_mel_controls`; optional `agen_stage0_rfb_amp`.  
4. Fill settled insights; Stage 1 vs Stage 1a (G6) vs park.  
5. Do **not** build phase/state plumbing until Stage 0 mel result is in
   (reuse `ResonatorBank` readout modes when G6 opens).

---

## 8. Non-goals / intellectual honesty

- Not SOTA TTS / music generation / beating WaveNet–Transformer leaderboards.  
- Not “digit accuracy transfers to generation.”  
- Not “CIFAR / physics-prior image wins transfer as numbers.”  
- Not “audio is oscillatory ⇒ we must win.”  
- Not “analog hardware someday” as a Stage 0 success criterion.  
- Prefer a clean **park** over a beautiful failure scaled to a vocoder.  
- Prefer escalating to **phase-rich latents** once before declaring the
  entire ONN audio-gen line dead on mel-magnitude alone.
