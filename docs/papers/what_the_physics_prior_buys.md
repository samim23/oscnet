# What Does the Physics Prior Buy You?

## Oscillatory neural networks vs. free-form recurrence, measured properly

*OscNet project synthesis — July 2026*

---

## Abstract

Oscillatory neural networks (ONNs) are widely promoted as a path toward
brain-like, energy-efficient, physically realizable computation. The
literature, however, rarely pits them against controls that are matched in
everything except the oscillation, and almost never scores them on anything
other than in-distribution accuracy. Over several weeks we ran a long series
of controlled experiments on generative image modeling and corruption
recovery (MNIST, CIFAR-10), comparing damped coupled-oscillator networks
(HORN-style, second-order dynamics) against a same-stack recurrent MLP
control that shares the encoder, decoder, training objective, data, seeds,
and settling protocol — differing only in the recurrent update rule.

The headline is a two-regime result. **Inside the training envelope, the
free-form recurrent control wins everything**: reconstruction, generation
quality, occlusion fill-in, clean PSNR — consistently, across seeds, at every
scale we tried. Attempts to close the gap by adding the ingredients theory
suggested (slow/global carriers, timescale separation, non-local and fractal
coupling topologies) either failed outright or recovered only a fraction of
the deficit. **But when test conditions leave the training envelope, the
ranking inverts.** Under out-of-distribution occlusion at 2–2.4× the trained
severity, the oscillator's fill-in error grows 1.8× while the control's grows
3.7–4.3×, making the oscillator absolutely better — replicated over four
seeds, and robust to a regularized control that rules out overfitting as the
explanation. The physics-constrained update trades nominal accuracy for
graceful extrapolation.

Along the way we isolated *why* each result happens: oscillator settling is
contraction onto a learned manifold, which repairs distributed corruption for
free but cannot transport structure across contiguous holes; coupling
locality is a genuine confound worth about a third of the recovery gap; and
one apparent oscillator win (3-bit quantization robustness) failed to
replicate at four seeds and was retracted. This article presents the full
arc, the numbers, and the methodology that made the conclusion trustworthy.

---

## 1. The question

The promise of oscillatory neural networks rests on an analogy: brains and
many physical systems compute with coupled oscillators, cheaply and robustly,
so artificial networks built from oscillator dynamics should inherit some of
that. The problem is that the claim is usually tested in a way that cannot
falsify it. Typical ONN papers compare an oscillator model against a
*different architecture* (different capacity, different connectivity,
different training recipe), score it on in-distribution accuracy, and
attribute whatever survives to "the dynamics."

We wanted the honest version of the question: **holding everything else
fixed, what does constraining the recurrent update to damped oscillator
physics actually buy — and what does it cost?**

Concretely, the oscillator under test is a HORN-style network: each of
256–512 sites on a retinotopic grid carries position/velocity state evolving
under second-order damped dynamics with learned natural frequencies and a
static spatial coupling profile; multimode variants stack 2 frequency modes
per site. Class conditioning drives the field, settling runs for a fixed
number of steps, and a small convolutional decoder reads the state out as an
image.

The control — and this is the load-bearing methodological choice of the whole
project — is a **same-stack StateMLP**: identical encoder, identical decoder,
identical conditioning, identical training objective and data and seeds and
settling protocol, with the second-order oscillator update replaced by a
small residual MLP acting on the same state vector. Any surviving difference
is attributable to the update rule, not to the scaffolding around it.

---

## 2. Act I — The generation contest, and losing it fairly

The project began with the ambitious framing: oscillator settling as a
generative engine — random state in, class-conditional image out after the
dynamics relax, with the hope that attractor structure would provide
semantic organization, diversity, and sample quality.

The early MNIST-era work (Winfree phase fields, rate-phase hybrids, masked
completion) established a pattern that never subsequently broke:

- On clean reconstruction, feedforward models beat every oscillatory variant.
- On masked completion, oscillatory local fields beat naive controls but the
  advantage shrank or vanished against *matched* recurrent controls
  (recurrent-conv, ConvLSTM), and feedforward models still won overall.
- Oscillatory machinery was most defensible as a local *refinement field* on
  top of a conventional prior — a hybrid result, not an ONN victory.

The CIFAR-10 RGB branch sharpened this with the same-stack StateMLP control.
Across capacity probes, retinotopic readouts, state-anchor training, prior
sampling, and patch objectives, the paired-by-seed comparison came back
either null or StateMLP-favoring. The healthy training pipeline transferred
wholesale to the non-oscillatory control; nothing about sample quality
required oscillation.

**Lesson of Act I:** if you only look at in-distribution quality metrics, the
oscillator constraint is pure cost. Every "oscillator advantage" we had seen
before adding matched controls was scaffolding, not dynamics.

---

## 3. Act II — The asymmetry: oscillators denoise, but cannot fill holes

The first genuine, sign-consistent oscillator result appeared when we stopped
asking "does it generate better?" and started asking "what do the dynamics
*do* to a corrupted state?"

**Noise-then-settle probe (eval-only).** Fit oscillator states to real CIFAR
images through the frozen decoder, corrupt the state with Gaussian noise,
settle, decode, and measure how much of the injected noise the dynamics
absorb (controlling for each arm's own clean-state drift):

| Arm | mean denoise fraction | cells where settling *amplified* noise |
| --- | ---: | ---: |
| HORN (global prior) | 0.53 | 0 / 40 |
| HORN (class prior) | 0.40 | 0 / 40 |
| StateMLP (same stack) | 0.24 | 9 / 40 |

Both HORN arms denoised in **every one of 80 cells** (2 seeds × 4 noise
scales × 5 settle depths), monotonically in depth. The StateMLP control was
unreliable — settling amplified corruption in nearly a quarter of its cells.
This is attractor-style contraction, and it is real.

**Occlusion probe (eval-only, same checkpoints).** Zero out a contiguous
patch of the state grid instead, and the picture inverts completely: *no* arm
filled the hole in. Occluded-region error never dropped below the "render
gray" floor at any settling depth, and settling degraded the intact region
while doing it. The oscillator's celebrated pattern-completion story was
absent — and notably, the StateMLP was marginally *less* bad at shallow
depths.

So the dynamics had a real, control-beating competence — but only for
*distributed* corruption, not *contiguous* damage.

---

## 4. Act III — Why: contraction is not transport

The distributed/contiguous split has a clean physical explanation, and
getting it explicit shaped everything after.

A locally-coupled field moves information at finite speed: per settle step,
influence spreads by roughly one coupling radius (a light-cone limit). Under
distributed corruption, every damaged site has intact evidence adjacent to
it, and local relaxation repairs it in a few steps. The interior of a large
contiguous hole, by contrast, is many coupling radii from any evidence;
within a bounded settling budget the boundary information physically cannot
reach it, and even in the limit a local operator produces the soap-film
interpolant — smooth, low-frequency, detail-free — because the interior is
genuinely underdetermined by local information.

What oscillator dynamics specifically add over generic recurrence is
**contraction toward a learned attractor manifold**: damped bounded dynamics
relax the off-manifold component of a perturbation while preserving
on-manifold structure. Isotropic noise is almost entirely off-manifold, so it
gets absorbed. A hole is not off-manifold noise — its correct fill lies
*along* the manifold and requires spatial **transport**. Contraction does not
transport. One mechanism, one win, one loss.

Nature agrees, which is the sobering part: no biological or physical system
we could find solves contiguous-hole completion with a single homogeneous
local oscillator field. Cortical filling-in uses long-range feedback from
areas with large receptive fields; synchronization physics uses mean-field or
long-range coupling; room acoustics handles "holes" because its low-frequency
basis is global standing waves; holography survives occlusion because the
encoding is distributed in the first place. The recurring answer is
hierarchy, long-range connectivity, or a global prior — never bare local
relaxation. Modern diffusion models make the same division of labor: dynamics
do the denoising, a learned global prior (U-Net receptive field, attention)
does the transport.

This predicted two interventions: train recovery explicitly, and give the
oscillator long-range structure. We did both.

---

## 5. Act IV — Training for recovery: the objective works, the theories mostly die

We built a corrupted-anchor training objective: occlude the input image
before encoding, add state-space noise, settle, decode, score against the
clean image — plus an explicit clean-fixed-point term so uncorrupted states
are trained to be genuine attractors. Evaluation scores occluded-region MSE
(the direct fill-in metric) separately from the intact region, across
settling depths.

Results across three GPU sweeps (all arms trained identically, 40 epochs,
CIFAR-10, seeds paired):

**Training for fill-in works.** Mixed-corruption training improved fill-in
3–5× over noise-only training (occluded-region MSE 0.04–0.10 vs 0.21–0.26).
No model fills holes for free; all of them can be taught to try.

**The slow/global carrier failed — three times.** The theory-favored fix
(add a slow, spatially coarse oscillator band to carry long-wavelength
structure) showed no reliable effect over its matched baseline, in either its
Winfree-era form or two current-stack implementations, including a "fair"
version on the strongest substrate. Falsified and retired.

**The multimode surprise.** The one arm where settling dynamics *actively
improved* fill-in on both seeds was the multimode oscillator (two frequency
modes per site): occluded-region MSE 0.072 → 0.058 from depth 0 to 16, while
StateMLP's settling gain was negligible (its fill-in comes almost entirely
from the encoder/decoder). A follow-up ablation showed the active ingredient
is per-site state capacity and mode coupling — the *equal-frequency* variant
was best, killing the timescale-separation interpretation. There is also a
hard ceiling: all oscillator arms peak around depth 8–16 and degrade badly by
depth 32–64. There is no iterate-to-perfection regime.

**The control still won.** StateMLP's fill-in (0.041) remained about 30–40%
better than the best oscillator number, and far more stable at depth. The
dynamics-driven fill-in was real but small, and the static encode/decode
pathway — where the free-form equilibrium is simply better — dominated the
total score.

---

## 6. Act V — The locality confound: real, but only a third of the story

At this point a reviewer-grade objection surfaced in our own comparison.
Every probe had treated "oscillator vs. not" as the independent variable, but
the two arms also differed on a second axis nobody had isolated: **coupling
range**. The HORN used spatially local coupling (each site reaches its grid
neighbours); StateMLP's hidden-to-hidden layers are dense — every unit
reaches every other in one step. By the transport argument of Act III, a
dense matrix can move structure across a hole in a single step; local
coupling provably cannot. Maybe StateMLP was winning for exactly the reason
HORN was losing, and oscillation itself was innocent.

So we equalized it. We added a **fractal coupling profile** (self-similar
ultrametric kernel: recursive quadrant blocks, direct long-range links,
strength decaying by block divergence level) alongside flat dense coupling,
and swept topology as the only variable — local vs. dense vs. fractal, on
both the single-mode and multimode substrates, with StateMLP as the
dense-linear ceiling.

Contiguous-hole fill-in (occluded-region MSE at depth 8, two-seed means):

| Arm | fill-in error |
| --- | ---: |
| multimode, local coupling | 0.0692 |
| multimode, fractal coupling | 0.0639 |
| multimode, dense coupling | 0.0567 |
| StateMLP (dense linear) | 0.0378 |

Three findings:

1. **The confound was real.** Non-local coupling improved oscillator fill-in
   by ~18% on the multimode substrate, both seeds, exactly as the transport
   theory predicted. Locality had genuinely been holding the oscillator back.
2. **But it explains only about a third of the gap.** With coupling range
   equalized, StateMLP still won by ~50%. The residual deficit belongs to the
   oscillatory update rule itself — the second-order dynamics constraint —
   not to topology.
3. **Fractal ≈ dense; and topology needs capacity.** The self-similar
   structure matched flat all-to-all coupling (non-locality is the active
   ingredient, not scale invariance), and topology did nothing on the
   single-mode substrate — long-range links only pay off where per-site
   capacity lets the dynamics use them.

This made the negative result publishable-grade: the oscillator-vs-control
comparison now stood on an unconfounded footing, and the oscillator still
lost. Which forced the last, most productive question of the project.

---

## 7. Act VI — Changing the fitness function

Every comparison so far had scored **exact reconstruction, at
infinite-precision parity, on in-distribution data** — the fitness function
digital free-form networks are optimized for. The claimed advantages of
physical and oscillatory computation live on entirely different axes:
tolerance to component noise, low-precision operation, and graceful
degradation when conditions drift off-nominal. None of that had ever been on
the scorecard. So we put it there.

The robustness evaluation scores contiguous-occlusion fill-in and clean PSNR
at a fixed settling depth while stressing the *model itself* and the *test
conditions*: Gaussian weight jitter (emulating analog component imprecision),
uniform weight quantization down to 3 bits (emulating low-precision
hardware), and occlusion far beyond the trained level (all models trained at
25% occlusion, tested up to 60%).

A first probe (2 seeds) showed a striking inversion under severe stress. We
then ran a confirmation sweep designed to kill it: **four seeds instead of
two, five weight-noise draws, intermediate stress levels to bracket the
crossover, and a regularized StateMLP arm (10× weight decay)** to test the
skeptic's explanation that the control's collapse was mere overfitting.

Fill-in error under out-of-distribution occlusion (4-seed means):

| Test occlusion | mm2 local | mm2 dense | StateMLP | StateMLP reg. |
| --- | ---: | ---: | ---: | ---: |
| 0.25 (trained) | 0.0708 | 0.0626 | 0.0414 | **0.0398** |
| 0.4 | 0.0926 | 0.0884 | **0.0779** | 0.0781 |
| 0.5 | 0.1107 | **0.1077** | 0.1264 | 0.1391 |
| 0.6 | 0.1243 | **0.1226** | 0.1546 | 0.1713 |

Relative degradation at 0.6 occlusion (error / own baseline): oscillator
1.80×, StateMLP 3.73×, *regularized* StateMLP 4.30×.

The findings, in order of importance:

1. **The crossover is real, replicated, and localized.** It opens between
   0.4 and 0.5 occlusion. Below it, StateMLP wins on every seed; above it,
   both oscillator arms beat both StateMLP arms, with the multimode-local
   oscillator winning 3/4 seeds against plain StateMLP and 4/4 against the
   regularized one. This is the first absolute oscillator win in the project
   that survived a designed attempt to kill it.
2. **The overfitting explanation is dead.** Regularization slightly improved
   StateMLP's nominal score and made its off-nominal collapse *worse*. The
   brittleness of the free-form update under distribution shift is
   structural, not a symptom of under-regularization. This is the result that
   converts "oscillators degrade gracefully" from a slogan into a controlled
   finding.
3. **One earlier claim was retracted.** The first probe's 3-bit quantization
   win did not replicate at four seeds (oscillator 2.12× degradation vs
   StateMLP 2.16× — a wash), and the regularized StateMLP was actually the
   most quantization-robust arm, for a mundane reason: weight decay shrinks
   weight range, which shrinks per-level quantization error. The 2-seed
   result was seed luck. We retired the claim.
4. **Weight noise gives relative, not absolute, robustness.** The oscillator
   degrades proportionally less at every jitter scale from 0.15 up
   (1.04–1.46× vs 1.27–1.83×), but StateMLP's nominal head start keeps it
   ahead in absolute terms throughout.

---

## 8. Synthesis — what the physics prior buys, and what it costs

Putting all six acts together, the ledger is:

**The physics prior costs:**

- ~40–70% higher error on every in-distribution task we measured:
  reconstruction, generation quality, trained-level occlusion fill-in, clean
  PSNR. This held at every scale, every topology, every substrate.
- A shallow settling window (depth ~8–16) beyond which iteration actively
  hurts — there is no iterate-to-perfection regime.
- An inability to transport structure across contiguous gaps with local
  coupling, of which non-local topology recovers only about a third.

**The physics prior buys:**

- **Reliable contraction of distributed corruption.** State denoising in
  80/80 measured cells, where the matched free-form control failed in 9/40.
  This is genuine attractor behavior, sign-consistent and monotone in depth.
- **Graceful extrapolation.** When corruption goes far beyond the training
  envelope, the oscillator's error grows ~1.8× where the free-form control's
  grows ~3.7–4.3× — enough to invert the absolute ranking, replicated over
  four seeds, and *not* purchasable for the control via regularization.
- **Relative robustness to weight perturbation**, though never enough to
  overcome the nominal deficit in absolute terms.

The one-sentence version:

> **Free-form recurrence wins inside and near the training envelope;
> physics-constrained oscillator dynamics fill in better when conditions
> leave it — and this is a structural property of the update rule, not an
> artifact of an under-regularized control.**

Mechanistically, both sides of the ledger come from the same place. The
damped second-order update is a *contractive, bounded* operator: it cannot
express the sharp, data-specific equilibria that make the free-form MLP
accurate on-nominal, and for the same reason it cannot express the sharp,
data-specific failure modes that make the MLP collapse off-nominal. You are
buying a smoother function class. Smoothness is a tax in-distribution and
insurance out-of-distribution. Nothing we measured contradicts that single
account.

---

## 9. What this does not claim

Honesty about scope is the point of the exercise, so, explicitly:

- **This does not rescue ONN generative modeling.** The oscillator makes
  worse images than its own matched control, which itself is nowhere near
  modern generative models. Nobody should build an image generator out of
  this on digital hardware.
- **The robustness win is about *unanticipated* damage.** If you know the
  deployment corruption level, you can almost certainly train the free-form
  control on it (or augment) and win. The oscillator's advantage applies when
  the stressor was not in the training distribution — which is real and
  practically important, but narrower than "better at heavy damage."
- **One dataset, one architecture family per side, two axes of stress.**
  CIFAR-10, HORN-style oscillators, one StateMLP design (plus its regularized
  variant). The crossover was demonstrated on the occlusion axis;
  quantization showed no oscillator advantage and weight noise only a
  relative one.
- **No hardware claim is made.** Everything here ran in float32 on GPUs. The
  results are the *digital shadow* of the analog-robustness story — evidence
  that the dynamics themselves carry the property, not evidence about any
  particular physical implementation.

---

## 10. Methodological lessons (arguably the real contribution)

1. **The same-stack control is everything.** Every oscillator "win" we found
   before installing the matched StateMLP control evaporated under it. Any
   ONN claim not accompanied by an equal-scaffolding, equal-budget,
   seed-paired free-form control should be treated as unattributed.
2. **Isolate axes one at a time.** "Oscillator vs. not" silently bundled
   coupling range with update rule for weeks. Unbundling it (Act V) both
   vindicated part of the theory and made the surviving negative result far
   stronger.
3. **Replicate before celebrating.** The 3-bit quantization win looked clean
   at two seeds and vanished at four. Two seeds is a hypothesis generator,
   not a result.
4. **Try to kill your positive result with the skeptic's control.** The
   regularized-StateMLP arm was added specifically because "it just overfit"
   was the best available deflationary story. Its failure is what gives the
   crossover its weight.
5. **Score on the axis your hypothesis actually lives on.** The graceful-
   degradation result was invisible for weeks not because it wasn't there,
   but because no metric was looking at it. If a system's claimed advantage
   is robustness, put robustness on the scorecard before concluding the
   system is worthless.
6. **Write down the negatives.** The carrier hypothesis died three times in
   three implementations. Without the running experiment log recording each
   death, we would have rediscovered and re-run it a fourth time. The
   negatives are what let the final claim be narrow *and* confident.

---

## 11. Where this could go

The result suggests directions we did not pursue, in rough order of expected
value:

- **Map the robustness frontier properly.** More datasets, more stress axes
  (input noise families, adversarial-adjacent perturbations, temporal drift),
  and the direct counterfactual we flagged: does an augmentation-trained
  free-form control catch the oscillator outside *its* new training envelope,
  or does the crossover just move? That experiment decides whether the
  advantage is fundamental or an arms race.
- **Hybrids, by design rather than concession.** Everything measured points
  to the division of labor nature uses: a free-form or global-prior pathway
  for on-nominal accuracy and transport, an oscillatory field for
  off-manifold cleanup and off-nominal insurance. A model that routes between
  them based on estimated distribution shift is the constructive version of
  this article's conclusion.
- **The efficiency angle we did not score.** The fractal coupling profile
  matched dense coupling with far more structured, compressible connectivity.
  On hardware where connectivity is the cost driver, "fractal ≈ dense at a
  fraction of the wiring" may matter more than any accuracy number here.
- **Actual analog validation.** The entire robustness result is a
  digital emulation of analog failure modes. If the dynamics carry the
  property in float32, the natural next test is carrying it onto genuinely
  imprecise substrates — which is the regime where the nominal-accuracy tax
  stops being a fair comparison, because the free-form control cannot run
  there at all.


## 12.  Synthesis — How Nature Keeps the Upsides Without the Downsides  

The two-regime result reduces to one mechanism: the damped second-order
oscillator update is a **smoother, bounded, contractive function class**. It
is too smooth to express the sharp data-specific equilibria that make the
free-form StateMLP accurate on-nominal, and for exactly the same reason it
cannot express the sharp data-specific failure modes that make StateMLP
collapse off-nominal. Smoothness is a tax in-distribution and insurance
out-of-distribution. The physics prior buys **predictability, not
performance**: ~40-70% worse on every in-distribution metric, but ~1.8x
degradation vs 3.7-4.3x under out-of-distribution occlusion, and the
regularized control made the collapse *worse*, not better — so the
brittleness of free-form recurrence is structural, not under-regularization.

The natural follow-on question — how does nature keep the oscillator upsides
(contraction, denoising, graceful degradation) *without* the downsides
(nominal-accuracy tax, no contiguous transport, shallow settling window)? —
has a single answer that unifies every negative result in this report:
**nature never runs a pure homogeneous local oscillator field in isolation.**
All three downsides are properties of that specific object, and biology and
physics remove each by wrapping the dynamics in surrounding structure rather
than by purifying the oscillator:

**Negative 1 — the nominal-accuracy tax (too smooth to be precise).** Nature doesn't ask the oscillatory dynamics to be the precise readout. It uses them for coordination and cleanup, and puts precision elsewhere — feedforward pathways, learned synaptic weights, sparse codes. In the brain, oscillations gate and bind (theta/gamma phase coding, synchronization for routing), while the actual content is carried by firing-rate/synaptic structure. The dynamics do the robust part; a different substrate does the sharp part. That's a division of labor, not a better oscillator.

**Negative 2 — can't transport across contiguous holes (contraction ≠ transport).** Nature adds long-range and top-down structure. Cortical filling-in of the blind spot or a scotoma is driven by feedback from higher visual areas with large receptive fields and learned priors, plus long-range cortico-cortical loops — not by lateral relaxation in one local sheet. This is the same lesson our fractal/dense coupling experiment showed in miniature: give the field non-local links and hole-filling improves. Nature just wires that non-locality in anatomically from the start, and backs it with a global prior for the part that has to be invented rather than transported.

**Negative 3 — shallow settling window (iterate-to-perfection doesn't exist).** Physical systems don't iterate a homogeneous field to convergence either; they use multiple timescales and hierarchy so different scales settle at different rates, and standing-wave / global-mode bases so large-scale structure is represented directly instead of having to propagate step by step. A room's low frequencies live in global eigenmodes; a hologram spreads every region across the whole plate so occlusion costs resolution, not a patch. When the representation itself is non-local, the "hole" problem partly disappears.

**The unifying principle:** nature keeps the oscillator's contraction/robustness and offloads everything the oscillator is bad at onto a surrounding architecture — hierarchy, long-range feedback, learned priors, multiple timescales, and non-local (even distributed/holographic) representations. Every biological or physical system we looked at that "shrugs off damage" is hybrid and multiscale. None of them is a pure local oscillator field that somehow lost its downsides.

---

## Appendix — Where the evidence lives

All experiments are reproducible from this repository. Key pointers:

- **Full chronological log with every probe, number, and decision:**
  `docs/experiment_report.md` (the sections from "CIFAR RGB State Recovery
  Probe" onward cover Acts II–VI). Modal run IDs and launch details:
  `docs/modal_runs.md`.
- **Models:** `oscnet/models/generative/` (HORN, multimode, coarse-to-fine,
  StateMLP control). Coupling profiles including the fractal/ultrametric
  kernel: `oscnet/core/coupling.py` (`hierarchical_coupling_profile`).
- **Recovery and robustness evaluation:**
  `oscnet/experiments/mnist_generator/metrics.py`
  (`compute_generator_recovery_metrics`,
  `compute_generator_robustness_metrics`, weight perturbation/quantization
  helpers).
- **Sweep definitions:** `scripts/modal_mnist_generator.py` — in particular
  `mnist_generator_cifar10_rgb_recovery_training_probe`,
  `..._multimode_carrier_probe`, `..._coupling_topology_probe`,
  `..._robustness_probe`, and `..._robustness_confirmation`.
- **Result CSVs:** `outputs/analysis/modal_mnist_generator_cifar10_rgb_*.csv`
  (one row per trained run, all metrics flattened).

Headline sweeps behind the central claim: coupling-topology probe
(12 runs, app `ap-qTl1hsn2kkqoRTqfulAyKz`), robustness probe (8 runs, app
`ap-CYvg8hNf6VjCCJH9Q9xxHa`), robustness confirmation (16 runs, app
`ap-sYXsSdYEHPBDjeOebDUIaL`). All runs: A10G GPUs on Modal, 40 epochs,
CIFAR-10, 10k training images, seeds paired across arms.

*Written as the closing synthesis of the OscNet generative/recovery research
arc, July 2026.*