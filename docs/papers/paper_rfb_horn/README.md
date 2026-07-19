# RFB–HORN: compact spoken-digit recognition

Paper sources for the OscNet RFB→HORN audio encoding study.

| File | Role |
| --- | --- |
| [`main.tex`](main.tex) | Camera-style article |
| [`references.bib`](references.bib) | Bibliography |
| [`figures/`](figures/) | Generated figures (PDF + PNG) |
| [`main.pdf`](main.pdf) | Compiled PDF (8 pages; fair controls + iso-param GRU) |

**Author:** Samim A. Winiger (AI agents disclosed in acknowledgments only).

**Claim (post fair controls):** temporal integration is load-bearing; HORN is a compact sequential head (~7k ≈ mel→GRU at ~19k); RFB helps white-noise robustness; mel ≥ RFB on clean digits with fair heads.

### Build

```bash
cd docs/paper_rfb_horn
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

### Regenerate figures

From the repo root (requires Modal CSV summaries under `outputs/analysis/`):

```bash
python docs/paper_rfb_horn/make_figures.py
python docs/paper_rfb_horn/make_hero_figure.py
```

### Key Modal presets

```bash
modal run scripts/modal_audio_digit.py --sweep-preset rfb_plus_controls
modal run scripts/modal_audio_digit.py --sweep-preset rfb_plus_isoparam
```

A narrative markdown mirror also lives at [`../paper_oscnet_rfb_horn.md`](../paper_oscnet_rfb_horn.md).
