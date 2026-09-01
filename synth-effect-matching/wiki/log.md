---
tags: [log, meta]
date: 2026-05-29
sources: 0
---

# Operations Log

## 2026-04-06: Initial bulk ingest of 37 papers

Ingested all 36 available text files (1 PDF text not found: ASA-2025-SynthMatch-handout).

### Source summaries created (30 files)

**Deep summaries (19 key papers):**
- barkan-deep-synth-pe-2017.md -- Barkan 2017, first deep CNN for synth parameter estimation
- inversynth-2018.md -- InverSynth journal extension
- flowsynth-2020.md -- Flow Synthesizer with VAE + normalizing flows
- shier-thesis-2021.md -- Shier M.Sc. thesis, hybrid neural+GA, SpiegeLib, torchsynth
- masuda-ddsp-ismir21.md -- First differentiable synth for sound matching
- sound2synth-2022.md -- SOUND2SYNTH for Dexed/DX7 FM synth
- yang-white-box-ismir23.md -- White-box differentiable rendering for synths
- yang-white-box-fm-2023.md -- White-box FM variant
- han-inverse-problem-2023.md -- PNP preprint (same as TASLP 2024)
- han-pnp-taslp-2024.md -- PNP loss, TASLP journal paper
- han-thesis-2025.md -- Han Ph.D. thesis on physicality of timbre
- hayes-equivariant-flow-2025.md -- Equivariant flow matching for ill-posed inversion
- synthrl-2025.md -- Reinforcement learning for cross-domain matching
- sound-similarity-metrics-2025.md -- Comprehensive loss function comparison
- neural-proxies-2025.md -- Neural proxies for black-box synths
- diffmoog-2024.md -- DiffMoog differentiable modular synthesizer
- modulation-discovery-2025.md -- Modulation extraction with DDSP
- fdn-reverb-matching-2025.md -- FDN reverb matching

**Shorter summaries (12 papers):**
- han-pnp-icassp-2023.md -- PNP conference paper
- han-gradient-clipping-2025.md -- Gradient clipping for PSM
- deepafx-2021.md -- DeepAFx black-box audio effects
- preset-gen-vae-2021.md -- VAE for DX7 preset generation
- masuda-saito-taslp-2023.md -- Semi-supervised DDSP matching extended
- masuda-qd-2023.md -- Quality-diversity GA matching
- sterling-bowed-string-2010.md -- Empirical bowed string physical model
- su-rnn-strings-2002.md -- RNN plucked string physical model
- itoyama-okuno-icmc14.md -- Linear regression parameter estimation
- neural-source-filter-2019.md -- Neural source-filter speech synthesis
- wav2shape-2020.md -- Shape from drum sounds
- torchsynth-synth1b1-2021.md -- 1B sounds dataset and GPU synth
- diff-tf-scattering-2022.md -- Differentiable JTFS on GPU
- mesostructures-2023.md -- Beyond spectrogram loss
- diaz-hayes-rigid-body-icassp23.md -- Differentiable modal resonators
- bruford-ast-dafx-2024.md -- AST encoder for sound matching

### Concept pages created (5)
- synth-parameter-estimation.md -- Core problem definition and historical arc
- sound-matching-approaches.md -- Taxonomy of methods with comparison table
- audio-similarity-metrics.md -- Loss functions and evaluation metrics
- plug-and-play-priors.md -- PNP loss formulation and properties
- reinforcement-learning-sound-matching.md -- RL paradigm for matching

### Overview and index updated
- overview.md -- Full narrative synthesis of the field
- index.md -- Complete catalog with thematic grouping

### Note on duplicate text files
- SynthMatch-PNP-HanHan-TASLP-2311.14213v2.txt and SynthMatchingAsInverseProblem-2311.14213v2.txt are the same paper (preprint of TASLP 2024)
- SynthMatchingDDSP-2023.txt and MasudaSaito23-... are the same paper (TASLP 2023)
- Both handled via cross-references in the source summaries

## 2026-05-29: DRC parameter-estimation ingest (4 papers, +1 concept page)

From a HANDOFF tied to an ISMIR-2026 paper-review session (black-box optimisation for DRC parameter estimation in a perceptual feature space). 37→41 papers.

### Source summaries created (4)
- [gorlow-drc-inversion-2013.md](sources/gorlow-drc-inversion-2013.md) — Gorlow & Reiss, model-based DRC inversion (= Zölzer Ch. 7 forward model, hard knee / 0 dB makeup; characteristic-function root search).
- [peladeau-blind-afx-2024.md](sources/peladeau-blind-afx-2024.md) — Peladeau & Peeters ICASSP, auto-encoder + DDSP blind audio-effect estimation; parameter distance ≠ perceptual distance.
- [sun-drc-neural-inversion-2024.md](sources/sun-drc-neural-inversion-2024.md) — Sun, Fourer & Maaref, hybrid neural DRC inversion (AST classify / MEE regress → Gorlow inversion). Reuses Gorlow's model unchanged.
- [peladeau-param-distributions-2025.md](sources/peladeau-param-distributions-2025.md) — Peladeau, Fourer & Peeters DAFx, estimating parameter *distributions* (normalizing flows + entropy) instead of point values.

### Concept page created (1)
- [drc-parameter-estimation.md](drc-parameter-estimation.md) — cross-topic hub. Captures session findings not in the PDFs: Gorlow=Zölzer equation-for-equation match; Sun's reuse + LM root-finder; the two ballistics pairs (detector + gain smoother) with canonical refs (Giannoulis–Massberg–Reiss 2012, McNally 1984, Stikvoort 1986); threshold-dominates; many-to-one ⇒ distributions; the {derivative-free + DRC + perceptual} prior-art gap.

### Updated
- [index.md](index.md) (sources 37→41; new concept-page row; Gorlow in Foundational, the rest in Current Frontiers).
- [overview.md](overview.md), [sound-matching-approaches.md](sound-matching-approaches.md) (new §6 inversion/blind/distributional), [audio-similarity-metrics.md](audio-similarity-metrics.md) (entropy/SOT losses; parameter-vs-perceptual finding).
- Cross-linked to `../../ddsp/wiki/` (Wright grey-box DRC) and the top-level forward-model writeup [`DRC/DRC.tex`](../../DRC/DRC.tex).
