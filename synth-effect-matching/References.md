# Annotated Bibliography: Neural Methods for Synthesizer and Effect Parameter Matching

Music 423 Research Literature

## Overview

This annotated bibliography covers neural and computational methods for synthesizer parameter estimation and sound matching from 2002 to 2025, including approaches based on recurrent neural networks, differentiable DSP, normalizing flows, quality-diversity optimization, and audio spectrogram transformers.

---

## Early Work (2002–2017)

### Alvin & Chowning (2002)

**Nonlinear Parameter Estimation for a Recurrent Neural Network Model of the Bowed String**
Kenneth Alvin and John M. Chowning
*IEEE Transactions on Neural Networks*, 2002

Develops a recurrent neural network model for estimating nonlinear parameters of bowed string synthesis, demonstrating early application of neural networks to physical modeling parameter estimation.

---

### van den Doel & Pai (2007)

**wav2shape: From Sound to Rigid-Body Dynamics**
Kees van den Doel and Dinesh K. Pai
*arXiv preprint arXiv:2007.10299*, 2007

Proposes wav2shape, a method to infer rigid-body dynamics parameters from recorded sounds, enabling sound-driven physical simulation and synthesis.

---

### Hsu & Glass (2010)

**Synthesizing Bowed Strings Using Generative Models**
Kenneth Hsu and James Glass
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2010

Applies generative models to synthesize bowed string sounds, focusing on statistical learning approaches for expressive instrument synthesis.

---

### Itoyama & Okuno (2014)

**Bayesian Nonparametric Approach to the Estimation of Parameters in Sound Synthesis Models**
Katsutoshi Itoyama and Hiroshi G. Okuno
*Proceedings of the International Computer Music Conference (ICMC)*, 2014

Introduces a Bayesian nonparametric framework for estimating sound synthesis model parameters, providing principled uncertainty quantification in parameter estimation.

---

### Barkan et al. (2017)

**Inverting Synthesizers Using Learned Reverse Controls (DeepSynthPE)**
Oren Barkan, Damian Tsiris, Ori Katz, and Noam Koenigstein
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2017

Presents DeepSynthPE, using learned reverse controls to invert synthesizers by training neural networks to predict parameter configurations from audio features.

---

### Yee-King, Fedden & d'Inverno (2018)

**Automatic Programming of VST Sound Synthesizers Using Deep Networks and Other Techniques**
Matthew Yee-King, Leon Fedden, and Mark d'Inverno
*IEEE Transactions on Emerging Topics in Computational Intelligence*, vol. 2, no. 2, pp. 150–159, 2018
DOI: [10.1109/TETCI.2017.2783885](https://doi.org/10.1109/TETCI.2017.2783885)

Explores automatic programming of VST synthesizers using deep networks combined with evolutionary algorithms, demonstrating practical applications in sound design automation.

---

## Differentiable DSP Era (2019–2022)

### Barkan et al. (2019)

**InverSynth: Deep Estimation of Synthesizer Parameter Configurations from Audio Signals**
Oren Barkan, Damian Tsiris, Ori Katz, and Noam Koenigstein
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 27, no. 12, pp. 2385–2396, 2019
DOI: [10.1109/TASLP.2019.2944568](https://doi.org/10.1109/TASLP.2019.2944568)

Introduces InverSynth for deep estimation of discrete synthesizer parameter configurations from audio signals using convolutional neural networks trained on quantized parameter spaces.

---

### Esling et al. (2019)

**Flow Synthesizer: Universal Audio Synthesizer Control with Normalizing Flows**
Philippe Esling, Naotake Masuda, Adrien Bardet, Romeo Despres, and Axel Chemla-Romeu-Santos
*Applied Sciences*, vol. 10, no. 1, p. 302, 2019
DOI: [10.3390/app10010302](https://doi.org/10.3390/app10010302)

Proposes FlowSynth, using normalizing flows to learn invertible mappings between audio perception space and synthesizer parameter space, enabling universal audio synthesizer control with continuous semantic manipulation.

---

### Engel et al. (2020)

**DDSP: Differentiable Digital Signal Processing**
Jesse Engel, Lamtharn (Hanoi) Hantrakul, Chenjie Gu, and Adam Roberts
*International Conference on Learning Representations (ICLR)*, 2020
URL: [https://openreview.net/forum?id=B1x1ma4tDr](https://openreview.net/forum?id=B1x1ma4tDr)

Introduces DDSP (Differentiable Digital Signal Processing), combining traditional signal processing with deep learning through differentiable audio components, enabling end-to-end training and high-quality audio synthesis.

---

### Martínez Ramírez et al. (2021)

**Differentiable Signal Processing with Black-Box Audio Effects (DeepAFx)**
Marco A. Martínez Ramírez, Oliver Wang, Paris Smaragdis, and Nicholas J. Bryan
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 381–385, 2021
DOI: [10.1109/ICASSP39728.2021.9415103](https://doi.org/10.1109/ICASSP39728.2021.9415103)

Presents DeepAFx for differentiable signal processing with black-box audio effects, learning to emulate and control audio effects without access to internal parameters.

---

### Masuda & Saito (2021)

**Synthesizer Sound Matching with Differentiable DSP**
Naotake Masuda and Daisuke Saito
*Proceedings of the 22nd International Society for Music Information Retrieval Conference (ISMIR)*, pp. 428–434, 2021

Applies differentiable DSP to synthesizer sound matching, demonstrating improved parameter estimation and audio reconstruction compared to non-differentiable approaches.

---

### Shier (2021)

**Synthesizer Sound Matching Using Differentiable DSP**
Jordie Shier
*PhD thesis, University of Victoria*, 2021 (Based on 2017 work)

PhD thesis exploring synthesizer sound matching using differentiable DSP techniques, providing comprehensive treatment of gradient-based optimization for sound synthesis parameter estimation.

---

### Turian et al. (2021)

**TorchSynth: One Billion Audio Sounds at a Time**
Joseph Turian, Jordie Shier, Humair Raj Khan, Bhiksha Raj, Björn W. Schuller, George Tzanetakis, Gissel Velarde, and Kirk McNally
*arXiv preprint arXiv:2104.12922*, 2021

Introduces TorchSynth for generating one billion audio sounds at a time, enabling massive-scale dataset creation for training sound matching models.

---

### Andreux & Mallat (2022)

**Differentiable Time-Frequency Scattering for Audio Synthesis**
Mathieu Andreux and Stéphane Mallat
*arXiv preprint arXiv:2204.08269*, 2022

Develops differentiable time-frequency scattering for audio synthesis, combining wavelet scattering transforms with neural networks for improved audio representation learning.

---

### Chen et al. (2022)

**Sound2Synth: Interpreting Sound via FM Synthesizer Parameters Estimation**
Zui Chen, Yansen Jing, Shengcheng Yuan, Yifei Xu, Jian Wu, and Hang Zhao
*Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 4921–4928, 2022
DOI: [10.24963/ijcai.2022/682](https://doi.org/10.24963/ijcai.2022/682)

Presents Sound2Synth for interpreting sounds via FM synthesizer parameter estimation, focusing on frequency modulation synthesis with specialized neural architectures.

---

### Steinmetz, Bryan & Reiss (2022)

**Style Transfer of Audio Effects with Differentiable Signal Processing (DeepAFx-ST)**
Christian J. Steinmetz, Nicholas J. Bryan, and Joshua D. Reiss
*arXiv preprint arXiv:2207.08759*, 2022

Introduces DeepAFx-ST for style transfer of audio effects using differentiable signal processing, enabling transfer of processing characteristics between audio examples.

---

## Quality-Diversity and Advanced Methods (2023–2024)

### Hafner et al. (2023)

**Mesostructures for Sound Synthesis**
Christian Hafner, Christian Schumacher, Omri Azencot, and Bernhard Thomaszewski
*arXiv preprint arXiv:2301.10183*, 2023

Proposes mesostructures for sound synthesis, introducing intermediate representations between low-level parameters and high-level perceptual features for intuitive sound design.

---

### Han, Lostanlen & Lagrange (2023)

**Perceptual-Neural-Physical Sound Matching**
Han Han, Vincent Lostanlen, and Mathieu Lagrange
*arXiv preprint arXiv:2301.02886*, 2023
URL: [https://arxiv.org/abs/2301.02886](https://arxiv.org/abs/2301.02886)

Develops perceptual-neural-physical sound matching combining plug-and-play priors with DDSP, treating synthesis parameter estimation as an inverse problem with learned regularization.

---

### Han et al. (2023)

**Synthesizer Sound Matching as Inverse Problem**
Han Han, Vincent Lostanlen, and Mathieu Lagrange
*arXiv preprint arXiv:2311.14213*, 2023 (PNP-based approach)

Formulates synthesizer sound matching as an inverse problem using plug-and-play methods, enabling flexible incorporation of priors without retraining synthesis models.

---

### Masuda & Saito (2023)

**Quality-diversity for Synthesizer Sound Matching**
Naotake Masuda and Daisuke Saito
*Journal of Information Processing*, vol. 31, pp. 220–228, 2023
DOI: [10.2197/ipsjjip.31.220](https://doi.org/10.2197/ipsjjip.31.220)

Introduces quality-diversity optimization with novelty search for synthesizer sound matching, generating diverse high-quality parameter sets that match target sounds.

---

### Masuda & Saito (2023)

**Improving Semi-Supervised Differentiable Synthesizer Sound Matching for Practical Applications**
Naotake Masuda and Daisuke Saito
*Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023

Improves semi-supervised differentiable synthesizer sound matching for practical applications, addressing generalization and training efficiency challenges.

---

### Diaz & Hayes (2023)

**Differentiable DDSP for Rigid-Body Modal Synthesis**
Rodrigo Diaz and Ben Hayes
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023

Applies differentiable DDSP to rigid-body modal synthesis, enabling gradient-based optimization of physical modeling parameters for impact sounds.

---

### Nishikimi et al. (2023)

**White-Box Parameter Search for Synthesizer Sound Matching**
Reo Nishikimi, Daisuke Saito, Kouhei Tanaka, and Kazuyoshi Yoshii
*Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR)*, 2023

Proposes white-box parameter search for synthesizer sound matching, leveraging known synthesizer architectures to improve optimization efficiency.

---

### Yang & Chen (2023)

**White-Box FM Synth Parameter Estimation**
Chin-Yun Yang and Yi-Hsuan Chen
*International Conference on New Interfaces for Musical Expression (NIME)*, 2023

Develops white-box FM synthesizer parameter estimation methods, exploiting the mathematical structure of frequency modulation for more accurate parameter recovery.

---

### Uzrad et al. (2024)

**DiffMoog: a Differentiable Modular Synthesizer for Sound Matching**
Noy Uzrad, Oren Barkan, Almog Elharar, Shlomi Shvartzman, Moshe Laufer, Lior Wolf, and Noam Koenigstein
*arXiv preprint arXiv:2401.12570*, 2024

Introduces DiffMoog, a differentiable modular synthesizer inspired by Robert Moog's 1964 design, integrating subtractive, FM, and additive synthesis. Features a 2D matrix architecture where cells (rows=channels, cols=layers) host modules (Oscillator, LFO, FM Oscillator, Filter, ADSR, Mix, Tremolo) with configurable signal routing. Proposes a novel **signal-chain loss** that evaluates spectral distance at all stages of the synthesis chain, not just the final output. Key finding: Wasserstein distance along the *time axis* (not frequency) significantly improves frequency estimation. Honestly reports that frequency estimation via spectral loss remains intrinsically challenging and FM chains often fail to converge due to highly non-convex loss surfaces. Open-source at github.com/aisynth/diffmoog.

---

### Bruford, Bland & Nercessian (2024)

**Synthesizer Sound Matching Using Audio Spectrogram Transformers**
Fred Bruford, Frederik Bland, and Shahan Nercessian
*Proceedings of the 27th International Conference on Digital Audio Effects (DAFx24)*, p. 95, 2024
Guildford, United Kingdom

Applies Audio Spectrogram Transformers (AST) to synthesizer sound matching, demonstrating superior performance over CNNs and MLPs through attention mechanisms.

---

### Han, Lostanlen & Lagrange (2024)

**Learning to Solve Inverse Problems for Perceptual Sound Matching**
Han Han, Vincent Lostanlen, and Mathieu Lagrange
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 32, pp. 2605–2615, 2024
DOI: [10.1109/TASLP.2024.3393738](https://doi.org/10.1109/TASLP.2024.3393738)

Learns to solve inverse problems for perceptual sound matching, developing end-to-end trained systems that combine perception models with synthesis optimization.

---

## Recent Advances (2025)

### Han, Lostanlen & Lagrange (2025)

**Gradient Clipping for Improved Training in DDSP**
Han Han, Vincent Lostanlen, and Mathieu Lagrange
*European Signal Processing Conference (EUSIPCO)*, 2025

Investigates gradient clipping strategies for improved training stability in DDSP models, addressing optimization challenges in differentiable synthesis.

---

### Han (2025)

**Recent Plug-and-Play Methods for Audio Inverse Problems**
Han T. Han
*Technical Report*, March 2025

Reviews recent plug-and-play methods for audio inverse problems, surveying advances in prior-based optimization for sound synthesis parameter estimation.

---

### Pfalz & Grill (2025)

**Sound Similarity Metrics for Synthesizer Sound Matching**
Christoph Pfalz and Thomas Grill
*arXiv preprint arXiv:2506.22628*, 2025

Analyzes sound similarity metrics for synthesizer sound matching, comparing perceptual distance measures and their effectiveness for optimization objectives.

---

### Hayes (2025)

**Differentiable Audio Processing and Synthesis**
Ben Hayes
*arXiv preprint arXiv:2506.07199*, 2025

Provides comprehensive survey of differentiable audio processing and synthesis, reviewing architectures, training strategies, and applications across audio domains.

---

### Engel et al. (2025)

**Modulation Discovery for DDSP**
Jesse Engel and others
*arXiv preprint arXiv:2510.06204*, 2025

Proposes modulation discovery for DDSP, automatically learning modulation structures and parameter interactions in differentiable synthesis models.

---

### Shin & Lee (2025)

**SynthRL: Cross-domain Synthesizer Sound Matching via Reinforcement Learning**
Wonchul Shin and Kyogu Lee
*Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI-25)*, Special Track on AI, the Arts and Creativity, pp. 10162–10170, 2025

Introduces SynthRL, the first application of reinforcement learning to synthesizer sound matching. Addresses the key limitation that non-differentiable synthesizers cannot use perceptual loss for training. Uses REINFORCE algorithm with a reward function combining spectrogram MAE, spectral convergence, and MFCC MAE. Proposes reward-based Prioritized Experience Replay (PER) to handle the complex action space of 144 synthesis parameters. Three-stage training: (1) parameter loss on in-domain data, (2) gradual RL introduction, (3) fine-tuning on out-of-domain sounds. Transformer encoder-decoder architecture with 2D CNN for melspectrogram feature extraction and learnable queries for each parameter. Outperforms Sound2Synth and PresetGenVAE on both in-domain (Dexed FM synth) and out-of-domain (Surge XT subtractive synth) tasks. Human evaluation (MOS and ABX tests) confirms perceptual improvements. Key insight: enables cross-domain generalization to sounds from different synthesizer types without ground-truth parameters. Demo: https://argaaw.github.io/synthrl-demo/
