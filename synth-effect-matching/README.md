# Papers on neural methods for matching synthesizer/effect parameters to desired sounds

The accompanying PDFs and text extractions are maintained as a local research corpus and are not
redistributed in this repository. This index links to public source records; see the
[annotated bibliography](References.md) for summaries and citations.

- [2002 - Nonlinear Parameter Estimation for a Recurrent Neural Network Model of the Bowed String](https://doi.org/10.1109/TNN.2002.1031945)
- [2007 - wav2shape: From Sound to Rigid-Body Dynamics](https://arxiv.org/abs/2007.10299)
- [2010 - Synthesizing Bowed Strings Using Generative Models](References.md)
- [2013 - Model-Based Inversion of Dynamic Range Compression (model-based precursor to the neural DRC inversion, Sun et al. 2024-11)](References.md)
- [2014 - Bayesian Nonparametric Approach to the Estimation of Parameters in Sound Synthesis Models](References.md)
- [2017 - Inverting Synthesizers Using Learned Reverse Controls (DeepSynthPE)](References.md)
- [2018-04 - Automatic Programming of VST Sound Synthesizers Using Deep Networks and Other Techniques](https://ieeexplore-ieee-org.stanford.idm.oclc.org/stamp/stamp.jsp?tp=&arnumber=8323327)
- [2018-12 - InverSynth: Deep Estimation of Synthesizer Parameter Configurations from Audio Signals](https://doi.org/10.1109/TASLP.2019.2944568)
- [2019-04 - Neural source-filter waveform models for statistical parametric speech synthesis](https://arxiv.org/abs/1904.12088)
- [2019-12 - Flow Synthesizer: Universal Audio Synthesizer Control with Normalizing Flows](https://doi.org/10.3390/app10010302)
- [2021-?? - Synthesizer Sound Matching Using Differentiable DSP (Shier thesis)](References.md)
- [2021-05 - [DeepAFx] Differentiable Signal Processing with Black-Box Audio Effects](https://doi.org/10.1109/ICASSP39728.2021.9415103)
- [2021-09 - Improving Synthesizer Programming from Variational Autoencoders Latent Space (DAFx-21)](References.md)
- [2021-11 - Synthesizer Sound Matching with Differentiable DSP (ISMIR-21)](https://archives.ismir.net/ismir2021/paper/000053.pdf)
- [2022-04 - Differentiable Time-Frequency Scattering for Audio Synthesis](https://arxiv.org/abs/2204.08269)
- [2022-05 - Sound2Synth: Interpreting Sound via FM Synthesizer Parameters Estimation](https://doi.org/10.24963/ijcai.2022/682)
- [2022-07 - [DeepAFx-ST] Style Transfer of Audio Effects with Differentiable Signal Processing](https://arxiv.org/abs/2207.08759) [(code)](https://github.com/adobe-research/DeepAFx-ST)
- [2022-07 - TorchSynth: One Billion Audio Sounds at a Time](https://arxiv.org/abs/2104.12922)
- [2023-01 - Improving Semi-Supervised Differentiable Synthesizer Sound Matching for Practical Applications](References.md) [(code)](https://gwendal-lv.github.io/preset-gen-vae/)
- [2023-01 - Mesostructures for Sound Synthesis](https://arxiv.org/abs/2301.10183)
- [2023-01 - Plug-and-Play Prior for Audio Inverse Problems (PNP)](https://arxiv.org/abs/2301.02886)
- [2023-03 - Quality-diversity for Synthesizer Sound Matching](https://doi.org/10.2197/ipsjjip.31.220)
- [2023-04 - Differentiable DDSP for Rigid-Body Modal Synthesis](References.md)
- [2023-09 - White-Box Parameter Search for Synthesizer Sound Matching (ISMIR-23)](References.md)
- [2023-11 - Synthesizer Sound Matching as Inverse Problem (PNP-based)](https://arxiv.org/abs/2311.14213)
- [2023-11 - PNP methods for Audio Inverse Problems (TASLP submission)](https://doi.org/10.1109/TASLP.2024.3393738)
- [2023-11 - White-Box FM Synth Parameter Estimation](References.md)
- [2024-01 - DIFFMOOG: A Differentiable Modular Synthesizer for Sound Matching](https://arxiv.org/abs/2401.12570)
- [2024-04 - Blind Estimation of Audio Effects Using an Auto-Encoder Approach and Differentiable DSP (ICASSP-24)](https://peladeaucome.github.io/ICASSP-2024-BEAFX-using-DDSP/)
- [2024-07 - Synthesizer Sound Matching Using Audio Spectrogram Transformers (DAFx-24)](References.md)
            (based on [AST](https://arxiv.org/abs/2104.01778))
- [2024-11 - Neural-Enhanced Dynamic Range Compression Inversion: A Hybrid Approach for Restoring Audio Dynamics (Sun et al.)](https://github.com/SunHaoRanCN/DRC-Inversion)
- [2025-01 - Gradient Clipping for Improved Training in DDSP](References.md)
- [2025-03 - Unearthing the Physicality of Instrumental Timbre (Han Han Thesis: PNP Wrap-Up)](References.md)
- [2025-06 - Sound Similarity Metrics for Synthesizer Sound Matching](https://arxiv.org/abs/2506.22628)
- [2025-06 - Differentiable Audio Processing and Synthesis](https://arxiv.org/abs/2506.07199)
- [2025-08 - SynthRL: Cross-domain Synthesizer Sound Matching via Reinforcement Learning (IJCAI-25)](https://argaaw.github.io/synthrl-demo/) [(code)](https://argaaw.github.io/synthrl-demo/)
- [2025-09 - Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations](https://arxiv.org/abs/2509.07635)
- [2025-09 - Audio Processor Parameters: Estimating Distributions Instead of Deterministic Values (DAFx-25)](https://github.com/peladeaucome/DAFx_Params_Distrib)
- [2025-10 - Modulation Discovery for DDSP](https://arxiv.org/abs/2510.06204)
- [2025-10 - Matching Reverberant Speech Through Learned Acoustic Embeddings and Feedback Delay Networks](https://arxiv.org/abs/2510.23158)
- [2025-12 - Neural Parameter Estimation for Musical Sound Synthesis (ASA-25 Overheads)](References.md)

---

[1st DAFx Parameter Estimation Challenge 2025](https://github.com/LOGUNIVPM/1st-DAFx-Challenge) --- [Details](https://github.com/LOGUNIVPM/1st-DAFx-Challenge/blob/main/DAFxChallengeDetails.pdf)

---

[Annotated Bibliography](References.md)

---

[Notes](Notes.md)

---

## Local Software

- [Neural Spectral Modeling Template](https://github.com/josmithiii/neural-spectral-modeling-template.git)
  - Image Processing approach
  - Spectrograms and/or Feature Maps processed by CNNs and/or Transformers
  - Conditioning Inputs supported
  - Classification and Regression supported
  - Simple Example Synth Matching Example: Decay Rate and Wah Pedal Angle
  - [Video](https://youtu.be/hRglC84nWoc?t=12921)
  - [Overheads](https://ccrma.stanford.edu/~jos/pdf/NSMT.pdf)

---

#### [Music 423 2023 GitLab Project](https://cm-gitlab.stanford.edu/jos/music423-2023/-/blob/master/README.md)
