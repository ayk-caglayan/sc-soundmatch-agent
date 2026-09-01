---
tags: [neural-proxy, pretrained-embeddings, automatic-synthesizer-programming, black-box]
date: 2025-09-09
sources: 1
---

# Neural Proxies for Sound Synthesizers: Learning Perceptually Informed Preset Representations

**Authors:** Paolo Combes, Stefan Weinzierl, Klaus Obermayer
**Venue:** Journal of the Audio Engineering Society (accepted 2025)
**PDF:** SynthMatch-NeuralProxies-2509.07635v1.txt

## Key Contributions

- Method to approximate **arbitrary black-box synthesizers** via a neural proxy that maps presets to an audio embedding space
- Neural proxy trained on pretrained audio model embeddings (e.g., from VGGish, PANNs, CLAP)
- Enables integration of **audio embedding loss** into neural ASP systems for non-differentiable synthesizers
- Evaluated feedforward, recurrent, and transformer-based architectures as proxies
- Tested on three popular software synthesizers with both synthetic and hand-crafted presets

## Method

1. Generate (preset, audio) pairs from the black-box synthesizer
2. Extract audio embeddings using a pretrained model
3. Train a neural network to map preset parameters directly to the embedding space
4. Use the proxy in place of the synthesizer during training, enabling audio-domain loss without differentiable synthesis

## Key Results

- Encouraging results across all three synthesizers
- Transformer-based proxies show promise for complex parameter spaces
- Nuanced by computational resource requirements vs. quality tradeoff
- Paves the way for applying neural ASP to any commercial synthesizer

## Connections

- Complementary to [white-box](yang-white-box-ismir23.md) and [DDSP](masuda-ddsp-ismir21.md) approaches
- Neural proxy concept also used as baseline in [Yang et al.](yang-white-box-ismir23.md)
- See concept: [sound matching approaches](../sound-matching-approaches.md)
