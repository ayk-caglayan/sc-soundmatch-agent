# FluCoMa UGens in sc_claw_flucoma

FluCoMa (Fluid Corpus Manipulation) classes are already registered in the
sc_claw_flucoma validator (`sc_classes.txt` and `sc_ugen_signatures.json`), so they
pass Layer 1 and Layer 1b validation without any extra setup.

Whether they work at Layer 2 (sclang SynthDef build check) and at render time
depends on whether the FluCoMa SuperCollider plugin is installed in the local
`sclang` environment.

## Most Useful Real-Time UGens for Sound Matching

### FluidSines.ar — sinusoidal re-synthesis
Tracks and re-synthesizes the sinusoidal components of an input signal.
Useful when the target is a tonal, harmonic sound and you want to model its
partial structure directly.

```supercollider
// Signature
FluidSines.ar(in, bandwidth: 76, detectionThreshold: -96,
              birthLowThreshold: -24, birthHighThreshold: -60,
              minTrackLen: 15, trackMethod: 0,
              trackMagRange: 15, trackFreqRange: 50, trackProb: 1.0,
              windowSize: 1024, hopSize: -1, fftSize: -1, maxFFTSize: 16384)
```

Example — re-synthesize sinusoidal layer of a live input:
```supercollider
var src, sines;
src = SinOsc.ar(440) + (SinOsc.ar(880) * 0.5);
sines = FluidSines.ar(src, detectionThreshold: -60, birthLowThreshold: -30);
Out.ar(0, sines.dup);
```

### FluidSineFeature.kr — sinusoidal peak analysis
Outputs the frequencies and magnitudes of the N strongest sinusoidal peaks
each hop. Useful for analysis rather than synthesis.

```supercollider
// Signature
FluidSineFeature.kr(in, numPeaks: 10, detectionThreshold: -96,
                    order: 0, freqUnit: 0, magUnit: 0,
                    windowSize: 1024, hopSize: -1, fftSize: -1,
                    maxFFTSize: 16384, maxNumPeaks: 10)
```

### FluidHPSS.ar — harmonic/percussive separation
Separates the harmonic and percussive layers of a signal.

```supercollider
// Signature
FluidHPSS.ar(in, harmFilterSize: 17, percFilterSize: 31,
             maskingMode: 0, ...)
// Returns [harmonic, percussive, residual]
```

### FluidTransients.ar — transient extraction
Detects and extracts transients from a signal.

```supercollider
// Signature
FluidTransients.ar(in, order: 20, blockSize: 256, padSize: 128,
                   skew: 0, threshFwd: 2, threshBack: 1.1,
                   windowSize: 14, clumpLength: 25)
// Returns [transients, residual]
```

## How to Add FluCoMa Classes to sc_classes.txt / sc_ugen_signatures.json

The current registries were generated from the installed SC class library.
If new FluCoMa versions add classes not yet present, run:

```bash
# From an sclang session, dump all class names:
# Class.allClasses.collect(_.name).sort.do { |n| n.postln }
# Then append new Fluid* entries to sc_classes.txt.
```

For `sc_ugen_signatures.json`, add entries in the format:
```json
"FluidNewUGen.ar": ["param1", "param2", ...],
"FluidNewUGen.new": ["maxSize"]
```
