# RAW and Sensor-Native Perception Basis

This note distills the useful engineering guidance from
`deep-research-report-4.md`. That source file is treated as research input, not
as a directly publishable citation set, because its `turn...` citations are
not portable outside the Codex session.

## Main Takeaways

- The wrong claim is "RAW always beats sRGB." The defensible claim is that
  sensor-native information helps when the front end is task-aware and the DNN
  is adapted to the representation.
- Naive RAW is an important negative baseline. It often underperforms sRGB
  because the detector must learn dynamic range, CFA structure, noise, white
  balance, and sensor-specific statistics that a normal ISP usually hides.
- Domain gap and pretraining dominate. sRGB-pretrained detectors do not
  automatically become good RAW detectors, and training from scratch on small
  RAW datasets is usually weak.
- The most relevant system architecture is a split output: a human display
  stream, a perception stream, auxiliary confidence maps, and sensor metadata.
- Strong claims need paired or matched evaluation, fixed detector recipe,
  condition-specific metrics plus a robustness gate, confidence intervals, and
  compute/latency/memory reporting.

## Evaluation Requirements

The minimum benchmark matrix for a RAW or sensor-native superiority claim should
include these rows under a fixed detector and training recipe:

1. Standard HumanISP/sRGB baseline.
2. Naive RAW or RAW-like baseline.
3. Classical lightweight RAW transform, such as demosaic plus gamma or log
   compression.
4. Task-aware PerceptionISP or adapter path.
5. RAW-domain adaptation or pretraining compared with sRGB-domain pretraining.
6. Condition-specific metrics and a condition robustness gate for low light,
   HDR/glare, weather, visibility, and other adverse slices.
7. Optional auxiliary-map variants, such as noise, saturation, HDR-source, or
   demosaic-confidence channels.

The current project now covers rows 1-4 at the software-reference level on
KITTI-derived evidence, and it exercises auxiliary-map tensors through export,
training, and direct dense evaluation. It does not yet cover claim-quality
RAW-domain pretraining or large adverse-condition RAW datasets.

## Dataset Direction

The current KITTI/COCO smoke and validation reports are useful for engineering
iteration, but they are not enough for a broad external performance claim. The
next benchmark tier should add public RAW/adverse-condition datasets where
possible:

- PASCALRAW-style daylight sanity checks for naive RAW versus lightweight
  adaptation.
- Low-light RAW detection data for dark-scene behavior.
- HDR driving RAW data for glare and dynamic-range behavior.
- Adverse-condition RAW data for low light, rain, fog, and mixed conditions.
- Real RAW segmentation data if segmentation claims are added.

When those datasets are not locally available, the project should keep the
claim narrow and explicitly state that the current evidence is simulated or
KITTI-derived rather than real multi-condition RAW evidence.

## Output Contract

The implemented PerceptionISP output contract is aligned with the research
direction:

- Primary display stream: human-tuned RGB/YUV equivalent.
- Primary perception stream: log or linear camera-native RGB tensor.
- Auxiliary maps: noise/SNR, saturation/clipping, HDR confidence/source,
  demosaic confidence, lens gain/reliability, color confidence, focus/blur, and
  optional Clear/IR evidence.
- Metadata: exposure, gain, CFA pattern, calibration profile, timing, and
  sensor provenance.

Existing RGB DNNs will not use these maps by themselves. They need a trained
RGB+aux stem, an auxiliary branch, a proposal calibration head, or an adapter
that preserves class-label behavior while using aux evidence.

## Current Claim Position

The current evidence supports this narrow statement:

```text
PerceptionISP can expose sensor-native auxiliary maps to DNN-facing tensors,
and a calibrated aux-assisted proposal path can reduce false positives versus
HumanISP under a recall budget on the current KITTI-derived validation setup.
```

It does not support this broader statement yet:

```text
PerceptionISP is generally superior to HumanISP for detection.
```

That broader claim still needs a real adapted detector, stronger held-out
datasets, real adverse-condition slices, and a passing broad-superiority gate.
