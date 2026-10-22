# Architecture audit and scope of the rerun

The authorized rerun uses **LaneDG-Semantic-Reimplementation-v1**, a documented
semantic adaptation. It does not reproduce the cited instance model exactly.
Its name, output task, parameter count and loss must be updated consistently
throughout any revised manuscript.

Reference inspected on 22 October 2026:
[Zheng et al., Lane Detection Algorithm Based on Improved YOLOv8](https://www.mdpi.com/2073-431X/15/9/561),
DOI 10.3390/computers15090561.

The cited paper describes YOLOv8n-seg instance outputs, DGF backbone changes,
a C2f FPN-PAN neck, EUCB upsampling and a multi-task continuity loss. Its
comparison table reports 7.2 million parameters for its proposed model. An
exact executable author implementation and complete layer configuration were
not recovered. Architectural equivalence cannot be established from a shared
name or a forced parameter count.

| Property | Historical local model | Real-image semantic rerun |
|---|---|---|
| Output | One full-resolution binary mask | One full-resolution binary mask |
| Instance boxes, coefficients, prototypes | Absent | Absent; explicitly outside the semantic task |
| Backbone | Small custom convolution/DGF stages | DGF-inspired stages at P2, P3 and P5, C2f and SPPF |
| Fusion neck | Top-down fusion only | Top-down and bottom-up C2f FPN-PAN |
| Upsampling | Bilinear/depthwise/shuffle/pointwise | Explicit EUCB-inspired decoder |
| C2f | Simplified split and recombination | Concatenation of both split branches and each repeated bottleneck |
| Parameters | 405,088 including 10,399 scalar-calibration elements | 1,141,793 actual structural parameters |
| Count targeting | Explicit calibration vector sized to hit a target count | No padding, target count or calibration vector |
| Loss | Binary BCE/Dice/vertical continuity | Masked BCE, per-image Dice and both-direction continuity |
| Localization/classification tasks | Absent despite manuscript claims | Absent and not claimed |
| AP/mAP | Image-IoU threshold rates labeled as mAP | Rates labeled as mask-IoU success rates |

The legacy 10,399-element vector affects the forward pass only through its
mean and a scalar multiplier. Its structural parameter subtotal is 394,689.
This observation describes the current implementation; it does not establish
which code produced the original manuscript tables.

The reimplementation fixes the omitted bottom-up path and avoids count
padding, but its channel widths, repeat counts, static half-channel split,
adaptive gate and semantic decoder are explicit reconstruction choices. It
does not claim the cited paper's exact DGF channel-selection mechanism,
instance-segmentation loss, parameter count, pretrained state or scores.

Runtime checks verify the binary output shape and finite gradients reaching
every parameter. Tests cover both-direction masked continuity, invalid-pixel
exclusion and annotation geometry. These checks establish that the documented
semantic model executes and trains; they do not prove equivalence to the
unavailable author implementation or validate benchmark convergence.

For a paper retaining the exact YOLOv8n-LaneDG instance-model claim, the author
implementation and its task-specific loader, losses and evaluation would
still be required. The present results support a clearly renamed exploratory
semantic cross-dataset study, as authorized by the user.
