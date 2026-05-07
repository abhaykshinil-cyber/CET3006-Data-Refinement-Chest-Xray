# UGDR Research Paper — Diagram Reference
## Publication-Quality Figures for "Uncertainty-Guided Data Refinement for Robust Medical Image Classification"

---

# DIAGRAM 1 — UGDR END-TO-END PIPELINE

## Figure Title
**Figure 1.** Overview of the Uncertainty-Guided Data Refinement (UGDR) Pipeline.

---

## ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        UGDR PIPELINE OVERVIEW                           │
└─────────────────────────────────────────────────────────────────────────┘

 ┌────────────────────┐
 │   CHEST X-RAY      │  5,855 images  |  NORMAL (27%) + PNEUMONIA (73%)
 │   DATASET          │  Source: Guangzhou Women & Children's Medical Centre
 └─────────┬──────────┘
           │  Stratified split  (70 / 15 / 15,  seed = 42)
           ▼
 ┌──────────────────────────────────────────────────────────┐
 │   DATA PARTITIONING                                       │
 │   D_train (3,669)  |  D_val (786)  |  D_test (878)       │
 └─────────┬────────────────────────────────────────────────┘
           │  Preprocessing:  Resize 224x224, RGB, ImageNet normalisation
           │  Augmentation (train only):  H-flip p=0.5, Rotate +/-10 deg
           ▼
 ╔══════════════════════╗
 ║  STAGE 1             ║
 ║  CLEAN BASELINE      ║── ResNet18 + Dropout(0.5) + FC(512→2)
 ║  MODEL TRAINING      ║   Adam lr=1e-4, Batch 32, Max 15 epochs
 ╚══════════╦═══════════╝   Early stop patience=5
            ║
            ║  (controlled experiment only)
            ▼
 ╔══════════════════════╗
 ║  STAGE 2             ║
 ║  NOISE INJECTION     ║── Symmetric label flip  rho = 20%
 ║  (Synthetic)         ║   ~734 labels corrupted (NORMAL <-> PNEUMONIA)
 ╚══════════╦═══════════╝   Ground-truth set G stored (NOT used by UGDR)
            ║
            ▼
 ╔══════════════════════╗
 ║  STAGE 3             ║
 ║  NOISY MODEL         ║── Same architecture + same hyperparameters
 ║  TRAINING            ║   --> NOISY checkpoint  (val loss 0.293)
 ╚══════════╦═══════════╝
            ║
            ▼
 ╔══════════════════════════════════════════════════════════╗
 ║  STAGE 4  -  MC DROPOUT UNCERTAINTY ESTIMATION           ║
 ║                                                          ║
 ║  BatchNorm → eval mode   |   Dropout → train mode        ║
 ║                                                          ║
 ║  For each x_i in D_train:                                ║
 ║    T = 30 stochastic forward passes                      ║
 ║    mu(x)    = mean prediction across passes              ║
 ║    sigma2(x)= variance across passes                     ║
 ║    u(x)     = max_c  sigma2_c(x)   [scalar uncertainty]  ║
 ╚══════════╦═══════════════════════════════════════════════╝
            ║
            ▼
 ╔══════════════════════════════════════════════════════════╗
 ║  STAGE 5  -  DUAL-CRITERION ERROR DETECTION              ║
 ║                                                          ║
 ║  Criterion A (High Variance):                            ║
 ║    u(x_i) > tau_u   (tau_u = 0.02)                       ║
 ║                                                          ║
 ║  Criterion B (Confident Mismatch):                       ║
 ║    y_hat_i ≠ y_i  AND  max_c mu_c(x_i) > tau_c = 0.70   ║
 ║                                                          ║
 ║  Flagged set:  F = { i : Crit-A OR Crit-B }              ║
 ╚══════╦═══════════════════════════════════════════╦═══════╝
        ║                                           ║
        ║ Flagged (~726 samples)    Retained (~2,943 samples)
        ▼                                           ▼
 ┌─────────────────┐                   ┌────────────────────────┐
 │  REMOVE         │                   │  D_refined             │
 │  Unreliable     │                   │  (80% of D_train kept) │
 │  Samples        │                   └────────────┬───────────┘
 └─────────────────┘                                │
                                                    ▼
                                        ╔══════════════════════╗
                                        ║  STAGE 6             ║
                                        ║  RETRAINING ON       ║── Same arch.
                                        ║  REFINED DATA        ║   Same params
                                        ╚══════════╦═══════════╝
                                                    ║
                                                    ▼
                                        ┌───────────────────────┐
                                        │  FINAL EVALUATION     │
                                        │  on D_test (n=878)    │
                                        │                       │
                                        │  CLEAN   F1 = 92.9%  │
                                        │  NOISY   F1 = 87.1%  │
                                        │  CLEANED F1 = 91.9%  │
                                        │  Recovery    = 83%   │
                                        └───────────────────────┘
```

---

## Publication-Level Caption

**Figure 1.** Overview of the Uncertainty-Guided Data Refinement (UGDR) pipeline for chest X-ray pneumonia classification. The pipeline comprises six sequential stages: (1) a clean-data baseline model is trained on the stratified corpus; (2) synthetic label noise is injected at a 20% rate to simulate real-world annotation error; (3) a noisy-data model is trained under identical hyperparameters; (4) Monte Carlo Dropout is applied over T = 30 stochastic forward passes to compute per-sample predictive variance; (5) a dual-criterion mechanism — combining a high-variance criterion (Criterion A) and a confident-mismatch criterion (Criterion B) — identifies and removes unreliable training samples; and (6) a fresh classifier is retrained on the refined corpus and evaluated against the held-out test set. Ground-truth noise indices are recorded for post-hoc evaluation only and remain inaccessible to the UGDR mechanism throughout, ensuring the pipeline operates without noise supervision.

---

## Explanation Paragraph (Section 3.1 — body text)

Figure 1 presents the complete UGDR pipeline as a sequential flow from raw dataset input through three model variants to final three-model evaluation. The diagram makes explicit the closed-loop nature of the framework: the NOISY model trained in Stage 3 is the same model whose uncertainty estimates are used in Stage 4 to audit the corpus on which it was trained. This self-referential design is a key property — it requires no external information, no additional annotation effort, and no architectural changes beyond those already present in a standard dropout-regularised training workflow. The branching structure at Stage 5 illustrates the dual-criterion mechanism central to UGDR: Criterion A and Criterion B together partition the training set into flagged and retained subsets, and only retained samples proceed to Stage 6 retraining. The evaluation panel at the base of the figure summarises the three-model comparison that constitutes the paper's primary experimental evidence, showing that UGDR recovers 83% of noise-induced F1-score degradation without access to noise labels.

---

---

# DIAGRAM 2 — MODEL ARCHITECTURE

## Figure Title
**Figure 2.** ResNet18-Based Classifier with Monte Carlo Dropout Head.

---

## ASCII Diagram

```
             MODEL ARCHITECTURE — UGDR BINARY CLASSIFIER
┌────────────────────────────────────────────────────────────────────┐

  INPUT IMAGE
  224 × 224 × 3  (RGB, ImageNet-normalised)
       │
       ▼
 ╔═══════════════════════════════════════════════════════════════╗
 ║             ResNet18 BACKBONE                                  ║
 ║   (pretrained ImageNet-1K  |  ~11.18M parameters)             ║
 ║                                                               ║
 ║  Conv1  7×7, stride 2, 64 filters                             ║
 ║  MaxPool 3×3, stride 2                                         ║
 ║     │                                                         ║
 ║  Stage 1: 2 x BasicBlock  (64 filters,  56×56)                ║
 ║  Stage 2: 2 x BasicBlock  (128 filters, 28×28)                ║
 ║  Stage 3: 2 x BasicBlock  (256 filters, 14×14)                ║
 ║  Stage 4: 2 x BasicBlock  (512 filters,  7×7 )                ║
 ║     │                                                         ║
 ║  Global Average Pooling  →  512-d feature vector               ║
 ╚═══════════════════════════════════════════════════╦═══════════╝
                                                     ║
                                           512-d feature vector
                                                     ║
                                                     ▼
                                          ┌──────────────────────┐
                                          │  DROPOUT LAYER       │
                                          │  p = 0.5             │
                                          │                      │
                                          │  Training:  active   │
                                          │  Inference: active   │  <-- MC Dropout
                                          │  (non-standard)      │
                                          └──────────┬───────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  LINEAR LAYER        │
                                          │  FC(512 → 2)         │
                                          │  [NORMAL, PNEUMONIA] │
                                          └──────────┬───────────┘
                                                     │
                      ┌──────────────────────────────┤
                      │                              │
                      ▼                              ▼
             ┌─────────────────┐          ┌──────────────────────┐
             │  TRAINING MODE  │          │  MC DROPOUT MODE     │
             │                 │          │                      │
             │  Softmax        │          │  T = 30 passes       │
             │  + Cross-entropy│          │  p_t per pass        │
             │  loss           │          │  mu(x)  = mean       │
             │                 │          │  s2(x)  = variance   │
             │  Predicted      │          │  u(x)   = max s2_c   │
             │  class label    │          │                      │
             └─────────────────┘          └──────────────────────┘
```

---

## Component Explanations

| Component | Role in UGDR |
|---|---|
| **Input 224×224×3** | Standardised resolution for ImageNet backbone compatibility. Grayscale X-rays replicated across 3 channels. |
| **ResNet18 Backbone** | 4 residual stage groups with skip connections preventing vanishing gradients. Pretrained features (edges, textures) transfer effectively to radiology. Full fine-tuning applied. |
| **Global Average Pooling** | Collapses each feature map to a scalar, producing a 512-d vector invariant to spatial position — appropriate for pneumonia which may appear in any pulmonary region. |
| **Dropout (p=0.5)** | During training: regularisation. During MC Dropout inference: stochastic sub-network sampling from the approximate Bayesian posterior. Kept active at inference — the key non-standard design choice. |
| **FC(512→2)** | Linear projection to two logits (NORMAL, PNEUMONIA). Softmax applied externally during MC Dropout so raw probabilities are comparable across passes. |
| **MC Dropout Mode** | T=30 independent passes with different random dropout masks produce a distribution over predictions. Variance of this distribution is the uncertainty signal used for noise detection. |

---

## Why Dropout Is Critical for Uncertainty Estimation

Standard neural network inference is deterministic: given the same input, the model always produces the same output. This means a single forward pass cannot reveal whether a prediction reflects clear visual evidence or overfit pattern memorisation. Monte Carlo Dropout resolves this by treating each forward pass as sampling a distinct sub-network from an approximate Bayesian posterior over model weights. For a reliably-labelled training sample, different sub-networks consistently assign the same class because the visual evidence is unambiguous — producing low variance. For a mislabelled sample, different sub-networks disagree: some respond to the visual evidence (true class) while others reflect the corrupted label signal (wrong class), producing high variance. This asymmetry is the theoretical basis for UGDR's noise detection mechanism. Variance under MC Dropout is not a generic uncertainty signal — it is a specific diagnostic for label-evidence conflict within the training corpus, the precise failure mode that causes clinical AI systems to degrade under real-world annotation noise.

---

---

# DIAGRAM RECREATION GUIDE

## Option A — draw.io (Recommended)

**Download / use online:** https://app.diagrams.net

### Diagram 1 — UGDR Pipeline Setup

**Canvas:** 1200 × 1800 px, Portrait, White background (print) or #1A1A2E (slides)

**Shape list (top to bottom):**

| Stage | Shape | Fill | Border | Label |
|---|---|---|---|---|
| Dataset | Rounded rectangle | #E3F2FD | #1565C0 | "Chest X-Ray Dataset" |
| Partitioning | Plain rectangle | #F3E5F5 | #6A1B9A | "D_train / D_val / D_test" |
| Stage 1 | Thick-border rectangle | #E8F5E9 | #2E7D32 | "Stage 1 — Clean Baseline" |
| Stage 2 | Plain rectangle | #FFEBEE | #C62828 | "Stage 2 — Noise Injection" |
| Stage 3 | Thick-border rectangle | #FFF3E0 | #E65100 | "Stage 3 — Noisy Training" |
| Stage 4 | Hexagon | #EDE7F6 | #4527A0 | "Stage 4 — MC Dropout" |
| Stage 5 | Diamond | #FFF9C4 | #F57F17 | "Stage 5 — Dual Criterion" |
| Flagged | Parallelogram | #FFCDD2 | #B71C1C | "Remove Flagged (~726)" |
| Retained | Parallelogram | #C8E6C9 | #1B5E20 | "D_refined (~2,943)" |
| Stage 6 | Thick-border rectangle | #E8F5E9 | #2E7D32 | "Stage 6 — Retraining" |
| Evaluation | Rounded rectangle | #E1F5FE | #0277BD | "Final Evaluation" |
| Results | Note/callout shape | #F9FBE7 | #558B2F | F1 scores |

**Arrows:** Solid dark grey (#424242), open arrowhead, 1.5pt weight.
Label the two arrows out of Stage 5: "Flagged" (red #C62828) and "Retained" (green #2E7D32).

**Font:** Roboto or Open Sans, 12pt bold for stage titles, 10pt regular for sub-labels.

**Export:** Extras → Edit Diagram (XML) to save source. File → Export → PNG at 300 dpi.

---

### Diagram 2 — Model Architecture Setup

**Canvas:** 900 × 700 px, Portrait

| Component | Shape | Fill | Notes |
|---|---|---|---|
| Input | 3D cube or rounded rect | #B3E5FC | "224×224×3 Input" |
| ResNet18 | Large outer rectangle | #E8EAF6 | Contains 4 sub-boxes |
| Each Stage | Small inner rectangles | #9FA8DA | "Stage N: 2× BasicBlock" |
| GAP | Trapezoid (narrowing down) | #C5CAE9 | "Global Avg Pool → 512d" |
| Dropout | Dashed-border rectangle | #FFE082 | "Dropout p=0.5 ← MC active" |
| FC layer | Rectangle | #A5D6A7 | "Linear 512→2" |
| Split | Diamond | #FFFFFF | "Inference mode?" |
| Training branch | Rounded rect | #EF9A9A | "Softmax + Cross-entropy" |
| MC branch | Rounded rect | #80CBC4 | "T=30 passes → u(x)" |

**Key visual tip:** Add a yellow star or lightning bolt icon inside the Dropout box
to highlight its role as the key methodological novelty. Use a dashed border (not solid)
to visually distinguish it from the backbone layers.

---

## Option B — Microsoft PowerPoint

**Slide setup:** 16:9 widescreen (33.87 × 19.05 cm), blank layout.

**Step-by-step for Diagram 1:**
1. Insert → Shapes → Flowchart: Process (rectangles for stages)
2. Insert → Shapes → Flowchart: Decision (diamond for Stage 5)
3. Insert → Shapes → Lines → Elbow Arrow Connector (all inter-stage arrows)
4. Right-click each shape → Format Shape → Fill colour (use table above)
5. Home → Arrange → Align → Distribute Vertically for even spacing
6. Insert → Text Box for sub-label annotations beside each stage
7. Ctrl+A → Group → right-click → Save as Picture → PNG 300 dpi

**Colour palette for PowerPoint:**

```
Clean/train stages  Fill: #E8F5E9   Border: #43A047   Text: #1B5E20
Noisy stages        Fill: #FFEBEE   Border: #E53935   Text: #B71C1C
MC/Detection stages Fill: #EDE7F6   Border: #7E57C2   Text: #311B92
Evaluation box      Fill: #E1F5FE   Border: #0288D1   Text: #01579B
Flagged arrows      Colour: #C62828
Retained arrows     Colour: #2E7D32
```

**Font recommendations:**
- Stage titles: Calibri Bold 12pt
- Sub-labels: Calibri 9pt, colour #616161
- Arrow labels: Calibri Italic 9pt

**Export:** File → Export → Change File Type → PNG (300 dpi if prompted),
or Save as PDF for lossless vector embedding in LaTeX.

---

## Academic Styling Checklist

- [ ] White background for journal/report submission (dark only for slides)
- [ ] Consistent border weight: 1.5pt for main stage boxes, 1pt for sub-labels
- [ ] Maximum 4 colours (blue=data, red=noisy/flagged, green=clean/retained, purple=uncertainty)
- [ ] Every decision node has labelled outgoing arrows ("Flagged" / "Retained")
- [ ] Captions placed **below** figures in academic papers
- [ ] Captions are self-contained — no reference to surrounding text required
- [ ] PNG at 300 dpi minimum for submission; SVG/PDF for LaTeX
- [ ] Figure numbers in order of appearance in the text (Fig. 1 = pipeline, Fig. 2 = architecture)
- [ ] Consistent font across all figures in the paper

---

*Diagrams ready for insertion into:*
*research_paper_final.md — Section 3 (Methodology), Figures 1 and 2*
