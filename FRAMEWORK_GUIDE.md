# Building a New Multi-Modal Deep Learning Framework (DNA + sn/scATAC-seq)

This guide maps DeepDETAILS components to their functional roles, then proposes how to design a **new** framework with similar inputs while keeping your implementation original.

## 1) What is the key code in this repository?

If your goal is to understand the architectural backbone, focus on:

1. **Model blocks and fusion logic**
   - `src/deepdetails/model/deconvolution.py`
   - Core sequence path: motif-style Conv1d trunk with residual/dilated layers.
   - Core chromatin path: per-cluster ATAC handling, per-cluster gating, and profile/count heads.

2. **Reusable residual modules**
   - `src/deepdetails/model/__init__.py`
   - Defines residual 1D convolutions used in the main model.

3. **Training objective and optimization loop**
   - `src/deepdetails/model/wrapper.py`
   - Lightning module with profile loss, correlation-driven regularization, prior consistency penalty, and optimizer/scheduler setup.

4. **Data contracts and sampling behavior**
   - `src/deepdetails/data.py`
   - Defines how sequence/ATAC/targets/cluster-load priors are assembled and returned.

5. **Loss utilities**
   - `src/deepdetails/model/loss.py`
   - Custom loss helpers and matrix utilities used by training.

## 2) Should you “just modify current code” for a new publication?

Short answer: **No, not as your primary strategy**.

- Ethically and scientifically, if your method is derived from this implementation, you should cite it.
- Simply renaming layers or changing a few constants is usually not a sufficiently novel method.
- You can still build a new system inspired by high-level ideas, but do a **fresh implementation** and target **clear methodological novelty**:
  - Different modality fusion mechanism,
  - Different decoder/objective,
  - Different uncertainty model,
  - Different biological constraints and evaluation protocol.

## 3) Necessary components of a deep learning framework for DNA + sn/scATAC

At minimum, you need seven layers of design.

1. **Problem formulation**
   - Define outputs precisely: deconvolution proportions, per-cell-type profiles, counts, denoised accessibility, etc.
   - Decide which outputs are supervised vs weakly supervised.

2. **Data model / schema**
   - Region definition, coordinate system, input window length.
   - Modalities per region: one-hot DNA, ATAC pseudo-bulk/per-cluster tracks, optional priors.
   - Split logic to avoid leakage (chromosome-level splits are common).

3. **Neural architecture**
   - DNA encoder.
   - ATAC encoder.
   - Fusion block.
   - Multi-head decoders (e.g., profile + counts, or occupancy + intensity).

4. **Loss stack**
   - Task losses (profile/count regression, classification, etc.).
   - Calibration/consistency losses (cross-modal consistency, sparsity, smoothness, prior alignment).
   - Regularization preventing branch collapse and over-correlation.

5. **Training framework**
   - Batch assembly and augmentation.
   - Optimizer, schedule, mixed precision, early stopping.
   - Reproducibility controls (seeds, deterministic options, config snapshots).

6. **Evaluation framework**
   - Metrics per task and per cell type.
   - Generalization tests (held-out chromosomes, donors, assays).
   - Ablations for each architectural component.

7. **Deployment/interpretation outputs**
   - Export tracks/tensors in standard formats.
   - Feature importance / motif attribution / uncertainty reporting.
   - Versioned model artifacts and metadata.

## 4) Suggested architecture rewrite (original design template)

Below is a robust blueprint you can implement from scratch.

### A. Inputs
- **DNA branch**: one-hot sequence `B x 4 x L`.
- **ATAC branch**: cluster-conditioned accessibility `B x C x L` (or sparse matrix form).
- **Optional metadata**: donor, batch, assay, QC covariates.

### B. Encoders
1. **DNA encoder (new design)**
   - Conv stem -> dilated residual stack -> lightweight Transformer (local attention windows).
   - Output: `H_dna in R^(B x D x L')`.

2. **ATAC encoder (new design)**
   - Two options:
     - Track-based Conv encoder over `C x L`, or
     - Cell graph encoder (if cell-level matrix retained) with pooling to regional representation.
   - Output: `H_atac in R^(B x D x L')` and optional cluster embedding `E_c`.

### C. Fusion (where novelty can live)
Use one of these instead of simple scaling:
- **Cross-attention fusion**: DNA queries, ATAC keys/values.
- **FiLM conditioning**: ATAC-conditioned affine modulation of DNA channels.
- **Mixture-of-experts** with cluster-aware router.

### D. Decoders
- **Profile decoder**: predicts strand-aware base-resolution distribution.
- **Count/intensity decoder**: predicts per-target totals.
- **Optional proportion decoder**: predicts cell-type mixture weights if not provided.

### E. Losses
- Profile distribution loss (e.g., multinomial/NLL or transformed MSE).
- Count regression loss (Poisson/NB or RMSLE depending on noise profile).
- Cross-modal consistency loss between aggregated cluster predictions and bulk target.
- Diversity/orthogonality regularizer across cell-type heads to avoid collapse.
- Optional prior-matching loss (if using external biological prior matrices).

## 5) Migration strategy from this repository without copy-paste dependence

1. Write your own data contract first (tensor names/shapes and split policies).
2. Re-implement blocks from clean pseudocode, not line-by-line translation.
3. Replace at least one major subsystem in each tier:
   - encoder,
   - fusion,
   - decoder,
   - loss.
4. Add ablations proving each replacement matters.
5. Cite prior methods that inspired design principles, even with fresh code.

## 6) Practical MVP roadmap (8-10 weeks)

1. Week 1-2: data schema + loader + reproducible split tooling.
2. Week 3-4: baseline dual-branch model (DNA+ATAC) + two-head decoder.
3. Week 5: robust loss balancing + metric dashboard.
4. Week 6-7: introduce your novel fusion module + ablations.
5. Week 8: held-out donor/chromosome evaluation + error analysis.
6. Week 9-10: interpretation tooling + packaging + manuscript figures.

## 7) Publication and attribution guidance

- If your concept is inspired by DeepDETAILS or related models, cite them.
- Novel publications are about **new scientific contribution**, not avoiding citation.
- A strong paper has:
  - clear novelty statement,
  - theoretically/biologically motivated architecture,
  - rigorous benchmarking and ablations,
  - reproducible code/data protocol.
