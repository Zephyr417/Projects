# TissueNet Cell Segmentation Project

## Project Summary

This project explores cell segmentation as a foundational step in a spatial omics analysis pipeline. The main goal is to understand how well a deep learning-based segmentation model can detect cells and nucleus from microscopy images, how this capability generalizes across different tissue types, and whether the model can identify its unreliabale predictions. In parallel, the project aims to develop a simple but effective U-Net-based baseline and evaluate its reliability under distribution shift.

## Purpose

The project is designed to:

1. Understand the role of cell segmentation in a spatial omics workflow.
2. Develop and train a U-Net model for cell/nuclear segmentation in PyTorch.
3. Study generalization performance across unseen tissue types.
4. Analyze model failure cases and uncertainty under domain shift.
5. Connect segmentation quality with trustworthy AI and downstream biological interpretation.

## Working Title

**Can Cell Segmentation Generalize and identify failure? A TissueNet Study Across Tissues**

## Research Questions

- **RQ1:** Can a simple U-Net provide a strong baseline for cell and nuclear segmentation on TissueNet?
- **RQ2:** How much does segmentation performance degrade on unseen tissue types?
- **RQ3:** Can uncertainty or image-level indicators help identify unreliable predictions?

## Project Approach

The study will be organized around the three research questions in a stepwise manner:

1. **RQ1: Baseline segmentation performance**
   - Train a simple, reproducible PyTorch U-Net on the available training data.
   - Evaluate segmentation quality using standard metrics such as Dice score and related overlap measures.
   - Establish a baseline performance level for cell and nuclear segmentation on TissueNet.

2. **RQ2: Generalization to unseen tissues**
   - Test the trained model on a tissue-held-out evaluation setting, with colon held out as a representative out-of-distribution case.
   - Compare performance on seen versus unseen tissue types to quantify generalization degradation.
   - Avoid leakage between training and evaluation, especially at the experiment or sample level.

3. **RQ3: Uncertainty and failure analysis**
   - Examine uncertainty or image-level indicators to identify cases where predictions are likely to be unreliable.
   - Analyze failure cases under domain shift and assess whether uncertainty can support more trustworthy segmentation.
   - Report results by tissue as well as in aggregate, and preserve instance-level information where possible.

Additional implementation principles:
- Record preprocessing steps, split definitions, random seeds, checkpoints, and package versions.
- Use semantic segmentation as an initial baseline while preserving the possibility of later instance-level analysis.

## Initial Dataset Audit Notes

- Observed image/label shapes:
  - train: `(2580, 512, 512, 2)`
  - validation: `(3118, 256, 256, 2)`
  - test: `(1324, 256, 256, 2)`
- Metadata arrays currently store headers as rows rather than NumPy field names.
- The test metadata contains repeated header rows and requires cleaning before alignment with the test images.
- The provided train, validation, and test splits share similar tissue categories and overlapping experiment/file identifiers, so they are not sufficient on their own to evaluate unseen-tissue generalization.
- A separate tissue-held-out evaluation protocol is needed for robust generalization analysis.

## Expected Outcome

The project is expected to produce a baseline segmentation model, a clearer understanding of cross-tissue generalization, and initial insights into uncertainty and failure behavior for future trustworthy AI applications in spatial omics.
