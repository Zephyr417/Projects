# TissueNet Cell Segmentation Project Context

## Purpose

This project is preparation for a PhD interview with Dr. Denis Schapiro's
Spatial Omics Group at Heidelberg University Hospital.

The project is not only an exercise in training a U-Net. It should:

1. Explain the role of cell segmentation in a spatial omics pipeline.
2. Demonstrate the ability to build a medical image segmentation pipeline in PyTorch.
3. Study generalization across tissue types.
4. Analyze failure under domain shift.
5. Connect uncertainty, failure analysis, and applicability boundaries to trustworthy AI.

## Working title

**When Does Cell Segmentation Generalize? A TissueNet Study Across Tissues**

## Research questions

- **RQ1:** Can a simple U-Net provide a reasonable baseline for cell/nuclear segmentation on TissueNet?
- **RQ2:** How much does segmentation performance degrade on unseen tissue types?
- **RQ3:** Can uncertainty or image-level indicators identify unreliable predictions?

## Scope and current phase

- Approximate duration: four weeks.
- Current phase: week 1, data understanding and baseline pipeline construction.

## Principles for future implementation and experiments

- Keep a simple, reproducible PyTorch U-Net as the primary baseline.
- Treat cross-tissue generalization as a first-class experimental design requirement.
- Prevent leakage between training and evaluation, especially at experiment/sample level.
- Report results per tissue as well as aggregated results; do not let large tissues dominate conclusions.
- Preserve instance labels even if an initial baseline temporarily uses binary semantic targets.
- Record preprocessing, split definitions, random seeds, checkpoints, and package versions.
- Include qualitative failure analysis and uncertainty/reliability analysis, not only mean accuracy.
- Make claims about trustworthy AI only when supported by measurable reliability or failure-detection evidence.

## Initial TissueNet v1.1 audit notes

- Image/label shapes currently observed:
  - train: `(2580, 512, 512, 2)`
  - validation: `(3118, 256, 256, 2)`
  - test: `(1324, 256, 256, 2)`
- The metadata arrays store headers as rows rather than NumPy field names.
- `test_meta` contains four header rows in total. All repeated header rows must be removed before aligning metadata with the 1324 test images.
- The packaged train, validation, and test splits contain the same tissue categories and substantial overlap in experiment/file identifiers. They therefore cannot, by themselves, answer the unseen-tissue generalization question.
- A separate tissue-held-out evaluation protocol is required for RQ2, with grouping checks to prevent experiment/sample leakage.
