# Attacking Fake Image Detectors with STIG-PGD
## Description
🚀 **Project Overview and Goals**

This project introduces a **novel adversarial image transformation technique (STIG-PGD)** designed to successfully neutralize existing fake image detectors based on both Artifact and Spectrum analysis.

Goal: To achieve a significantly higher attack success rate than conventional single-attack methods by uniquely combining the Spectral Transformation for Image Generation (STIG) and Projected Gradient Descent (PGD) techniques.

Significance: Our work aims to demonstrate the vulnerabilities of current detection models, thereby encouraging future researchers to design more robust and resilient defense strategies.

💡 Proposed Method
The newly proposed STIG-PGD technique generates a Refined Fake Image by merging the effects of spectrum improvement and adversarial artifact injection. This refined image forces the detector to misclassify the fake input as belonging to the Real Class.

1. STIG Framework (Spectrum Improvement)
The STIG framework focuses on refining the image's spectral characteristics.

Improved Transformation: We replace the traditional Fourier Transform used in standard STIG with the low-loss Fast Wavelet Transform (FWT) when converting to the spectral domain. This better preserves structural properties.

Targeted HF Transformation: Since spectral differences mainly appear in the High-Frequency (HF) regions, we apply Patch-wise Contrastive Learning. This ensures the transformation is concentrated in the HF areas, effectively making the fake image's spectrum resemble that of a real image.

2. PGD Framework (Adversarial Artifact Generation)
The PGD framework generates artifacts to cause misclassification.

Hybrid Loss Function (Novelty): Unlike standard PGD, which only uses the detector's Classification Loss, we incorporate the Reconstruction Loss from the STIG framework. This ensures the STIG's spectral improvement is factored into the generation of the PGD artifacts.

Adversarial Artifact Coefficient: The framework calculates the gradient of the detector's Loss and Weights, updating the image in the direction of the Real Class. This process yields an optimal Adversarial Artifact Coefficient that induces misclassification.

3. Final Image Refinement and Combination
The core innovation is the frequency-selective combination of the STIG and PGD results.

Low-Frequency (LF) Modification: We selectively apply the PGD artifact coefficients only to the LF region of the spectrum. This prevents the conspicuous, overall noise often seen in standard PGD, which results from adding artifacts uniformly across the image.

High-Frequency (HF) Preservation: The HF region maintains the STIG-transformed spectrum. This blocks noise generation while successfully improving the spectral characteristics related to image manipulation.

Finally, the combined spectral components are inversely transformed to create the Refined Fake Image, which is then submitted to the detector to be successfully neutralized (misclassified as Real).
## Performance Comparison
## Additional Results
## Requirements and Installation
## Getting Started
## Reference
## Team Introduction
| Name | Student ID | Major |
| :--- | :--- | :--- |
| **Hyeonjun Cha** | 202011378 | Computer Science and Engineering |
| **Euntaek Lee** | 201911203 | Computer Science and Engineering |
| **Kyeongbeom Park** | 202011291 | Computer Science and Engineering |

