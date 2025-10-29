# Attacking Fake Image Detectors with STIG-PGD

<p align="center">
    <img src="https://github.com/user-attachments/assets/7a03e788-baa7-4ce7-9a09-a8286cd6b6a0" alt="Result 1" width="80%">
    <img src="https://github.com/user-attachments/assets/853d8507-81cc-4dca-ae74-c32cfd47e7fa" alt="Result 3" width="80%">
    <img src="https://github.com/user-attachments/assets/77edcbc0-b746-4c97-9017-db8786bc6590" alt="Result 5" width="80%">
    <img src="https://github.com/user-attachments/assets/57d7eea8-e8c5-4ce6-bdd3-80c8b07ba0c6" alt="Result 7" width="80%">
</p>

## Description
### 🚀 Project Overview
This repository contains the official PyTorch implementation of the STIG-PGD method, a novel adversarial image transformation technique designed to neutralize state-of-the-art fake image detectors.

Our method uniquely combines Spectral Transformation for refinement of Image Generation (STIG) and Projected Gradient Descent (PGD) to create Refined Fake Images that evade detection systems focusing on both spectral artifacts and visual inconsistencies.

### Core Contributions
**Hybrid Attack Superiority**: Achieves a significantly higher attack success rate against fake image detectors compared to single-technique approaches.

**Frequency-Selective Artifact Injection**: Addresses the limitations of traditional PGD by applying adversarial artifacts selectively to the Low-Frequency (LF) spectrum while using STIG to refine the High-Frequency (HF) spectrum.

**Novel Loss Function**: Introduces a hybrid PGD loss that integrates the STIG framework's Reconstruction Loss, ensuring the generated artifacts complement spectral refinement.

**Vulnerability Proof**: Demonstrates the fragility of current fake image detection models, guiding future research toward more robust defense mechanisms.

### STIG-PGD Framework
The framework is largely composed of the **STIG Framework**, the **PGD Framework**, and the **result merging step**.

<img width="1107" height="507" alt="image" src="https://github.com/user-attachments/assets/30c8ea5a-a2b0-48c9-9a33-222c5197cb13" />

**1. STIG Framework**

The STIG component refines the fake image's spectral quality. It uses Fast Wavelet Transform (FWT) for low-loss conversion and Patch-wise Contrastive Learning to align the High-Frequency (HF) spectrum with real images, preserving image structure.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1bc2eea2-e755-4527-9dfa-c9aa22bdc686" width="465" height="223" alt="image" />
</p>

**2. PGD Framework**

PGD generates adversarial perturbations. It updates the gradient toward the Real Class using a Hybrid Loss Function that combines the Detector's Classification Loss with STIG's Reconstruction Loss to produce the PGD Artifact Coefficient.

<p align="center">
<img width="185" height="191" alt="image" src="https://github.com/user-attachments/assets/6ba33806-c174-442f-ad53-4c8ca1042942" />
</p>

**3. Merging (The Core Innovation)**

This step strategically combines the STIG and PGD results to mitigate noise. It applies the PGD Artifact Coefficient only to the Low-Frequency (LF) region of the STIG spectrum, while maintaining the STIG-refined spectrum in the HF region. The combined result yields the final Refined Fake Image via Inverse Wavelet Transform (IWT).

<p align="center">
<img width="234" height="238" alt="image" src="https://github.com/user-attachments/assets/cf7d7738-a6ce-44a1-86e5-afc94455e72d" />
</p>

**4. Evasion Verification**

The final Refined Fake Image is confirmed to be effective by forcing the VtDIF Detector to misclassify it as Real.

<p align="center">
<img width="321" height="113" alt="image" src="https://github.com/user-attachments/assets/e8ef8ffd-2251-4780-897b-a43596edf372" />
</p>

## Performance Comparison
<table>
  <thead>
    <tr>
      <th align="left">Detector Type</th>
      <th align="left">Test Images</th>
      <th align="left">Accuracy</th>
      <th align="left">F1-Score</th>
      <th align="left">Remark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left" rowspan="4"><b>ViT</b></td>
      <td align="left">Real Images (2400) <br> + Original AI Generated Images (2400)</td>
      <td align="left">93.5%</td>
      <td align="left">0.9351</td>
      <td align="left">Baseline accuracy against unaugmented fake images.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + STIG Augmented AI Images (2400)</td>
      <td align="left">81.1%</td>
      <td align="left">0.8107</td>
      <td align="left">Spectral refinement offers moderate evasion.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + PGD Augmented AI Images (2400)</td>
      <td align="left">74.9%</td>
      <td align="left">0.7460</td>
      <td align="left">Adversarial artifacts provide significant evasion.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + STIG-PGD Augmented AI Images (2400)</td>
      <td align="left"><b>43.5%</b></td>
      <td align="left"><b>0.3033</b></td>
      <td align="left"><b>Lowest F1-Score 👑</b></td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
      <th align="left">Detector Type</th>
      <th align="left">Test Images</th>
      <th align="left">Accuracy</th>
      <th align="left">F1-Score</th>
      <th align="left">Remark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left" rowspan="4"><b>DIF</b></td>
      <td align="left">Real Images (2400) <br> + Original AI Generated Images (2400)</td>
      <td align="left">99.6%</td>
      <td align="left">0.9965</td>
      <td align="left">Baseline accuracy against unaugmented fake images.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + STIG Augmented AI Images (2400)</td>
      <td align="left">89.1%</td>
      <td align="left">0.8782</td>
      <td align="left">Spectral refinement offers moderate evasion.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + PGD Augmented AI Images (2400)</td>
      <td align="left">74.7%</td>
      <td align="left">0.6626</td>
      <td align="left">PGD artifacts significantly decrease detection performance.</td>
    </tr>
    <tr>
      <td align="left">Real Images (2400) <br> + STIG-PGD Augmented AI Images (2400)</td>
      <td align="left"><b>68.9%</b></td>
      <td align="left"><b>0.5496</b></td>
      <td align="left"><b>Lowest F1-Score 👑</b></td>
    </tr>
  </tbody>
</table>

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

