# Attacking Fake-Image Detectors with STIG-PGD

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

The final Refined Fake Image is confirmed to be effective by forcing the ViT, DIF Detector to misclassify it as Real.

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

<p align="center">
<img width="80%" alt="1_8400" src="https://github.com/user-attachments/assets/8507f27a-e9a5-4cb3-be8f-50d1b610bef8" />
<img width="80%" alt="1_8300" src="https://github.com/user-attachments/assets/4ce1bbcf-2442-42da-97a9-6d7c9c9f2b03" />
<img width="80%" alt="1_8200" src="https://github.com/user-attachments/assets/47cdf877-cd6f-42a1-b04e-d5b898d790b8" />
<img width="80%" alt="1_8100" src="https://github.com/user-attachments/assets/73c624b4-aeb6-4840-a020-9e495e0eb66d" />

</p>

## Requirements and Installation

We recommend running our code using:
* NVIDIA GPU + CUDA
* Python 3, Anaconda

To install our implementation, clone our repository and run following commands to install necessary packages:
   ```shell script
conda create -n stig-pgd python=3.13.3
conda activate stig-pgd
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
### Preparing Dataset
Put real and fake datasets in the folder ```datasets/```.

The real images should be located in ```datasets/{dataset_name}/real/```.<br>
The generated images should be located in ```datasets/{dataset_name}/fake/```.<br>
We suppose the type of image file is ```.png```.<br>
And its size is 256 * 256.<br>
The real and fake datasets must be prepared with an equal number of samples.<br>

We generated the AI images using the Stable Diffusion v1.5 Realistic Vision (SD 1.5 RV) model(https://github.com/lllyasviel/Fooocus).<br>
For the real dataset, we used cat images from https://www.kaggle.com/datasets/crawford/cat-dataset<br> and human images from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00002.png.<br>
You can generate or acquire images using other Stable Diffusion models, and you should use real images that match the domain of those AI pictures.

## Getting Started

### Training

```shell script
python train.py --size 256 --data {dataset_name} --epoch 10 --batch_size 1 --lr 0.00008 --device {gpu_ids} --dst {experiment_name}
```
Enter the command. You can change the GPU device by modifying the option ```--device```.<br>

The sampled results are visualized in the ```results/{experiment_name}/sample/``` during the training.<br> 

After training, the results image and magnitude spectrum will be saved at each folder in ```results/{experiment_name}/eval/```.<br>

Inside the ```eval``` folder, you will find three subfolders: ```clean```, ```denoised```, and ```noise```. The ```clean``` folder contains the original real images, the ```noise``` folder contains the original AI-generated images, and the ```denoised``` folder contains the refined AI images.
### Inference
```shell script
python inference.py --size 256 --inference_data datasets\{inference_dataset_name}\fake --device {gpu_ids} --inference_params results\{experiment_name}\parameters_{}_epoch.pt --dst {experiment_name}
```
Enter the command.

The inference results are saved at ```results/{experiment_name}/inference/```.<br>

Put the folder path of the inference data into ```--inference_data```.<br>

And also put the path of model parameters onto ```--inference_params```.<br>

Inside the ```inference``` folder, you will find two subfolders: ```noise```, and ```denoised```. The ```noise``` folder contains the original AI-generated images, and the ```denoised``` folder contains the refined AI images.

### Detection(Evaluation)
We have prepared two detectors: the ViT and DIF detectors.

**ViT**
```shell script
python detect.py --is_train False --classifier vit --lr 0.0002 --size 256 --device {gpu_ids} --class_batch_size 32 --dst {experiment_name} --eval_root results\{experiment_name}\inference\denoised
```


**DIF**
```shell script
python detect.py --is_train False --classifier dif --lr 0.0002 --size 256 --device {gpu_ids} --class_batch_size 32 --dst {experiment_name} --eval_root results\{experiment_name}\inference\denoised
```

## Reference
| Type | Title & Source | GitHub / Codebase |
| :--- | :--- | :--- |
| **STIG** (Base Method) | Lee, S., Jung, S. W., & Seo, H. (2024). **Spectrum translation for refinement of image generation (STIG) based on contrastive learning and spectral filter profile.** *In Proceedings of the AAAI Conference on Artificial Intelligence.* | [ykykyk112/STIG](https://github.com/ykykyk112/STIG) |
| **PGD** (Codebase) | Madry, A., et al. (2018). **Towards deep learning models resistant to adversarial attacks.** *In International Conference on Learning Representations (ICLR).* <br> (Implementation used for this project) | [Harry24k/adversarial-attacks-pytorch](https://github.com/Harry24k/adversarial-attacks-pytorch/tree/master) |
## Team Introduction
| Name | Student ID | Major |
| :--- | :--- | :--- |
| **Hyeonjun Cha** | 202011378 | Computer Science and Engineering |
| **Euntaek Lee** | 201911203 | Computer Science and Engineering |
| **Kyeongbeom Park** | 202011291 | Computer Science and Engineering |

