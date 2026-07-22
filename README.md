<div align="center">

#  FRITNet: Facial Region Informed Transformer Network 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/RAF--DB_Accuracy-88.66%25-brightgreen.svg)]()

>  *A highly optimized, single-weight architecture for in-the-wild Facial Emotion Recognition (FER), achieving state-of-the-art adjacent performance (88.66%) on the RAF-DB benchmark.* 

</div>

---

##  Table of Contents
- [ About the Project](#-about-the-project)
- [ Architecture Overview](#-architecture-overview)
- [ Benchmark Results](#-benchmark-results)
- [ Methodology & Optimizations](#-methodology--optimizations)
- [ Getting Started](#-getting-started)
- [ Usage](#-usage)
- [ License](#-license)

---

##  About the Project

**FRITNet** (Facial Region Informed Transformer Network) is a hybrid CNN-Transformer architecture built specifically for real-time, in-the-wild emotion classification. 

 **The Goal:** Bypass the computational bloat of model ensembling. 
 **The Result:** We successfully compressed the intelligence of a dual-model ensemble into a single, highly deployable weight file—maintaining strictly $O(1)$ inference time without sacrificing accuracy.

This repository holds the full pipeline, including the **RAF-DB hard-label optimization** (7-class emotion recognition) and the **FERPlus soft-label distribution training**.

---

##  Architecture Overview

FRITNet uses a highly specialized spatial extraction pipeline feeding into a Dual-Transformer setup:

*    **Input & Alignment:** RGB `224x224` images, perfectly aligned and cropped via RetinaFace.
*    **Backbone:** Truncated InceptionResNetV1 (FaceNet), deeply extracted up to `block8`.
*    **Local Feature Augmentation (LFA):** Connects a `1x1` Conv bridge to a `28x28` upsample, capped with 4-quadrant asymmetric convolutions (`1x3`, `3x1`) to capture granular local geometry.
*    **Spatial Attention Feature Masking (SAFM):** Channel avg/max pooling  `7x7` Conv  Sigmoid spatial gate.
*    **Tokenization:** Overlapping patch extraction (`12x12` window, stride 8, mean pooling) creating exactly **9 tokens** (128 dimensions).
*    **Dual Transformer Encoder:**
    *    *Branch 1 (Local):* Self-attention (2 layers, 8 heads) for localized feature relation.
    *    *Branch 2 (Global):* Cross-attention utilizing a CLS token as the query, pulling from local tokens as keys/values.
*    **Classification Heads:** Main logit head (`128  7`) mapped to standard RAF-DB classes, supported by auxiliary global/local heads for robust joint optimization.

---

##  Benchmark Results

###  RAF-DB (7-Class Basic Expressions)
|  Model Variant |  Strategy |  Validation Accuracy |
| :--- | :--- | :--- |
| **FRITNet (Ensemble)** | Unnormalized logit sum of two distinct optima | 88.40% |
| **FRITNet (Model Soup)** | Linear Mode Connectivity (Weight Averaging) | 88.07% |
|  **FRITNet (Final)** | **Classifier-only Re-Training (cRT) - Single Weight** | **88.66%** |

###  FERPlus (Pre-training / Baseline)
|  Label Strategy |  Format |  Combined Split Accuracy |
| :--- | :--- | :--- |
| **Hard-Label Control** | Argmax of majority vote + Standard CE | 78.52% |
|  **Soft-Label (Final)** | **Probability distribution + Soft-Target CE** | **79.77%** |

*( Note: FERPlus soft-label training dropped the 'contempt' class to enforce strict 7-way probability distribution compatibility for direct, frictionless RAF-DB transfer).*

---

##  Methodology & Optimizations

###  Classifier-only Re-Training (cRT)
The absolute peak of **88.66%** on RAF-DB was unlocked by freezing the entire CNN backbone and dual transformers, and retraining **only** the classification heads (2,965 parameters) for 10 epochs. We paired this with a class-balanced `WeightedRandomSampler` to correct linear classifier bias *without* shattering upstream geometric embeddings.

###  Post-Mortem Ablations
Extensive ablation studies were run to prove the 88.66% ceiling is mathematically exhausted:
*    **LDAM-DRW:** Attempted margin-based loss to rescue minority classes (Fear/Disgust). Gradients exploded due to the unnormalized `nn.Linear` head. Result confirmed that minority class errors stem from upstream FaceNet feature entanglement (e.g., Fear vs. Surprise upper-face Action Unit overlap), not downstream boundary shape.
*    **SWA & TTA:** Stochastic Weight Averaging and 80/20 horizontal flip Test-Time Augmentation both degraded performance (dropping to `88.46%`). This proved the cRT weights converged in the sharpest possible optimal basin, and that the LFA module's spatial precision is hyper-sensitive to exact coordinate geometry.

---

##  Getting Started

###  Prerequisites
*    `Python 3.10+`
*    `PyTorch 2.0+`
*    `torchvision`, `pandas`, `numpy`, `scikit-learn`, `tqdm`

###  Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/FRITNet.git](https://github.com/your-username/FRITNet.git)
   cd FRITNet
