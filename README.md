# MMFDAHNet

**MMFDAHNet: A Multi-band Multi-feature Fusion Domain Adversarial Hybrid Neural Network for Cross-Subject Spatial Cognitive EEG Signals Classification**

This repository contains the official PyTorch implementation of **MMFDAHNet**. The model is designed to decode cross-subject spatial cognitive states using electroencephalography (EEG) signals. By effectively integrating multi-band and multi-view features (temporal dynamics and brain functional connectivity) and employing a domain adversarial strategy, MMFDAHNet mitigates inter-subject variability and achieves state-of-the-art generalization performance.

## 🚀 Repository Structure

To keep the repository clean and focused, we provide the core implementation files and a pre-trained testing pipeline:

* `model.py`: Contains the core MMFDAHNet architecture, including the Temporal Feature Extraction Module, Brain Functional Connectivity Feature Extraction Module, Frequency Band Fusion Module (FFM), and Domain Adversarial Predictors.
* `FCF.py`: Implementation of the novel Feature Complementarity Fusion (FCF) module, which uses a two-stage cross-attention mechanism to explicitly extract differential features and inject common information.
* `datasets.py`: Data processing script for loading, normalizing, and formatting the multi-band EEG signals.
* `train.py`: The complete training pipeline implementing Leave-One-Subject-Out (LOSO) cross-validation and two-phase domain adversarial training.
* `test_sub6.py`: A ready-to-use testing script to evaluate the model's performance on a specific target domain (Subject 6).
* `feature_extractor_sub6.pth` and `classifier_sub6.pth`:  The pre-trained model weights for Subject 6
* `sub6_data.npy` and `sub6——labels.npy`:  The processed data and labels of Subject 6

## 🛠️ Dependencies

Ensure you have the following dependencies installed:

```bash
# Core requirements
torch>=1.12.0
torch-geometric
numpy
pandas
scikit-learn
openpyxl
```

## 📊 Data Preparation

The model expects multi-band EEG data (Delta, Beta2, Gamma) formatted in `.xlsx` files. The data should be organized with columns representing EEG channels and rows representing sampling points.

*Note: Due to privacy and ethical guidelines, the original clinical EEG datasets are not directly included in this repository. Users can structure their own EEG data following the loading logic in `datasets.py` or request the original dataset from the corresponding authors upon reasonable request.*

## 💻 Usage

### Quick Test (Subject 6)

You can directly evaluate the model's performance on Subject 6 using the provided pre-trained weights. This will load the weights, perform a forward pass on Subject 6's data, and output comprehensive metrics (Accuracy, F1, Recall, Precision, AUC) along with the confusion matrix.

Bash

```
python test_sub6.py
```

### Full Training (Leave-One-Subject-Out)

To train the model from scratch across all subjects using the domain adversarial strategy, run:

Bash

```
python train.py
```

This script will iterate through all subjects, utilizing each as a target domain while using the rest as the source domain, and will save the final evaluation results to `results.xlsx`.

## 📜 Acknowledgments and References

Our code implementations for specific network modules were inspired by and adapted from the following outstanding open-source works. We express our gratitude to the authors for sharing their research:

1. **Rethinking Cross-Attention for Infrared and Visible Image Fusion** 
   - Paper: https://arxiv.org/html/2401.11675v1
2. **Subject-independent emotion recognition based on EEG frequency band features and self-adaptive graph construction** 
   - Paper: https://www.mdpi.com/2076-3425/14/3/271
