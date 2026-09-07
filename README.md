# Explainable Deep Learning for Fine-Grained Flower Species Classification

A comparative deep learning study for five-class flower species classification using a custom Convolutional Neural Network (CNN), MobileNetV2, and EfficientNetB0, with Grad-CAM-based explainability analysis.

## 📌 Overview

Flower species classification is a challenging computer vision task because different species can exhibit similar colors, shapes, petal structures, and visual patterns.

This project investigates whether transfer learning can improve flower classification performance compared with a lightweight custom CNN baseline. In addition to classification accuracy, the study evaluates computational efficiency and uses Grad-CAM to analyze the visual regions influencing model predictions.

The project compares:

- Custom CNN
- MobileNetV2 with frozen backbone
- MobileNetV2 with fine-tuning
- EfficientNetB0 with frozen backbone
- EfficientNetB0 with fine-tuning

## 🎯 Objectives

The main objectives of this study are:

1. Develop a lightweight CNN baseline for flower classification.
2. Evaluate ImageNet-pretrained MobileNetV2 and EfficientNetB0 models.
3. Compare frozen-backbone and fine-tuned transfer-learning approaches.
4. Analyze classification performance using accuracy, precision, recall, F1-score, and confusion matrices.
5. Compare model parameter count, storage size, and inference latency.
6. Apply Grad-CAM to investigate model decision-making and failure cases.

## 🌸 Dataset

The dataset contains five flower classes:

- Daisy
- Dandelion
- Rose
- Sunflower
- Tulip

After duplicate and near-duplicate screening, the final dataset contains **4,298 images**.

### Dataset Split

| Split | Images |
|---|---:|
| Training | 3,007 |
| Validation | 643 |
| Test | 648 |
| **Total** | **4,298** |

The dataset was divided using a fixed random seed (`42`) with an approximately 70/15/15 train-validation-test split.

### Dataset Source

The original dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/alxmamaev/flowers-recognition

The local dataset used in this study differs from the currently displayed Kaggle dataset statistics. Therefore, the exact dataset version used for the experiments is documented through the final cleaned and split dataset statistics above.

## 🧹 Data Cleaning

Before model training, dataset quality was examined through:

- Image integrity checking
- Exact duplicate detection using MD5 hashing
- Perceptual near-duplicate screening using perceptual hashing (pHash)

Duplicate and near-duplicate images were removed before creating the final train-validation-test split.

The final cleaned dataset contains **4,298 images**.

## 🧠 Models

### 1. Custom CNN Baseline

A lightweight custom CNN was implemented using:

- Convolutional layers
- Max pooling
- Global Average Pooling
- Fully connected layers
- Dropout regularization
- Data augmentation

The baseline contains **110,405 trainable parameters**.

### 2. MobileNetV2

ImageNet-pretrained MobileNetV2 was evaluated in two configurations:

- Frozen backbone
- Fine-tuned backbone

For fine-tuning, the final 30 layers of the backbone were made trainable.

### 3. EfficientNetB0

ImageNet-pretrained EfficientNetB0 was also evaluated using:

- Frozen backbone
- Fine-tuned backbone

For fine-tuning, the final 30 layers of the backbone were made trainable.

## 📊 Results

The best-performing model was the fine-tuned EfficientNetB0.

| Model | Test Accuracy | Test Loss | Parameters |
|---|---:|---:|---:|
| Custom CNN | 70.06% | 0.7326 | 110,405 |
| MobileNetV2 Frozen | 87.19% | 0.3363 | 2,422,597 |
| MobileNetV2 Fine-Tuned | 90.74% | 0.2643 | 2,422,597 |
| EfficientNetB0 Frozen | 93.52% | 0.1867 | 4,214,184 |
| EfficientNetB0 Fine-Tuned | **93.83%** | **0.1837** | 4,214,184 |

### Best Model

**EfficientNetB0 Fine-Tuned**

- Test Accuracy: **93.83%**
- Weighted F1-score: **93.86%**
- Macro F1-score: **93.73%**

## 🔍 Class-Wise Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Daisy | 97.22% | 91.30% | 94.17% |
| Dandelion | 96.23% | 96.23% | 96.23% |
| Rose | 86.72% | 94.87% | 90.61% |
| Sunflower | 96.23% | 92.73% | 94.44% |
| Tulip | 93.20% | 93.20% | 93.20% |

The primary remaining classification difficulty was observed between Rose and Tulip, with 13 combined misclassifications in the final EfficientNetB0 model. This suggests that fine-grained morphological similarities between these classes remain challenging for the classifier.
## ⚡ Computational Efficiency

Inference measurements were performed on the experimental Colab GPU environment.

| Model | Parameters | Saved Model Size | Inference Time/Image | Throughput |
|---|---:|---:|---:|---:|
| Custom CNN | 110,405 | 1.326 MB | 3.30 ms | 303.23 img/s |
| MobileNetV2 Fine-Tuned | 2,422,597 | 22.745 MB | 7.89 ms | 126.75 img/s |
| EfficientNetB0 Fine-Tuned | 4,214,184 | 29.589 MB | 11.74 ms | 85.15 img/s |

These results demonstrate a trade-off between predictive performance and computational efficiency.

The custom CNN provides the lowest computational cost, while EfficientNetB0 provides the highest classification accuracy among the evaluated models. MobileNetV2 provides an intermediate accuracy-efficiency balance.

> Inference latency is hardware-dependent and should not be interpreted as a universal deployment speed.

## 🔬 Explainability with Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualize the regions contributing to model predictions.

The analysis included:

- Correctly classified Rose images
- Correctly classified Tulip images
- Rose → Tulip misclassification
- Tulip → Rose misclassification

Correct predictions generally showed activation over relevant floral regions.

Some failure cases showed different patterns. A selected Rose → Tulip error exhibited substantial activation over dominant contextual regions when the target flower was relatively small. In contrast, a selected Tulip → Rose error showed activation directly over the flower structure despite the incorrect prediction.

These observations suggest that classification errors can arise from both contextual influence and fine-grained visual similarity.

## 📁 Repository Structure

```text
flower-species-classification/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   └── flower_classification.ipynb
│
├── results/
│   ├── flower_model_results.csv
│   └── efficientnetb0_finetuned_classification_report.csv
│
└── figures/
    ├── efficientnetb0_confusion_matrix.png
    ├── efficientnetb0_gradcam_4_cases.png
    ├── efficientnetb0_training_accuracy.png
    └── efficientnetb0_validation_loss.png

## 🛠️ Technologies**

- Python
- TensorFlow 2.20
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- ImageHash
- Google Colab
- GitHub

## 🔁 Reproducibility

The experiments were conducted using a fixed dataset split and a random seed of `42`.

The final cleaned dataset contains 4,298 images and was divided into:

- Training: 3,007 images
- Validation: 643 images
- Test: 648 images

All models were trained using the same train-validation-test partition to ensure a controlled comparison.

The repository provides the main experimental notebook, model comparison results, classification metrics, confusion matrix, training curves, and Grad-CAM visualizations.

## ⚠️ Limitations

- The dataset contains only five flower classes and may not represent the full diversity of real-world flower species.
- The dataset size and visual diversity are limited compared with large-scale image datasets.
- The reported inference latency is specific to the experimental hardware environment.
- Grad-CAM provides qualitative visual explanations and does not establish causal relationships between image regions and predictions.
- External validation on an independent flower dataset was not performed.
- Statistical evaluation across multiple independent training runs was not included.

## 🚀 Future Work

Future extensions of this work may include:

- Evaluation on larger and more diverse flower datasets
- External validation using independent datasets
- Inclusion of additional flower species
- Quantitative evaluation of explainability methods
- Robustness testing under different backgrounds and lighting conditions
- Model compression and quantization for edge deployment
- Comparison with additional lightweight architectures
- Multiple training runs for statistical significance analysis
- Investigation of fine-grained feature extraction and attention mechanisms

## 🔬 Research Focus

This project focuses on the combined evaluation of:

1. Classification performance
2. Transfer learning effectiveness
3. Computational efficiency
4. Model explainability

Rather than evaluating models only by accuracy, the study considers predictive performance, computational requirements, and visual explanations together.

## 📌 Project Status

**Status:** Experimental study completed.

The current best-performing model is the fine-tuned EfficientNetB0, achieving **93.83% test accuracy** and **93.86% weighted F1-score** on the cleaned five-class flower dataset.

## 📚 References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.

2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.

3. Selvaraju, R. R., Cogswell, M., Das, A., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

## 👤 Author

**Harish Alwar S**

This repository contains the experimental implementation and analysis associated with the flower species classification study.
