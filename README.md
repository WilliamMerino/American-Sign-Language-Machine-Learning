# American Sign Language Image Classification

## Project Overview

This project focuses on applying machine learning and deep learning techniques to classify hand gesture images representing letters in American Sign Language (ASL). Using the Sign Language MNIST dataset, convolutional neural networks (CNNs) are trained to recognize static hand signs.

The dataset includes **24 classes (A–Y excluding J and Z)**, as these letters require motion and cannot be captured in static images.

---

## Repository Structure

| Component | Description |
|----------|------------|
| `sign_language_compare_models_scheduler_test.py` | Main script comparing multiple models and scheduler configurations |
| `sign_language_customcnn_only.py` | Clean script for training a single Custom CNN model |
| `data/` | Contains training and test CSV datasets |
| `outputs_compare/` | Outputs for model comparison (plots, metrics, reports) |
| `outputs_customcnn_only/` | Outputs for single model run |
| `README.md` | Project documentation |

---

## Project Workflow

| Step | Description |
|------|------------|
| 01 | Data loading and preprocessing |
| 02 | Dataset construction and augmentation |
| 03 | Model design (LeNet-5 and Custom CNN) |
| 04 | Training pipeline implementation |
| 05 | Validation and early stopping |
| 06 | Learning rate scheduling experiments |
| 07 | Model evaluation and comparison |
| 08 | Visualization (EDA, confusion matrices, learning curves) |
| 09 | Final model selection |
| 10 | Result analysis and reporting |

---

## Methodology

### Data Preprocessing
- Conversion from CSV to image tensors (28×28 grayscale)
- Normalization of pixel values to `[0, 1]`
- Label remapping to contiguous indices

### Data Splitting
- Training set: 85%  
- Validation set: 15%  
- Test set: separate dataset  

### Data Augmentation
Applied only to training data:
- Small brightness variations  
- Gaussian noise  

This improves generalization while preserving hand structure.

---

## Models

### LeNet-5
A classical convolutional neural network used as a baseline:
- Two convolutional layers
- Average pooling
- Fully connected classifier

### Custom CNN
A deeper and more robust architecture:
- Multiple convolutional blocks (32 → 64 → 128 filters)
- Batch normalization for stability
- Dropout for regularization
- Max pooling for feature reduction

---

## Training Setup

| Component | Configuration |
|----------|--------------|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 30 |

### Learning Rate Scheduler
- ReduceLROnPlateau
- Reduces learning rate when validation loss plateaus

### Early Stopping
- Patience: 5 epochs
- Prevents overfitting and unnecessary training

---

## Evaluation

Model performance is evaluated using:
- Accuracy (training, validation, test)
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Learning curves (loss and accuracy)

---

## Results

The project compares:
- LeNet-5  
- Custom CNN (different scheduler factors)

Key observations:
- The Custom CNN outperforms LeNet-5 due to deeper feature extraction
- Learning rate scheduling improves convergence and stability
- Some visually similar letters are more difficult to distinguish

---

## How to Run

### Install dependencies
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn

---

Conclusion
----------

This project demonstrates a complete deep learning workflow for image classification. It highlights the importance of model architecture, training strategies, and systematic evaluation in achieving strong performance on real-world tasks such as sign language recognition.

Run model comparison
python sign_language_compare_models_scheduler_test.py

Run single model
python sign_language_customcnn_only.py
Outputs

Generated outputs include:

Training and validation curves
Confusion matrices
Accuracy comparison plots
Classification reports (TXT, CSV, JSON)
Learning Outcomes

## By completing this project, we were able to:

understand and apply convolutional neural networks for image classification
implement a full deep learning pipeline from preprocessing to evaluation
systematically compare model architectures and training strategies
analyze model performance using quantitative and visual methods
Contributors

This project was developed by:

Marcela Santos
Daria Fedorova
William Merino-Galindo

Conclusion

This project demonstrates a complete deep learning workflow for image classification. It highlights the importance of model architecture, training strategies, and systematic evaluation in achieving strong performance on real-world tasks such as sign language recognition.
