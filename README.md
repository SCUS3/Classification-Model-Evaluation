# Classification Model Evaluation

This repository contains Python code for evaluating classification models using various algorithms such as Support Vector Machine (SVM), k-Nearest Neighbors (kNN), Multi-Layer Perceptron (MLP), Linear Discriminant Analysis (LDA), and Naive Bayes (NB). The evaluation is based on precision, recall, and F1 score metrics using cross-validation and testing on unseen data.

## Project Overview

The project involves the following key steps:

1. **Loading Data**: Loading the training and testing data from CSV files.
2. **Model Evaluation and Tuning**:
    - Tuning the C parameter for SVM.
    - Tuning the k parameter for kNN.
    - Evaluating different configurations for MLP.
3. **Model Comparison**: Comparing the performance of various classifiers.
4. **Metrics Calculation**: Computing precision, recall, and F1 score for each classifier.

## Features

- **Model Tuning**: Optimize hyperparameters for SVM, kNN, and MLP.
- **Cross-Validation**: Employ k-fold cross-validation for model evaluation.
- **Model Comparison**: Compare the performance of multiple classifiers.
- **Metrics Calculation**: Compute precision, recall, and F1 score for each classifier.

## Technologies Used

- **Python**
- **Pandas**
- **Matplotlib**
- **scikit-learn**

## Getting Started

### Prerequisites

Ensure you have Python and the required libraries installed:

```bash
pip install pandas matplotlib scikit-learn
