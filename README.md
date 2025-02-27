# Drug Discovery Using Molecular Fingerprints
Master thesis project: Using multi-label classification model to discover new drugs using molecular fingerprint 

## Overview
This project focuses on drug discovery by leveraging molecular fingerprints to predict chemical properties and reactions. Various machine learning and deep learning models are applied for multi-label classification, enabling the prediction of multiple molecular properties simultaneously. The project utilizes traditional machine learning approaches like Random Forest and SVM, along with deep learning methods for improved predictive performance.

## Repository Structure
This repository contains multiple Jupyter notebooks, each performing specific tasks in the drug discovery workflow. Below is a summary of each notebook:

### 1. **All_Metrics_Generator.ipynb**
- **Purpose:** Trains and evaluates multi-label classification models on different molecular fingerprint datasets.
- **Functionality:**
  - Loads multiple fingerprint datasets.
  - Splits data using iterative stratification to preserve label distribution.
  - Trains models using Binary Relevance (BR), Classifier Chains (CC), and Label Powerset (LP) approaches.
  - Evaluates models using various metrics (accuracy, Hamming loss, precision, recall, F1-score).
  - Consolidates results for comparison across fingerprint types and classification methods.

### 2. **Multi-label_Classification_Stratification.ipynb**
- **Purpose:** Demonstrates the importance of stratified train/test splitting for multi-label chemical datasets.
- **Functionality:**
  - Compares random vs. stratified train/test splits.
  - Evaluates label distribution differences quantitatively.
  - Trains sample multi-label classifiers to validate the impact of stratification.

### 3. **FP-online.ipynb**
- **Purpose:** Runs multi-label classification experiments and records results per fingerprint dataset.
- **Functionality:**
  - Iterates through multiple fingerprint datasets.
  - Trains models and evaluates performance for each fingerprint.
  - Saves results as separate CSV files for individual fingerprint analysis.

### 4. **Multi-label-Validation.ipynb**
- **Purpose:** Provides a streamlined validation pipeline for a specific fingerprint type.
- **Functionality:**
  - Loads a specified fingerprint dataset.
  - Splits data, trains models, and evaluates performance.
  - Saves results in a structured format for further analysis.

### 5. **ChemoTest.ipynb**
- **Purpose:** Tests deep learning on a single-label classification problem derived from the multi-label dataset.
- **Functionality:**
  - Filters dataset for a specific reaction label (e.g., "C-C Bond Formation (Acylation)").
  - Trains a small neural network on fingerprint data.
  - Evaluates model performance on a binary classification task.

### 6. **2020RecSys.ipynb**
- **Purpose:** Explores deep learning approaches for multi-label classification and reframes the problem as a recommendation system task.
- **Functionality:**
  - Prepares data for a deep learning model.
  - Implements a multi-label neural network classifier.
  - Examines feature-label interactions and collaborative filtering perspectives.

### 7. **SklearnMetrics.ipynb**
- **Purpose:** Provides clarification on scikit-learn classification metrics to ensure correct interpretation of results.
- **Functionality:**
  - Demonstrates how scikit-learn computes precision, recall, F1-score in different settings.
  - Explains micro, macro, weighted, and sample-averaged metrics for multi-label problems.

## How to Use
1. **Installation**: Ensure you have Python 3.x installed along with the necessary dependencies:
   ```sh
   pip install numpy pandas scikit-learn scikit-multilearn tensorflow
   ```
2. **Run Notebooks**: Open and execute Jupyter notebooks as needed:
   ```sh
   jupyter notebook
   ```
3. **Experiment with Fingerprint Data**: Modify datasets and models within the notebooks to explore different chemical representations and classification techniques.

## License
This project is open-source and available under the MIT License.

