# Fake News Detection using Random Forest

## Overview

This repository contains an implementation of a **Random Forest Classifier** for detecting fake news across three different datasets: **Liar**, **Fake or Real**, and **Combined Corpus**. This implementation extends methodologies presented in the paper *"A benchmark study of machine learning models for online fake news detection"* by Junaed Younus Khan et al. (2021).

---

## Repository Structure
```
├── datasets
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── code
│   ├── random_forest_liar.py
│   ├── random_forest_fake_or_real_and_combined_corpus.py
│   report.ipynb
├── requirements.txt
└── README.md
```

---

## Installation
Clone the repository and install dependencies using:

```bash
git clone (https://github.com/thomasbritnell/mlpfinal.git)
cd Fake-News-Detection-Random-Forest
pip install -r requirements.txt
```

---

## Dependencies
- pandas
- numpy
- scikit-learn
- imblearn (SMOTE)
- scipy

To install dependencies directly, run:

```bash
pip install pandas numpy scikit-learn imblearn scipy
```

---

## Running the Code
Execute the notebook:

```bash
jupyter notebook notebooks/report.ipynb
```

The notebook includes detailed steps for:
- Data Loading
- Feature Engineering
- SMOTE Oversampling
- Model Training with Hyperparameter Tuning
- Performance Evaluation
- Feature Importance Analysis

---

## Results Summary

| Dataset           | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | ROC AUC (%) |
|-------------------|--------------|---------------|------------|--------------|-------------|
| Liar              | 61.90        | 62.14         | 75.90      | 68.34        | 64.59       |
| Fake or Real      | 75.24        | 72.90         | 79.45      | 76.04        | 82.77       |
| Combined Corpus   | 77.65        | 84.21         | 82.64      | 83.42        | 82.82       |

---

## Feature Importance
The most impactful features across datasets include:
- Article length
- Word count
- Number count
- Adjective density

---

## Model Artifacts
Trained models are stored in the `root` directory and can be loaded for predictions using Python’s pickle module.

Example:
```python
import pickle

with open('models/random_forest_liar.pkl', 'rb') as file:
    model_info = pickle.load(file)

model = model_info['model']
feature_columns = model_info['feature_columns']
```

---

## Contributing
Feel free to fork, enhance, or optimize the code. Contributions are welcome through pull requests.

---

## References
- [Original Research Paper](https://github.com/JunaedYounusKhan51/FakeNewsDetection)
- Khan, J. Y., et al. (2021). *A benchmark study of machine learning models for online fake news detection.* Machine Learning with Applications.

