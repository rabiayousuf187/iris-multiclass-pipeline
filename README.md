# Iris Multi-Class Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, production-ready Machine Learning pipeline for multi-class classification on the Iris dataset. Built using object-oriented Python software architecture, packaged with `pyproject.toml` standards, and designed for full end-to-end experiment reproducibility.

---

## 📌 Project Overview

This repository transitions machine learning experimentation from monolithic Jupyter Notebooks into a maintainable, pip-installable Python package. 

### Key Features
* **Modular Software Design**: Separates data ingestion, model fitting, and evaluation into isolated modules under `src/iris_pipeline/`.
* **Reproducible Pipelines**: Stratified train-test splitting and deterministic seed control to prevent dataset drift and data leakage.
* **Pip-Installable Architecture**: Fully configured build system (`pyproject.toml`) allowing single-command package installations via Git.
* **Automated Visual Evaluation**: Automatically generates classification metrics and exports high-resolution confusion matrix heatmaps.

---

## 📐 Mathematical Formulation

The core model utilizes a **Random Forest Classifier**, an ensemble method combining $N$ decision trees trained via bootstrap aggregation (bagging).

### 1. Stratified Multi-Class Target
Given a feature vector $x_i \in \mathbb{R}^d$ (representing 4 sepal and petal measurements), the classifier maps inputs to a categorical target label $y_i \in \{0, 1, 2\}$ representing flower species:
* $0$: *Iris setosa*
* $1$: *Iris versicolor*
* $2$: *Iris virginica*

### 2. Gini Impurity Decision Trees
Each node splitting decision minimizes the Gini Impurity $H(Q_m)$ across regions:

$$H(Q_m) = \sum_{k=0}^{K-1} p_{mk} (1 - p_{mk})$$

where $p_{mk}$ represents the proportion of class $k$ training observations in node $m$.

### 3. Metric Evaluation
Model efficacy is evaluated using class-wise Precision, Recall, and Macro $F_1$-score:

$$\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}$$

$$F_{1, \text{macro}} = \frac{1}{K} \sum_{k=0}^{K-1} 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

---

## 📁 Repository Architecture

```text
iris-multiclass-pipeline/
├── src/
│   └── iris_pipeline/          # Core package root
│       ├── __init__.py         # Package versioning metadata
│       ├── data.py             # Ingestion & stratified processing
│       ├── model.py            # Random Forest estimator logic
│       └── evaluate.py         # Evaluation metrics & visualization
├── main.py                     # Primary execution pipeline script
├── pyproject.toml              # Build setup & pip metadata
├── requirements.txt            # Dependency specification
├── confusion_matrix.png        # Saved metric artifact
├── .gitignore
└── README.md