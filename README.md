# Star-Galaxy-Quasar Classification

Supervised machine learning pipeline for multiclass astronomical classification using photometric and spectroscopic observations from the Sloan Digital Sky Survey (SDSS Data Release 17). The objective is to classify celestial objects into three classes: **Star**, **Galaxy**, and **Quasar (QSO)**.

---

## Overview

The Sloan Digital Sky Survey (SDSS DR17) provides deep optical imaging and spectroscopy of celestial bodies. This project processes 100,000 observations, cleans instrument error artifacts, validates class separations with One-Way ANOVA hypothesis testing, trains three baseline and ensemble models, and delivers an interactive Streamlit application.

### Key Metrics
- **Best Model:** Random Forest Classifier
- **Held-Out Test Accuracy:** 97.94%
- **Macro F1-Score:** 97.58%
- **Dataset Size:** 99,999 valid observations (after removing 1 sensor error artifact)

---

## Dataset and Features

### Target Classes
- **Galaxy:** 59,445 observations (59.45%)
- **Star:** 21,593 observations (21.59%)
- **Quasar (QSO):** 18,961 observations (18.96%)

### Selected Features
Only 6 physical attributes are used for model training. Survey identifiers and spatial coordinates are excluded to avoid data leakage and non-physical correlations.

| Feature | Description | Spectral Range / Formula |
| :--- | :--- | :--- |
| `u` | Ultraviolet photometric band | ~3543 Å |
| `g` | Green photometric band | ~4770 Å |
| `r` | Red photometric band | ~6231 Å |
| `i` | Near-infrared photometric band | ~7625 Å |
| `z` | Infrared photometric band | ~9134 Å |
| `redshift` | Spectroscopic redshift | $z = \Delta\lambda / \lambda_0$ |

### Data Cleaning
- **Sensor Error Record:** Row 79543 contained corrupted magnitude readings (`u = -9999.0`, `g = -9999.0`, `z = -9999.0`), the SDSS instrument error code. It was cleanly removed, leaving 99,999 valid samples.
- **Negative Redshifts:** 13,724 observations exhibit small negative redshifts (down to -0.00997), representing authentic Doppler blueshift (99.7% are Milky Way stars moving toward the Solar System). These were validated and retained.

---

## Hypothesis Testing (One-Way ANOVA)

A One-Way Analysis of Variance (ANOVA) was conducted across the three classes (`STAR`, `GALAXY`, `QSO`) to determine whether feature means differ significantly prior to model training:

- **Null Hypothesis ($H_0$):** Mean feature values are identical across classes ($\mu_{\text{Star}} = \mu_{\text{Galaxy}} = \mu_{\text{QSO}}$).
- **Alternative Hypothesis ($H_1$):** At least one class mean is statistically different.

| Feature | F-Statistic | p-Value | Outcome |
| :--- | :---: | :---: | :--- |
| `redshift` | 83,427.75 | < 1e-100 | Reject $H_0$ (Dominant discriminator) |
| `u` | 4,204.60 | < 1e-100 | Reject $H_0$ |
| `g` | 4,008.38 | < 1e-100 | Reject $H_0$ |
| `r` | 3,923.49 | < 1e-100 | Reject $H_0$ |
| `i` | 4,228.61 | < 1e-100 | Reject $H_0$ |
| `z` | 4,942.34 | < 1e-100 | Reject $H_0$ |

All features yield $p < 0.05$, confirming statistically significant separation across classes.

---

## Model Benchmark

Models were evaluated on a held-out test set (20,000 observations, 20% stratified split) after training on 64% and validating on 16%.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 95.57% | 95.22% | 94.75% | 94.97% | ~0.81 s |
| Decision Tree | 97.43% | 97.35% | 96.65% | 96.99% | ~0.74 s |
| **Random Forest** | **97.94%** | **98.02%** | **97.17%** | **97.58%** | **~3.94 s** |

### Per-Class Performance (Random Forest)
- **Star:** Precision: 99.63% | Recall: 99.95% | F1-Score: 99.79%
- **Galaxy:** Precision: 97.68% | Recall: 98.87% | F1-Score: 98.27%
- **Quasar (QSO):** Precision: 96.75% | Recall: 92.70% | F1-Score: 94.68%

### Feature Importance (Gini Impurity)
1. `redshift`: 67.38%
2. `z` band: 10.26%
3. `g` band: 7.07%
4. `u` band: 6.14%
5. `i` band: 5.95%
6. `r` band: 3.20%

---

## Project Structure

```
Star-Galaxy-Quasar-Classification/
├── notebook.ipynb                  # End-to-end data pipeline, ANOVA test, models, and evaluation
├── app.py                          # Interactive Streamlit classification web dashboard
├── best_star_classifier.pkl        # Trained Random Forest model artifact and metadata
├── sdss_stellar_classification.csv # SDSS DR17 dataset (100,000 observations)
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/oknidhiiiii/Star-Galaxy-Quasar-Classification.git
cd Star-Galaxy-Quasar-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```
