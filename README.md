# Star–Galaxy–Quasar Classification Using Machine Learning

An end-to-end, multiclass supervised machine learning project that classifies astronomical observations from the **Sloan Digital Sky Survey (SDSS DR17)** into three celestial categories: **Star**, **Galaxy**, and **Quasar (QSO)**.

---

## 📁 Project Structure & Deliverables

```
Star Galaxy-SI/
│
├── sdss_stellar_classification.csv         # Verified SDSS17 dataset (100,000 observations, 18 columns)
├── notebook.ipynb                          # Project notebook (ANOVA, models, evaluation)
├── best_star_classifier.pkl                # Trained model artifact (Random Forest + complete metadata)
├── app.py                                  # Standalone interactive Streamlit web dashboard
├── requirements.txt                        # Python dependencies
└── README.md                               # Usage documentation & viva guide
```

---

## 🔭 Dataset Overview

- **Source:** Kaggle — [Stellar Classification Dataset - SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) (published by Fedesoriano).
- **Origin:** Sloan Digital Sky Survey Data Release 17 (SDSS DR17), Apache Point Observatory.
- **Original Dimensions:** 100,000 rows × 18 columns.
- **Data Quality & Astronomy Insight:**
  - **Sensor Artifact Filtering:** Exactly one observation (`obj_ID = 1.237649e+18`, row 79543) contained corrupted magnitude values (`u = -9999.0`, `g = -9999.0`, `z = -9999.0`), which is the standard SDSS instrument error code. It was cleanly removed, leaving **99,999 valid observations**.
  - **Doppler Blueshift:** 13,724 observations exhibit small negative redshifts (down to $-0.00997$). Domain verification confirmed that 99.7% of these are Milky Way stars moving radially toward the Solar System (Doppler blueshift). They are physically authentic and preserved.
- **Class Breakdown:**
  - `GALAXY`: 59,445 observations (59.45%)
  - `STAR`: 21,593 observations (21.59%)
  - `QSO`: 18,961 observations (18.96%)

---

## 🧪 Machine Learning Pipeline

### 1. Selected Features (6 Physical Attributes)
- `u`: Ultraviolet photometric filter ($\lambda \approx 3543$ Å)
- `g`: Green optical filter ($\lambda \approx 4770$ Å)
- `r`: Red optical filter ($\lambda \approx 6231$ Å)
- `i`: Near-infrared filter ($\lambda \approx 7625$ Å)
- `z`: Infrared filter ($\lambda \approx 9134$ Å)
- `redshift`: Spectroscopic Doppler/cosmological redshift ($z = \Delta\lambda / \lambda_0$)

*Excluded Metadata:* Identifiers and survey metadata (`obj_ID`, `spec_obj_ID`, `run_ID`, `rerun_ID`, `cam_col`, `field_ID`, `plate`, `MJD`, `fiber_ID`, `alpha`, `delta`) are excluded to prevent spurious correlation and preserve scientific validity.

### 2. Data Partitioning (Zero Data Leakage)
- **Training Set (64%):** 63,999 observations (used to fit models and scalers).
- **Validation Set (16%):** 16,000 observations (used exclusively for model selection).
- **Held-Out Test Set (20%):** 20,000 observations (quarantined until final benchmark).

### 3. Evaluated Models
1. **Logistic Regression (Baseline):** Encapsulated in a Scikit-Learn `Pipeline` with `StandardScaler` (`max_iter=1000`).
2. **Decision Tree Classifier:** Non-linear decision rules regularized with `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=5`.
3. **Random Forest Classifier (Selected Best Model):** Bagging ensemble of 100 decision trees (`n_estimators=100`, `max_depth=15`, `min_samples_split=5`, `n_jobs=-1`).

---

## 📊 Benchmark Results (Held-Out Test Set — 20,000 Observations)

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 95.57% | 95.22% | 94.75% | 94.97% | ~0.81 s |
| **Decision Tree** | 97.43% | 97.35% | 96.65% | 96.99% | ~0.74 s |
| **Random Forest (Winner)** | **97.94%** | **98.02%** | **97.17%** | **97.58%** | **~3.94 s** |

### Per-Class Test Performance (Random Forest)
- **`STAR`:** Precision: 99.63% | Recall: 99.95% | **F1-Score: 99.79%** (Support: 4,319)
- **`GALAXY`:** Precision: 97.68% | Recall: 98.87% | **F1-Score: 98.27%** (Support: 11,889)
- **`QSO`:** Precision: 96.75% | Recall: 92.70% | **F1-Score: 94.68%** (Support: 3,792)

### Feature Importance (Gini Impurity)
1. `redshift`: **67.38%** (Primary discriminator distinguishing Milky Way stars from cosmological objects)
2. `z` band: **10.26%**
3. `g` band: **7.07%**
4. `u` band: **6.14%**
5. `i` band: **5.95%**
6. `r` band: **3.20%**

---

## 🚀 How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `notebook.ipynb`.
3. Upload `sdss_stellar_classification.csv` to the session storage using the folder icon on the left pane. *(If omitted, the notebook will automatically download the verified SDSS17 dataset).*
4. Click **Runtime** > **Run all**.
5. All cells will execute in sequence, training the models, plotting confusion matrices, generating `best_star_classifier.pkl`, creating `app.py`, and running verification tests.
6. In Section 22, uncomment `files.download('best_star_classifier.pkl')` and `files.download('app.py')` to download them to your computer.

---

## 💻 How to Launch the Streamlit Web Application Locally

1. Open your terminal / command prompt in this directory.
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. The dashboard will automatically open in your browser at `http://localhost:8501`.
5. Features of the App:
   - **Overview & Dataset:** Project summary, SDSS context, and class proportions.
   - **Model Comparison:** Interactive comparison table and performance bar charts.
   - **Evaluation & Confusion Matrices:** Interactive confusion matrix viewer and per-class reports.
   - **Feature Importance:** Random Forest Gini importance horizontal bar chart with scientific explanations.
   - **Interactive Classification:** Enter $u, g, r, i, z$ and $redshift$ values (or click the quick presets for typical Star, Galaxy, Quasar) to receive real-time predictions, class probabilities, and astronomical descriptions.

---

## 🎓 College Viva Preparation (Q&A)

- **Q: Why was Macro F1-score chosen to select the winning model rather than Accuracy?**  
  *A:* The dataset is imbalanced (59.4% Galaxy, 21.6% Star, 19.0% QSO). Accuracy favors the majority class. Macro F1 calculates the arithmetic mean of F1-scores across all three classes, ensuring the minority class (`QSO`) is evaluated with equal weight.

- **Q: Why scale features for Logistic Regression but not for Random Forest or Decision Trees?**  
  *A:* Logistic Regression calculates linear dot products $\mathbf{w}^T \mathbf{x} + b$; unscaled features with large ranges distort the loss surface and hinder gradient descent. Tree models split on monotonic thresholds ($x_i \ge \theta$), making them completely invariant to feature scaling.

- **Q: Why does `redshift` have negative values?**  
  *A:* Negative redshift represents Doppler blueshift. In our dataset, 99.7% of negative redshift objects are stars within the Milky Way moving toward the observer. It is genuine astronomical physics.

- **Q: What is the main scientific limitation of this model?**  
  *A:* Spectroscopic redshift accounts for 67.4% of the model's predictive power. In real-world sky surveys, obtaining spectroscopic redshift requires targeted spectrograph exposures, which are not available when an object is first detected. A purely photometric classification pipeline would need to use photometric color indices instead.
