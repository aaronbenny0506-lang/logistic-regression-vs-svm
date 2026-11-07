# Logistic Regression vs SVM – Crop / Fertilizer Classification

Task 6: apply **Logistic Regression** and **Support Vector Machine (SVM)** to a multi-class dataset, tune the regularisation strength **C**, compare them with **4 multi-class metrics**, visualise the results and analyse the features.

## Dataset
`data/fertilizer.csv` - 500 rows, 5 balanced classes (100 each).

| Column | Meaning |
|---|---|
| ID | Row counter (dropped) |
| Carbon, Moisture, Phosphorous, Potassium | Soil properties |
| Rainfall, Temperature | Weather conditions |
| **Crop** | Target class (0-4) |

## Project structure
```
data/        fertilizer.csv (raw) · cleaned_dataset.csv (output of preprocessing)
plots/       01-04 EDA plots · 05–09 model / feature plots
results/     model_comparison.csv · logistic_regression_predictions.csv · svm_predictions.csv
             c_tuning_cv_scores.csv · feature_importance.csv
preprocessing.py   cleaning script -> data/cleaned_dataset.csv
eda_plots.py       4 exploratory plots
model_comparison.py  tuning, training, evaluation, visualisation
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python preprocessing.py      
python eda_plots.py          
python model_comparison.py   
```

## 1. Preprocessing (`preprocessing.py`)
* No missing values and no duplicate rows were found.
* Dropped `ID` (just a row index, no predictive value).
* **Found physically impossible negative readings** : Carbon (16), Phosphorous (12), Potassium (9), Rainfall (8). These were treated as invalid and replaced with the column median.
* Feature scaling (`StandardScaler`) is applied **inside the model pipelines**, so the scaler is fitted only on training folds (no data leakage).

## 2. Exploratory plots
| Plot | What it shows |
|---|---|
| `01_histograms.png` | Distribution of every feature (Phosphorous, Potassium and Rainfall are right-skewed) |
| `02_class_distribution.png` | Target is perfectly balanced (100 per crop) |
| `03_boxplots_by_crop.png` | **Temperature and Rainfall separate the crops clearly**; Carbon and Moisture look the same for every crop |
| `04_scatter_and_correlation.png` | Rainfall vs Temperature scatter forms distinct clusters; features are almost uncorrelated |

## 3. Models and regularisation (C)
* 80/20 stratified train/test split (400 train / 100 test), `random_state=42`.
* `05_effect_of_C.png` : 5-fold CV accuracy for C = 0.001 … 1000. Both models under-fit with small C (strong regularisation) and peak around **C = 10**; the SVM drops again at very large C (over-fitting).
* `GridSearchCV` (5-fold, stratified):
  * Logistic Regression: C ∈ {0.001 … 1000}
  * SVM: C ∈ {0.001 … 1000} × kernel ∈ {linear, rbf, poly} × gamma ∈ {scale, 0.01, 0.1, 1}
* Best settings: **LR C = 10**, **SVM linear kernel, C = 10**.

## 4. Results (test set, 100 samples)
Metrics: accuracy + macro-averaged precision, recall and F1.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression (default) | 0.830 | 0.837 | 0.830 | 0.830 |
| SVM (default, RBF) | 0.820 | 0.815 | 0.820 | 0.816 |
| **Logistic Regression (tuned)** | 0.830 | 0.849 | 0.830 | 0.831 |
| **SVM (tuned)** | **0.870** | **0.877** | **0.870** | **0.869** |

Both tuned models are well above the 0.7 target. Visuals: `06_metrics_comparison.png` (4 metrics side by side), `07_confusion_matrices.png`, `08_per_class_f1.png`.

## 5. Feature analysis (`09_feature_importance.png`)
* Logistic-regression coefficients and permutation importance agree: **Temperature** and **Rainfall** drive almost all of the predictions (≈ 0.43 and 0.33–0.39 accuracy drop when shuffled).
* Phosphorous and Potassium contribute a little; **Carbon and Moisture are essentially noise** (≈ 0 importance).

## Conclusions
* Tuning C matters: very small C costs 10–15 points of accuracy for both models.
* The tuned SVM scored higher on the test set (0.87 vs 0.83), but the two models are **essentially tied in 5-fold cross-validation (0.878 vs 0.880)**, and a 100-row test set is small, so the gap should not be over-interpreted. Logistic Regression is simpler and as good in CV; the SVM has a slight edge on the held-out split.
* Most confusion is between crops 1/2 (similar temperature, both low rainfall) and crop 4 vs 0 (both warm).
* Note: the C-effect plot uses an RBF SVM; the grid search later found a linear kernel to be best.
