"""Step 3 - Logistic Regression vs SVM: tune C, evaluate with 4 multi-class metrics, visualise."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sns.set_theme(style="whitegrid")
SEED = 42
df = pd.read_csv("data/cleaned_dataset.csv")
X, y = df.drop(columns="Crop"), df["Crop"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
C_RANGE = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# ---- 1. Effect of regularisation strength C (5-fold CV on the training set) ----
curves = {}
for name, make in {"Logistic Regression": lambda C: LogisticRegression(C=C, max_iter=5000),
                   "SVM (RBF)": lambda C: SVC(C=C, kernel="rbf", gamma="scale")}.items():
    curves[name] = [cross_val_score(make_pipeline(StandardScaler(), make(C)), X_tr, y_tr, cv=cv).mean()
                    for C in C_RANGE]
plt.figure(figsize=(7, 4.5))
for name, sc in curves.items():
    plt.semilogx(C_RANGE, sc, marker="o", label=name)
plt.xlabel("C (inverse regularisation strength)"); plt.ylabel("5-fold CV accuracy")
plt.title("Effect of C on model performance"); plt.legend(); plt.tight_layout()
plt.savefig("plots/05_effect_of_C.png", dpi=150); plt.close()
pd.DataFrame(curves, index=C_RANGE).rename_axis("C").to_csv("results/c_tuning_cv_scores.csv")

# ---- 2. Grid search for the best settings ----
lr_grid = GridSearchCV(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    {"logisticregression__C": C_RANGE}, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
svm_grid = GridSearchCV(
    make_pipeline(StandardScaler(), SVC(random_state=SEED)),
    {"svc__C": C_RANGE, "svc__kernel": ["linear", "rbf", "poly"],
     "svc__gamma": ["scale", 0.01, 0.1, 1]}, cv=cv, scoring="accuracy", n_jobs=-1).fit(X_tr, y_tr)
print("Best LR :", lr_grid.best_params_, f"CV={lr_grid.best_score_:.3f}")
print("Best SVM:", svm_grid.best_params_, f"CV={svm_grid.best_score_:.3f}")

# ---- 3. Untuned baselines for before/after comparison ----
base = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X_tr, y_tr),
        "SVM": make_pipeline(StandardScaler(), SVC(random_state=SEED)).fit(X_tr, y_tr)}
tuned = {"Logistic Regression": lr_grid.best_estimator_, "SVM": svm_grid.best_estimator_}


def metrics(model, name):
    p = model.predict(X_te)
    return {"Model": name,
            "Accuracy": accuracy_score(y_te, p),
            "Precision (macro)": precision_score(y_te, p, average="macro"),
            "Recall (macro)": recall_score(y_te, p, average="macro"),
            "F1 (macro)": f1_score(y_te, p, average="macro")}, p


rows, preds = [], {}
for n, m in base.items():
    r, _ = metrics(m, f"{n} (default)"); rows.append(r)
for n, m in tuned.items():
    r, preds[n] = metrics(m, f"{n} (tuned)"); rows.append(r)
res = pd.DataFrame(rows).round(4)
res.to_csv("results/model_comparison.csv", index=False)
print("\n", res.to_string(index=False))

pd.DataFrame({"actual": y_te.values, "predicted": preds["Logistic Regression"]}, index=y_te.index)\
    .to_csv("results/logistic_regression_predictions.csv")
pd.DataFrame({"actual": y_te.values, "predicted": preds["SVM"]}, index=y_te.index)\
    .to_csv("results/svm_predictions.csv")
for n, p in preds.items():
    print(f"\n--- {n} (tuned) classification report ---\n", classification_report(y_te, p, digits=3))

# ---- 4. Visualise performance ----
tun = res[res.Model.str.contains("tuned")].set_index("Model")
tun.index = [i.replace(" (tuned)", "") for i in tun.index]
ax = tun.T.plot(kind="bar", figsize=(9, 5), color=["#2a7f8e", "#e8b630"], edgecolor="black", rot=0)
ax.set_ylim(0.5, 1.0); ax.set_title("Tuned model comparison - 4 multi-class metrics (test set)")
ax.axhline(0.7, color="red", ls="--", lw=1, label="0.7 target"); ax.legend()
for c in ax.containers[:2]: ax.bar_label(c, fmt="%.3f", fontsize=9)
plt.tight_layout(); plt.savefig("plots/06_metrics_comparison.png", dpi=150); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (n, p) in zip(axes, preds.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_te, p), display_labels=sorted(y.unique())).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{n} (tuned)\naccuracy={accuracy_score(y_te, p):.3f}")
plt.tight_layout(); plt.savefig("plots/07_confusion_matrices.png", dpi=150); plt.close()

pc = pd.DataFrame({n: f1_score(y_te, p, average=None) for n, p in preds.items()}, index=sorted(y.unique()))
pc.plot(kind="bar", figsize=(8, 4.5), color=["#2a7f8e", "#e8b630"], edgecolor="black", rot=0)
plt.xlabel("Crop class"); plt.ylabel("F1 score"); plt.ylim(0, 1.05); plt.title("Per-class F1 (tuned models)")
plt.tight_layout(); plt.savefig("plots/08_per_class_f1.png", dpi=150); plt.close()

# ---- 5. Feature analysis ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
coef = pd.DataFrame(lr_grid.best_estimator_[-1].coef_, columns=X.columns)
sns.heatmap(coef, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0], yticklabels=sorted(y.unique()))
axes[0].set_ylabel("Crop class"); axes[0].set_title("Logistic Regression coefficients (scaled features)")
imp = {}
for n, m in tuned.items():
    pi = permutation_importance(m, X_te, y_te, n_repeats=30, random_state=SEED, scoring="accuracy")
    imp[n] = pi.importances_mean
imp = pd.DataFrame(imp, index=X.columns).sort_values("SVM")
imp.plot(kind="barh", ax=axes[1], color=["#2a7f8e", "#e8b630"], edgecolor="black")
axes[1].set_title("Permutation importance (drop in accuracy)")
plt.tight_layout(); plt.savefig("plots/09_feature_importance.png", dpi=150); plt.close()
imp.round(4).to_csv("results/feature_importance.csv")
print("\nFeature importance:\n", imp.sort_values("SVM", ascending=False).round(3))
