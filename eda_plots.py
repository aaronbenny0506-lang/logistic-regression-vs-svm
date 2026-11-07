"""Step 2 - Four exploratory plots saved to plots/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/cleaned_dataset.csv")
FEATURES = [c for c in df.columns if c != "Crop"]

# 1. Histograms of every feature
df[FEATURES].hist(bins=25, figsize=(12, 7), color="#2a7f8e", edgecolor="white")
plt.suptitle("1. Feature distributions (histograms)", fontsize=14)
plt.tight_layout(); plt.savefig("plots/01_histograms.png", dpi=150); plt.close()

# 2. Class balance
plt.figure(figsize=(6, 4))
sns.countplot(x="Crop", data=df, palette="viridis", hue="Crop", legend=False)
plt.title("2. Crop class distribution (target is balanced)")
plt.tight_layout(); plt.savefig("plots/02_class_distribution.png", dpi=150); plt.close()

# 3. Box plots of each feature by crop
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), FEATURES):
    sns.boxplot(x="Crop", y=col, data=df, ax=ax, palette="viridis", hue="Crop", legend=False)
    ax.set_title(col)
plt.suptitle("3. Features by crop (box plots)", fontsize=14)
plt.tight_layout(); plt.savefig("plots/03_boxplots_by_crop.png", dpi=150); plt.close()

# 4. Scatter plot: the two most separating features + correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.scatterplot(data=df, x="Rainfall", y="Temperature", hue="Crop", palette="viridis", ax=axes[0], s=35)
axes[0].set_title("4a. Rainfall vs Temperature coloured by crop")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
axes[1].set_title("4b. Correlation heatmap")
plt.tight_layout(); plt.savefig("plots/04_scatter_and_correlation.png", dpi=150); plt.close()
print("Saved 4 EDA plots to plots/")
