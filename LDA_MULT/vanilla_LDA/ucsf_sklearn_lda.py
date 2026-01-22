# ============================================================
# Vanilla LDA for Regional Brain Atrophy
# Unsupervised discovery of latent atrophy patterns
# ============================================================

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# -------------------------------
# 1. Load data
# -------------------------------

# Expected CSV structure:
# Columns:
#   - 'RID' (optional subject ID)
#   - 'DX'  (diagnosis; NOT used for fitting)
#   - regional atrophy columns (non-negative, higher = more atrophy)

df = pd.read_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/data/ucsf_regional_wscore.csv")

region_cols = df.loc[:, '1002':'2035'].columns
X = df[region_cols].values.astype(float)

# -------------------------------
# 2. Preprocess for LDA
# -------------------------------

# Ensure non-negativity (required for LDA)
X[X < 0] = 0.0
# X = X / (X.sum(axis=1, keepdims=True) + 1e-8)

# Normalize per subject so topics reflect spatial patterns
# X = X / (X.sum(axis=1, keepdims=True) + 1e-8)

# -------------------------------
# 3. Fit vanilla LDA
# -------------------------------

n_topics = 6  # try 3–6 in sensitivity analyses

lda = LatentDirichletAllocation(
    n_components=n_topics,
    doc_topic_prior=1.0,      # alpha
    topic_word_prior=0.1,     # beta
    learning_method='batch',
    max_iter=500,
    random_state=42
)

theta = lda.fit_transform(X)     # subject × topic
beta = lda.components_           # topic × region

print("LDA fitting complete")

# -------------------------------
# 4. Normalize topic maps
# -------------------------------

beta_norm = beta / beta.sum(axis=1, keepdims=True)

topic_df = pd.DataFrame(
    beta_norm.T,
    index=region_cols,
    columns=[f"Topic_{k}" for k in range(n_topics)]
)

# -------------------------------
# 5. Save topic (atrophy pattern) maps
# -------------------------------

topic_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/vanilla_LDA/ucsf_results/lda_topic_atrophy_patterns.csv")
print("Saved topic atrophy patterns")

# -------------------------------
# 6. Subject-level topic mixtures
# -------------------------------

theta_df = pd.DataFrame(
    theta,
    columns=[f"Topic_{k}" for k in range(n_topics)]
)

theta_df["DX"] = df["DX"].values
theta_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/vanilla_LDA/ucsf_results/lda_subject_topic_weights.csv", index=False)

print("Saved subject-level topic mixtures")

# -------------------------------
# 7. Diagnosis-wise topic expression (post hoc)
# -------------------------------

group_means = theta_df.groupby("DX").mean()
group_means.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/vanilla_LDA/ucsf_results/lda_diagnosis_topic_expression.csv")

print("Saved diagnosis-wise topic expression")

# -------------------------------
# 8. Reconstruct diagnosis-specific atrophy maps
# -------------------------------

dx_maps = {}

for dx in group_means.index:
    weights = group_means.loc[dx].values
    dx_map = np.dot(weights, beta_norm)
    dx_maps[dx] = dx_map

dx_maps_df = pd.DataFrame(dx_maps, index=region_cols)
dx_maps_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/vanilla_LDA/ucsf_results/lda_diagnosis_atrophy_maps.csv")

print("Saved diagnosis-specific atrophy maps")

# -------------------------------
# 9. Quick sanity checks
# -------------------------------

print("\nTop regions per topic:")
for k in range(n_topics):
    top_regions = (
        topic_df[f"Topic_{k}"]
        .sort_values(ascending=False)
        .head(8)
    )
    print(f"\nTopic {k}:")
    print(top_regions)

print("\nMean topic expression per diagnosis:")
print(group_means)

print("\nDone.")
