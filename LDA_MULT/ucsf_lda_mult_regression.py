# ============================================================
# Step 2: Vanilla LDA + Multinomial Regression on Topic Weights
# ============================================================

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and prepare data
# -------------------------------
df = pd.read_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/data/ucsf_regional_wscore.csv")

region_cols = df.loc[:, '1002':'2035'].columns
X = df[region_cols].values.astype(float)


print(f"Patients: {X.shape[0]} subjects")
y = df['DX'].values

X[X < 0] = 0.0

# -------------------------------
# 3. Fit vanilla LDA
# -------------------------------

n_topics = 10  # start here

lda = LatentDirichletAllocation(
    n_components=n_topics,
    doc_topic_prior=1.0,
    topic_word_prior=0.1,
    learning_method='batch',
    max_iter=500,
    random_state=42
)

theta = lda.fit_transform(X)   # subject × topic
beta = lda.components_                 # topic × region

print("LDA fitted.")

# -------------------------------
# 4. Save topic maps (optional)
# -------------------------------

beta_norm = beta / beta.sum(axis=1, keepdims=True)

topic_df = pd.DataFrame(
    beta_norm.T,
    index=region_cols,
    columns=[f"Topic_{k}" for k in range(n_topics)]
)

topic_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/lda_with_reg/ucsf_results/lda_topic_atrophy_patterns.csv")

# -------------------------------
# 5. Multinomial regression on topic weights
# -------------------------------

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(
    clf,
    theta,
    y,
    cv=cv
)

print("\nCross-validated classification report:")
print(classification_report(y, y_pred))

# -------------------------------
# 6. Fit final model for interpretation
# -------------------------------

clf.fit(theta, y)
coef = clf.named_steps["logreg"].coef_

coef_df = pd.DataFrame(
    coef,
    columns=[f"Topic_{k}" for k in range(n_topics)],
    index=clf.named_steps["logreg"].classes_
)

coef_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/lda_with_reg/ucsf_results/topic_diagnosis_coefficients.csv")

print("Saved topic–diagnosis coefficients.")
print("/nTopic coefficients by diagnosis:")
print(coef_df)

print("/nDone.")
