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
df_all = pd.read_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/data/260108_wsev_final_df.csv")
hc_df = df_all[df_all['DX'] == 'HC']
df = df_all[df_all['DX'] != 'HC']

# -------------------------------
# 1. Select regional data
# -------------------------------

region_cols = df_all.loc[:, 'ctx_lh_caudalanteriorcingulate':'ctx_rh_insula'].columns

X_hc = hc_df[region_cols].values.astype(float)
X_pat = df[region_cols].values.astype(float)

print(f"HC: {X_hc.shape[0]} subjects")
print(f"Patients: {X_pat.shape[0]} subjects")
y = df['DX'].values
# -------------------------------
# 2. Compute HC reference stats
# -------------------------------

hc_mean = X_hc.mean(axis=0, keepdims=True)
hc_std  = X_hc.std(axis=0, keepdims=True) + 1e-8  # avoid divide-by-zero

# -------------------------------
# 3. Z-score patients relative to HC
# -------------------------------

Z = (X_pat - hc_mean) / hc_std

# -------------------------------
# 4. Convert to atrophy scores
#    (higher = more atrophy)
# -------------------------------

X_atrophy = np.maximum(-Z, 0.0)

X = X_atrophy

X[X < 0] = 0.0

# -------------------------------
# 3. Fit vanilla LDA
# -------------------------------

n_topics = 6  # start here

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

topic_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/lda_with_reg/lda_topic_atrophy_patterns.csv")

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

coef_df.to_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/lda_with_reg/topic_diagnosis_coefficients.csv")

print("Saved topic–diagnosis coefficients.")
print("/nTopic coefficients by diagnosis:")
print(coef_df)

print("/nDone.")

################## inference ##
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load external patients CSV
# -------------------------------
df_ext = pd.read_csv("C:/Users/BREIN/Desktop/copathology_visualization_temp/data/inference.csv")

# Extract regional columns (make sure they match training data)
region_cols = df_ext.loc[:, 'ctx_lh_caudalanteriorcingulate':'ctx_rh_insula'].columns
X_ext_raw = df_ext[region_cols].values.astype(float)
subject_ids = df_ext['PTID'].values  # replace with your actual column
dx_ext = df_ext['DX'].values              # true diagnosis

# -------------------------------
# 2. Z-score relative to original HC
# hc_mean, hc_std: from your training cohort
# -------------------------------
Z_ext = (X_ext_raw - hc_mean) / hc_std
X_atrophy_ext = np.maximum(-Z_ext, 0.0)
X_atrophy_ext[X_atrophy_ext < 0] = 0.0

# -------------------------------
# 3. Project onto LDA topics
# -------------------------------
theta_ext = lda.transform(X_atrophy_ext)  # shape: (n_subjects x n_topics)
n_topics = theta_ext.shape[1]

# -------------------------------
# 4. Create output folder for plots
# -------------------------------
output_dir = "C:/Users/BREIN/Desktop/copathology_visualization_temp/external_topic_ranking"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 5. Topic label mapping
# -------------------------------
label_map = {'Topic_0': 'TO', 'Topic_1': 'MT', 'Topic_2': 'TP', 
             'Topic_3': 'C', 'Topic_4': 'FL', 'Topic_5': 'F'}

# -------------------------------
# 6. Initialize dataframe to store topic ranks
# -------------------------------
topic_rank_list = []

# -------------------------------
# 7. Loop over each subject
# -------------------------------
for i, subj_id in enumerate(subject_ids):
    theta_patient = theta_ext[i]
    
    # Rank topics
    rank_idx = np.argsort(theta_patient)[::-1]
    ranked_topics = [f"Topic_{k}" for k in rank_idx]
    ranked_labels = [label_map[t] for t in ranked_topics]
    ranked_weights = theta_patient[rank_idx]
    
    # Save to dataframe
    for rank, (topic, weight) in enumerate(zip(ranked_topics, ranked_weights), start=1):
        topic_rank_list.append({
            "SubjectID": subj_id,
            "DX": dx_ext[i],
            "Rank": rank,
            "Topic": topic,
            "Weight": weight
        })
    
    # -------------------------------
    # 8. Plot ranked topics for this subject
    # -------------------------------
    plt.figure(figsize=(6,4))
    plt.barh(range(len(ranked_weights)), ranked_weights[::-1], color='skyblue')
    plt.yticks(range(len(ranked_weights)), ranked_labels[::-1])
    plt.xlabel("Topic Weight")
    plt.title(f"{subj_id} — {dx_ext[i]}")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{subj_id}_topic_ranking.png"))
    plt.close()

# -------------------------------
# 9. Save topic ranks CSV
# -------------------------------
rank_df = pd.DataFrame(topic_rank_list)
rank_df.to_csv(os.path.join(output_dir, "external_subject_topic_ranks.csv"), index=False)

print(f"Saved topic ranking plots and CSV for {len(subject_ids)} subjects in '{output_dir}'.")

################

# Topic weights from LDA
# theta_ext: shape (n_subjects x n_topics)
# Topic labels mapped to your short codes
label_map = {'Topic_0': 'TO', 'Topic_1': 'MT', 'Topic_2': 'TP',
             'Topic_3': 'C', 'Topic_4': 'FL', 'Topic_5': 'F'}
topic_labels = [label_map[f"Topic_{i}"] for i in range(theta_ext.shape[1])]

# -------------------------------
# 2. Create DataFrame with subject IDs and DX
# -------------------------------
theta_df = pd.DataFrame(theta_ext, columns=topic_labels)
theta_df['SubjectID'] = subject_ids
theta_df['DX'] = dx_ext

# -------------------------------
# 3. Sort subjects by DX
# -------------------------------
theta_df_sorted = theta_df.sort_values(by='DX')
theta_sorted_values = theta_df_sorted[topic_labels].values
subject_labels_sorted = [f"{sid} ({dx})" for sid, dx in zip(theta_df_sorted['SubjectID'], theta_df_sorted['DX'])]

# -------------------------------
# 4. Plot heatmap (red color scheme)
# -------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(theta_sorted_values, cmap='Reds', cbar_kws={'label': 'Topic weight'}, xticklabels=topic_labels, yticklabels=subject_labels_sorted)
plt.title("Topic Weight Heatmap (sorted by DX)")
plt.xlabel("Topics")
plt.ylabel("Subjects")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"inf_heatmap.png"))


# -------------------------------
# 5. Plot stacked bar chart
# -------------------------------
import matplotlib.cm as cm
# Prepare sorted theta for external subjects
theta_df_sorted = theta_df.sort_values(by='DX')
theta_sorted_values = theta_df_sorted[topic_labels].values
subject_labels_sorted = [f"{sid} ({dx})" for sid, dx in zip(theta_df_sorted['SubjectID'], theta_df_sorted['DX'])]

# Use tab20 colormap for topics
cmap = cm.get_cmap('tab20', len(topic_labels))  # discrete colors

plt.figure(figsize=(14,6))

bottom = np.zeros(theta_sorted_values.shape[0])
for i, topic in enumerate(topic_labels):
    plt.bar(range(theta_sorted_values.shape[0]), theta_sorted_values[:,i],
            bottom=bottom, color=cmap(i), label=topic)
    bottom += theta_sorted_values[:,i]

plt.xticks(range(theta_sorted_values.shape[0]), subject_labels_sorted, rotation=90)
plt.ylabel("Topic weight")
plt.title("Stacked Topic Contributions")
plt.xticks(rotation=90)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"stacked_barchart.png"))


################################26011433333333333333333333333333333333333333
# ============================================================
# Inference function: project new subjects onto LDA + DX model
# ============================================================

def infer_subjects_lda_dx(
    df_new,
    lda,
    clf,
    region_cols,
    hc_mean,
    hc_std,
    subject_id_col="PTID",
    true_dx_col="DX",
    plot=True,
    output_dir=None
):
    """
    Infer topic proportions and diagnostic probabilities
    for new subjects using trained LDA + multinomial model.

    Returns:
        results_df : DataFrame with topic weights, predicted DX,
                     and diagnostic probabilities
    """

    import os
    import matplotlib.pyplot as plt

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------
    # 1. Extract and preprocess data
    # -------------------------------
    X_raw = df_new[region_cols].values.astype(float)

    Z = (X_raw - hc_mean) / hc_std
    X_atrophy = np.maximum(-Z, 0.0)
    X_atrophy[X_atrophy < 0] = 0.0

    # -------------------------------
    # 2. Infer topic proportions
    # -------------------------------
    theta_new = lda.transform(X_atrophy)
    n_topics = theta_new.shape[1]

    topic_cols = [f"Topic_{k}" for k in range(n_topics)]

    theta_df = pd.DataFrame(theta_new, columns=topic_cols)
    theta_df["SubjectID"] = df_new[subject_id_col].values

    if true_dx_col in df_new.columns:
        theta_df["True_DX"] = df_new[true_dx_col].values

    # -------------------------------
    # 3. Predict diagnostic probabilities
    # -------------------------------
    dx_probs = clf.predict_proba(theta_new)
    dx_labels = clf.named_steps["logreg"].classes_

    dx_prob_df = pd.DataFrame(
        dx_probs,
        columns=[f"P({dx})" for dx in dx_labels]
    )

    pred_dx = clf.predict(theta_new)

    # -------------------------------
    # 4. Combine results
    # -------------------------------
    results_df = pd.concat(
        [
            theta_df.reset_index(drop=True),
            dx_prob_df.reset_index(drop=True)
        ],
        axis=1
    )

    results_df["Predicted_DX"] = pred_dx

    # -------------------------------
    # 5. Optional plotting (per subject)
    # -------------------------------
    if plot:
        for i in range(theta_new.shape[0]):
            sid = results_df.loc[i, "SubjectID"]

            # Topic composition plot
            plt.figure(figsize=(6, 3))
            plt.bar(topic_cols, theta_new[i] * 100)
            plt.ylabel("Topic proportion (%)")
            plt.title(f"{sid} — Topic composition")
            plt.xticks(rotation=45)
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{sid}_topics.png"))
                plt.close()
            else:
                plt.show()

            # Diagnostic probability plot
            plt.figure(figsize=(5, 3))
            plt.bar(dx_labels, dx_probs[i] * 100)
            plt.ylabel("Probability (%)")
            plt.title(f"{sid} — Diagnostic mixture")
            plt.ylim(0, 100)
            plt.tight_layout()

            if output_dir:
                plt.savefig(os.path.join(output_dir, f"{sid}_dx_probs.png"))
                plt.close()
            else:
                plt.show()

    return results_df

df_ext = pd.read_csv(
    "C:/Users/BREIN/Desktop/copathology_visualization_temp/data/inference.csv"
)

results_df = infer_subjects_lda_dx(
    df_new=df_ext,
    lda=lda,
    clf=clf,
    region_cols=region_cols,
    hc_mean=hc_mean,
    hc_std=hc_std,
    subject_id_col="PTID",
    true_dx_col="DX",
    plot=True,
    output_dir="C:/Users/BREIN/Desktop/copathology_visualization_temp/wsev_results/external_inference_results"
)

results_df.to_csv(
    "C:/Users/BREIN/Desktop/copathology_visualization_temp/wsev_results/external_inference_results/inference_summary.csv",
    index=False
)

print("Inference complete.")
print(results_df.head())
