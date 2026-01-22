# Co-Pathology Analysis Pipeline

A Python pipeline for discovering and classifying brain atrophy patterns using Latent Dirichlet Allocation (LDA) topic modeling and XGBoost classification.

## Overview

This pipeline analyzes regional brain volumetric data to:

1. **Normalize** patient volumes against healthy controls (Z-scoring)
2. **Discover** latent atrophy patterns ("topics") using LDA
3. **Classify** diagnoses based on topic mixtures using XGBoost
4. **Visualize** results with heatmaps, spider charts, and more

## Installation

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

## Quick Start

### Single Cohort Analysis

```python
from pipeline import CopathologyPipeline

import pandas as pd

# Load data
df = pd.read_csv("patient_data.csv")
region_cols = list(df.loc[:, "first_region":"last_region"].columns)

# Separate healthy controls
hc_df = df[df["DX"] == "HC"]
patient_df = df[df["DX"] != "HC"]

# Create and fit pipeline
pipeline = CopathologyPipeline(n_topics=6, output_dir="results")
results = pipeline.fit(patient_df, hc_df, region_cols, dx_col="DX")

# Save results and generate visualizations
pipeline.save_results(results["theta"], results["labels"], results["subject_ids"])
pipeline.generate_visualizations(results["theta"], results["labels"])

# Save model for later use
pipeline.save("models/pipeline.pkl")
```

### Multiple Cohorts with Different Healthy Controls

When you have datasets with different healthy control groups (e.g., one uses "HC", another uses "CN"):

```python
import pandas as pd
import numpy as np
from lda_model import LDATopicModel
from classifier import TopicClassifier

# Load datasets
df1 = pd.read_csv("dataset1.csv")  # HC as healthy control
df2 = pd.read_csv("dataset2.csv")  # CN as healthy control

# Define region columns
region_cols = list(df1.loc[:, "first_region":"last_region"].columns)

# Separate patients and healthy controls
hc1 = df1[df1["DX"] == "HC"]
patients1 = df1[df1["DX"] != "HC"]

hc2 = df2[df2["DX"] == "CN"]
patients2 = df2[df2["DX"] != "CN"]

# Z-score function
def zscore_against_hc(patient_df, hc_df, region_cols):
    hc_mean = hc_df[region_cols].mean()
    hc_std = hc_df[region_cols].std()
    z_scores = (patient_df[region_cols] - hc_mean) / hc_std
    return z_scores.clip(lower=0)  # Keep only atrophy (positive z)

# Compute z-scores for each cohort against its own HC
z1 = zscore_against_hc(patients1, hc1, region_cols)
z2 = zscore_against_hc(patients2, hc2, region_cols)

# Combine z-scores and labels
combined_z = pd.concat([z1, z2], ignore_index=True)
combined_labels = pd.concat([patients1["DX"], patients2["DX"]], ignore_index=True).values
combined_ids = pd.concat([patients1["PTID"], patients2["PTID"]], ignore_index=True).values

# Fit LDA on combined z-scores
lda = LDATopicModel(n_topics=6)
theta = lda.fit_transform(combined_z.values)

# Fit classifier and run cross-validation
classifier = TopicClassifier(n_splits=5)
classifier.fit(theta, combined_labels)
cv_results = classifier.cross_validate(theta, combined_labels, combined_ids, verbose=True)

print(f"CV Accuracy: {cv_results['accuracy']:.4f}")
```

### Inference on New Data

```python
# Load new patients
new_df = pd.read_csv("new_patients.csv")

# Z-score against the appropriate HC baseline
new_z = zscore_against_hc(new_df, hc1, region_cols)  # Use matching HC group

# Get topic weights and predictions
theta_new = lda.transform(new_z.values)
predictions = classifier.predict(theta_new)
probabilities = classifier.predict_proba(theta_new)

# Build results DataFrame
results = pd.DataFrame(theta_new, columns=[f"Topic_{k}" for k in range(6)])
results["Predicted_DX"] = predictions
for i, dx in enumerate(classifier.classes):
    results[f"P({dx})"] = probabilities[:, i]

print(results)
```

## Data Requirements

Your CSV files should contain:

| Column | Description |
|--------|-------------|
| `DX` | Diagnosis label (e.g., "AD", "MCI", "HC", "CN") |
| `PTID` | Subject identifier |
| Region columns | Volumetric data for each brain region (consecutive columns) |

## Pipeline Components

### DataProcessor

Handles Z-score normalization against healthy controls.

```python
from data_processor import DataProcessor

processor = DataProcessor()
processor.fit_baseline(hc_df, region_cols)
atrophy_scores = processor.compute_atrophy_scores(patient_df)
```

### LDATopicModel

Discovers latent atrophy patterns using Latent Dirichlet Allocation.

```python
from lda_model import LDATopicModel

lda = LDATopicModel(n_topics=6, alpha=1.0, beta=0.1)
theta = lda.fit_transform(atrophy_scores)  # Subject-topic weights
topic_patterns = lda.get_topic_patterns()  # Topic-region weights
```

**Parameters:**
- `n_topics`: Number of latent topics to discover
- `alpha`: Document-topic prior (higher = more uniform topic distributions)
- `beta`: Topic-word prior (lower = sparser topic-region associations)

### TopicClassifier

XGBoost classifier for diagnosis prediction from topic weights.

```python
from classifier import TopicClassifier

classifier = TopicClassifier(n_splits=5)
classifier.fit(theta, labels)
cv_results = classifier.cross_validate(theta, labels, subject_ids)

# Inference
predictions = classifier.predict(theta_new)
probabilities = classifier.predict_proba(theta_new)
```

### CopathologyVisualizer

Generates publication-ready visualizations.

```python
from visualizer import CopathologyVisualizer

viz = CopathologyVisualizer(output_dir="results/figures")

# Topic-region heatmap
viz.plot_topic_heatmap(topic_patterns, region_cols)

# Diagnosis profiles (spider charts)
viz.plot_diagnosis_topic_profiles(theta, labels)

# Subject topic mixtures (stacked bars)
viz.plot_copathology_stacked_bars(theta, labels, predictions=predictions)

# Confusion matrix
viz.plot_confusion_matrix(cm, class_names, accuracy=0.85)

# Top regions per topic
viz.plot_top_regions_per_topic(topic_patterns, region_cols)
```

## Output Files

The pipeline generates:

| File | Description |
|------|-------------|
| `lda_topic_atrophy_patterns.csv` | Topic-region weight matrix |
| `lda_subject_topic_weights.csv` | Subject-topic weight matrix |
| `lda_diagnosis_topic_expression.csv` | Mean topic weights per diagnosis |
| `lda_cv_predictions.csv` | Cross-validation predictions |
| `figures/topic_heatmap.png` | Topic-region association heatmap |
| `figures/diagnosis_topic_profiles.png` | Spider charts per diagnosis |
| `figures/copathology_stacked_bars.png` | Subject topic mixtures |
| `figures/confusion_matrix.png` | Classification confusion matrix |
| `figures/top_regions_per_topic.png` | Top regions for each topic |

## Saving and Loading Models

```python
# Save fitted pipeline
pipeline.save("models/copathology_pipeline.pkl")

# Load for inference
from pipeline import CopathologyPipeline
pipeline = CopathologyPipeline.load("models/copathology_pipeline.pkl")

# Predict on new data
results = pipeline.predict_new_subjects(new_df)
```

## API Reference

### CopathologyPipeline

| Method | Description |
|--------|-------------|
| `fit(patient_df, hc_df, region_cols)` | Fit the complete pipeline |
| `predict_new_subjects(new_df)` | Predict diagnosis for new subjects |
| `save_results(theta, labels, subject_ids)` | Save analysis results to CSV |
| `generate_visualizations(theta, labels)` | Generate all plots |
| `save(path)` / `load(path)` | Persist/restore fitted pipeline |

### LDATopicModel

| Method | Description |
|--------|-------------|
| `fit(X)` / `fit_transform(X)` | Fit LDA model |
| `transform(X_new)` | Infer topic weights for new subjects |
| `get_topic_patterns()` | Get topic-region weight matrix |
| `get_top_regions_per_topic(region_names)` | Get top regions for each topic |

### TopicClassifier

| Method | Description |
|--------|-------------|
| `fit(X, y)` | Fit final model on all data |
| `cross_validate(X, y, subject_ids)` | Run stratified K-fold CV |
| `predict(X)` / `predict_proba(X)` | Predict diagnoses |
| `get_confusion_matrix()` | Get CV confusion matrix |
| `get_feature_importance()` | Get topic importance scores |

## License

MIT
