"""
End-to-end pipeline for co-pathology analysis.

Orchestrates data processing, LDA topic modeling, classification,
and provides a unified interface for training and inference.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any, Union

from .data_processor import DataProcessor
from .lda_model import LDATopicModel
from .classifier import TopicClassifier
from .visualizer import CopathologyVisualizer


class CopathologyPipeline:
    """
    End-to-end pipeline for co-pathology analysis.

    Orchestrates:
    1. Data preprocessing (Z-scoring against healthy controls)
    2. LDA topic discovery
    3. XGBoost classification
    4. Visualization

    Parameters
    ----------
    n_topics : int
        Number of LDA topics.
    alpha : float
        LDA document-topic prior.
    beta : float
        LDA topic-word prior.
    n_cv_splits : int
        Number of cross-validation folds.
    output_dir : str
        Directory for saving outputs.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_topics: int = 6,
        alpha: float = 1.0,
        beta: float = 0.1,
        n_cv_splits: int = 5,
        output_dir: str = "results",
        random_state: int = 42
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_cv_splits = n_cv_splits
        self.output_dir = output_dir
        self.random_state = random_state

        # Initialize components
        self.data_processor = DataProcessor()
        self.lda_model = LDATopicModel(
            n_topics=n_topics,
            alpha=alpha,
            beta=beta,
            random_state=random_state
        )
        self.classifier = TopicClassifier(
            n_splits=n_cv_splits,
            random_state=random_state
        )
        self.visualizer = CopathologyVisualizer(
            output_dir=os.path.join(output_dir, "figures")
        )

        # Training metadata
        self._region_cols: Optional[List[str]] = None
        self._dx_classes: Optional[np.ndarray] = None
        self._is_fitted = False

        os.makedirs(output_dir, exist_ok=True)

    def fit(
        self,
        train_df: pd.DataFrame,
        region_cols : List[str],
        standardize = True,
        ref_df: pd.DataFrame = None,
        dx_col: str = "DX",
        subject_col: str = "PTID",
        run_cv: bool = True,
        verbose: bool = True
    ):
        """
        Fit the complete pipeline on training data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Patient data with regional volumes and diagnosis.
        ref_df : pd.DataFrame
            Healthy control data for baseline.
        region_cols : list
            Column names for brain regions.
        standardize : bool
            Flag to whether apply z score computation on input data
        dx_col : str
            Column name for diagnosis labels.
        subject_col : str
            Column name for subject IDs.
        run_cv : bool
            Run cross-validation for classifier evaluation.
        verbose : bool
            Print progress information.

        Returns
        -------
        dict
            Dictionary containing:
            - 'theta': Subject-topic weights
            - 'topic_patterns': Topic-region patterns
            - 'cv_results': Cross-validation results (if run_cv=True)
        """
        self._region_cols = list(region_cols)
        self.data_processor.region_cols = self._region_cols
        self.data_processor.dx_col = dx_col
        self.data_processor.subject_col = subject_col
        if standardize:
            assert ref_df is not None, "Please Provide Reference DF For Z-score Computation..."
            if verbose:
                print("Step 1: Fitting healthy control baseline...")

            # Fit baseline on healthy controls
            self.data_processor.fit_baseline(ref_df, region_cols)

            if verbose:
                print("Step 2: Computing atrophy scores...")

        # Compute atrophy scores for patients
            X = self.data_processor.compute_atrophy_scores(train_df)
        else:
            print("Scores Already Stadardized...Skipping Step 1 and 2...")
            X = train_df[region_cols].values
            
        y = self.data_processor.get_labels(train_df)
        subject_ids = self.data_processor.get_subject_ids(train_df)

        if verbose:
            print(f"   Patients: {X.shape[0]}, Regions: {X.shape[1]}")

        if verbose:
            print(f"Step 3: Fitting LDA with {self.n_topics} topics...")

        # Fit LDA
        theta = self.lda_model.fit_transform(X)
        topic_patterns = self.lda_model.get_topic_patterns()

        if verbose:
            print("Step 4: Fitting classifier...")

        # Fit classifier
        self.classifier.fit(theta, y)
        self._dx_classes = self.classifier.classes
        self._is_fitted = True

        results = {
            "theta": theta,
            "topic_patterns": topic_patterns,
            "labels": y,
            "subject_ids": subject_ids
        }

        # Run cross-validation if requested
        if run_cv:
            if verbose:
                print("Step 5: Running cross-validation...")
            cv_results = self.classifier.cross_validate(
                theta, y, subject_ids, verbose=verbose
            )
            results["cv_results"] = cv_results
            if verbose:
                print(f"\nCV Accuracy: {cv_results['accuracy']:.4f}")

        if verbose:
            print("\nPipeline fitting complete!")

        return results

    def predict_new_subjects(
        self,
        inp_df : pd.DataFrame,
        subject_col: Optional[str] = None,
        standardize = False,
        dx_col = 'DX',
        cn_dx = None
    ):
        """
        Predict diagnosis for new subjects.

        Parameters
        ----------
        X_new : 
        subject_col : str, optional
            Column name for subject IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - Subject ID (if provided)
            - Topic_0 through Topic_N weights
            - Predicted_DX
            - P(DX) for each diagnosis class
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        if inp_df[self._region_cols].isna().any().any():
            raise ValueError("Validation VA data contains nan")

        # Compute atrophy scores
        if standardize:
            X_new = self.data_processor.compute_atrophy_scores(inp_df)
        elif cn_dx is not None:
            print('Calculating Z scores from external cohort reference...')
            ext_data_processor = DataProcessor(region_cols=self._region_cols, dx_col=dx_col, subject_col=subject_col)
            ext_cn_df = inp_df[inp_df[dx_col]==cn_dx]
            ext_data_processor.fit_baseline(ext_cn_df)
            X_new = ext_data_processor.compute_atrophy_scores(data=inp_df)
        else:
            X_new = inp_df[self._region_cols].values
            
        # Get topic weights                
        theta_new = self.lda_model.transform(X_new)

        # Get predictions
        y_pred = self.classifier.predict(theta_new)
        y_proba = self.classifier.predict_proba(theta_new)

        # Build results DataFrame
        results = pd.DataFrame(
            theta_new,
            columns=[f"Topic_{k}" for k in range(self.n_topics)]
        )

        # Add subject IDs if available
        subj_col = subject_col or self.data_processor.subject_col
        if subj_col in inp_df.columns:
            results.insert(0, "SUBJ_ID", inp_df[subj_col].values)

        # Add predictions
        results["pred_DX"] = y_pred

        # Add probabilities
        for i, dx in enumerate(self._dx_classes):
            results[f"P({dx})"] = y_proba[:, i]

        return results

    def save_results(
        self,
        theta: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        prefix: str = "lda"
    ):
        """
        Save analysis results to CSV files.

        Parameters
        ----------
        theta : np.ndarray
            Subject-topic weights.
        labels : np.ndarray
            Diagnosis labels.
        subject_ids : np.ndarray, optional
            Subject identifiers.
        prefix : str
            Filename prefix.

        Returns
        -------
        dict
            Paths to saved files.
        """
        paths = {}

        # Topic patterns
        topic_df = self.lda_model.get_topic_dataframe(self._region_cols)
        topic_path = os.path.join(self.output_dir, f"{prefix}_topic_atrophy_patterns.csv")
        topic_df.to_csv(topic_path)
        paths["topic_patterns"] = topic_path

        # Subject topic weights
        theta_df = self.lda_model.get_subject_topic_dataframe(labels, subject_ids)
        theta_path = os.path.join(self.output_dir, f"{prefix}_subject_topic_weights.csv")
        theta_df.to_csv(theta_path, index=False)
        paths["subject_weights"] = theta_path

        # Diagnosis topic expression
        group_means = self.lda_model.get_group_topic_means(labels)
        group_path = os.path.join(self.output_dir, f"{prefix}_diagnosis_topic_expression.csv")
        group_means.to_csv(group_path)
        paths["group_means"] = group_path

        # CV results if available
        if self.classifier._cv_results_df is not None:
            cv_path = os.path.join(self.output_dir, f"{prefix}_cv_predictions.csv")
            self.classifier.get_cv_results().to_csv(cv_path, index=False)
            paths["cv_results"] = cv_path

        print(f"Results saved to {self.output_dir}/")
        return paths

    def generate_internal_visualizations(
        self,
        theta: np.ndarray,
        labels: np.ndarray,
        topic_label_map: Optional[Dict[str, str]] = None,
        region_names = None,
        label_order = None
    ):
        """
        Generate standard visualization plots.

        Parameters
        ----------
        theta : np.ndarray
            Subject-topic weights.
        labels : np.ndarray
            Diagnosis labels.
        topic_label_map : dict, optional
            Mapping from topic names to display labels.
        """
        topic_patterns = self.lda_model.get_topic_patterns()
        topic_names = self.lda_model.topic_names
        if region_names is None:
            region_names = self._region_cols

        print("Generating visualizations...")

        # Topic heatmap
        self.visualizer.plot_topic_heatmap(
            topic_patterns, region_names, topic_names
        )

        # Diagnosis profiles (spider charts)
        self.visualizer.plot_diagnosis_topic_profiles(
            theta, labels, topic_names, topic_label_map
        )

        # Top regions per topic
        self.visualizer.plot_top_regions_per_topic(
            topic_patterns, region_names, topic_names=topic_names
        )

        # If CV was run, plot results
        if self.classifier._cv_results_df is not None:
            cv_results = self.classifier.get_cv_results()
            predictions = cv_results["DX_pred"].values

            # Reorder to match theta
            # Note: CV results may be in different order
            self.visualizer.plot_copathology_stacked_bars(
                theta, labels, topic_names, topic_label_map, predictions
            )

            self.visualizer.plot_confusion_matrix(
                self.classifier.get_confusion_matrix(),
                list(self.classifier.classes),
                accuracy=self.classifier.cv_accuracy,
                label_order=label_order
            )

            self.visualizer.plot_prediction_probabilities(cv_results)
            self.visualizer.plot_probability_heatmap(cv_results)

        print(f"Visualizations saved to {self.visualizer.output_dir}/")

    def save(self, path: str):
        """
        Save the fitted pipeline to a file.

        Parameters
        ----------
        path : str
            Output file path (.pkl).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        state = {
            "n_topics": self.n_topics,
            "alpha": self.alpha,
            "beta": self.beta,
            "n_cv_splits": self.n_cv_splits,
            "output_dir": self.output_dir,
            "random_state": self.random_state,
            "region_cols": self._region_cols,
            "dx_classes": self._dx_classes,
            "data_processor": {
                "region_cols": self.data_processor.region_cols,
                "dx_col": self.data_processor.dx_col,
                "subject_col": self.data_processor.subject_col,
                "hc_mean": self.data_processor._hc_mean,
                "hc_std": self.data_processor._hc_std
            },
            "lda_state": {
                "lda": self.lda_model._lda,
                "theta": self.lda_model._theta,
                "beta_raw": self.lda_model._beta_raw,
                "beta_norm": self.lda_model._beta_norm
            },
            "classifier_state": {
                "label_encoder": self.classifier._label_encoder,
                "final_model": self.classifier._final_model,
                "classes": self.classifier._classes
            }
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

        print(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str):
        """
        Load a fitted pipeline from file.

        Parameters
        ----------
        path : str
            Input file path (.pkl).

        Returns
        -------
        CopathologyPipeline
            Loaded pipeline instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        pipeline = cls(
            n_topics=state["n_topics"],
            alpha=state["alpha"],
            beta=state["beta"],
            n_cv_splits=state["n_cv_splits"],
            output_dir=state["output_dir"],
            random_state=state["random_state"]
        )

        # Restore data processor
        dp_state = state["data_processor"]
        pipeline.data_processor.region_cols = dp_state["region_cols"]
        pipeline.data_processor.dx_col = dp_state["dx_col"]
        pipeline.data_processor.subject_col = dp_state["subject_col"]
        pipeline.data_processor._hc_mean = dp_state["hc_mean"]
        pipeline.data_processor._hc_std = dp_state["hc_std"]
        pipeline.data_processor._is_fitted = True

        # Restore LDA model
        lda_state = state["lda_state"]
        pipeline.lda_model._lda = lda_state["lda"]
        pipeline.lda_model._theta = lda_state["theta"]
        pipeline.lda_model._beta_raw = lda_state["beta_raw"]
        pipeline.lda_model._beta_norm = lda_state["beta_norm"]
        pipeline.lda_model._is_fitted = True

        # Restore classifier
        clf_state = state["classifier_state"]
        pipeline.classifier._label_encoder = clf_state["label_encoder"]
        pipeline.classifier._final_model = clf_state["final_model"]
        pipeline.classifier._classes = clf_state["classes"]
        pipeline.classifier._is_fitted = True

        # Restore pipeline state
        pipeline._region_cols = state["region_cols"]
        pipeline._dx_classes = state["dx_classes"]
        pipeline._is_fitted = True

        print(f"Pipeline loaded from {path}")
        return pipeline

    @property
    def is_fitted(self):
        """Check if pipeline has been fitted."""
        return self._is_fitted

    @property
    def topic_names(self):
        """Get topic names."""
        return self.lda_model.topic_names

    @property
    def diagnosis_classes(self):
        """Get diagnosis class labels."""
        if self._dx_classes is None:
            raise RuntimeError("Pipeline must be fitted first")
        return self._dx_classes


def run_analysis(
    patient_csv: str,
    hc_csv: str,
    region_start: str,
    region_end: str,
    hc_label: str = "HC",
    dx_col: str = "DX",
    n_topics: int = 6,
    output_dir: str = "results",
    save_model: bool = True,
    model_path: str = "models/copathology_pipeline.pkl"
):
    """
    Convenience function to run the full analysis pipeline.

    Parameters
    ----------
    patient_csv : str
        Path to patient data CSV.
    hc_csv : str
        Path to healthy control data CSV (can be same as patient_csv).
    region_start : str
        First region column name.
    region_end : str
        Last region column name.
    hc_label : str
        Label for healthy controls in dx_col.
    dx_col : str
        Diagnosis column name.
    n_topics : int
        Number of LDA topics.
    output_dir : str
        Output directory.
    save_model : bool
        Save fitted model.
    model_path : str
        Path for saved model.

    Returns
    -------
    CopathologyPipeline
        Fitted pipeline.
    """
    # Load data
    print(f"Loading data from {patient_csv}...")
    patient_df = pd.read_csv(patient_csv)
    region_cols = list(patient_df.loc[:, region_start:region_end].columns)

    # Separate HC if in same file
    if hc_csv == patient_csv:
        hc_df = patient_df[patient_df[dx_col] == hc_label]
        patient_df = patient_df[patient_df[dx_col] != hc_label]
    else:
        hc_df = pd.read_csv(hc_csv)

    print(f"Patients: {len(patient_df)}, HC: {len(hc_df)}, Regions: {len(region_cols)}")

    # Create and fit pipeline
    pipeline = CopathologyPipeline(
        n_topics=n_topics,
        output_dir=output_dir
    )

    results = pipeline.fit(
        patient_df, hc_df, region_cols,
        dx_col=dx_col,
        verbose=True
    )

    # Save results
    pipeline.save_results(
        results["theta"],
        results["labels"],
        results["subject_ids"]
    )

    # Generate visualizations
    pipeline.generate_internal_visualizations(
        results["theta"],
        results["labels"]
    )

    # Save model
    if save_model:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pipeline.save(model_path)

    return pipeline
