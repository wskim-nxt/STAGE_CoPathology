"""
Classification module for co-pathology analysis.

XGBoost classifier for predicting diagnosis from LDA topic weights,
with stratified cross-validation and inference on new subjects.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight


class TopicClassifier:
    """
    XGBoost classifier for diagnosis prediction from topic weights.

    Supports:
    - Stratified K-fold cross-validation for evaluation
    - Training a final model on all data
    - Inference on new subjects

    Parameters
    ----------
    n_splits : int
        Number of folds for cross-validation.
    xgb_params : dict, optional
        XGBoost parameters. Defaults to multiclass softmax.
    random_state : int
        Random seed for reproducibility.
    class_weight : str or None
        If 'balanced', automatically adjusts weights inversely proportional
        to class frequencies. If None, all classes have equal weight.
    """

    def __init__(
        self,
        n_splits: int = 5,
        xgb_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        class_weight: Optional[str] = None
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.class_weight = class_weight

        self.xgb_params = xgb_params or {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "random_state": random_state
        }

        self._label_encoder = LabelEncoder()
        self._final_model: Optional[XGBClassifier] = None

        # Cross-validation results
        self._cv_results_df: Optional[pd.DataFrame] = None
        self._cv_confusion_matrix: Optional[np.ndarray] = None
        self._cv_accuracy: Optional[float] = None
        self._classes: Optional[np.ndarray] = None
        self._is_fitted = False

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        verbose: bool = True
    ):
        """
        Run stratified K-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (topic weights), shape (n_subjects, n_topics).
        y : np.ndarray
            Diagnosis labels.
        subject_ids : np.ndarray, optional
            Subject identifiers for tracking.
        verbose : bool
            Print progress information.

        Returns
        -------
        dict
            Cross-validation results including accuracy, confusion matrix,
            and per-subject predictions.
        """
        if subject_ids is None:
            subject_ids = np.arange(len(y))

        # Encode labels
        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = self._label_encoder.classes_

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        all_idx = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
            if verbose:
                print(f"\n===== Fold {fold} =====")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            model = XGBClassifier(
                num_class=len(self._classes),
                **self.xgb_params
            )

            # Compute sample weights if class_weight is specified
            sample_weights = None
            if self.class_weight == "balanced":
                sample_weights = compute_sample_weight("balanced", y_train)

            model.fit(X_train, y_train, sample_weight=sample_weights)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            all_y_proba.append(y_proba)
            all_idx.append(subject_ids[test_idx])

        # Aggregate results
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)
        y_proba_all = np.vstack(all_y_proba)
        idx_all = np.concatenate(all_idx)

        # Compute metrics
        self._cv_confusion_matrix = confusion_matrix(y_true_all, y_pred_all)
        self._cv_accuracy = accuracy_score(y_true_all, y_pred_all)

        # Build results DataFrame
        results_df = pd.DataFrame(
            y_proba_all,
            columns=[f"P({dx})" for dx in self._classes]
        )
        results_df["DX_true"] = self._label_encoder.inverse_transform(y_true_all)
        results_df["DX_pred"] = self._label_encoder.inverse_transform(y_pred_all)
        results_df["subject_id"] = idx_all

        # Reorder to match original input order
        results_df = results_df.set_index("subject_id").loc[subject_ids].reset_index()

        self._cv_results_df = results_df

        return {
            "accuracy": self._cv_accuracy,
            "confusion_matrix": self._cv_confusion_matrix,
            "results_df": self._cv_results_df,
            "classes": self._classes
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit final model on all data for inference.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (topic weights).
        y : np.ndarray
            Diagnosis labels.

        Returns
        -------
        self
        """
        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = self._label_encoder.classes_

        self._final_model = XGBClassifier(
            num_class=len(self._classes),
            **self.xgb_params
        )

        # Compute sample weights if class_weight is specified
        sample_weights = None
        if self.class_weight == "balanced":
            sample_weights = compute_sample_weight("balanced", y_encoded)

        self._final_model.fit(X, y_encoded, sample_weight=sample_weights)
        self._is_fitted = True

        return self

    def predict(self, X: np.ndarray):
        """
        Predict diagnosis for new subjects.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (topic weights).

        Returns
        -------
        np.ndarray
            Predicted diagnosis labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict()")

        y_pred_encoded = self._final_model.predict(X)
        return self._label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: np.ndarray):
        """
        Predict diagnosis probabilities for new subjects.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (topic weights).

        Returns
        -------
        np.ndarray
            Probability matrix, shape (n_subjects, n_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba()")

        return self._final_model.predict_proba(X)

    def predict_with_proba(
        self,
        X: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ):
        """
        Predict diagnosis with probabilities as a DataFrame.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (topic weights).
        subject_ids : np.ndarray, optional
            Subject identifiers.

        Returns
        -------
        pd.DataFrame
            Predictions with probabilities for each class.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        df = pd.DataFrame(
            y_proba,
            columns=[f"P({dx})" for dx in self._classes]
        )
        df["DX_pred"] = y_pred

        if subject_ids is not None:
            df.insert(0, "subject_id", subject_ids)

        return df

    def get_classification_report(self, as_dict: bool = False):
        """
        Get classification report from cross-validation.

        Parameters
        ----------
        as_dict : bool
            Return as dictionary instead of printing.

        Returns
        -------
        str or dict
            Classification report.
        """
        if self._cv_results_df is None:
            raise RuntimeError("Must run cross_validate() first")

        y_true = self._label_encoder.transform(self._cv_results_df["DX_true"])
        y_pred = self._label_encoder.transform(self._cv_results_df["DX_pred"])

        if as_dict:
            return classification_report(
                y_true, y_pred,
                target_names=self._classes,
                output_dict=True
            )
        else:
            report = classification_report(
                y_true, y_pred,
                target_names=self._classes
            )
            print("\n===== CV Classification Report =====")
            print(report)
            print(f"CV Accuracy: {self._cv_accuracy:.4f}")
            return report

    def get_confusion_matrix(self):
        """
        Get confusion matrix from cross-validation.

        Returns
        -------
        np.ndarray
            Confusion matrix.
        """
        if self._cv_confusion_matrix is None:
            raise RuntimeError("Must run cross_validate() first")
        return self._cv_confusion_matrix

    def get_cv_results(self):
        """
        Get detailed cross-validation results.

        Returns
        -------
        pd.DataFrame
            Per-subject CV predictions and probabilities.
        """
        if self._cv_results_df is None:
            raise RuntimeError("Must run cross_validate() first")
        return self._cv_results_df.copy()

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ):
        """
        Get feature importance from the final model.

        Parameters
        ----------
        feature_names : list, optional
            Names of features (topics).

        Returns
        -------
        pd.DataFrame
            Feature importances sorted by importance.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        importance = self._final_model.feature_importances_

        if feature_names is None:
            feature_names = [f"Topic_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str):
        """
        Save fitted classifier to file.

        Parameters
        ----------
        path : str
            Output file path (.pkl).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        state = {
            "n_splits": self.n_splits,
            "xgb_params": self.xgb_params,
            "random_state": self.random_state,
            "class_weight": self.class_weight,
            "label_encoder": self._label_encoder,
            "final_model": self._final_model,
            "classes": self._classes,
            "cv_results_df": self._cv_results_df,
            "cv_confusion_matrix": self._cv_confusion_matrix,
            "cv_accuracy": self._cv_accuracy
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str):
        """
        Load classifier from file.

        Parameters
        ----------
        path : str
            Input file path (.pkl).

        Returns
        -------
        TopicClassifier
            Loaded classifier instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        classifier = cls(
            n_splits=state["n_splits"],
            xgb_params=state["xgb_params"],
            random_state=state["random_state"],
            class_weight=state.get("class_weight")
        )

        classifier._label_encoder = state["label_encoder"]
        classifier._final_model = state["final_model"]
        classifier._classes = state["classes"]
        classifier._cv_results_df = state["cv_results_df"]
        classifier._cv_confusion_matrix = state["cv_confusion_matrix"]
        classifier._cv_accuracy = state["cv_accuracy"]
        classifier._is_fitted = True

        return classifier

    @property
    def is_fitted(self):
        """Check if final model has been fitted."""
        return self._is_fitted

    @property
    def classes(self):
        """Get class labels."""
        if self._classes is None:
            raise RuntimeError("Model must be fitted first")
        return self._classes

    @property
    def cv_accuracy(self):
        """Get cross-validation accuracy."""
        if self._cv_accuracy is None:
            raise RuntimeError("Must run cross_validate() first")
        return self._cv_accuracy
