"""
LDA Topic Model for atrophy pattern discovery.

Wraps sklearn's LatentDirichletAllocation with methods for
topic extraction, inference on new subjects, and persistence.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from sklearn.decomposition import LatentDirichletAllocation


class LDATopicModel:
    """
    Latent Dirichlet Allocation for discovering atrophy patterns.

    Discovers latent "topics" (atrophy patterns) from regional brain data.
    Each topic represents a spatial pattern of atrophy, and each subject
    is characterized by a mixture of topics.

    Parameters
    ----------
    n_topics : int
        Number of latent topics to discover.
    alpha : float
        Document-topic prior (Dirichlet concentration).
        Higher values = more uniform topic distributions per subject.
    beta : float
        Topic-word prior (Dirichlet concentration).
        Lower values = sparser topic-region associations.
    max_iter : int
        Maximum iterations for batch learning.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_topics: int = 6,
        alpha: float = 1.0,
        beta: float = 0.1,
        max_iter: int = 500,
        random_state: int = 42
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state

        self._lda = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            learning_method="batch",
            max_iter=max_iter,
            random_state=random_state
        )

        # Fitted parameters
        self._theta: Optional[np.ndarray] = None  # Subject-topic weights
        self._beta_raw: Optional[np.ndarray] = None  # Raw topic-region weights
        self._beta_norm: Optional[np.ndarray] = None  # Normalized topic-region weights
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> "LDATopicModel":
        """
        Fit LDA model to atrophy data.

        Parameters
        ----------
        X : np.ndarray
            Atrophy data, shape (n_subjects, n_regions).
            Values should be non-negative.

        Returns
        -------
        self
        """
        self._theta = self._lda.fit_transform(X)
        self._beta_raw = self._lda.components_
        self._beta_norm = self._beta_raw / self._beta_raw.sum(axis=1, keepdims=True)
        self._is_fitted = True

        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit LDA model and return subject-topic weights.

        Parameters
        ----------
        X : np.ndarray
            Atrophy data, shape (n_subjects, n_regions).

        Returns
        -------
        np.ndarray
            Subject-topic weights (theta), shape (n_subjects, n_topics).
        """
        self.fit(X)
        return self._theta

    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Infer topic weights for new subjects.

        Parameters
        ----------
        X_new : np.ndarray
            New atrophy data, shape (n_new_subjects, n_regions).

        Returns
        -------
        np.ndarray
            Topic weights for new subjects, shape (n_new_subjects, n_topics).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform()")

        return self._lda.transform(X_new)

    def get_topic_patterns(self, normalized: bool = True):
        """
        Get topic-region weight matrix (beta).

        Parameters
        ----------
        normalized : bool
            If True, return row-normalized weights (sum to 1 per topic).

        Returns
        -------
        np.ndarray
            Topic patterns, shape (n_topics, n_regions).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self._beta_norm if normalized else self._beta_raw

    def get_topic_weights(self) -> np.ndarray:
        """
        Get subject-topic weight matrix (theta) from training.

        Returns
        -------
        np.ndarray
            Subject-topic weights, shape (n_subjects, n_topics).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self._theta

    def get_top_regions_per_topic(
        self,
        region_names: List[str],
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Get top regions for each topic.

        Parameters
        ----------
        region_names : list
            Names of regions (must match n_regions).
        top_n : int
            Number of top regions to return per topic.

        Returns
        -------
        dict
            Dictionary mapping topic names to DataFrames of top regions.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        result = {}
        for k in range(self.n_topics):
            topic_weights = self._beta_norm[k]
            indices = np.argsort(topic_weights)[::-1][:top_n]

            result[f"Topic_{k}"] = pd.DataFrame({
                "region": [region_names[i] for i in indices],
                "weight": topic_weights[indices]
            })

        return result

    def get_topic_dataframe(
        self,
        region_names: List[str]
    ) -> pd.DataFrame:
        """
        Get topic patterns as a DataFrame.

        Parameters
        ----------
        region_names : list
            Names of regions.

        Returns
        -------
        pd.DataFrame
            DataFrame with regions as rows and topics as columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        return pd.DataFrame(
            self._beta_norm.T,
            index=region_names,
            columns=[f"Topic_{k}" for k in range(self.n_topics)]
        )

    def get_subject_topic_dataframe(
        self,
        labels: Optional[np.ndarray] = None,
        subject_ids: Optional[np.ndarray] = None,
        label_col: str = "DX"
    ) -> pd.DataFrame:
        """
        Get subject-topic weights as a DataFrame.

        Parameters
        ----------
        labels : np.ndarray, optional
            Diagnosis labels for each subject.
        subject_ids : np.ndarray, optional
            Subject IDs.
        label_col : str
            Column name for labels.

        Returns
        -------
        pd.DataFrame
            DataFrame with subjects as rows and topics as columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        df = pd.DataFrame(
            self._theta,
            columns=[f"Topic_{k}" for k in range(self.n_topics)]
        )

        if subject_ids is not None:
            df.insert(0, "SUBJ_ID", subject_ids)

        if labels is not None:
            df[label_col] = labels

        return df

    def get_group_topic_means(
        self,
        labels: np.ndarray,
        theta: Optional[np.ndarray] = None
    ):
        """
        Compute mean topic weights per diagnosis group.

        Parameters
        ----------
        labels : np.ndarray
            Diagnosis labels.
        theta : np.ndarray, optional
            Subject-topic weights. Uses training theta if None.

        Returns
        -------
        pd.DataFrame
            Mean topic weights per group.
        """
        if theta is None:
            if not self._is_fitted:
                raise RuntimeError("Model must be fitted first")
            theta = self._theta

        df = pd.DataFrame(
            theta,
            columns=[f"Topic_{k}" for k in range(self.n_topics)]
        )
        df["DX"] = labels

        return df.groupby("DX").mean()

    def save(self, path: str) -> None:
        """
        Save fitted model to file.

        Parameters
        ----------
        path : str
            Output file path (.pkl).
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted model")

        state = {
            "n_topics": self.n_topics,
            "alpha": self.alpha,
            "beta": self.beta,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "lda": self._lda,
            "theta": self._theta,
            "beta_raw": self._beta_raw,
            "beta_norm": self._beta_norm
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "LDATopicModel":
        """
        Load model from file.

        Parameters
        ----------
        path : str
            Input file path (.pkl).

        Returns
        -------
        LDATopicModel
            Loaded model instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(
            n_topics=state["n_topics"],
            alpha=state["alpha"],
            beta=state["beta"],
            max_iter=state["max_iter"],
            random_state=state["random_state"]
        )

        model._lda = state["lda"]
        model._theta = state["theta"]
        model._beta_raw = state["beta_raw"]
        model._beta_norm = state["beta_norm"]
        model._is_fitted = True

        return model

    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted

    @property
    def n_regions(self) -> int:
        """Number of regions (features)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")
        return self._beta_norm.shape[1]

    @property
    def topic_names(self) -> List[str]:
        """List of topic names."""
        return [f"Topic_{k}" for k in range(self.n_topics)]
