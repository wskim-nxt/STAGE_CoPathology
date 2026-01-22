"""
Data processing module for co-pathology analysis.

Handles data loading, Z-score normalization against healthy controls,
and atrophy score computation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union


class DataProcessor:
    """
    Handles data loading, preprocessing, and normalization for atrophy analysis.

    The preprocessing pipeline:
    1. Load raw volumetric data (CSV)
    2. Extract region columns (cortical/subcortical)
    3. Compute Z-scores against healthy control baseline
    4. Convert to atrophy scores: atrophy = max(-Z, 0)

    Parameters
    ----------
    region_cols : list, optional
        List of region column names. If None, auto-detected from data.
    dx_col : str
        Column name containing diagnosis labels.
    subject_col : str
        Column name containing subject IDs.
    """

    def __init__(
        self,
        region_cols: Optional[List[str]] = None,
        dx_col: str = "DX",
        subject_col: str = "PTID"
    ):
        self.region_cols = region_cols
        self.dx_col = dx_col
        self.subject_col = subject_col

        # Healthy control baseline statistics
        self._hc_mean: Optional[np.ndarray] = None
        self._hc_std: Optional[np.ndarray] = None
        self._is_fitted = False

    def load_csv(
        self,
        path: str,
        region_start: Optional[str] = None,
        region_end: Optional[str] = None
    ):
        """
        Load data from CSV file.

        Parameters
        ----------
        path : str
            Path to CSV file.
        region_start : str, optional
            First region column name (for auto-detection).
        region_end : str, optional
            Last region column name (for auto-detection).

        Returns
        -------
        pd.DataFrame
            Loaded dataframe.
        """
        df = pd.read_csv(path)

        # Auto-detect region columns if not set
        if self.region_cols is None and region_start and region_end:
            self.region_cols = list(df.loc[:, region_start:region_end].columns)

        return df

    def fit_baseline(
        self,
        hc_data: Union[pd.DataFrame, np.ndarray],
        region_cols: Optional[List[str]] = None
    ):
        """
        Fit healthy control baseline statistics for Z-score normalization.

        Parameters
        ----------
        hc_data : pd.DataFrame or np.ndarray
            Healthy control data. If DataFrame, region_cols are used to extract values.
        region_cols : list, optional
            Region columns to use (overrides instance region_cols for this call).

        Returns
        -------
        self
        """
        cols = region_cols or self.region_cols

        if isinstance(hc_data, pd.DataFrame):
            if cols is None:
                raise ValueError("region_cols must be specified for DataFrame input")
            X_hc = hc_data[cols].values.astype(float)
        else:
            X_hc = hc_data.astype(float)

        self._hc_mean = X_hc.mean(axis=0, keepdims=True)
        self._hc_std = X_hc.std(axis=0, keepdims=True) + 1e-8  # Avoid divide-by-zero
        self._is_fitted = True

        return self

    def compute_atrophy_scores(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        region_cols: Optional[List[str]] = None
    ):
        """
        Compute atrophy scores from raw volumetric data.

        Atrophy = max(-Z, 0) where Z = (X - HC_mean) / HC_std

        Parameters
        ----------
        data : pd.DataFrame or np.ndarray
            Patient volumetric data.
        region_cols : list, optional
            Region columns to use.

        Returns
        -------
        np.ndarray
            Non-negative atrophy scores, shape (n_subjects, n_regions).
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit_baseline() before compute_atrophy_scores()")

        cols = region_cols or self.region_cols

        if isinstance(data, pd.DataFrame):
            if cols is None:
                raise ValueError("region_cols must be specified for DataFrame input")
            X = data[cols].values.astype(float)
        else:
            X = data.astype(float)

        # Z-score normalization
        Z = (X - self._hc_mean) / self._hc_std

        # Convert to atrophy (positive = more atrophy)
        atrophy = np.maximum(-Z, 0.0)

        return atrophy

    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        region_cols: Optional[List[str]] = None
    ):
        """
        Extract feature matrix from DataFrame (without Z-scoring).

        Use this for data that is already preprocessed.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        region_cols : list, optional
            Region columns to extract.

        Returns
        -------
        np.ndarray
            Feature matrix, shape (n_subjects, n_regions).
        """
        cols = region_cols or self.region_cols
        if cols is None:
            raise ValueError("region_cols must be specified")

        X = df[cols].values.astype(float)
        X[X < 0] = 0.0  # Ensure non-negative
        return X

    def get_labels(self, df: pd.DataFrame):
        """
        Extract diagnosis labels from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        np.ndarray
            Diagnosis labels.
        """
        return df[self.dx_col].values

    def get_subject_ids(self, df: pd.DataFrame):
        """
        Extract subject IDs from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.

        Returns
        -------
        np.ndarray
            Subject IDs.
        """
        if self.subject_col in df.columns:
            return df[self.subject_col].values
        return np.arange(len(df))

    def get_region_names(self):
        """
        Get the list of region column names.

        Returns
        -------
        list
            Region column names.
        """
        if self.region_cols is None:
            raise ValueError("region_cols not set")
        return list(self.region_cols)

    def filter_by_diagnosis(
        self,
        df: pd.DataFrame,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ):
        """
        Filter DataFrame by diagnosis labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        include : list, optional
            Diagnoses to include (mutually exclusive with exclude).
        exclude : list, optional
            Diagnoses to exclude.

        Returns
        -------
        pd.DataFrame
            Filtered dataframe.
        """
        if include is not None:
            return df[df[self.dx_col].isin(include)].copy()
        elif exclude is not None:
            return df[~df[self.dx_col].isin(exclude)].copy()
        return df.copy()

    def save(self, path: str):
        """
        Save processor configuration and fitted parameters.

        Parameters
        ----------
        path : str
            Output file path (.pkl).
        """
        state = {
            "region_cols": self.region_cols,
            "dx_col": self.dx_col,
            "subject_col": self.subject_col,
            "hc_mean": self._hc_mean,
            "hc_std": self._hc_std,
            "is_fitted": self._is_fitted
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str):
        """
        Load processor from saved file.

        Parameters
        ----------
        path : str
            Input file path (.pkl).

        Returns
        -------
        DataProcessor
            Loaded processor instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        processor = cls(
            region_cols=state["region_cols"],
            dx_col=state["dx_col"],
            subject_col=state["subject_col"]
        )
        processor._hc_mean = state["hc_mean"]
        processor._hc_std = state["hc_std"]
        processor._is_fitted = state["is_fitted"]

        return processor

    @property
    def is_fitted(self):
        """Check if baseline has been fitted."""
        return self._is_fitted

    @property
    def n_regions(self) -> int:
        """Number of regions."""
        if self.region_cols is None:
            raise ValueError("region_cols not set")
        return len(self.region_cols)
