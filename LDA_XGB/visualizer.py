"""
2D Visualization module for co-pathology analysis.

Provides plotting utilities for topic models, classification results,
and subject-level analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
from sklearn.metrics import ConfusionMatrixDisplay


class CopathologyVisualizer:
    """
    Visualization utilities for topic models and classification results.

    Parameters
    ----------
    output_dir : str
        Directory for saving figures.
    figsize : tuple
        Default figure size.
    style : str
        Matplotlib style to use.
    """

    def __init__(
        self,
        output_dir: str = "results/figures",
        figsize: Tuple[int, int] = (10, 8),
        style: str = "whitegrid"
    ):
        self.output_dir = output_dir
        self.figsize = figsize
        self.style = style

        os.makedirs(output_dir, exist_ok=True)
        sns.set_style(style)

    def plot_topic_heatmap(
        self,
        topic_patterns: np.ndarray,
        region_names: List[str],
        topic_names: Optional[List[str]] = None,
        title: str = "Topic-Region Associations",
        cmap: str = "Reds",
        save: bool = True,
        filename: str = "topic_heatmap.png"
    ):
        """
        Plot heatmap of topic-region associations.

        Parameters
        ----------
        topic_patterns : np.ndarray
            Topic-region weights, shape (n_topics, n_regions).
        region_names : list
            Names of brain regions.
        topic_names : list, optional
            Names of topics.
        title : str
            Plot title.
        cmap : str
            Colormap.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        n_topics = topic_patterns.shape[0]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        fig, ax = plt.subplots(figsize=(14, max(8, len(region_names) * 0.15)))

        sns.heatmap(
            topic_patterns.T,
            xticklabels=topic_names,
            yticklabels=region_names,
            cmap=cmap,
            ax=ax,
            cbar_kws={"label": "Weight"}
        )

        ax.set_title(title)
        ax.set_xlabel("Topics")
        ax.set_ylabel("Regions")

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_topic_distribution_by_dx(
        self,
        theta: np.ndarray,
        dx_labels: np.ndarray,
        topic_names: Optional[List[str]] = None,
        title: str = "Topic Distribution by Diagnosis",
        save: bool = True,
        filename: str = "topic_distribution_by_dx.png"
    ):
        """
        Plot boxplots of topic weights by diagnosis group.

        Parameters
        ----------
        theta : np.ndarray
            Subject-topic weights, shape (n_subjects, n_topics).
        dx_labels : np.ndarray
            Diagnosis labels for each subject.
        topic_names : list, optional
            Names of topics.
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        n_topics = theta.shape[1]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        # Build dataframe
        df = pd.DataFrame(theta, columns=topic_names)
        df["DX"] = dx_labels
        df_melted = df.melt(id_vars=["DX"], var_name="Topic", value_name="Weight")

        fig, ax = plt.subplots(figsize=self.figsize)

        sns.boxplot(
            data=df_melted,
            x="Topic",
            y="Weight",
            hue="DX",
            ax=ax
        )

        ax.set_title(title)
        ax.legend(title="Diagnosis", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_diagnosis_topic_profiles(
        self,
        theta: np.ndarray,
        dx_labels: np.ndarray,
        topic_names: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None,
        title: str = "Diagnosis Topic Profiles",
        save: bool = True,
        filename: str = "diagnosis_topic_profiles.png"
    ):
        """
        Plot radar/spider charts of mean topic weights per diagnosis.

        Parameters
        ----------
        theta : np.ndarray
            Subject-topic weights.
        dx_labels : np.ndarray
            Diagnosis labels.
        topic_names : list, optional
            Names of topics.
        label_map : dict, optional
            Mapping from topic names to display labels.
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        n_topics = theta.shape[1]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        # Compute group means
        df = pd.DataFrame(theta, columns=topic_names)
        df["DX"] = dx_labels
        group_means = df.groupby("DX").mean()

        # Labels for radar
        if label_map:
            labels = [label_map.get(t, t) for t in topic_names]
        else:
            labels = topic_names

        # Radar chart setup
        num_vars = len(topic_names)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Grid layout
        n_dx = len(group_means)
        n_cols = min(3, n_dx + 1)
        n_rows = int(np.ceil((n_dx + 1) / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 5 * n_rows),
            subplot_kw=dict(polar=True)
        )
        axes = axes.flatten()

        colors = plt.cm.tab10.colors

        # First subplot: all diagnoses combined
        ax = axes[0]
        for idx, (dx, values) in enumerate(group_means.iterrows()):
            values_closed = values.tolist() + [values.iloc[0]]
            ax.plot(
                angles, values_closed, "o-",
                linewidth=2,
                color=colors[idx % len(colors)],
                label=dx,
                alpha=0.7
            )
            ax.fill(angles, values_closed, alpha=0.1, color=colors[idx % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=12)
        ax.set_ylim(0, group_means.values.max() * 1.1)
        ax.set_title("All Diagnoses", size=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.0), fontsize=8)

        # Individual diagnosis plots
        for idx, (dx, values) in enumerate(group_means.iterrows()):
            ax = axes[idx + 1]
            values_closed = values.tolist() + [values.iloc[0]]

            ax.plot(
                angles, values_closed, "o-",
                linewidth=2,
                color=colors[idx % len(colors)]
            )
            ax.fill(angles, values_closed, alpha=0.25, color=colors[idx % len(colors)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=12)
            ax.set_ylim(0, group_means.values.max() * 1.1)
            ax.set_title(dx, size=14)

        # Hide empty subplots
        for idx in range(n_dx + 1, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(title, size=16, y=1.02)
        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_copathology_stacked_bars(
        self,
        theta: np.ndarray,
        dx_labels: np.ndarray,
        topic_names: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None,
        predictions: Optional[np.ndarray] = None,
        proba_df: Optional[pd.DataFrame] = None,
        dx_order: Optional[List[str]] = None,
        title: str = "Subject Topic Mixtures",
        save: bool = True,
        filename: str = "copathology_stacked_bars.png"
    ):
        """
        Plot stacked bar chart of topic mixtures per subject.

        Parameters
        ----------
        theta : np.ndarray
            Subject-topic weights.
        dx_labels : np.ndarray
            True diagnosis labels.
        topic_names : list, optional
            Names of topics.
        label_map : dict, optional
            Mapping from topic names to display labels.
        predictions : np.ndarray, optional
            Predicted diagnosis labels (for marking misclassifications).
        proba_df : pd.DataFrame, optional
            DataFrame with probability columns P(DX) for sorting subjects
            within each diagnosis group by descending P(true DX).
        dx_order : list of str, optional
            Manual ordering of diagnosis groups (left → right).
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        n_topics = theta.shape[1]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        # Build dataframe
        df = pd.DataFrame(theta, columns=topic_names)
        df["DX"] = dx_labels

        if predictions is not None:
            df["DX_pred"] = predictions

        # Determine diagnosis order
        if dx_order is None:
            dx_order = sorted(df["DX"].unique())

        # Sort subjects: by diagnosis group, then by P(true DX) descending if proba_df provided
        if proba_df is not None:
            # Add probability columns to df
            proba_cols = [c for c in proba_df.columns if c.startswith("P(")]
            for col in proba_cols:
                df[col] = proba_df[col].values

            sorted_blocks = []
            for dx in dx_order:
                dx_block = df[df["DX"] == dx].copy()
                true_dx_col = f"P({dx})"
                if true_dx_col in dx_block.columns:
                    dx_block = dx_block.sort_values(by=true_dx_col, ascending=False)
                sorted_blocks.append(dx_block)

            df = pd.concat(sorted_blocks).reset_index(drop=True)
        else:
            # Just sort by diagnosis order
            df["_dx_order"] = df["DX"].map({dx: i for i, dx in enumerate(dx_order)})
            df = df.sort_values("_dx_order").reset_index(drop=True)
            df = df.drop(columns=["_dx_order"])

        fig, ax = plt.subplots(figsize=(14, 5))

        bottom = np.zeros(len(df))
        colors = sns.color_palette("tab20", n_topics)

        for i, topic in enumerate(topic_names):
            display_label = label_map.get(topic, topic) if label_map else topic
            ax.bar(
                np.arange(len(df)),
                df[topic],
                bottom=bottom,
                color=colors[i],
                label=display_label
            )
            bottom += df[topic].values

        ax.set_xticks([])
        ax.set_ylabel("Topic proportion")
        ax.set_ylim(0, 1.08)
        ax.set_title(title)

        # Add diagnosis group labels and separators
        current = 0
        for dx in df["DX"].unique():
            count = (df["DX"] == dx).sum()
            ax.text(
                current + count / 2 - 0.5,
                -0.05,
                dx,
                ha="center",
                va="top",
                fontsize=12
            )
            ax.axvline(current - 0.5, color="black", linewidth=1.5)
            current += count
        ax.axvline(current - 0.5, color="black", linewidth=1.5)

        # Mark misclassified subjects
        if predictions is not None:
            misclassified = df["DX"] != df["DX_pred"]
            ax.scatter(
                np.where(misclassified)[0],
                np.ones(misclassified.sum()) * 1.03,
                color="red",
                marker="x",
                s=60,
                linewidths=2,
                label="Misclassified",
                zorder=10
            )

        ax.legend(
            title="Topics",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        accuracy: Optional[float] = None,
        normalize: bool = False,
        cmap: str = "Blues",
        title: str = "Confusion Matrix",
        save: bool = True,
        filename: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix.
        class_names : list
            Class labels.
        accuracy : float, optional
            Overall accuracy to display in title.
        normalize : bool
            Normalize values.
        cmap : str
            Colormap.
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(
            cmap=cmap,
            values_format=".2f" if normalize else "d",
            ax=ax
        )
        # ---- FIX GRIDLINE ARTIFACTS ----
        ax.grid(False)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)

        if accuracy is not None:
            ax.set_title(f"{title} (Accuracy = {accuracy:.3f})")
        else:
            ax.set_title(title)

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                bbox_inches="tight"
            )

        return fig

    def plot_prediction_probabilities(
        self,
        proba_df: pd.DataFrame,
        dx_true_col: str = "DX_true",
        dx_order: Optional[List[str]] = None,
        title: str = "Predicted Diagnosis Probabilities",
        save: bool = True,
        filename: str = "prediction_probabilities.png"
    ):
        """
        Plot stacked bar chart of prediction probabilities.

        Parameters
        ----------
        proba_df : pd.DataFrame
            DataFrame with probability columns P(DX) and DX_true column.
        dx_true_col : str
            Column name for true diagnosis.
        dx_order : list of str, optional
            Manual ordering of true diagnosis blocks (left → right).
            Example: ["CN", "MCI", "AD"]
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        proba_cols = [c for c in proba_df.columns if c.startswith("P(")]
        dx_labels = [c.replace("P(", "").replace(")", "") for c in proba_cols]

        if dx_order is None:
            dx_order = list(proba_df[dx_true_col].unique())
        else:
            # safety check
            missing = set(dx_order) - set(proba_df[dx_true_col].unique())
            if len(missing) > 0:
                raise ValueError(f"dx_order contains unknown labels: {missing}")

        # Sort by true DX and P(true DX)
        sorted_blocks = []
        for dx in dx_order:
        # for dx in proba_df[dx_true_col].unique():
            dx_block = proba_df[proba_df[dx_true_col] == dx].copy()
            true_dx_col = f"P({dx})"
            if true_dx_col in dx_block.columns:
                dx_block = dx_block.sort_values(by=true_dx_col, ascending=False)
            sorted_blocks.append(dx_block)

        df_sorted = pd.concat(sorted_blocks).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 5))

        bottom = np.zeros(len(df_sorted))
        colors = sns.color_palette("tab10", len(dx_labels))

        for i, (dx, col) in enumerate(zip(dx_labels, proba_cols)):
            ax.bar(
                np.arange(len(df_sorted)),
                df_sorted[col],
                bottom=bottom,
                color=colors[i],
                label=dx
            )
            bottom += df_sorted[col].values

        ax.set_xticks([])
        ax.set_ylabel("Predicted DX probability")
        ax.set_ylim(0, 1)
        ax.set_title(title)

        # Add diagnosis group labels
        current = 0
        for dx in df_sorted[dx_true_col].unique():
            count = (df_sorted[dx_true_col] == dx).sum()
            ax.text(
                current + count / 2 - 0.5,
                -0.05,
                dx,
                ha="center",
                va="top",
                fontsize=12
            )
            ax.axvline(current - 0.5, color="black", linewidth=1.5)
            current += count
        ax.axvline(current - 0.5, color="black", linewidth=1.5)

        ax.legend(
            title="Predicted DX",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_probability_heatmap(
        self,
        proba_df: pd.DataFrame,
        dx_true_col: str = "DX_true",
        dx_order: Optional[List[str]] = None,
        cmap: str = "Reds",
        title: str = "Prediction Probability Heatmap",
        save: bool = True,
        filename: str = "probability_heatmap.png"
    ):
        """
        Plot heatmap of prediction probabilities per subject.

        Parameters
        ----------
        proba_df : pd.DataFrame
            DataFrame with probability columns and true diagnosis.
        dx_true_col : str
            Column name for true diagnosis.
        cmap : str
            Colormap.
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        # proba_cols = [c for c in proba_df.columns if c.startswith("P(")]
        # dx_labels = [c.replace("P(", "").replace(")", "") for c in proba_cols]
        
        # if dx_order is None:
        #     dx_order = list(proba_df[dx_true_col].unique())
        # else:
        #     # safety check
        #     missing = set(dx_order) - set(proba_df[dx_true_col].unique())
        #     if len(missing) > 0:
        #         raise ValueError(f"dx_order contains unknown labels: {missing}")
        if dx_order is None:
            dx_order = list(proba_df[dx_true_col].unique())
        else:
            missing = set(dx_order) - set(proba_df[dx_true_col].unique())
            if len(missing) > 0:
                raise ValueError(f"dx_order contains unknown labels: {missing}")

        # enforce same order for predicted + true DX
        proba_cols = [f"P({dx})" for dx in dx_order]

        missing_probs = [c for c in proba_cols if c not in proba_df.columns]
        if len(missing_probs) > 0:
            raise ValueError(f"Missing probability columns: {missing_probs}")

        dx_labels = dx_order

            
        # Sort by true DX and P(true DX)
        sorted_blocks = []
        # for dx in proba_df[dx_true_col].unique():
        for dx in dx_order:
            dx_block = proba_df[proba_df[dx_true_col] == dx].copy()
            true_dx_col = f"P({dx})"
            if true_dx_col in dx_block.columns:
                dx_block = dx_block.sort_values(by=true_dx_col, ascending=False)
            sorted_blocks.append(dx_block)

        df_sorted = pd.concat(sorted_blocks).reset_index(drop=True)
        heatmap_data = df_sorted[proba_cols].values

        fig, ax = plt.subplots(figsize=(8, 10))

        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Probability"},
            yticklabels=False,
            xticklabels=dx_labels,
            ax=ax
        )

        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("Subjects")
        ax.set_title(title)

        # Add group separators and labels
        current = 0
        yticks = []
        ylabels = []

        # for dx in df_sorted[dx_true_col].unique():
        for dx in dx_order:
            count = (df_sorted[dx_true_col] == dx).sum()
            midpoint = current + count / 2
            yticks.append(midpoint)
            ylabels.append(dx)
            ax.hlines(current, xmin=0, xmax=len(dx_labels), colors="black", linewidth=1.5)
            current += count

        ax.hlines(current, xmin=0, xmax=len(dx_labels), colors="black", linewidth=1.5)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, rotation=0)

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_top_regions_per_topic(
        self,
        topic_patterns: np.ndarray,
        region_names: List[str],
        top_n: int = 10,
        topic_names: Optional[List[str]] = None,
        save: bool = True,
        filename: str = "top_regions_per_topic.png"
    ):
        """
        Plot horizontal bar charts of top regions for each topic.

        Parameters
        ----------
        topic_patterns : np.ndarray
            Topic-region weights, shape (n_topics, n_regions).
        region_names : list
            Names of brain regions.
        top_n : int
            Number of top regions to show.
        topic_names : list, optional
            Names of topics.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        n_topics = topic_patterns.shape[0]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        n_cols = min(3, n_topics)
        n_rows = int(np.ceil(n_topics / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_topics > 1 else [axes]

        colors = plt.cm.tab10.colors

        for k in range(n_topics):
            ax = axes[k]
            weights = topic_patterns[k]
            indices = np.argsort(weights)[::-1][:top_n]

            regions = [region_names[i] for i in indices]
            values = weights[indices]

            ax.barh(range(top_n), values[::-1], color=colors[k % len(colors)])
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(regions[::-1], fontsize=9)
            ax.set_xlabel("Weight")
            ax.set_title(topic_names[k])

        # Hide empty subplots
        for idx in range(n_topics, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        save: bool = True,
        filename: str = "feature_importance.png"
    ):
        """
        Plot feature importance from classifier.

        Parameters
        ----------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns.
        title : str
            Plot title.
        save : bool
            Save figure to file.
        filename : str
            Output filename.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.barh(
            importance_df["feature"],
            importance_df["importance"],
            color="steelblue"
        )

        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.invert_yaxis()

        plt.tight_layout()

        if save:
            fig.savefig(
                os.path.join(self.output_dir, filename),
                dpi=150,
                bbox_inches="tight"
            )

        return fig
