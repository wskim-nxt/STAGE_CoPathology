"""
3D Brain Surface Visualization module (optional).

Provides brain surface plotting using PyVista and DKT atlas.
Requires the atlas_vis package to be installed.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict

# Optional imports - will fail gracefully if not available
try:
    from atlas_vis import DKTAtlas62ROIPlotter
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class BrainVisualizer:
    """
    3D brain surface visualization using PyVista and DKT atlas.

    This class provides methods for rendering topic patterns and
    atrophy maps on brain surface meshes.

    Note: Requires atlas_vis and pyvista packages.

    Parameters
    ----------
    output_dir : str
        Directory for saving figures.
    cmap : str
        Colormap for surface rendering.
    clim : tuple
        Color limits (min, max) for the colormap.
    window_size : tuple
        Window size for rendering.
    background : str
        Background color.
    template_key : str
        Surface template ('pial', 'inflated', etc.).
    """

    def __init__(
        self,
        output_dir: str = "results/surface_maps",
        cmap: str = "Reds",
        clim: Tuple[float, float] = (0, 0.1),
        window_size: Tuple[int, int] = (1200, 1000),
        background: str = "white",
        template_key: str = "pial"
    ):
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "BrainVisualizer requires atlas_vis and pyvista. "
                "Install with: pip install pyvista atlas_vis"
            )

        self.output_dir = output_dir
        self.cmap = cmap
        self.clim = clim
        self.window_size = window_size
        self.background = background
        self.template_key = template_key

        os.makedirs(output_dir, exist_ok=True)

        self._plotter = DKTAtlas62ROIPlotter(
            cmap=cmap,
            clim=clim,
            window_size=window_size,
            nan_color="lightgray",
            background=background,
            template_key=template_key
        )

    def plot_topic_surface(
        self,
        topic_patterns: np.ndarray,
        topic_idx: int,
        region_names: Optional[List[str]] = None,
        filename: Optional[str] = None,
        n_cortical_per_hemi: int = 31
    ):
        """
        Plot a single topic's pattern on the brain surface.

        Parameters
        ----------
        topic_patterns : np.ndarray
            Topic-region weights, shape (n_topics, n_regions).
            Should contain 62 cortical regions (31 left, 31 right).
        topic_idx : int
            Index of the topic to plot.
        region_names : list, optional
            Names of regions (for filtering cortical only).
        filename : str, optional
            Output filename. Defaults to Topic_{idx}.png.
        n_cortical_per_hemi : int
            Number of cortical regions per hemisphere.

        Returns
        -------
        str
            Path to saved image.
        """
        if filename is None:
            filename = f"Topic_{topic_idx}.png"

        save_path = os.path.join(self.output_dir, filename)

        # Extract topic weights
        weights = topic_patterns[topic_idx]

        # Split into left and right hemispheres
        # Assumes first 31 are left, next 31 are right (standard DKT order)
        if len(weights) == 62:
            l_values = weights[:n_cortical_per_hemi].tolist()
            r_values = weights[n_cortical_per_hemi:].tolist()
        elif len(weights) > 62:
            # If more regions (e.g., subcortical), take last 62
            cortical_weights = weights[-62:]
            l_values = cortical_weights[:n_cortical_per_hemi].tolist()
            r_values = cortical_weights[n_cortical_per_hemi:].tolist()
        else:
            raise ValueError(
                f"Expected 62 cortical regions, got {len(weights)}. "
                "Make sure topic_patterns contains DKT cortical regions."
            )

        self._plotter(l_values, r_values, save_path=save_path)

        return save_path

    def plot_all_topics(
        self,
        topic_patterns: np.ndarray,
        topic_names: Optional[List[str]] = None,
        subdir: str = "topicwise"
    ):
        """
        Plot all topics on brain surfaces.

        Parameters
        ----------
        topic_patterns : np.ndarray
            Topic-region weights, shape (n_topics, n_regions).
        topic_names : list, optional
            Names of topics for filenames.
        subdir : str
            Subdirectory for topic images.

        Returns
        -------
        list
            Paths to saved images.
        """
        n_topics = topic_patterns.shape[0]
        if topic_names is None:
            topic_names = [f"Topic_{k}" for k in range(n_topics)]

        out_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        paths = []
        for k in range(n_topics):
            filename = f"{topic_names[k]}.png"
            save_path = os.path.join(out_dir, filename)

            # Get cortical weights (last 62 if more regions)
            weights = topic_patterns[k]
            if len(weights) >= 62:
                cortical = weights[-62:] if len(weights) > 62 else weights
                l_values = cortical[:31].tolist()
                r_values = cortical[31:].tolist()

                print(f"Plotting {topic_names[k]}...")
                self._plotter(l_values, r_values, save_path=save_path)
                paths.append(save_path)

        return paths

    def plot_diagnosis_atrophy_map(
        self,
        atrophy_map: np.ndarray,
        diagnosis: str,
        filename: Optional[str] = None
    ):
        """
        Plot diagnosis-specific atrophy pattern on brain surface.

        Parameters
        ----------
        atrophy_map : np.ndarray
            Atrophy values per region, shape (n_regions,).
        diagnosis : str
            Diagnosis name for filename.
        filename : str, optional
            Output filename.

        Returns
        -------
        str
            Path to saved image.
        """
        if filename is None:
            filename = f"{diagnosis}_atrophy.png"

        save_path = os.path.join(self.output_dir, filename)

        # Extract cortical regions
        if len(atrophy_map) >= 62:
            cortical = atrophy_map[-62:] if len(atrophy_map) > 62 else atrophy_map
            l_values = cortical[:31].tolist()
            r_values = cortical[31:].tolist()

            self._plotter(l_values, r_values, save_path=save_path)

        return save_path

    def plot_from_dataframe(
        self,
        df: pd.DataFrame,
        value_col: str,
        filename: str,
        n_cortical: int = 62
    ):
        """
        Plot values from a DataFrame on brain surface.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with region values. Expects last n_cortical rows
            to be cortical regions in DKT order.
        value_col : str
            Column name containing values to plot.
        filename : str
            Output filename.
        n_cortical : int
            Number of cortical regions (default 62 for DKT).

        Returns
        -------
        str
            Path to saved image.
        """
        save_path = os.path.join(self.output_dir, filename)

        # Get cortical values from last 62 rows
        values = df[value_col].values
        if len(values) >= n_cortical:
            cortical = values[-n_cortical:]
            l_values = cortical[:31].tolist()
            r_values = cortical[31:].tolist()

            self._plotter(l_values, r_values, save_path=save_path)

        return save_path

    def update_colormap(
        self,
        cmap: str = "Reds",
        clim: Optional[Tuple[float, float]] = None
    ):
        """
        Update the colormap and color limits.

        Parameters
        ----------
        cmap : str
            New colormap.
        clim : tuple, optional
            New color limits.
        """
        self.cmap = cmap
        if clim is not None:
            self.clim = clim

        self._plotter = DKTAtlas62ROIPlotter(
            cmap=self.cmap,
            clim=self.clim,
            window_size=self.window_size,
            nan_color="lightgray",
            background=self.background,
            template_key=self.template_key
        )


def is_available():
    """Check if brain visualization dependencies are available."""
    return PYVISTA_AVAILABLE
