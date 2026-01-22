"""
Co-Pathology Analysis Pipeline

A modular framework for discovering and classifying brain atrophy patterns
using Latent Dirichlet Allocation (LDA) and XGBoost classification.

Modules
-------
- data_processor: Data loading, Z-scoring, atrophy computation
- lda_model: LDA topic model for pattern discovery
- classifier: XGBoost classification with cross-validation
- visualizer: 2D plotting utilities
- brain_visualizer: 3D brain surface visualization (optional)
- pipeline: End-to-end orchestration

Example Usage
-------------
# Training
>>> from copathology_final_script import CopathologyPipeline
>>> pipeline = CopathologyPipeline(n_topics=6)
>>> results = pipeline.fit(patient_df, hc_df, region_cols)
>>> pipeline.save("models/pipeline.pkl")

# Inference on new subjects
>>> pipeline = CopathologyPipeline.load("models/pipeline.pkl")
>>> predictions = pipeline.predict_new_subjects(new_df)
"""

from .data_processor import DataProcessor
from .lda_model import LDATopicModel
from .classifier import TopicClassifier
from .visualizer import CopathologyVisualizer
from .pipeline import CopathologyPipeline, run_analysis

# Optional brain visualizer
try:
    from .brain_visualizer import BrainVisualizer, is_available as brain_vis_available
except ImportError:
    BrainVisualizer = None
    brain_vis_available = lambda: False

__all__ = [
    "DataProcessor",
    "LDATopicModel",
    "TopicClassifier",
    "CopathologyVisualizer",
    "CopathologyPipeline",
    "BrainVisualizer",
    "run_analysis",
    "brain_vis_available"
]

__version__ = "1.0.0"
