"""
Plotting utilities for probe analysis and choice model visualization.

Provides visualization functions for:
- Classification accuracy heatmaps
- Regression metric heatmaps (MAE, R², normalized MAE, MSE)
- Text visualization with accuracy-based coloring
- Choice model comparison plots
- Camera-ready figure export
"""

from .common import (
    TokenPositionInfo,
    build_accuracy_matrix,
    build_regression_matrix,
    format_token_position_label,
    get_camera_ready_filename,
    get_probe_type_info,
)
from .heatmaps import (
    create_accuracy_heatmap,
    create_regression_heatmap,
    create_regression_metric_heatmap,
    create_unit_comparison_plot,
)
from .text_viz import (
    create_text_only_visualization,
    render_sample_text_with_accuracy,
)
from .export import export_camera_ready

__all__ = [
    # Common utilities
    "TokenPositionInfo",
    "build_accuracy_matrix",
    "build_regression_matrix",
    "format_token_position_label",
    "get_camera_ready_filename",
    "get_probe_type_info",
    # Heatmaps
    "create_accuracy_heatmap",
    "create_regression_heatmap",
    "create_regression_metric_heatmap",
    "create_unit_comparison_plot",
    # Text visualization
    "create_text_only_visualization",
    "render_sample_text_with_accuracy",
    # Export
    "export_camera_ready",
]
