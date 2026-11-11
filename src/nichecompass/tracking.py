"""
This module contains helpers for the optional experiment tracking with mlflow.
"""

from types import ModuleType


def import_mlflow() -> ModuleType:
    """
    Import mlflow, which is only needed when an `mlflow_experiment_id` is
    given.

    Returns
    ----------
    mlflow:
        The mlflow module.
    """
    try:
        import mlflow
    except ImportError as error:
        raise ImportError(
            "Tracking with an mlflow_experiment_id needs mlflow. Install it "
            "with 'pip install nichecompass[tracking]'.") from error
    return mlflow
