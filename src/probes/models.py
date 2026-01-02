"""
Probe model definitions for analyzing model internals.

Provides:
- ProbeType enum for different probe types
- BaseProbe abstract class
- ChoiceProbe for binary choice classification
- TimeHorizonCategoryProbe for time horizon category classification
- TimeHorizonValueProbe for time horizon regression
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Probe Types
# =============================================================================


class ProbeType(Enum):
    """Types of probes for analyzing model internals."""
    CHOICE = "choice"  # Binary: short_term (0) vs long_term (1)
    TIME_HORIZON_CATEGORY = "time_horizon_category"  # Binary: short <=1yr (0) vs long >1yr (1)
    TIME_HORIZON_VALUE = "time_horizon_value"  # Regression: months


# =============================================================================
# Result Structures
# =============================================================================


@dataclass
class ClassificationMetrics:
    """Metrics for classification probes."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int


@dataclass
class RegressionMetrics:
    """Metrics for regression probes."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mse: float  # Mean Squared Error (L² loss)
    r2: float  # R² (coefficient of determination)
    normalized_mae: float  # MAE / std(y) - <1 means better than predicting mean
    n_samples: int


@dataclass
class ProbeResult:
    """
    Result from training/evaluating a single probe.

    Attributes:
        layer: Layer index
        token_position_idx: Token position index
        probe_type: Type of probe
        cv_accuracy_mean: Mean cross-validation accuracy (classification only)
        cv_accuracy_std: Std of cross-validation accuracy (classification only)
        train_metrics: Metrics on training data
        test_metrics: Metrics on test data (if test data provided)
        confusion_matrix: Confusion matrix (classification only)
        n_train: Number of training samples
        n_test: Number of test samples
        n_features: Feature dimension (d_model)
    """
    layer: int
    token_position_idx: int
    probe_type: ProbeType
    cv_accuracy_mean: float = 0.0
    cv_accuracy_std: float = 0.0
    train_metrics: Optional[ClassificationMetrics | RegressionMetrics] = None
    test_metrics: Optional[ClassificationMetrics | RegressionMetrics] = None
    confusion_matrix: Optional[list[list[int]]] = None
    n_train: int = 0
    n_test: int = 0
    n_features: int = 0


# =============================================================================
# Base Probe Class
# =============================================================================


class BaseProbe(ABC):
    """Abstract base class for all probe types."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the probe to training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions array (n_samples,)
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassificationMetrics | RegressionMetrics:
        """
        Evaluate probe on data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: True labels/values (n_samples,)

        Returns:
            Metrics object
        """
        pass


# =============================================================================
# Classification Probes
# =============================================================================


class ChoiceProbe(BaseProbe):
    """
    Binary classifier for choice prediction (short_term vs long_term).

    Uses logistic regression with L2 regularization and balanced class weights
    to handle class imbalance (ensures ~50% baseline accuracy).
    """

    def __init__(self, random_state: int = 42, C: float = 1.0, max_iter: int = 1000):
        """
        Initialize choice probe.

        Args:
            random_state: Random seed for reproducibility
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
        """
        self.model = LogisticRegression(
            random_state=random_state,
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",  # Critical: ensures ~50% random baseline
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the probe to training data."""
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassificationMetrics:
        """Evaluate probe on data."""
        y_pred = self.predict(X)
        accuracy = balanced_accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            n_samples=len(y),
        )

    @property
    def coef_(self) -> np.ndarray:
        """Get learned coefficients."""
        return self.model.coef_

    @property
    def intercept_(self) -> np.ndarray:
        """Get learned intercept."""
        return self.model.intercept_


class TimeHorizonCategoryProbe(BaseProbe):
    """
    Binary classifier for time horizon category (short <=1yr vs long >1yr).

    Uses logistic regression with L2 regularization and balanced class weights
    to handle class imbalance (ensures ~50% baseline accuracy).
    """

    def __init__(self, random_state: int = 42, C: float = 1.0, max_iter: int = 1000):
        """
        Initialize time horizon category probe.

        Args:
            random_state: Random seed for reproducibility
            C: Inverse regularization strength
            max_iter: Maximum iterations for solver
        """
        self.model = LogisticRegression(
            random_state=random_state,
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",  # Critical: ensures ~50% random baseline
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the probe to training data."""
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassificationMetrics:
        """Evaluate probe on data."""
        y_pred = self.predict(X)
        accuracy = balanced_accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average="binary", zero_division=0
        )
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            n_samples=len(y),
        )

    @property
    def coef_(self) -> np.ndarray:
        """Get learned coefficients."""
        return self.model.coef_


# =============================================================================
# Regression Probes
# =============================================================================


class TimeHorizonValueProbe(BaseProbe):
    """
    Regressor for time horizon value in months.

    Uses Ridge regression with L2 regularization.
    """

    def __init__(self, random_state: int = 42, alpha: float = 1.0):
        """
        Initialize time horizon value probe.

        Args:
            random_state: Random seed for reproducibility
            alpha: Regularization strength
        """
        self.model = Ridge(
            random_state=random_state,
            alpha=alpha,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the probe to training data."""
        self.model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise RuntimeError("Probe must be fitted before prediction")
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> RegressionMetrics:
        """Evaluate probe on data."""
        y_pred = self.predict(X)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)

        # Normalized MAE: MAE / std(y)
        # Values < 1 mean probe is better than predicting mean
        y_std = np.std(y)
        normalized_mae = mae / y_std if y_std > 0 else float('inf')

        return RegressionMetrics(
            mae=mae,
            rmse=rmse,
            mse=mse,
            r2=r2,
            normalized_mae=normalized_mae,
            n_samples=len(y),
        )

    @property
    def coef_(self) -> np.ndarray:
        """Get learned coefficients."""
        return self.model.coef_


# =============================================================================
# Factory Function
# =============================================================================


def create_probe(
    probe_type: ProbeType,
    random_state: int = 42,
    **kwargs,
) -> BaseProbe:
    """
    Create a probe of the specified type.

    Args:
        probe_type: Type of probe to create
        random_state: Random seed
        **kwargs: Additional arguments for probe constructor

    Returns:
        Probe instance
    """
    if probe_type == ProbeType.CHOICE:
        return ChoiceProbe(random_state=random_state, **kwargs)
    elif probe_type == ProbeType.TIME_HORIZON_CATEGORY:
        return TimeHorizonCategoryProbe(random_state=random_state, **kwargs)
    elif probe_type == ProbeType.TIME_HORIZON_VALUE:
        return TimeHorizonValueProbe(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
