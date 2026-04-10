"""ML-powered router that learns optimal model selection from past traces.

Uses a scikit-learn RandomForestClassifier to predict the best
``provider::model`` for a task based on features extracted from the task
specification.  Falls back to the parent ``Router`` heuristic when:

- No model has been trained yet (cold start)
- The predicted model is no longer in the registered pool
- Prediction confidence is below ``min_confidence``

Auto-retraining fires every ``retrain_every_n`` new traces, keeping the
model fresh without manual intervention.

Optional dependencies: ``scikit-learn>=1.4``, ``numpy>=1.26``.
Install with ``pip install 'kortex-ai[ml]'``.

Example::

    from kortex.router.learned_router import LearnedRouter
    from kortex.core.trace import TaskTrace

    router = LearnedRouter()
    router.register_model(model_a)
    router.register_model(model_b)

    # Train on historical traces
    report = router.train(past_traces)
    print(report)

    # Route as normal — ML kicks in automatically
    decision = await router.route(task)
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from kortex.core.router import Router

if TYPE_CHECKING:
    from kortex.core.trace import TaskTrace
    from kortex.core.types import RoutingDecision, TaskSpec

logger = structlog.get_logger(component="learned_router")

# Canonical capability order used for feature encoding (must be stable)
_CAP_ORDER = [
    "reasoning",
    "code_generation",
    "content_generation",
    "vision",
    "analysis",
    "math",
    "function_calling",
    "data_analysis",
]

_COMPLEXITY_MAP = {"simple": 0.0, "moderate": 1.0, "complex": 2.0}


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------


@dataclass
class TrainingReport:
    """Summary of a training run.

    Args:
        num_samples: Number of (task, step) training examples used.
        num_classes: Number of distinct models in the training set.
        accuracy: Out-of-bag accuracy estimate (0–1).
        feature_importances: Mapping of feature name → importance score.
        trained_at: ISO timestamp of when training completed.
        model_labels: List of model keys in label order.
    """

    num_samples: int
    num_classes: int
    accuracy: float
    feature_importances: dict[str, float]
    trained_at: str
    model_labels: list[str]

    def __str__(self) -> str:
        top = sorted(
            self.feature_importances.items(), key=lambda kv: kv[1], reverse=True
        )[:3]
        top_str = ", ".join(f"{k}={v:.3f}" for k, v in top)
        return (
            f"TrainingReport(samples={self.num_samples}, "
            f"classes={self.num_classes}, "
            f"oob_accuracy={self.accuracy:.3f}, "
            f"top_features=[{top_str}], "
            f"trained_at={self.trained_at})"
        )


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_FEATURE_NAMES = ["complexity", "num_required_caps"] + [
    f"cap_{c}" for c in _CAP_ORDER
]


def _task_to_features(task: "TaskSpec") -> list[float]:
    """Extract a fixed-length feature vector from a TaskSpec.

    Args:
        task: The task specification.

    Returns:
        A list of floats encoding complexity and capability flags.
    """
    complexity = _COMPLEXITY_MAP.get(task.complexity_hint or "moderate", 1.0)
    caps = set(task.required_capabilities or [])
    num_caps = float(len(caps))
    cap_flags = [1.0 if c in caps else 0.0 for c in _CAP_ORDER]
    return [complexity, num_caps, *cap_flags]


def _extract_training_pairs(
    traces: list["TaskTrace"],
) -> tuple[list[list[float]], list[str]]:
    """Build (features, label) pairs from a list of traces.

    Only successful steps are included. The label is ``provider::model``.

    Args:
        traces: Historical TaskTrace objects.

    Returns:
        Tuple of (X_rows, y_labels).
    """
    from kortex.core.types import TaskSpec

    X: list[list[float]] = []
    y: list[str] = []

    for trace in traces:
        if not trace.success:
            continue
        for step in trace.steps:
            rd = step.routing_decision
            model_key = f"{rd.get('chosen_provider', '')}::{rd.get('chosen_model', '')}"
            if not rd.get("chosen_provider") or not rd.get("chosen_model"):
                continue

            # Reconstruct a minimal TaskSpec for feature extraction
            caps: list[str] = []
            if trace.pipeline and step.step_index < len(trace.pipeline):
                # Use whatever required_capabilities were set in the input payload
                caps = step.input_payload.get("required_capabilities", [])

            task = TaskSpec(
                content=trace.task_content,
                complexity_hint=trace.task_complexity or "moderate",
                required_capabilities=caps,
            )
            X.append(_task_to_features(task))
            y.append(model_key)

    return X, y


# ---------------------------------------------------------------------------
# LearnedRouter
# ---------------------------------------------------------------------------


class LearnedRouter(Router):
    """A Router subclass that predicts model selection via a RandomForest.

    Extends the heuristic ``Router`` with an optional ML prediction layer.
    When the model is trained and confidence is sufficient the ML prediction
    overrides the heuristic; otherwise falls back transparently.

    Args:
        model_dir: Directory where the trained model is persisted.
        min_training_samples: Minimum samples needed before training is viable.
        min_confidence: Minimum class probability to trust the ML prediction.
        retrain_every_n: Accumulate this many new traces before auto-retraining.
        n_estimators: Number of trees in the RandomForest.
        **router_kwargs: Forwarded to ``Router.__init__``.
    """

    def __init__(
        self,
        model_dir: str = ".kortex/learned_router",
        min_training_samples: int = 10,
        min_confidence: float = 0.6,
        retrain_every_n: int = 50,
        n_estimators: int = 100,
        **router_kwargs: Any,
    ) -> None:
        super().__init__(**router_kwargs)
        self._model_dir = Path(model_dir)
        self._min_training_samples = min_training_samples
        self._min_confidence = min_confidence
        self._retrain_every_n = retrain_every_n
        self._n_estimators = n_estimators

        # Trained artifacts
        self._clf: Any = None  # sklearn RandomForestClassifier
        self._label_encoder: Any = None  # sklearn LabelEncoder
        self._training_report: TrainingReport | None = None

        # Auto-retraining state
        self._pending_traces: list[TaskTrace] = []
        self._total_traces_trained = 0

        # Comparison tracking
        self._ml_routes = 0
        self._heuristic_routes = 0
        self._ml_cost_usd = 0.0
        self._heuristic_cost_usd = 0.0

        self._log = structlog.get_logger(component="learned_router")

        # Try to load a previously saved model
        self.load()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, traces: list["TaskTrace"]) -> TrainingReport:
        """Train the RandomForest on a batch of historical traces.

        Args:
            traces: List of TaskTrace objects (any mix of successful/failed).

        Returns:
            A TrainingReport summarising accuracy and feature importances.

        Raises:
            ImportError: If scikit-learn or numpy are not installed.
            ValueError: If fewer than ``min_training_samples`` samples are found.
        """
        try:
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and numpy are required for LearnedRouter training. "
                "Install with: pip install 'kortex-ai[ml]'"
            ) from exc

        X_raw, y_raw = _extract_training_pairs(traces)

        if len(X_raw) < self._min_training_samples:
            raise ValueError(
                f"Need at least {self._min_training_samples} training samples, "
                f"got {len(X_raw)}. Provide more traces or lower min_training_samples."
            )

        X = np.array(X_raw, dtype=np.float32)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)

        clf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X, y)

        self._clf = clf
        self._label_encoder = le
        self._total_traces_trained += len(traces)
        self._pending_traces.clear()

        importances = dict(zip(_FEATURE_NAMES, clf.feature_importances_.tolist()))
        report = TrainingReport(
            num_samples=len(X_raw),
            num_classes=len(le.classes_),
            accuracy=float(clf.oob_score_),
            feature_importances=importances,
            trained_at=datetime.now(timezone.utc).isoformat(),
            model_labels=list(le.classes_),
        )
        self._training_report = report

        self._log.info(
            "learned_router_trained",
            samples=report.num_samples,
            classes=report.num_classes,
            oob_accuracy=f"{report.accuracy:.3f}",
        )
        return report

    def add_trace(self, trace: "TaskTrace") -> None:
        """Add a new trace for future auto-retraining.

        Triggers retraining if ``retrain_every_n`` new traces have accumulated.

        Args:
            trace: The newly completed TaskTrace.
        """
        if trace.success:
            self._pending_traces.append(trace)
        if len(self._pending_traces) >= self._retrain_every_n:
            self._auto_retrain()

    def _auto_retrain(self) -> None:
        """Retrain on pending traces if viable."""
        if len(self._pending_traces) < self._min_training_samples:
            return
        try:
            self.train(self._pending_traces)
            self.save()
            self._log.info(
                "learned_router_auto_retrained",
                pending_traces=len(self._pending_traces),
            )
        except Exception as exc:
            self._log.warning("learned_router_auto_retrain_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(self, task: "TaskSpec") -> "RoutingDecision":
        """Route task, preferring ML prediction over heuristic.

        Falls back to heuristic when:
        - No trained model exists
        - Predicted model is not in the registered pool
        - Prediction confidence is below ``min_confidence``

        Args:
            task: The task to route.

        Returns:
            A RoutingDecision.
        """
        if self._clf is not None and self._label_encoder is not None:
            ml_decision = await self._predict(task)
            if ml_decision is not None:
                self._ml_routes += 1
                self._ml_cost_usd += ml_decision.estimated_cost_usd or 0.0
                return ml_decision

        # Heuristic fallback
        decision = await super().route(task)
        self._heuristic_routes += 1
        self._heuristic_cost_usd += decision.estimated_cost_usd or 0.0
        return decision

    async def _predict(self, task: "TaskSpec") -> "RoutingDecision | None":
        """Attempt to predict the best model using the trained classifier.

        Returns None if confidence is insufficient or model not available.
        """
        try:
            import numpy as np
        except ImportError:
            return None

        features = np.array([_task_to_features(task)], dtype=np.float32)
        proba = self._clf.predict_proba(features)[0]
        best_idx = int(proba.argmax())
        confidence = float(proba[best_idx])

        if confidence < self._min_confidence:
            return None

        predicted_label: str = self._label_encoder.inverse_transform([best_idx])[0]
        provider_name, model_name = predicted_label.split("::", 1)

        # Verify model is still registered
        registered = self._models  # dict[str, ProviderModel] from Router
        # Iterate registered models to find matching one
        for model in registered.values():
            if model.provider == provider_name and model.model == model_name:
                from kortex.core.types import RoutingDecision

                return RoutingDecision(
                    task_id=task.task_id,
                    chosen_provider=provider_name,
                    chosen_model=model_name,
                    estimated_cost_usd=model.estimated_cost(),
                    estimated_latency_ms=model.avg_latency_ms,
                    reasoning=(
                        f"ML prediction (confidence={confidence:.2f}): "
                        f"{predicted_label} selected via RandomForest"
                    ),
                    fallback_model=None,
                    fallback_provider=None,
                )

        # Predicted model no longer available
        self._log.info(
            "learned_router_prediction_unavailable",
            predicted=predicted_label,
            confidence=f"{confidence:.2f}",
        )
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist the trained model to disk.

        Creates ``model_dir`` if it does not exist.
        """
        if self._clf is None:
            return

        self._model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._model_dir / "model.pkl"
        meta_path = self._model_dir / "meta.json"

        with model_path.open("wb") as f:
            pickle.dump({"clf": self._clf, "le": self._label_encoder}, f)

        meta: dict[str, Any] = {
            "trained_at": self._training_report.trained_at if self._training_report else "",
            "num_samples": self._training_report.num_samples if self._training_report else 0,
            "num_classes": self._training_report.num_classes if self._training_report else 0,
            "accuracy": self._training_report.accuracy if self._training_report else 0.0,
            "model_labels": self._training_report.model_labels if self._training_report else [],
        }
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        self._log.info("learned_router_saved", path=str(model_path))

    def load(self) -> bool:
        """Load a previously saved model from disk.

        Returns:
            True if a model was loaded, False if none exists.
        """
        model_path = self._model_dir / "model.pkl"
        meta_path = self._model_dir / "meta.json"

        if not model_path.exists():
            return False

        try:
            with model_path.open("rb") as f:
                artifacts = pickle.load(f)
            self._clf = artifacts["clf"]
            self._label_encoder = artifacts["le"]

            if meta_path.exists():
                with meta_path.open() as f:
                    meta = json.load(f)
                self._training_report = TrainingReport(
                    num_samples=meta.get("num_samples", 0),
                    num_classes=meta.get("num_classes", 0),
                    accuracy=meta.get("accuracy", 0.0),
                    feature_importances={},
                    trained_at=meta.get("trained_at", ""),
                    model_labels=meta.get("model_labels", []),
                )

            self._log.info("learned_router_loaded", path=str(model_path))
            return True
        except Exception as exc:
            self._log.warning("learned_router_load_failed", error=str(exc))
            return False

    # ------------------------------------------------------------------
    # Comparison stats
    # ------------------------------------------------------------------

    def comparison_stats(self) -> dict[str, Any]:
        """Return routing comparison stats: ML vs heuristic.

        Returns:
            Dict with route counts, costs, and savings percentage.
        """
        total = self._ml_routes + self._heuristic_routes
        ml_pct = self._ml_routes / total * 100 if total else 0.0
        savings = self._heuristic_cost_usd - self._ml_cost_usd
        savings_pct = savings / self._heuristic_cost_usd * 100 if self._heuristic_cost_usd else 0.0
        return {
            "total_routes": total,
            "ml_routes": self._ml_routes,
            "heuristic_routes": self._heuristic_routes,
            "ml_pct": round(ml_pct, 1),
            "ml_cost_usd": round(self._ml_cost_usd, 6),
            "heuristic_cost_usd": round(self._heuristic_cost_usd, 6),
            "estimated_savings_usd": round(savings, 6),
            "estimated_savings_pct": round(savings_pct, 1),
            "training_report": str(self._training_report) if self._training_report else None,
        }

    @property
    def is_trained(self) -> bool:
        """Return True if a trained model is loaded."""
        return self._clf is not None

    @property
    def training_report(self) -> TrainingReport | None:
        """Return the most recent training report, or None."""
        return self._training_report
