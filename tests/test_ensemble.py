"""Unit tests for orcetra.models.ensemble module."""
import numpy as np
import pytest
from dataclasses import dataclass
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
)
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from orcetra.models.ensemble import compute_weights, select_diverse_models, build_ensemble, try_ensemble


# ── Helpers ──────────────────────────────────────────────────────────

@dataclass
class FakeProposal:
    description: str
    model: object
    preprocessor: object = None


class FakeMetric:
    def __init__(self, direction):
        self.direction = direction
        self.name = "fake"

    def compute(self, y_true, y_pred):
        from sklearn.metrics import mean_squared_error, accuracy_score
        if self.direction == "minimize":
            return mean_squared_error(y_true, y_pred)
        return accuracy_score(y_true, y_pred)


class FakeCache:
    def __init__(self, proposals):
        self.proposals = proposals


def _make_regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte, "task_type": "regression"}


def _make_classification_data():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte, "task_type": "classification"}


# ── Tests: compute_weights ───────────────────────────────────────────

class TestComputeWeights:
    def test_maximize_higher_score_gets_higher_weight(self):
        weights = compute_weights([0.9, 0.8, 0.7], "maximize")
        assert weights[0] > weights[1] > weights[2]

    def test_minimize_lower_score_gets_higher_weight(self):
        weights = compute_weights([1.0, 2.0, 3.0], "minimize")
        assert weights[0] > weights[1] > weights[2]

    def test_weights_sum_to_one(self):
        weights = compute_weights([0.5, 0.8, 0.6], "maximize")
        assert abs(sum(weights) - 1.0) < 1e-6

    def test_equal_scores_equal_weights(self):
        weights = compute_weights([0.5, 0.5, 0.5], "maximize")
        assert abs(weights[0] - weights[1]) < 1e-6
        assert abs(weights[1] - weights[2]) < 1e-6


# ── Tests: select_diverse_models ─────────────────────────────────────

class TestSelectDiverseModels:
    def test_selects_diverse_model_types(self):
        proposals = [
            (FakeProposal("rf1", RandomForestRegressor()), 1.0),
            (FakeProposal("rf2", RandomForestRegressor()), 1.1),
            (FakeProposal("gbm", GradientBoostingRegressor()), 1.2),
            (FakeProposal("et", ExtraTreesRegressor()), 1.5),
        ]
        selected = select_diverse_models(proposals, "minimize", top_k=3)
        types = [type(p.model).__name__ for p, _ in selected]
        assert len(types) == len(set(types)), "Should pick distinct model types"

    def test_skips_proposals_with_preprocessor(self):
        proposals = [
            (FakeProposal("rf", RandomForestRegressor(), preprocessor="scaler"), 1.0),
            (FakeProposal("gbm", GradientBoostingRegressor()), 1.2),
            (FakeProposal("et", ExtraTreesRegressor()), 1.5),
        ]
        selected = select_diverse_models(proposals, "minimize", top_k=3)
        assert len(selected) == 2

    def test_returns_empty_if_fewer_than_2_eligible(self):
        proposals = [
            (FakeProposal("rf", RandomForestRegressor()), 1.0),
        ]
        assert select_diverse_models(proposals, "minimize") == []

    def test_respects_top_k(self):
        proposals = [
            (FakeProposal("rf", RandomForestRegressor()), 1.0),
            (FakeProposal("gbm", GradientBoostingRegressor()), 1.2),
            (FakeProposal("et", ExtraTreesRegressor()), 1.5),
        ]
        selected = select_diverse_models(proposals, "minimize", top_k=2)
        assert len(selected) == 2


# ── Tests: build_ensemble ────────────────────────────────────────────

class TestBuildEnsemble:
    def test_regression_returns_score_and_description(self):
        data = _make_regression_data()
        selected = [
            (FakeProposal("rf", RandomForestRegressor(n_estimators=10, random_state=42)), 1.0),
            (FakeProposal("et", ExtraTreesRegressor(n_estimators=10, random_state=42)), 1.2),
        ]
        result = build_ensemble(selected, data, FakeMetric("minimize"))
        assert result is not None
        score, desc = result
        assert isinstance(score, float)
        assert "Ensemble" in desc

    def test_classification_returns_score_and_description(self):
        data = _make_classification_data()
        selected = [
            (FakeProposal("rf", RandomForestClassifier(n_estimators=10, random_state=42)), 0.8),
            (FakeProposal("et", ExtraTreesClassifier(n_estimators=10, random_state=42)), 0.75),
        ]
        result = build_ensemble(selected, data, FakeMetric("maximize"))
        assert result is not None
        score, desc = result
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_time_budget_zero_returns_none(self):
        data = _make_regression_data()
        selected = [
            (FakeProposal("rf", RandomForestRegressor(n_estimators=10, random_state=42)), 1.0),
            (FakeProposal("et", ExtraTreesRegressor(n_estimators=10, random_state=42)), 1.2),
        ]
        result = build_ensemble(selected, data, FakeMetric("minimize"), time_budget=0)
        assert result is None


# ── Tests: try_ensemble (integration) ────────────────────────────────

class TestTryEnsemble:
    def test_returns_none_if_fewer_than_3_proposals(self):
        cache = FakeCache([])
        result = try_ensemble(cache, {}, FakeMetric("minimize"), None)
        assert result is None

    def test_full_pipeline_regression(self):
        from rich.console import Console
        data = _make_regression_data()
        metric = FakeMetric("minimize")

        models = [
            RandomForestRegressor(n_estimators=10, random_state=42),
            GradientBoostingRegressor(n_estimators=10, random_state=42),
            ExtraTreesRegressor(n_estimators=10, random_state=42),
        ]
        proposals = []
        for m in models:
            p = FakeProposal(type(m).__name__, m)
            m.fit(data["X_train"], data["y_train"])
            score = metric.compute(data["y_test"], m.predict(data["X_test"]))
            proposals.append((p, score))

        cache = FakeCache(proposals)
        console = Console(quiet=True)
        result = try_ensemble(cache, data, metric, console)
        assert result is not None
        score, desc = result
        assert isinstance(score, float)
