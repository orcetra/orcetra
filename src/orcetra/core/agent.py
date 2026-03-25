"""
AI Agent interface for the AutoResearch loop.

The agent reads the current pipeline state, proposes a modification,
and the loop tests whether it improves the metric.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import random
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
    VotingRegressor, VotingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA

# Optional: XGBoost / LightGBM
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


@dataclass
class Proposal:
    """A proposed modification to the prediction pipeline."""
    description: str
    rationale: str
    model: object  # sklearn model instance
    preprocessor: object = None  # optional preprocessor
    
    def evaluate(self, data_info, metric_fn):
        """Evaluate this proposal on the given data."""
        X_train = data_info["X_train"].copy()
        X_test = data_info["X_test"].copy()
        
        if self.preprocessor:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)
        
        self.model.fit(X_train, data_info["y_train"])
        preds = self.model.predict(X_test)
        return metric_fn.compute(data_info["y_test"], preds)


class Agent(ABC):
    """Base class for AutoResearch agents."""
    
    @abstractmethod
    def propose(self, state: dict) -> Proposal:
        """Given current state, propose a pipeline modification."""
        ...


# Weighted model pools for tabular data
TREE_MODELS = ["rf", "gbm", "histgbm", "extra"]
if HAS_XGB:
    TREE_MODELS.append("xgb")
if HAS_LGBM:
    TREE_MODELS.append("lgbm")

LINEAR_MODELS = ["linear", "ridge", "lasso", "elastic"]
OTHER_MODELS = ["knn", "ada", "svr"]


def _weighted_model_choice(best_model_type: Optional[str] = None) -> str:
    """Pick a model type with smart weighting for tabular data."""
    # If we know what's winning, 50% chance to refine it
    if best_model_type and random.random() < 0.5:
        return best_model_type
    
    # Otherwise: 70% tree, 20% linear, 10% other
    r = random.random()
    if r < 0.70:
        return random.choice(TREE_MODELS)
    elif r < 0.90:
        return random.choice(LINEAR_MODELS)
    else:
        return random.choice(OTHER_MODELS)


class RandomSearchAgent(Agent):
    """Agent that explores models using weighted random search."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.best_model_type = None  # Track winning model family
        self.has_improvement = False
        
    def propose(self, state: dict) -> Proposal:
        # Track what's winning
        best_model = state.get("best_model", "")
        if state.get("iteration", 0) > 1:
            self.best_model_type = self._detect_model_type(best_model)
        
        # Ensemble attempt: 10% chance after iteration 15
        if state.get("iteration", 0) > 15 and random.random() < 0.10:
            return self._propose_ensemble()
        
        model_choice = _weighted_model_choice(
            self.best_model_type if self.has_improvement else None
        )
        
        if self.task_type == "regression":
            model, description = self._get_regression_model(model_choice)
        else:
            model, description = self._get_classification_model(model_choice)
        
        # Preprocessing: only for linear/knn/svr models (tree models don't need it)
        if model_choice in LINEAR_MODELS + ["knn", "svr", "svc"]:
            preprocessor, prep_desc = self._get_preprocessor()
        else:
            preprocessor, prep_desc = None, ""
        
        if prep_desc:
            description = f"{prep_desc} + {description}"
        
        return Proposal(
            description=description,
            rationale=f"Weighted search (bias={self.best_model_type})",
            model=model,
            preprocessor=preprocessor,
        )
    
    def _detect_model_type(self, model_name: str) -> Optional[str]:
        """Detect model family from description string."""
        name_lower = model_name.lower()
        if "xgb" in name_lower: return "xgb"
        if "lgbm" in name_lower or "lightgbm" in name_lower: return "lgbm"
        if "histgbm" in name_lower or "histgradient" in name_lower: return "histgbm"
        if "gradientboosting" in name_lower or "gbm(" in name_lower: return "gbm"
        if "randomforest" in name_lower or "rf(" in name_lower: return "rf"
        if "extratrees" in name_lower: return "extra"
        if "ridge" in name_lower: return "ridge"
        if "lasso" in name_lower: return "lasso"
        return None

    def _get_regression_model(self, choice: str):
        if choice == "linear":
            return LinearRegression(), "LinearRegression"
        elif choice == "ridge":
            alpha = random.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
            return Ridge(alpha=alpha, random_state=42), f"Ridge(alpha={alpha})"
        elif choice == "lasso":
            alpha = random.choice([0.001, 0.01, 0.1, 0.5, 1.0])
            return Lasso(alpha=alpha, random_state=42, max_iter=2000), f"Lasso(alpha={alpha})"
        elif choice == "elastic":
            alpha = random.choice([0.001, 0.01, 0.1, 0.5, 1.0])
            l1 = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
            return ElasticNet(alpha=alpha, l1_ratio=l1, random_state=42, max_iter=2000), f"ElasticNet(a={alpha},l1={l1})"
        elif choice == "rf":
            n = random.choice([100, 200, 300, 500])
            depth = random.choice([5, 10, 15, 20, None])
            min_split = random.choice([2, 5, 10])
            return RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_split, random_state=42, n_jobs=-1), f"RF(n={n},d={depth},ms={min_split})"
        elif choice == "gbm":
            n = random.choice([100, 200, 300, 500, 800])
            lr = random.choice([0.01, 0.03, 0.05, 0.1, 0.2])
            depth = random.choice([3, 4, 5, 6, 7])
            sub = random.choice([0.7, 0.8, 0.9, 1.0])
            return GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=sub, random_state=42), f"GBM(n={n},lr={lr},d={depth},sub={sub})"
        elif choice == "histgbm":
            n = random.choice([100, 200, 300, 500, 800])
            lr = random.choice([0.01, 0.03, 0.05, 0.1, 0.2])
            depth = random.choice([3, 5, 7, 10, None])
            leaf = random.choice([10, 20, 30, 50])
            return HistGradientBoostingRegressor(max_iter=n, learning_rate=lr, max_depth=depth, min_samples_leaf=leaf, random_state=42), f"HistGBM(n={n},lr={lr},d={depth},leaf={leaf})"
        elif choice == "xgb" and HAS_XGB:
            n = random.choice([100, 200, 300, 500, 800])
            lr = random.choice([0.01, 0.03, 0.05, 0.1, 0.2])
            depth = random.choice([3, 4, 5, 6, 7, 8])
            sub = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
            col = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
            alpha = random.choice([0, 0.01, 0.1, 1.0])
            lam = random.choice([0, 0.01, 0.1, 1.0])
            return XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=sub, colsample_bytree=col, reg_alpha=alpha, reg_lambda=lam, random_state=42, n_jobs=-1, verbosity=0), f"XGB(n={n},lr={lr},d={depth},sub={sub},col={col})"
        elif choice == "lgbm" and HAS_LGBM:
            n = random.choice([100, 200, 300, 500, 800])
            lr = random.choice([0.01, 0.03, 0.05, 0.1, 0.2])
            depth = random.choice([-1, 3, 5, 7, 10])
            leaves = random.choice([15, 31, 50, 80, 127])
            sub = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
            col = random.choice([0.6, 0.7, 0.8, 0.9, 1.0])
            return LGBMRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, num_leaves=leaves, subsample=sub, colsample_bytree=col, random_state=42, n_jobs=-1, verbose=-1), f"LGBM(n={n},lr={lr},d={depth},lv={leaves})"
        elif choice == "extra":
            n = random.choice([100, 200, 300, 500])
            depth = random.choice([5, 10, 15, 20, None])
            return ExtraTreesRegressor(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1), f"ExtraTrees(n={n},d={depth})"
        elif choice == "knn":
            k = random.choice([3, 5, 7, 10, 15])
            w = random.choice(["uniform", "distance"])
            return KNeighborsRegressor(n_neighbors=k, weights=w, n_jobs=-1), f"KNN(k={k},w={w})"
        elif choice == "ada":
            n = random.choice([50, 100, 200])
            lr = random.choice([0.1, 0.5, 1.0])
            return AdaBoostRegressor(n_estimators=n, learning_rate=lr, random_state=42), f"AdaBoost(n={n},lr={lr})"
        elif choice == "svr":
            C = random.choice([0.1, 1.0, 10.0])
            return SVR(C=C), f"SVR(C={C})"
        else:
            # Fallback to GBM
            return GradientBoostingRegressor(n_estimators=200, random_state=42), "GBM(n=200,default)"

    def _get_classification_model(self, choice: str):
        if choice == "linear" or choice == "logistic":
            C = random.choice([0.01, 0.1, 1.0, 10.0])
            return LogisticRegression(C=C, max_iter=1000, random_state=42), f"LogReg(C={C})"
        elif choice == "rf":
            n = random.choice([100, 200, 300, 500])
            depth = random.choice([5, 10, 15, None])
            return RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1), f"RF(n={n},d={depth})"
        elif choice == "gbm":
            n = random.choice([100, 200, 300, 500])
            lr = random.choice([0.01, 0.05, 0.1, 0.2])
            depth = random.choice([3, 5, 7])
            return GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, random_state=42), f"GBM(n={n},lr={lr},d={depth})"
        elif choice == "histgbm":
            n = random.choice([100, 200, 300, 500])
            lr = random.choice([0.01, 0.05, 0.1, 0.2])
            depth = random.choice([3, 5, 7, None])
            return HistGradientBoostingClassifier(max_iter=n, learning_rate=lr, max_depth=depth, random_state=42), f"HistGBM(n={n},lr={lr},d={depth})"
        elif choice == "xgb" and HAS_XGB:
            n = random.choice([100, 200, 300, 500])
            lr = random.choice([0.01, 0.05, 0.1, 0.2])
            depth = random.choice([3, 5, 6, 7])
            return XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='logloss'), f"XGB(n={n},lr={lr},d={depth})"
        elif choice == "lgbm" and HAS_LGBM:
            n = random.choice([100, 200, 300, 500])
            lr = random.choice([0.01, 0.05, 0.1, 0.2])
            leaves = random.choice([15, 31, 50, 80])
            return LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves, random_state=42, n_jobs=-1, verbose=-1), f"LGBM(n={n},lr={lr},lv={leaves})"
        elif choice == "extra":
            n = random.choice([100, 200, 300])
            depth = random.choice([5, 10, 15, None])
            return ExtraTreesClassifier(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1), f"ExtraTrees(n={n},d={depth})"
        elif choice == "knn":
            k = random.choice([3, 5, 7, 10])
            return KNeighborsClassifier(n_neighbors=k, n_jobs=-1), f"KNN(k={k})"
        elif choice == "ada":
            n = random.choice([50, 100, 200])
            return AdaBoostClassifier(n_estimators=n, random_state=42), f"AdaBoost(n={n})"
        elif choice in ("svr", "svc"):
            C = random.choice([0.1, 1.0, 10.0])
            return SVC(C=C, random_state=42), f"SVC(C={C})"
        elif choice in ("ridge", "lasso", "elastic"):
            return LogisticRegression(max_iter=1000, random_state=42), "LogReg(default)"
        else:
            return GradientBoostingClassifier(n_estimators=200, random_state=42), "GBM(n=200,default)"

    def _get_preprocessor(self):
        choice = random.choice(["standard", "minmax", "robust", "quantile", "power"])
        if choice == "standard":
            return StandardScaler(), "StdScale"
        elif choice == "minmax":
            return MinMaxScaler(), "MinMax"
        elif choice == "robust":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler(), "Robust"
        elif choice == "quantile":
            return QuantileTransformer(n_quantiles=100, random_state=42), "Quantile"
        elif choice == "power":
            return PowerTransformer(method="yeo-johnson"), "PowerTx"
        return None, ""

    def _propose_ensemble(self):
        if self.task_type == "regression":
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
                ("histgbm", HistGradientBoostingRegressor(max_iter=200, random_state=42)),
                ("gbm", GradientBoostingRegressor(n_estimators=200, random_state=42)),
            ]
            if HAS_XGB:
                estimators.append(("xgb", XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbosity=0)))
            if HAS_LGBM:
                estimators.append(("lgbm", LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)))
            model = VotingRegressor(estimators=estimators, n_jobs=-1)
            return Proposal(description="Ensemble(RF+HistGBM+GBM" + ("+XGB" if HAS_XGB else "") + ("+LGBM" if HAS_LGBM else "") + ")", rationale="Ensemble of best tree models", model=model)
        else:
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
                ("histgbm", HistGradientBoostingClassifier(max_iter=200, random_state=42)),
            ]
            model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
            return Proposal(description="Ensemble(RF+HistGBM)", rationale="Ensemble", model=model)
