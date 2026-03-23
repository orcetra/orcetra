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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier, VotingRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, QuantileTransformer, PowerTransformer
from sklearn.decomposition import PCA

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

class RandomSearchAgent(Agent):
    """Agent that intelligently explores different models using guided random search."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        self.history = []  # Track what's been tried and their scores
        self.model_success_rates = {}  # Track which model types work well
        self.best_hyperparams = {}  # Remember good hyperparameter ranges
        
    def propose(self, state: dict) -> Proposal:
        """Propose a guided random model configuration."""
        # Update history based on current state
        if "best_score" in state and state["best_score"] is not None:
            self._update_history(state)
        
        # Decide whether to try ensemble (10% chance after iteration 20)
        if state.get("iteration", 0) > 20 and random.random() < 0.1:
            return self._propose_ensemble()
        
        # Guided model selection based on history
        if self.task_type == "regression":
            model, description = self._get_guided_regression_model()
        else:  # classification
            model, description = self._get_guided_classification_model()
        
        # Smart preprocessing selection
        preprocessor, preprocess_desc = self._get_guided_preprocessor()
        
        if preprocess_desc:
            description = f"{preprocess_desc} + {description}"
        
        return Proposal(
            description=description,
            rationale="Guided random search with history",
            model=model,
            preprocessor=preprocessor,
        )
    
    def _update_history(self, state: dict):
        """Update model performance history for smarter future choices."""
        # Simple tracking of which model types are performing well
        best_model = state.get("best_model", "")
        if "RF(" in best_model:
            self.model_success_rates["rf"] = self.model_success_rates.get("rf", 0) + 1
        elif "GBM(" in best_model or "HistGBM(" in best_model:
            self.model_success_rates["gbm"] = self.model_success_rates.get("gbm", 0) + 1
        elif "XGB(" in best_model:
            self.model_success_rates["xgb"] = self.model_success_rates.get("xgb", 0) + 1
    
    def _get_guided_regression_model(self):
        """Get a guided regression model based on success history."""
        # Prefer models that have shown success, otherwise random
        successful_models = [k for k, v in self.model_success_rates.items() if v > 0]
        
        if successful_models and random.random() < 0.3:  # 30% chance to exploit successful models
            model_choice = random.choice(successful_models)
        else:  # 70% chance to explore, including new models
            model_choice = random.choice([
                "linear", "ridge", "lasso", "elastic", "rf", "gbm", "histgbm", "knn", "extra", "ada", "svr"
            ])
        
        return self._get_regression_model(model_choice)
    
    def _get_regression_model(self, model_choice):
        """Get a random regression model with random hyperparameters."""
        model_choice = random.choice([
            "linear", "ridge", "lasso", "elastic", "rf", "gbm", "knn", "extra", "ada", "svr"
        ])
        
        if model_choice == "linear":
            return LinearRegression(), "LinearRegression"
        
        elif model_choice == "ridge":
            alpha = random.choice([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
            return Ridge(alpha=alpha, random_state=42), f"Ridge(alpha={alpha})"
        
        elif model_choice == "lasso":
            alpha = random.choice([0.01, 0.1, 0.5, 1.0, 2.0])
            return Lasso(alpha=alpha, random_state=42, max_iter=2000), f"Lasso(alpha={alpha})"
        
        elif model_choice == "elastic":
            alpha = random.choice([0.01, 0.1, 0.5, 1.0])
            l1_ratio = random.choice([0.1, 0.3, 0.5, 0.7, 0.9])
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000), f"ElasticNet(alpha={alpha}, l1={l1_ratio})"
        
        elif model_choice == "rf":
            n_estimators = random.choice([50, 100, 200, 300])
            max_depth = random.choice([3, 5, 10, 15, None])
            min_samples_split = random.choice([2, 5, 10, 20])
            return RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42, 
                n_jobs=-1
            ), f"RF(n={n_estimators}, depth={max_depth}, min_split={min_samples_split})"
        
        elif model_choice == "gbm":
            n_estimators = random.choice([50, 100, 200, 300])
            learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
            max_depth = random.choice([3, 5, 7, 10])
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ), f"GBM(n={n_estimators}, lr={learning_rate}, depth={max_depth})"
        
        elif model_choice == "knn":
            n_neighbors = random.choice([3, 5, 7, 10, 15])
            weights = random.choice(["uniform", "distance"])
            return KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1), f"KNN(k={n_neighbors}, weights={weights})"
        
        elif model_choice == "extra":
            n_estimators = random.choice([50, 100, 200, 300])
            max_depth = random.choice([3, 5, 10, 15, None])
            min_samples_split = random.choice([2, 5, 10, 20])
            return ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            ), f"ExtraTrees(n={n_estimators}, depth={max_depth}, min_split={min_samples_split})"
        
        elif model_choice == "ada":
            n_estimators = random.choice([50, 100, 200])
            learning_rate = random.choice([0.1, 0.5, 1.0, 2.0])
            return AdaBoostRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            ), f"AdaBoost(n={n_estimators}, lr={learning_rate})"
        
        elif model_choice == "histgbm":
            max_iter = random.choice([50, 100, 200, 300])
            learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
            max_depth = random.choice([None, 3, 5, 7, 10])
            return HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ), f"HistGBM(iter={max_iter}, lr={learning_rate}, depth={max_depth})"
        
        elif model_choice == "svr":
            C = random.choice([0.1, 1.0, 10.0, 100.0])
            epsilon = random.choice([0.01, 0.1, 0.2])
            return SVR(C=C, epsilon=epsilon), f"SVR(C={C}, eps={epsilon})"
    
    def _get_guided_classification_model(self):
        """Get a guided classification model based on success history."""
        successful_models = [k for k, v in self.model_success_rates.items() if v > 0]
        
        if successful_models and random.random() < 0.3:
            model_choice = random.choice(successful_models)
        else:
            model_choice = random.choice([
                "logistic", "rf", "gbm", "histgbm", "knn", "extra", "ada", "svc"
            ])
        
        return self._get_classification_model(model_choice)
    
    def _get_classification_model(self, model_choice):
        """Get a random classification model with random hyperparameters."""
        model_choice = random.choice([
            "logistic", "rf", "gbm", "knn", "extra", "ada", "svc"
        ])
        
        if model_choice == "logistic":
            C = random.choice([0.01, 0.1, 1.0, 10.0, 100.0])
            return LogisticRegression(C=C, max_iter=1000, random_state=42), f"LogisticRegression(C={C})"
        
        elif model_choice == "rf":
            n_estimators = random.choice([50, 100, 200, 300])
            max_depth = random.choice([3, 5, 10, 15, None])
            min_samples_split = random.choice([2, 5, 10, 20])
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            ), f"RF(n={n_estimators}, depth={max_depth}, min_split={min_samples_split})"
        
        elif model_choice == "gbm":
            n_estimators = random.choice([50, 100, 200, 300])
            learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
            max_depth = random.choice([3, 5, 7, 10])
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ), f"GBM(n={n_estimators}, lr={learning_rate}, depth={max_depth})"
        
        elif model_choice == "knn":
            n_neighbors = random.choice([3, 5, 7, 10, 15])
            weights = random.choice(["uniform", "distance"])
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, n_jobs=-1), f"KNN(k={n_neighbors}, weights={weights})"
        
        elif model_choice == "extra":
            n_estimators = random.choice([50, 100, 200, 300])
            max_depth = random.choice([3, 5, 10, 15, None])
            min_samples_split = random.choice([2, 5, 10, 20])
            return ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            ), f"ExtraTrees(n={n_estimators}, depth={max_depth}, min_split={min_samples_split})"
        
        elif model_choice == "ada":
            n_estimators = random.choice([50, 100, 200])
            learning_rate = random.choice([0.1, 0.5, 1.0, 2.0])
            return AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            ), f"AdaBoost(n={n_estimators}, lr={learning_rate})"
        
        elif model_choice == "histgbm":
            max_iter = random.choice([50, 100, 200, 300])
            learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
            max_depth = random.choice([None, 3, 5, 7, 10])
            return HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            ), f"HistGBM(iter={max_iter}, lr={learning_rate}, depth={max_depth})"
        
        elif model_choice == "svc":
            C = random.choice([0.1, 1.0, 10.0, 100.0])
            return SVC(C=C, random_state=42), f"SVC(C={C})"
    
    def _get_guided_preprocessor(self):
        """Smart preprocessing selection with new transformers."""
        preprocessing_choice = random.choice([None, "standard", "minmax", "quantile", "power", "poly", "pca"])
        
        if preprocessing_choice is None:
            return None, ""
        elif preprocessing_choice == "standard":
            return StandardScaler(), "StandardScaler"
        elif preprocessing_choice == "minmax":
            return MinMaxScaler(), "MinMaxScaler"
        elif preprocessing_choice == "quantile":
            n_quantiles = random.choice([100, 500, 1000])
            return QuantileTransformer(n_quantiles=n_quantiles, random_state=42), f"QuantileTransformer(n={n_quantiles})"
        elif preprocessing_choice == "power":
            method = random.choice(["yeo-johnson", "box-cox"])
            try:
                return PowerTransformer(method=method, standardize=True), f"PowerTransformer({method})"
            except:
                # Fallback to standard scaler if power transform fails
                return StandardScaler(), "StandardScaler"
        elif preprocessing_choice == "poly":
            degree = random.choice([2, 3])
            return PolynomialFeatures(degree=degree, include_bias=False), f"PolyFeatures(degree={degree})"
        elif preprocessing_choice == "pca":
            n_components = random.choice([0.8, 0.9, 0.95])
            return PCA(n_components=n_components, random_state=42), f"PCA(n_comp={n_components})"
    
    def _propose_ensemble(self):
        """Propose an ensemble of multiple models."""
        if self.task_type == "regression":
            # Create simple ensemble of 3 fast models
            estimators = [
                ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ("histgbm", HistGradientBoostingRegressor(max_iter=100, random_state=42)),
                ("linear", LinearRegression())
            ]
            model = VotingRegressor(estimators=estimators, n_jobs=-1)
            description = "VotingRegressor(RF+HistGBM+Linear)"
        else:
            estimators = [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                ("histgbm", HistGradientBoostingClassifier(max_iter=100, random_state=42)),
                ("logistic", LogisticRegression(random_state=42, max_iter=1000))
            ]
            model = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
            description = "VotingClassifier(RF+HistGBM+Logistic)"
        
        return Proposal(
            description=description,
            rationale="Ensemble of diverse models",
            model=model,
            preprocessor=StandardScaler(),  # Ensembles often benefit from scaling
        )
        """Randomly decide whether to add preprocessing."""
        preprocessing_choice = random.choice([None, "standard", "minmax", "poly", "pca"])
        
        if preprocessing_choice is None:
            return None, ""
        elif preprocessing_choice == "standard":
            return StandardScaler(), "StandardScaler"
        elif preprocessing_choice == "minmax":
            return MinMaxScaler(), "MinMaxScaler"
        elif preprocessing_choice == "poly":
            degree = random.choice([2, 3])
            return PolynomialFeatures(degree=degree, include_bias=False), f"PolyFeatures(degree={degree})"
        elif preprocessing_choice == "pca":
            n_components = random.choice([0.8, 0.9, 0.95])
            return PCA(n_components=n_components, random_state=42), f"PCA(n_comp={n_components})"

class LLMAgent(Agent):
    """Agent that uses an LLM to propose improvements (requires API key)."""
    
    def propose(self, state: dict) -> Proposal:
        # TODO: Call LLM API to propose improvements
        raise NotImplementedError("LLM agent requires groq or openai API key. Install with: pip install orcetra[llm]")