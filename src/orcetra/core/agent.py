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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
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
    """Agent that randomly explores different models, hyperparameters, and preprocessing."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
        
    def propose(self, state: dict) -> Proposal:
        """Propose a random model configuration."""
        # Randomly choose model and hyperparameters
        if self.task_type == "regression":
            model, description = self._get_random_regression_model()
        else:  # classification
            model, description = self._get_random_classification_model()
        
        # Randomly decide whether to add preprocessing
        preprocessor, preprocess_desc = self._get_random_preprocessor()
        
        if preprocess_desc:
            description = f"{preprocess_desc} + {description}"
        
        return Proposal(
            description=description,
            rationale="Random search exploration",
            model=model,
            preprocessor=preprocessor,
        )
    
    def _get_random_regression_model(self):
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
        
        elif model_choice == "svr":
            C = random.choice([0.1, 1.0, 10.0, 100.0])
            epsilon = random.choice([0.01, 0.1, 0.2])
            return SVR(C=C, epsilon=epsilon), f"SVR(C={C}, eps={epsilon})"
    
    def _get_random_classification_model(self):
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
        
        elif model_choice == "svc":
            C = random.choice([0.1, 1.0, 10.0, 100.0])
            return SVC(C=C, random_state=42), f"SVC(C={C})"
    
    def _get_random_preprocessor(self):
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