"""Model registry — maps task types to available models."""
from .baseline import (
    linear_regression, random_forest_regression, gradient_boosting_regression,
    logistic_regression, random_forest_classification, gradient_boosting_classification,
)

def get_baselines(task_type: str) -> dict:
    if task_type == "regression":
        return {
            "LinearRegression": linear_regression,
            "RandomForest": random_forest_regression,
            "GradientBoosting": gradient_boosting_regression,
        }
    else:
        return {
            "LogisticRegression": logistic_regression,
            "RandomForest": random_forest_classification,
            "GradientBoosting": gradient_boosting_classification,
        }