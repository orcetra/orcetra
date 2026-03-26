"""Model registry — maps task types to available models."""
from .baseline import (
    # Regression
    linear_regression, ridge_regression,
    random_forest_regression, extra_trees_regression,
    gradient_boosting_regression, hist_gradient_boosting_regression,
    xgb_regression, lgbm_regression,
    # Classification
    logistic_regression,
    random_forest_classification, extra_trees_classification,
    gradient_boosting_classification, hist_gradient_boosting_classification,
    xgb_classification, lgbm_classification,
)

def get_baselines(task_type: str) -> dict:
    if task_type == "regression":
        return {
            "LinearRegression": linear_regression,
            "Ridge": ridge_regression,
            "RandomForest": random_forest_regression,
            "ExtraTrees": extra_trees_regression,
            "GradientBoosting": gradient_boosting_regression,
            "HistGradientBoosting": hist_gradient_boosting_regression,
            "XGBoost": xgb_regression,
            "LightGBM": lgbm_regression,
        }
    else:
        return {
            "LogisticRegression": logistic_regression,
            "RandomForest": random_forest_classification,
            "ExtraTrees": extra_trees_classification,
            "GradientBoosting": gradient_boosting_classification,
            "HistGradientBoosting": hist_gradient_boosting_classification,
            "XGBoost": xgb_classification,
            "LightGBM": lgbm_classification,
        }
