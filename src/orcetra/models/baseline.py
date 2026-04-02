"""Standard baseline models — expanded pool for competitive AutoML."""
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier,
)
from sklearn.preprocessing import StandardScaler

# Optional: XGBoost / LightGBM
try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


def _safe_fit_predict(model, data_info, metric_fn, needs_scaling=False):
    """Fit model and return metric score. Handles scaling if needed."""
    X_train = data_info["X_train"]
    X_test = data_info["X_test"]
    
    if needs_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, data_info["y_train"])
        preds = model.predict(X_test)
    return metric_fn.compute(data_info["y_test"], preds)


# ── Regression baselines ──────────────────────────────────────────────

def linear_regression(data_info, metric_fn):
    return _safe_fit_predict(LinearRegression(), data_info, metric_fn)

def ridge_regression(data_info, metric_fn):
    return _safe_fit_predict(Ridge(alpha=1.0, random_state=42), data_info, metric_fn, needs_scaling=True)

def random_forest_regression(data_info, metric_fn):
    return _safe_fit_predict(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        data_info, metric_fn)

def extra_trees_regression(data_info, metric_fn):
    return _safe_fit_predict(
        ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        data_info, metric_fn)

def gradient_boosting_regression(data_info, metric_fn):
    return _safe_fit_predict(
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        data_info, metric_fn)

def hist_gradient_boosting_regression(data_info, metric_fn):
    return _safe_fit_predict(
        HistGradientBoostingRegressor(max_iter=200, random_state=42),
        data_info, metric_fn)

def xgb_regression(data_info, metric_fn):
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    return _safe_fit_predict(
        XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
        data_info, metric_fn)

def lgbm_regression(data_info, metric_fn):
    if not _HAS_LGBM:
        raise ImportError("lightgbm not installed")
    return _safe_fit_predict(
        LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1),
        data_info, metric_fn)

def catboost_regression(data_info, metric_fn):
    if not _HAS_CATBOOST:
        raise ImportError("catboost not installed")
    return _safe_fit_predict(
        CatBoostRegressor(iterations=100, random_state=42, verbose=False),
        data_info, metric_fn)


# ── Classification baselines ──────────────────────────────────────────

def logistic_regression(data_info, metric_fn):
    return _safe_fit_predict(
        LogisticRegression(max_iter=1000, random_state=42),
        data_info, metric_fn, needs_scaling=True)

def random_forest_classification(data_info, metric_fn):
    return _safe_fit_predict(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        data_info, metric_fn)

def extra_trees_classification(data_info, metric_fn):
    return _safe_fit_predict(
        ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        data_info, metric_fn)

def gradient_boosting_classification(data_info, metric_fn):
    return _safe_fit_predict(
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        data_info, metric_fn)

def hist_gradient_boosting_classification(data_info, metric_fn):
    return _safe_fit_predict(
        HistGradientBoostingClassifier(max_iter=200, random_state=42),
        data_info, metric_fn)

def xgb_classification(data_info, metric_fn):
    if not _HAS_XGB:
        raise ImportError("xgboost not installed")
    return _safe_fit_predict(
        XGBClassifier(n_estimators=100, random_state=42, verbosity=0, 
                      use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
        data_info, metric_fn)

def lgbm_classification(data_info, metric_fn):
    if not _HAS_LGBM:
        raise ImportError("lightgbm not installed")
    return _safe_fit_predict(
        LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1, n_jobs=-1),
        data_info, metric_fn)

def catboost_classification(data_info, metric_fn):
    if not _HAS_CATBOOST:
        raise ImportError("catboost not installed")
    return _safe_fit_predict(
        CatBoostClassifier(iterations=100, random_state=42, verbose=False),
        data_info, metric_fn)
