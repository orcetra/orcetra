"""Standard baseline models."""
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

def linear_regression(data_info, metric_fn):
    model = LinearRegression()
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)

def random_forest_regression(data_info, metric_fn):
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)

def gradient_boosting_regression(data_info, metric_fn):
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)

def logistic_regression(data_info, metric_fn):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)

def random_forest_classification(data_info, metric_fn):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)

def gradient_boosting_classification(data_info, metric_fn):
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(data_info["X_train"], data_info["y_train"])
    preds = model.predict(data_info["X_test"])
    return metric_fn.compute(data_info["y_test"], preds)