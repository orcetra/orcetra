"""Auto-analyze and load datasets."""
import pandas as pd
import numpy as np
from typing import Dict, Any

def analyze_and_load(data_path: str, target: str) -> Dict[str, Any]:
    """
    Load a CSV and automatically determine task type, features, etc.
    """
    df = pd.read_csv(data_path)
    
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)}")
    
    y = df[target]
    X = df.drop(columns=[target])
    
    # Auto-detect task type
    if y.dtype in ["object", "bool"] or y.nunique() <= 20:
        task_type = "classification"
    else:
        task_type = "regression"
    
    # Handle categorical features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Simple preprocessing: fill NaN, encode categoricals
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for col in cat_cols:
        X[col] = X[col].fillna("_missing_")
        X[col] = X[col].astype("category").cat.codes
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "shape": df.shape,
        "task_type": task_type,
        "n_features": X.shape[1],
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target_name": target,
    }