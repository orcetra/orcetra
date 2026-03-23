"""Train/validation/test split utilities."""
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Tuple

def split_data(X, y, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> Tuple:
    """Split data into train/val/test sets."""
    # First split: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train / val
    val_ratio = val_size / (1 - test_size)  # Adjust val_size relative to temp set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test