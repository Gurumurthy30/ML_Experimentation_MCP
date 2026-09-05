import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.compose import ColumnTransformer

def split_dataset(dataset_id: str, target_column: str, split_strategy='auto', 
                group_column=None, time_column=None, test_size=0.2, task_type='classification') -> dict:
    """Writes _cache/splits/{split_id}/{train,test}.parquet. split_id is a
    hash of (dataset_id, target_column, test_size, strategy, random_state)
    -- same inputs always resolve to the same split_id (idempotent)."""
    