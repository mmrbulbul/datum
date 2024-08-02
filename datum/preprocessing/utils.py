import numpy as np
import pandas as pd


def analyze_column(input_series: pd.Series) -> str:
    """
    Check whether a column is numerical or categorical
    """
    if pd.api.types.is_numeric_dtype(input_series):
        return 'numerical'
    else:
        return 'categorical'

def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Get the categorical columns
    """
    return [col for col in df.columns if analyze_column(df[col]) == 'categorical']


def get_numerical_columns(df: pd.DataFrame) -> list:
    """
    Get the numerical columns
    """
    return [col for col in df.columns if analyze_column(df[col]) == 'numerical']