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


def subset_by_iqr(df, column, whisker_width=1.5):
    """Remove outliers from a dataframe by column, including optional 
       whiskers, removing rows for which the column value are 
       less than Q1-1.5IQR or greater than Q3+1.5IQR.
    """
    # Calculate Q1, Q2 and IQR
    q1 = df[column].quantile(0.25)                 
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    # Apply filter with respect to IQR, including optional whiskers
    filter = (df[column] >= q1 - whisker_width*iqr) & (df[column] <= q3 + whisker_width*iqr)
    return df.loc[filter]                                                     
