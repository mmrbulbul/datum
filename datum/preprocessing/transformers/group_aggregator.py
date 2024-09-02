import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GroupAggEncoder(BaseEstimator, TransformerMixin):
    """
    ref: https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575#643288
    """
    def __init__(self, group, columns, agg=np.mean, replace_na=-1, verbose=False):
        self.group = group if type(group) is list else [group]
        self.columns = columns if type(columns) is list else[columns]
        self.agg = agg if type(agg) in [list, dict] else [agg]
        if type(self.agg) is not dict:
            self.agg = {a.__name__: a for a in self.agg}
        self.agg_encode_map = {}
        self.replace_na = replace_na
        self.verbose = verbose

    def fit(self, df):
        for column in self.columns:
            encode_df = df[self.group + [column]].groupby(self.group)[column].agg(list(self.agg.values()))
            encode_column_names = ['_'.join(self.group) + '_' + column + '_' + agg_name for agg_name in self.agg.keys()]
            encode_df.columns = encode_column_names
            self.agg_encode_map[column] = encode_df
            if self.verbose: print(f'{column} fit processed {encode_df.shape}')
        return self
    
    def transform(self, df):
        result_df = df[self.group].set_index(self.group)
        for column in self.columns:
            encode_df = self.agg_encode_map[column]
            for encode_col in encode_df.columns:
                result_df[encode_col] = result_df.index.map(encode_df[encode_col].to_dict())
            if self.verbose: print(f'{column} transformed')
        result_df = result_df.fillna(self.replace_na)
        result_df.index = df.index
        return result_df