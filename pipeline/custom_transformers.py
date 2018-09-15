from sklearn.base import TransformerMixin
import pandas as pd

class NAEncoder(TransformerMixin):
    """encodes all NA values as 0, non-NA-s as 1"""

    def __init__(self, columns = []):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].apply(lambda x: int(pd.notna(x)))
        return X
    
class ColumnDropper(TransformerMixin):
    """Expects a pandas dataframe, drops the columns specified"""

    def __init__(self, columns = []):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.drop(columns=self.columns)
