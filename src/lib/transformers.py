from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].clip(
                lower=self.lower_bounds_[col], 
                upper=self.upper_bounds_[col]
            )
            
        return X_transformed
