import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config.settings import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ENGINEERED_FEATURES
from preprocessing.feature_engineering import add_engineered_features
from utils.logger import get_logger

logger = get_logger("DataCleaner")

class StudentDataPreprocessor(BaseEstimator, TransformerMixin):
    """Production data cleaning, imputation, outlier handling, feature engineering, and encoding pipeline."""
    
    def __init__(self, handle_outliers: bool = True):
        self.handle_outliers = handle_outliers
        self.column_transformer = None
        self.feature_names_out = []
        self.all_numeric_cols = NUMERICAL_FEATURES + ENGINEERED_FEATURES
        self.all_cat_cols = CATEGORICAL_FEATURES
        
    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caps extreme outliers in numerical columns using the IQR method."""
        df_capped = df.copy()
        for col in NUMERICAL_FEATURES:
            if col in df_capped.columns and pd.api.types.is_numeric_dtype(df_capped[col]):
                q1 = df_capped[col].quantile(0.25)
                q3 = df_capped[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 2.5 * iqr
                upper_bound = q3 + 2.5 * iqr
                df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
        return df_capped

    def fit(self, X: pd.DataFrame, y=None):
        """Fits the underlying imputer, scaler, and categorical encoders."""
        df = X.copy()
        
        # 1. Clean duplicates
        df = df.drop_duplicates()
        
        # 2. Add engineered features
        df = add_engineered_features(df)
        
        # 3. Handle outliers if configured
        if self.handle_outliers:
            df = self._cap_outliers(df)
            
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.all_numeric_cols),
                ("cat", cat_pipeline, self.all_cat_cols)
            ]
        )
        
        self.column_transformer.fit(df)
        
        # Extract feature names after encoding
        cat_encoder = self.column_transformer.named_transformers_["cat"].named_steps["encoder"]
        cat_encoded_cols = list(cat_encoder.get_feature_names_out(self.all_cat_cols))
        self.feature_names_out = self.all_numeric_cols + cat_encoded_cols
        
        logger.info(f"Preprocessor successfully fitted. Total transformed features: {len(self.feature_names_out)}")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame using fitted pipelines."""
        if self.column_transformer is None:
            raise ValueError("Preprocessor has not been fitted yet. Call fit or fit_transform first.")
            
        df = X.copy()
        df = add_engineered_features(df)
        if self.handle_outliers:
            df = self._cap_outliers(df)
            
        transformed = self.column_transformer.transform(df)
        return transformed

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> list:
        return self.feature_names_out
