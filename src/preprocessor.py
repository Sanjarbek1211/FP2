import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class PreprocessingBaseline:
    def __init__(self, threshold=2):
        self.threshold = threshold
        self.encoders = {}
        self.scalers = {}
        self.train_columns = None
        self.fill_values = {}

    # -------------------------------
    # 1. Missing value handling
    # -------------------------------
    def fit_fillna(self, X):
        for col in X.columns:
            if X[col].dtype in ["int64"]:
                self.fill_values[col] = X[col].mean()
            else:
                self.fill_values[col] = X[col].mode()[0]
        return self

    def transform_fillna(self, X):
        X = X.copy()
        for col, val in self.fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X

    # -------------------------------
    # 2. Encoding
    # -------------------------------
    def fit_encode(self, X):
        X = X.copy()

        for col in X.columns:
            if X[col].dtype == "object":
                if X[col].nunique() <= self.threshold:
                    dummies = pd.get_dummies(X[col], prefix=col, dtype=int)
                    X = pd.concat([X.drop(columns=col), dummies], axis=1)
                else:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    self.encoders[col] = le

        self.train_columns = X.columns
        return X

    def transform_encode(self, X):
        X = X.copy()

        # Label encoding
        for col, le in self.encoders.items():
            X[col] = X[col].astype(str)
            X[col] = X[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        # One-hot alignment
        X = pd.get_dummies(X)
        X = X.reindex(columns=self.train_columns, fill_value=0)

        return X

    # -------------------------------
    # 3. Scaling
    # -------------------------------
    def fit_scale(self, X):
        X = X.copy()
        for col in X.columns:
            scaler = StandardScaler()
            X[col] = scaler.fit_transform(X[[col]])
            self.scalers[col] = scaler
        return X

    def transform_scale(self, X):
        X = X.copy()
        for col, scaler in self.scalers.items():
            if col in X.columns:
                X[col] = scaler.transform(X[[col]])
        return X

    # -------------------------------
    # 4. Full pipeline
    # -------------------------------
    def fit_transform(self, X):
        X = self.fit_fillna(X).transform_fillna(X)
        X = self.fit_encode(X)
        X = self.fit_scale(X)
        return X

    def transform(self, X):
        X = self.transform_fillna(X)
        X = self.transform_encode(X)
        X = self.transform_scale(X)
        return X
