import pandas as pd
import holidays
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import loguniform
from datetime import datetime

class ModelService:
    def __init__(self):
        self.trained_model = None
        self.model_metadata = {
            "max_refill": None,
            "column_names": None
        }

    def train_model(self, csv_path: str):
        clean_path = csv_path.strip('\"\'')
        if not clean_path.lower().endswith('.csv'):
            clean_path += '.csv'
        path = Path(clean_path)
        if not path.exists():
            raise FileNotFoundError("File not found")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV file is empty")
        headers = df.columns.tolist()
        if len(headers) < 3:
            raise ValueError("CSV needs at least 3 columns")
        date_col, amount_col, balance_col = headers[0], headers[1], headers[2]
        self.model_metadata["column_names"] = {
            "date": date_col,
            "amount": amount_col,
            "balance": balance_col
        }
        df[date_col] = pd.to_datetime(df[date_col])
        df["date"] = df[date_col].dt.date
        df["day"] = df[date_col].dt.day
        df["dayofweek"] = df[date_col].dt.dayofweek
        df["month"] = df[date_col].dt.month
        df["year"] = df[date_col].dt.year
        df2 = df[[amount_col, balance_col, "date", "year", "day", "dayofweek", "month"]].groupby(
            ["date", "year", "month", "day", "dayofweek"]
        ).agg({amount_col: "sum", balance_col: "min"}).reset_index()
        df2["Month_end"] = df2["day"].apply(lambda x: 1 if x > 27 or x < 3 else 0)
        df2["is_weekend"] = df2["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
        df3 = df2
        NGN_holidays = holidays.Nigeria()
        df3['is_holiday'] = df3['date'].apply(lambda x: 1 if x in NGN_holidays else 0)
        df3 = df3.drop(columns=["date", "day"])
        df3['is_refill'] = df3[balance_col] > df3[balance_col].shift(1)
        df3['refill_group'] = df3['is_refill'].cumsum()
        grouped = df3.groupby('refill_group').agg({
            amount_col: "sum",
            "Month_end": "sum",
            "is_weekend": "sum",
            "is_holiday": "sum",
            "dayofweek": "sum",
            "year": "sum"
        }).reset_index()
        self.model_metadata["max_refill"] = grouped[amount_col].max()
        df4 = df3[df3[balance_col] != 0]
        df5 = df4[[amount_col, "Month_end", "is_weekend", "is_holiday", "month", "dayofweek", "year"]]
        X = df5[["month", "Month_end", "is_weekend", "is_holiday", "dayofweek", "year"]]
        y = df5[amount_col]
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42
        )
        xgb_param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        xgb = XGBRegressor(random_state=42)
        xgb_search = RandomizedSearchCV(
            xgb, xgb_param_grid, n_iter=50, scoring='neg_mean_squared_error',
            cv=5, verbose=1, random_state=42, n_jobs=-1
        )
        xgb_search.fit(x_train, y_train)
        best_xgb = xgb_search.best_estimator_
        rf_param_grid = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 5, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
        rf = RandomForestRegressor(random_state=42)
        rf_search = RandomizedSearchCV(
            rf, rf_param_grid, n_iter=50, scoring='neg_mean_squared_error',
            cv=5, verbose=1, random_state=42, n_jobs=-1
        )
        rf_search.fit(x_train, y_train)
        best_rf = rf_search.best_estimator_
        ridge_param_grid = {
            'alpha': loguniform(1e-4, 100),
            'fit_intercept': [True, False],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
            'max_iter': [100, 500, 1000, 2000],
            'tol': [1e-4, 1e-3, 1e-2]
        }
        ridge = Ridge(random_state=42)
        ridge_search = RandomizedSearchCV(
            ridge, ridge_param_grid, n_iter=50, scoring='neg_mean_squared_error',
            cv=5, verbose=1, random_state=42, n_jobs=-1
        )
        ridge_search.fit(x_train, y_train)
        best_ridge = ridge_search.best_estimator_
        base_learners = [
            ("xgb", best_xgb),
            ("rf", best_rf),
            ("ridge", best_ridge)
        ]
        stacking = StackingRegressor(
            estimators=base_learners,
            final_estimator=LinearRegression()
        )
        stacking.fit(X, y)
        self.trained_model = stacking
        return {
            "message": "Model trained successfully",
            "max_refill_amount": self.model_metadata["max_refill"],
            "columns_used": self.model_metadata["column_names"]
        }

    def predict_refill(self, current_date: str, days: int):
        if not self.trained_model:
            raise ValueError("Model not trained. Please train first.")
        current_date = pd.to_datetime(current_date)
        date_range = pd.date_range(start=current_date, periods=365)
        pred_df = pd.DataFrame({'date': date_range})
        pred_df['day'] = pred_df['date'].dt.day
        pred_df['month'] = pred_df['date'].dt.month
        pred_df['year'] = pred_df['date'].dt.year
        pred_df['dayofweek'] = pred_df['date'].dt.dayofweek
        pred_df['Month_end'] = pred_df['day'].apply(lambda x: 1 if x > 27 or x < 3 else 0)
        pred_df['is_weekend'] = pred_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        NGN_holidays = holidays.Nigeria()
        pred_df['is_holiday'] = pred_df['date'].apply(lambda x: 1 if x in NGN_holidays else 0)
        pred_df = pred_df.drop(columns=["date", "day"])
        predictions = []
        total = 0
        for day in range(days):
            daily_pred = float(self.trained_model.predict(pred_df.iloc[[day]][["month", "Month_end", "is_weekend", "is_holiday", "dayofweek", "year"]])[0])
            total += daily_pred
            predictions.append({
                "day_number": day + 1,
                "date": date_range[day].strftime('%Y-%m-%d'),
                "predicted_withdrawal": daily_pred,
                "running_total": total
            })
        return {
            "total_predicted_amount": total,
            "message": f"{total} would be sufficient for {days} days",
            "daily_predictions": predictions,
            "prediction_date": current_date.strftime('%Y-%m-%d'),
            "days_requested": days
        }
