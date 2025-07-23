from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import holidays
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import numpy as np
import datetime
import warnings
import joblib
from typing import Optional

warnings.filterwarnings("ignore")

app = FastAPI()

trained_model = None
MAX = None
columns_map = {}

@app.get("/train-model/")
async def train_model(csv_path: str = Query(..., description="Path to CSV file (can use forward slashes or double backslashes)")):
    global trained_model, MAX, columns_map
    
    try:
        clean_path = csv_path.strip('\"\'')
        if not clean_path.lower().endswith('.csv'):
            clean_path += '.csv'
        
        path = Path(clean_path)
        if not path.exists():
            raise HTTPException(status_code=400, detail="File not found at the specified path")
        
        df = pd.read_csv(path)
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        headers = df.columns.tolist()
        if len(headers) < 3:
            raise HTTPException(status_code=400, detail="CSV needs at least 3 columns")
        
        columns_map = {
            'date_col': headers[0],
            'amount_col': headers[1],
            'balance_col': headers[2]
        }
        
        a, b, c = headers[0], headers[1], headers[2]
        
        df[a] = pd.to_datetime(df[a], errors='coerce')
        if df[a].isnull().any():
            raise HTTPException(status_code=400, detail="Invalid date format in first column")
            
        df['year'] = df[a].dt.year
        df['month'] = df[a].dt.month
        df['day'] = df[a].dt.day
        df['date'] = df[a].dt.date
        df['dayofweek'] = df[a].dt.dayofweek
        df = df.drop(columns=a)
        
        df2 = df.groupby(["year", "month", "day", "dayofweek", "date"]).agg({b: "sum", c: "min"}).reset_index()
        df2["Month_end"] = df2["day"].apply(lambda x: True if x > 27 or x < 3 else False)
        df2["is_weekend"] = df2["dayofweek"].apply(lambda x: True if x >= 5 else False)
        df3 = df2.drop(columns="dayofweek")
        
        NGN_holidays = holidays.Nigeria()
        df3['is_holiday'] = df3['date'].apply(lambda x: x in NGN_holidays)
        df3 = df3.replace(True, 1).replace(False, 0)
        df3["Day_number"] = df3["year"].apply(lambda x: 1 if x > 0 else 0)
        df4 = df3.drop(columns=["year", "month", "day", "date"])
        
        df4['is_refill'] = df4[c] > df4[c].shift(1)
        df4['refill_group'] = df4['is_refill'].cumsum()
        grouped = df4.groupby('refill_group').agg({
            c: "last", 
            b: "sum", 
            "Month_end": "sum", 
            "is_weekend": "sum", 
            "is_holiday": "sum", 
            "Day_number": "sum"
        }).reset_index()
        
        MAX = grouped[b].max()
        X = grouped[[b, "Month_end", "is_weekend", "is_holiday"]]
        y = grouped["Day_number"]
        
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
        
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2],
            'min_child_weight': [1, 3, 5]
        }
        
        xgb = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_grid,
            n_iter=50,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        random_search.fit(x_train, y_train)
        best_xgb = random_search.best_estimator_
        
        RF = RandomForestRegressor()
        ridge = Ridge()
        base_learners = [
            ("xgb", best_xgb),
            ("Rf", RF),
            ("Ridge", ridge)
        ]
        
        stacking = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
        stacking.fit(X, y)
        trained_model = stacking
        
        return {
            "message": "Model trained successfully",
            "max_daily_withdrawal": MAX,
            "columns_used": columns_map
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/predict/")
async def predict(
    current_date: str = Query(..., description="Current date in YYYY-MM-DD format"),
    current_amount: float = Query(..., description="Current cash amount in ATM")
):
    global trained_model, MAX, columns_map
    
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained. Please train the model first.")
    
    try:
        current_date = pd.to_datetime(current_date)
        if pd.isnull(current_date):
            raise ValueError("Invalid date format")
            
        current_amount = float(current_amount)
        if not MAX or MAX <= 0:
            raise ValueError("Invalid MAX value")
            
        future_days = min(30, max(5, int(current_amount/(MAX/10))))
        
        date_range = pd.date_range(start=current_date, periods=future_days)
        pred_df = pd.DataFrame({'date': date_range})
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        amount_col = columns_map.get('amount_col', 'amount')
        pred_df[amount_col] = current_amount
        
        pred_df['day'] = pred_df['date'].dt.day
        pred_df['dayofweek'] = pred_df['date'].dt.dayofweek
        pred_df['Month_end'] = pred_df['day'].apply(lambda x: 1 if x > 27 or x < 3 else 0)
        pred_df['is_weekend'] = pred_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
        NGN_holidays = holidays.Nigeria()
        pred_df['is_holiday'] = pred_df['date'].apply(lambda x: 1 if x in NGN_holidays else 0)
        pred_df = pred_df.drop(columns=["date", "day", "dayofweek"])
        
        X_pred = pred_df.groupby(amount_col).sum().reset_index()
        preds = trained_model.predict(X_pred)
        depletion_day = round(preds[0])
        
        return {
            "depletion_day": depletion_day,
            "message": f"ATM will run out of cash in approximately {depletion_day} days",
            "prediction_date": current_date.strftime('%Y-%m-%d'),
            "current_amount": current_amount
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)