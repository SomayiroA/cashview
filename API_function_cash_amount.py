# from fastapi import FastAPI, HTTPException, Query, Depends, status
# from fastapi.security import OAuth2PasswordBearer
# from pydantic import BaseModel, Field
# from typing import Optional, List, Dict
# import pandas as pd
# import holidays
# from pathlib import Path
# from xgboost import XGBRegressor
# from sklearn.ensemble import RandomForestRegressor, StackingRegressor
# from sklearn.linear_model import Ridge, LinearRegression
# from sklearn.model_selection import RandomizedSearchCV, train_test_split
# from scipy.stats import loguniform
# import numpy as np
# import warnings
# from datetime import datetime
# import uuid

# warnings.filterwarnings("ignore")

# app = FastAPI()

# # --- Authentication and Role Management (Stub) ---

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Simple in-memory user store for demonstration
# users_db = {
#     "atm_ops": {"username": "atm_ops", "role": "ATM Operations Staff"},
#     "branch_mgr": {"username": "branch_mgr", "role": "Branch Operations Manager"},
#     "vault_mgr": {"username": "vault_mgr", "role": "Vault Manager"},
#     "hoao": {"username": "hoao", "role": "Head Office Authorization Officer"},
# }

# def get_current_user(token: str = Depends(oauth2_scheme)):
#     # In real app, decode token and verify user
#     user = users_db.get(token)
#     if not user:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
#     return user

# def role_required(allowed_roles: List[str]):
#     def role_checker(user: dict = Depends(get_current_user)):
#         if user["role"] not in allowed_roles:
#             raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Operation not permitted for your role")
#         return user
#     return role_checker

# # --- Machine Learning Model and Metadata ---

# trained_model = None
# model_metadata = {
#     "max_refill": None,
#     "column_names": None
# }

# # --- Refill Request Data Model and Store ---

# class ApprovalRecord(BaseModel):
#     approver: str
#     role: str
#     action: str  # "approved" or "refused"
#     timestamp: datetime
#     comment: Optional[str] = None

# class RefillRequest(BaseModel):
#     request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
#     atm_id: str
#     requested_amount: float
#     requestor: str
#     status: str = "Pending"  # Pending, Approved, Refused
#     created_at: datetime = Field(default_factory=datetime.utcnow)
#     updated_at: datetime = Field(default_factory=datetime.utcnow)
#     approval_history: List[ApprovalRecord] = []

# # In-memory store for refill requests
# refill_requests: Dict[str, RefillRequest] = {}

# # --- Existing Model Training Endpoint ---

# @app.get("/train-model/")
# async def train_model(
#     csv_path: str = Query(..., description="Path to CSV file (use forward slashes or double backslashes)")
# ):
#     global trained_model, model_metadata
    
#     try:
#         clean_path = csv_path.strip('\"\'')
#         if not clean_path.lower().endswith('.csv'):
#             clean_path += '.csv'
        
#         path = Path(clean_path)
#         if not path.exists():
#             raise HTTPException(status_code=400, detail="File not found")
        
#         df = pd.read_csv(path)
#         if df.empty:
#             raise HTTPException(status_code=400, detail="CSV file is empty")
        
#         headers = df.columns.tolist()
#         if len(headers) < 3:
#             raise HTTPException(status_code=400, detail="CSV needs at least 3 columns")
        
#         date_col, amount_col, balance_col = headers[0], headers[1], headers[2]
#         model_metadata["column_names"] = {
#             "date": date_col,
#             "amount": amount_col,
#             "balance": balance_col
#         }
        
#         df[date_col] = pd.to_datetime(df[date_col])
#         df["date"] = df[date_col].dt.date
#         df["day"] = df[date_col].dt.day
#         df["dayofweek"] = df[date_col].dt.dayofweek
#         df["month"] = df[date_col].dt.month
        
#         df2 = df[[amount_col, balance_col, "date", "day", "dayofweek", "month"]].groupby(
#             ["date", "day", "dayofweek", "month"]
#         ).agg({amount_col: "sum", balance_col: "min"}).reset_index()
        
#         df2["Month_end"] = df2["day"].apply(lambda x: 1 if x > 27 or x < 3 else 0)
#         df2["is_weekend"] = df2["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
#         df3 = df2
        
#         NGN_holidays = holidays.Nigeria()
#         df3['is_holiday'] = df3['date'].apply(lambda x: 1 if x in NGN_holidays else 0)
#         df3 = df3.drop(columns=["date", "day"])
        
#         # Refill detection
#         df3['is_refill'] = df3[balance_col] > df3[balance_col].shift(1)
#         df3['refill_group'] = df3['is_refill'].cumsum()
#         grouped = df3.groupby('refill_group').agg({
#             amount_col: "sum", 
#             "Month_end": "sum", 
#             "is_weekend": "sum", 
#             "is_holiday": "sum",
#             "dayofweek": "sum"
#         }).reset_index()
        
#         model_metadata["max_refill"] = grouped[amount_col].max()
        
#         df4 = df3[df3[balance_col] != 0]
#         df5 = df4[[amount_col, "Month_end", "is_weekend", "is_holiday", "month", "dayofweek"]]
#         X = df5[["month", "Month_end", "is_weekend", "is_holiday", "dayofweek"]]
#         y = df5[amount_col]
#         x_train, x_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.25, shuffle=True, random_state=42
#         )
        
#         xgb_param_grid = {
#             'n_estimators': [100, 200, 300, 400, 500],
#             'max_depth': [3, 5, 7, 9],
#             'learning_rate': [0.01, 0.05, 0.1, 0.2],
#             'subsample': [0.6, 0.8, 1.0],
#             'colsample_bytree': [0.6, 0.8, 1.0],
#             'gamma': [0, 0.1, 0.2],
#             'min_child_weight': [1, 3, 5]
#         }
#         xgb = XGBRegressor(random_state=42)
#         xgb_search = RandomizedSearchCV(
#             xgb, xgb_param_grid, n_iter=50, scoring='neg_mean_squared_error',
#             cv=5, verbose=1, random_state=42, n_jobs=-1
#         )
#         xgb_search.fit(x_train, y_train)
#         best_xgb = xgb_search.best_estimator_
        
#         rf_param_grid = {
#             'n_estimators': [100, 200, 300, 400, 500],
#             'max_depth': [None, 5, 10, 20, 30],
#             'min_samples_split': [2, 5, 10],
#             'min_samples_leaf': [1, 2, 4],
#             'max_features': ['auto', 'sqrt', 'log2'],
#             'bootstrap': [True, False]
#         }
#         rf = RandomForestRegressor(random_state=42)
#         rf_search = RandomizedSearchCV(
#             rf, rf_param_grid, n_iter=50, scoring='neg_mean_squared_error',
#             cv=5, verbose=1, random_state=42, n_jobs=-1
#         )
#         rf_search.fit(x_train, y_train)
#         best_rf = rf_search.best_estimator_
        
#         ridge_param_grid = {
#             'alpha': loguniform(1e-4, 100),
#             'fit_intercept': [True, False],
#             'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
#             'max_iter': [100, 500, 1000, 2000],
#             'tol': [1e-4, 1e-3, 1e-2]
#         }
#         ridge = Ridge(random_state=42)
#         ridge_search = RandomizedSearchCV(
#             ridge, ridge_param_grid, n_iter=50, scoring='neg_mean_squared_error',
#             cv=5, verbose=1, random_state=42, n_jobs=-1
#         )
#         ridge_search.fit(x_train, y_train)
#         best_ridge = ridge_search.best_estimator_
        
#         base_learners = [
#             ("xgb", best_xgb),
#             ("rf", best_rf),
#             ("ridge", best_ridge)
#         ]
#         stacking = StackingRegressor(
#             estimators=base_learners, 
#             final_estimator=LinearRegression()
#         )
#         stacking.fit(X, y)
#         trained_model = stacking
        
#         return {
#             "message": "Model trained successfully",
#             "max_refill_amount": model_metadata["max_refill"],
#             "columns_used": model_metadata["column_names"]
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

# # --- Existing Prediction Endpoint ---

# @app.get("/predict-refill/")
# async def predict_refill(
#     current_date: str = Query(..., description="Current date in YYYY-MM-DD format"),
#     days: int = Query(..., description="Number of days to predict"),
#     user: dict = Depends(role_required(["ATM Operations Staff", "Branch Operations Manager", "Vault Manager", "Head Office Authorization Officer"]))
# ):
#     global trained_model, model_metadata
    
#     if not trained_model:
#         raise HTTPException(status_code=400, detail="Model not trained. Please train first.")
    
#     try:
#         current_date = pd.to_datetime(current_date)
#         date_range = pd.date_range(start=current_date, periods=365)
#         pred_df = pd.DataFrame({'date': date_range})
        
#         pred_df['day'] = pred_df['date'].dt.day
#         pred_df['month'] = pred_df['date'].dt.month
#         pred_df['dayofweek'] = pred_df['date'].dt.dayofweek
#         pred_df['Month_end'] = pred_df['day'].apply(lambda x: 1 if x > 27 or x < 3 else 0)
#         pred_df['is_weekend'] = pred_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
        
#         NGN_holidays = holidays.Nigeria()
#         pred_df['is_holiday'] = pred_df['date'].apply(lambda x: 1 if x in NGN_holidays else 0)
#         pred_df = pred_df.drop(columns=["date", "day"])
        
#         predictions = []
#         total = 0
#         for day in range(days):
#             daily_pred = float(trained_model.predict(pred_df.iloc[[day]][["month", "Month_end", "is_weekend", "is_holiday", "dayofweek"]])[0])
#             total += daily_pred
#             predictions.append({
#                 "day_number": day + 1,
#                 "date": date_range[day].strftime('%Y-%m-%d'),
#                 "predicted_withdrawal": daily_pred,
#                 "running_total": total
#             })
        
#         return {
#             "total_predicted_amount": total,
#             "message": f"{total} would be sufficient for {days} days",
#             "daily_predictions": predictions,
#             "prediction_date": current_date.strftime('%Y-%m-%d'),
#             "days_requested": days
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# # --- New Refill Request Endpoints ---

# class RefillRequestCreate(BaseModel):
#     atm_id: str
#     requested_amount: float
#     comment: Optional[str] = None

# @app.post("/refill-requests/", status_code=201)
# async def create_refill_request(
#     request_data: RefillRequestCreate,
#     user: dict = Depends(role_required(["ATM Operations Staff"]))
# ):
#     new_request = RefillRequest(
#         atm_id=request_data.atm_id,
#         requested_amount=request_data.requested_amount,
#         requestor=user["username"],
#         approval_history=[]
#     )
#     if request_data.comment:
#         new_request.approval_history.append(
#             ApprovalRecord(
#                 approver=user["username"],
#                 role=user["role"],
#                 action="requested",
#                 timestamp=datetime.utcnow(),
#                 comment=request_data.comment
#             )
#         )
#     refill_requests[new_request.request_id] = new_request
#     return {"message": "Refill request created", "request_id": new_request.request_id}

# @app.get("/refill-requests/", response_model=List[RefillRequest])
# async def list_refill_requests(
#     status_filter: Optional[str] = Query(None, description="Filter by status: Pending, Approved, Refused"),
#     user: dict = Depends(get_current_user)
# ):
#     # Role-based filtering
#     if user["role"] == "ATM Operations Staff":
#         # Show only requests created by this user
#         filtered = [r for r in refill_requests.values() if r.requestor == user["username"]]
#     elif user["role"] == "Branch Operations Manager":
#         # Show all pending requests for approval
#         filtered = [r for r in refill_requests.values() if r.status == "Pending"]
#     elif user["role"] == "Vault Manager":
#         # Show only approved requests
#         filtered = [r for r in refill_requests.values() if r.status == "Approved"]
#     elif user["role"] == "Head Office Authorization Officer":
#         # Show all requests
#         filtered = list(refill_requests.values())
#     else:
#         filtered = []
    
#     if status_filter:
#         filtered = [r for r in filtered if r.status == status_filter]
    
#     return filtered

# class RefillRequestAction(BaseModel):
#     action: str  # "approve" or "refuse"
#     comment: Optional[str] = None

# @app.post("/refill-requests/{request_id}/action")
# async def take_action_on_refill_request(
#     request_id: str,
#     action_data: RefillRequestAction,
#     user: dict = Depends(role_required(["Branch Operations Manager", "Head Office Authorization Officer"]))
# ):
#     refill_request = refill_requests.get(request_id)
#     if not refill_request:
#         raise HTTPException(status_code=404, detail="Refill request not found")
#     if refill_request.status != "Pending":
#         raise HTTPException(status_code=400, detail="Refill request already processed")
    
#     if action_data.action.lower() not in ["approve", "refuse"]:
#         raise HTTPException(status_code=400, detail="Invalid action")
    
#     if action_data.action.lower() == "approve":
#         refill_request.status = "Approved"
#     else:
#         refill_request.status = "Refused"
    
#     refill_request.updated_at = datetime.utcnow()
#     refill_request.approval_history.append(
#         ApprovalRecord(
#             approver=user["username"],
#             role=user["role"],
#             action=refill_request.status.lower(),
#             timestamp=refill_request.updated_at,
#             comment=action_data.comment
#         )
#     )
#     return {"message": f"Refill request {refill_request.status.lower()}"}

# # --- Audit Trail Endpoint ---

# @app.get("/refill-requests/{request_id}/audit", response_model=List[ApprovalRecord])
# async def get_refill_request_audit(
#     request_id: str,
#     user: dict = Depends(get_current_user)
# ):
#     refill_request = refill_requests.get(request_id)
#     if not refill_request:
#         raise HTTPException(status_code=404, detail="Refill request not found")
#     # Only allow involved users or managers to view audit
#     if user["role"] not in ["Branch Operations Manager", "Head Office Authorization Officer", "Vault Manager"] and user["username"] != refill_request.requestor:
#         raise HTTPException(status_code=403, detail="Not authorized to view audit trail")
#     return refill_request.approval_history

# # --- Token Endpoint for Authentication Stub ---

# from fastapi.security import OAuth2PasswordRequestForm

# @app.post("/token")
# async def login(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = users_db.get(form_data.username)
#     if not user or form_data.password != "password":  # Simplified password check
#         raise HTTPException(status_code=400, detail="Incorrect username or password")
#     # Return token as username for simplicity
#     return {"access_token": user["username"], "token_type": "bearer"}

# # --- Main ---

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)
