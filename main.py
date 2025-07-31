from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Optional, List
import warnings
from fastapi import APIRouter
from pydantic import BaseModel

from database import get_db
from sqlalchemy.orm import Session
from models_db import RefillRequestDB, ApprovalRecordDB, RequestStatus
from auth import get_current_user, role_required, users_db
from services import create_refill_request, list_refill_requests, take_action_on_refill_request
from model_service import ModelService

warnings.filterwarnings("ignore")

app = FastAPI()

model_service = ModelService()

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or form_data.password != "password":  # Simplified password check
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    return {"access_token": user["username"], "token_type": "bearer"}

@app.get("/train-model/")
async def train_model(
    csv_path: str = Query(..., description="Path to CSV file (use forward slashes or double backslashes)")
):
    try:
        result = model_service.train_model(csv_path)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File not found")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/predict-refill/")
async def predict_refill(
    current_date: str = Query(..., description="Current date in YYYY-MM-DD format"),
    days: int = Query(..., description="Number of days to predict"),
    user: dict = Depends(role_required(["ATM Operations Staff", "Branch Operations Manager", "Vault Manager", "Head Office Authorization Officer"]))
):
    if not model_service.trained_model:
        raise HTTPException(status_code=400, detail="Model not trained. Please train first.")
    try:
        result = model_service.predict_refill(current_date, days)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

refill_router = APIRouter()

class RefillRequestCreate(BaseModel):
    atm_id: str
    requested_amount: float
    comment: Optional[str] = None

@refill_router.post("/refill-requests/", status_code=201)
async def create_refill_request_endpoint(
    request_data: RefillRequestCreate,
    user: dict = Depends(role_required(["ATM Operations Staff"])),
    db: Session = Depends(get_db)
):
    new_request = create_refill_request(
        db=db,
        atm_id=request_data.atm_id,
        requested_amount=request_data.requested_amount,
        requestor=user["username"],
        comment=request_data.comment
    )
    return {"message": "Refill request created", "request_id": new_request.request_id}

@refill_router.get("/refill-requests/", response_model=List[RefillRequest])
async def list_refill_requests_endpoint(
    status_filter: Optional[str] = Query(None, description="Filter by status: Pending, Approved, Refused"),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    filtered = list_refill_requests(db=db, user_role=user["role"], username=user["username"], status_filter=status_filter)
    return filtered

class RefillRequestAction(BaseModel):
    action: str  # "approve" or "refuse"
    comment: Optional[str] = None

@refill_router.post("/refill-requests/{request_id}/action")
async def take_action_on_refill_request_endpoint(
    request_id: str,
    action_data: RefillRequestAction,
    user: dict = Depends(role_required(["Branch Operations Manager", "Head Office Authorization Officer"])),
    db: Session = Depends(get_db)
):
    try:
        updated_request = take_action_on_refill_request(
            db=db,
            request_id=request_id,
            action=action_data.action,
            approver=user["username"],
            role=user["role"],
            comment=action_data.comment
        )
        return {"message": f"Refill request {updated_request.status.lower()}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@refill_router.get("/refill-requests/{request_id}/audit", response_model=List[ApprovalRecord])
async def get_refill_request_audit_endpoint(
    request_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    refill_request = next((r for r in list_refill_requests(db=db, user_role=user["role"], username=user["username"]) if r.request_id == request_id), None)
    if not refill_request:
        raise HTTPException(status_code=404, detail="Refill request not found")
    if user["role"] not in ["Branch Operations Manager", "Head Office Authorization Officer", "Vault Manager"] and user["username"] != refill_request.requestor:
        raise HTTPException(status_code=403, detail="Not authorized to view audit trail")
    return refill_request.approval_history

app.include_router(refill_router)
