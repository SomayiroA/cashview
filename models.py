from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ApprovalRecord(BaseModel):
    approver: str
    role: str
    action: str  # "approved" or "refused" or "requested"
    timestamp: datetime
    comment: Optional[str] = None

class RefillRequest(BaseModel):
    request_id: str = Field(default_factory=str)
    atm_id: str
    requested_amount: float
    requestor: str
    status: str = "Pending"  # Pending, Approved, Refused
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    approval_history: List[ApprovalRecord] = []
