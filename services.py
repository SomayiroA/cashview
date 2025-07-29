from typing import Dict, List
from datetime import datetime
from models import RefillRequest, ApprovalRecord
import uuid

# In-memory store for refill requests (to be replaced with DB later)
refill_requests: Dict[str, RefillRequest] = {}

def create_refill_request(atm_id: str, requested_amount: float, requestor: str, comment: str = None) -> RefillRequest:
    new_request = RefillRequest(
        request_id=str(uuid.uuid4()),
        atm_id=atm_id,
        requested_amount=requested_amount,
        requestor=requestor,
        approval_history=[]
    )
    if comment:
        new_request.approval_history.append(
            ApprovalRecord(
                approver=requestor,
                role="ATM Operations Staff",
                action="requested",
                timestamp=datetime.utcnow(),
                comment=comment
            )
        )
    refill_requests[new_request.request_id] = new_request
    return new_request

def list_refill_requests(user_role: str, username: str, status_filter: str = None) -> List[RefillRequest]:
    if user_role == "ATM Operations Staff":
        filtered = [r for r in refill_requests.values() if r.requestor == username]
    elif user_role == "Branch Operations Manager":
        filtered = [r for r in refill_requests.values() if r.status == "Pending"]
    elif user_role == "Vault Manager":
        filtered = [r for r in refill_requests.values() if r.status == "Approved"]
    elif user_role == "Head Office Authorization Officer":
        filtered = list(refill_requests.values())
    else:
        filtered = []
    if status_filter:
        filtered = [r for r in filtered if r.status == status_filter]
    return filtered

def take_action_on_refill_request(request_id: str, action: str, approver: str, role: str, comment: str = None) -> RefillRequest:
    refill_request = refill_requests.get(request_id)
    if not refill_request:
        raise ValueError("Refill request not found")
    if refill_request.status != "Pending":
        raise ValueError("Refill request already processed")
    if action.lower() not in ["approve", "refuse"]:
        raise ValueError("Invalid action")
    if action.lower() == "approve":
        refill_request.status = "Approved"
    else:
        refill_request.status = "Refused"
    refill_request.updated_at = datetime.utcnow()
    refill_request.approval_history.append(
        ApprovalRecord(
            approver=approver,
            role=role,
            action=refill_request.status.lower(),
            timestamp=refill_request.updated_at,
            comment=comment
        )
    )
    return refill_request
