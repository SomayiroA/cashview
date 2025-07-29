from sqlalchemy.orm import Session
from models_db import RefillRequestDB, ApprovalRecordDB, RequestStatus
from datetime import datetime
import uuid

def create_refill_request(db: Session, atm_id: str, requested_amount: float, requestor: str, comment: str = None) -> RefillRequestDB:
    new_request = RefillRequestDB(
        request_id=uuid.uuid4(),
        atm_id=atm_id,
        requested_amount=requested_amount,
        requestor=requestor,
        status=RequestStatus.Pending,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(new_request)
    db.commit()
    db.refresh(new_request)

    if comment:
        approval_record = ApprovalRecordDB(
            refill_request_id=new_request.request_id,
            approver=requestor,
            role="ATM Operations Staff",
            action="requested",
            timestamp=datetime.utcnow(),
            comment=comment
        )
        db.add(approval_record)
        db.commit()
    return new_request

def list_refill_requests(db: Session, user_role: str, username: str, status_filter: str = None):
    query = db.query(RefillRequestDB)
    if user_role == "ATM Operations Staff":
        query = query.filter(RefillRequestDB.requestor == username)
    elif user_role == "Branch Operations Manager":
        query = query.filter(RefillRequestDB.status == RequestStatus.Pending)
    elif user_role == "Vault Manager":
        query = query.filter(RefillRequestDB.status == RequestStatus.Approved)
    elif user_role == "Head Office Authorization Officer":
        pass  # no filter
    else:
        return []

    if status_filter:
        try:
            status_enum = RequestStatus[status_filter]
            query = query.filter(RefillRequestDB.status == status_enum)
        except KeyError:
            return []

    return query.all()

def take_action_on_refill_request(db: Session, request_id: str, action: str, approver: str, role: str, comment: str = None):
    refill_request = db.query(RefillRequestDB).filter(RefillRequestDB.request_id == request_id).first()
    if not refill_request:
        raise ValueError("Refill request not found")
    if refill_request.status != RequestStatus.Pending:
        raise ValueError("Refill request already processed")
    if action.lower() not in ["approve", "refuse"]:
        raise ValueError("Invalid action")
    if action.lower() == "approve":
        refill_request.status = RequestStatus.Approved
    else:
        refill_request.status = RequestStatus.Refused
    refill_request.updated_at = datetime.utcnow()
    approval_record = ApprovalRecordDB(
        refill_request_id=refill_request.request_id,
        approver=approver,
        role=role,
        action=refill_request.status.value.lower(),
        timestamp=refill_request.updated_at,
        comment=comment
    )
    db.add(approval_record)
    db.commit()
    return refill_request
