from typing import List
from datetime import datetime
from sqlalchemy.orm import Session
from models import RefillRequest, ApprovalRecord
from services_db import create_refill_request as db_create_refill_request, list_refill_requests as db_list_refill_requests, take_action_on_refill_request as db_take_action_on_refill_request

def create_refill_request(db: Session, atm_id: str, requested_amount: float, requestor: str, comment: str = None) -> RefillRequest:
    new_request_db = db_create_refill_request(db, atm_id, requested_amount, requestor, comment)
    approval_history = [
        ApprovalRecord(
            approver=record.approver,
            role=record.role,
            action=record.action,
            timestamp=record.timestamp,
            comment=record.comment
        ) for record in new_request_db.approval_history
    ]
    return RefillRequest(
        request_id=str(new_request_db.request_id),
        atm_id=new_request_db.atm_id,
        requested_amount=new_request_db.requested_amount,
        requestor=new_request_db.requestor,
        status=new_request_db.status.value,
        created_at=new_request_db.created_at,
        updated_at=new_request_db.updated_at,
        approval_history=approval_history
    )

def list_refill_requests(db: Session, user_role: str, username: str, status_filter: str = None) -> List[RefillRequest]:
    requests_db = db_list_refill_requests(db, user_role, username, status_filter)
    result = []
    for r in requests_db:
        approval_history = [
            ApprovalRecord(
                approver=record.approver,
                role=record.role,
                action=record.action,
                timestamp=record.timestamp,
                comment=record.comment
            ) for record in r.approval_history
        ]
        result.append(
            RefillRequest(
                request_id=str(r.request_id),
                atm_id=r.atm_id,
                requested_amount=r.requested_amount,
                requestor=r.requestor,
                status=r.status.value,
                created_at=r.created_at,
                updated_at=r.updated_at,
                approval_history=approval_history
            )
        )
    return result

def take_action_on_refill_request(db: Session, request_id: str, action: str, approver: str, role: str, comment: str = None) -> RefillRequest:
    updated_request_db = db_take_action_on_refill_request(db, request_id, action, approver, role, comment)
    approval_history = [
        ApprovalRecord(
            approver=record.approver,
            role=record.role,
            action=record.action,
            timestamp=record.timestamp,
            comment=record.comment
        ) for record in updated_request_db.approval_history
    ]
    return RefillRequest(
        request_id=str(updated_request_db.request_id),
        atm_id=updated_request_db.atm_id,
        requested_amount=updated_request_db.requested_amount,
        requestor=updated_request_db.requestor,
        status=updated_request_db.status.value,
        created_at=updated_request_db.created_at,
        updated_at=updated_request_db.updated_at,
        approval_history=approval_history
    )
