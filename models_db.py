from sqlalchemy import Column, String, Float, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
import uuid
from database import Base
from datetime import datetime

class RequestStatus(enum.Enum):
    Pending = "Pending"
    Approved = "Approved"
    Refused = "Refused"

class RefillRequestDB(Base):
    __tablename__ = "refill_requests"

    request_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    atm_id = Column(String, nullable=False)
    requested_amount = Column(Float, nullable=False)
    requestor = Column(String, nullable=False)
    status = Column(Enum(RequestStatus), default=RequestStatus.Pending, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    approval_history = relationship("ApprovalRecordDB", back_populates="refill_request", cascade="all, delete-orphan")

class ApprovalRecordDB(Base):
    __tablename__ = "approval_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    refill_request_id = Column(UUID(as_uuid=True), ForeignKey("refill_requests.request_id"), nullable=False)
    approver = Column(String, nullable=False)
    role = Column(String, nullable=False)
    action = Column(String, nullable=False)  # "approved", "refused", "requested"
    timestamp = Column(DateTime, default=datetime.utcnow)
    comment = Column(String, nullable=True)

    refill_request = relationship("RefillRequestDB", back_populates="approval_history")
