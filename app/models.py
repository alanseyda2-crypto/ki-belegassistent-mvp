from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Numeric, Boolean, ForeignKey
from datetime import datetime
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(50), nullable=False)
    status = Column(String(50), default="erkannt")

    invoice_date = Column(Date, nullable=True)
    vendor = Column(String(255), nullable=True)
    invoice_number = Column(String(255), nullable=True)
    gross_amount = Column(Numeric(10, 2), nullable=True)
    net_amount = Column(Numeric(10, 2), nullable=True)
    vat_amount = Column(Numeric(10, 2), nullable=True)
    vat_rate = Column(Numeric(5, 2), nullable=True)
    currency = Column(String(10), default="EUR")

    booking_category = Column(String(255), nullable=True)
    account = Column(String(50), nullable=True)
    contra_account = Column(String(50), nullable=True)
    payment_method = Column(String(100), nullable=True)
    vat_key = Column(String(100), nullable=True)
    booking_text = Column(String(255), nullable=True)
    booking_confidence = Column(Numeric(4, 2), nullable=True)
    booking_source = Column(String(50), default="regelwerk")
    is_confirmed = Column(Boolean, default=False)

    ocr_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class BookingRule(Base):
    __tablename__ = "booking_rules"

    id = Column(Integer, primary_key=True, index=True)
    vendor = Column(String(255), nullable=False, index=True)
    category = Column(String(255), nullable=True)
    account = Column(String(50), nullable=False)
    contra_account = Column(String(50), nullable=False)
    payment_method = Column(String(100), nullable=True)
    vat_key = Column(String(100), nullable=True)
    booking_text = Column(String(255), nullable=True)
    times_used = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)



class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    field_name = Column(String(100), nullable=False)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    source = Column(String(50), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
