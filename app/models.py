from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Numeric
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
    vat_amount = Column(Numeric(10, 2), nullable=True)
    currency = Column(String(10), default="EUR")

    ocr_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
