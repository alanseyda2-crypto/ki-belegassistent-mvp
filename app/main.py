import csv
import io
import shutil
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import desc, text

from .database import Base, engine, SessionLocal
from .models import Document, BookingRule
from .extractor import extract_text, extract_fields
from .accounting_ai import ai_skr03_suggestion, rule_based_skr03

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

Base.metadata.create_all(bind=engine)


def ensure_columns():
    doc_columns = [
        ("vat_rate", "NUMERIC(5,2)"),
        ("booking_category", "VARCHAR(255)"),
        ("account", "VARCHAR(50)"),
        ("contra_account", "VARCHAR(50)"),
        ("payment_method", "VARCHAR(100)"),
        ("vat_key", "VARCHAR(100)"),
        ("booking_text", "VARCHAR(255)"),
        ("booking_confidence", "NUMERIC(4,2)"),
        ("is_confirmed", "BOOLEAN DEFAULT 0"),
    ]
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(documents)"))}
        for name, sql_type in doc_columns:
            if name not in existing:
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}"))

ensure_columns()

app = FastAPI(title="KI-Belegassistent MVP")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def apply_learned_rule(db: Session, doc: Document, fields: dict) -> dict:
    vendor = fields.get("vendor")
    if not vendor:
        return fields
    rule = db.query(BookingRule).filter(BookingRule.vendor == vendor).order_by(desc(BookingRule.times_used)).first()
    if not rule:
        return fields
    fields.update({
        "booking_category": rule.category,
        "account": rule.account,
        "contra_account": rule.contra_account,
        "payment_method": rule.payment_method,
        "vat_key": rule.vat_key,
        "booking_text": rule.booking_text,
        "booking_confidence": Decimal("0.96"),
    })
    return fields


def create_or_update_rule(db: Session, doc: Document):
    if not doc.vendor or not doc.account or not doc.contra_account:
        return
    rule = db.query(BookingRule).filter(BookingRule.vendor == doc.vendor, BookingRule.account == doc.account).first()
    if rule:
        rule.category = doc.booking_category
        rule.contra_account = doc.contra_account
        rule.payment_method = doc.payment_method
        rule.vat_key = doc.vat_key
        rule.booking_text = doc.booking_text
        rule.times_used = (rule.times_used or 0) + 1
        rule.updated_at = datetime.utcnow()
    else:
        rule = BookingRule(
            vendor=doc.vendor,
            category=doc.booking_category,
            account=doc.account,
            contra_account=doc.contra_account,
            payment_method=doc.payment_method,
            vat_key=doc.vat_key,
            booking_text=doc.booking_text,
        )
        db.add(rule)


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db), month: Optional[str] = None):
    query = db.query(Document)
    if month:
        year, m = month.split("-")
        query = query.filter(Document.invoice_date >= f"{year}-{m}-01")
        if m == "12":
            query = query.filter(Document.invoice_date < f"{int(year)+1}-01-01")
        else:
            query = query.filter(Document.invoice_date < f"{year}-{int(m)+1:02d}-01")
    documents = query.order_by(desc(Document.invoice_date), desc(Document.created_at)).all()
    return templates.TemplateResponse("index.html", {"request": request, "documents": documents, "month": month})


@app.post("/upload")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Keine Datei hochgeladen")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Nur PDF, JPG, PNG oder WEBP erlaubt")

    stored_name = f"{uuid.uuid4().hex}{suffix}"
    target = UPLOAD_DIR / stored_name
    with target.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ocr_text = extract_text(str(target), file.content_type or "")
    fields = extract_fields(ocr_text)

    ai_booking = ai_skr03_suggestion(
        ocr_text,
        fields.get("vendor"),
        fields.get("invoice_date"),
        fields.get("gross_amount"),
        fields.get("vat_amount"),
        fields.get("vat_rate"),
    )
    if ai_booking:
        fields.update(ai_booking)

    dummy = Document(vendor=fields.get("vendor"))
    fields = apply_learned_rule(db, dummy, fields)

    doc = Document(
        filename=stored_name,
        original_filename=file.filename,
        file_path=str(target),
        file_type=file.content_type or suffix,
        status="erkannt" if fields.get("invoice_date") or fields.get("gross_amount") else "prüfen",
        invoice_date=fields.get("invoice_date"),
        vendor=fields.get("vendor"),
        invoice_number=fields.get("invoice_number"),
        gross_amount=fields.get("gross_amount"),
        vat_amount=fields.get("vat_amount"),
        vat_rate=fields.get("vat_rate"),
        booking_category=fields.get("booking_category"),
        account=fields.get("account"),
        contra_account=fields.get("contra_account"),
        payment_method=fields.get("payment_method"),
        vat_key=fields.get("vat_key"),
        booking_text=fields.get("booking_text"),
        booking_confidence=fields.get("booking_confidence"),
        currency=fields.get("currency") or "EUR",
        ocr_text=ocr_text,
    )
    db.add(doc)
    db.commit()
    return RedirectResponse(url="/", status_code=303)


@app.get("/documents/{document_id}", response_class=HTMLResponse)
def document_detail(document_id: int, request: Request, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    return templates.TemplateResponse("detail.html", {"request": request, "doc": doc})


@app.post("/documents/{document_id}/ai-booking")
def recalc_ai_booking(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    ai_booking = ai_skr03_suggestion(doc.ocr_text or "", doc.vendor, doc.invoice_date, doc.gross_amount, doc.vat_amount, doc.vat_rate)
    if not ai_booking:
        ai_booking = rule_based_skr03(doc.ocr_text or "", doc.vendor, doc.vat_rate)
    for key, value in ai_booking.items():
        setattr(doc, key, value)
    db.commit()
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.post("/documents/{document_id}/correct")
def correct_booking(
    document_id: int,
    booking_category: str = Form(""),
    account: str = Form(...),
    contra_account: str = Form(...),
    payment_method: str = Form("Bank"),
    vat_key: str = Form("Vorsteuer prüfen"),
    booking_text: str = Form(""),
    save_rule: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    doc.booking_category = booking_category
    doc.account = account
    doc.contra_account = contra_account
    doc.payment_method = payment_method
    doc.vat_key = vat_key
    doc.booking_text = booking_text
    doc.booking_confidence = Decimal("1.00")
    doc.is_confirmed = True
    if save_rule:
        create_or_update_rule(db, doc)
    db.commit()
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.post("/documents/{document_id}/confirm")
def confirm_booking(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    doc.is_confirmed = True
    create_or_update_rule(db, doc)
    db.commit()
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)


@app.get("/documents/{document_id}/file")
def document_file(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    return FileResponse(doc.file_path, filename=doc.original_filename, media_type=doc.file_type)


@app.get("/documents/{document_id}/download")
def document_download(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    return FileResponse(
        doc.file_path,
        filename=doc.original_filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={doc.original_filename}"},
    )


@app.post("/documents/{document_id}/delete")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if doc:
        try:
            Path(doc.file_path).unlink(missing_ok=True)
        except Exception:
            pass
        db.delete(doc)
        db.commit()
    return RedirectResponse(url="/", status_code=303)


@app.get("/export/datev")
def export_datev(db: Session = Depends(get_db), confirmed_only: bool = False):
    query = db.query(Document)
    if confirmed_only:
        query = query.filter(Document.is_confirmed == True)
    docs = query.order_by(Document.invoice_date.asc(), Document.id.asc()).all()

    output = io.StringIO()
    writer = csv.writer(output, delimiter=";")

    # Kompakter Buchungsstapel-Export für die Weiterverarbeitung/Import-Vorbereitung.
    # DATEV-Programme erwarten je nach Version/Berater-Setup ggf. zusätzliche Stapel-Metadaten.
    writer.writerow([
        "Umsatz",
        "Soll/Haben-Kennzeichen",
        "WKZ Umsatz",
        "Konto",
        "Gegenkonto",
        "BU-Schlüssel",
        "Belegdatum",
        "Belegfeld 1",
        "Buchungstext",
        "Lieferant",
        "Kategorie",
        "Zahlungsart",
        "MwSt-Satz",
        "MwSt-Betrag",
        "Bestätigt",
        "Dateiname",
    ])

    for d in docs:
        vat_rate = float(d.vat_rate or 0)
        if vat_rate == 19:
            bu = "9"
        elif vat_rate == 7:
            bu = "8"
        elif vat_rate == 0:
            bu = ""
        else:
            bu = ""

        amount = f"{float(d.gross_amount or 0):.2f}".replace(".", ",")
        vat_amount = f"{float(d.vat_amount or 0):.2f}".replace(".", ",") if d.vat_amount is not None else ""
        vat_rate_text = f"{vat_rate:.2f}".replace(".", ",") if d.vat_rate is not None else ""
        belegdatum = d.invoice_date.strftime("%d%m") if d.invoice_date else ""
        text_value = d.booking_text or d.vendor or d.original_filename or "Beleg"

        writer.writerow([
            amount,
            "S",
            d.currency or "EUR",
            d.account or "",
            d.contra_account or "",
            bu,
            belegdatum,
            d.invoice_number or "",
            text_value[:60],
            d.vendor or "",
            d.booking_category or "",
            d.payment_method or "",
            vat_rate_text,
            vat_amount,
            "ja" if d.is_confirmed else "nein",
            d.original_filename or d.filename,
        ])

    output.seek(0)
    suffix = "bestaetigt" if confirmed_only else "alle"
    filename = f"datev_buchungsstapel_{suffix}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        iter(["\ufeff" + output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/rules", response_class=HTMLResponse)
def rules(request: Request, db: Session = Depends(get_db)):
    rules = db.query(BookingRule).order_by(desc(BookingRule.times_used), BookingRule.vendor.asc()).all()
    return templates.TemplateResponse("rules.html", {"request": request, "rules": rules})


@app.post("/rules/create")
def create_rule(
    vendor: str = Form(...),
    category: str = Form(""),
    account: str = Form(...),
    contra_account: str = Form("1200"),
    payment_method: str = Form("Bank"),
    vat_key: str = Form(""),
    booking_text: str = Form(""),
    db: Session = Depends(get_db),
):
    rule = BookingRule(
        vendor=vendor.strip(),
        category=category.strip() or None,
        account=account.strip(),
        contra_account=contra_account.strip(),
        payment_method=payment_method.strip() or None,
        vat_key=vat_key.strip() or None,
        booking_text=booking_text.strip() or None,
    )
    db.add(rule)
    db.commit()
    return RedirectResponse(url="/rules", status_code=303)


@app.post("/rules/{rule_id}/update")
def update_rule(
    rule_id: int,
    vendor: str = Form(...),
    category: str = Form(""),
    account: str = Form(...),
    contra_account: str = Form("1200"),
    payment_method: str = Form("Bank"),
    vat_key: str = Form(""),
    booking_text: str = Form(""),
    db: Session = Depends(get_db),
):
    rule = db.get(BookingRule, rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="Lernregel nicht gefunden")
    rule.vendor = vendor.strip()
    rule.category = category.strip() or None
    rule.account = account.strip()
    rule.contra_account = contra_account.strip()
    rule.payment_method = payment_method.strip() or None
    rule.vat_key = vat_key.strip() or None
    rule.booking_text = booking_text.strip() or None
    rule.updated_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(url="/rules", status_code=303)


@app.post("/rules/{rule_id}/delete")
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.get(BookingRule, rule_id)
    if rule:
        db.delete(rule)
        db.commit()
    return RedirectResponse(url="/rules", status_code=303)
