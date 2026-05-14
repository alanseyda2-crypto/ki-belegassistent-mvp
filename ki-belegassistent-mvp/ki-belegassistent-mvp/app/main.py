import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from .database import Base, engine, SessionLocal
from .models import Document
from .extractor import extract_text, extract_fields

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="KI-Belegassistent MVP")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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

    text = extract_text(str(target), file.content_type or "")
    fields = extract_fields(text)

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
        currency=fields.get("currency") or "EUR",
        ocr_text=text,
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


@app.get("/documents/{document_id}/file")
def document_file(document_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Beleg nicht gefunden")
    return FileResponse(doc.file_path, filename=doc.original_filename)


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
