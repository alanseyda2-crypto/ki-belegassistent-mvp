import re
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


def extract_text(file_path: str, content_type: str) -> str:
    path = Path(file_path)

    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts).strip()
        except Exception as exc:
            return f"PDF konnte nicht gelesen werden: {exc}"

    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            from PIL import Image
            import pytesseract
            return pytesseract.image_to_string(Image.open(file_path), lang="deu+eng").strip()
        except Exception as exc:
            return (
                "Bild-OCR ist lokal noch nicht verfügbar. "
                "Installiere Tesseract OCR oder nutze später Azure Document Intelligence. "
                f"Fehler: {exc}"
            )

    return ""


def _parse_decimal(value: str) -> Optional[Decimal]:
    cleaned = value.replace(".", "").replace(",", ".")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def find_date(text: str):
    patterns = [
        r"(\d{2}\.\d{2}\.\d{4})",
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{2}/\d{2}/\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1)
        for fmt in ["%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"]:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                pass
    return None


def find_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r"(?:Rechnungsnummer|Rechnung Nr\.?|Invoice No\.?|Invoice Number|Belegnummer)[:\s#-]*([A-Z0-9\-/]+)",
        r"(?:Rechnung|Invoice)[:\s#-]*([A-Z0-9\-/]{4,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def find_amount(text: str, labels: list[str]) -> Optional[Decimal]:
    for label in labels:
        pattern = rf"{label}[^\d]{{0,30}}(\d{{1,3}}(?:\.\d{{3}})*,\d{{2}}|\d+,\d{{2}})\s?(?:€|EUR)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_decimal(match.group(1))
    return None


def find_largest_amount(text: str) -> Optional[Decimal]:
    values = re.findall(r"(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})\s?(?:€|EUR)?", text)
    decimals = [_parse_decimal(v) for v in values]
    decimals = [d for d in decimals if d is not None]
    return max(decimals) if decimals else None


def find_vendor(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bad_keywords = ["rechnung", "invoice", "datum", "seite", "betrag", "summe", "total"]
    for line in lines[:12]:
        if len(line) < 3 or len(line) > 80:
            continue
        lower = line.lower()
        if any(k in lower for k in bad_keywords):
            continue
        if re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            return line
    return None


def extract_fields(text: str) -> dict:
    gross = find_amount(text, ["Gesamtbetrag", "Bruttobetrag", "Summe", "Total", "Amount due", "Zu zahlen"])
    if gross is None:
        gross = find_largest_amount(text)

    vat = find_amount(text, ["MwSt", "USt", "Umsatzsteuer", "VAT", "Tax"])

    return {
        "invoice_date": find_date(text),
        "vendor": find_vendor(text),
        "invoice_number": find_invoice_number(text),
        "gross_amount": gross,
        "vat_amount": vat,
        "currency": "EUR" if "€" in text or "EUR" in text.upper() else None,
    }
