import re
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from .accounting_ai import rule_based_skr03


GERMAN_MONTHS = {
    "januar": 1, "jan": 1,
    "februar": 2, "feb": 2,
    "märz": 3, "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4,
    "mai": 5,
    "juni": 6, "jun": 6,
    "juli": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9,
    "oktober": 10, "okt": 10,
    "november": 11, "nov": 11,
    "dezember": 12, "dez": 12,
}


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
    value = value.strip()
    value = value.replace("€", "").replace("EUR", "").strip()
    cleaned = value.replace(".", "").replace(",", ".")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def find_date(text: str):
    # 08.05.2026, 2026-05-08, 08/05/2026
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

    # 08. Mai 2026 / 8 Mai 2026
    match = re.search(
        r"\b(\d{1,2})\.?:?\s+([A-Za-zÄÖÜäöüß]+)\s+(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower().replace("ä", "ae")
        year = int(match.group(3))
        month = GERMAN_MONTHS.get(month_name)
        if month:
            try:
                return datetime(year, month, day).date()
            except ValueError:
                return None
    return None


def find_invoice_number(text: str) -> Optional[str]:
    patterns = [
        r"(?:Rechnung\s*#|Rechnungsnummer|Rechnung\s*Nr\.?|Invoice\s*No\.?|Invoice\s*Number|Belegnummer)[:\s#-]*([A-Z0-9\-/]+)",
        r"(?:Rechnung|Invoice)[:\s#-]*([A-Z0-9\-/]{4,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lstrip("#")
    return None


def _amount_regex() -> str:
    # erkennt 50,00 / 1.250,00 mit optionalem € vor oder nach dem Betrag
    return r"(?:€\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+,\d{2})\s*(?:€|EUR)?"


def find_amount(text: str, labels: list[str]) -> Optional[Decimal]:
    for label in labels:
        pattern = rf"{label}[^\n\d€]{{0,40}}{_amount_regex()}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_decimal(match.group(1))
    return None


def find_vat(text: str) -> Optional[Decimal]:
    # Beispiel: MwSt. 7% €3,27 oder VAT 19% 11,17 EUR
    patterns = [
        rf"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{{0,60}}{_amount_regex()}",
        rf"(?:7%|19%)[^\n]{{0,30}}{_amount_regex()}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_decimal(match.group(1))
    return None


def find_largest_amount(text: str) -> Optional[Decimal]:
    values = re.findall(_amount_regex(), text)
    decimals = [_parse_decimal(v) for v in values]
    decimals = [d for d in decimals if d is not None]
    return max(decimals) if decimals else None


def find_vendor(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    legal_forms = ["gmbh", "ug", "gbr", "ag", "kg", "ohg", "e.k", "mbh", "ltd"]

    # bevorzugt echte Firmenzeile statt Domain/Shop-Zeile
    for line in lines[:20]:
        lower = line.lower()
        if any(form in lower for form in legal_forms):
            return line

    bad_keywords = ["rechnung", "invoice", "datum", "seite", "betrag", "summe", "total", "shop", "www", ".de", ".com"]
    for line in lines[:15]:
        if len(line) < 3 or len(line) > 80:
            continue
        lower = line.lower()
        if any(k in lower for k in bad_keywords):
            continue
        if re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            return line
    return None



def find_vat_rate(text: str) -> Optional[Decimal]:
    # Erkennt MwSt.-Sätze wie: MwSt. 7%, USt 19%, VAT 19%, 7% €3,27
    patterns = [
        r"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{0,30}?(\d{1,2}(?:,\d{1,2})?)\s*%",
        r"\b(7|19)\s*%[^\n]{0,40}(?:€|EUR|\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(',', '.')
            try:
                return Decimal(raw).quantize(Decimal("0.01"))
            except (InvalidOperation, ValueError):
                return None
    return None


def suggest_booking(text: str, vendor: Optional[str], vat_rate: Optional[Decimal]) -> dict:
    return rule_based_skr03(text, vendor, vat_rate)


def extract_fields(text: str) -> dict:
    gross = find_amount(text, [
        "Gesamtsumme", "Gesamtbetrag", "Gesamtpreis", "Bruttobetrag", "Rechnungssumme",
        "Summe", "Total", "Amount due", "Zu zahlen"
    ])
    if gross is None:
        gross = find_largest_amount(text)

    vat = find_vat(text)
    vat_rate = find_vat_rate(text)
    booking = suggest_booking(text, find_vendor(text), vat_rate)

    return {
        "invoice_date": find_date(text),
        "vendor": find_vendor(text),
        "invoice_number": find_invoice_number(text),
        "gross_amount": gross,
        "vat_amount": vat,
        "vat_rate": vat_rate,
        **booking,
        "currency": "EUR" if "€" in text or "EUR" in text.upper() else None,
    }
