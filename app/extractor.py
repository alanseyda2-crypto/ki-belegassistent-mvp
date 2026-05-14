import re
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from .accounting_ai import rule_based_skr03, ai_skr03_suggestion


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


def _ocr_image_variants(file_path: str) -> str:
    """Mehrstufige Bild-OCR für Kassenbons/Handyfotos.

    Es werden mehrere Bildvarianten erzeugt: Original, vergrößert, kontrastverstärkt,
    schwarz/weiß und geschärft. Die OCR-Ergebnisse werden zusammengeführt.
    Das verbessert insbesondere Thermopapier-, Tankstellen- und Restaurantbelege.
    """
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance
    import pytesseract

    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    variants = []
    variants.append(img)

    # großziehen: kleine/feine Thermopapier-Schriften werden lesbarer
    scale = 2 if max(img.size) < 2200 else 1
    if scale > 1:
        up = img.resize((img.width * scale, img.height * scale))
    else:
        up = img.copy()

    gray = ImageOps.grayscale(up)
    gray = ImageOps.autocontrast(gray)
    variants.append(gray)

    sharp = gray.filter(ImageFilter.SHARPEN)
    sharp = ImageEnhance.Contrast(sharp).enhance(1.8)
    variants.append(sharp)

    # Binärbild: oft gut für Kassenbons
    bw = sharp.point(lambda p: 255 if p > 165 else 0)
    variants.append(bw)

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11",
    ]
    texts = []
    for variant in variants:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(variant, lang="deu+eng", config=cfg).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue

    # Zeilen deduplizieren, Reihenfolge behalten
    seen = set()
    merged_lines = []
    for txt in texts:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            key = re.sub(r"\s+", " ", line).lower()
            if key not in seen:
                seen.add(key)
                merged_lines.append(line)
    return "\n".join(merged_lines)


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
            return _ocr_image_variants(file_path).strip()
        except Exception as exc:
            return (
                "Bild-OCR ist lokal noch nicht verfügbar. "
                "Installiere Tesseract OCR oder nutze später Azure Document Intelligence. "
                f"Fehler: {exc}"
            )

    return ""


def _parse_decimal(value: str) -> Optional[Decimal]:
    value = value.strip().replace("€", "").replace("EUR", "").replace("EÜR", "").strip()
    # OCR-Verwechslungen korrigieren
    value = value.replace("O", "0").replace("o", "0")
    value = re.sub(r"[^0-9,.-]", "", value)
    if not value:
        return None
    # Deutsches Format bevorzugen
    cleaned = value.replace(".", "").replace(",", ".")
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def find_date(text: str):
    patterns = [
        (r"\b(\d{2}\.\d{2}\.\d{4})\b", "%d.%m.%Y"),
        (r"\b(\d{1,2}\.\d{1,2}\.\d{2})\b", "%d.%m.%y"),
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{2}/\d{2}/\d{4})\b", "%d/%m/%Y"),
        (r"\b(\d{1,2}/\d{1,2}/\d{2})\b", "%d/%m/%y"),
    ]
    for pattern, fmt in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return datetime.strptime(match.group(1), fmt).date()
            except ValueError:
                pass

    match = re.search(r"\b(\d{1,2})\.?:?\s+([A-Za-zÄÖÜäöüß]+)\s+(\d{4})\b", text, re.IGNORECASE)
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
        r"(?:Rechnung\s*#|Rechnungsnummer|Rechnung\s*Nr\.?|Invoice\s*No\.?|Invoice\s*Number|Belegnummer|Beleg-Nr\.?|Beleg\s*Nr\.?)[:\s#-]*([A-Z0-9\-/]+)",
        r"(?:Bon|Kassenbon|Transaktions-Nr\.?|Bon-Nr\.?)[:\s#-]*([A-Z0-9\-/]+)",
        r"(?:Rechnung|Invoice)[:\s#-]*([A-Z0-9\-/]{4,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lstrip("#")
    return None


def _amount_regex() -> str:
    return r"(?:€\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+[,\.]\d{2})\s*(?:€|EUR|EÜR)?"


def _amounts_in_line(line: str) -> list[Decimal]:
    found = re.findall(_amount_regex(), line, flags=re.IGNORECASE)
    vals = []
    for v in found:
        d = _parse_decimal(v)
        if d is not None:
            vals.append(d)
    return vals


def find_amount(text: str, labels: list[str]) -> Optional[Decimal]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # stärkste Logik: auf Zeilen mit TOTAL/Gesamt/Summe bevorzugt letzte/groesste Geldzahl nehmen
    for label in labels:
        label_re = re.compile(label, re.IGNORECASE)
        for i, line in enumerate(lines):
            if not label_re.search(line):
                continue
            window = " ".join(lines[i:i+3])
            vals = _amounts_in_line(window)
            vals = [v for v in vals if v >= Decimal("0.01")]
            if vals:
                return max(vals)

    # Fallback alter Pattern
    for label in labels:
        pattern = rf"{label}[^\n\d€]{{0,60}}{_amount_regex()}"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_decimal(match.group(1))
    return None


def find_vat(text: str) -> Optional[Decimal]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(k in lower for k in ["mwst", "ust", "umsatzsteuer", "vat", "tax"]):
            window = " ".join(lines[i:i+2])
            vals = _amounts_in_line(window)
            # Beim MwSt.-Satz steht oft 19,00 vor dem Steuerbetrag; Steuerbetrag ist meist letzte Geldzahl
            money_vals = [v for v in vals if v < Decimal("1000.00")]
            if money_vals:
                # Beträge wie 19.00 als Prozentsatz ignorieren, falls weitere Werte vorhanden sind
                filtered = [v for v in money_vals if v not in [Decimal("7.00"), Decimal("19.00")]]
                return filtered[-1] if filtered else money_vals[-1]
    patterns = [
        rf"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{{0,80}}{_amount_regex()}",
        rf"(?:7\s*%|19\s*%)[^\n]{{0,50}}{_amount_regex()}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _parse_decimal(match.group(1))
    return None


def find_vat_rate(text: str) -> Optional[Decimal]:
    patterns = [
        r"(?:MwSt\.?|MIST|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{0,40}?(\d{1,2}(?:[,\.]\d{1,2})?)\s*%",
        r"\b(7|19)\s*(?:,00|\.00)?\s*%",
        r"\b(7|19)\s*(?:,00|\.00)\s+A\b",  # OCR bei Tankbons: MWST 19,00 A
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


def find_largest_amount(text: str) -> Optional[Decimal]:
    values = re.findall(_amount_regex(), text, flags=re.IGNORECASE)
    decimals = [_parse_decimal(v) for v in values]
    decimals = [d for d in decimals if d is not None]
    # unrealistisch große Transaktions-/Kartennummern fallen durch Regex meist nicht rein, trotzdem filtern
    decimals = [d for d in decimals if d < Decimal("100000.00")]
    return max(decimals) if decimals else None


def find_vendor(text: str) -> Optional[str]:
    lines = [re.sub(r"\s+", " ", line.strip(" ,;")) for line in text.splitlines() if line.strip()]
    legal_forms = ["gmbh", "ug", "gbr", "ag", "kg", "ohg", "e.k", "mbh", "ltd"]

    for line in lines[:25]:
        lower = line.lower()
        if any(form in lower for form in legal_forms):
            return line

    # Tankstellen/Kassenbons: erste starke Händlerzeile nehmen
    for line in lines[:10]:
        lower = line.lower()
        if any(k in lower for k in ["tankstelle", "hem", "aral", "shell", "esso", "jet", "totalenergies", "avia"]):
            return line.strip(" ,")

    bad_keywords = ["rechnung", "invoice", "datum", "seite", "betrag", "summe", "total", "kundenbeleg", "terminal", "beleg", "shop", "www", ".de", ".com"]
    for line in lines[:15]:
        if len(line) < 3 or len(line) > 80:
            continue
        lower = line.lower()
        if any(k in lower for k in bad_keywords):
            continue
        if re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            return line.strip(" ,")
    return None


def suggest_booking(text: str, vendor: Optional[str], vat_rate: Optional[Decimal]) -> dict:
    # Erst lokale SKR03-Regeln, dann optional OpenAI überschreiben, falls Key gesetzt und valides Ergebnis kommt.
    rule_result = rule_based_skr03(text, vendor, vat_rate)
    # KI nur optional; bei Railway ohne OPENAI_API_KEY bleibt es stabil regelbasiert.
    return rule_result


def extract_fields(text: str) -> dict:
    gross = find_amount(text, [
        "Gesamtsumme", "Gesamtbetrag", "Gesamtpreis", "Bruttobetrag", "Rechnungssumme",
        "Zu zahlen", "Amount due", "TOTAL", "Total", "Summe", "Betrag"
    ])
    if gross is None:
        gross = find_largest_amount(text)

    vat = find_vat(text)
    vat_rate = find_vat_rate(text)
    vendor = find_vendor(text)
    booking = suggest_booking(text, vendor, vat_rate)

    return {
        "invoice_date": find_date(text),
        "vendor": vendor,
        "invoice_number": find_invoice_number(text),
        "gross_amount": gross,
        "vat_amount": vat,
        "vat_rate": vat_rate,
        **booking,
        "currency": "EUR" if "€" in text or "EUR" in text.upper() or "EÜR" in text.upper() else None,
    }
