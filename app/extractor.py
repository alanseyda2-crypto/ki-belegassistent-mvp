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


def _normalize_ocr_text(text: str) -> str:
    """Korrigiert typische OCR-Fehler auf Kassenbons, ohne Originaldatei zu ändern."""
    replacements = {
        "MIST": "MWST",
        "MWSI": "MWST",
        "MW5T": "MWST",
        "UST.": "UST",
        "€UR": "EUR",
        "EÜR": "EUR",
        "SROS5O": "BRUTTO",
    }
    out = text
    for wrong, right in replacements.items():
        out = re.sub(re.escape(wrong), right, out, flags=re.IGNORECASE)
    return out


def _clean_vendor_line(line: str) -> str:
    line = re.sub(r"[^A-Za-zÄÖÜäöüß0-9& .,'/-]", "", line)
    return re.sub(r"\s+", " ", line).strip(" ,;:-")


def _is_likely_money_amount(value: Decimal) -> bool:
    """Filtert typische OCR-Störwerte wie MwSt.-Sätze, Literpreise und Nullwerte."""
    return Decimal("0.01") <= value < Decimal("100000.00")


def _line_context(lines: list[str], index: int, before: int = 1, after: int = 2) -> str:
    start = max(0, index - before)
    end = min(len(lines), index + after + 1)
    return " ".join(lines[start:end])


def find_receipt_total(text: str) -> Optional[Decimal]:
    """Robuste Speziallogik für Handyfotos, Kassenbons und Tankbelege.

    Priorität: explizite Brutto-/Total-/Kartenzahlung-Zeilen. Liter, Literpreise,
    MwSt-Sätze, Netto-Beträge und einzelne Steuersätze werden konsequent ignoriert.
    """
    text = _normalize_ocr_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates: list[tuple[int, Decimal]] = []

    strong_total = ["total", "gesamtsumme", "gesamtbetrag", "gesamtpreis", "summe", "zu zahlen", "brutto"]
    payment_words = ["kartenzahlung", "mastercard", "visa", "girocard", "ec-karte", "debit", "kreditkarte", "karte"]
    ignore_words = ["liter", "eur/l", "eur / l", "l/", "stk", "menge", "einzelpreis", "netto"]

    # 1) Wenn eine Zeile BRUTTO enthält, Wert direkt nach BRUTTO bevorzugen.
    for line in lines:
        lower = line.lower()
        m = re.search(r"brutto[^0-9]{0,25}" + _amount_regex(), line, re.IGNORECASE)
        if m:
            val = _parse_decimal(m.group(1))
            if val and _is_likely_money_amount(val) and val not in [Decimal("7.00"), Decimal("19.00")]:
                candidates.append((260 + min(int(val), 80), val))

    # 2) Explizite Total-/Gesamt-/Zahlungszeilen. Bei mehreren Zahlen meist letzte/groesste Zahl.
    for i, line in enumerate(lines):
        lower = line.lower()
        has_total = any(k in lower for k in strong_total)
        has_payment = any(k in lower for k in payment_words)
        has_betrag = bool(re.search(r"\bbetrag\b", lower))
        if not (has_total or has_payment or has_betrag):
            continue
        # Nächste Zeile mitnehmen, weil OCR Betrag manchmal in Folgezeile setzt.
        window = _line_context(lines, i, before=0, after=1)
        vals = _amounts_in_line(window)
        for val in vals:
            if not _is_likely_money_amount(val) or val in [Decimal("7.00"), Decimal("19.00")]:
                continue
            score = 0
            if has_total:
                score += 210
            if "brutto" in lower:
                score += 230
            if has_payment:
                score += 170
            if has_betrag:
                score += 120
            if any(w in lower for w in ignore_words):
                score -= 180
            if "mwst" in lower or "ust" in lower or "steuer" in lower:
                score -= 130
            if "netto" in lower and "brutto" not in lower:
                score -= 160
            # Höhere Bruttowerte innerhalb derselben Zeile leicht bevorzugen.
            score += min(int(val), 80)
            candidates.append((score, val))

    # 3) Muster: NETTO x MWST y BRUTTO z oder NETTO x BRUTTO z.
    brutto_matches = re.finditer(r"brutto[^0-9]{0,25}" + _amount_regex(), text, re.IGNORECASE)
    for m in brutto_matches:
        val = _parse_decimal(m.group(1))
        if val and _is_likely_money_amount(val) and val not in [Decimal("7.00"), Decimal("19.00")]:
            candidates.append((240 + min(int(val), 80), val))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1]
    return None

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
    text = _normalize_ocr_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Direktmuster auf MwSt-Zeilen: "MWST 19,00% A 12,07 EUR".
    for i, line in enumerate(lines):
        lower = line.lower()
        if not any(k in lower for k in ["mwst", "ust", "umsatzsteuer", "vat", "tax"]):
            continue
        if "netto" in lower and "brutto" in lower and "mwst" not in lower:
            continue
        window = _line_context(lines, i, before=0, after=0)
        vals = _amounts_in_line(window)
        vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
        if vals:
            # Steuerbetrag ist auf MwSt-Zeilen meist der letzte Geldwert.
            return vals[-1]

    # Wenn Steuerbetrag in der Folgezeile steht, maximal eine Zeile weiter suchen.
    for i, line in enumerate(lines):
        lower = line.lower()
        if not any(k in lower for k in ["mwst", "ust", "umsatzsteuer", "vat", "tax"]):
            continue
        window = _line_context(lines, i, before=0, after=1)
        # Nicht die Netto-/Brutto-Zeile als Steuerbetrag nehmen.
        window = " ".join([part for part in window.split("  ") if "netto" not in part.lower() and "brutto" not in part.lower()])
        vals = _amounts_in_line(window)
        vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
        if vals:
            return vals[-1]

    patterns = [
        rf"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{{0,80}}{_amount_regex()}",
        rf"(?:7\s*%|19\s*%)[^\n]{{0,50}}{_amount_regex()}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = _parse_decimal(match.group(1))
            if val and val not in [Decimal("7.00"), Decimal("19.00")]:
                return val
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
    text = _normalize_ocr_text(text)
    vals: list[Decimal] = []
    for line in text.splitlines():
        lower = line.lower()
        if any(k in lower for k in ["liter", "eur/l", "eur / l", "stk", "menge", "einzelpreis", "art.-nr", "terminal", "transaktion"]):
            continue
        for v in _amounts_in_line(line):
            if v is None or not _is_likely_money_amount(v):
                continue
            if v in [Decimal("7.00"), Decimal("19.00")]:
                continue
            vals.append(v)
    return max(vals) if vals else None

def find_vendor(text: str) -> Optional[str]:
    text = _normalize_ocr_text(text)
    raw_lines = [line for line in text.splitlines() if line.strip()]
    lines = [_clean_vendor_line(line) for line in raw_lines if _clean_vendor_line(line)]
    full_lower = "\n".join(lines).lower()

    # Bekannte Händler/Tankstellen/Receipt-Brands zuerst.
    known = [
        ("hem", "HEM Tankstelle"),
        ("hem-tankstelle", "HEM Tankstelle"),
        ("aral", "ARAL"),
        ("shell", "Shell"),
        ("esso", "Esso"),
        ("jet", "JET"),
        ("totalenergies", "TotalEnergies"),
        ("avia", "AVIA"),
        ("edeka", "EDEKA"),
        ("rewe", "REWE"),
        ("lidl", "Lidl"),
        ("aldi", "ALDI"),
        ("kaufland", "Kaufland"),
        ("dm-drogerie", "dm-drogerie markt"),
        ("rossmann", "Rossmann"),
        ("ikea", "IKEA"),
        ("amazon", "Amazon"),
    ]
    for needle, name in known:
        if needle in full_lower:
            return name

    # Unternehmenszeile mit Rechtsform.
    legal_forms = ["gmbh", "ug", "gbr", "ag", "kg", "ohg", "e.k", "mbh", "ltd"]
    for line in lines[:30]:
        lower = line.lower()
        if any(form in lower for form in legal_forms):
            return line

    # Bei Kassenbons steht der Händler fast immer in den ersten 3-6 Zeilen.
    bad_keywords = [
        "rechnung", "invoice", "datum", "seite", "betrag", "summe", "total", "kundenbeleg",
        "terminal", "beleg", "shop", "www", ".de", ".com", "terminalnummer", "kartenzahlung",
        "straße", "strasse", "telefon", "tel", "fax", "ust", "steuer", "iban", "bic", "bon", "kasse",
        "kundenservice", "versenden", "käufer", "zahlungsmethode"
    ]
    for line in lines[:10]:
        lower = line.lower()
        if len(line) < 3 or len(line) > 60:
            continue
        if any(k in lower for k in bad_keywords):
            continue
        # Zeilen mit zu vielen Zahlen sind meist Adresse/Terminal.
        if len(re.findall(r"\d", line)) > 4:
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
    text = _normalize_ocr_text(text)
    gross = find_receipt_total(text)
    if gross is None:
        gross = find_amount(text, [
            "Gesamtsumme", "Gesamtbetrag", "Gesamtpreis", "Bruttobetrag", "Rechnungssumme",
            "Zu zahlen", "Amount due", "TOTAL", "Total", "Summe", "Betrag"
        ])
    if gross is None:
        gross = find_largest_amount(text)

    vat = find_vat(text)
    vat_rate = find_vat_rate(text)
    # Plausibilisierung: Bei 19% ist Vorsteuer ca. Brutto * 19/119, bei 7% ca. Brutto * 7/107.
    if gross is not None and vat_rate is not None:
        expected = None
        if vat_rate == Decimal("19.00"):
            expected = (gross * Decimal("19") / Decimal("119")).quantize(Decimal("0.01"))
        elif vat_rate == Decimal("7.00"):
            expected = (gross * Decimal("7") / Decimal("107")).quantize(Decimal("0.01"))
        if expected is not None:
            if vat is None or vat <= 0 or vat >= gross or abs(vat - expected) > Decimal("1.00"):
                vat = expected

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
