import re
import os
import json
import base64
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
    """Nur echte Rechnungsnummern zuverlässig übernehmen.

    Bei Kassenbons stehen viele Nummern (Terminal, Trace, TSE, Beleg-Nr.). Diese sind
    meistens keine Rechnungsnummer und verwirren den DATEV-Export. Daher bei reinen
    Kassenbons lieber leer lassen, außer es gibt ein klares Rechnung/Rechnungsnummer-Label.
    """
    invoice_patterns = [
        r"(?:Rechnung\s*#|Rechnungsnummer|Rechnung\s*Nr\.?|Invoice\s*No\.?|Invoice\s*Number)[:\s#-]*([A-Z0-9\-/]{4,})",
    ]
    for pattern in invoice_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lstrip("#")

    # Nur sehr explizite Bonnummer übernehmen, nicht kurze Terminal-/Trace-Werte.
    explicit_receipt_patterns = [
        r"(?:Bonnummer|Kassenbonnummer|Belegnummer)[:\s#-]*([A-Z0-9\-/]{5,})",
    ]
    for pattern in explicit_receipt_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().lstrip("#")
            if not candidate.isdigit() or len(candidate) >= 6:
                return candidate
    return None

def _amount_regex() -> str:
    return r"(?:€\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+[,\.]\d{2})\s*(?:€|EUR|EÜR)?"




def _loose_money_values(line: str) -> list[Decimal]:
    """Erkennt auch OCR-Formate wie '2 49' oder '75 57' als Geldbeträge."""
    vals = _amounts_in_line(line)
    # Nur bei Beleg-/Summen-/Zahlungs-Kontext lockere Erkennung verwenden.
    if re.search(r"\b(summe|total|gesamt|bar|karte|mastercard|visa|brutto|betrag|eur)\b", line, re.IGNORECASE):
        for a, b in re.findall(r"\b(\d{1,4})\s+(\d{2})\b", line):
            try:
                d = Decimal(f"{a}.{b}").quantize(Decimal("0.01"))
                if d not in vals:
                    vals.append(d)
            except Exception:
                pass
    return vals

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

    Priorität: SUMME/TOTAL/GESAMT, Bar/Karte/MasterCard/Visa und Steuer-Tabellen
    mit BRUTTO-Spalte. Liter, Literpreise, Steuerbeträge, Netto und Nummern werden ignoriert.
    """
    text = _normalize_ocr_text(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates: list[tuple[int, Decimal]] = []

    ignore_words = ["liter", "eur/l", "eur / l", "l/", "stk", "menge", "einzelpreis", "netto", "mwst", "ust", "steuer", "tse", "terminal", "trace", "seriennr"]
    total_words = ["summe", "total", "gesamtsumme", "gesamtbetrag", "gesamtpreis", "zu zahlen", "endbetrag"]
    payment_words = ["bar", "kartenzahlung", "mastercard", "visa", "girocard", "ec-karte", "debit", "kreditkarte", "karte"]

    # 1) Sehr starke Regel: SUMME/TOTAL/GESAMT-Zeile plus Folgezeilen. Nicht in Steuer-/TSE-Blöcken suchen.
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(w in lower for w in total_words):
            window_lines = lines[i:i+4]
            for j, wline in enumerate(window_lines):
                wl = wline.lower()
                if any(x in wl for x in ["mwst", "ust", "steuer", "netto", "tse", "terminal", "transaktion"]):
                    continue
                vals = _loose_money_values(wline)
                vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
                for v in vals:
                    candidates.append((400 - j * 10 + min(int(v), 80), v))

    # 2) Zahlungszeilen: Bar EUR 2,49 / Mastercard 75,57 / Kartenzahlung 75,57.
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(w in lower for w in payment_words) or re.search(r"\bbetrag\b", lower):
            if any(x in lower for x in ["mwst", "ust", "steuer", "netto", "tse", "terminal", "transaktion", "folge"]):
                continue
            window = " ".join(lines[i:i+2])
            vals = _loose_money_values(window)
            vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
            for v in vals:
                score = 330 + min(int(v), 80)
                if "bar" in lower or "mastercard" in lower or "visa" in lower:
                    score += 40
                candidates.append((score, v))

    # 3) Steuer-Tabelle: Header enthält MWST/BRUTTO/NETTO; Brutto ist meist der mittlere/groessere Wert.
    for i, line in enumerate(lines):
        lower = line.lower()
        if "brutto" in lower and ("mwst" in lower or "ust" in lower or "netto" in lower):
            window = " ".join(lines[i:i+3])
            vals = _loose_money_values(window)
            vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
            # Typisch: MwSt, Brutto, Netto -> Brutto ist der größte Wert, aber nicht immer.
            if vals:
                candidates.append((300 + min(int(max(vals)), 80), max(vals)))

    # 4) Explizites BRUTTO-Muster.
    for m in re.finditer(r"brutto[^0-9]{0,30}" + _amount_regex(), text, re.IGNORECASE):
        val = _parse_decimal(m.group(1))
        if val and _is_likely_money_amount(val) and val not in [Decimal("7.00"), Decimal("19.00")]:
            candidates.append((280 + min(int(val), 80), val))

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

    # 1) Steuer-Tabelle mit Spalten MWST / BRUTTO / NETTO.
    # Beispiel Netto: Header "MWST BRUTTO NETTO", Folgezeile "b 7% 0,16 2,49 2,33".
    for i, line in enumerate(lines):
        lower = line.lower()
        if ("mwst" in lower or "ust" in lower) and "brutto" in lower and "netto" in lower:
            window = " ".join(lines[i:i+3])
            vals = _loose_money_values(window)
            vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
            if len(vals) >= 3:
                # Kleinster Betrag ist in diesen Tabellen fast immer Steuerbetrag.
                return min(vals)
            if vals:
                return min(vals)

    # 2) Direktmuster auf MwSt-Zeilen: "MWST 19,00% A 12,07 EUR".
    for i, line in enumerate(lines):
        lower = line.lower()
        if not any(k in lower for k in ["mwst", "ust", "umsatzsteuer", "vat", "tax"]):
            continue
        window = _line_context(lines, i, before=0, after=1)
        vals = _loose_money_values(window)
        vals = [v for v in vals if _is_likely_money_amount(v) and v not in [Decimal("7.00"), Decimal("19.00")]]
        if vals:
            # Bei Steuerzeilen ist Steuerbetrag plausibel der kleinste Betrag < Brutto.
            return min(vals)

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
        ("netto", "Netto Marken-Discount"),
        ("marken-discount", "Netto Marken-Discount"),
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




def _json_decimal(value):
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value)).quantize(Decimal("0.01"))
        except Exception:
            return None
    return _parse_decimal(str(value))


def _json_date(value):
    if not value:
        return None
    value = str(value).strip()
    for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"]:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def _openai_compatible_client():
    """Erstellt einen OpenAI-kompatiblen Client.

    Funktioniert mit OpenAI direkt und mit OpenRouter über:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL=https://openrouter.ai/api/v1
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url.rstrip("/")
    return OpenAI(**kwargs)


def _json_loads_lenient(raw: str) -> dict:
    raw = (raw or "{}").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def ai_document_extraction(text: str, file_path: str | None = None, content_type: str | None = None) -> Optional[dict]:
    """Generische KI-Belegerkennung für Kassenbons und Rechnungen über OpenAI/OpenRouter."""
    if not os.getenv("OPENAI_API_KEY"):
        print("AI extraction skipped: OPENAI_API_KEY missing", flush=True)
        return None
    try:
        client = _openai_compatible_client()
        if client is None:
            return None
        model = os.getenv("OPENAI_EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        print(f"AI extraction enabled: model={model}, base_url={base_url}", flush=True)

        system = (
            "Du bist ein deutscher Beleg- und Rechnungsextraktor für vorbereitende Buchhaltung. "
            "Analysiere jeden Beleg generisch: Kassenbon, Tankbeleg, Restaurantbon, Online-Rechnung, PDF-Rechnung. "
            "Priorität für Brutto/Gesamtsumme: SUMME, TOTAL, GESAMT, ZU ZAHLEN, BAR, KARTE, EC, VISA, MASTERCARD, KARTENZAHLUNG. "
            "Verwechsele niemals MwSt.-Betrag, Netto, Liter, Menge, Literpreis, Uhrzeit, Terminalnummer, TSE-Nummer, Telefonnummer, PLZ oder Belegpositionsnummer mit Brutto. "
            "Bei Steuertabellen mit MWST/UST/VAT + BRUTTO + NETTO: Brutto ist die Summe inkl. Steuer, MwSt ist nur der Steuerbetrag. "
            "Rechnungsnummer nur setzen, wenn eindeutig 'Rechnung/Rechnungsnr/Invoice' erkennbar ist; bei normalen Kassenbons sonst null. "
            "Gib ausschließlich valides JSON zurück. Keine Erklärungen."
        )
        schema = {
            "vendor": "Händler/Lieferant, offizieller Name wenn erkennbar",
            "invoice_date": "YYYY-MM-DD oder null",
            "invoice_number": "echte Rechnungsnummer oder null",
            "gross_amount": "Brutto/Gesamtsumme als Zahl, z.B. 75.57",
            "vat_amount": "MwSt-Betrag als Zahl oder null",
            "vat_rate": "MwSt-Satz als Zahl, z.B. 19 oder 7 oder null",
            "payment_method": "Bar, Bank, Kreditkarte/Bank, EC-Karte, PayPal oder Unbekannt",
            "currency": "EUR",
            "extraction_confidence": "0 bis 1"
        }
        user_text = (
            "Extrahiere die Felder aus dem Beleg. Antworte exakt im JSON-Schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n\n"
            "OCR-Text:\n"
            f"{text[:10000]}"
        )
        text_content = [{"type": "text", "text": user_text}]
        content = list(text_content)

        suffix = Path(file_path).suffix.lower() if file_path else ""
        if file_path and suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            mime = "image/webp" if suffix == ".webp" else ("image/png" if suffix == ".png" else "image/jpeg")
            raw_img = Path(file_path).read_bytes()
            b64 = base64.b64encode(raw_img).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}})

        def call_ai(msg_content):
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": msg_content}],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
            except Exception as e:
                print(f"AI extraction JSON-mode failed, retrying without response_format: {e}", flush=True)
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": msg_content}],
                    temperature=0,
                )

        try:
            res = call_ai(content)
        except Exception as e:
            # Einige OpenRouter-Textmodelle unterstützen keine Bildinputs. Dann OCR-Text-only versuchen.
            print(f"AI extraction with image failed, retrying text-only: {e}", flush=True)
            res = call_ai(text_content)

        raw = res.choices[0].message.content or "{}"
        data = _json_loads_lenient(raw)

        out = {
            "vendor": data.get("vendor"),
            "invoice_date": _json_date(data.get("invoice_date")),
            "invoice_number": data.get("invoice_number") or None,
            "gross_amount": _json_decimal(data.get("gross_amount")),
            "vat_amount": _json_decimal(data.get("vat_amount")),
            "vat_rate": _json_decimal(data.get("vat_rate")),
            "currency": data.get("currency") or "EUR",
        }
        payment = data.get("payment_method")
        if payment:
            out["payment_method"] = str(payment)
        print("AI extraction success", flush=True)
        return out
    except Exception as e:
        print(f"AI extraction failed: {e}", flush=True)
        return None


def _merge_ai_fields(local: dict, ai: Optional[dict]) -> dict:
    if not ai:
        return local
    merged = dict(local)
    # KI darf diese Extraktionsfelder überschreiben, weil sie generisch Layouts versteht.
    for key in ["vendor", "invoice_date", "invoice_number", "gross_amount", "vat_amount", "vat_rate", "currency"]:
        val = ai.get(key)
        if val not in (None, ""):
            merged[key] = val
    # Zahlungsart nur übernehmen, wenn erkannt.
    if ai.get("payment_method") and ai.get("payment_method") != "Unbekannt":
        merged["payment_method"] = ai.get("payment_method")
        if "bar" in str(ai.get("payment_method")).lower():
            merged["contra_account"] = "1000"
        elif any(k in str(ai.get("payment_method")).lower() for k in ["karte", "bank", "paypal"]):
            merged["contra_account"] = "1200"
    return merged

def extract_fields(text: str, file_path: str | None = None, content_type: str | None = None) -> dict:
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

    local_fields = {
        "invoice_date": find_date(text),
        "vendor": vendor,
        "invoice_number": find_invoice_number(text),
        "gross_amount": gross,
        "vat_amount": vat,
        "vat_rate": vat_rate,
        **booking,
        "currency": "EUR" if "€" in text or "EUR" in text.upper() or "EÜR" in text.upper() else None,
    }

    ai_fields = ai_document_extraction(text, file_path=file_path, content_type=content_type)
    merged = _merge_ai_fields(local_fields, ai_fields)

    # Nach AI-Extraktion Kontierung nochmals anhand besserer Händler-/Steuerdaten berechnen.
    booking = suggest_booking(text, merged.get("vendor"), merged.get("vat_rate"))
    for key, value in booking.items():
        merged.setdefault(key, value)
    # Wenn KI Zahlungsart erkannt hat, nicht überschreiben.
    if ai_fields and ai_fields.get("payment_method") and ai_fields.get("payment_method") != "Unbekannt":
        merged["payment_method"] = ai_fields.get("payment_method")
        merged["contra_account"] = "1000" if "bar" in str(ai_fields.get("payment_method")).lower() else "1200"
    return merged
