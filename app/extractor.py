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
    "januar": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3, "mrz": 3,
    "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6, "juli": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "oktober": 10, "okt": 10,
    "november": 11, "nov": 11, "dezember": 12, "dez": 12,
}

# ------------------------- OCR -------------------------
def _ocr_image_variants(file_path: str) -> str:
    from PIL import Image, ImageOps, ImageFilter, ImageEnhance
    import pytesseract

    img = Image.open(file_path)
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    scale = 2 if max(img.size) < 2400 else 1
    up = img.resize((img.width * scale, img.height * scale)) if scale > 1 else img.copy()
    gray = ImageOps.grayscale(up)
    gray = ImageOps.autocontrast(gray)
    sharp = ImageEnhance.Contrast(gray.filter(ImageFilter.SHARPEN)).enhance(1.9)
    bw = sharp.point(lambda p: 255 if p > 160 else 0)

    variants = [img, gray, sharp, bw]
    configs = ["--oem 3 --psm 6", "--oem 3 --psm 4", "--oem 3 --psm 11"]
    texts = []
    for variant in variants:
        for cfg in configs:
            try:
                txt = pytesseract.image_to_string(variant, lang="deu+eng", config=cfg).strip()
                if txt:
                    texts.append(txt)
            except Exception:
                continue

    seen, merged = set(), []
    for txt in texts:
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            key = re.sub(r"\s+", " ", line).lower()
            if key not in seen:
                seen.add(key)
                merged.append(line)
    return "\n".join(merged)


def extract_text(file_path: str, content_type: str) -> str:
    path = Path(file_path)
    if content_type == "application/pdf" or path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as exc:
            return f"PDF konnte nicht gelesen werden: {exc}"
    if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            return _ocr_image_variants(file_path).strip()
        except Exception as exc:
            return f"Bild-OCR ist lokal noch nicht verfügbar. Fehler: {exc}"
    return ""

# ------------------------- Normalisierung -------------------------
def _normalize_ocr_text(text: str) -> str:
    replacements = {
        "MIST": "MWST", "MWSI": "MWST", "MW5T": "MWST", "Mwst": "MwSt", "UST.": "USt",
        "€UR": "EUR", "EÜR": "EUR", "SROS5O": "BRUTTO", "0,0O": "0,00",
        "Rechnungsnr.": "Rechnungsnummer", "Re.-Nr.": "Rechnungsnummer",
    }
    out = text or ""
    for wrong, right in replacements.items():
        out = re.sub(re.escape(wrong), right, out, flags=re.IGNORECASE)
    return out


def _lines(text: str) -> list[str]:
    return [l.strip() for l in text.splitlines() if l.strip()]


def _parse_decimal(value: str) -> Optional[Decimal]:
    if value is None:
        return None
    value = str(value).strip().replace("€", "").replace("EUR", "").replace("EÜR", "")
    value = value.replace("O", "0").replace("o", "0")
    value = re.sub(r"[^0-9,.-]", "", value)
    if not value:
        return None
    # 1.234,56 oder 1234,56
    if "," in value:
        cleaned = value.replace(".", "").replace(",", ".")
    else:
        cleaned = value
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def _json_decimal(v) -> Optional[Decimal]:
    if v in (None, "", "null"):
        return None
    return _parse_decimal(str(v))


def _json_date(v):
    if not v or str(v).lower() == "null":
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).date()
        except Exception:
            pass
    return None


def _amount_regex() -> str:
    return r"(?:€\s*)?(\d{1,3}(?:\.\d{3})*,\d{2}|\d+[,\.]\d{2})\s*(?:€|EUR|EÜR)?"


def _is_date_like(line: str) -> bool:
    return bool(re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b", line))


def _amounts_in_line(line: str, loose: bool = False) -> list[Decimal]:
    vals = []
    money_re = re.compile(_amount_regex(), flags=re.IGNORECASE)
    for m in money_re.finditer(line):
        raw = m.group(1)
        # Datumsfragmente wie 27.04.2026 dürfen nicht als 27.04 EUR gewertet werden.
        before = line[max(0, m.start()-3):m.start()]
        after = line[m.end():m.end()+6]
        if re.search(r"\d[./-]$", before) or re.match(r"^[./-]\d{2,4}", after):
            continue
        # Telefon-/IBAN-/ID-Kontexte ignorieren, wenn kein EUR/€ oder Summenwort in der Nähe steht.
        ctx = line[max(0, m.start()-20):m.end()+20].lower()
        if any(k in ctx for k in ["tel", "fax", "iban", "hrb", "kundennummer"]) and not re.search(r"€|eur|betrag|summe|total|brutto", ctx):
            continue
        d = _parse_decimal(raw)
        if d is not None:
            vals.append(d)
    # Nur sehr kontrollierte lockere Erkennung für Bons: "2 49". Niemals in Datumszeilen.
    if loose and not _is_date_like(line) and re.search(r"\b(summe|total|gesamt|bar|karte|mastercard|visa|brutto|betrag|eur|zu zahlen)\b", line, re.IGNORECASE):
        for a, b in re.findall(r"\b(\d{1,4})\s+(\d{2})\b", line):
            try:
                d = Decimal(f"{a}.{b}").quantize(Decimal("0.01"))
                if d not in vals:
                    vals.append(d)
            except Exception:
                pass
    return vals


def _is_money(v: Decimal) -> bool:
    return v is not None and Decimal("0.01") <= v < Decimal("100000.00") and v not in (Decimal("7.00"), Decimal("19.00"))


def _clean_vendor_line(line: str) -> str:
    line = re.sub(r"[^A-Za-zÄÖÜäöüß0-9& .,'/-]", "", line)
    return re.sub(r"\s+", " ", line).strip(" ,;:-")

# ------------------------- Datum -------------------------
def find_date(text: str):
    text = _normalize_ocr_text(text)
    lines = _lines(text)
    # Label-Priorität: echte Rechnungs-/Belegdaten vor Zahlungs-/TSE-/Vertragsdaten.
    positive = ["rechnungsdatum", "datum", "belegdatum", "bon-datum", "kaufdatum", "ausgestellt"]
    negative = ["wird am", "eingezogen", "zahlungsziel", "fällig", "faellig", "tse-start", "tse-stop", "vertragsbeginn", "kündigung", "kuendigung", "mindestvertragslaufzeit"]
    date_patterns = [
        (r"\b(\d{2}\.\d{2}\.\d{4})\b", "%d.%m.%Y"),
        (r"\b(\d{1,2}\.\d{1,2}\.\d{2})\b", "%d.%m.%y"),
        (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
        (r"\b(\d{2}/\d{2}/\d{4})\b", "%d/%m/%Y"),
        (r"\b(\d{1,2}/\d{1,2}/\d{2})\b", "%d/%m/%y"),
    ]
    def parse_from(s):
        for pattern, fmt in date_patterns:
            m = re.search(pattern, s)
            if m:
                try:
                    return datetime.strptime(m.group(1), fmt).date()
                except ValueError:
                    pass
        m = re.search(r"\b(\d{1,2})\.?:?\s+([A-Za-zÄÖÜäöüß]+)\s+(\d{4})\b", s, re.I)
        if m:
            month = GERMAN_MONTHS.get(m.group(2).lower().replace("ä", "ae"))
            if month:
                try:
                    return datetime(int(m.group(3)), month, int(m.group(1))).date()
                except ValueError:
                    pass
        return None

    for line in lines:
        low = line.lower()
        if any(n in low for n in negative):
            continue
        if any(p in low for p in positive):
            d = parse_from(line)
            if d:
                return d
    for line in lines:
        low = line.lower()
        if any(n in low for n in negative):
            continue
        d = parse_from(line)
        if d:
            return d
    return None

# ------------------------- Rechnungsnummer -------------------------
def find_invoice_number(text: str) -> Optional[str]:
    text = _normalize_ocr_text(text)
    patterns = [
        r"(?:Rechnungsnummer|Rechnung\s*#|Rechnung\s*Nr\.?|Rechnungs\s*Nr\.?|Invoice\s*(?:No\.?|Number))[:\s#-]*([A-Z0-9][A-Z0-9\-/]{3,})",
        r"(?:Dokumentnummer|Belegnummer|Beleg-Nr\.?)[:\s#-]*([A-Z0-9][A-Z0-9\-/]{5,})",
    ]
    bad_context = ["terminal", "trace", "tse", "transaktion", "kartenzahlung", "seriennr", "verwendungszweck", "mandatsreferenz"]
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.I):
            # Nur die aktuelle Zeile prüfen. Eine Kundennummer in der Zeile davor darf die Rechnungsnummer nicht blockieren.
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            ctx = text[line_start:line_end].lower()
            if any(b in ctx for b in bad_context):
                continue
            candidate = m.group(1).strip().strip("# ,;:")
            if len(candidate) >= 4:
                return candidate
    return None

# ------------------------- Beträge -------------------------
def _amount_candidates_near_label(text: str, label_groups: list[tuple[list[str], int]]) -> list[tuple[int, Decimal, str]]:
    lines = _lines(text)
    candidates = []
    ignore = ["forderung aus vormonat", "netto", "mwst", "ust", "umsatzsteuer", "steuer", "liter", "eur/l", "einzelpreis", "menge", "stk", "tse", "terminal", "trace", "telefon", "iban"]
    for i, line in enumerate(lines):
        low = line.lower()
        for labels, base_score in label_groups:
            if any(lbl in low for lbl in labels):
                window_lines = lines[i:i+3]
                for j, wline in enumerate(window_lines):
                    wlow = wline.lower()
                    if any(x in wlow for x in ignore) and not any(lbl in wlow for lbl in labels):
                        continue
                    vals = _amounts_in_line(wline, loose=True)
                    for v in vals:
                        if _is_money(v):
                            candidates.append((base_score - j*20 + min(int(v), 50), v, wline))
    return candidates


def find_receipt_total(text: str) -> Optional[Decimal]:
    text = _normalize_ocr_text(text)
    label_groups = [
        (["zu zahlender betrag", "rechnungsbetrag", "zahlbetrag", "betrag fällig", "betrag faellig"], 1000),
        (["gesamtsumme", "gesamtbetrag", "gesamtpreis", "endsumme", "endbetrag"], 900),
        (["total", "summe", "zu zahlen"], 850),
        (["bar eur", "bar", "kartenzahlung", "mastercard", "visa", "girocard", "ec-karte", "kreditkarte", "karte"], 750),
        (["brutto"], 650),
    ]
    candidates = _amount_candidates_near_label(text, label_groups)
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][1]
    return None


def find_amount(text: str, labels: list[str]) -> Optional[Decimal]:
    groups = [([lbl.lower() for lbl in labels], 700)]
    c = _amount_candidates_near_label(text, groups)
    if c:
        c.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return c[0][1]
    return None


def find_largest_amount(text: str) -> Optional[Decimal]:
    text = _normalize_ocr_text(text)
    vals: list[Decimal] = []
    hard_ignore = ["telefon", "tel", "fax", "iban", "bic", "ust.id", "hrb", "plz", "kundennummer", "verwendungszweck", "mandatsreferenz", "tse", "terminal", "trace", "liter", "eur/l", "menge", "stk", "einzelpreis"]
    for line in _lines(text):
        low = line.lower()
        if _is_date_like(line) or any(k in low for k in hard_ignore):
            continue
        for v in _amounts_in_line(line):
            if _is_money(v):
                vals.append(v)
    return max(vals) if vals else None

# ------------------------- MwSt -------------------------
def find_vat_rate(text: str) -> Optional[Decimal]:
    text = _normalize_ocr_text(text)
    # Erst nahe Steuer-Keywords.
    patterns = [
        r"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{0,60}?(\d{1,2}(?:[,\.]\d{1,2})?)\s*%",
        r"(?:Steuersatz|USt-Satz|MwSt-Satz)[^\n]{0,40}?(\d{1,2}(?:[,\.]\d{1,2})?)\s*%",
        r"\b(7|19)\s*(?:,00|\.00)?\s*%",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            try:
                val = Decimal(m.group(1).replace(",", ".")).quantize(Decimal("0.01"))
                if val in (Decimal("7.00"), Decimal("19.00"), Decimal("0.00")):
                    return val
            except Exception:
                pass
    return None


def find_vat(text: str) -> Optional[Decimal]:
    text = _normalize_ocr_text(text)
    lines = _lines(text)
    # 1) Vodafone/Telekom-Style: Umsatzsteuer 19% 1,91
    for line in lines:
        low = line.lower()
        if any(k in low for k in ["umsatzsteuer", "mwst", "ust", "vat", "tax"]):
            vals = [v for v in _amounts_in_line(line, loose=False) if _is_money(v)]
            # In Steuerzeile ist meist der kleinste Geldbetrag die Steuer. Prozentwerte sind gefiltert.
            if vals:
                return min(vals)
    # 2) Steuertabelle: MWST BRUTTO NETTO / b 7% 0,16 2,49 2,33
    for i, line in enumerate(lines):
        low = line.lower()
        if ("mwst" in low or "ust" in low) and ("brutto" in low or "netto" in low):
            window = " ".join(lines[i:i+3])
            vals = [v for v in _amounts_in_line(window, loose=True) if _is_money(v)]
            if vals:
                return min(vals)
    # 3) Fallback
    m = re.search(r"(?:MwSt\.?|USt\.?|Umsatzsteuer|VAT|Tax)[^\n]{0,80}" + _amount_regex(), text, re.I)
    if m:
        val = _parse_decimal(m.group(1))
        if val and _is_money(val):
            return val
    return None

# ------------------------- Lieferant/Zahlung -------------------------
def find_vendor(text: str) -> Optional[str]:
    text = _normalize_ocr_text(text)
    raw_lines = _lines(text)
    lines = [_clean_vendor_line(l) for l in raw_lines if _clean_vendor_line(l)]
    full = "\n".join(lines).lower()
    known = [
        ("vodafone", "Vodafone West GmbH"), ("telekom", "Deutsche Telekom"), ("o2", "O2"),
        ("netto", "Netto Marken-Discount"), ("marken-discount", "Netto Marken-Discount"),
        ("hem", "HEM Tankstelle"), ("aral", "ARAL"), ("shell", "Shell"), ("esso", "Esso"), ("jet", "JET"),
        ("totalenergies", "TotalEnergies"), ("avia", "AVIA"), ("edeka", "EDEKA"), ("rewe", "REWE"),
        ("lidl", "Lidl"), ("aldi", "ALDI"), ("kaufland", "Kaufland"), ("dm-drogerie", "dm-drogerie markt"),
        ("rossmann", "Rossmann"), ("ikea", "IKEA"), ("amazon", "Amazon"),
    ]
    for needle, name in known:
        if needle in full:
            return name
    legal_forms = ["gmbh", "ug", "gbr", "ag", "kg", "ohg", "e.k", "mbh", "ltd"]
    for line in lines[:35]:
        if any(form in line.lower() for form in legal_forms):
            return line
    bad = ["rechnung", "invoice", "datum", "seite", "betrag", "summe", "total", "kundenbeleg", "terminal", "beleg", "www", ".de", ".com", "straße", "strasse", "telefon", "tel", "fax", "ust", "steuer", "iban", "bic", "bon", "kasse", "kundenservice", "käufer", "zahlungsmethode"]
    for line in lines[:10]:
        low = line.lower()
        if 3 <= len(line) <= 60 and not any(k in low for k in bad) and len(re.findall(r"\d", line)) <= 4 and re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            return line.strip(" ,")
    return None


def detect_payment_method(text: str) -> tuple[Optional[str], Optional[str]]:
    low = (text or "").lower()
    if any(k in low for k in ["bar eur", "barzahlung", " bar ", "cash"]):
        return "Bar", "1000"
    if any(k in low for k in ["lastschrift", "eingezogen", "iban", "sepa"]):
        return "Lastschrift/Bank", "1200"
    if any(k in low for k in ["kreditkarte", "mastercard", "visa", "girocard", "ec-karte", "kartenzahlung", "debit"]):
        return "Kreditkarte/Bank", "1200"
    if "paypal" in low:
        return "PayPal", "1200"
    return None, None

# ------------------------- KI -------------------------
def _openai_compatible_client():
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
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end+1]
    return json.loads(raw)


def ai_document_extraction(text: str, file_path: str | None = None, content_type: str | None = None) -> Optional[dict]:
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
            "Du bist ein extrem genauer deutscher Belegextraktor für vorbereitende Buchhaltung. "
            "Extrahiere Rechnungen und Kassenbons. Wichtig: Datumswerte wie 27.04.2026 dürfen NIEMALS als Geldbetrag 2704.00 interpretiert werden. "
            "Brutto/Gesamtsumme nur aus expliziten Feldern wie Rechnungsbetrag, Zu zahlender Betrag, Gesamtbetrag, Gesamtsumme, SUMME, TOTAL, Bar, Karte, Mastercard, Visa. "
            "Ignoriere Telefonnummern, IBAN, Kundennummern, Terminalnummern, Trace, TSE, Vertragsdaten, Liter, Menge, Einzelpreise, Steuersätze und Netto, wenn Brutto gesucht ist. "
            "Bei Steuerzeilen ist Umsatzsteuer/MwSt der Steuerbetrag, nicht Brutto. Gib ausschließlich JSON zurück."
        )
        schema = {
            "vendor": "Lieferant/Händler",
            "invoice_date": "YYYY-MM-DD oder null; echtes Rechnungs-/Belegdatum, nicht Zahlungsdatum",
            "invoice_number": "echte Rechnungsnummer, sonst null bei Kassenbons",
            "gross_amount": "Brutto/Gesamtsumme als Zahl",
            "vat_amount": "MwSt/USt-Betrag als Zahl oder null",
            "vat_rate": "MwSt-Satz als Zahl, z.B. 19 oder 7 oder null",
            "payment_method": "Bar, Lastschrift/Bank, Kreditkarte/Bank, EC-Karte, PayPal oder Unbekannt",
            "currency": "EUR",
            "extraction_confidence": "0 bis 1"
        }
        user_text = "Extrahiere exakt dieses JSON-Schema:\n" + json.dumps(schema, ensure_ascii=False) + "\n\nOCR-Text:\n" + text[:12000]
        text_content = [{"type": "text", "text": user_text}]
        content = list(text_content)
        suffix = Path(file_path).suffix.lower() if file_path else ""
        if file_path and suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            mime = "image/webp" if suffix == ".webp" else ("image/png" if suffix == ".png" else "image/jpeg")
            b64 = base64.b64encode(Path(file_path).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"}})
        def call(msg_content):
            try:
                return client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": msg_content}], temperature=0, response_format={"type": "json_object"})
            except Exception as e:
                print(f"AI extraction JSON-mode failed, retrying without response_format: {e}", flush=True)
                return client.chat.completions.create(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": msg_content}], temperature=0)
        try:
            res = call(content)
        except Exception as e:
            print(f"AI extraction with image failed, retrying text-only: {e}", flush=True)
            res = call(text_content)
        data = _json_loads_lenient(res.choices[0].message.content or "{}")
        out = {
            "vendor": data.get("vendor"),
            "invoice_date": _json_date(data.get("invoice_date")),
            "invoice_number": data.get("invoice_number") or None,
            "gross_amount": _json_decimal(data.get("gross_amount")),
            "vat_amount": _json_decimal(data.get("vat_amount")),
            "vat_rate": _json_decimal(data.get("vat_rate")),
            "currency": data.get("currency") or "EUR",
        }
        if data.get("payment_method"):
            out["payment_method"] = str(data.get("payment_method"))
        print("AI extraction success", flush=True)
        return out
    except Exception as e:
        print(f"AI extraction failed: {e}", flush=True)
        return None


def _close_money(a: Optional[Decimal], b: Optional[Decimal]) -> bool:
    if a is None or b is None:
        return False
    tolerance = max(Decimal("1.00"), abs(a) * Decimal("0.10"))
    return abs(a - b) <= tolerance


def _merge_ai_fields(local: dict, ai: Optional[dict]) -> dict:
    if not ai:
        return local
    merged = dict(local)
    # Textfelder: KI darf ergänzen, aber lokale klare Anbieter/Rechnungsnummer behalten.
    for key in ["vendor", "invoice_date", "invoice_number", "currency"]:
        if not merged.get(key) and ai.get(key):
            merged[key] = ai.get(key)
    # Geldfelder: KI darf nur überschreiben, wenn lokal leer oder nah am lokalen Wert. Verhindert 27.04.2026 -> 2704.00.
    for key in ["gross_amount", "vat_amount", "vat_rate"]:
        aval, lval = ai.get(key), merged.get(key)
        if aval is None:
            continue
        if lval is None or _close_money(lval, aval):
            merged[key] = aval
        else:
            print(f"AI field rejected by plausibility: {key} local={lval} ai={aval}", flush=True)
    if ai.get("payment_method") and ai.get("payment_method") != "Unbekannt":
        merged["payment_method"] = ai.get("payment_method")
        merged["contra_account"] = "1000" if "bar" in str(ai.get("payment_method")).lower() else "1200"
    return merged

# ------------------------- Kontierung/Extraktion Gesamt -------------------------
def suggest_booking(text: str, vendor: Optional[str], vat_rate: Optional[Decimal]) -> dict:
    rule_result = rule_based_skr03(text, vendor, vat_rate)
    ai_result = ai_skr03_suggestion(text, vendor, None, None, None, vat_rate)
    if ai_result:
        rule_result.update(ai_result)
    return rule_result


def extract_fields(text: str, file_path: str | None = None, content_type: str | None = None) -> dict:
    text = _normalize_ocr_text(text)
    gross = find_receipt_total(text)
    if gross is None:
        gross = find_amount(text, ["Gesamtsumme", "Gesamtbetrag", "Gesamtpreis", "Bruttobetrag", "Rechnungssumme", "Rechnungsbetrag", "Zu zahlender Betrag", "Amount due", "TOTAL", "Summe", "Betrag"])
    if gross is None:
        gross = find_largest_amount(text)

    vat_rate = find_vat_rate(text)
    vat = find_vat(text)
    if gross is not None and vat_rate is not None:
        expected = None
        if vat_rate == Decimal("19.00"):
            expected = (gross * Decimal("19") / Decimal("119")).quantize(Decimal("0.01"))
        elif vat_rate == Decimal("7.00"):
            expected = (gross * Decimal("7") / Decimal("107")).quantize(Decimal("0.01"))
        if expected is not None and (vat is None or vat <= 0 or vat >= gross or abs(vat - expected) > max(Decimal("0.20"), gross * Decimal("0.03"))):
            vat = expected

    vendor = find_vendor(text)
    payment, contra = detect_payment_method(text)
    booking = suggest_booking(text, vendor, vat_rate)
    if payment:
        booking["payment_method"] = payment
    if contra:
        booking["contra_account"] = contra

    local_fields = {
        "invoice_date": find_date(text),
        "vendor": vendor,
        "invoice_number": find_invoice_number(text),
        "gross_amount": gross,
        "vat_amount": vat,
        "vat_rate": vat_rate,
        **booking,
        "currency": "EUR" if "€" in text or "EUR" in text.upper() else None,
    }
    ai_fields = ai_document_extraction(text, file_path=file_path, content_type=content_type)
    merged = _merge_ai_fields(local_fields, ai_fields)

    # Nach finalen Extraktionswerten Kontierung nochmal sauber berechnen.
    final_booking = rule_based_skr03(text, merged.get("vendor"), merged.get("vat_rate"))
    # vorhandene Zahlungsart aus Extraktion behalten
    if merged.get("payment_method"):
        final_booking["payment_method"] = merged["payment_method"]
        final_booking["contra_account"] = "1000" if "bar" in str(merged["payment_method"]).lower() else "1200"
    merged.update(final_booking)
    return merged
