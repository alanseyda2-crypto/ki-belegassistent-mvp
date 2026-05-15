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
        "BRUTT0": "BRUTTO", "BRUTTD": "BRUTTO", "NEITD": "NETTO", "NETTD": "NETTO",
        "Nefto": "Netto", "Neft0": "Netto", "Nctto": "Netto", "NettO": "Netto", "NEITO": "NETTO",
        "MUST": "MWST", "MUSI": "MWST", "MW5I": "MWST", "MWST.": "MWST",
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
def _safe_date_from_parts(day, month, year):
    try:
        d = int(str(day).replace("O", "0").replace("o", "0"))
        m = int(str(month).replace("O", "0").replace("o", "0"))
        y = int(str(year).replace("O", "0").replace("o", "0"))
        if y < 100:
            y += 2000 if y <= 69 else 1900
        if not (2000 <= y <= 2099 and 1 <= m <= 12 and 1 <= d <= 31):
            return None
        return datetime(y, m, d).date()
    except Exception:
        return None


def _date_candidates(text: str) -> list[tuple[int, object, str]]:
    """Robuste Datums-Kandidaten mit Scoring.
    Ziel: echtes Beleg-/Rechnungsdatum vor Zahlungs-/TSE-/Vertragsdatum.
    Erkennt auch Kassenzettel-Schreibweisen unten rechts: 14:37 27 04 23.
    """
    text = _normalize_ocr_text(text or "")
    lines = _lines(text)
    candidates: list[tuple[int, object, str]] = []
    positive = [
        "rechnungsdatum", "rechnung datum", "datum", "belegdatum", "bon-datum", "kaufdatum", "ausgestellt",
        "rechnung vom", "invoice date", "date"
    ]
    negative = [
        "wird am", "eingezogen", "zahlungsziel", "fällig", "faellig", "zahlbar", "lastschrift am",
        "tse-start", "tse stop", "tse-stop", "vertragsbeginn", "kündigung", "kuendigung",
        "mindestvertragslaufzeit", "lieferdatum", "leistungszeitraum", "bis ", "von "
    ]

    def score_line(line: str, index: int) -> int:
        low = line.lower()
        score = 100
        if any(p in low for p in positive):
            score += 500
        if re.search(r"\b\d{1,2}:\d{2}\b", line):
            score += 220
        if any(k in low for k in ["kasse", "bon", "steuer-nr", "tse", "transaktion", "summe", "bar eur"]):
            score += 120
        # bei Kassenbons steht das Datum häufig in den letzten Zeilen
        if index >= max(0, len(lines) - 8):
            score += 140
        if any(n in low for n in negative):
            score -= 500
        return score

    for idx, line in enumerate(lines):
        low = line.lower()
        base = score_line(line, idx)
        # dd.mm.yy/yyyy oder dd/mm/yyyy; OCR erlaubt O statt 0
        for m in re.finditer(r"\b([0-3]?\d)\s*[./,;:-]\s*([O0]?[1-9]|1[0-2])\s*[./,;:-]\s*(20\d{2}|\d{2})\b", line, re.I):
            d = _safe_date_from_parts(m.group(1), m.group(2), m.group(3))
            if d:
                candidates.append((base + 80, d, line))
        # yyyy-mm-dd
        for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", line):
            d = _safe_date_from_parts(m.group(3), m.group(2), m.group(1))
            if d:
                candidates.append((base + 70, d, line))
        # 14:37 27 04 23 oder 7507 ... 14:37 27 04 23
        for m in re.finditer(r"(?:\b\d{1,2}:\d{2}\b\s*)\b([0-3]?\d)\s+([O0]?[1-9]|1[0-2])\s+(20\d{2}|\d{2})\b", line, re.I):
            d = _safe_date_from_parts(m.group(1), m.group(2), m.group(3))
            if d:
                candidates.append((base + 260, d, line))
        # Nur bei starkem Bon-/Zeitkontext: dd mm yy ohne Trennzeichen
        if re.search(r"\b(\d{1,2}:\d{2}|kasse|bon|tse|steuer-nr|transaktion)\b", line, re.I):
            for m in re.finditer(r"\b([0-3]?\d)\s+([O0]?[1-9]|1[0-2])\s+(20\d{2}|\d{2})\b", line, re.I):
                d = _safe_date_from_parts(m.group(1), m.group(2), m.group(3))
                if d:
                    candidates.append((base + 160, d, line))
        # 08. Mai 2026
        for m in re.finditer(r"\b(\d{1,2})\.?\s+([A-Za-zÄÖÜäöüß]+)\s+(20\d{2})\b", line, re.I):
            month = GERMAN_MONTHS.get(m.group(2).lower().replace("ä", "ae"))
            if month:
                d = _safe_date_from_parts(m.group(1), month, m.group(3))
                if d:
                    candidates.append((base + 120, d, line))

    # Zusätzlicher globaler Fallback: Datum unten rechts/letzte Zeile mit Uhrzeit, auch wenn OCR alles in eine Zeile klebt.
    tail = " ".join(lines[-10:])
    for m in re.finditer(r"\b\d{1,2}:\d{2}\b\s+([0-3]?\d)\s*[./,;:\- ]\s*([O0]?[1-9]|1[0-2])\s*[./,;:\- ]\s*(20\d{2}|\d{2})\b", tail, re.I):
        d = _safe_date_from_parts(m.group(1), m.group(2), m.group(3))
        if d:
            candidates.append((700, d, "tail-time-date"))
    return candidates


def find_date(text: str):
    candidates = _date_candidates(text)
    if not candidates:
        return None
    # Höchster Score gewinnt. Bei Gleichstand frühere echte Rechnungsdatumszeile vor Zahlungsdatum.
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

# ------------------------- Rechnungsnummer -------------------------
def find_invoice_number(text: str) -> Optional[str]:
    text = _normalize_ocr_text(text)
    receipt_like = _looks_like_receipt(text)
    patterns = [
        r"(?:Rechnungsnummer|Rechnung\s*#|Rechnung\s*Nr\.?|Rechnungs\s*Nr\.?|Invoice\s*(?:No\.?|Number))[:\s#-]*([A-Z0-9][A-Z0-9\-/]{3,})",
        # Belegnummer nur bei Nicht-Kassenbons verwenden; bei Bons sind das oft Terminal/TSE/Transaktionsnummern.
    ]
    if not receipt_like:
        patterns.append(r"(?:Dokumentnummer|Belegnummer|Beleg-Nr\.?)[:\s#-]*([A-Z0-9][A-Z0-9\-/]{5,})")
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




def _looks_like_receipt(text: str) -> bool:
    low = (text or "").lower()
    return any(k in low for k in [
        "summe", "total", "bar eur", "kartenzahlung", "mastercard", "visa", "girocard",
        "tse", "terminal", "kasse", "kundenbeleg", "bon", "brutto", "netto", "mwst"
    ])


def _safe_amounts_after_keyword(text: str, keywords: list[str], max_chars: int = 120) -> list[tuple[int, Decimal, str]]:
    """Findet Beträge direkt nach starken Summen-/Zahlungswörtern.
    Robust gegen Datumswerte, TSE-Zeiten, Mengen, Liter und Nummern.
    """
    out: list[tuple[int, Decimal, str]] = []
    flat = re.sub(r"[ \t]+", " ", _normalize_ocr_text(text or ""))
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.I)
        for m in pattern.finditer(flat):
            snippet = flat[m.start():m.end() + max_chars]
            low = snippet.lower()
            if any(bad in low for bad in ["mwst", "umsatzsteuer", "ust ", " ust", "netto für", "steuersatz"]):
                # Summe/Total-Zeilen dürfen bleiben; reine Steuer-/Netto-Kontexte nicht.
                if not any(good in low for good in ["summe", "total", "bar eur", "karte", "rechnungsbetrag", "zu zahlender"]):
                    continue
            vals = []
            # Formate 2,49 / 2.49 / €2,49 / 2 49 nach starkem Keyword
            vals.extend(_amounts_in_line(snippet, loose=True))
            # OCR bei Bons trennt Dezimalstellen oft als Leerzeichen, z.B. "SUMME [T] 2 49".
            if not vals:
                for a,b in re.findall(r"\b(\d{1,4})\s+(\d{2})\b", snippet):
                    try:
                        vals.append(Decimal(f"{a}.{b}").quantize(Decimal("0.01")))
                    except Exception:
                        pass
            for v in vals:
                if _is_money(v):
                    # Starke Keywords gewinnen; kleinere Werte werden bei Kassenbons NICHT schlechter bewertet.
                    score = 1000
                    if kw.lower() in ["rechnungsbetrag", "zu zahlender betrag", "gesamtsumme", "gesamtbetrag"]:
                        score += 200
                    if kw.lower() in ["summe", "total", "bar eur"]:
                        score += 150
                    out.append((score, v, snippet[:160]))
    return out


def _receipt_tax_table(text: str) -> tuple[Optional[Decimal], Optional[Decimal], Optional[Decimal], Optional[Decimal]]:
    """Erkennt typische Kassenbon-Steuertabellen:
    MWST BRUTTO NETTO
    b 7% 0,16 2,49 2,33
    Rückgabe: (vat_rate, vat_amount, gross, net)
    """
    text = _normalize_ocr_text(text)
    lines = _lines(text)
    for i, line in enumerate(lines):
        low = line.lower()
        if ("mwst" in low or "ust" in low) and "brutto" in low and "netto" in low:
            window = " ".join(lines[i:i+4])
            # Typisch: b 7% 0,16 2,49 2,33 oder b 73 0.16 2.49 2.33 (OCR)
            m = re.search(r"\b(?:[a-z]\s*)?(7|19|73|1900|700|19,00|7,00)\s*%?\s+" + _amount_regex() + r"\s+" + _amount_regex() + r"\s+" + _amount_regex(), window, re.I)
            if m:
                raw_rate = m.group(1).replace(",", ".")
                if raw_rate in ["73", "700"]:
                    rate = Decimal("7.00")
                elif raw_rate in ["1900"]:
                    rate = Decimal("19.00")
                else:
                    rate = Decimal(raw_rate).quantize(Decimal("0.01"))
                nums = [_parse_decimal(m.group(2)), _parse_decimal(m.group(3)), _parse_decimal(m.group(4))]
                nums = [n for n in nums if n is not None]
                if len(nums) >= 3:
                    vat, gross, net = nums[0], nums[1], nums[2]
                    if vat < gross and net < gross:
                        return rate, vat, gross, net
            # Fallback: Nach Header alle Geldwerte; Reihenfolge ist meist Steuer, Brutto, Netto.
            vals = [v for v in _amounts_in_line(window, loose=True) if _is_money(v)]
            if len(vals) >= 3:
                # Prozentwerte 7/19 werden durch _is_money teils entfernt, bei OCR 73 ignorieren wir.
                vals_sorted = vals[:3]
                vat, gross, net = vals_sorted[0], vals_sorted[1], vals_sorted[2]
                if vat < gross and net < gross:
                    # Satz aus der Nähe erkennen; wenn Brutto-Netto ungefähr 7/19 passt, ableiten.
                    diff = gross - net
                    rate = None
                    if abs(diff - (gross * Decimal("7") / Decimal("107")).quantize(Decimal("0.01"))) <= Decimal("0.05"):
                        rate = Decimal("7.00")
                    elif abs(diff - (gross * Decimal("19") / Decimal("119")).quantize(Decimal("0.01"))) <= Decimal("0.10"):
                        rate = Decimal("19.00")
                    return rate, vat, gross, net
    return None, None, None, None

def find_receipt_total(text: str) -> Optional[Decimal]:
    text = _normalize_ocr_text(text)
    # 1) Strukturierte Kassenbon-Steuertabelle: Brutto aus Tabelle ist oft zuverlässiger als OCR-Zufall.
    _rate, _vat, table_gross, _net = _receipt_tax_table(text)
    # 2) Starke Summen-/Zahlungswörter direkt auswerten.
    strong = _safe_amounts_after_keyword(text, [
        "zu zahlender betrag", "rechnungsbetrag", "gesamtsumme", "gesamtbetrag", "gesamtpreis",
        "total", "summe", "bar eur", "kartenzahlung", "mastercard", "visa", "girocard", "ec-karte"
    ])
    if strong:
        # Wenn ein Wert doppelt vorkommt, z.B. SUMME und Bar EUR, ist er sehr plausibel.
        counts = {}
        for _, v, _ in strong:
            counts[v] = counts.get(v, 0) + 1
        strong.sort(key=lambda x: (counts.get(x[1], 0), x[0], -abs(x[1] - (table_gross or x[1]))), reverse=True)
        return strong[0][1]
    if table_gross is not None:
        return table_gross
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
    table_rate, _table_vat, _table_gross, _table_net = _receipt_tax_table(text)
    if table_rate is not None:
        return table_rate
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
    _table_rate, table_vat, _table_gross, _table_net = _receipt_tax_table(text)
    if table_vat is not None:
        return table_vat
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

MERCHANT_BLACKLIST_WORDS = [
    "rechnung", "invoice", "datum", "seite", "betrag", "summe", "total", "kundenbeleg", "terminal",
    "beleg", "www", "telefon", "tel", "fax", "ust", "steuer", "iban", "bic", "bon", "kasse",
    "kundenservice", "käufer", "zahlungsmethode", "transaktion", "tse", "seriennr", "signatur",
    "signaturzähler", "pruefwert", "prüfwert", "trace", "kartenzahlung", "mastercard", "visa",
    "brutto", "netto", "mwst", "ust.", "eur", "bar", "karte", "kreditkarte", "debit", "uhr",
    "menge", "stk", "preis", "einzelpreis", "zwischensumme", "postfach", "kundennummer",
]

MERCHANT_ADDRESS_WORDS = [
    "straße", "strasse", "str.", "allee", "platz", "weg", "gasse", "damm", "ring", "chaussee",
    "berlin", "kerpen", "dresden", "köln", "koeln", "hamburg", "münchen", "muenchen", "deutschland",
]

# Händlerlexikon mit OCR-Varianten. Das ist nicht als einzelner Sonderfall gedacht,
# sondern als Merchant-Classification-Layer: bekannte Marken gewinnen gegen technische TSE-/Kassenzeilen.
KNOWN_MERCHANTS = [
    ("Netto Marken-Discount", [r"\bnetto\b", r"\bnett[o0]\b", r"\bne[t7][t7][o0]\b", r"\bneft[o0]\b", r"\bnert[o0]\b", r"marken[\s\.-]*discount", r"netto[\s\.-]*online"]),
    ("HEM Tankstelle", [r"\bhem\b", r"hem[\s\.-]*tank", r"hem[\s\.-]*tankstelle"]),
    ("Vodafone West GmbH", [r"vodafone", r"v[o0]daf[o0]ne"]),
    ("Deutsche Telekom", [r"deutsche\s+telekom", r"\btelekom\b", r"t[-\s]*mobile"]),
    ("O2", [r"\bo2\s*(shop|store|telefonica|germany|rechnung)\b", r"\btelefonica\b"]),
    ("ARAL", [r"\baral\b"]), ("Shell", [r"\bshell\b"]), ("Esso", [r"\besso\b"]),
    ("JET Tankstelle", [r"\bjet\s+tank", r"\bjet\b"]),
    ("TotalEnergies", [r"total\s*energies", r"totalenergies"]), ("AVIA", [r"\bavia\b"]),
    ("REWE", [r"\brewe\b"]), ("EDEKA", [r"\bedeka\b"]), ("Lidl", [r"\blidl\b"]),
    ("ALDI", [r"\baldi\b"]), ("Kaufland", [r"\bkaufland\b"]), ("PENNY", [r"\bpenny\b"]),
    ("NORMA", [r"\bnorma\b"]), ("METRO", [r"\bmetro\b"]),
    ("dm-drogerie markt", [r"dm[\s\.-]*drogerie", r"\bdm\s+markt\b"]),
    ("Rossmann", [r"rossmann"]), ("Amazon", [r"amazon"]),
    ("OBI", [r"\bobi\b"]), ("Hornbach", [r"hornbach"]), ("BAUHAUS", [r"bauhaus"]),
    ("IKEA", [r"\bikea\b"]), ("McDonald's", [r"mcdonald", r"mc donald"]),
    ("Burger King", [r"burger\s+king"]), ("Starbucks", [r"starbucks"]),
]


def _merchant_match_text(s: str) -> str:
    """OCR-normalisierte Zeichenkette nur fürs Händler-Matching."""
    s = (s or "").lower()
    s = s.replace("0", "o").replace("1", "l").replace("|", "l")
    s = s.replace("€", "e")
    s = re.sub(r"[^a-zäöüß0-9 .\-/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _bad_vendor_candidate(line: str, *, receipt_like: bool = False) -> bool:
    low = line.lower().strip()
    if not low or len(low) < 2:
        return True
    if any(w in low for w in MERCHANT_BLACKLIST_WORDS):
        # Ausnahme: Netto als Markenname darf nicht durch Wort "netto" als Rechnungsfeld geblockt werden,
        # wenn die Zeile wirklich wie ein Händlerkopf aussieht.
        if not re.search(r"\bnetto\b|marken[\s\.-]*discount", low, re.I):
            return True
    if any(w in low for w in MERCHANT_ADDRESS_WORDS):
        return True
    digit_count = len(re.findall(r"\d", line))
    alpha_count = len(re.findall(r"[A-Za-zÄÖÜäöüß]", line))
    if digit_count > max(3, alpha_count):
        return True
    if re.search(r"\b\d{4,}\b", line):
        return True
    if re.search(r"\b\d{1,2}[:.]\d{2}\b", line):
        return True
    if receipt_like and len(line) > 45:
        return True
    return False


def _known_merchant_candidates(lines: list[str]) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    joined_all = " ".join(_merchant_match_text(l) for l in lines[:40])
    joined_head = " ".join(_merchant_match_text(l) for l in lines[:10])

    # Sehr starke globale Händler-Indikatoren. Diese gewinnen gegen zufällige OCR-Tokens.
    if re.search(r"netto[\s\.-]*online|marken[\s\.-]*discount|\bnetto\b", joined_head, re.I) or re.search(r"netto[\s\.-]*online|marken[\s\.-]*discount", joined_all, re.I):
        candidates.append((2600, "Netto Marken-Discount", "global:netto"))
    if re.search(r"hem[\s\.-]*tank|hem[\s\.-]*tankstelle|\bhem\b.{0,25}tank", joined_head, re.I):
        candidates.append((2600, "HEM Tankstelle", "global:hem-tank"))
    if re.search(r"vodafone\s+west|vodafone", joined_head, re.I) or re.search(r"vodafone\s+west\s+gmbh", joined_all, re.I):
        candidates.append((2500, "Vodafone West GmbH", "global:vodafone"))

    # Kopfbereich stärker gewichten; Footer/TSE-Zeilen sehr schwach.
    # WICHTIG: kurze Marken wie HEM/O2 dürfen nicht fuzzy/zufällig irgendwo im Bon matchen.
    for idx, line in enumerate(lines[:30]):
        norm = _merchant_match_text(line)
        joined_next = norm
        if idx + 1 < len(lines):
            joined_next += " " + _merchant_match_text(lines[idx + 1])
        for merchant, patterns in KNOWN_MERCHANTS:
            for pat in patterns:
                if not re.search(pat, joined_next, re.I):
                    continue

                # HEM nur akzeptieren, wenn es im Kopfbereich steht und Tankstellen-Kontext hat.
                # Sonst können zufällige OCR-Fragmente aus TSE/Prüfwert als HEM fehlinterpretiert werden.
                if merchant == "HEM Tankstelle":
                    if idx > 8:
                        continue
                    if not re.search(r"hem|tankstelle|tank|diesel|säulen|saeulen", joined_next, re.I):
                        continue
                    if re.search(r"netto|marken\s*discount|netto\s*online", joined_all, re.I) and not re.search(r"hem\s*tank|tankstelle", joined_head, re.I):
                        continue

                # O2 nur mit echtem Telekommunikations-Kontext akzeptieren, nie aus TSE-Müll.
                if merchant == "O2" and not re.search(r"o2\s*(shop|store|rechnung)|telefonica|mobilfunk|dsl|internet", joined_next, re.I):
                    continue

                score = 1300 - idx * 18
                # Markenlogo in den ersten 5 Zeilen ist extrem stark.
                if idx <= 5:
                    score += 250
                # TSE-/Kassen-Umfeld darf bekannte Händler nicht fälschen.
                if any(b in norm for b in ["tse", "serien", "kasse", "transaktion", "terminal", "pruefwert", "prüfwert", "signatur"]):
                    score -= 900
                candidates.append((score, merchant, f"known:{pat}:{line}"))
    return candidates

def _fuzzy_brand_candidates(lines: list[str]) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    try:
        from difflib import SequenceMatcher
        brand_words = {
            "netto": "Netto Marken-Discount", "hem": "HEM Tankstelle", "vodafone": "Vodafone West GmbH",
            "telekom": "Deutsche Telekom", "edeka": "EDEKA", "rewe": "REWE", "lidl": "Lidl",
            "aldi": "ALDI", "kaufland": "Kaufland", "rossmann": "Rossmann", "aral": "ARAL",
            "shell": "Shell", "esso": "Esso", "penny": "PENNY", "norma": "NORMA", "metro": "METRO",
            "hornbach": "Hornbach", "bauhaus": "BAUHAUS", "amazon": "Amazon",
        }
        for idx, line in enumerate(lines[:15]):
            if _bad_vendor_candidate(line, receipt_like=True) and not re.search(r"netto|hem|vodafone|telekom", line, re.I):
                continue
            compact = re.sub(r"[^a-z0-9äöüß]", "", _merchant_match_text(line))
            tokens = [compact] + [re.sub(r"[^a-z0-9äöüß]", "", _merchant_match_text(t)) for t in re.split(r"\s+", line)]
            for brand, canonical in brand_words.items():
                # Keine Fuzzy-Matches für sehr kurze Marken wie HEM, O2, dm, OBI.
                # Diese erzeugen bei OCR-Müll extrem viele False Positives.
                if len(brand) <= 3:
                    continue
                best = max((SequenceMatcher(None, tok[:max(len(brand)+2, 4)], brand).ratio() for tok in tokens if tok), default=0)
                # Bei Kassenbons etwas strenger, damit "NTO/Kasse" nicht zu falschen Marken wird.
                threshold = 0.82
                if brand == "netto" and re.search(r"netto|nett[o0]|neft[o0]|nert[o0]|marken|discount", line, re.I):
                    threshold = 0.72
                if best >= threshold:
                    candidates.append((1050 - idx * 20 + int(best * 100), canonical, f"fuzzy:{line}"))
    except Exception:
        pass
    return candidates


def _generic_header_candidates(lines: list[str], receipt_like: bool) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    legal_forms = ["gmbh", "ug", "gbr", "ag", "kg", "ohg", "e.k", "mbh", "ltd"]
    # 1) Juristische Namen und Firmennamen im oberen Dokumentbereich.
    for idx, line in enumerate(lines[:35]):
        low = line.lower()
        if any(form in low for form in legal_forms) and not _bad_vendor_candidate(line, receipt_like=receipt_like):
            score = 900 - idx * 8
            candidates.append((score, line, f"legal:{line}"))
    # 2) Bei Kassenbons: erste plausible Logo-/Kopfzeile. Keine Technik-/Adresszeile.
    head_limit = 8 if receipt_like else 15
    for idx, line in enumerate(lines[:head_limit]):
        if _bad_vendor_candidate(line, receipt_like=receipt_like):
            continue
        if not re.search(r"[A-Za-zÄÖÜäöüß]{3,}", line):
            continue
        score = 620 - idx * 25
        # Sehr kurze, markenartige Zeilen bevorzugen.
        if len(line) <= 24:
            score += 120
        candidates.append((score, line.strip(" ,;:-"), f"header:{line}"))
    return candidates


def _vendor_candidates(text: str) -> list[tuple[int, str, str]]:
    text = _normalize_ocr_text(text or "")
    raw_lines = _lines(text)
    lines = [_clean_vendor_line(l) for l in raw_lines if _clean_vendor_line(l)]
    receipt_like = _looks_like_receipt(text)
    candidates: list[tuple[int, str, str]] = []
    candidates.extend(_known_merchant_candidates(lines))
    candidates.extend(_fuzzy_brand_candidates(lines))
    candidates.extend(_generic_header_candidates(lines, receipt_like))

    # Dedup: gleicher Händler nur mit bestem Score.
    best: dict[str, tuple[int, str, str]] = {}
    for score, name, reason in candidates:
        clean_name = _clean_vendor_line(name)
        if not clean_name:
            continue
        low = clean_name.lower()
        # Finale harte Sperre gegen TSE-/Kassen-/Footer-Müll als Lieferant.
        if _bad_vendor_candidate(clean_name, receipt_like=receipt_like) and clean_name not in {m[0] for m in KNOWN_MERCHANTS}:
            continue
        if low not in best or score > best[low][0]:
            best[low] = (score, clean_name, reason)
    return sorted(best.values(), key=lambda x: x[0], reverse=True)

def find_vendor(text: str) -> Optional[str]:
    candidates = _vendor_candidates(text)
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


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
            "invoice_date": "YYYY-MM-DD oder null; echtes Rechnungs-/Belegdatum, nicht Zahlungsdatum. Bei Kassenbons steht das Datum oft unten rechts nach der Uhrzeit, z.B. 14:37 27 04 23 = 2023-04-27",
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
    # Textfelder: KI darf ergänzen. Datum/Lieferant dürfen überschrieben werden, wenn lokal leer oder offensichtlich schwach.
    weak_vendors = {"o2", "02", "kan disgauni", "disgauni", "unknown", "unbekannt"}
    if ai.get("vendor") and (not merged.get("vendor") or str(merged.get("vendor")).strip().lower() in weak_vendors or len(str(merged.get("vendor"))) <= 3):
        merged["vendor"] = ai.get("vendor")
    if ai.get("invoice_date") and not merged.get("invoice_date"):
        merged["invoice_date"] = ai.get("invoice_date")
    for key in ["invoice_number", "currency"]:
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

    # Finale Plausibilitätskorrektur: starker lokaler Lieferant/Datum aus Regelwerk gewinnt.
    # Bei Kassenbons darf ein sehr starker Händler-Match auch eine falsche KI-/OCR-Antwort überschreiben.
    vendor_cands = _vendor_candidates(text)
    strong_vendor = vendor_cands[0][1] if vendor_cands else find_vendor(text)
    strong_vendor_score = vendor_cands[0][0] if vendor_cands else 0
    current_vendor = str(merged.get("vendor") or "").strip().lower()
    weak_vendor_values = {"o2", "02", "unbekannt", "unknown", "", "none"}
    if strong_vendor and (current_vendor in weak_vendor_values or strong_vendor_score >= 1800):
        merged["vendor"] = strong_vendor
    strong_date = find_date(text)
    if strong_date:
        merged["invoice_date"] = strong_date

    # Nach finalen Extraktionswerten Kontierung nochmal sauber berechnen.
    final_booking = rule_based_skr03(text, merged.get("vendor"), merged.get("vat_rate"))
    # vorhandene Zahlungsart aus Extraktion behalten
    if merged.get("payment_method"):
        final_booking["payment_method"] = merged["payment_method"]
        final_booking["contra_account"] = "1000" if "bar" in str(merged["payment_method"]).lower() else "1200"
    merged.update(final_booking)
    return merged
