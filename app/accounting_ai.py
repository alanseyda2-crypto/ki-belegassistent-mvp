import json
import os
from decimal import Decimal
from typing import Optional

SKR03_ACCOUNTS = {
    "1000": "Kasse",
    "1200": "Bank",
    "1360": "Geldtransit/Kreditkarte prüfen",
    "1571": "Abziehbare Vorsteuer 7%",
    "1576": "Abziehbare Vorsteuer 19%",
    "3400": "Wareneingang 19% Vorsteuer",
    "3300": "Wareneingang 7% Vorsteuer",
    "4210": "Miete/Pacht",
    "4360": "Versicherungen",
    "4530": "Laufende Kfz-Betriebskosten",
    "4650": "Bewirtungskosten",
    "4660": "Reisekosten Arbeitnehmer",
    "4806": "Wartungskosten Hardware/Software",
    "4910": "Porto",
    "4920": "Telefon",
    "4930": "Bürobedarf",
    "4964": "Aufwendungen für Lizenzen/Konzessionen",
    "4980": "Sonstiger Betriebsbedarf",
}


def vat_key_from_rate(vat_rate: Optional[Decimal]) -> str:
    if vat_rate is None:
        return "Vorsteuer prüfen"
    if Decimal(vat_rate) == Decimal("19.00"):
        return "19% Vorsteuer"
    if Decimal(vat_rate) == Decimal("7.00"):
        return "7% Vorsteuer"
    if Decimal(vat_rate) == Decimal("0.00"):
        return "ohne Vorsteuer"
    return f"{vat_rate}% Vorsteuer prüfen"


def rule_based_skr03(text: str, vendor: Optional[str], vat_rate: Optional[Decimal]) -> dict:
    combined = f"{vendor or ''}\n{text}".lower()

    payment_method = "Bank"
    contra_account = "1200"
    if any(k in combined for k in ["barzahlung", "bar", "cash"]):
        payment_method = "Bar"
        contra_account = "1000"
    elif any(k in combined for k in ["kreditkarte", "credit card", "visa", "mastercard", "amex"]):
        payment_method = "Kreditkarte/Bank"
        contra_account = "1200"

    rules = [
        (["adobe", "software", "saas", "lizenz", "license", "cloud", "openai", "chatgpt"], "Software/Lizenzen", "4964", "Software- und Lizenzkosten"),
        (["hosting", "domain", "server", "aws", "azure", "google cloud", "hetzner", "strato", "ionos"], "IT/Hosting", "4806", "IT- und Hostingkosten"),
        (["telefon", "telekom", "vodafone", "internet", "mobilfunk", " o2 ", "o2 germany"], "Telefon/Internet", "4920", "Telefon- und Internetkosten"),
        (["porto", "dhl", "dpd", "ups", "hermes", "versand", "standardversand"], "Porto/Versand", "4910", "Porto und Versand"),
        (["bürobedarf", "papier", "drucker", "toner", "stift", "notiz", "amazon business"], "Bürobedarf", "4930", "Bürobedarf"),
        (["supermarkt", "discounter", "marken-discount", "netto", "edeka", "rewe", "lidl", "aldi", "kaufland", "lebensmittel", "eier", "bäckerei"], "Lebensmittel/Kassenbon prüfen", "4980", "Kassenbon/Lebensmittel prüfen"),
        (["bewirtung", "restaurant", "cafe", "café", "bäckerei", "essen", "lieferando"], "Bewirtung", "4650", "Bewirtungskosten"),
        (["hotel", "booking.com", "reise", "bahn", "db fernverkehr", "flug", "lufthansa"], "Reisekosten", "4660", "Reisekosten"),
        (["aral", "shell", "esso", "total", "totalenergies", "hem", "tankstelle", "kraftstoff", "diesel", "benzin", "säule", "saeule"], "Kfz-Kosten", "4530", "Kfz-Betriebskosten"),
        (["ware", "waren", "artikel", "produkt", "shop", "einkauf", "schwarzkümmelöl", "patch", "patches"], "Wareneinkauf", "3300" if vat_rate == Decimal("7.00") else "3400", "Wareneinkauf"),
        (["versicherung", "allianz", "huk", "axa"], "Versicherung", "4360", "Versicherungsbeitrag"),
        (["miete", "pacht", "bürofläche", "gewerberaum"], "Miete/Pacht", "4210", "Miete/Pacht"),
    ]

    category, account, booking_text, confidence = "Allgemeine Betriebsausgabe", "4980", "Sonstige betriebliche Aufwendungen", Decimal("0.58")
    for keywords, cat, acc, label in rules:
        if any(k in combined for k in keywords):
            category, account, booking_text, confidence = cat, acc, label, Decimal("0.84")
            break

    return {
        "booking_category": category,
        "account": account,
        "contra_account": contra_account,
        "payment_method": payment_method,
        "vat_key": vat_key_from_rate(vat_rate),
        "booking_text": booking_text,
        "booking_confidence": confidence,
    }


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
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start:end + 1]
    return json.loads(raw)


def ai_skr03_suggestion(text: str, vendor: Optional[str], invoice_date, gross_amount, vat_amount, vat_rate) -> Optional[dict]:
    """SKR03-Kontierung über OpenAI-kompatible API, z.B. OpenRouter."""
    if not os.getenv("OPENAI_API_KEY"):
        print("AI SKR03 skipped: OPENAI_API_KEY missing", flush=True)
        return None
    try:
        client = _openai_compatible_client()
        if client is None:
            return None
        model = os.getenv("OPENAI_ACCOUNTING_MODEL", os.getenv("OPENAI_EXTRACTION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")))
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        print(f"AI SKR03 enabled: model={model}, base_url={base_url}", flush=True)
        prompt = f"""
Du bist ein deutscher Buchhaltungsassistent. Erstelle einen plausiblen Kontierungsvorschlag nach SKR03.
Es ist nur ein Vorschlag für vorbereitende Buchhaltung, keine Steuerberatung.
Gib ausschließlich valides JSON zurück.

Beleg:
Lieferant: {vendor}
Datum: {invoice_date}
Brutto: {gross_amount}
MwSt-Betrag: {vat_amount}
MwSt-Satz: {vat_rate}%
OCR-Text:
{text[:8000]}

JSON-Felder exakt:
{{
  "booking_category": "Kategorie",
  "account": "SKR03-Aufwandskonto, z.B. 4530, 4920, 4930, 4980",
  "contra_account": "Zahlungskonto, meist 1200 Bank oder 1000 Kasse",
  "payment_method": "Bar, Bank, Kreditkarte/Bank, EC-Karte, PayPal oder Unbekannt",
  "vat_key": "z.B. 19% Vorsteuer, 7% Vorsteuer, Vorsteuer prüfen",
  "booking_text": "kurzer Buchungstext",
  "booking_confidence": 0.0
}}
"""
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            print(f"AI SKR03 JSON-mode failed, retrying without response_format: {e}", flush=True)
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
        raw = res.choices[0].message.content
        data = _json_loads_lenient(raw)
        required = ["booking_category", "account", "contra_account", "payment_method", "vat_key", "booking_text", "booking_confidence"]
        if all(k in data for k in required):
            data["booking_confidence"] = Decimal(str(data.get("booking_confidence", 0.75))).quantize(Decimal("0.01"))
            print("AI SKR03 success", flush=True)
            return data
    except Exception as e:
        print(f"AI SKR03 failed: {e}", flush=True)
        return None
    return None
