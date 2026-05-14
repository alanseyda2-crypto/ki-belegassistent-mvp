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
        (["telefon", "telekom", "vodafone", "o2", "internet", "mobilfunk"], "Telefon/Internet", "4920", "Telefon- und Internetkosten"),
        (["porto", "dhl", "dpd", "ups", "hermes", "versand", "standardversand"], "Porto/Versand", "4910", "Porto und Versand"),
        (["bürobedarf", "papier", "drucker", "toner", "stift", "notiz", "amazon business"], "Bürobedarf", "4930", "Bürobedarf"),
        (["bewirtung", "restaurant", "cafe", "café", "bäckerei", "essen", "lieferando"], "Bewirtung", "4650", "Bewirtungskosten"),
        (["hotel", "booking.com", "reise", "bahn", "db fernverkehr", "flug", "lufthansa"], "Reisekosten", "4660", "Reisekosten"),
        (["aral", "shell", "esso", "total", "tankstelle", "kraftstoff", "diesel", "benzin"], "Kfz-Kosten", "4530", "Kfz-Betriebskosten"),
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
        "booking_source": "regelwerk",
    }


def ai_skr03_suggestion(text: str, vendor: Optional[str], invoice_date, gross_amount, vat_amount, vat_rate) -> Optional[dict]:
    """Optional: nutzt OpenAI nur, wenn OPENAI_API_KEY gesetzt ist. Sonst None."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
        client = OpenAI()
        prompt = f"""
Du bist ein deutscher Buchhaltungsassistent. Erstelle einen Kontierungsvorschlag nach SKR03.
Wichtig: Keine Steuerberatung, nur Vorschlag. Gib ausschließlich valides JSON zurück.

Beleg:
Lieferant: {vendor}
Datum: {invoice_date}
Brutto: {gross_amount}
MwSt-Betrag: {vat_amount}
MwSt-Satz: {vat_rate}%
OCR-Text:
{text[:6000]}

JSON-Felder:
booking_category, account, contra_account, payment_method, vat_key, booking_text, booking_confidence
booking_confidence als Zahl zwischen 0 und 1.
"""
        res = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = res.choices[0].message.content.strip().strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()
        data = json.loads(raw)
        required = ["booking_category", "account", "contra_account", "payment_method", "vat_key", "booking_text", "booking_confidence"]
        if all(k in data for k in required):
            data["booking_source"] = "openai"
            data["booking_confidence"] = Decimal(str(data.get("booking_confidence", 0.75))).quantize(Decimal("0.01"))
            return data
    except Exception:
        return None
    return None
