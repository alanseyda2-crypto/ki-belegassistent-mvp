# KI-Belegassistent MVP

Railway-ready MVP für vorbereitende Buchhaltung:

- Upload von PDF/JPG/PNG/WEBP
- Belegdaten-Erkennung: Datum, Lieferant, Rechnungsnummer, Brutto, MwSt.-Betrag, MwSt.-Satz
- SKR03-Kontierungsvorschläge
- manuelle Korrektur und Lernregeln
- Originalrechnung öffnen und herunterladen
- DATEV-CSV Export für alle oder nur bestätigte Belege

## Lokal starten

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Railway Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Optional für AI-Kontierung:

```text
OPENAI_API_KEY=dein_key
```

Hinweis: Der DATEV-CSV-Export ist eine Import-Vorbereitung/Buchungsstapel-Struktur. Je nach DATEV-Setup beim Steuerberater können zusätzliche Stapel-Metadaten erforderlich sein.

## Generische KI-Belegerkennung

Für allgemein bessere Kassenbon-/Handyfoto-Erkennung ohne erneutes Redeploy:

1. In Railway unter Variables setzen:
   `OPENAI_API_KEY=...`
2. Optional Modell setzen:
   `OPENAI_EXTRACTION_MODEL=gpt-4o-mini`
3. Redeploy/Restart.

Dann analysiert die App bei Bildern zusätzlich das Belegfoto direkt und muss nicht für jeden Händler hardcodiert werden. Ohne Key läuft weiterhin die lokale OCR-Regel-Logik.
