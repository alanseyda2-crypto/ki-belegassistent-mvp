# KI-Belegassistent MVP — Railway Ready

Ein erster MVP für einen KI-Assistenten für vorbereitende Buchhaltung.

## Funktionen

- Upload von PDF/JPG/PNG/WEBP
- Speichern der Belege
- Texterkennung bei PDFs
- Bild-OCR über Tesseract
- automatische Extraktion von Datum, Lieferant, Betrag, MwSt. und Rechnungsnummer
- Sortierung nach Rechnungsdatum
- Dashboard und Detailansicht

## Lokal starten

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Dann öffnen:

```text
http://127.0.0.1:8000
```

## Railway Deployment

Dieses Projekt ist bereits Railway-ready.

Wichtige Dateien:

- `Dockerfile`
- `railway.json`
- `start.sh`
- `Procfile`
- `requirements.txt`

Railway Start Command:

```bash
sh start.sh
```

Falls Railway fragt: Builder = Dockerfile.

## Hinweis

SQLite und lokale Uploads sind für den MVP okay. Für Produktion später ersetzen durch:

- PostgreSQL
- S3/Supabase Storage
- Azure Document Intelligence
- OpenAI Kontierungsvorschläge
