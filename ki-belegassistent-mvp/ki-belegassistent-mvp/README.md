# KI-Belegassistent für vorbereitende Buchhaltung — MVP

Ein erster lauffähiger MVP für Upload, Erkennung, Speicherung und Sortierung von Belegen/Rechnungen.

## Funktionen

- Upload von PDF/JPG/PNG
- Speicherung der Originaldatei
- Textextraktion aus PDFs per `pypdf`
- optionale Bilderkennung per `pytesseract`, falls Tesseract lokal installiert ist
- automatische Extraktion von:
  - Datum
  - Lieferant/Kandidat
  - Bruttobetrag
  - MwSt.-Betrag
  - Rechnungsnummer/Kandidat
- automatische Sortierung nach Datum
- Dashboard mit Status und Belegdetails
- SQLite-Datenbank, später leicht auf PostgreSQL umstellbar

## Start lokal

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Danach öffnen:

```text
http://127.0.0.1:8000
```

## Hinweis zu OCR für Bilder

Für JPG/PNG brauchst du lokal Tesseract OCR.

macOS:

```bash
brew install tesseract tesseract-lang
```

Ubuntu/Debian:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-deu
```

Windows:

Tesseract installieren und Pfad in PATH aufnehmen.

## Nächster sinnvoller Schritt

- Azure Document Intelligence anbinden
- OpenAI-KI-Kontierung ergänzen
- SKR03/SKR04 auswählbar machen
- DATEV-CSV-Export bauen
