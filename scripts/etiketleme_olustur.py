"""
Scraper CSV'sinden elle etiketleme dosyası oluşturur.
Çıktı: data/etiketleme.csv
  → id, comment, star, label (boş - sen dolduracaksın)
"""

import csv
import re
from pathlib import Path

PROJE = Path(__file__).resolve().parent.parent
GIRDI = PROJE / "src" / "scraper" / "data" / "user_contents.csv"
CIKTI = PROJE / "data" / "etiketleme.csv"


def temizle(metin: str) -> str:
    if not isinstance(metin, str):
        return ""
    metin = metin.replace("\n", " ").replace("\r", " ")
    metin = re.sub(r"[^\w\s.,!?;:'\"\-()/ıİğĞüÜşŞöÖçÇ]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


rows = []
with open(GIRDI, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, start=1):
        rows.append({
            "id": i,
            "comment": temizle(row["comment"]),
            "star": row["star"],
            "label": "",
        })

CIKTI.parent.mkdir(parents=True, exist_ok=True)
with open(CIKTI, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "comment", "star", "label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} yorum → {CIKTI}")
