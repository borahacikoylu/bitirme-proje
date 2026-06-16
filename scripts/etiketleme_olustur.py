"""
Scraper CSV'sinden elle etiketleme dosyası oluşturur/günceller.
Çıktı: data/etiketleme.csv  →  id, comment, star, label (boş), kategori

Her çalışmada user_contents.csv'deki TÜM yorumları işler.
Mevcut etiketleme.csv'deki dolu label'lar korunur (comment eşleşmesiyle).
Duplicate satır oluşturmaz.

Kullanım:
  python scripts/etiketleme_olustur.py
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


def main():
    if not GIRDI.exists() or GIRDI.stat().st_size == 0:
        print(f"HATA: Girdi dosyası bulunamadı veya boş: {GIRDI}")
        print("Önce scraper'ı çalıştırın.")
        return

    # ── Mevcut label'ları oku (comment -> label eşleşmesi) ──
    mevcut_label: dict[str, str] = {}
    if CIKTI.exists() and CIKTI.stat().st_size > 0:
        with open(CIKTI, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lbl = row.get("label", "").strip()
                if lbl in ("0", "1"):
                    mevcut_label[row["comment"]] = lbl

    # ── user_contents.csv'yi oku, duplicate'leri temizle ──
    gorulmusler: set[tuple[str, str]] = set()  # (item, order)
    satirlar = []
    with open(GIRDI, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anahtar = (row.get("item", ""), row.get("order", ""))
            if anahtar in gorulmusler:
                continue
            gorulmusler.add(anahtar)

            yorum = temizle(row.get("comment", ""))
            if not yorum:
                continue

            satirlar.append({
                "comment": yorum,
                "star":    row.get("star", ""),
                "kategori": row.get("kategori", ""),
            })

    # ── Çıktıyı yaz; id'leri 1'den başlat ──
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    with open(CIKTI, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "comment", "star", "label", "kategori"])
        writer.writeheader()
        for i, satir in enumerate(satirlar, start=1):
            writer.writerow({
                "id":       i,
                "comment":  satir["comment"],
                "star":     satir["star"],
                "label":    mevcut_label.get(satir["comment"], ""),
                "kategori": satir["kategori"],
            })

    etiketli = sum(1 for l in mevcut_label.values() if l in ("0", "1"))
    print(f"Toplam {len(satirlar)} yorum yazildi: {CIKTI}")
    print(f"Korunan label sayisi: {etiketli}")
    print()
    print("data/etiketleme.csv dosyasini acip 'label' sutununu doldurun:")
    print("  1 = Olumlu yorum")
    print("  0 = Olumsuz yorum")
    print("  Bos birakilan satirlar egitimde kullanilmaz.")


if __name__ == "__main__":
    main()
