"""
Tahmin ve özetleme modülü.

Ham cümleleri sınıflandırmak yerine:
  1. Her cümleyi pozitif/negatif sınıflandır
  2. Cümlenin konusunu tespit et (kalite, orijinallik, kargo vs.)
  3. Konu bazlı grupla ve say
  4. Çelişkili konuları ayır (hem pozitif hem negatif varsa)
  5. İnsan tarafından okunabilir özetler üret

Çıktı formatı:
  {
    "HBCV00005QG5TV": {
      "ozet": { "toplam_yorum": 319, "pozitif": 145, "negatif": 174 },
      "artilar": [
        { "baslik": "Kullanıcılar ürünü şık ve güzel buluyor.", "sayi": 87 },
        ...
      ],
      "eksiler": [...],
      "tartismali": [
        { "baslik": "Orijinallik konusunda görüşler bölünmüş.",
          "detay": "45 kullanıcı orijinal bulurken, 78 kullanıcı sahte olduğunu düşünüyor.",
          "pozitif": 45, "negatif": 78 }
      ]
    }
  }
"""

import re
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.preprocess import csv_oku, metin_temizle

# ── Sabitler ──
PROJE_KOKU = Path(__file__).resolve().parent.parent
MODEL_YOLU = PROJE_KOKU / "sonuclar" / "model" / "best_model"
CSV_YOLU = PROJE_KOKU / "src" / "scraper" / "data" / "user_contents.csv"
JSON_CIKTI = PROJE_KOKU / "sonuclar" / "urun_analiz.json"
MIN_KELIME = 3

# ─────────────────────────────────────────────────────
# KONU TANIMLARI
# Her konu için:
#   - kelimeler: O konuyla ilgili anahtar kelimeler
#   - pozitif:   Pozitif özetleme şablonu
#   - negatif:   Negatif özetleme şablonu
#   - tartismali: Çelişki durumu şablonu
# ─────────────────────────────────────────────────────
KONULAR = {
    "kalite": {
        "kelimeler": [
            "kalite", "kaliteli", "kalitesiz", "sağlam", "dayanıklı",
            "dandik", "ucuz görün",
        ],
        "pozitif": "Ürün kalitesi beğeniliyor.",
        "negatif": "Ürün kalitesi beğenilmiyor.",
        "tartismali": "Kalite konusunda görüşler bölünmüş.",
    },
    "orijinallik": {
        "kelimeler": [
            "orijinal", "orjinal", "sahte", "çakma", "taklit",
            "imitasyon", "imtasyon", "emitasyon", "imitsoyon",
            "barkod", "barkot", "qr", "karekod",
            "fake", "gerçek değil",
        ],
        "pozitif": "Kullanıcılar ürünün orijinal olduğunu düşünüyor.",
        "negatif": "Kullanıcılar ürünün orijinal olmadığını düşünüyor.",
        "tartismali": "Orijinallik konusunda görüşler bölünmüş.",
    },
    "beden_kalip": {
        "kelimeler": [
            "dar", "geniş", "büyük", "küçük", "tam kalıp", "kalıp",
            "numara", "beden", "sıkı", "sıktı", "olmadı",
            "ayağım", "ayakta", "taraklı", "buçuk",
        ],
        "pozitif": "Beden/kalıp uygun bulunuyor.",
        "negatif": "Beden/kalıp sorunlu bulunuyor, farklı numara öneriliyor.",
        "tartismali": "Beden/kalıp konusunda görüşler farklılık gösteriyor.",
    },
    "kargo_paketleme": {
        "kelimeler": [
            "kargo", "paketleme", "paketlenme", "kutu", "kutusuz",
            "ezilmiş", "kırık", "yırtık", "poşet", "korunaklı",
            "teslimat", "teslim", "hızlı geldi", "geç geldi",
        ],
        "pozitif": "Kargo ve paketleme beğeniliyor.",
        "negatif": "Kargo ve paketleme konusunda ciddi şikayetler var.",
        "tartismali": "Kargo/paketleme konusunda görüşler bölünmüş.",
    },
    "fiyat": {
        "kelimeler": [
            "fiyat", "ucuz", "pahalı", "uygun", "indirim",
            "fiyat performans", "para",
        ],
        "pozitif": "Fiyat-performans oranı beğeniliyor.",
        "negatif": "Fiyatı yüksek bulunuyor.",
        "tartismali": "Fiyat konusunda görüşler bölünmüş.",
    },
    "rahatlik": {
        "kelimeler": [
            "rahat", "rahatsız", "vuruyor", "vurdu", "sıkıyor",
            "konforlu", "ağrı", "acıtı", "kanattı", "kanat",
        ],
        "pozitif": "Ürün rahat bulunuyor.",
        "negatif": "Ürün rahatsız bulunuyor, ayağı vurduğu belirtiliyor.",
        "tartismali": "Rahatlık konusunda görüşler bölünmüş.",
    },
    "gorunum": {
        "kelimeler": [
            "şık", "güzel", "tatlı", "hoş", "harika", "mükemmel",
            "çirkin", "model", "tasarım", "görünüm", "duruş",
        ],
        "pozitif": "Ürünün görünümü ve tasarımı beğeniliyor.",
        "negatif": "Ürünün görünümü beklentiyi karşılamıyor.",
        "tartismali": "Görünüm konusunda görüşler bölünmüş.",
    },
    "koku": {
        "kelimeler": [
            "koku", "kokuyor", "bali", "yapıştırıcı koku",
            "plastik koku", "leş",
        ],
        "pozitif": "Üründe koku sorunu yaşanmamış.",
        "negatif": "Üründe kötü koku şikayetleri var.",
        "tartismali": "Koku konusunda görüşler bölünmüş.",
    },
    "malzeme": {
        "kelimeler": [
            "deri", "kırılma", "kırışık", "kırıldı", "kırışıklık",
            "soyulma", "yırtıl", "yamuk", "defolu", "defo",
            "yapıştırıcı iz", "dikiş",
        ],
        "pozitif": "Malzeme kalitesi beğeniliyor.",
        "negatif": "Malzemede kırılma, kırışma veya dikiş hataları görülüyor.",
        "tartismali": "Malzeme kalitesi konusunda görüşler bölünmüş.",
    },
    "iade": {
        "kelimeler": [
            "iade", "değişim", "geri gönderdim", "geri yolladım",
        ],
        "pozitif": "İade/değişim süreci sorunsuz işlemiş.",
        "negatif": "İade veya değişim talebi oluşmuş.",
        "tartismali": "İade süreçleri konusunda görüşler bölünmüş.",
    },
}

# Bir konunun "tartışmalı" sayılması için azınlık tarafın
# çoğunluğa oranının bu değeri aşması gerekir
TARTISMA_ESIGI = 0.30


def cumlelere_bol(metin: str) -> list[str]:
    """Türkçe metni cümlelere böler."""
    cumleler = re.split(r"(?<=[.!?])\s+", metin)
    if len(cumleler) == 1 and len(metin.split()) > 10:
        cumleler = re.split(r"(?<=,)\s+", metin)
    sonuc = [c.strip() for c in cumleler if len(c.strip().split()) >= MIN_KELIME]
    return sonuc if sonuc else [metin]


def tahmin_yap(
    metin: str,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    cihaz: torch.device,
) -> tuple[int, float]:
    """Tek bir metin için (etiket, güven) döndürür."""
    kodlama = tokenizer(
        metin, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = kodlama["input_ids"].to(cihaz)
    attention_mask = kodlama["attention_mask"].to(cihaz)

    with torch.no_grad():
        cikti = model(input_ids=input_ids, attention_mask=attention_mask)

    olasiliklar = torch.softmax(cikti.logits, dim=1)
    etiket = torch.argmax(olasiliklar, dim=1).item()
    guven = olasiliklar[0][etiket].item()
    return etiket, guven


def konu_tespit_et(cumle: str) -> list[str]:
    """
    Bir cümlenin hangi konu(lar)a ait olduğunu tespit eder.
    Birden fazla konu dönebilir.
    Hiçbir konu eşleşmezse ["genel"] döner.
    """
    cumle_kucuk = cumle.lower()
    bulunan_konular = []

    for konu_adi, konu_bilgi in KONULAR.items():
        for kelime in konu_bilgi["kelimeler"]:
            if kelime.lower() in cumle_kucuk:
                bulunan_konular.append(konu_adi)
                break

    return bulunan_konular if bulunan_konular else ["genel"]


def ozetler_uret(konu_sayilari: dict) -> dict:
    """
    Konu bazlı pozitif/negatif sayılardan özetler üretir.

    Her konu için:
      - Çoğunluk pozitifse → artılara ekle
      - Çoğunluk negatifse → eksilere ekle
      - İkisi de önemliyse → tartışmalıya ekle
    """
    artilar = []
    eksiler = []
    tartismali = []

    for konu, sayilar in sorted(
        konu_sayilari.items(),
        key=lambda x: x[1]["pozitif"] + x[1]["negatif"],
        reverse=True,
    ):
        poz = sayilar["pozitif"]
        neg = sayilar["negatif"]
        toplam = poz + neg

        if toplam < 2:
            continue

        konu_bilgi = KONULAR.get(konu)

        # "genel" konusu için özel şablon
        if konu == "genel":
            if poz > neg:
                artilar.append({"baslik": "Genel olarak ürün beğeniliyor.", "sayi": poz})
            elif neg > poz:
                eksiler.append({"baslik": "Genel olarak ürün beğenilmiyor.", "sayi": neg})
            continue

        # Azınlık oranı: tartışmalı mı?
        azinlik = min(poz, neg)
        cogunluk = max(poz, neg)
        oran = azinlik / cogunluk if cogunluk > 0 else 0

        if oran >= TARTISMA_ESIGI and azinlik >= 3:
            tartismali.append({
                "baslik": konu_bilgi["tartismali"],
                "detay": f"{poz} kullanıcı olumlu bulurken, {neg} kullanıcı olumsuz buluyor.",
                "pozitif": poz,
                "negatif": neg,
            })
        elif poz >= neg:
            artilar.append({"baslik": konu_bilgi["pozitif"], "sayi": poz})
        else:
            eksiler.append({"baslik": konu_bilgi["negatif"], "sayi": neg})

    return {
        "artilar": artilar,
        "eksiler": eksiler,
        "tartismali": tartismali,
    }


def modeli_yukle(model_yolu: str | Path = MODEL_YOLU):
    """Model ve tokenizer'ı yükleyip (model, tokenizer, cihaz) döndürür."""
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_yolu))
    model = BertForSequenceClassification.from_pretrained(str(model_yolu))
    model.to(cihaz)
    model.eval()
    return model, tokenizer, cihaz


def yorumlari_analiz_et(
    yorumlar: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    cihaz: torch.device,
    guven_esigi: float = 0.6,
) -> dict:
    """
    Yorum listesini alıp konu bazlı analiz döndürür.
    CSV'den bağımsız — doğrudan yorum listesiyle çalışır.
    """
    konu_sayilari: dict[str, dict[str, int]] = defaultdict(
        lambda: {"pozitif": 0, "negatif": 0}
    )
    toplam_poz = 0
    toplam_neg = 0

    for yorum in yorumlar:
        yorum_temiz = metin_temizle(str(yorum))
        if not yorum_temiz:
            continue

        for cumle in cumlelere_bol(yorum_temiz):
            etiket, guven = tahmin_yap(cumle, model, tokenizer, cihaz)
            if guven < guven_esigi:
                continue

            konular = konu_tespit_et(cumle)
            sentiment_key = "pozitif" if etiket == 1 else "negatif"
            for konu in konular:
                konu_sayilari[konu][sentiment_key] += 1

            if etiket == 1:
                toplam_poz += 1
            else:
                toplam_neg += 1

    ozetler = ozetler_uret(dict(konu_sayilari))
    return {
        "ozet": {
            "toplam_yorum": len(yorumlar),
            "pozitif_cumle": toplam_poz,
            "negatif_cumle": toplam_neg,
        },
        **ozetler,
    }


def urun_analizi_uret(
    csv_yolu: str | Path = CSV_YOLU,
    model_yolu: str | Path = MODEL_YOLU,
    cikti_yolu: str | Path = JSON_CIKTI,
    guven_esigi: float = 0.6,
) -> dict:
    """Tüm ürünleri analiz edip konu bazlı özetler üretir."""

    print("Model yükleniyor...")
    model, tokenizer, cihaz = modeli_yukle(model_yolu)
    print(f"  Cihaz: {cihaz}")

    df = csv_oku(csv_yolu)
    urun_listesi = df["item"].unique()
    print(f"\n{len(urun_listesi)} ürün bulundu.\n")

    sonuc = {}

    for urun_id in urun_listesi:
        urun_yorumlari = df[df["item"] == urun_id]["comment"].tolist()
        analiz = yorumlari_analiz_et(
            urun_yorumlari, model, tokenizer, cihaz, guven_esigi
        )
        sonuc[urun_id] = analiz

        print(f"  {urun_id}:")
        print(f"    {len(urun_yorumlari)} yorum → "
              f"{analiz['ozet']['pozitif_cumle']} pozitif, "
              f"{analiz['ozet']['negatif_cumle']} negatif")
        print(f"    {len(analiz['artilar'])} artı, {len(analiz['eksiler'])} eksi, "
              f"{len(analiz['tartismali'])} tartışmalı konu")

    cikti_yolu = Path(cikti_yolu)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)

    with open(cikti_yolu, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)

    print(f"\nJSON kaydedildi → {cikti_yolu}")
    return sonuc


if __name__ == "__main__":
    urun_analizi_uret()
