"""
Tahmin ve özetleme modülü — kategori bazlı analiz.

Her kategori (ayakkabı / kıyafet) için ayrı konu tanımları ve
doğal dil özetleri üretir.

Çıktı formatı:
  {
    "kategori": "ayakkabı",
    "ozet": { "toplam_yorum": 100, "pozitif_cumle": 60, "negatif_cumle": 40 },
    "ozet_maddeleri": [
      "Ayakkabıların rahatlığı ve konforu beğeniliyor...",
      "Kalıp dar; yarım veya bir numara büyük almanız tavsiye ediliyor.",
      ...
    ],
    "artilar":    [{"baslik": "...", "sayi": 40}, ...],
    "eksiler":    [...],
    "tartismali": [...]
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

PROJE_KOKU = Path(__file__).resolve().parent.parent
MODEL_YOLU = PROJE_KOKU / "sonuclar" / "model" / "best_model"
CSV_YOLU   = PROJE_KOKU / "src" / "scraper" / "data" / "user_contents.csv"
JSON_CIKTI = PROJE_KOKU / "sonuclar" / "urun_analiz.json"

MIN_KELIME    = 3
TARTISMA_ESIGI = 0.30
MAKS_MADDE    = 7   # Özet listesinde maksimum madde sayısı
MIN_TOPLAM    = 3   # Bir konu için minimum bahsedilme sayısı

# ──────────────────────────────────────────────────────────────────────────────
# AYAKKABI — Konu tanımları
# ──────────────────────────────────────────────────────────────────────────────
KONULAR_AYAKKABI: dict[str, dict] = {
    "rahatlık": {
        "kelimeler": [
            "rahat", "rahatsız", "konforlu", "ağrı", "vuruyor", "vurdu",
            "sıkıyor", "sıktı", "kanattı", "kanat", "acıtı",
        ],
        "pozitif": "Ayakkabıların rahatlığı ve konforu sıkça övülüyor, günlük ve uzun süreli kullanım için ideal bulunuyor.",
        "negatif": "Ayakkabılar konfor açısından şikayet alıyor; topuk vurması ve ayağı sıkması gibi sorunlar dile getiriliyor.",
        "tartismali": "Rahatlık konusunda görüşler ikiye bölünmüş; bir kısım rahat bulurken diğerleri topuk vurması gibi sorunlar yaşadığını belirtiyor.",
    },
    "numara_kalıp": {
        "kelimeler": [
            "dar", "geniş", "numara", "beden", "kalıp", "tam kalıp",
            "buçuk", "taraklı", "ayağım", "sıkı",
        ],
        "pozitif": "Kullanıcıların büyük çoğunluğu kendi numaralarının tam geldiğini bildiriyor.",
        "negatif": "Ayakkabının kalıbı dar bulunuyor; yarım veya bir numara büyük alınması tavsiye ediliyor.",
        "tartismali": "Birçok kullanıcı kendi numarasını almanın yeterli olduğunu söylerken, bazıları yarım numara büyük almanın daha iyi sonuç verdiğini ifade ediyor.",
    },
    "orijinallik": {
        "kelimeler": [
            "orijinal", "orjinal", "sahte", "çakma", "taklit", "imitasyon",
            "imtasyon", "emitasyon", "imitsoyon", "fake", "barkod", "qr", "karekod",
        ],
        "pozitif": "Kullanıcılar ürünün orijinalliğini QR/barkod sorgulamasıyla doğruluyor; orijinal olduğu teyit ediliyor.",
        "negatif": "Ürünün orijinal olmadığına dair ciddi şüpheler dile getiriliyor; sahte veya taklit olduğunu belirten kullanıcılar mevcut.",
        "tartismali": "Orijinallik konusunda görüşler bölünmüş; bir kısım QR kodu ile doğrularken diğerleri şüphelerini paylaşıyor.",
    },
    "görünüm": {
        "kelimeler": [
            "şık", "güzel", "tatlı", "hoş", "harika", "mükemmel", "çirkin",
            "model", "tasarım", "görünüm", "duruş",
        ],
        "pozitif": "Ayakkabıların şıklığı ve görünümü sıkça övülüyor; her kombinle uyumlu ve şık durduğu belirtiliyor.",
        "negatif": "Ayakkabının görünümü beklentiyi karşılamadığını dile getiren kullanıcılar mevcut.",
        "tartismali": "Görünüm ve estetik konusunda görüşler farklılık gösteriyor.",
    },
    "kalite": {
        "kelimeler": [
            "kalite", "kaliteli", "kalitesiz", "sağlam", "dayanıklı", "dandik",
        ],
        "pozitif": "Ayakkabıların kalitesi ve malzeme sağlamlığı övgü alıyor; uzun süreli kullanım için uygun olduğu düşünülüyor.",
        "negatif": "Kalite beklentinin altında kaldığını belirten kullanıcılar var; malzeme zayıf bulunuyor.",
        "tartismali": "Kalite konusunda görüşler bölünmüş; bir kısım sağlam ve kaliteli bulurken diğerleri yetersiz buluyor.",
    },
    "malzeme": {
        "kelimeler": [
            "deri", "kırılma", "kırışık", "kırıldı", "soyulma", "yırtıl",
            "yamuk", "dikiş", "yapıştırıcı",
        ],
        "pozitif": "Ayakkabının derisi ve malzeme kalitesi beğeniliyor; deri yumuşaklığı özellikle övgü alıyor.",
        "negatif": "Malzemede kırılma, kırışma veya açılma sorunları yaşandığı belirtiliyor.",
        "tartismali": "Malzeme kalitesi konusunda görüşler bölünmüş; derinin dayanıklılığı tartışılıyor.",
    },
    "koku": {
        "kelimeler": [
            "koku", "kokuyor", "plastik koku", "bali", "yapıştırıcı koku", "leş",
        ],
        "pozitif": "Üründe belirgin bir koku sorunu yaşanmadığı belirtiliyor.",
        "negatif": "Plastik veya yapıştırıcı kokusu şikayeti dile getiriliyor; rahatsız edici bulunuyor.",
        "tartismali": "Koku konusunda görüşler bölünmüş.",
    },
    "kargo_paketleme": {
        "kelimeler": [
            "kargo", "paketleme", "paketlenme", "kutu", "kutusuz",
            "ezilmiş", "yırtık", "poşet", "özensiz", "korunaklı", "teslimat",
        ],
        "pozitif": "Kargo ve paketleme özenli bulunuyor; ürün sağlam ve zamanında teslim ediliyor.",
        "negatif": "Kargo ve paketleme konusunda şikayetler dile getiriliyor; kutusu hasarlı, eksik veya özensiz geliyor.",
        "tartismali": "Kargo ve paketleme konusunda görüşler farklılık gösteriyor.",
    },
    "fiyat": {
        "kelimeler": [
            "fiyat", "ucuz", "pahalı", "uygun", "indirim", "fiyat performans", "para",
        ],
        "pozitif": "Fiyat-performans açısından olumlu değerlendirmeler yapılıyor; indirimli fiyatlarla satın alanlar memnuniyetlerini dile getiriyor.",
        "negatif": "Fiyatın yüksek ve kaliteyle orantısız olduğunu düşünen kullanıcılar mevcut.",
        "tartismali": "Fiyat konusunda görüşler farklılık gösteriyor; bazıları fiyat-performansı iyi bulurken bazıları pahalı buluyor.",
    },
    "iade": {
        "kelimeler": [
            "iade", "değişim", "geri gönderdim", "geri yolladım",
        ],
        "pozitif": "İade ve değişim sürecinin sorunsuz işlediği belirtiliyor.",
        "negatif": "Bazı kullanıcılar beden uyumsuzluğu veya ürün kalitesi nedeniyle iade süreçlerine başvurduklarını belirtiyor.",
        "tartismali": "İade süreçleri konusunda görüşler bölünmüş.",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# KIYAFET — Konu tanımları
# ──────────────────────────────────────────────────────────────────────────────
KONULAR_KIYAFET: dict[str, dict] = {
    "kumaş": {
        "kelimeler": [
            "kumaş", "pamuk", "polyester", "kalın", "ince", "nefes",
            "yumuşak", "sert", "parlak", "mat", "döküm", "dokuma",
        ],
        "pozitif": "Kumaşın kalitesi, dokusu ve nefes alabilirliği sıkça övülüyor; vücuda rahat oturduğu belirtiliyor.",
        "negatif": "Kumaş ince, sert veya tahriş edici bulunuyor; beklentinin altında kaldığı dile getiriliyor.",
        "tartismali": "Kumaş kalitesi konusunda görüşler bölünmüş; bir kısım yumuşak ve kaliteli bulurken diğerleri hayal kırıklığı yaşadığını belirtiyor.",
    },
    "beden_uyumu": {
        "kelimeler": [
            "beden", "dar", "geniş", "küçük", "büyük", "kalıp", "fit",
            "tam oturdu", "büyük geldi", "küçük geldi", "sıktı",
        ],
        "pozitif": "Kıyafetin bedeni standart ölçülere uygun geliyor; doğru beden seçildiğinde tam oturduğu belirtiliyor.",
        "negatif": "Kalıp dar veya geniş geliyor; bir beden büyük ya da küçük alınması tavsiye ediliyor.",
        "tartismali": "Beden uyumu konusunda görüşler farklılık gösteriyor; bir kısım tam oturduğunu söylerken bazıları farklı beden almak zorunda kaldıklarını belirtiyor.",
    },
    "renk": {
        "kelimeler": [
            "renk", "solma", "solar", "boya", "rengi", "soluk", "canlı",
            "tam renk", "yıkamada", "soldu",
        ],
        "pozitif": "Renk canlı ve görseldekiyle uyumlu olduğu belirtiliyor; yıkamada solma yaşanmadığı aktarılıyor.",
        "negatif": "Rengin görseldekinden farklı veya yıkamada soldukça soluklaştığı şikayet ediliyor.",
        "tartismali": "Renk konusunda görüşler farklılık gösteriyor; bazıları canlı bulurken diğerleri soluk veya farklı olduğunu belirtiyor.",
    },
    "dikiş": {
        "kelimeler": [
            "dikiş", "dikişler", "koptu", "yırtıldı", "açıldı", "dağıldı",
            "yamuk", "kötü dikiş",
        ],
        "pozitif": "Dikişler sağlam ve düzgün bulunuyor; uzun süreli kullanımda dağılma yaşanmadığı aktarılıyor.",
        "negatif": "Dikişlerde kopma, açılma veya yamukluğa dair şikayetler dile getiriliyor.",
        "tartismali": "Dikiş kalitesi konusunda görüşler bölünmüş.",
    },
    "görünüm": {
        "kelimeler": [
            "şık", "güzel", "tatlı", "hoş", "harika", "model", "tasarım",
            "görünüm", "mükemmel", "çirkin", "kombinle",
        ],
        "pozitif": "Kıyafetin görünümü ve tasarımı sıkça övülüyor; şık ve kombinlemeye uygun olduğu belirtiliyor.",
        "negatif": "Görünümün fotoğraftaki gibi durmadığını veya beklentiyi karşılamadığını belirten kullanıcılar mevcut.",
        "tartismali": "Görünüm ve estetik konusunda görüşler farklılık gösteriyor.",
    },
    "kalite": {
        "kelimeler": [
            "kalite", "kaliteli", "kalitesiz", "sağlam", "dayanıklı", "dandik",
        ],
        "pozitif": "Kıyafetin genel kalitesi ve sağlamlığı iyi bulunuyor; fiyatına göre kaliteli olduğu vurgulanıyor.",
        "negatif": "Kalite beklentinin altında kaldığını belirten kullanıcılar mevcut; fiyatla orantılı bulunmuyor.",
        "tartismali": "Kalite konusunda görüşler bölünmüş; bir kısım sağlam bulurken diğerleri yetersiz buluyor.",
    },
    "yıkama": {
        "kelimeler": [
            "yıkama", "çekme", "çekti", "bozuldu", "büzüldü", "soldu",
            "yıkandı", "makinede",
        ],
        "pozitif": "Yıkama sonrası şekil ve renk korunuyor; makine yıkamaya dayanıklı olduğu belirtiliyor.",
        "negatif": "Yıkamada çekme, büzülme veya solma yaşandığı dile getiriliyor; dikkatli yıkama öneriliyor.",
        "tartismali": "Yıkama sonrası dayanıklılık konusunda görüşler bölünmüş.",
    },
    "kargo_paketleme": {
        "kelimeler": [
            "kargo", "paketleme", "kutu", "özensiz", "hasarlı", "kırık",
            "teslimat", "poşet",
        ],
        "pozitif": "Kargo ve paketleme özenli bulunuyor; ürün sağlam ve zamanında teslim ediliyor.",
        "negatif": "Kargo özensiz; ürün hasarlı veya kötü paketlenmiş şekilde geldiği belirtiliyor.",
        "tartismali": "Kargo ve paketleme konusunda görüşler farklılık gösteriyor.",
    },
    "fiyat": {
        "kelimeler": [
            "fiyat", "ucuz", "pahalı", "uygun", "indirim", "fiyat performans", "para",
        ],
        "pozitif": "Fiyat-performans açısından olumlu değerlendirmeler yapılıyor; indirimli dönemlerde satın alanlar memnuniyetlerini dile getiriyor.",
        "negatif": "Fiyatın kaliteyle orantısız olduğunu düşünen kullanıcılar mevcut; pahalı bulunuyor.",
        "tartismali": "Fiyat konusunda görüşler farklılık gösteriyor; bazıları uygun bulurken bazıları pahalı buluyor.",
    },
    "iade": {
        "kelimeler": [
            "iade", "değişim", "geri gönderdim", "geri yolladım",
        ],
        "pozitif": "İade ve değişim sürecinin sorunsuz ilerlediği belirtiliyor.",
        "negatif": "Beden uyumsuzluğu veya ürün kalitesi nedeniyle iade süreçlerine başvurulduğu belirtiliyor.",
        "tartismali": "İade süreçleri konusunda görüşler bölünmüş.",
    },
}

KATEGORI_KONULAR = {
    "ayakkabı": KONULAR_AYAKKABI,
    "kıyafet":  KONULAR_KIYAFET,
}


# ──────────────────────────────────────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────────────────────────────────────

def cumlelere_bol(metin: str) -> list[str]:
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
    kodlama = tokenizer(
        metin, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids: torch.Tensor      = kodlama["input_ids"]  # type: ignore[assignment]
    attention_mask: torch.Tensor = kodlama["attention_mask"]  # type: ignore[assignment]
    input_ids      = input_ids.to(cihaz)
    attention_mask = attention_mask.to(cihaz)

    with torch.no_grad():
        cikti = model(input_ids=input_ids, attention_mask=attention_mask)

    olasiliklar = torch.softmax(cikti.logits, dim=1)
    etiket = int(torch.argmax(olasiliklar, dim=1).item())
    guven  = float(olasiliklar[0][etiket].item())
    return etiket, guven


def konu_tespit_et(cumle: str, konular: dict) -> list[str]:
    cumle_kucuk = cumle.lower()
    bulunan = []
    for konu_adi, konu_bilgi in konular.items():
        for kelime in konu_bilgi["kelimeler"]:
            if kelime.lower() in cumle_kucuk:
                bulunan.append(konu_adi)
                break
    return bulunan if bulunan else ["genel"]


def ozet_maddeleri_uret(konu_sayilari: dict, konular: dict) -> list[str]:
    """
    Konu bazlı pozitif/negatif sayılardan madde bazlı özet oluşturur.
    En çok bahsedilen konular önce gelir.
    """
    maddeler = []

    sirali = sorted(
        konu_sayilari.items(),
        key=lambda x: x[1]["pozitif"] + x[1]["negatif"],
        reverse=True,
    )

    for konu, sayilar in sirali:
        poz    = sayilar["pozitif"]
        neg    = sayilar["negatif"]
        toplam = poz + neg

        if toplam < MIN_TOPLAM:
            continue

        if konu == "genel":
            continue

        konu_bilgi = konular.get(konu)
        if konu_bilgi is None:
            continue

        azinlik  = min(poz, neg)
        cogunluk = max(poz, neg)
        oran     = azinlik / cogunluk if cogunluk > 0 else 0

        if oran >= TARTISMA_ESIGI and azinlik >= 3:
            maddeler.append(konu_bilgi["tartismali"])
        elif poz >= neg:
            maddeler.append(konu_bilgi["pozitif"])
        else:
            maddeler.append(konu_bilgi["negatif"])

        if len(maddeler) >= MAKS_MADDE:
            break

    return maddeler


def yapili_ozetler_uret(konu_sayilari: dict, konular: dict) -> dict:
    """artilar / eksiler / tartismali yapısını üretir (detay kartları için)."""
    artilar   = []
    eksiler   = []
    tartismali = []

    for konu, sayilar in sorted(
        konu_sayilari.items(),
        key=lambda x: x[1]["pozitif"] + x[1]["negatif"],
        reverse=True,
    ):
        poz    = sayilar["pozitif"]
        neg    = sayilar["negatif"]
        toplam = poz + neg

        if toplam < 2:
            continue

        if konu == "genel":
            if poz > neg:
                artilar.append({"baslik": "Genel olarak ürün beğeniliyor.", "sayi": poz})
            elif neg > poz:
                eksiler.append({"baslik": "Genel olarak ürün beğenilmiyor.", "sayi": neg})
            continue

        konu_bilgi = konular.get(konu)
        if konu_bilgi is None:
            continue

        azinlik  = min(poz, neg)
        cogunluk = max(poz, neg)
        oran     = azinlik / cogunluk if cogunluk > 0 else 0

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

    return {"artilar": artilar, "eksiler": eksiler, "tartismali": tartismali}


def modeli_yukle(model_yolu: str | Path = MODEL_YOLU):
    cihaz     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_yolu))
    model     = BertForSequenceClassification.from_pretrained(str(model_yolu))
    model.to(cihaz)
    model.eval()
    return model, tokenizer, cihaz


def yorumlari_analiz_et(
    yorumlar: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    cihaz: torch.device,
    kategori: str = "ayakkabı",
    guven_esigi: float = 0.6,
) -> dict:
    """
    Yorum listesini alıp kategori bazlı analiz döndürür.

    Parametreler
    ------------
    kategori : "ayakkabı" veya "kıyafet"
    """
    if kategori not in KATEGORI_KONULAR:
        raise ValueError(f"Geçersiz kategori: {kategori!r}. Seçenekler: {list(KATEGORI_KONULAR)}")

    konular = KATEGORI_KONULAR[kategori]

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

            bulunan_konular = konu_tespit_et(cumle, konular)
            sentiment_key   = "pozitif" if etiket == 1 else "negatif"
            for konu in bulunan_konular:
                konu_sayilari[konu][sentiment_key] += 1

            if etiket == 1:
                toplam_poz += 1
            else:
                toplam_neg += 1

    konu_sayilari_dict = dict(konu_sayilari)
    ozet_maddeleri     = ozet_maddeleri_uret(konu_sayilari_dict, konular)
    yapili             = yapili_ozetler_uret(konu_sayilari_dict, konular)

    return {
        "kategori": kategori,
        "ozet": {
            "toplam_yorum":   len(yorumlar),
            "pozitif_cumle":  toplam_poz,
            "negatif_cumle":  toplam_neg,
        },
        "ozet_maddeleri": ozet_maddeleri,
        **yapili,
    }


def urun_analizi_uret(
    csv_yolu:    str | Path = CSV_YOLU,
    model_yolu:  str | Path = MODEL_YOLU,
    cikti_yolu:  str | Path = JSON_CIKTI,
    guven_esigi: float = 0.6,
) -> dict:
    """Tüm ürünleri analiz edip konu bazlı özetler üretir."""
    print("Model yükleniyor...")
    model, tokenizer, cihaz = modeli_yukle(model_yolu)
    print(f"  Cihaz: {cihaz}")

    df = csv_oku(csv_yolu)

    # Kategori sütunu yoksa varsayılan olarak ayakkabı kullan
    if "kategori" not in df.columns:
        df["kategori"] = "ayakkabı"

    urun_listesi = df["item"].unique()
    print(f"\n{len(urun_listesi)} ürün bulundu.\n")

    sonuc = {}

    for urun_id in urun_listesi:
        urun_df      = df[df["item"] == urun_id]
        urun_yorumlar = urun_df["comment"].tolist()

        # Ürünün kategorisini al (aynı ürünün tüm yorumları aynı kategoride olmalı)
        kategori = urun_df["kategori"].iloc[0] if "kategori" in urun_df.columns else "ayakkabı"

        analiz = yorumlari_analiz_et(
            urun_yorumlar, model, tokenizer, cihaz,
            kategori=kategori,
            guven_esigi=guven_esigi,
        )
        sonuc[urun_id] = analiz

        print(f"  [{kategori}] {urun_id}:")
        print(f"    {len(urun_yorumlar)} yorum → "
              f"{analiz['ozet']['pozitif_cumle']} pozitif, "
              f"{analiz['ozet']['negatif_cumle']} negatif")
        print(f"    {len(analiz['ozet_maddeleri'])} özet maddesi, "
              f"{len(analiz['artilar'])} artı, "
              f"{len(analiz['eksiler'])} eksi")

    cikti_yolu = Path(cikti_yolu)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    with open(cikti_yolu, "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)

    print(f"\nJSON kaydedildi → {cikti_yolu}")
    return sonuc


if __name__ == "__main__":
    urun_analizi_uret()
