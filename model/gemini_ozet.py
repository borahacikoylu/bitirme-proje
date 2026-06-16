"""
Gemini Flash ile analiz maddelerinden akıcı Türkçe özet üretir.

Kullanım:
    from model.gemini_ozet import gemini_ozet_uret
    ozet = gemini_ozet_uret(maddeler, kategori="ayakkabı", api_key="...")
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass


def gemini_ozet_uret(
    maddeler: list[str],
    kategori: str = "ürün",
    api_key: str | None = None,
) -> str | None:
    """
    ozet_maddeleri listesini Gemini 1.5 Flash'a gönderip akıcı
    bir Türkçe değerlendirme metni döndürür.

    Başarısız olursa None döner — uygulama yine de çalışmaya devam eder.
    """
    if not maddeler:
        return None

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    client = genai.Client(api_key=key)

    maddeler_metin = "\n".join(f"- {m}" for m in maddeler)

    prompt = f"""Sana bir {kategori} ürününe ait müşteri yorumlarından otomatik çıkarılmış analiz maddeleri veriyorum.
Bu maddeleri oku ve kullanıcıya sunmak için doğal, akıcı bir Türkçe değerlendirme özeti yaz.

Kurallar:
- 3-5 cümle yaz, her cümle yeni satırda başlasın
- "belirtiliyor", "dile getiriliyor", "aktarılıyor", "öne çıkıyor" gibi pasif gazetecilik dili kullan
- Maddeleri birleştir ve zenginleştir — kopyala-yapıştır yapma
- Rakam veya yüzde kullanma
- Sadece Türkçe yaz

Analiz maddeleri:
{maddeler_metin}

Değerlendirme özeti:"""

    try:
        yanit = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return yanit.text.strip()
    except Exception as e:
        print(f"[gemini_ozet] Hata: {e}")
        return None
