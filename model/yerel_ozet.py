"""
Gemini API ile Türkçe özet üretici.
Üretim başarısız olursa şablon tabanlı özete döner.
"""

import os
from pathlib import Path

_TARTISMALI_IPUCU = (
    "görüşler bölünmüş", "görüşler farklı", "ikiye bölünmüş",
    "görüş ayrılığı", "bir kısım", "görüşler ikiye",
)

_NEGATIF_IPUCU = (
    "şikayet", "sorun ", "sorunlar", "olumsuz", "yetersiz",
    "beklentinin altında", "hayal kırıklığı", "şüphe", "sahte",
    "taklit", "dar bulun", "kopma", "açılma", "büzülme",
    "hasarlı", "özensiz", "orantısız", "pahalı bulun",
    "kalitesiz", "kötü dikiş", "plastik koku", "bozuldu",
)


def _api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    return key or None


def _gemini_ozet_uret(maddeler: list[str], kategori: str) -> str | None:
    try:
        import google.generativeai as genai

        key = _api_key()
        if not key:
            return None

        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        madde_listesi = "\n".join(f"- {m}" for m in maddeler)
        prompt = (
            f"Aşağıdaki {kategori} ürünü hakkındaki değerlendirme maddelerini "
            f"akıcı, doğal bir Türkçe paragraf hâline getir. "
            f"Maddelerin içeriğini koru, kelimesi kelimesine kopyalama. "
            f"Sadece paragrafı yaz, başlık veya ek açıklama ekleme.\n\n"
            f"{madde_listesi}"
        )

        response = model.generate_content(prompt)
        ozet = response.text.strip()
        return ozet if ozet else None

    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Şablon tabanlı özet (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _ton(madde: str) -> str:
    m = madde.lower()
    if any(ip in m for ip in _TARTISMALI_IPUCU):
        return "tartismali"
    if any(ip in m for ip in _NEGATIF_IPUCU):
        return "negatif"
    return "pozitif"


def _gecis(onceki: str, simdi: str) -> str:
    if simdi == "tartismali":
        return "Öte yandan, "
    if simdi == "negatif":
        return "Öte yandan, " if onceki == "pozitif" else "Ayrıca, "
    return "Bununla birlikte, " if onceki == "negatif" else "Bunun yanı sıra, "


def _sablon_ozet(maddeler: list[str]) -> str | None:
    if not maddeler:
        return None
    tonlar   = [_ton(m) for m in maddeler]
    satirlar = []
    for i, (madde, ton) in enumerate(zip(maddeler, tonlar)):
        if i == 0:
            satirlar.append(madde)
        else:
            gecis = _gecis(tonlar[i - 1], ton)
            satirlar.append(gecis + madde[0].lower() + madde[1:])
    return "\n".join(satirlar)


# ──────────────────────────────────────────────────────────────────────────────
# Dışa açık fonksiyon
# ──────────────────────────────────────────────────────────────────────────────

def yerel_ozet_uret(
    maddeler: list[str],
    kategori: str = "ürün",
) -> str | None:
    """
    Önce Gemini API ile özet üretmeyi dener.
    API key yoksa veya çağrı başarısız olursa şablon özete döner.
    """
    ozet = _gemini_ozet_uret(maddeler, kategori)
    if ozet:
        return ozet
    return _sablon_ozet(maddeler)
