"""
Streamlit demo uygulaması — Kategori bazlı sentiment analizi.

İki mod:
  1) Kayıtlı Ürünler  — Önceden analiz edilmiş JSON'dan sonuçları gösterir.
  2) Yeni Ürün Analizi — Ürün ID'si + kategori girerek canlı scrape + analiz yapar.

Çalıştırma:
    streamlit run app.py
"""

import sys
import json
import streamlit as st
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parent
JSON_YOLU  = PROJE_KOKU / "sonuclar" / "urun_analiz.json"
MODEL_YOLU = PROJE_KOKU / "sonuclar" / "model" / "best_model"

sys.path.insert(0, str(PROJE_KOKU))
sys.path.insert(0, str(PROJE_KOKU / "src" / "scraper"))

# .env dosyasını yükle (varsa)
try:
    from dotenv import load_dotenv
    load_dotenv(PROJE_KOKU / ".env")
except ImportError:
    pass

KATEGORI_ETIKETLER = {
    "ayakkabı": "👟 Ayakkabı",
    "kıyafet":  "👕 Kıyafet",
}


@st.cache_data
def json_yukle(json_yolu: str) -> dict:
    yol = Path(json_yolu)
    if not yol.exists():
        return {}
    with open(yol, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def model_yukle():
    from model.predict import modeli_yukle
    return modeli_yukle(MODEL_YOLU)



def ozet_bolumu_goster(maddeler: list[str], kategori: str):
    """Gemini ile üretilmiş akıcı özeti gösterir."""
    st.subheader("✨ Değerlendirme")
    from model.yerel_ozet import yerel_ozet_uret

    with st.spinner("Özet üretiliyor..."):
        ozet = yerel_ozet_uret(maddeler, kategori=kategori)

    if ozet:
        st.info(ozet)
    else:
        st.warning("Değerlendirme üretilemedi.")


def ozet_maddeleri_goster(maddeler: list[str], kategori: str):
    if maddeler:
        ozet_bolumu_goster(maddeler, kategori)
    else:
        st.info("Özet oluşturmak için yeterli veri bulunamadı.")


def detay_kartlari_goster(urun: dict):
    """Artı / eksi / tartışmalı konu kartlarını gösterir."""
    ozet       = urun.get("ozet", {})
    artilar    = urun.get("artilar", [])
    eksiler    = urun.get("eksiler", [])
    tartismali = urun.get("tartismali", [])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Yorum",   ozet.get("toplam_yorum", 0))
    k2.metric("Pozitif Cümle",  ozet.get("pozitif_cumle", 0))
    k3.metric("Negatif Cümle",  ozet.get("negatif_cumle", 0))
    k4.metric("Tartışmalı Konu", len(tartismali))

    st.divider()
    sol, sag = st.columns(2)

    with sol:
        st.subheader("✅ Güçlü Yönler")
        if artilar:
            for item in artilar:
                st.success(f"**{item['baslik']}**\n\n_{item.get('sayi', 0)} kullanıcı bu görüşte_")
        else:
            st.info("Belirgin güçlü yön bulunamadı.")

    with sag:
        st.subheader("❌ Zayıf Yönler")
        if eksiler:
            for item in eksiler:
                st.error(f"**{item['baslik']}**\n\n_{item.get('sayi', 0)} kullanıcı bu görüşte_")
        else:
            st.info("Belirgin zayıf yön bulunamadı.")

    if tartismali:
        st.divider()
        st.subheader("⚖️ Tartışmalı Konular")
        for item in tartismali:
            poz      = item.get("pozitif", 0)
            neg      = item.get("negatif", 0)
            toplam   = poz + neg
            poz_yuzde = int(poz / toplam * 100) if toplam > 0 else 0
            st.warning(f"**{item['baslik']}**\n\n{item.get('detay', '')}")
            st.progress(poz_yuzde / 100, text=f"Olumlu %{poz_yuzde}  —  Olumsuz %{100 - poz_yuzde}")


def sonuclari_goster(urun: dict):
    """Bir ürünün analiz sonuçlarını gösterir: önce özet, sonra detaylar."""
    kategori      = urun.get("kategori", "ayakkabı")
    ozet_maddeleri = urun.get("ozet_maddeleri", [])

    ozet_maddeleri_goster(ozet_maddeleri, kategori)

    st.divider()
    with st.expander("📊 Konu Bazlı Detay", expanded=False):
        detay_kartlari_goster(urun)

    with st.expander("Ham JSON çıktısı", expanded=False):
        st.json(urun)


def tab_kayitli_urunler():
    veri = json_yukle(str(JSON_YOLU))

    if not veri:
        st.warning(
            "Analiz sonuçları bulunamadı. "
            "Önce `python model/predict.py` komutunu çalıştırın."
        )
        return

    urun_ids = list(veri.keys())
    secilen  = st.selectbox("Ürün ID seçin:", urun_ids, index=0)

    if secilen:
        sonuclari_goster(veri[secilen])


def tab_yeni_urun():
    st.info(
        "HepsiBurada ürün sayfasındaki SKU kodunu ve ürün kategorisini girin. "
        "Sistem yorumları çekip analiz edecektir."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        urun_id = st.text_input(
            "Ürün ID (SKU):",
            placeholder="Örn: HBCV00005QG5TV",
        )
    with col2:
        kategori = st.selectbox(
            "Kategori:",
            options=list(KATEGORI_ETIKETLER.keys()),
            format_func=lambda k: KATEGORI_ETIKETLER[k],
        )

    if not urun_id:
        return

    urun_id = urun_id.strip()

    if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
        if not MODEL_YOLU.exists():
            st.error(
                "Eğitilmiş model bulunamadı! "
                "Colab'da eğittiğin modeli `sonuclar/model/best_model/` klasörüne koy."
            )
            return

        with st.status("Analiz devam ediyor...", expanded=True) as durum:
            st.write("📥 Yorumlar çekiliyor...")
            try:
                from asd import yorumlari_cek
            except ImportError:
                from src.scraper.asd import yorumlari_cek

            yorumlar_raw = yorumlari_cek(urun_id)

            if not yorumlar_raw:
                durum.update(label="Hata!", state="error")
                st.error(f"'{urun_id}' için yorum bulunamadı. SKU kodunu kontrol edin.")
                return

            yorum_metinleri = [r["comment"] for r in yorumlar_raw]
            st.write(f"✅ {len(yorum_metinleri)} yorum çekildi.")

            st.write("🧠 Model yükleniyor...")
            model, tokenizer, cihaz = model_yukle()
            st.write(f"✅ Model hazır. (Cihaz: {cihaz})")

            st.write("📊 Yorumlar analiz ediliyor...")
            from model.predict import yorumlari_analiz_et
            analiz = yorumlari_analiz_et(
                yorum_metinleri, model, tokenizer, cihaz,
                kategori=kategori,
            )

            durum.update(label="Analiz tamamlandı!", state="complete")

        st.divider()
        st.subheader(f"📋 Sonuçlar — {urun_id}")
        sonuclari_goster(analiz)


def ana_sayfa():
    st.set_page_config(
        page_title="Yorum Sentiment Analizi",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Türkçe Yorum Sentiment Analizi")
    st.markdown(
        "Ürün yorumlarını **kategori bazlı** analiz eden BERT tabanlı sistem. "
        "Ayakkabı ve kıyafet için ayrı konu tanımları ve doğal dil özetleri üretir."
    )
    st.divider()

    tab1, tab2 = st.tabs(["📦 Kayıtlı Ürünler", "🆕 Yeni Ürün Analizi"])

    with tab1:
        tab_kayitli_urunler()

    with tab2:
        tab_yeni_urun()


if __name__ == "__main__":
    ana_sayfa()
