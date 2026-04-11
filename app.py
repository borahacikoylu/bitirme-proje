"""
Streamlit demo uygulaması.

İki mod:
  1) Kayıtlı Ürünler — Önceden analiz edilmiş JSON'dan sonuçları gösterir.
  2) Yeni Ürün Analizi — Ürün ID'si girilerek canlı scrape + analiz yapılır.

Çalıştırma:
    streamlit run app.py
"""

import sys
import json
import streamlit as st
from pathlib import Path

PROJE_KOKU = Path(__file__).resolve().parent
JSON_YOLU = PROJE_KOKU / "sonuclar" / "urun_analiz.json"
MODEL_YOLU = PROJE_KOKU / "sonuclar" / "model" / "best_model"

sys.path.insert(0, str(PROJE_KOKU))
sys.path.insert(0, str(PROJE_KOKU / "src" / "scraper"))


@st.cache_data
def json_yukle(json_yolu: str) -> dict:
    yol = Path(json_yolu)
    if not yol.exists():
        return {}
    with open(yol, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def model_yukle():
    """Modeli bir kez yükleyip cache'le."""
    from model.predict import modeli_yukle
    return modeli_yukle(MODEL_YOLU)


def sonuclari_goster(urun: dict):
    """Bir ürünün analiz sonuçlarını gösterir."""
    ozet = urun.get("ozet", {})
    artilar = urun.get("artilar", [])
    eksiler = urun.get("eksiler", [])
    tartismali = urun.get("tartismali", [])

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Yorum", ozet.get("toplam_yorum", 0))
    k2.metric("Pozitif Cümle", ozet.get("pozitif_cumle", 0))
    k3.metric("Negatif Cümle", ozet.get("negatif_cumle", 0))
    k4.metric("Tartışmalı Konu", len(tartismali))

    st.divider()

    sol, sag = st.columns(2)

    with sol:
        st.subheader("✅ Güçlü Yönler")
        if artilar:
            for item in artilar:
                sayi = item.get("sayi", 0)
                st.success(f"**{item['baslik']}**\n\n_{sayi} kullanıcı bu görüşte_")
        else:
            st.info("Bu ürün için belirgin güçlü yön bulunamadı.")

    with sag:
        st.subheader("❌ Zayıf Yönler")
        if eksiler:
            for item in eksiler:
                sayi = item.get("sayi", 0)
                st.error(f"**{item['baslik']}**\n\n_{sayi} kullanıcı bu görüşte_")
        else:
            st.info("Bu ürün için belirgin zayıf yön bulunamadı.")

    if tartismali:
        st.divider()
        st.subheader("⚖️ Tartışmalı Konular")
        st.caption("Bu konularda kullanıcı görüşleri bölünmüş durumda.")
        for item in tartismali:
            poz = item.get("pozitif", 0)
            neg = item.get("negatif", 0)
            toplam = poz + neg
            poz_yuzde = int(poz / toplam * 100) if toplam > 0 else 0
            st.warning(f"**{item['baslik']}**\n\n{item.get('detay', '')}")
            st.progress(poz_yuzde / 100, text=f"Olumlu %{poz_yuzde}  —  Olumsuz %{100 - poz_yuzde}")

    with st.expander("Ham JSON çıktısı"):
        st.json(urun)


def tab_kayitli_urunler():
    """Önceden analiz edilmiş ürünleri gösterir."""
    veri = json_yukle(str(JSON_YOLU))

    if not veri:
        st.warning(
            "Analiz sonuçları bulunamadı. "
            "Önce `python model/predict.py` komutunu çalıştırın."
        )
        return

    urun_ids = list(veri.keys())
    secilen = st.selectbox("Ürün ID seçin:", urun_ids, index=0)

    if secilen:
        sonuclari_goster(veri[secilen])


def tab_yeni_urun():
    """Yeni bir ürün ID'si girip canlı analiz yapma."""
    st.info(
        "HepsiBurada ürün sayfasındaki SKU kodunu girin. "
        "Sistem yorumları çekip analiz edecektir."
    )

    urun_id = st.text_input(
        "Ürün ID (SKU):",
        placeholder="Örn: HBCV00005QG5TV",
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
            # 1) Yorumları çek
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

            # 2) Model yükle
            st.write("🧠 Model yükleniyor...")
            model, tokenizer, cihaz = model_yukle()
            st.write(f"✅ Model hazır. (Cihaz: {cihaz})")

            # 3) Analiz
            st.write("📊 Yorumlar analiz ediliyor...")
            from model.predict import yorumlari_analiz_et
            analiz = yorumlari_analiz_et(
                yorum_metinleri, model, tokenizer, cihaz
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
        "Ürün yorumlarını konu bazlı analiz eden BERT tabanlı sistem. "
        "Benzer yorumlar gruplanır, çelişkili konular ayrıca gösterilir."
    )
    st.divider()

    tab1, tab2 = st.tabs(["📦 Kayıtlı Ürünler", "🆕 Yeni Ürün Analizi"])

    with tab1:
        tab_kayitli_urunler()

    with tab2:
        tab_yeni_urun()


if __name__ == "__main__":
    ana_sayfa()
