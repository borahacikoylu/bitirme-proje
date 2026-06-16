# Türkçe Yorum Sentiment Analizi — Kategori Bazlı

HepsiBurada ürün yorumlarını **ayakkabı** ve **kıyafet** kategorilerine göre analiz eden BERT tabanlı NLP sistemi.
Her kategori için ayrı konu tanımları kullanılır ve çıktı doğal dil madde listesi olarak sunulur.

**Model:** [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased)

---

## Proje Yapısı

```
bitirme-proje/
├── src/scraper/
│   ├── asd.py                  ← HepsiBurada yorum scraper (--kategori flag'i var)
│   └── data/user_contents.csv  ← Scraper çıktısı (item, order, comment, star, kategori)
├── data/
│   ├── preprocess.py           ← CSV okuma, temizleme, train/test split
│   └── etiketleme.csv          ← Elle etiketlenmiş veri (id, comment, star, label, kategori)
├── model/
│   ├── dataset.py              ← PyTorch Dataset sınıfı
│   ├── train.py                ← BERT fine-tuning (Trainer API)
│   └── predict.py              ← Kategori bazlı analiz → doğal dil özet
├── scripts/
│   └── etiketleme_olustur.py   ← Scraper çıktısından etiketleme dosyası üretir
├── notebooks/
│   └── colab_egitim.ipynb
├── sonuclar/                   ← Eğitim sonrası otomatik oluşur
│   ├── model/best_model/
│   └── urun_analiz.json
├── app.py                      ← Streamlit demo arayüzü
└── requirements.txt
```

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Adım Adım İş Akışı

### 1. Veri Toplama (Scraping)

`asd.py`'yi `--kategori` parametresiyle çalıştır. Her kategori için ayrı çalıştır.

```bash
# URUN_LISTESI'ni asd.py içinde doldur, sonra:
python src/scraper/asd.py --kategori ayakkabı

# Kıyafet ürünlerini scrape etmek için SKU'ları değiştirip:
python src/scraper/asd.py --kategori kıyafet
```

`src/scraper/data/user_contents.csv` dosyasına `kategori` sütunuyla birlikte kaydeder.

---

### 2. Etiketleme Dosyası Oluşturma

```bash
python scripts/etiketleme_olustur.py
```

Bu komut `user_contents.csv`'den `data/etiketleme.csv` dosyasını üretir.
Sütunlar: `id, comment, star, label, kategori`

> `label` sütunu boş gelir — sen dolduracaksın.

---

### 3. Elle Etiketleme

`data/etiketleme.csv` dosyasını Excel veya herhangi bir CSV editörüyle aç.
Her yorumun `label` sütununa şu değerleri yaz:

| Değer | Anlam |
|-------|-------|
| `1`   | **Olumlu** yorum — ürünü beğenmiş, tavsiye ediyor |
| `0`   | **Olumsuz** yorum — şikayet ediyor, memnun değil |
| boş   | Etiketlenmedi — eğitimde kullanılmaz |

#### Etiketleme İpuçları

**Olumlu (1) örnekler:**
- "Çok rahat, tam kalıp, kesinlikle tavsiye ederim."
- "Kumaşı çok güzel, fiyatına göre kaliteli."
- "Hızlı geldi, görseldeki gibi."

**Olumsuz (0) örnekler:**
- "Kalıp dar, iade ettim, çok kötüydü."
- "Sahte ürün, orijinal kutusunda gelmedi."
- "Dikiş hataları var, kumaş ince."

**Dikkat edilecekler:**
- Karma yorumlarda (hem iyi hem kötü yan) ağırlıklı duyguya bakarak karar ver.
- Çok kısa yorumları ("Güzel", "Teşekkürler") olumlu (1) etiketleyebilirsin.
- Yıldızı yüksek ama içerik olumsuz olan yorumlar var — içeriğe bak, yıldıza değil.
- `star` sütunu rehber olabilir ama her zaman doğru değil (bazıları yanlışlıkla 5 veriyor).

#### Tavsiye Edilen Etiketleme Miktarı

| Kategori  | Minimum | İdeal  |
|-----------|---------|--------|
| Ayakkabı  | 200     | 500+   |
| Kıyafet   | 200     | 500+   |

Her kategoride olumlu/olumsuz oranı yaklaşık **50/50** olursa model daha iyi öğrenir.

---

### 4. Model Eğitimi

#### Colab'da (Önerilen — ücretsiz GPU)

1. [Google Colab](https://colab.research.google.com/) açın
2. `notebooks/colab_egitim.ipynb` dosyasını yükleyin
3. **Runtime → T4 GPU** seçin
4. `data/etiketleme.csv` dosyasını yükleyin
5. Hücreleri sırayla çalıştırın (~2-3 dk)
6. Eğitilen modeli `sonuclar/model/best_model/` klasörüne indirin

#### Lokal (GPU varsa)

```bash
python model/train.py
```

---

### 5. Tahmin / Analiz

Kayıtlı ürünler için JSON çıktısı üret:

```bash
python model/predict.py
```

---

### 6. Streamlit Arayüzü

```bash
streamlit run app.py
```

`http://localhost:8501` adresinde açılır.

**Yeni Ürün Analizi** sekmesinde:
1. HepsiBurada ürün SKU kodunu gir
2. Kategoriyi seç (Ayakkabı veya Kıyafet)
3. "Analiz Et" butonuna bas

Sistem yorumları çekip **kategori bazlı doğal dil özeti** üretir:

```
Değerlendirme Özeti — 👟 Ayakkabı
• Ayakkabıların rahatlığı ve konforu beğeniliyor, günlük kullanım için uygun bulunuyor.
• Kalıp dar; kullanıcılar yarım veya bir numara büyük almayı tavsiye ediyor.
• Fiyat-performans oranı olumlu değerlendiriliyor; indirimli alım yapanlar memnun.
• Ayakkabının derisi ve malzemesi kaliteli bulunuyor.
• Kargo ve paketleme konusunda ciddi şikayetler var.
```

---

## Pipeline Özeti

```
src/scraper/asd.py  (--kategori ayakkabı/kıyafet)
  │  user_contents.csv  [item, order, comment, star, kategori]
  ▼
scripts/etiketleme_olustur.py
  │  data/etiketleme.csv  [id, comment, star, label, kategori]
  ▼
Elle etiketleme  (label: 0 veya 1)
  ▼
model/train.py  →  BERT fine-tuning
  │  sonuclar/model/best_model/
  ▼
model/predict.py  →  Kategori bazlı konu analizi + doğal dil özet
  │  sonuclar/urun_analiz.json
  ▼
app.py  →  Streamlit arayüzü
```
