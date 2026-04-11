# Türkçe Yorum Sentiment Analizi

HepsiBurada ürün yorumlarını **pozitif** ve **negatif** cümlelere ayıran BERT tabanlı NLP sistemi.

**Model:** [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased)

---

## Proje Yapısı

```
bitirme-proje/
├── src/scraper/              ← HepsiBurada yorum scraper
│   ├── asd.py
│   └── data/user_contents.csv
├── data/
│   └── preprocess.py         ← CSV okuma, temizleme, train/test split
├── model/
│   ├── dataset.py            ← PyTorch Dataset sınıfı
│   ├── train.py              ← BERT fine-tuning (Trainer API)
│   └── predict.py            ← Yorum → artılar/eksiler pipeline
├── notebooks/
│   └── colab_egitim.ipynb    ← Adım adım Colab notebook
├── sonuclar/                 ← Eğitim sonrası otomatik oluşur
│   ├── model/best_model/
│   └── urun_analiz.json
├── app.py                    ← Streamlit demo arayüzü
├── requirements.txt
└── README.md
```

---

## Kurulum

```bash
# Repoyu klonla
git clone <repo-url>
cd bitirme-proje

# Bağımlılıkları kur
pip install -r requirements.txt
```

> **Not:** GPU destekli PyTorch için [pytorch.org](https://pytorch.org/get-started/locally/) adresinden uygun CUDA sürümünü seçip kurun.

---

## Colab'da Nasıl Çalıştırılır

1. [Google Colab](https://colab.research.google.com/) açın
2. **File → Upload notebook** ile `notebooks/colab_egitim.ipynb` dosyasını yükleyin
3. **Runtime → Change runtime type → T4 GPU** seçin
4. Hücreleri sırayla çalıştırın:
   - İlk hücre CSV dosyasını yüklemenizi isteyecek → `src/scraper/data/user_contents.csv` dosyasını seçin
   - Eğitim T4 GPU'da yaklaşık 2-3 dakika sürer
   - Son hücre eğitilmiş modeli ve JSON çıktısını indirir

---

## Lokal Eğitim (GPU varsa)

```bash
# Proje kökünden çalıştır
python model/train.py
```

Eğitim tamamlandığında model `sonuclar/model/best_model/` dizinine kaydedilir.

---

## Tahmin (JSON Çıktısı Üretme)

```bash
python model/predict.py
```

Bu komut:
- Eğitilmiş modeli yükler
- Tüm yorumları cümlelere böler
- Her cümleyi pozitif/negatif olarak sınıflandırır
- `sonuclar/urun_analiz.json` dosyasını üretir

---

## Streamlit Demo

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresinde açılır. Ürün ID'si seçildiğinde artılar ve eksiler yan yana gösterilir.

---

## Örnek Çıktı

```json
{
  "HBCV00005QG5TV": {
    "artilar": [
      "Ürün çok güzel tam hayak ettiğim gibi indirimle birlikte çok iyi bir fiyata denk geldi.",
      "Çok rahat bir ayakkabı.",
      "Barkodu okuttum adidas sitesine yönlendirdi.",
      "Kargo hızlıydı."
    ],
    "eksiler": [
      "Ürün sahte almayın kesinlikle ve iğrenç kokuyor.",
      "Kalıbı küçük geldi.",
      "Kutusu paramparçaydı.",
      "Arkası ayağımı vurdu."
    ]
  }
}
```

---

## Pipeline Özeti

```
CSV (scraper çıktısı)
  │
  ▼
data/preprocess.py     →  Temizleme + Train/Test split
  │
  ▼
model/dataset.py       →  PyTorch Dataset (tokenize)
  │
  ▼
model/train.py         →  BERT fine-tuning (3 epoch)
  │
  ▼
model/predict.py       →  Cümle bazlı sınıflandırma → JSON
  │
  ▼
app.py                 →  Streamlit ile görselleştirme
```
