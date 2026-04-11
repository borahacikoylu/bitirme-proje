"""
BERT fine-tuning modülü.
HuggingFace Trainer API ile dbmdz/bert-base-turkish-cased modelini
Türkçe yorum sentiment analizi için eğitir.

Sınıf dengesizliğini çözmek için ağırlıklı CrossEntropyLoss kullanır.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Proje kökünü Python yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.preprocess import veriyi_hazirla
from model.dataset import YorumDataset, MODEL_ADI

# ── Sabitler ──
PROJE_KOKU = Path(__file__).resolve().parent.parent
CIKTI_DIZINI = PROJE_KOKU / "sonuclar" / "model"

EPOCH_SAYISI = 5       # Elle etiketli veri için 5 epoch daha iyi sonuç verir
BATCH_BOYUTU = 16      # Colab T4 GPU belleğine uygun
OGRENME_HIZI = 2e-5    # BERT fine-tuning için standart değer


class DengeliTrainer(Trainer):
    """
    Sınıf dengesizliğini çözen özel Trainer.

    Azınlık sınıfına (negatif) daha fazla ağırlık verir,
    böylece model "hep pozitif de" stratejisini öğrenemez.
    """

    def __init__(self, sinif_agirliklari=None, **kwargs):
        super().__init__(**kwargs)
        self.sinif_agirliklari = sinif_agirliklari

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.sinif_agirliklari is not None:
            weight = torch.tensor(
                self.sinif_agirliklari, dtype=torch.float32
            ).to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def metrikleri_hesapla(eval_pred) -> dict:
    """
    Trainer her epoch sonunda bu fonksiyonu çağırır.
    Accuracy, F1, Precision ve Recall hesaplar.
    """
    tahminler, gercekler = eval_pred
    tahmin_etiketleri = np.argmax(tahminler, axis=1)

    return {
        "accuracy": accuracy_score(gercekler, tahmin_etiketleri),
        "f1": f1_score(gercekler, tahmin_etiketleri, average="binary"),
        "precision": precision_score(gercekler, tahmin_etiketleri, average="binary"),
        "recall": recall_score(gercekler, tahmin_etiketleri, average="binary"),
    }


def sinif_agirliklarini_hesapla(etiketler: list[int]) -> list[float]:
    """
    Sınıf ağırlıklarını hesaplar.
    Az temsil edilen sınıfa daha yüksek ağırlık verir.

    Örnek: 200 pozitif, 100 negatif varsa
      → negatif ağırlığı: 300 / (2 * 100) = 1.5
      → pozitif ağırlığı: 300 / (2 * 200) = 0.75
    """
    toplam = len(etiketler)
    sinif_sayilari = {}
    for e in etiketler:
        sinif_sayilari[e] = sinif_sayilari.get(e, 0) + 1

    sinif_sayisi = len(sinif_sayilari)
    agirliklar = [0.0, 0.0]
    for sinif, sayi in sinif_sayilari.items():
        agirliklar[sinif] = toplam / (sinif_sayisi * sayi)

    print(f"  Sınıf ağırlıkları → Negatif: {agirliklar[0]:.2f}, Pozitif: {agirliklar[1]:.2f}")
    return agirliklar


def modeli_egit(
    csv_yolu: str | Path | None = None,
    cikti_dizini: str | Path = CIKTI_DIZINI,
    epoch: int = EPOCH_SAYISI,
    batch: int = BATCH_BOYUTU,
    lr: float = OGRENME_HIZI,
):
    """
    Uçtan uca eğitim pipeline'ı.

    Adımlar:
      1. Veriyi hazırla (elle etiketli CSV veya yıldız tabanlı)
      2. Tokenizer ve önceden eğitilmiş modeli yükle
      3. PyTorch Dataset nesneleri oluştur
      4. Sınıf ağırlıklarını hesapla (dengesizlik çözümü)
      5. DengeliTrainer ile fine-tuning yap
      6. Test setinde değerlendir
      7. En iyi modeli diske kaydet
    """
    # ── 1) Veri hazırlama ──
    print("=" * 55)
    print("ADIM 1 ▸ Veri hazırlanıyor")
    print("=" * 55)
    if csv_yolu:
        train_df, test_df = veriyi_hazirla(csv_yolu)
    else:
        train_df, test_df = veriyi_hazirla()

    # ── 2) Tokenizer ve model yükleme ──
    print("\nADIM 2 ▸ Tokenizer ve model yükleniyor...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_ADI)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_ADI,
        num_labels=2,
    )

    # ── 3) Dataset oluştur ──
    print("ADIM 3 ▸ Dataset nesneleri oluşturuluyor...")
    train_dataset = YorumDataset(
        metinler=train_df["comment"].tolist(),
        etiketler=train_df["label"].tolist(),
        tokenizer=tokenizer,
    )
    test_dataset = YorumDataset(
        metinler=test_df["comment"].tolist(),
        etiketler=test_df["label"].tolist(),
        tokenizer=tokenizer,
    )

    # ── 4) Sınıf ağırlıkları ──
    print("\nADIM 3b ▸ Sınıf ağırlıkları hesaplanıyor...")
    agirliklar = sinif_agirliklarini_hesapla(train_df["label"].tolist())

    # ── 5) Eğitim ayarları ──
    cikti_dizini = Path(cikti_dizini)
    cikti_dizini.mkdir(parents=True, exist_ok=True)

    egitim_ayarlari = TrainingArguments(
        output_dir=str(cikti_dizini),
        num_train_epochs=epoch,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        seed=42,
        report_to="none",
    )

    trainer = DengeliTrainer(
        sinif_agirliklari=agirliklar,
        model=model,
        args=egitim_ayarlari,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrikleri_hesapla,
    )

    # ── 6) Eğitimi başlat ──
    print(f"\nADIM 4 ▸ Eğitim başlıyor  (epoch={epoch}, batch={batch}, lr={lr})")
    print("-" * 55)
    trainer.train()

    # ── 7) Test seti değerlendirmesi ──
    print("\nADIM 5 ▸ Test seti değerlendirmesi")
    print("-" * 55)
    sonuclar = trainer.evaluate()
    for key, val in sonuclar.items():
        if isinstance(val, float):
            print(f"  {key:25s} : {val:.4f}")

    # ── 8) Modeli kaydet ──
    kayit_yolu = cikti_dizini / "best_model"
    trainer.save_model(str(kayit_yolu))
    tokenizer.save_pretrained(str(kayit_yolu))
    print(f"\nADIM 6 ▸ Model kaydedildi → {kayit_yolu}")

    return trainer, sonuclar


if __name__ == "__main__":
    modeli_egit()
