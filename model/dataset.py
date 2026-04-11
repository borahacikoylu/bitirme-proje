"""
PyTorch Dataset sınıfı.
BERT tokenizer ile yorumları model-girdisi tensor formatına çevirir.
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# ── Sabitler ──
MODEL_ADI = "dbmdz/bert-base-turkish-cased"

# BERT maksimum 512 token destekler; kısa yorumlar için 128 yeterli
MAX_UZUNLUK = 128


class YorumDataset(Dataset):
    """
    Her bir yorumu tokenize edip BERT'e uygun tensor sözlüğü döndürür.

    Parametreler
    ----------
    metinler   : Yorum metinlerinin listesi
    etiketler  : 0 (negatif) / 1 (pozitif) etiket listesi
    tokenizer  : HuggingFace BertTokenizer (None ise varsayılan yüklenir)
    max_uzunluk: Maksimum token sayısı
    """

    def __init__(
        self,
        metinler: list[str],
        etiketler: list[int],
        tokenizer: BertTokenizer | None = None,
        max_uzunluk: int = MAX_UZUNLUK,
    ):
        self.metinler = metinler
        self.etiketler = etiketler
        self.max_uzunluk = max_uzunluk

        # Tokenizer verilmezse varsayılanı HuggingFace Hub'dan indir
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(MODEL_ADI)

    def __len__(self) -> int:
        """Veri kümesindeki toplam örnek sayısı."""
        return len(self.metinler)

    def __getitem__(self, idx: int) -> dict:
        """
        Tek bir örneği BERT girdisine dönüştürür.

        Döndürdüğü anahtarlar:
          - input_ids      : Token ID'leri             (shape: max_uzunluk)
          - attention_mask  : Gerçek token = 1, pad = 0 (shape: max_uzunluk)
          - labels          : 0 veya 1                   (skaler)
        """
        metin = str(self.metinler[idx])
        etiket = int(self.etiketler[idx])

        kodlama = self.tokenizer(
            metin,
            max_length=self.max_uzunluk,
            padding="max_length",      # Kısa metinleri [PAD] ile doldur
            truncation=True,           # Uzun metinleri kes
            return_tensors="pt",       # PyTorch tensörü döndür
        )

        return {
            "input_ids": kodlama["input_ids"].squeeze(0),
            "attention_mask": kodlama["attention_mask"].squeeze(0),
            "labels": torch.tensor(etiket, dtype=torch.long),
        }
