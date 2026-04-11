"""
CSV okuma, metin temizleme ve eğitim/test veri kümesi oluşturma modülü.

İki mod destekler:
  1. Elle etiketlenmiş veri (data/etiketleme.csv) → label sütunu dolu
  2. Otomatik etiketleme (yıldız tabanlı)        → yedek mod
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Proje kök dizini (bu dosyanın bir üst klasörü) ──
PROJE_KOKU = Path(__file__).resolve().parent.parent

# Elle etiketlenmiş veri (birincil kaynak)
ETIKETLEME_YOLU = PROJE_KOKU / "data" / "etiketleme.csv"

# Scraper'ın ürettiği ham CSV (tahmin aşamasında kullanılır)
CSV_YOLU = PROJE_KOKU / "src" / "scraper" / "data" / "user_contents.csv"


def csv_oku(csv_yolu: str | Path = CSV_YOLU) -> pd.DataFrame:
    """
    Scraper çıktısı CSV'yi okur.
    Sütunlar: item, order, comment, star
    """
    df = pd.read_csv(csv_yolu, encoding="utf-8")
    print(f"  [csv_oku] {len(df)} yorum okundu.")
    return df


def metin_temizle(metin: str) -> str:
    """
    Tek bir yorum metnini temizler:
      - Satır sonlarını boşluğa çevirir
      - Emoji ve kontrol karakterlerini boşlukla değiştirir
      - Ardışık boşlukları teke indirir
    """
    if not isinstance(metin, str):
        return ""

    metin = metin.replace("\n", " ").replace("\r", " ")
    metin = re.sub(r"[^\w\s.,!?;:'\"\-()/]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()

    return metin


def etiketleme_csv_oku(
    csv_yolu: str | Path = ETIKETLEME_YOLU,
) -> pd.DataFrame:
    """
    Elle etiketlenmiş CSV'yi okur.
    Beklenen sütunlar: id, comment, star, label
    Sadece label sütunu dolu olan satırları alır.
    """
    df = pd.read_csv(csv_yolu, encoding="utf-8")

    # label sütunu boş olan (henüz etiketlenmemiş) satırları çıkar
    df = df.dropna(subset=["label"])
    df = df[df["label"].astype(str).str.strip() != ""]

    # label'ı int'e çevir (0 veya 1)
    df["label"] = df["label"].astype(int)

    # Geçersiz etiketleri filtrele
    df = df[df["label"].isin([0, 1])]

    print(f"  [etiketleme_csv_oku] {len(df)} etiketli yorum okundu.")
    return df


def veriyi_hazirla(
    csv_yolu: str | Path | None = None,
    test_orani: float = 0.2,
    rastgele_tohum: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uçtan uca veri hazırlama pipeline'ı.

    Öncelik sırası:
      1. Elle etiketlenmiş data/etiketleme.csv varsa ve içinde etiket
         doluysa → onu kullan
      2. Yoksa → csv_yolu parametresindeki ham CSV'den yıldız tabanlı
         etiketleme yap (eski yöntem, yedek)

    Returns
    -------
    (train_df, test_df)  –  her ikisi de ``comment`` ve ``label`` sütunlu
    """

    # ── 1) Elle etiketlenmiş veri var mı kontrol et ──
    etiketleme_dosyasi = Path(csv_yolu) if csv_yolu else ETIKETLEME_YOLU

    if etiketleme_dosyasi.exists() and etiketleme_dosyasi.suffix == ".csv":
        df_test = pd.read_csv(etiketleme_dosyasi, encoding="utf-8")
        # label sütunu var ve en az bir satır doluysa elle etiketleme modunu kullan
        if "label" in df_test.columns and df_test["label"].notna().any():
            return _elle_etiketli_hazirla(etiketleme_dosyasi, test_orani, rastgele_tohum)

    # ── 2) Yedek: yıldız tabanlı etiketleme ──
    print("  [!] Elle etiketlenmiş veri bulunamadı, yıldız tabanlı etiketleme kullanılıyor.")
    ham_csv = Path(csv_yolu) if csv_yolu else CSV_YOLU
    return _yildiz_tabanlı_hazirla(ham_csv, test_orani, rastgele_tohum)


def _elle_etiketli_hazirla(
    csv_yolu: Path,
    test_orani: float,
    rastgele_tohum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elle etiketlenmiş CSV'den train/test hazırlar."""
    print("=" * 55)
    print("Elle etiketlenmiş veri kullanılıyor")
    print("=" * 55)

    df = etiketleme_csv_oku(csv_yolu)

    # Metin temizleme
    df["comment"] = df["comment"].apply(metin_temizle)

    # Boş yorumları çıkar
    df = df[df["comment"].str.strip().astype(bool)]

    # Dağılım
    dagilim = df["label"].value_counts()
    print(f"  Pozitif (1): {dagilim.get(1, 0)}")
    print(f"  Negatif (0): {dagilim.get(0, 0)}")

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df,
        test_size=test_orani,
        random_state=rastgele_tohum,
        stratify=df["label"],
    )

    print(f"  Train: {len(train_df)},  Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _yildiz_tabanlı_hazirla(
    csv_yolu: Path,
    test_orani: float,
    rastgele_tohum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yıldız puanlarından otomatik etiket üretir (yedek yöntem)."""
    print("=" * 55)
    print("Yıldız tabanlı otomatik etiketleme (yedek mod)")
    print("=" * 55)

    df = csv_oku(csv_yolu)

    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip().astype(bool)]

    df["comment"] = df["comment"].apply(metin_temizle)

    # 1-2 → negatif (0),  4-5 → pozitif (1),  3 → çıkar
    df["label"] = df["star"].apply(
        lambda y: 0 if y <= 2 else (1 if y >= 4 else -1)
    )
    df = df[df["label"] != -1].copy()

    dagilim = df["label"].value_counts()
    print(f"  Pozitif (1): {dagilim.get(1, 0)},  Negatif (0): {dagilim.get(0, 0)}")

    train_df, test_df = train_test_split(
        df,
        test_size=test_orani,
        random_state=rastgele_tohum,
        stratify=df["label"],
    )

    print(f"  Train: {len(train_df)},  Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    train_df, test_df = veriyi_hazirla()
    print("\n── Train (ilk 5) ──")
    print(train_df[["comment", "label"]].head())
    print("\n── Test  (ilk 5) ──")
    print(test_df[["comment", "label"]].head())
