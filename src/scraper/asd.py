import requests
import csv
import re
import json
from pathlib import Path


def get_user_contents(sku: str, from_value: int = 0, size: int = 100):
    url = "https://user-content-gw-hermes.hepsiburada.com/queryapi/v2/ApprovedUserContents"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    }

    params = {
        "sku": sku,
        "from": from_value,
        "size": size,
        "includeSiblingVariantContents": "true",
        "includeSummary": "true",
    }

    response = requests.get(
        url,
        headers=headers,
        params=params,
        allow_redirects=False,
    )

    return response


def _extract_total_item_count(raw_text: str) -> int:
    match = re.search(r'"totalItemCount"\s*:\s*(\d+)', raw_text)
    if not match:
        return 0
    return int(match.group(1))


def _extract_reviews(raw_text: str) -> list[dict[str, object]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    items = (
        payload.get("data", {})
        .get("approvedUserContent", {})
        .get("approvedUserContentList", [])
    )

    reviews: list[dict[str, object]] = []
    for item in items:
        comment = (item.get("review") or {}).get("content")
        if comment is None or not str(comment).strip():
            continue
        reviews.append(
            {
                "comment": str(comment).strip(),
                "star": item.get("star"),
            }
        )
    return reviews


def _write_comments_to_csv(
    sku: str, reviews: list[dict[str, object]], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item", "order", "comment", "star"]
    existing_rows: list[dict[str, object]] = []
    write_mode = "a"

    if not output_path.exists() or output_path.stat().st_size == 0:
        write_mode = "w"
    else:
        with output_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                write_mode = "w"
                for row in reader:
                    existing_rows.append(
                        {
                            "item": row.get("item"),
                            "order": row.get("order"),
                            "comment": row.get("comment"),
                            "star": row.get("star", ""),
                        }
                    )

    with output_path.open(write_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
            if existing_rows:
                writer.writerows(existing_rows)
        for idx, review in enumerate(reviews, start=1):
            writer.writerow(
                {
                    "item": sku,
                    "order": idx,
                    "comment": review["comment"],
                    "star": review.get("star"),
                }
            )


def _sku_zaten_var(sku: str, csv_path: Path) -> bool:
    """CSV'de bu SKU'nun yorumları zaten varsa True döndürür."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("item") == sku:
                return True
    return False


def get_all_user_contents(sku: str) -> tuple[int, int]:
    csv_path = Path(__file__).resolve().parent / "data" / "user_contents.csv"

    if _sku_zaten_var(sku, csv_path):
        print(f"  [ATLANDI] {sku} zaten CSV'de mevcut.")
        return 0, 0

    from_value = 0
    size = 100
    total_item_count = 0
    all_reviews: list[dict[str, object]] = []

    while True:
        response = get_user_contents(sku=sku, from_value=from_value, size=size)
        if response.status_code != 200:
            break

        raw_text = response.text
        if total_item_count == 0:
            total_item_count = _extract_total_item_count(raw_text)

        reviews = _extract_reviews(raw_text)
        all_reviews.extend(reviews)

        from_value += size
        if (total_item_count - from_value) <= 0:
            break

    _write_comments_to_csv(sku=sku, reviews=all_reviews, output_path=csv_path)
    return total_item_count, len(all_reviews)


def yorumlari_cek(sku: str) -> list[dict[str, object]]:
    """
    Verilen SKU için tüm yorumları çeker ve liste olarak döndürür.
    CSV'ye yazmaz — doğrudan bellekte kullanım için.
    """
    from_value = 0
    size = 100
    total_item_count = 0
    all_reviews: list[dict[str, object]] = []

    while True:
        response = get_user_contents(sku=sku, from_value=from_value, size=size)
        if response.status_code != 200:
            break

        raw_text = response.text
        if total_item_count == 0:
            total_item_count = _extract_total_item_count(raw_text)

        reviews = _extract_reviews(raw_text)
        all_reviews.extend(reviews)

        from_value += size
        if (total_item_count - from_value) <= 0:
            break

    return all_reviews


if __name__ == "__main__":
    # Yorumlarını çekmek istediğin ürün ID'lerini buraya ekle
    URUN_LISTESI = [
        "HBCV00005QG5TV",
        "HBCV0000CJ0SKA",
        "HBCV000095F10F",
        "HBV00000GU1HH",
    ]

    toplam_yorum = 0
    for sku in URUN_LISTESI:
        print(f"\n{'─'*40}")
        print(f"Ürün: {sku}")
        item_count, comment_count = get_all_user_contents(sku)
        print(f"  totalItemCount : {item_count}")
        print(f"  saved comments : {comment_count}")
        toplam_yorum += comment_count

    print(f"\n{'═'*40}")
    print(f"TOPLAM: {toplam_yorum} yorum kaydedildi.")
