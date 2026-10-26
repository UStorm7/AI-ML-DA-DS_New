"""Download the Olist Brazilian E-Commerce dataset (public, CC BY-NC-SA 4.0).

Source: https://github.com/olist/work-at-olist-data (also on Kaggle as
"Brazilian E-Commerce Public Dataset by Olist").
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"
FILES = [
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if target.exists():
            print(f"skip  {name} (already downloaded)")
            continue
        print(f"fetch {name}")
        urlretrieve(f"{BASE}/{name}", target)
    print(f"Done -> {RAW_DIR}")


if __name__ == "__main__":
    main()
