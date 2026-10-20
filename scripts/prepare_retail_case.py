"""从 UCI Online Retail II 生成案例数据（德国 + 法国，2011）。

用法：
    python scripts/prepare_retail_case.py --source path/to/online_retail_II.xlsx

原始文件不进仓库（45 MB）；这里只做筛选与改名，不改金额、不补缺失、不删退货行。
处理口径与标准答案见 docs/case-study.md。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COUNTRIES = ("Germany", "France")
START = pd.Timestamp("2011-01-01")
END = pd.Timestamp("2012-01-01")
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "retail"


def build(source: Path, out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_excel(source, sheet_name="Year 2010-2011")
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")

    subset = frame[
        frame["Country"].isin(COUNTRIES)
        & (frame["InvoiceDate"] >= START)
        & (frame["InvoiceDate"] < END)
    ].copy()

    orders = subset.rename(
        columns={
            "Invoice": "invoice_id",
            "StockCode": "stock_code",
            "Description": "description",
            "Quantity": "quantity",
            "InvoiceDate": "invoice_ts",
            "Price": "unit_price",
            "Customer ID": "customer_id",
            "Country": "country",
        }
    )
    orders["customer_id"] = orders["customer_id"].astype("Int64")
    # 原始 StockCode 有纯数字也有字母码，统一成字符串
    orders["stock_code"] = orders["stock_code"].astype(str).str.strip()
    orders["is_return"] = (orders["quantity"] < 0).astype(int)
    orders["invoice_ts"] = orders["invoice_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    orders = orders[
        [
            "invoice_id",
            "stock_code",
            "description",
            "quantity",
            "invoice_ts",
            "unit_price",
            "customer_id",
            "country",
            "is_return",
        ]
    ].sort_values(["invoice_ts", "invoice_id", "stock_code"])

    orders_path = out_dir / "orders_eu_2011.csv"
    orders.to_csv(orders_path, index=False, encoding="utf-8")

    products = (
        orders.dropna(subset=["stock_code"])
        .groupby("stock_code", as_index=False)
        .agg(
            description=(
                "description",
                lambda values: values.dropna().iloc[0] if values.notna().any() else "",
            ),
            median_price=("unit_price", "median"),
            first_seen=("invoice_ts", "min"),
        )
        .sort_values("stock_code")
    )
    products_path = out_dir / "products.csv"
    products.to_csv(products_path, index=False, encoding="utf-8")

    return orders_path, products_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成电商案例数据")
    parser.add_argument("--source", required=True, type=Path, help="online_retail_II.xlsx 路径")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"找不到源文件：{args.source}")

    orders_path, products_path = build(args.source)
    print(f"写出 {orders_path}（{orders_path.stat().st_size / 1024:.0f} KB）")
    print(f"写出 {products_path}（{products_path.stat().st_size / 1024:.0f} KB）")


if __name__ == "__main__":
    main()
