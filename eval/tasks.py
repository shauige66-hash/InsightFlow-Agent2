"""P5-A：标准评测任务集合。

## 设计原则

1. **标准答案独立算出。** 每个任务的期望值由本模块用纯 Python 从同一份
   原始数据直接计算，**不经过被测工作流的任何代码路径**。
   拿 Agent 自己的输出去当答案等于什么都没测。

2. **任务之间必须真的有区别。** 本模块只收录**语义不同**的任务：
   不同的数据源类型、不同的算子（聚合 / 过滤 / 分组 / 连接 / Top-N /
   去重计数 / 极值 / 派生列）、不同的输出形态（标量 / 分组表 / 明细 / 图表）。
   参数的机械复制（只换一个数字）不算新任务，宁可不凑数。

3. **可重复。** 数据由代码生成、不依赖网络与随机数；同样的任务在任何
   机器上跑出同样的期望值。

## 与"评测真实模型能力"的关系

这些任务用 Mock LLM 执行时，验证的是**工程链路与判分逻辑**，
**不能**用来声称真实模型的自主分析能力。真实模型的评测入口见
`scripts/run_eval.py --provider real`（需要显式提供 API Key 与预算）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 固定的评测数据（由代码生成，可重复）
# ---------------------------------------------------------------------------

REGIONS = ("East", "West", "North", "South")
CATEGORIES = ("hardware", "software", "service")

#: CSV 行：(order_id, region, category, units, unit_price)
CSV_ROWS: list[tuple[int, str, str, int, float]] = []
for _index in range(120):
    CSV_ROWS.append(
        (
            1000 + _index,
            REGIONS[_index % len(REGIONS)],
            CATEGORIES[_index % len(CATEGORIES)],
            (_index % 9) + 1,
            round(10.0 + (_index % 7) * 2.5, 2),
        )
    )

#: 小数金额先量化再求和，避免浮点累加在不同实现下产生最后一位差异
def _money(value: float) -> float:
    return round(value, 2)


CSV_REVENUE: dict[int, float] = {
    row[0]: _money(row[3] * row[4]) for row in CSV_ROWS
}


@dataclass(frozen=True)
class EvalTask:
    """一个可自动判分的评测任务。

    `expected` 是**独立算出**的标准答案；`check` 说明怎么比。
    判分逻辑集中在这里，避免每个任务各写一套比较代码而产生口径差异。
    """

    task_id: str
    family: str
    description: str
    data_source: str  # csv | excel | sqlite
    target_tables: list[str]
    code: str
    expected: Any
    #: scalar / rows / rows_grouped
    shape: str = "scalar"
    #: 用于取值的列（rows_grouped 时是分组键与指标列）
    key_column: str | None = None
    value_column: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# CSV 与 Excel 的标准答案
# ---------------------------------------------------------------------------


def csv_total_revenue() -> float:
    return _money(sum(CSV_REVENUE.values()))


def csv_revenue_by_region() -> dict[str, float]:
    grouped: dict[str, float] = {}
    for order_id, region, _category, units, price in CSV_ROWS:
        grouped[region] = _money(grouped.get(region, 0.0) + _money(units * price))
    return grouped


def csv_revenue_by_category() -> dict[str, float]:
    grouped: dict[str, float] = {}
    for _order_id, _region, category, units, price in CSV_ROWS:
        grouped[category] = _money(grouped.get(category, 0.0) + _money(units * price))
    return grouped


def csv_orders_in_east() -> int:
    return sum(1 for row in CSV_ROWS if row[1] == "East")


def csv_units_over_5() -> int:
    return sum(1 for row in CSV_ROWS if row[3] > 5)


def csv_max_row() -> dict[str, Any]:
    best = max(CSV_ROWS, key=lambda row: _money(row[3] * row[4]))
    return {"order_id": best[0], "revenue": _money(best[3] * best[4])}


def csv_top3_regions() -> list[tuple[str, float]]:
    by_region = csv_revenue_by_region()
    ordered = sorted(by_region.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:3]


def csv_hardware_east_count() -> int:
    return sum(1 for row in CSV_ROWS if row[2] == "hardware" and row[1] == "East")


def csv_avg_units() -> float:
    return round(sum(row[3] for row in CSV_ROWS) / len(CSV_ROWS), 4)


def csv_distinct_categories() -> int:
    return len({row[2] for row in CSV_ROWS})


def csv_min_revenue() -> float:
    return min(CSV_REVENUE.values())


def csv_revenue_std() -> float:
    values = list(CSV_REVENUE.values())
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return round(variance ** 0.5, 4)


def csv_east_share() -> float:
    total = csv_total_revenue()
    east = csv_revenue_by_region()["East"]
    return round(east / total, 6)


def csv_units_sum() -> int:
    return sum(row[3] for row in CSV_ROWS)


def csv_service_count() -> int:
    return sum(1 for row in CSV_ROWS if row[2] == "service")


def csv_both_east_west() -> int:
    return sum(1 for row in CSV_ROWS if row[1] in ("East", "West"))


def csv_revenue_spread() -> float:
    return _money(max(CSV_REVENUE.values()) - min(CSV_REVENUE.values()))


def csv_revenue_by_region_category() -> dict[str, float]:
    """双键分组：`region|category` → 营收。"""
    grouped: dict[str, float] = {}
    for _order, region, category, units, price in CSV_ROWS:
        key = f"{region}|{category}"
        grouped[key] = _money(grouped.get(key, 0.0) + _money(units * price))
    return grouped


def csv_top2_categories() -> dict[str, float]:
    by_category = csv_revenue_by_category()
    ordered = sorted(by_category.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:2])


def csv_high_value_order_count() -> int:
    """派生列 + 过滤：营收超过 40 的订单数。"""
    return sum(1 for value in CSV_REVENUE.values() if value > 40.0)


def csv_east_avg_units() -> float:
    units = [row[3] for row in CSV_ROWS if row[1] == "East"]
    return round(sum(units) / len(units), 4)


def csv_lowest_region() -> str:
    by_region = csv_revenue_by_region()
    return min(by_region.items(), key=lambda item: (item[1], item[0]))[0]


#: Excel 两张 sheet 的数据（与 CSV 不同，单独定义，避免"同数据换格式"）
EXCEL_ORDERS: list[tuple[int, str, int, float]] = [
    (1, "East", 2, 10.0),
    (2, "West", 1, 25.0),
    (3, "East", 5, 10.0),
    (4, "North", 3, 15.0),
    (5, "South", 4, 12.5),
    (6, "West", 2, 20.0),
]
EXCEL_REGIONS: list[tuple[str, str]] = [
    ("East", "Alice"),
    ("West", "Bob"),
    ("North", "Cara"),
    ("South", "Dan"),
]


def excel_total_revenue() -> float:
    return _money(sum(_money(units * price) for _o, _r, units, price in EXCEL_ORDERS))


def excel_east_revenue() -> float:
    return _money(
        sum(
            _money(units * price)
            for _o, region, units, price in EXCEL_ORDERS
            if region == "East"
        )
    )


def excel_order_count() -> int:
    return len(EXCEL_ORDERS)


def excel_revenue_by_manager() -> dict[str, float]:
    manager_of = dict(EXCEL_REGIONS)
    grouped: dict[str, float] = {}
    for _order, region, units, price in EXCEL_ORDERS:
        manager = manager_of[region]
        grouped[manager] = _money(grouped.get(manager, 0.0) + _money(units * price))
    return grouped


def excel_regions_sheet_rows() -> int:
    return len(EXCEL_REGIONS)


# ---------------------------------------------------------------------------
# SQLite 的标准答案
# ---------------------------------------------------------------------------

SQLITE_CUSTOMERS: list[tuple[int, str, str]] = [
    (1, "Alice", "East"),
    (2, "Bob", "West"),
    (3, "Cara", "East"),
    (4, "Dan", "North"),
    (5, "Eve", "South"),
]
SQLITE_ORDERS: list[tuple[int, int, float]] = [
    (101, 1, 10.0),
    (102, 1, 15.5),
    (103, 2, 20.0),
    (104, 3, 5.25),
    (105, 4, 100.0),
    (106, 5, 42.0),
    (107, 2, 7.5),
]
#: 一张**未授权**的表：用于确认授权边界在评测里也没被放宽
SQLITE_SECRETS: list[tuple[str]] = [("top-secret",)]


def sqlite_total() -> float:
    return _money(sum(amount for _o, _c, amount in SQLITE_ORDERS))


def sqlite_order_count() -> int:
    return len(SQLITE_ORDERS)


def sqlite_revenue_by_region() -> dict[str, float]:
    region_of = {cid: region for cid, _name, region in SQLITE_CUSTOMERS}
    grouped: dict[str, float] = {}
    for _order, customer, amount in SQLITE_ORDERS:
        region = region_of[customer]
        grouped[region] = _money(grouped.get(region, 0.0) + amount)
    return grouped


def sqlite_revenue_by_customer() -> dict[str, float]:
    name_of = {cid: name for cid, name, _region in SQLITE_CUSTOMERS}
    grouped: dict[str, float] = {}
    for _order, customer, amount in SQLITE_ORDERS:
        name = name_of[customer]
        grouped[name] = _money(grouped.get(name, 0.0) + amount)
    return grouped


def sqlite_orders_over_15() -> int:
    return sum(1 for _o, _c, amount in SQLITE_ORDERS if amount > 15.0)


def sqlite_max_amount() -> float:
    return _money(max(amount for _o, _c, amount in SQLITE_ORDERS))


def sqlite_east_total() -> float:
    region_of = {cid: region for cid, _name, region in SQLITE_CUSTOMERS}
    return _money(
        sum(
            amount
            for _order, customer, amount in SQLITE_ORDERS
            if region_of[customer] == "East"
        )
    )


def sqlite_avg_amount() -> float:
    return round(sum(amount for _o, _c, amount in SQLITE_ORDERS) / len(SQLITE_ORDERS), 4)


def sqlite_distinct_customers() -> int:
    return len({customer for _o, customer, _a in SQLITE_ORDERS})


def sqlite_min_amount() -> float:
    return _money(min(amount for _o, _c, amount in SQLITE_ORDERS))


def sqlite_orders_between() -> int:
    return sum(1 for _o, _c, amount in SQLITE_ORDERS if 10.0 <= amount <= 50.0)


def sqlite_high_value_customers() -> dict[str, float]:
    """连接 + 分组 + 过滤：金额超过 15 的客户合计。"""
    name_of = {cid: name for cid, name, _region in SQLITE_CUSTOMERS}
    grouped: dict[str, float] = {}
    for _order, customer, amount in SQLITE_ORDERS:
        if amount <= 15.0:
            continue
        name = name_of[customer]
        grouped[name] = _money(grouped.get(name, 0.0) + amount)
    return grouped


def sqlite_order_count_by_region() -> dict[str, float]:
    region_of = {cid: region for cid, _name, region in SQLITE_CUSTOMERS}
    counts: dict[str, float] = {}
    for _order, customer, _amount in SQLITE_ORDERS:
        region = region_of[customer]
        counts[region] = counts.get(region, 0.0) + 1
    return counts


def sqlite_west_or_north_total() -> float:
    region_of = {cid: region for cid, _name, region in SQLITE_CUSTOMERS}
    return _money(
        sum(
            amount
            for _order, customer, amount in SQLITE_ORDERS
            if region_of[customer] in ("West", "North")
        )
    )


def sqlite_max_order_customer() -> dict[str, Any]:
    best = max(SQLITE_ORDERS, key=lambda row: row[2])
    name_of = {cid: name for cid, name, _region in SQLITE_CUSTOMERS}
    return {"name": name_of[best[1]], "amount": _money(best[2])}


# ---------------------------------------------------------------------------
# 数据落盘
# ---------------------------------------------------------------------------


def write_csv(path: Path) -> Path:
    lines = ["order_id,region,category,units,unit_price"]
    for order_id, region, category, units, price in CSV_ROWS:
        lines.append(f"{order_id},{region},{category},{units},{price}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_excel(path: Path) -> Path:
    import pandas as pd

    orders = pd.DataFrame(
        [
            {"order_id": o, "region": r, "units": u, "unit_price": p}
            for o, r, u, p in EXCEL_ORDERS
        ]
    )
    regions = pd.DataFrame(
        [{"region": r, "manager": m} for r, m in EXCEL_REGIONS]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        orders.to_excel(writer, sheet_name="orders", index=False)
        regions.to_excel(writer, sheet_name="regions", index=False)
    return path


def write_sqlite(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                amount REAL NOT NULL
            );
            CREATE TABLE secrets (value TEXT);
            """
        )
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?)", SQLITE_CUSTOMERS)
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?)", SQLITE_ORDERS)
        connection.executemany("INSERT INTO secrets VALUES (?)", SQLITE_SECRETS)
        connection.commit()
    finally:
        connection.close()
    return path


def materialise(root: Path) -> dict[str, Path]:
    """把评测数据写到 `root` 下，返回 {类型: 路径}。"""
    root.mkdir(parents=True, exist_ok=True)
    return {
        "csv": write_csv(root / "orders.csv"),
        "excel": write_excel(root / "workbook.xlsx"),
        "sqlite": write_sqlite(root / "sales.sqlite"),
    }


# ---------------------------------------------------------------------------
# 任务集合
# ---------------------------------------------------------------------------

_CSV_GROUP_BY_REGION = (
    "frame = df.copy()\n"
    "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
    "result = frame.groupby('region', as_index=False)['revenue'].sum().round(2)\n"
)

_CSV_GROUP_BY_CATEGORY = (
    "frame = df.copy()\n"
    "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
    "result = frame.groupby('category', as_index=False)['revenue'].sum().round(2)\n"
)


def build_tasks() -> list[EvalTask]:
    """构造评测任务集合。

    每个条目都必须与其他条目在**语义**上不同；只换常量的重复条目
    不应出现在这里。
    """
    tasks: list[EvalTask] = [
        # --- CSV：聚合 -------------------------------------------------
        EvalTask(
            task_id="csv_total_revenue",
            family="csv_aggregate",
            description="统计全部订单的总营收",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'total': float((df['units'] * df['unit_price']).round(2).sum().round(2))}]\n",
            expected=csv_total_revenue(),
            shape="scalar",
            value_column="total",
            tags=("aggregate", "derive_column"),
        ),
        EvalTask(
            task_id="csv_revenue_by_region",
            family="csv_group",
            description="按区域汇总营收",
            data_source="csv",
            target_tables=["orders"],
            code=_CSV_GROUP_BY_REGION,
            expected=csv_revenue_by_region(),
            shape="rows_grouped",
            key_column="region",
            value_column="revenue",
            tags=("group_by", "derive_column"),
        ),
        EvalTask(
            task_id="csv_revenue_by_category",
            family="csv_group",
            description="按品类汇总营收",
            data_source="csv",
            target_tables=["orders"],
            code=_CSV_GROUP_BY_CATEGORY,
            expected=csv_revenue_by_category(),
            shape="rows_grouped",
            key_column="category",
            value_column="revenue",
            tags=("group_by", "derive_column"),
        ),
        # --- CSV：过滤 -------------------------------------------------
        EvalTask(
            task_id="csv_east_order_count",
            family="csv_filter",
            description="统计 East 区域的订单数",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int((df['region'] == 'East').sum())}]\n",
            expected=csv_orders_in_east(),
            shape="scalar",
            value_column="value",
            tags=("filter", "count"),
        ),
        EvalTask(
            task_id="csv_units_gt_5",
            family="csv_filter",
            description="统计单量大于 5 的订单数",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int((df['units'] > 5).sum())}]\n",
            expected=csv_units_over_5(),
            shape="scalar",
            value_column="value",
            tags=("filter", "count"),
        ),
        EvalTask(
            task_id="csv_hardware_east_count",
            family="csv_filter",
            description="统计 East 区域的 hardware 品类订单数（双条件）",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "mask = (df['region'] == 'East') & (df['category'] == 'hardware')\n"
                "result = [{'value': int(mask.sum())}]\n"
            ),
            expected=csv_hardware_east_count(),
            shape="scalar",
            value_column="value",
            tags=("filter", "multi_condition"),
        ),
        # --- CSV：极值与排序 -------------------------------------------
        EvalTask(
            task_id="csv_max_revenue_order",
            family="csv_extreme",
            description="找出营收最高的订单及其营收",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "best = frame.loc[frame['revenue'].idxmax()]\n"
                "result = [{'order_id': int(best['order_id']), "
                "'revenue': float(best['revenue'])}]\n"
            ),
            expected=csv_max_row(),
            # 返回的是一行"复合结果"（订单号 + 营收），不是按订单号分组。
            shape="row",
            tags=("extreme", "argmax"),
        ),
        EvalTask(
            task_id="csv_top3_regions",
            family="csv_ranking",
            description="按营收降序取前 3 个区域",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "grouped = frame.groupby('region', as_index=False)['revenue'].sum().round(2)\n"
                "result = grouped.sort_values(['revenue', 'region'], "
                "ascending=[False, True]).head(3)\n"
            ),
            expected=dict(csv_top3_regions()),
            shape="rows_grouped",
            key_column="region",
            value_column="revenue",
            tags=("ranking", "top_n"),
        ),
        EvalTask(
            task_id="csv_min_revenue",
            family="csv_extreme",
            description="找出最低的单笔营收",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'value': float(frame['revenue'].min())}]\n"
            ),
            expected=csv_min_revenue(),
            shape="scalar",
            value_column="value",
            tags=("extreme", "min"),
        ),
        # --- CSV：去重与均值 -------------------------------------------
        EvalTask(
            task_id="csv_distinct_categories",
            family="csv_distinct",
            description="统计不同品类的数量",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int(df['category'].nunique())}]\n",
            expected=csv_distinct_categories(),
            shape="scalar",
            value_column="value",
            tags=("distinct",),
        ),
        EvalTask(
            task_id="csv_avg_units",
            family="csv_aggregate",
            description="计算平均单量（保留 4 位）",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': round(float(df['units'].mean()), 4)}]\n",
            expected=csv_avg_units(),
            shape="scalar",
            value_column="value",
            tags=("aggregate", "mean"),
        ),
        # --- Excel：单表 -----------------------------------------------
        EvalTask(
            task_id="excel_total_revenue",
            family="excel_single_sheet",
            description="Excel 单表：订单总营收",
            data_source="excel",
            target_tables=["workbook.orders"],
            code=(
                "frame = dfs['workbook.orders'].copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'total': float(frame['revenue'].sum().round(2))}]\n"
            ),
            expected=excel_total_revenue(),
            shape="scalar",
            value_column="total",
            tags=("excel", "aggregate"),
        ),
        EvalTask(
            task_id="excel_order_count",
            family="excel_single_sheet",
            description="Excel 单表：订单条数",
            data_source="excel",
            target_tables=["workbook.orders"],
            code="result = [{'value': int(len(dfs['workbook.orders']))}]\n",
            expected=excel_order_count(),
            shape="scalar",
            value_column="value",
            tags=("excel", "count"),
        ),
        EvalTask(
            task_id="excel_east_revenue",
            family="excel_filter",
            description="Excel 单表：East 区域营收",
            data_source="excel",
            target_tables=["workbook.orders"],
            code=(
                "frame = dfs['workbook.orders']\n"
                "frame = frame[frame['region'] == 'East'].copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'value': float(frame['revenue'].sum().round(2))}]\n"
            ),
            expected=excel_east_revenue(),
            shape="scalar",
            value_column="value",
            tags=("excel", "filter"),
        ),
        EvalTask(
            task_id="excel_regions_sheet_rows",
            family="excel_single_sheet",
            description="Excel 指定另一张 sheet：条数",
            data_source="excel",
            target_tables=["workbook.regions"],
            code="result = [{'value': int(len(dfs['workbook.regions']))}]\n",
            expected=excel_regions_sheet_rows(),
            shape="scalar",
            value_column="value",
            tags=("excel", "sheet_selection"),
        ),
        # --- Excel：跨表 -----------------------------------------------
        EvalTask(
            task_id="excel_revenue_by_manager",
            family="excel_cross_sheet",
            description="Excel 跨表：按负责人汇总营收",
            data_source="excel",
            target_tables=["workbook.orders", "workbook.regions"],
            code=(
                "orders = dfs['workbook.orders'].copy()\n"
                "regions = dfs['workbook.regions']\n"
                "merged = orders.merge(regions, on='region', how='left')\n"
                "merged['revenue'] = (merged['units'] * merged['unit_price']).round(2)\n"
                "result = merged.groupby('manager', as_index=False)['revenue'].sum().round(2)\n"
            ),
            expected=excel_revenue_by_manager(),
            shape="rows_grouped",
            key_column="manager",
            value_column="revenue",
            tags=("excel", "join", "group_by"),
        ),
        # --- SQLite：单表 ----------------------------------------------
        EvalTask(
            task_id="sqlite_total",
            family="sqlite_aggregate",
            description="SQLite 单表：订单总额",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT SUM(amount) AS total FROM orders",
            expected=sqlite_total(),
            shape="scalar",
            value_column="total",
            tags=("sqlite", "aggregate"),
        ),
        EvalTask(
            task_id="sqlite_order_count",
            family="sqlite_aggregate",
            description="SQLite 单表：订单数",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT COUNT(*) AS value FROM orders",
            expected=sqlite_order_count(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "count"),
        ),
        EvalTask(
            task_id="sqlite_max_amount",
            family="sqlite_extreme",
            description="SQLite 单表：最大单笔金额",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT MAX(amount) AS value FROM orders",
            expected=sqlite_max_amount(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "max"),
        ),
        EvalTask(
            task_id="sqlite_orders_over_15",
            family="sqlite_filter",
            description="SQLite 单表：金额大于 15 的订单数",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT COUNT(*) AS value FROM orders WHERE amount > 15.0",
            expected=sqlite_orders_over_15(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "filter"),
        ),
        # --- SQLite：连接 ----------------------------------------------
        EvalTask(
            task_id="sqlite_revenue_by_region",
            family="sqlite_join",
            description="SQLite 连接 + 分组：按区域汇总金额",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT c.region AS region, SUM(o.amount) AS revenue "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "GROUP BY c.region ORDER BY c.region"
            ),
            expected=sqlite_revenue_by_region(),
            shape="rows_grouped",
            key_column="region",
            value_column="revenue",
            tags=("sqlite", "join", "group_by"),
        ),
        EvalTask(
            task_id="sqlite_revenue_by_customer",
            family="sqlite_join",
            description="SQLite 连接 + 分组：按客户汇总金额",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT c.name AS name, SUM(o.amount) AS revenue "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "GROUP BY c.name ORDER BY c.name"
            ),
            expected=sqlite_revenue_by_customer(),
            shape="rows_grouped",
            key_column="name",
            value_column="revenue",
            tags=("sqlite", "join", "group_by"),
        ),
        EvalTask(
            task_id="sqlite_east_total",
            family="sqlite_join",
            description="SQLite 连接 + 过滤：East 区域金额",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT SUM(o.amount) AS value "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "WHERE c.region = 'East'"
            ),
            expected=sqlite_east_total(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "join", "filter"),
        ),
        # --- CSV：派生指标 ---------------------------------------------
        EvalTask(
            task_id="csv_revenue_std",
            family="csv_statistics",
            description="单笔营收的总体标准差（保留 4 位）",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'value': round(float(frame['revenue'].std(ddof=0)), 4)}]\n"
            ),
            expected=csv_revenue_std(),
            shape="scalar",
            value_column="value",
            tags=("statistics", "std"),
        ),
        EvalTask(
            task_id="csv_east_share",
            family="csv_ratio",
            description="East 区域营收占比（保留 6 位）",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "total = frame['revenue'].sum()\n"
                "east = frame.loc[frame['region'] == 'East', 'revenue'].sum()\n"
                "result = [{'value': round(float(east / total), 6)}]\n"
            ),
            expected=csv_east_share(),
            shape="scalar",
            value_column="value",
            tags=("ratio", "filter"),
        ),
        EvalTask(
            task_id="csv_units_sum",
            family="csv_aggregate",
            description="全部订单的单量之和",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int(df['units'].sum())}]\n",
            expected=csv_units_sum(),
            shape="scalar",
            value_column="value",
            tags=("aggregate", "sum"),
        ),
        EvalTask(
            task_id="csv_service_count",
            family="csv_filter",
            description="service 品类的订单数",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int((df['category'] == 'service').sum())}]\n",
            expected=csv_service_count(),
            shape="scalar",
            value_column="value",
            tags=("filter", "count"),
        ),
        EvalTask(
            task_id="csv_east_or_west",
            family="csv_filter",
            description="East 或 West 区域的订单数（集合过滤）",
            data_source="csv",
            target_tables=["orders"],
            code="result = [{'value': int(df['region'].isin(['East', 'West']).sum())}]\n",
            expected=csv_both_east_west(),
            shape="scalar",
            value_column="value",
            tags=("filter", "set_membership"),
        ),
        EvalTask(
            task_id="csv_revenue_spread",
            family="csv_statistics",
            description="单笔最高与最低营收之差",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'spread': float((frame['revenue'].max() "
                "- frame['revenue'].min()).round(2))}]\n"
            ),
            expected=csv_revenue_spread(),
            shape="scalar",
            value_column="spread",
            tags=("statistics", "range"),
        ),
        EvalTask(
            task_id="csv_revenue_by_region_category",
            family="csv_multi_group",
            description="按区域×品类双键分组汇总营收",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "grouped = frame.groupby(['region', 'category'], as_index=False)['revenue'].sum().round(2)\n"
                "grouped['key'] = grouped['region'] + '|' + grouped['category']\n"
                "result = grouped[['key', 'revenue']]\n"
            ),
            expected=csv_revenue_by_region_category(),
            shape="rows_grouped",
            key_column="key",
            value_column="revenue",
            tags=("group_by", "multi_key"),
        ),
        EvalTask(
            task_id="csv_top2_categories",
            family="csv_ranking",
            description="按营收降序取前 2 个品类",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "grouped = frame.groupby('category', as_index=False)['revenue'].sum().round(2)\n"
                "result = grouped.sort_values(['revenue', 'category'], "
                "ascending=[False, True]).head(2)\n"
            ),
            expected=csv_top2_categories(),
            shape="rows_grouped",
            key_column="category",
            value_column="revenue",
            tags=("ranking", "top_n"),
        ),
        EvalTask(
            task_id="csv_high_value_orders",
            family="csv_derived_filter",
            description="派生列后过滤：营收超过 40 的订单数",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "result = [{'value': int((frame['revenue'] > 40).sum())}]\n"
            ),
            expected=csv_high_value_order_count(),
            shape="scalar",
            value_column="value",
            tags=("derive_column", "filter"),
        ),
        EvalTask(
            task_id="csv_east_avg_units",
            family="csv_filter",
            description="East 区域的平均单量（保留 4 位）",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df[df['region'] == 'East']\n"
                "result = [{'value': round(float(frame['units'].mean()), 4)}]\n"
            ),
            expected=csv_east_avg_units(),
            shape="scalar",
            value_column="value",
            tags=("filter", "mean"),
        ),
        EvalTask(
            task_id="csv_lowest_region",
            family="csv_ranking",
            description="营收最低的区域名（并列时取字典序最小）",
            data_source="csv",
            target_tables=["orders"],
            code=(
                "frame = df.copy()\n"
                "frame['revenue'] = (frame['units'] * frame['unit_price']).round(2)\n"
                "grouped = frame.groupby('region', as_index=False)['revenue'].sum().round(2)\n"
                "grouped = grouped.sort_values(['revenue', 'region'], ascending=[True, True])\n"
                "result = [{'region': str(grouped.iloc[0]['region'])}]\n"
            ),
            expected=csv_lowest_region(),
            shape="scalar",
            value_column="region",
            tags=("ranking", "argmin"),
        ),
        # --- SQLite：更多算子 -------------------------------------------
        EvalTask(
            task_id="sqlite_avg_amount",
            family="sqlite_aggregate",
            description="SQLite 单表：平均金额（保留 4 位）",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT ROUND(AVG(amount), 4) AS value FROM orders",
            expected=sqlite_avg_amount(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "aggregate", "mean"),
        ),
        EvalTask(
            task_id="sqlite_distinct_customers",
            family="sqlite_distinct",
            description="SQLite 单表：有订单的客户数",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT COUNT(DISTINCT customer_id) AS value FROM orders",
            expected=sqlite_distinct_customers(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "distinct"),
        ),
        EvalTask(
            task_id="sqlite_min_amount",
            family="sqlite_extreme",
            description="SQLite 单表：最小单笔金额",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code="SELECT MIN(amount) AS value FROM orders",
            expected=sqlite_min_amount(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "min"),
        ),
        EvalTask(
            task_id="sqlite_orders_in_range",
            family="sqlite_filter",
            description="SQLite 单表：金额在 [10, 50] 区间的订单数",
            data_source="sqlite",
            target_tables=["sales.orders"],
            code=(
                "SELECT COUNT(*) AS value FROM orders "
                "WHERE amount >= 10.0 AND amount <= 50.0"
            ),
            expected=sqlite_orders_between(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "filter", "range"),
        ),
        EvalTask(
            task_id="sqlite_high_value_customers",
            family="sqlite_join",
            description="SQLite 连接 + 过滤 + 分组：大额订单按客户汇总",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT c.name AS name, SUM(o.amount) AS revenue "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "WHERE o.amount > 15.0 GROUP BY c.name ORDER BY c.name"
            ),
            expected=sqlite_high_value_customers(),
            shape="rows_grouped",
            key_column="name",
            value_column="revenue",
            tags=("sqlite", "join", "filter", "group_by"),
        ),
        EvalTask(
            task_id="sqlite_order_count_by_region",
            family="sqlite_join",
            description="SQLite 连接 + 计数：各区域订单数",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT c.region AS region, COUNT(*) AS count "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "GROUP BY c.region ORDER BY c.region"
            ),
            expected=sqlite_order_count_by_region(),
            shape="rows_grouped",
            key_column="region",
            value_column="count",
            tags=("sqlite", "join", "count"),
        ),
        EvalTask(
            task_id="sqlite_west_or_north_total",
            family="sqlite_join",
            description="SQLite 连接 + IN 过滤：West/North 合计",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT SUM(o.amount) AS value "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "WHERE c.region IN ('West', 'North')"
            ),
            expected=sqlite_west_or_north_total(),
            shape="scalar",
            value_column="value",
            tags=("sqlite", "join", "set_membership"),
        ),
        EvalTask(
            task_id="sqlite_max_order_customer",
            family="sqlite_join",
            description="SQLite 连接 + 极值：最大单笔对应的客户",
            data_source="sqlite",
            target_tables=["sales.orders", "sales.customers"],
            code=(
                "SELECT c.name AS name, o.amount AS amount "
                "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
                "ORDER BY o.amount DESC LIMIT 1"
            ),
            # 返回的是"客户 + 金额"这一行复合结果，必须整行比对：
            # 只取 name 会丢掉 amount，期望里的两个键就对不上了。
            expected=sqlite_max_order_customer(),
            shape="row",
            tags=("sqlite", "join", "extreme"),
        ),
    ]

    return tasks


def task_by_id(task_id: str) -> EvalTask:
    for task in build_tasks():
        if task.task_id == task_id:
            return task
    raise KeyError(f"未知任务：{task_id}")


def families() -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in build_tasks():
        counts[task.family] = counts.get(task.family, 0) + 1
    return counts
