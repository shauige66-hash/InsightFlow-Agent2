"""Small deterministic business tasks, with disjoint data per evaluation split.

These are same-domain, synthetic task variants, not evidence of cross-domain
generalisation.  They contain inputs only; answers are computed by the host oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

SPLITS = ("development", "validation", "holdout")
RULES = """# Business rules
High-value customers have a cumulative order amount strictly greater than 400.
The threshold applies to each customer's total, not individual orders or averages.
Join orders to customers by customer_id. Aggregate qualifying customers' order
amounts by their region. Exclude regions with no qualifying customers.
Amounts are denominated in USD. Do not replace the threshold with a percentile.
"""


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class RealTask:
    task_id: str
    split: str
    customers_csv: str
    orders_csv: str
    rules: str = RULES
    question: str = "分析各地区高价值客户的销售贡献，给出各地区销售总额及结论。"

    def files(self) -> dict[str, str]:
        return {"customers.csv": self.customers_csv, "orders.csv": self.orders_csv,
                "business_rules.md": self.rules}

    @property
    def fingerprint(self) -> str:
        return _digest({"task_id": self.task_id, "question": self.question, "files": self.files()})

    def materialise(self, directory: Path) -> tuple[list[str], list[str]]:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in self.files().items():
            (directory / name).write_text(content, encoding="utf-8", newline="")
        return ([str(directory / "customers.csv"), str(directory / "orders.csv")],
                [str(directory / "business_rules.md")])

    def verify_files(self, directory: Path) -> None:
        for name, content in self.files().items():
            if (directory / name).read_bytes() != content.encode("utf-8"):
                raise ValueError(f"Evaluation input changed: {name}")


def build_real_tasks(split: str, *, holdout_seed: int = 0) -> tuple[RealTask, ...]:
    if split not in SPLITS:
        raise ValueError(f"Unknown evaluation split: {split}")
    if type(holdout_seed) is not int or holdout_seed < 0:
        raise ValueError("holdout_seed must be a nonnegative integer")
    split_number = SPLITS.index(split) + 1
    salt = int(_digest({"seed": holdout_seed})[:8], 16) if split == "holdout" else 0
    tasks = []
    for variant in range(2):
        tag = f"{split}-{variant}-{salt:x}"
        ids = [f"{tag}-c{i}" for i in range(1, 7)]
        regions = ("East", "West", "North", "East", "South", "West")
        customers = "customer_id,region\n" + "".join(f"{i},{r}\n" for i, r in zip(ids, regions))
        # Boundary (400), split orders, negative adjustments, and cents exercise
        # the fixed business rule without importing expected values into a prompt.
        offset = split_number * 17 + variant * 23 + salt % 97
        amounts = ((f"{250 + offset}.05", "200.05"), ("400.00",),
                   (f"{390 + offset}.00", "70.25"), ("300.00", "-10.00"),
                   (f"{500 + offset}.10", "-50.05"), ("0.00", "399.99"))
        orders = "order_id,customer_id,amount\n"
        serial = 0
        for customer, values in zip(ids, amounts):
            for amount in values:
                serial += 1
                orders += f"{tag}-o{serial},{customer},{amount}\n"
        tasks.append(RealTask(f"{split}-regions-{variant}", split, customers, orders))
    return tuple(tasks)


def dataset_fingerprint(tasks: tuple[RealTask, ...]) -> str:
    return _digest([task.fingerprint for task in tasks])
