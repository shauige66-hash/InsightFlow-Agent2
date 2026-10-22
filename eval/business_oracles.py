"""Independent CSV/Decimal answer oracles for the two paid regression cases.

No model call is made here. Missing machine-readable answers fail closed.
"""
from __future__ import annotations

import calendar
import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
import json
import math
from pathlib import Path
import re

NON_PRODUCT_CODES = {'M', 'POST', 'DOT', 'C2', 'BANK CHARGES', 'PADS', 'CRUK', 'D'}
ANSWER_CONTRACT = '\n报告末尾必须追加一个JSON代码块，格式为 {"business_answers": <答案对象>}，数值字段必须为JSON数字。'


def expected_answers(name: str, data_sources: list[str]) -> dict:
    paths = {Path(path).stem: Path(path) for path in data_sources}
    if name == 'toy-regions':
        with paths['customers'].open(encoding='utf-8-sig') as handle:
            customers = {row['customer_id']: row['region'] for row in csv.DictReader(handle)}
        amounts = defaultdict(Decimal)
        with paths['orders'].open(encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                amounts[row['customer_id']] += Decimal(row['amount'])
        regions = defaultdict(Decimal)
        for customer, amount in amounts.items():
            if amount > Decimal(400):
                regions[customers[customer]] += amount
        return {'high_value_sales_by_region': {key: float(value) for key, value in sorted(regions.items())}}
    if name != 'retail-gmv':
        raise ValueError(f'No business oracle for {name}')
    with paths['orders_eu_2011'].open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    last = max(datetime.fromisoformat(row['invoice_ts']) for row in rows)
    last_month = last.strftime('%Y-%m')
    monthly = defaultdict(Decimal)
    returns = defaultdict(Decimal)
    for row in rows:
        if row['stock_code'] in NON_PRODUCT_CODES or Decimal(row['unit_price']) <= 0:
            continue
        amount = Decimal(row['quantity']) * Decimal(row['unit_price'])
        if amount > 0:
            monthly[row['invoice_ts'][:7]] += amount
        elif amount < 0:
            returns[row['stock_code']] -= amount
    complete = sorted(month for month in monthly if month != last_month or last.day == calendar.monthrange(last.year, last.month)[1])
    months = complete[-2:]
    if len(months) != 2 or not returns or monthly[months[0]] == 0:
        raise ValueError('Insufficient complete months or return records')
    values = [monthly[month] for month in months]
    return {'gmv_months': months, 'gmv_values': [float(value) for value in values], 'gmv_growth_pct': float((values[1] / values[0] - 1) * 100), 'top_return_sku': max(returns, key=returns.get)}


def answer_fields(name: str) -> str:
    if name == 'retail-gmv':
        return ANSWER_CONTRACT + '答案字段：gmv_months（两个完整月YYYY-MM数组）、gmv_values（对应正向GMV数组）、gmv_growth_pct（环比百分数）、top_return_sku（退货金额最高的商品编码字符串）。'
    return ANSWER_CONTRACT + '答案字段：high_value_sales_by_region（地区到高价值客户销售总额的对象，仅保留有高价值客户的地区）。'


def grade_business_answers(name: str, report: str, data_sources: list[str]) -> dict:
    expected = expected_answers(name, data_sources)
    candidates = []
    for block in re.findall(r'```(?:json)?\s*\n(.*?)```', report or '', re.DOTALL | re.IGNORECASE):
        try:
            payload = json.loads(block)
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get('business_answers'), dict):
            candidates.append(payload['business_answers'])
    if len(candidates) != 1:
        return {'passed': False, 'reason': 'Require exactly one business_answers JSON block', 'expected': expected, 'actual': candidates}
    actual = candidates[0]
    def matches(a, e, key=''):
        if isinstance(e, dict):
            return isinstance(a, dict) and set(a) == set(e) and all(matches(a[k], v, k) for k, v in e.items())
        if isinstance(e, list):
            return isinstance(a, list) and len(a) == len(e) and all(matches(x, y, key) for x, y in zip(a, e))
        if isinstance(e, (int, float)):
            return type(a) in (int, float) and math.isfinite(a) and abs(a-e) <= (0.05 if key == 'gmv_growth_pct' else 0.01)
        return type(a) is type(e) and a == e
    passed = matches(actual, expected)
    return {'passed': passed, 'reason': 'Independent CSV/Decimal answers match' if passed else 'Business answer mismatch', 'expected': expected, 'actual': actual}
