# 案例数据：跨境小电商（德国 + 法国，2011）

| 文件 | 行数 | 说明 |
|---|---|---|
| `orders_eu_2011.csv` | 17,049 | 交易明细，一行一个商品行 |
| `products.csv` | 1,990 | 商品维表（由明细去重派生，供多表关联） |

## 来源

UCI Machine Learning Repository，**Online Retail II**（一家英国在线零售商的真实交易记录）。
本目录只取其中 `Year 2010-2011` sheet 里 **Germany 与 France** 两个市场、**2011-01-01 至 2011-12-31** 的行。

- 数据集主页：<https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- 引用：Chen, D. (2012). *Online Retail II*. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CG6D>

原始文件（45 MB xlsx）不进仓库；用 `scripts/prepare_retail_case.py` 重新生成。

## 列

| 列 | 含义 |
|---|---|
| `invoice_id` | 发票号；以 `C` 开头表示退货 |
| `stock_code` | 商品编码。`M`、`POST`、`DOT`、`BANK CHARGES` 等**不是商品**（手工调整、运费、银行费用） |
| `description` | 商品描述 |
| `quantity` | 数量，负数表示退货 |
| `invoice_ts` | 开票时间（原样保留，未做时区处理） |
| `unit_price` | 单价（原币种，英镑） |
| `customer_id` | 客户号；有 66 行缺失 |
| `country` | 市场（Germany / France） |
| `is_return` | 是否退货行（`quantity < 0`） |

## 刻意保留的"脏"

数据没有清洗，因为这些正是分析要处理的东西：

- **12 月只有 9 天**（截止 2011-12-09），月份环比不能以 12 月为终点；
- 678 行非商品编码（含 `M`、`POST`），会把退货率算歪；
- 66 行缺客户号；2 行零/负单价；64 组同一发票内重复的 `invoice_id + stock_code`。

处理口径写在 [`docs/case-study.md`](../case-study.md)，并给出每个问题的标准答案。
