# 定义不同数据源在系统中的统一数据结构。
from dataclasses import dataclass

import pandas as pd


@dataclass
class TableResource:
    name: str
    source_type: str
    source_path: str
    dataframe: pd.DataFrame | None = None
    database_table: str | None = None

    # 来源内容的 sha256，来自 TableSnapshot。用于让下游缓存（工具实例缓存、
    # 沙箱输入）与  的内容指纹保持一致：同一路径换了文件内容时，
    # 缓存键必须随之变化，不能复用旧 DataFrame。
    content_hash: str | None = None