# 定义 Business Knowledge RAG 的默认检索参数。

DEFAULT_TOP_K = 3
VECTOR_STORE_CACHE_SIZE = 8

#: 绝对阈值：与问句的余弦相似度（越大越相关）。
MIN_SIMILARITY_SCORE = 0.35

#: 相对阈值：低于最高分一定比例的结果视为噪音。
#:
#: 只有绝对阈值时，一个很短的知识库（比如两条各一段的规则）里所有片段都
#: 可能过线，于是"按任务检索"退化成"每个任务都拿到同一批知识"。相对阈值
#: 让第一名明显领先时只保留它，同时不影响真正并列相关的情况。
MIN_RELATIVE_SCORE = 0.75
