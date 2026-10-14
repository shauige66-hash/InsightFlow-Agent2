"""沙箱包。

宿主侧模块（`protocol`、`bootstrap`、`executor`）与容器侧执行入口（`run`）
共用同一目录：容器把它只读挂载为 `/sandbox`，因此两边不存在第二份实现。
"""
