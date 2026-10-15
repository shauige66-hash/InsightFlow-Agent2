"""MCP 桥接层：把一次 Run 的授权资源经 stdio 暴露给工作流。

模块职责：

- `contract.py`  两侧共享的工具名、参数名与结果形状
- `server.py`    独立子进程中的 MCP Server（三个只读工具）
- `client.py`    宿主侧的 stdio Client，负责握手、发现、调用与进程回收
- `registry.py`  从 ResourceResolver 的快照构造 Run 级授权配置
"""
