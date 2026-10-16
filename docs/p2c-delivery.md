# P2-C 交付说明：MCP Server / Client 集成

本文记录 P2-C 的 MCP 架构、真实 stdio 调用链、三个工具的 Schema、Run 级数据授权、
Tool Registry 与 LangGraph 的接入方式、进程生命周期与幂等边界，以及仍未解决的部分。

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 | Checkpoint、HITL、LLM Ledger、只读 SQL | 已完成 |
| P2-A | Docker 执行沙箱、staging 产物发布、错误分类短路 | 已完成 |
| P2-B | 逻辑尝试身份、SQL 纵深、大结果恢复 | 已完成 |
| **P2-C** | **MCP Server / Client 集成** | **本文** |

> **MCP Server 不是安全沙箱。** 它是宿主上的普通 Python 进程，权限与启动它的用户
> 相同。它的约束来自"只有一份授权"与"复用同一套 SQL 策略"，不来自进程隔离。
> Python / Plot 的执行沙箱是 P2-A 的 Docker 容器，与本阶段无关。

---

## 一、架构

```
LangGraph 节点
  └─ tools/tool_registry.invoke_tool(tool, tables, args, snapshots)
       ├─ "python" → DockerSandboxExecutor（P2-A，未改动）
       ├─ "plot"   → DockerSandboxExecutor（P2-A，未改动）
       └─ "sql"    → 按 RunRuntime.mcp.mode 分派
                      ├─ direct → tools/sql_tool.py        ─┐
                      └─ mcp    → tools/mcp_sql_tool.py     │
                                   └─ mcp_bridge/client.py   │
                                        │ stdio JSON-RPC    │
                                        ▼                   │
                             （独立子进程）mcp_bridge/server.py
                                        │                   │
                                        └───────────────────┘
                                                 ▼
                                    utils/sql_policy.execute_readonly_sql
                                    （direct 与 MCP 共用同一份策略）
```

两条后端路径**共用同一份 SQL 策略**，这是"引入 MCP 不放宽任何既有安全条件"的
实现方式：不是靠两边各写一遍再人工对齐，而是只有一份实现。

---

## 二、实际 stdio 调用链

工具调用时按序发生（`mcp_bridge/client.py`）：

| 步骤 | 动作 | 实现位置 |
|---|---|---|
| 1 | 起独立事件循环线程 | `_run_loop` / `_await_loop_ready` |
| 2 | `Subprocess` 启动 Server（结构化参数，不经 shell） | `StdioServerParameters(command=..., args=["-m","mcp_bridge.server","--config",...])` |
| 3 | `initialize` 握手 | `session.initialize()` |
| 4 | `list_tools` 动态发现 | `session.list_tools()` |
| 5 | 读参数 Schema 并记录 | `ToolDigest.input_schema` |
| 6 | 与本地允许清单求交集 | `allowed_tools()` |
| 7 | `call_tool` 调用 | `session.call_tool(name, arguments)` |
| 8 | 解析正常/错误结果 | `_call`（`isError` → `execution_error`） |
| 9 | 关闭 Session、终止子进程、兜底回收 | `close()` + `reap_process` |

实测握手结果（本机）：

```
initialize -> insightflow-mcp | protocolVersion 2025-11-25
list_tools -> ['discover_datasets', 'get_schema', 'run_readonly_sql']
```

**同步外观 + 专用事件循环线程**：工作流在宿主线程里同步执行，而 MCP SDK 是异步的。
不用"每次调用 `asyncio.run`"的做法——那样会话会在调用之间被拆掉，
握手与工具发现的结果无法复用，也无法保证进程回收。

**动态发现不等于自动授权**：`list_tools` 只说明服务端**声明**了什么。
实际可调用集合是 `发现结果 ∩ ALLOWED_MCP_TOOLS`，调用前还会再校验
工具名与参数 Schema。一个被替换过的 Server 多声明工具并不会因此获得执行权限。

---

## 三、三个工具的参数与结果 Schema

### `discover_datasets`

入参：无。

出参：

```json
{
  "run_id": "run-a",
  "count": 2,
  "datasets": [
    {
      "dataset_id": "run-a:sales.orders",
      "name": "sales.orders",
      "source_type": "sqlite",
      "row_count": null,
      "columns": [],
      "content_hash": "hash-1",
      "database_table": "orders",
      "able_to": ["schema", "readonly_sql"]
    }
  ]
}
```

**不含宿主绝对路径**，有测试断言返回内容里不出现文件路径。

### `get_schema`

入参：`dataset_id: str`

出参（CSV / Excel）：

```json
{
  "dataset_id": "run-a:customers",
  "source_type": "csv",
  "columns": [
    {"name": "amount", "inferred_type": "number", "sample_values": [1.5, 2.5]}
  ],
  "sampled_rows": 50,
  "row_count": null
}
```

出参（SQLite）：

```json
{
  "dataset_id": "run-a:sales.orders",
  "database_table": "orders",
  "columns": [
    {"name": "order_id", "declared_type": "INTEGER",
     "nullable": true, "primary_key": true}
  ]
}
```

类型是**粗推断**（integer / number / boolean / text / null），只用于让模型知道
怎么写查询，不声称是严格类型系统。CSV/Excel 读的是**表自身声明的**路径与
sheet 名，不接受调用方给路径或 sheet 参数。

### `run_readonly_sql`

入参：

| 参数 | 类型 | 说明 |
|---|---|---|
| `query` | `str` | 单条 SELECT / WITH |
| `dataset_ids` | `list[str] \| null` | 参与本次查询的授权数据集 |

出参：

```json
{"columns": ["total"], "rows": [{"total": 35.0}],
 "row_count": 1, "bytes": 15, "truncated": false}
```

与 direct 后端返回**完全相同**的结构（有测试逐字段比对）。

---

## 四、Run 级数据集授权

授权信息由宿主从 P1 的权威快照生成，写到
`<output_root>.mcp_private/<run_id>/authorizations.json`：

```json
{
  "run_id": "run-a",
  "workspace_root": "...",
  "datasets": [
    {"dataset_id": "run-a:sales.orders", "name": "sales.orders",
     "source_type": "sqlite", "source_path": "...",
     "locator": "orders", "content_hash": "hash-1",
     "database_table": "orders"}
  ]
}
```

几个关键点：

- **dataset_id 由服务端派生**：`f"{run_id}:{name}"`。只由稳定输入构成，
  因此同一 Run 内每次计算一致、跨进程恢复一致、不同 Run 之间不撞车。
- **模型只能引用 dataset_id**，不能传路径。工具签名里根本没有路径参数。
- **配置是私有位置**：在 `<output_root>.mcp_private/<run_id>/`，
  与 `output_root` **平级**，因此既不在 `output_root` 之内、
  也不在任何 Run 容器的挂载范围内（见 §9.2：这不只是"读不到"的问题，
  子目录名还会与 `run_id` 撞名）。
- **拒绝把私有路径配置到 `output_root` 之内**：`mcp_private_root()` 会把
  解析后的真实路径（展开符号链接与 `..`、统一大小写）与 `output_root` 比较，
  落在其中即抛 `PrivatePathConflict`。仅靠默认值不够——显式配置同样要守规矩。
- **Server 侧再校验一次路径**：`workspace_path()` 要求解析后的路径位于
  `workspace_root` 之内，越界即拒绝。配置即使被改动也读不到别处文件。
- **写入前拒绝符号链接**，且用临时文件 + `os.replace` 原子替换。
- **每次启动重写配置**：恢复后的新进程里内容指纹可能已变化，
  沿用旧配置等于绕过 P1 的"数据源变化必须失败"。
- **未知 id 与越权 id 走同一条拒绝路径**，错误信息不回显授权集合。
- **不传凭证**：子进程环境是逐项白名单（见第七节）。

---

## 五、Tool Registry 与 LangGraph 集成

`allow` 后端选择属于**可信运行期配置**，放在 `RunRuntime.mcp`（`McpSettings`）：

```python
McpSettings(mode="direct" | "mcp", workspace_root=..., startup_timeout=..., call_timeout=...)
```

`mode` 没有"自动挑一个"的默认值——那会在通道不通时把失败悄悄变成另一种执行路径。

`tool_registry.resolve_backend()` 决定分派：

- `python` / `plot` → `sandbox`（P2-A，未改动）；
- `sql` → `runtime.mcp.mode`；
- 取不到运行期上下文 → `direct`（那是"没有配置"，不是"配置失败"）。

后端进入工具缓存键：同一批数据在 direct / mcp 之间切换时不会拿到旧工具。

`tool_execution_node` 把 `state["table_snapshot"]` 作为 `snapshots` 传给
`invoke_tool`，因此 MCP 的授权来自 P1 的权威快照，而不是"本次点名的表"临时拼凑。

**完整链路已实测跑通**（`test_mcp_integration.py`）：

```
Planner → Router → Executor → MCP Client → MCP Server → SQL → Critic → Report
```

并且用计数子类证明 mcp 模式**确实启动了一次 Server 子进程、确实发生了一次
`run_readonly_sql` 调用**——而不是"结果一样所以大概走了 MCP"。

---

## 六、SQL 防护与错误分类

MCP 路径复用 `utils/sql_policy`，因此下列约束一条都没少（有测试逐项覆盖）：

| 层 | 机制 |
|---|---|
| 1 | `ensure_read_only` 词法校验（单条 SELECT/WITH、去注释与字面量、fail-closed） |
| 2 | SQLite Authorizer（4 个只读操作码白名单 + 授权表白名单 + 元数据表拒绝） |
| 3 | `mode=ro` + `uri=True` 连接 |
| 4 | `set_progress_handler` + 单调时钟查询预算 |
| 5 | 有界 `fetchmany`（行数上限 / 字节上限 / `truncated` 标记） |

错误分类映射到既有契约：

| 情况 | `error_kind` | 是否进入 Repair |
|---|---|---|
| MCP 启动失败、Server 崩溃、会话失效 | `unavailable` | **否**（环境问题） |
| MCP 操作超时 | `timeout` | **否** |
| SQL 语法错误、越权表、非只读语句 | `execution_error` | 是（有界修复） |

实测：MCP 通道不可用与启动超时两种情况下 `repair` 调用次数均为 **0**；
SQL 写错（表不存在）时 `repair` 被正常调用。

---

## 七、进程生命周期、超时与清理

| 项 | 策略 |
|---|---|
| 启动 + 初始化 | 共用一个预算（默认 60 s），任一步卡住即不可用 |
| `call_tool` | 独立预算（默认 60 s） |
| 超时后的会话 | **标记失效**，后续调用直接失败并要求重建客户端 |
| 正常关闭 | 退出 async 上下文 → 停止事件循环线程 → 兜底 `reap_process` |
| 子进程回收 | 实测关闭后进程确实结束；另有 `taskkill`/`SIGKILL` 兜底 |
| shell | 不使用 `shell=True`，参数为结构化列表 |
| 启动命令与环境 | 由客户端常量决定，模型无法影响 |

**超时即失效**这条是有实测依据的设计，不是保守猜测：
MCP 1.30.0 在 Windows 上超时会让 anyio 的 task group 以 `BaseExceptionGroup` 退出，
`stdio_client.__aexit__` 因此抛出。会话在超时后**不可继续使用**，
把它伪装成"还能用"会掩盖真实协议状态。因此 `client` 用 `flatten_exception()`
摊开嵌套异常取出根因（否则只能看到 "unhandled errors in a TaskGroup"），
再据此区分 timeout 与 unavailable。

**子进程环境是逐项白名单**：只传操作系统管道变量（`PATH` / `SYSTEMROOT` /
`TEMP` / `APPDATA` / `USERPROFILE` …）与编码变量，**不含** API Key、
ModelGate 凭证或完整 `os.environ`。

> 实现记录：最初把环境压到只剩 `PATH` + 编码，结果 CSV 的 `get_schema`
> 稳定超时——排查后确认真正原因是 §9.1 的 pandas 延迟导入，
> 但白名单里缺少 `TEMP` / `APPDATA` 这类变量仍然是不该有的风险，因此改成
> 逐项列举。白名单而非"排除敏感项"，是为了不因遗漏而泄漏。

---

## 八、跨进程恢复与幂等边界

### 不进 Checkpoint

`McpStdioClient`、`ClientSession`、事件循环线程与子进程句柄都**不进 state**，
因此不会进 Checkpoint。它们放在 `RunRuntime`，由 `use_runtime` 按上下文注入；
`McpSettings` 是纯数据（字符串与数字），恢复时重新构造。

**恢复时按调用重建会话**：每次 `call_mcp_sql` 创建一个新客户端，用完立刻关闭。
这样"恢复"不依赖任何上一个进程的存活状态。

### 已验证的恢复语义

| 场景 | 结果 |
|---|---|
| 重复 Resume（终态） | `execution_results` 条数不变，`run_status` 仍为 completed |
| **真实跨进程恢复** | 见下方"三个进程的实测" |
| 已完成工具结果 | 由 P2-B 的幂等登记跳过，不重复调用 |
| MCP 通道不可用 | `unavailable`，短路失败，不触发 Repair |
| Server 崩溃 / 启动失败 | `unavailable`，子进程被回收 |
| 两个 Run | 各自独立的 Server 进程、授权配置与输出目录 |

### 三个进程的实测（真实跨进程 MCP 恢复）

`tests/test_mcp_cross_process.py` 用三个**独立 Python 进程**验证：

| 进程 | 动作 | 实测结果 |
|---|---|---|
| 1 | mcp 模式启动，真实 MCP Server 完成一次 SQL 调用后强杀 | 退出码 **17**；stderr 有 `CallToolRequest`；Checkpoint 停在 `next_nodes` 指向待执行节点 |
| 2 | 全新进程，加载已有 Checkpoint 并在 mcp 模式下恢复 | `run_status=completed`、`analysis_status=completed`、`failed_tasks=0`；stderr **再次出现 `CallToolRequest`** |
| 3 | inspect 最终状态 | `run_status=completed`、`next_nodes == []` |

判据说明：

- **确实走了 MCP 而不是静默切 direct**：恢复进程的 stderr 上出现 MCP 协议活动
  （`ListToolsRequest` / `CallToolRequest`）。direct 后端不会产生这些输出——
  对照用例 `test_direct_mode_recovery_does_not_start_a_server` 断言 direct 下没有。
- **授权与指纹不变**：恢复前后 `dataset_id` 列表与 `content_hash` 完全一致，
  且都带本次 Run 的前缀。
- **无遗留子进程**：每个进程结束后查询系统进程表，`mcp_bridge.server` 数量为 0。
- **已完成的结果不重复**：终态 Run 再次 resume 会短路，`current_task_index` 不变。

崩溃点位于 `tools/mcp_sql_tool._mcp_crash_point("after_mcp_call")`：
MCP 调用**已经完成**、但节点 state 尚未提交。这正是"外部工具副作用与
Checkpoint 不在同一个事务里"的关键瞬间。

### exactly-once 边界（如实记录）

**不能声称 MCP 工具副作用恰好执行一次。** 原因是两层叠加：

1. Checkpoint 只在节点结束时提交，"调用已发出、节点未提交"的崩溃在 state 里
   没有痕迹；P1 的账本能记录"发出过"，但 `started` 无法区分
   "上游已处理"与"上游没收到"。
2. MCP 会话本身不持久：恢复后是新会话、新子进程。
   即使有账本记录，也无法向一个已经消失的会话查询上一次到底执行没有。

上面的三进程用例正好落在这个区间：进程 1 的 MCP 调用**确实执行了**，
但结果没有进 Checkpoint；恢复时它被重新执行了一次。因为 `run_readonly_sql`
是只读查询，重复执行不改变数据——但这属于"该工具恰好幂等"，
不是通用保证，不能被推广到将来的写类工具。

---

## 九、实现中发现并修复的缺陷

### 9.1 pandas 延迟导入导致 `get_schema` 永久卡住

`_read_tabular` 里 `import pandas as pd` 是**函数内**导入。
FastMCP 把同步工具函数放到工作线程里执行，于是首次导入 pandas 发生在
stdio 事件循环已运行之后——实测 CSV 路径**永久挂起**，
`call_tool` 一路超时（60 s、180 s 都试过）。

SQLite 路径走 `sqlite3` 不经过 pandas，所以现象表现为"只有 CSV 与 Excel 的
`get_schema` 有问题"，很容易被误判成"文件读取慢"或"环境变量缺失"。

修复：把 `import pandas as pd` 提到模块顶层，导入发生在 stdio 循环启动之前。
修复后 `get_schema` 从超时变为 0.01 s。

这个缺陷值得记下来，因为它只在"真实子进程 + 工作线程 + 首次导入"三者同时成立时
才出现——单测或同进程调用都不会暴露。

### 9.2 私有授权配置曾被放在 run 目录内（可被沙箱读取）

**这是一个真实的安全问题，已修复。**

`write_authorizations` 原先把配置写在
`<output_root>/<run_id>/mcp/authorizations.json`，而 P2-A 的
`DockerSandboxExecutor` 把整个 run 目录只读挂载为 `/history`。
容器里因此存在 `/history/mcp/authorizations.json`。

实测（真实容器 + 被注入的 `pandas`）证明模型代码能读到它，内容包含：

| 字段 | 是否敏感 |
|---|---|
| `workspace_root` | **是**：宿主绝对路径 |
| `datasets[].source_path` | **是**：每个数据源的宿主绝对路径 |
| `datasets[].dataset_id` / `name` | 是：授权集合与资源命名 |
| `datasets[].content_hash` | 是：可用于比对文件是否变化 |
| `datasets[].database_table` / `locator` | 是：物理表名 |

**只读挂载只保证"改不了"，不保证"读不到"。** 这条教训值得记下来。

修复（两步，缺一不可）：

**第一步：把私有配置移到 `output_root` 之外。**
路径改为 `<output_root>.mcp_private/<run_id>/authorizations.json`——
与 `output_root` **平级**。这样从文件系统结构上就不存在"run 目录 == 私有根目录"
的可能。

**第二步：显式配置也要守规矩。**
`mcp_private_root()` 会把解析后的真实路径与 `output_root` 比较，
落在其中即抛 `PrivatePathConflict`。比较用 `os.path.normcase(resolve())`，
因此 Windows 的大小写差异（`OUTPUTS` vs `outputs`）与
`..` / 符号链接拼出的等价路径都会被识别。

### 9.3 私有目录与 Run ID 的命名空间冲突（跨 Run 泄漏）

**这是第一步修复要解决的真实问题，单独记下来。**

`api_runs.RUN_ID_PATTERN = ^[A-Za-z0-9_-]{4,64}$` **合法接受** `mcp_private`
这个 run_id。而私有根目录原先叫 `<output_root>/mcp_private`、
run 目录叫 `<output_root>/<run_id>`——两者在 `run_id="mcp_private"` 时**重合**。

于是该 Run 的 `/history` 就等于私有根目录，容器里能列出**所有** Run 的配置：

```
攻击者 run 目录: <output_root>/mcp_private
该目录内容     : ['run_alpha', 'run_beta']
沙箱内读取 /history/run_alpha/authorizations.json → LEAKED
沙箱内读取 /history/run_beta/authorizations.json  → LEAKED
```

修复后：

```
私有根目录      : <output_root>.mcp_private   （在 output_root 内：False）
攻击者 run 目录 : <output_root>/mcp_private
该目录内容      : []
沙箱内读取两个别的 Run 配置 → 都 unreadable
```

保留的性质：仍按 Run 隔离（路径含 `<run_id>`，dataset_id 前缀不变）、
Server 的数据源 `workspace_root` 校验不变、内容指纹语义不变、
跨进程恢复不变（三进程用例依旧通过）。

对照与回归测试见 `tests/test_mcp_private_root_overlap.py`（19 项），
其中 `test_old_layout_would_have_leaked` 手工把私有树写回旧位置，
用同一容器复现泄漏，以证明新布局那条用例不是因为"路径写错"而通过。

### 9.4 `/runs` API 未暴露 MCP 模式（P2-C 时的实测结论）

> **后续状态（P2-D 已实现）**：本节描述的是 P2-C 交付时的事实。P2-D 已把后端
> 选择接入 HTTP：`POST /runs` 接受 `sql_backend`（`direct` / `mcp`，缺省 `direct`），
> 模式随 Run 持久化并在恢复时沿用。本节保留为当时的记录与 P2-D 的立项依据；
> 当前契约见 README 的「FastAPI」一节与 `/docs` 的 OpenAPI 描述。

`services/run_coordinator.py::_execute` 与 `api.py` 都没有把 `McpSettings`
传给 `run_analysis`，因此**API 发起的 Run 一律使用默认的 direct 后端**。
MCP 模式目前只能通过程序化的 `run_analysis(mcp=McpSettings(mode="mcp"))`
使用（或 `scripts/recovery_driver.py --mcp-mode mcp`）。

已固定的证据（P2-C 时）：

- `run_coordinator.py` 与 `api.py` / `api_runs.py` 中不存在 `McpSettings`；
- 真实 HTTP 创建一次 Run 后，`<output_root>.mcp_private` **不会出现**——
  而 mcp 模式每次调用都会写出该文件（有跨进程用例为证）。

这是**待办**而不是"已支持"：把 MCP 模式暴露到 API 属于 P2-D 的集成任务。
这一点在第十一节与 README 中同样标注。

### 9.5 关于"SQL 授权按位置错配"的更正

P2-C 初版报告声称 `tools/sql_tool.py` 原先用
`zip(tables.values(), table_names)` 推授权表、造成授权错配。

**这个说法是错的。** 核对方式与结论：

```
git show c1a1802:tools/sql_tool.py        # P2-B 基线
git log --all -S 'zip(tables.values()' -- tools/sql_tool.py
```

- P2-B 基线（第 291–334 行附近）已经用 `sqlite_resources` 生成 `allowed_tables`，
  而 `sqlite_resources` 是按 `table_names` 逐个查 `tables` 得到的；
- 整个仓库历史中**从未出现**过 `zip(tables.values(), ...)` 这种写法。

因此：**既不是 P2-B 基线缺陷，也不是开发中期缺陷，而是报告误述。**
工作区的代码从一开始就与基线语义一致（把授权来源收进
`_authorized_targets()` 只是随策略抽取做的整理）。
本文档已删除原先的缺陷描述，也没有为了配合报告去改本已正确的代码。


---

## 十、MCP SDK 版本与安装兼容性

选定 `mcp==1.30.0`（J-1 决策的 1.x 线），实际核验结果：

| 项 | 结果 |
|---|---|
| 安装方式 | `pip install mcp==1.30.0`，成功 |
| 新增包 | 14 个（PyJWT / attrs / cffi / cryptography / httpx-sse / jsonschema / jsonschema-specifications / mcp / pycparser / pydantic-settings / pywin32 / referencing / rpds-py / sse-starlette） |
| **既有包是否被升级** | **无**。pydantic 2.14.1、anyio 4.15.1、httpx 0.28.1、starlette 1.8.0、uvicorn 0.54.0、langchain-core 1.6.10、langgraph 1.2.15 全部保持原版本 |
| `FastMCP` | `mcp.server.fastmcp.FastMCP` 可用；`run("stdio")` 可用 |
| stdio 客户端 | `mcp.client.stdio.stdio_client`、`StdioServerParameters` 可用 |
| 会话 | `mcp.ClientSession`：`initialize` / `list_tools` / `call_tool(read_timeout_seconds: timedelta)` |
| 结果类型 | `CallToolResult` 字段为 `content` / `structuredContent` / `isError`（camelCase） |
| v2 路径 | `mcp.server.mcpserver.MCPServer` **不存在**（ModuleNotFoundError），确认仍在 1.x 线 |

安装前用 `pip install --dry-run --report` 核验过依赖解析结果，确认不会改动既有版本。

**stdout 纪律**：1.x 线没有把 fd0/fd1 从协议通道分离出去，
因此"stdout 只承载协议"必须靠代码约束。本实现的做法：

- Server 内所有日志与致命错误一律写 stderr；
- 工具函数不打印任何东西；
- 有测试用 `subprocess` 直接拉起 Server 并断言 **stdout 为空**；
- 另有测试验证 stderr 上有大量日志时握手与调用仍然正常（证明是通道分离，
  而不是"日志少"）。

---

## 十一、真实测试结果

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_mcp_protocol.py` | 23 | 握手、动态发现、Schema、三工具调用、错误结果、启动失败、启动超时、进程回收、异常摊平、stdout 纪律、配置私有性 |
| `tests/test_mcp_integration.py` | 23 | 数据集授权、越权拒绝、同名隔离、路径越界、CSV/Excel 多 Sheet/SQLite Schema、SQL 五类查询与拒绝、截断标记、库不变、direct/mcp 契约一致、真实经过 Server、环境失败零 Repair、SQL 错误进 Repair、Run 隔离、重复 Resume |
| `tests/test_mcp_config_isolation.py` | 13 | 配置位置、符号链接拒绝、原子写出、真实容器读不到配置、旧位置对照泄漏、模型可见结果不含宿主路径、stderr 不含配置 |
| `tests/test_mcp_cross_process.py` | 8 | 三进程真实恢复、协议活动证明、授权与指纹不变、配置在 run 目录外、无重复结果、direct 对照、`/runs` API 不暴露 MCP |
| `tests/test_mcp_private_root_overlap.py` | 19 | 私有根目录与 output_root 平级、特殊 run_id 不重合、显式配置落在 output_root 内被拒、大小写与 `..` 别名被拒、两 Run 隔离、真实容器读不到其他 Run 配置、旧布局对照泄漏 |

关键证据：

- **真实子进程**：断言 Server pid 与宿主 pid 不同且进程存活；
- **真实经过 MCP**：计数子类证明 `start()` 1 次、`run_readonly_sql` 调用 1 次；
- **direct 与 mcp 结果逐字段相同**：`{'total': 35.0}` 两侧一致；
- **越权拒绝**：`未授权的 dataset_id`、`未授权的数据表 secrets`、
  `不允许访问元数据表 sqlite_master`、`路径越界`；
- **库不变**：连续 4 类写操作后表结构与行内容指纹完全一致；
- **环境失败零 Repair**：`unavailable` 与 `timeout` 两种情况下 `repair` 调用次数为 0；
- **沙箱读不到私有配置**：真实容器内 `/history/mcp/authorizations.json` 返回
  `unreadable`，而同容器下把配置放回旧位置确实泄漏（对照用例）。

---

## 十二、限制与 P2-D 前置事项

### 12.1 限制

1. **MCP Server 不是安全沙箱**，是宿主上的普通进程，权限与启动它的用户相同。
   它不能执行模型生成的 Python 代码（本阶段三个工具都不接受代码），
   但这属于"工具集受限"，不是进程隔离。
2. **传输只用 stdio**。HTTP / SSE 未实现，因此不存在"跨主机调用"这条路径。
3. **每个 Run 一个 Server 子进程，每次工具调用重建会话**。
   长驻会话能省掉握手开销，但会让恢复语义依赖进程状态，因此没有采用；
   代价是每次 SQL 调用都付一次进程启动与握手的成本。
4. **`SQLITE_FUNCTION` 全量放行**（继承自 P2-B）：`load_extension` 被词法层与
   操作码双重拦截，其他内置函数未逐项审计。
5. **视图包装未单独阻止**（继承自 P2-B）：授权表按物理名匹配。
6. **MCP 调用的 exactly-once 无法保证**（见第八节）。
7. **`get_schema` 的类型是粗推断**，不是严格类型系统。
8. **子进程环境白名单是逐项维护的**：将来若某个依赖需要新的系统变量，
   需要显式加入；这不是"自动跟随宿主环境"的设计。
9. **超时后必须重建客户端**：当前实现不会自动重连，调用方拿到 `unavailable`
   就应当失败，而不是继续用同一个失效会话。
10. **`/runs` API 未暴露 MCP 模式**（见 §9.3）：API 发起的 Run 一律走 direct。
    MCP 模式目前只能程序化使用（`run_analysis(mcp=...)`）或经
    `scripts/recovery_driver.py --mcp-mode mcp`。

### 12.2 P2-D 前置事项

1. **把 MCP 模式接入 API**：`RunCoordinator` 需要接受并持久化
   `McpSettings`（模式属于可信运行期配置，必须随 Run 记录，否则恢复时
   无法保证与首次运行一致），`/runs` 需要提供显式的后端选择入口。
   在此之前不得声称 API 支持 MCP。
2. **先决定 MCP 工具的写权限边界**。本阶段三个工具全部只读。
   若要引入写类工具，"恰好一次"与补偿语义必须先设计好——
   当前只能做到"不静默重发"，不足以支撑写操作。
3. **会话复用与恢复语义的取舍**需要显式决定：长驻会话能降开销，
   但会引入"进程状态参与恢复"的问题，与 P1/P2-B 的语义有冲突。
4. **工具清单需要版本化**：`ALLOWED_MCP_TOOLS` 是硬编码集合。
   若工具集要扩展或来自第三方 Server，需要一个可信的清单来源与校验机制。
5. **授权粒度**：当前按数据集授权。若将来要按列或按行授权，
   `DatasetRef` 与 SQL 策略都需要扩展，不能靠"在提示词里要求模型别查"实现。
6. **可观测性**：MCP 的请求/响应目前只在 stderr 上，没有结构化审计记录。
   与 P1 的 LLM Ledger 对齐需要新的账本设计。
7. **跨平台**：本实现已在 Windows 上验证；Unix 上的进程回收路径
   （`start_new_session` + `killpg`）未实测。
