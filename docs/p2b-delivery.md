# P2-B 交付说明：逻辑尝试身份、SQL 纵深防护、大结果恢复

本文记录 P2-B 三个模块解决的问题、缺陷根因、修复方式与验收证据。

与前序文档的关系：`docs/p2a-delivery.md` 保留 P2-A 当时的实现与验收状态，
其中记录的三个问题（Repair 多轮调用去重、SQL 资源限制、大结果消费）
**已在本阶段修复**，本文是这些问题的当前状态说明。

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1 | Checkpoint、状态恢复、Human-in-the-Loop、只读 SQL | 已完成 |
| P2-A | Docker 执行沙箱、staging 产物发布、错误分类短路 | 已完成 |
| **P2-B** | **逻辑尝试身份、SQL 纵深、大结果恢复** | **本文** |
| P2-C | MCP 接入 | **尚未实施** |

---

## 零、P2-B 修的三个问题（先看这里）

### 问题 1：修复循环在跑，但模型只被问过一次

`utils/idempotency.llm_operation(run_id, node, subject)` 的 subject 只含
`task_id` / `chart_id`，不含逻辑尝试序号。第 2、3 轮修复算出与第 1 轮**相同**的
操作 ID，`already_done` 命中后直接复用第 1 轮生成的代码。

后果不只是"少调几次模型"：修复循环后两轮**从未真正尝试修复**，
却照常消耗了重试额度并记账，最终以"修复失败"收场——
从状态上看像是"试了三次都没修好"，实际上是"只试了一次，重复了两次"。

同一个缺陷存在于四条路径，`critic` 那条最隐蔽（见第一节）。

### 问题 2：非只读 SQL 只有词法一道闸门

P1 只有 `ensure_read_only` 词法校验 + `mode=ro` 连接。词法判定本质是**猜测**，
而 SQL 的表达方式极其丰富；一旦词法判定失效，就只剩文件级只读兜底，
且结果集没有任何大小或时间上限——一条自连接就能把宿主内存耗光。

### 问题 3：大结果外置后，下游拿摘要当数据用

`utils/result_artifacts.load_result()` 定义了但**没有任何调用点**。
结果超过 32 KiB 被外置后，state 里只剩
`{"__kind__": "externalized", "row_count": …, "size": …}`，
而绘图节点把它当分析结果送进了容器——**图画错或画空，工作流却不报错**。
报告节点则反过来，把整份数据 JSON 塞进 Prompt。

这两个方向都是错的：绘图需要真实数据，报告只需要概要。

---

## 一、P2-B1：逻辑尝试身份

### 为什么"只检查 repair 是不够的"

先枚举每个 Agent 在同一个 Run 内可能被调用几次。判据是图上的边：
`repair → tool_execution`、`refine → tool_execution`（都不回到 `execution_node`），
因此 `execution_node` 每个任务**只进入一次**，而 `tool_execution` 会被反复进入。

| 节点 | 每任务/图表调用次数 | subject（修复前） | 是否有缺陷 |
|---|---|---|---|
| `execution_node`（executor） | 1 | `task_id` | 否 |
| `repair_node` | ≤ MAX_RETRIES | `task_id` | **是** |
| `critic_node` | ≤ MAX_REFINES + 1 | `task_id` | **是** |
| `refine_node` | ≤ MAX_REFINES | `task_id` | **是** |
| `plot_generation_node`（plotter） | 每图表 1 | `chart_id` | 否 |
| `plot_repair_node` | ≤ MAX_PLOT_RETRIES | `chart_id` | **是** |
| `visualization_planner_node` | 1 | `"plan"` | 否 |
| `planner_node` / `report_node` | 1 | `"plan"` / `"report"` | 否 |

`critic_node` 那条最容易漏掉：每轮精修后都应**重新评审**新代码，
但第二轮评审复用了第一轮的结论——精修出来的代码从未被真正校验，
循环等于空转。"只要 critic 说通过就推进"这个前提被破坏了。

### 修复

在 `utils/idempotency.py` 增加逻辑尝试标识构造函数，让 subject 承担区分职责：

```python
_ATTEMPT_MARK = "#"

def attempt_subject(base: str, *counters: int | None) -> str:
    """把逻辑尝试序号编进 subject。
    例如 attempt_subject("task_1", 0) → "task_1#0"。
    """
```

三个关键决定，每个都有理由：

1. **不用时间戳、UUID 或随机数。** 恢复是从上一张 Checkpoint 重算操作 ID 的，
   ID 必须可复现，否则幂等直接失效。
2. **计数器编进字符串，而不是只传给账本。** 这样声明式幂等登记、账本核对与日志
   看到的是同一个值，不会出现"账本记的是第 2 次、幂等表按第 1 次去重"。
3. **传入的是入站计数器，不是它的 `+1`。** 节点里 `retry_count = state.get(...) + 1`
   发生在调用**之后**；用 `+1` 的话重放时也算得出同一个值，但语义会变成
   "第 N 次修复记作 N+1"，与 `execution_results` 里的 `retry_count` 对不上。
   用入站值可以保证：同一张 Checkpoint 恢复出的入站值相同 → 同一个 ID → 命中复用。

四个节点的调用点：

| 节点 | subject | 账本 attempt |
|---|---|---|
| `repair_node` | `attempt_subject(task_id, retry_count)` | `retry_count + 1` |
| `refine_node` | `attempt_subject(task_id, refine_count)` | `refine_count + 1` |
| `critic_node` | `attempt_subject(task_id, refine_count)` | `refine_count + 1` |
| `plot_repair_node` | `attempt_subject(chart_id, plot_retry_count)` | `plot_retry_count + 1` |

`critic` 与 `refine` 共用 `refine_count`，但节点名不同
（`llm:critic:…` vs `llm:refine:…`），因此不会互相碰撞。

`run_agent_call` 新增 `attempt` 参数，**只用于账本上的人读标注**，
不参与去重计算——真正的幂等身份始终来自 `operation_id`。

### 保留的语义

| 情形 | 行为 |
|---|---|
| 不同逻辑尝试 | 操作 ID 不同 → 允许新的模型请求 |
| 同一尝试的 Checkpoint 重放 | 操作 ID 相同 → `already_done` 复用，不产生新请求 |
| 请求已发出、结果未确认 | 账本记录未确认 → `guard_llm_calls` 中断等人工决定 retry/skip |

P1 的账本与人工审批**没有为这次修复让路**：没有关闭账本、没有跳过审批、
没有引入随机 ID。`tests/test_llm_attempt_identity.py` 用真实子进程 +
`os._exit(17)` 验证了这一点。

### 修复前后实测

| 循环 | 修复前模型调用次数 | 修复后 |
|---|---|---|
| Repair | `repair: 1` | `repair: 3`（ID `task_1#0/#1/#2`） |
| Refine | `refine: 1`、`critic: 1` | `refine: 3`、`critic: 4` |
| PlotRepair | `plot_repair: 1` | `plot_repair: 3`（ID `chart_1#0/#1/#2`） |

---

## 二、P2-B2：SQL 纵深防护

### 结构

```
一次 SQL 调用
  ├─ 1. 词法只读校验（utils/sql_guard）        拒绝非 SELECT/WITH、多语句、注释绕过
  ├─ 2. SQLite Authorizer                      按操作码 + 授权表白名单
  ├─ 3. mode=ro + uri=True 连接                文件级只读
  ├─ 4. set_progress_handler + 单调时钟预算    中断长查询
  └─ 5. 有界 fetchmany（每批 500）             行数与字节上限
```

前两层回答"**能否执行**"，后两层回答"**执行多久、返回多少**"。

### 第 1 层与第 3 层：保留原样

第 1 层词法校验（`utils/sql_guard.ensure_read_only`）与第 3 层
`mode=ro + uri=True` 连接都是 P1 的实现，**本阶段未改动**：
词法的保守判定（去注释与字面量后再匹配、只接受单条语句、无法确定即拒绝）
仍然有效，只读 URI 仍由 `sql_tool._read_only_uri` 构造。
P2-B 只是在它们之外增加了下面的三层。

### 第 2 层：Authorizer

采用**显式白名单**，只放行为完成一次只读 SELECT 所必需的操作码：

```python
_ALLOWED_ACTIONS = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
})
```

其余一律 `SQLITE_DENY`，覆盖写、DDL、`ATTACH`/`DETACH`、`PRAGMA`、事务控制。
另外两道表级检查：

- **授权表白名单**：只允许本次请求点名的表（按 `database_table` 物理名匹配），
  未点名的表一律拒绝；
- **元数据表拒绝**：`sqlite_master` / `sqlite_schema` / `sqlite_temp_master`
  显式拒绝，避免模型枚举库里还有哪些表或读到建表语句。

Authorizer 回调无法抛异常（SQLite 会把异常吞成通用错误），
因此拒绝原因先写进闭包里的列表，再由上层转成可读的 `SqlPolicyError`。

### 第 4 层：查询超时

用 `set_progress_handler` + `time.monotonic()` 的**绝对截止时间**（不是累加计时）：

```python
deadline = time.monotonic() + timeout_seconds
def progress_handler() -> int:
    if time.monotonic() >= deadline:
        timed_out["value"] = True
        return 1        # 非 0 → SQLite 中断当前语句
    return 0
```

这里的实现细节值得记下来：超时处理必须**同时覆盖 `execute()` 与取行阶段**。
大连接与排序都发生在 `cursor.execute()` 内部，最初只在取行循环里捕获
`OperationalError`，于是 `execute()` 阶段的超时被当成普通执行错误报出去，
而不是 `timeout`。测试用"预算 0 + 三重笛卡尔自连接"稳定复现并锁定了这个分支。

**能力边界**：进度回调只能中断 SQLite 虚拟机，**不是强制终止 Python 进程的
完整隔离手段**。SQL 仍在宿主进程内执行，不具备 Docker 级隔离，
本文不做相反的声称。

### 第 5 层：有界结果

不能先 `read_sql_query` 再截断——那等于把完整结果读进内存，
内存峰值由数据决定，"限制"形同虚设。这里用游标分批取：

```python
while len(rows) < max_rows:
    batch = cursor.fetchmany(min(500, max_rows - len(rows)))
    ...
    if total_bytes + size > max_bytes:
        truncated = True; break
```

默认预算：`max_rows=10_000`、`max_bytes=4 MiB`、`timeout_seconds=10.0`，
经 `create_sql_tool(...)` 可调。

三个细节：

- **行数与字节数都要有**：行数控制"多少条"，字节数控制"多大"，
  因为单行也可能极宽（例如 `group_concat` 的结果）。
- **行数正好用满时要再探一行**，只有确实还有更多才标记 `truncated`——
  否则"恰好 10 行的结果"会被误报为截断。
- **BLOB 显式标记**为 `{"__kind__": "blob", "size": n}`，
  不悄悄转成文本让调用方误当成字符串使用。

返回结构从"裸列表"改为带截断元数据的结构：

```json
{"columns": [...], "rows": [...], "row_count": 12, "bytes": 480, "truncated": false}
```

**触发截断时必须显式标记**，禁止把部分结果当完整结果。

### 分层事实（不得误报拦截层级）

DELETE / PRAGMA / ATTACH / UPDATE / INSERT 等语句在**词法层就被拒了**，
经 `run_query` 走不到 Authorizer。因此：

- 经工具入口的拒绝测试验证的是**第 1 层**；
- 第 2 层单独验证：用**可写连接**（刻意不用 `mode=ro`）直接施加 Authorizer，
  证明它独立于词法校验挡住写入；配合逐操作码的
  `SQLITE_INSERT/DELETE/UPDATE/CREATE_TABLE/DROP_TABLE/ALTER_TABLE/ATTACH/DETACH/PRAGMA/...`
  → `SQLITE_DENY` 参数化测试。

这是"纵深"的可验证含义：任何一层单独失效都不应导致写入或越权访问发生。

---

## 三、P2-B3：大结果外置、恢复与两种消费

### 外置策略（未改变）

小结果（≤ `INLINE_LIMIT_BYTES` = 32 KiB）内联进 state；
大结果落成带 sha256 的 JSON 产物，state 里只留**摘要**与引用。
摘要现在额外带上 `columns`，让下游知道形状。

### 两种消费者必须分开

| 消费者 | 需要什么 | 走哪条路 |
|---|---|---|
| 绘图（`plot_execution_node`） | **真实数据** | `load_result` 按引用读回 |
| 报告（`report_node` → reporter Agent） | **概要** | `build_preview` 有界预览 |

### 绘图路径

```python
analysis_result = load_result(
    source_execution_result.get("result"),
    source_execution_result.get("result_artifact"),
    run_id=state.get("run_id"),
    output_root=current_runtime().output_root,
)
...
"analysis_result": unwrap_records(analysis_result),
```

`load_result` 的校验顺序有讲究：**先归属、再路径、再存在性、最后内容哈希**。
归属检查放在最前，避免"读到别的 Run 的文件之后才发现不该读"。

| 校验项 | 失败时 |
|---|---|
| 引用含路径 | `结果产物引用缺少路径` |
| 路径在 `output_root` 之内 | `结果产物路径越界` |
| 属于本次 Run（`<root>/<run_id>/results/`） | `结果产物不属于本次 Run` |
| 存在 | `结果产物已丢失` |
| 是普通文件 | `结果产物不是普通文件` |
| 大小在读取预算内（64 MiB） | `结果产物超出读取预算` |
| sha256 一致 | `结果产物内容已变化` |

产物不可用时绘图节点**明确失败**：返回
`plot_error="绘图所需的分析结果不可用：…"` 且不记录任何图表——
不会静默画出一张空图。

`unwrap_records` 把 `{"__kind__": "dataframe", "records": [...]}` 拆成记录序列，
再交给 `sandbox.bootstrap.build_result_dataframe`。
这样沙箱侧不需要认识信封，P2-A 的数据协议与输出目录限制都没动。

### 报告路径

```python
execution_results=build_report_execution_previews(
    final_execution_results, run_id=..., output_root=...,
)
```

`build_preview` 返回**有界摘要**：`kind`、`available`、`columns`、`row_count`、
`total_row_count`、`preview_rows`、`sample`（≤20 条、≤8 KiB）、`truncated`，
以及说明：

> 完整结果已外置到产物文件，这里只是有界预览；不要把它当作完整数据集。

预览**可降级**：产物读不到时返回 `available: false` + 原因，
报告仍能生成，但会说明数据不可用——而不是让整个报告失败，
也不是假装数据可用。

### 修复中发现的第二个缺陷

`_row_count` 原先只认 `DataFrame` / `Series` / `list`。但**沙箱执行路径返回的
已经是 JSON 化的信封**（`{"__kind__": "dataframe", "records": [...]}`），
于是外置摘要的 `row_count` 变成 `None`——报告与预览都失去了行数信息。
已修正为同时识别信封形态。

### 不把完整内容写回 state

绘图读回的是 JSON 记录，用 `unwrap_records` 拆封后**只传给工具**，
不写回 LangGraph State。`test_state_stays_serializable` 断言
`execution_results` 整体可 `json.dumps`。

---

## 四、验收证据

### 测试规模

```
530 passed / 0 failed / 0 skipped        （全量 pytest）
```

| 分组 | 用例数 |
|---|---|
| P0 / P1 / P2-A 既有 | 388（全部保留） |
| P2-B 新增 | **142** |

新增分布：

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_sql_hardening.py` | 75 | 只读查询、Authorizer 逐操作码、越权表、超时、有界结果、库不变 |
| `tests/test_large_result_recovery.py` | 39 | 外置、还原、完整性失败、预览、类型往返、拆封 |
| `tests/test_llm_attempt_identity.py` | 20 | 多轮调用、反馈传递、重放、账本审批、Run 隔离 |
| `tests/test_large_result_consumers.py` | 8 | 两条消费路径的端到端行为 |

Docker 用例分类（实测口径，由 `scripts/sandbox_report.py` 汇总：
hook 观测 `sandbox.executor.containers_spawned()` 在每个用例期间的增量，
不靠人工标注）：

| 类别 | 数量 |
|---|---|
| A. 真实启动容器 | 48 |
| B. 故障模拟（需 Docker 可用，不启容器） | 23 |
| C. 静态断言（与 Docker 无关） | 4 |
| **合计** | **75** |
| 跳过 | **0** |

（`-m sandbox -rs` 实测 `75 passed, 0 skipped`，无跳过原因输出。）

### P2-B1 证据

- Mock 自己记录调用（不经账本）：`repair` 3 次、`refine` 3 次、`critic` 4 次、
  `plot_repair` 3 次；`executor`/`planner`/`reporter`/`visualizer` 各 1 次。
- 账本 operation_id：`…:llm:repair:task_1#0/#1/#2`、`…:llm:critic:task_1#0..#3`、
  `…:llm:plot_repair:chart_1#0..#2`。
- 每轮携带各自的错误反馈：断言第二轮修复的提示词含第二次的错误，
  第二轮评审的提示词含精修后的代码。
- 第二轮修复成功的用例断言代码真被执行：`successes[-1]["result"] == 3`。
- 重放不额外计费：同一 `SqliteSaver` 上续跑，`recorder.calls` 长度不变。
- 未确认仍等审批：`after_request` 在第一次修复崩溃 → 未确认记录恰 1 条且
  `endswith("#0")`；恢复不给许可 → `pending_approval["kind"] == "llm_replay"`
  且**上游调用记录逐字不变**；给 `--grant-llm-retry` → Run 完成，
  `repair` 身份仍只有 `#0`（重发的是同一逻辑尝试），`unconfirmed_operations` 为空。

### P2-B2 证据

- Authorizer：18 个写/DDL/ATTACH/PRAGMA/事务操作码逐个 `SQLITE_DENY`；
  4 个只读操作码 `SQLITE_OK`；越权表与元数据表读取被拒。
  另用**可写连接**验证它独立于词法层挡住 DELETE（拒绝后库指纹不变）。
- 词法层：15 类危险语句经 `invoke_tool` 被拒，`error_kind == "execution_error"`。
- 超时：400 行三重笛卡尔自连接（6.4×10⁷ 中间行）在 0.5 s 预算下被中断；
  预算 0 时经工具入口返回 `error_kind == "timeout"`。
- 有界结果：行数上限截断并标记；恰好等于上限不误报；
  单行超大（`group_concat` 超 1 KiB）返回 0 行 + `truncated`。
- 库不变：14 类攻击后库指纹（表清单 + 行数 + 内容 sha256 + 建表 SQL）逐字节不变；
  无 `-wal`/`-journal` 残留；失败查询后仍可正常写事务；超时后连接已释放。

### P2-B3 证据

- 20,000 行完整还原：`len == 20000`，首行 `C000000` / 末行 `C019999` 与
  `amount` 值抽查通过。
- 重启后重载：丢弃原始对象、只留 JSON 化后的摘要与引用，仍能还原。
- 完整性失败逐项验证：缺失、哈希不符、跨 Run、路径越界、目录冒充文件。
- 预览与真实数据的区别：`preview["truncated"] is True`、
  `len(preview["sample"]) < len(real)`、预览 JSON < 16 KiB。
- 端到端（8,000 行走完整工作流）：报告提示词含 `externalized`、预览说明、
  真实行数、列名，且 `< 100 KiB`；绘图经容器画出的 PNG > 5 KiB。

**数据量取值的说明**：走沙箱工作流的结果必须 < 1 MiB（沙箱结果上限），
因此端到端用 8,000 行（≈360 KiB，> 32 KiB 会外置）；
20,000 行的外置与还原由单元层直接测 `store_result` / `load_result` 覆盖。
这不是绕过限制，而是两个上限共同决定的取值。

---

## 五、与 P1、P2-A 的兼容性

| 项 | 状态 |
|---|---|
| LangGraph 工作流拓扑（19 节点 + 1 审批关口） | 未改动 |
| Checkpoint 与跨进程恢复 | 保留，相关测试全绿 |
| Human-in-the-Loop | 保留 |
| LLM Ledger 与未确认调用人工审批 | 保留，且新增修复路径下的验证 |
| Docker 执行沙箱（Python/Plot 一律过容器） | 保留，无同进程 exec 回退 |
| staging 产物发布与容器写入限制 | 保留，16 项写入限制测试全绿 |
| `error_kind` 环境错误短路 | 保留，环境错误修复调用次数为 0 |
| ModelGate 追踪注入 | 未改动 |
| P2-A 沙箱数据协议与输出目录限制 | 未改动（`unwrap_records` 只在宿主侧拆封） |

### 既有断言的四处更新

都先说明原因再改，未删除任何测试、未放宽任何安全断言、未修改 Mock 调用计数：

| 文件 | 原断言 | 改为 | 原因 |
|---|---|---|---|
| `test_error_classification.py` | `repair` 调用次数 == 1 | == `MAX_RETRIES` | 原断言锁定的正是 P2-B1 要修的缺陷 |
| `test_llm_crash_recovery.py` | 字面量 `…:llm:plotter:chart_1` | `plotter_operation(...)` 返回 `…#0` | 绘图 subject 现在带图表序号 |
| `test_sql_readonly_chain.py` | `len(outcome["result"]) == N` | `outcome["result"]["row_count"] == N` | SQL 结果改为带截断元数据的结构 |
| `test_tools.py` | `len(outcome["result"]) == 3` | `row_count` + `truncated` + `rows` 三项 | 同上 |

`test_llm_crash_recovery.py` 只改了操作 ID 的**构造方式**：
`test_resume_pauses_instead_of_resending`、`test_resume_does_not_silently_resend`、
`test_declining_produces_a_terminal_failed_state`、
`test_ledger_cannot_tell_whether_upstream_processed_the_request` 等
审批与账本语义断言全部原样保留。

---

## 六、已知限制

### 6.1 SQL（本阶段的主要残留风险）

1. **SQL 仍在宿主进程内执行**，隔离强度**低于** Python/Plot 的 Docker 沙箱。
   Authorizer 与进度回调都在同一进程内，**不具备容器级隔离**；
   恶意 SQL 仍可能消耗宿主 CPU（时间预算会中断，但中断前已消耗）。
2. **进度回调只能中断 SQLite 虚拟机**，无法强制终止 Python 进程。
3. **视图包装未单独阻止**：授权表按 `database_table` 物理名匹配，
   若库内存在与授权表同名的视图，`SQLITE_READ` 层面看到的是底层表。
   未单独阻止"视图包装越权表"这一构造。
4. **`SQLITE_FUNCTION` 全量放行**：`load_extension` 由词法层与
   `SQLITE_LOAD_EXTENSION` 操作码双重拦截，但其他内置函数的边界未逐项审计。
5. **结果上限可被调用方放宽**：`invoke_tool` 走默认值
   （10,000 行 / 4 MiB / 10 s），没有从配置中心统一下发，也未按 Run 分级。
6. **词法层与 Authorizer 高度重叠**：正常路径上危险语句都被词法层拦掉，
   Authorizer 防的是"词法层被绕过"这一情形——这一点在测试里是人为构造才验证到的，
   不应被描述为"Authorizer 拦住了日常攻击"。

### 6.2 大结果

7. **32 KiB 以下的结果仍原样进入 Checkpoint**：一个很宽的百行表可能不足 32 KiB，
   会完整写进 state。
8. **`LOAD_MAX_BYTES = 64 MiB` 是保护性上限，不是安全边界**：
   它防的是意外巨大文件，不防恶意构造的合法大小文件。
9. **预览的降级是"可读性降级"**：产物不可读时报告仍会生成，
   只是说明数据不可用。这不等于报告内容正确——需要人工判断。

### 6.3 继承自 P2-A（未改变）

10. 容器与宿主共享内核，Docker 隔离**不是绝对安全保证**；非 rootless。
11. 资源限制只约束单容器；单进程容器信号量不是跨服务实例的宿主级配额。
12. 符号链接拒绝逻辑未在有创建权限的 Windows 环境做真实端到端验证。
13. 基础镜像 digest 未与官方 manifest-list digest 比对（本机 registry 不可达）。
14. wheel 哈希与 Debian 系统包版本未完全固定。
15. 绘图产物发布非原子。

### 6.4 LLM 调用

16. **仍然不能声称恰好一次**。账本的 `started` 无法区分"上游已处理"与
    "上游没收到"，因此恢复时只能做到"不静默重发"，做不到精确计数。
    P2-B 扩大了"不同尝试"的区分度，但没有改变这个根本边界。

---

## 七、P2-C 前置事项

**MCP 尚未实施。** 本文与 README 都不声称已支持 MCP。若 P2-C 要做 MCP 接入：

1. **基础事实（P2-A 阶段已实测）**：PyPI `mcp` 最新为 2.x，
   而 2.x 中 `from mcp.server.fastmcp import FastMCP` 会抛 `ModuleNotFoundError`
   （已改名为 `MCPServer`，位于 `mcp.server.mcpserver`）。
   若选 1.x 线（如 `mcp==1.30.0`），`FastMCP`、`mcp.client.stdio.stdio_client`、
   `StdioServerParameters`、`ClientSession` 均可用。
2. **stdout 纪律**：1.x 线没有把 fd0/fd1 从协议通道里分出去，
   因此"stdout 只承载协议"必须靠代码约束（禁止 `print`）+ 静态测试保证。
3. **与现有沙箱的关系需要先决定**：MCP server 是本机子进程还是容器内进程，
   直接决定它是否受 P2-A 的挂载与网络限制约束。这一条应在动工前明确，
   不要先写代码再补边界。
4. **工具清单与授权**：MCP 工具会成为新的"可被模型调用的执行路径"，
   需要与 `tool_registry` 的契约（`{success, result, error, error_kind}`）、
   幂等登记与 `error_kind` 路由对齐，否则会出现第二条不受约束的执行路径。

### 其他可在 P2-C 一并处理的事项

- SQL 迁入容器或独立进程（消除第 6.1 节的第 1、2 条）。
- 宿主 `requirements.txt` 固定版本（沙箱侧已独立锁定）。
- 视图/授权边界的补齐（第 6.1 节第 3 条）。

---

## 八、修改文件清单

### 修改（10）

| 文件 | 改动 |
|---|---|
| `utils/idempotency.py` | 新增 `attempt_subject`；`run_agent_call` 增加 `attempt` |
| `nodes.py` | 四处 subject 带尝试序号；绘图经 `load_result`；报告用有界预览 |
| `tools/sql_tool.py` | Authorizer、进度中断、有界 fetchmany、授权表限制、截断元数据 |
| `utils/result_artifacts.py` | `load_result` 归属/路径校验；`from_jsonable`、`unwrap_records`、`build_preview`；`_row_count` 识别信封 |
| `scripts/recovery_driver.py` | `--crash-on-sequence` / `--fail-first-execution` / `--repair-sequence-fails` 与按序回复 |
| `tests/test_error_classification.py` | 更新锁定旧缺陷的断言 |
| `tests/test_llm_crash_recovery.py` | 绘图操作 ID 集中构造 |
| `tests/test_sql_readonly_chain.py`、`tests/test_tools.py` | SQL 结果结构断言更新 |
| `docs/evidence/modelgate-success.json` | 重跑取证文件刷新（预期行为） |

### 新增（5）

`tests/test_llm_attempt_identity.py`、`tests/test_sql_hardening.py`、
`tests/test_large_result_recovery.py`、`tests/test_large_result_consumers.py`、
`docs/p2b-delivery.md`（本文）

### P2-B 收尾阶段（文档整理与测试复核）

| 文件 | 改动 |
|---|---|
| `docs/p2b-delivery.md` | 新增（本文） |
| `docs/p2a-delivery.md` | **仅在顶部增加 5 行后续状态说明**（`git diff --numstat` = 5 增 0 删），历史内容不改写 |
| `README.md` | 核心能力补充沙箱与 SQL 强化；当前限制据实更新（Docker 已完成、SQL 仍在宿主、MCP 未实施）；后续计划移除已完成项并补 MCP |
| `scripts/sandbox_report.py` | 分类清单补上 `tests/test_large_result_consumers.py`（新增了一项真实容器的绘图用例） |
| `tests/test_sql_readonly_chain.py` | 静态守卫断言**加强**：新增 `set_authorizer` / `set_progress_handler` / `fetchmany` 必须出现在工具源码中 |

