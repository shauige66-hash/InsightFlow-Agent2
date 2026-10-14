# P1 交付说明：Checkpoint、状态恢复与 Human-in-the-Loop

本文记录 P1 的架构、状态生命周期、幂等设计与恢复语义。

---

## 零、恢复语义速查（先看这里）

以下每一条都由测试锁定；对应测试文件见表末。

### 1. 跨进程恢复能力

`SqliteSaver` 把图状态写到 `outputs/checkpoints/insightflow.sqlite`，
`thread_id` = `run_id`。恢复是**真的新进程**：测试用
`os._exit(17)` 强杀，再由独立子进程重新打开同一个文件继续执行。
覆盖场景：正常中断、进程崩溃、审批等待、绘图阶段中断、重复恢复、多 Run 并存、
数据源丢失/变更。**不是**在同一进程里换一个 Graph 对象。

### 2. 人工审批的实际执行与恢复路径

`require_plan_approval=True` 时，`approval_gate` 节点在 `router` **之前**
（此时无任何副作用）调用真实的 `interrupt()`。恢复路径：

```
POST /runs（require_plan_approval=true）
  → 图停在 approval_gate，返回 __interrupt__
  → RunCoordinator._finalize 落 waiting_approval + pending_approval
  → 进程可退出；重启后 GET /runs/{id} 仍能读到待审批内容
  → POST /runs/{id}/approval {decision: approve|reject}
  → Command(resume={...}) 交回节点，继续或终止
```

- `approve` → 继续执行到完成；
- `reject` → 直接 `END`，**不生成报告**，`run_status=failed`、
  `analysis_result` 记 `failed`、`status_reason="分析计划未获批准"`；
- 审批 ID 由 `run_id + 节点` 决定（稳定值），重复审批不会二次触发；
- `interrupt()` 之前没有任何不可重放的动作。

### 3. LLM Ledger 的 started / completed / uncertain 语义

模型调用与文件副作用的保护机制**不同**，原因见第 4 条。

| 记录 | 写入时机 | 含义 |
|---|---|---|
| `started` | 请求**发出之前**（`fsync`） | 这次调用已经可能到达上游 |
| `completed` | 请求返回之后 | 结果已拿到并会随节点提交 |
| （无 `completed` 的 `started`） | — | **未确认**：可能已计费，也可能没发出 |

`llm_ledger.summarize()` 输出 `unconfirmed_operations` 与 `replayed_operations`。
恢复时 `_unconfirmed_llm_calls()` 把它读出来写进 `state.pending_llm_calls`，
由 `guard_llm_calls` 在节点执行 Agent 之前拦截。

> `state` 中必须**显式声明** `pending_llm_calls`。未声明的键会被 LangGraph
> 静默丢弃，守卫就不会触发——这一点曾被实际踩到。

### 4. 无法判定上游是否计费（限制，不是能力）

`started` 无法区分"上游已处理"与"上游没收到"。因此：

- 账本**只报告"未确认"**，`summarize()` 中不存在 `processed` / `charged` 之类的字段；
- **不提供 exactly-once**；
- 人工批准重发**就是**会产生第二次上游请求（测试中 plotter 计数由 0 变 1）；
- 系统选择保守策略：**默认不重发**，把不确定交给人工，而不是静默再打一次。

### 5. retry / skip 之后的状态处理语义

分两段，因为第一次恢复时系统还不知道人要选什么：

| 步骤 | 行为 | 结果状态 |
|---|---|---|
| 第一次 resume | **仅呈现**中断，不执行任何节点 | 仍是 `waiting_approval` |
| 之后再 resume 且**未**带重发许可 | 视为放弃 | `run_status=failed`，`status_reason` 说明原因，并写 `llm_replay_declined` 审计事件 |
| 带 `{"llm_decision": "retry"}` | 清空未确认清单并重发 | 继续执行至完成 |

放弃是**受控终止**：通过一次纯状态写入（`update_state`）落成终态，
不推进任何节点，因此不产生副作用；终态之后再次 resume 只会返回同一状态，
不会重复执行、也不会重复询问。

### 6. 文件系统与 SQLite 不构成跨资源原子事务

明确不这样描述。可观测窗口：文件已落盘 → 进程被杀 → 节点 state 增量未提交
→ Checkpoint 里"没做过" → 恢复时重放该节点。

- **能保证**：产物路径稳定（同一 run 的同一产物始终同一路径，重放是覆盖而非产生副本）；
  产物清单与图表记录按 ID 去重；反复 resume 不增长。
- **不能保证**：写文件与写 Checkpoint 之间的原子性。写文件中途被杀可能留下半写文件。
  需要"读到的产物一定完整"时应改用 `utils/result_artifacts.py` 的做法
  （临时文件 + sha256 + 原子 rename）。

### 7. Run 心跳只用于识别可能失联的执行者

`RunStore` 记录 `owner_pid` 与 `heartbeat_at`，执行期间后台线程定期刷新。
`is_stale_running()` = 持有者进程不存在 **或** 心跳过期。

它**只**用于可观测性：API 暴露 `stale` / `resumable` 字段，**不触发自动重启**，
不构成安全判断。`running` 本身无法区分"正在执行"与"执行者已崩溃"，
心跳是启发式补充。

### 8. Python `exec` 尚未沙箱化

`tools/python_tool.py` 与 `tools/plot_tool.py` 仍是**同进程 `exec`**：
`safe_builtins` 可逃逸、`pd` / `plt` 已注入使文件系统可达、**全仓无超时**。
P1 没有改善这一点，也没有把它称为安全边界。
**禁止将系统暴露给不可信用户执行任意代码。**

SQL 路径另有说明：见第六节。

### 9. P1 指标均为本地 Mock 工程验证

所有测试使用脚本化假模型或本机 Mock Provider，**零真实付费调用**。
调用次数、恢复行为、心跳判定都是单机确定性验证，
**不代表真实模型服务的能力、计费或生产吞吐**。
不把 Mock 的 Token / 成本当作真实模型指标。

---

## 一、架构

```
                      ┌─────────────────────────────────────────┐
   POST /runs ───────►│ RunCoordinator（单进程有界线程池）        │
                      │  · RunStore：生命周期登记                │
                      │  · SqliteSaver：Checkpoint               │
                      └───────────────┬─────────────────────────┘
                                      │ use_runtime(ContextVar)
                                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ LangGraph：19 节点确定性工作流 + 1 个审批关口                 │
   │                                                              │
   │  loader → context → knowledge → planner                      │
   │                                    │                         │
   │                     ┌──────────────┴───────────────┐         │
   │                     │ require_plan_approval ?      │         │
   │                     └──────┬────────────────┬──────┘         │
   │                            │ 是             │ 否             │
   │                            ▼                │                │
   │                    approval_gate ───────────┘                │
   │                    （interrupt）  │ approve                  │
   │                                   ▼                          │
   │  router → execution → tool_execution → critic → …            │
   │                                                → report       │
   │                                                → artifact     │
   └──────────────────────────────────────────────────────────────┘
```

三个关键决策：

1. **运行期依赖用 ContextVar，不用 state。** `ModelFactory` 与
   `ResourceResolver` 不能序列化，跨进程恢复也没有意义。ContextVar 按上下文隔离，
   两个并发 Run 各自拿到自己的运行时。
   刻意不用 `config: RunnableConfig` 形态的节点参数：LangGraph 会把
   RunnableConfig 的字段名（`configurable`、`callbacks`、`tags` …）当成图通道，
   其中 `configurable` 是保留名，会直接导致 `compile()` 失败。

2. **state 只放可序列化的身份信息。** 表以 `table_snapshot`
   （路径 + 内容指纹 + sheet/表名）表示，真实 DataFrame 由 resolver 按需读回。

3. **审批关口在 router 之前。** 此时没有任何副作用，中断可安全重放。

---

## 二、状态生命周期

`run_status` 与 `analysis_status` 是两件事：

- `run_status`：这次**执行**走到哪一步；
- `analysis_status`：**分析本身**产出了什么（P0 已有语义，保持不变）。

| run_status | 含义 | 可恢复 |
|---|---|---|
| `pending` | 已登记，尚未开始 | — |
| `running` | 正在执行 | 进程崩溃后可用 resume 续跑 |
| `waiting_approval` | 已中断，等待人工审批 | 是（提交审批或 resume） |
| `completed` | 正常结束（含 partial） | 否 |
| `failed` | 执行失败或分析未产出 | 视 Checkpoint 情况而定 |
| `interrupted` | 被主动中断 | 是 |

流转：

```
pending → running ─┬─► waiting_approval ─┬─► running → completed
                   │                     └─► failed（拒绝）
                   ├─► completed
                   └─► failed
```

**进程重启后的任务处理语义**（明确边界）：

- `waiting_approval`、`completed`、`failed` 是持久化的，重启后可直接查询；
- 内存中的执行队列会丢失。重启时处于 `running` 的 Run **不会自动续跑**——
  自动续跑会在无人监督的情况下重放副作用。这些 Run 需要显式 `POST /runs/{id}/resume`；
- 因此本阶段是"单进程调度 + 持久化状态 + 显式恢复"，不是分布式任务队列。

---

## 三、Checkpoint 存储

| 项 | 值 |
|---|---|
| 实现 | `langgraph.checkpoint.sqlite.SqliteSaver` |
| 版本 | `langgraph-checkpoint-sqlite==3.1.2`（配 `langgraph-checkpoint` 4.3.0） |
| 文件 | `outputs/checkpoints/insightflow.sqlite`（V2 独立目录，已 gitignore） |
| thread_id | 与 `run_id` 相同 |
| 模式 | 默认 `WRITE`：每个 super-step 提交一次 |

**可靠性/性能取舍**：SQLite 单文件、零外部服务，适合单机；
每个 super-step 一次 fsync，因此写入延迟高于内存 Checkpointer。
换取的是"进程被强杀后仍能从最后提交点恢复"。
将来换 PostgreSQL 只需替换 `core/checkpoints.py` 的构造函数——
上层只依赖 `BaseCheckpointSaver` 接口。

**Checkpoint 中不含**：真实 API Key、模型客户端、resolver、DataFrame。
由 `tests/test_checkpoint_persistence.py` 以字节级断言锁定。

---

## 四、模型调用的可核对账本

P1 初版的文档声称"未确认的模型调用会被 `replay_guard` 标记为 uncertain 并拒绝自动重放"。
**这与代码不符**，已在补验收中修正：`llm_operation` 当时从未被调用，
`STATUS_UNCERTAIN` 也从未被赋值——真正接入守卫的只有 `artifact` 与 `chart` 两处。

根因是 **Checkpoint 的提交粒度**：它只在**节点执行结束**时提交。
进程若在"请求已发出、节点尚未返回"之间被杀，state 里不会留下任何痕迹，
因此仅靠 Checkpoint **无法发现**这种重复。这正是需要一个独立记录的原因。

修正后的机制：

| 组件 | 作用 | 落盘时机 |
|---|---|---|
| `utils/llm_ledger` | 追加式账本，记录每次调用的 `started` / `completed` | 请求**发出之前**写 `started` 并 fsync |
| `RunRuntime.unconfirmed_llm_calls` | 恢复时从账本比对出"已发出但未完成"的调用 | 每次 invoke 前重新计算 |
| `state.pending_llm_calls` | 把比对结果带进工作流 | 必须在 state 中**声明**，未声明的键会被 LangGraph 丢弃 |
| `guard_llm_calls` | 有未确认调用时 `interrupt`，默认不重发 | 节点执行 Agent 之前 |

### 三种崩溃时点的实测行为

| 场景 | 上游是否收到请求 | 恢复行为 | 重复请求 |
|---|---|---|---|
| 请求尚未发出时崩溃 | 否（plotter 计数 0） | 暂停，等人工决定 | **0** |
| 上游已返回、结果未持久化 | 是（plotter 计数 1） | 暂停，等人工决定 | **0** |
| 节点提交后崩溃 | 是（已登记） | 从 Checkpoint 继续 | **0** |

`started` 无法区分"上游已处理"与"上游没收到"，因此**不提供 exactly-once**。
账本只报告"未确认"，不推断是否已计费；人工批准重发是显式动作，
批准后确实会产生第二次请求（测试中 plotter 计数由 0 变为 1）。

---

## 五、幂等设计

Checkpoint 恢复是**至少一次**语义：节点执行完但提交前崩溃时，恢复会重放该节点。

| 副作用 | 幂等键 | 恢复时行为 |
|---|---|---|
| 工具执行 | `run:tool_execution:task:attempt_N` | 已完成则复用；结果按尝试序号去重 |
| 图表生成 | `run:plot_execution:chart_id` | 已完成则复用；写入按 chart_id 去重 |
| 报告产物 | `run:artifact` | 已完成则复用登记中的路径 |
| 审批 | `run:approval:plan_review` | 固定 ID，重复审批不会二次触发 |
| 模型调用 | `run:llm:node:subject` | 未确认则暂停等人工决定（见上） |

设计要点：

- 操作 ID 只由**稳定输入**构成（run、节点、ID），
  不含时间戳或随机数——否则恢复后算出的 ID 与之前不同，幂等失效；
- 图表刻意**不带** attempt：可视化结果按 chart_id 保留最后一条，
  "这张图是否已产出"与尝试次数无关；
- 终态（completed / failed）的 Run 再次 resume 时**直接返回持久化状态**，
  不再推进执行——否则每次 resume 都会重跑节点，让追加型字段持续增长。

### 文件系统与 Checkpoint 的一致性边界

**两者不属于同一个事务**，这一点必须明确，不能描述成跨资源原子提交。

可观测的窗口：文件已写入 → 进程被杀 → 节点的 state 增量从未提交
→ Checkpoint 里"没有做过" → 恢复时重新执行该节点。

实测结论（`tests/test_file_checkpoint_consistency.py`）：

| 情况 | 结果 |
|---|---|
| PNG 已写、登记未提交 | 恢复后重放，写回**同一路径**；只有 1 个 PNG，内容一致 |
| 报告三件套已写、state 未提交 | 恢复后重写同一路径；每个文件只有 1 份，产物清单无重复 |
| 反复 resume | 文件与 `execution_results` 都不再增长 |

**能保证**：产物路径稳定（同一 run 的同一产物始终是同一路径），
因此重放是"覆盖同一文件"而不是"产生副本"；产物清单与图表记录按 ID 去重。

**不能保证**：写文件与写 Checkpoint 之间的原子性。若进程在写文件中途被杀，
可能留下**半写的文件**（例如 HTML 写了一半）。P1 没有引入临时文件 + 原子替换，
因为当前产物都能被整体重新生成，且重放会覆盖它。
需要"读到的产物一定是完整的"时，应改用 `utils/result_artifacts.py` 的做法：
写临时文件 + 内容哈希 + 原子 rename。这一点列入已知限制。

---

## 六、SQL 只读是否真正接入工具链

P1 初版实现了 `utils/sql_guard.py` 并单独测试通过，但**工具链从未调用它**：
`tools/sql_tool.py` 只有 `query.strip().lower().startswith(("select", "with"))`，
而 `"/*x*/WITH t AS (SELECT 1) DELETE FROM orders"` 会通过前缀检查并真的执行删除。
连接也是 `sqlite3.connect(path)` 的**读写**模式。

修正后两层同时生效：

| 层 | 实现 | 作用 |
|---|---|---|
| 词法校验 | `ensure_read_only`（去注释/字面量 → 单语句 → 前缀 → 写关键字黑名单，fail-closed） | 拒绝非只读、多语句、注释绕过、PRAGMA / ATTACH 等 |
| 连接权限 | `sqlite3.connect(f"file:{path}?mode=ro", uri=True)` | 即使词法层被绕过，SQLite 也会拒绝写入 |

注意 `uri=True` 不可省：URI 字符串在缺省情况下会被当作普通文件名，
导致 `unable to open database file`。

`tests/test_sql_readonly_chain.py` 从真实入口
`tool_registry → sql_tool → sqlite3` 验证，并使用专用测试库，
断言 8 类写操作全部被拒且数据库内容哈希前后不变。
人工审批不能绕过：只读判定与审批记录无关。

---

## 七、State 序列化改造对比

### 改造前（P0）

| 字段 | 类型 | 问题 |
|---|---|---|
| `tables` | `dict[str, TableResource]` | 含 DataFrame，JSON 无法编码 |
| `model_factory` | `ModelFactory` | 运行期句柄，含 resolver 闭包 |
| `execution_results[].result` | `Any` | 可能是 DataFrame / ndarray / Timestamp |

实测：用真实 `SqliteSaver` 驱动工作流会在序列化 DataFrame 时失败。

### 改造后（P1）

| 原字段 | 新表示 | 恢复方式 |
|---|---|---|
| `tables` | `table_snapshot: dict[str, dict]` | `ResourceResolver` 按快照读回并校验内容指纹 |
| `model_factory` | 不在 state | ContextVar 注入，每次 invoke 重建 |
| `result` | 小结果内联 + 大结果 `result_artifact` 引用 | `load_result` 校验 sha256 后读取 |

新增可序列化字段：`run_status`、`pending_approval`、`plan_approval`、
`approvals`、`completed_operations`、`require_plan_approval`。

结果外置阈值：序列化后超过 32 KiB 落成
`<output_root>/<run_id>/results/<task_id>.json`，state 只留摘要与
`{path, sha256, size, row_count}`。

---

## 八、五个 tables 依赖节点的核查

| 节点 | 原写法 | 现写法 | 结果 |
|---|---|---|---|
| `loader_node` | `load_data_sources(...)` 写 `tables` | `build_snapshots` 写 `table_snapshot`，并试解析一次 | ✅ |
| `context_node` | `state["tables"]` | `resolve_tables(state)` | ✅ |
| `router_node` | `state["tables"]` | `resolve_tables(state)` | ✅ |
| `tool_execution_node` | `state["tables"]` | `resolve_tables(state)` | ✅ |
| `plot_execution_node` | `state["tables"]` | `resolve_tables(state)` | ✅ |

P0 文档只列了前四个，遗漏了 `plot_execution_node`——它同样把
`state["tables"]` 传给绘图工具。本阶段已改正，并由
场景 G（绘图阶段恢复）专门验证。

`state["tables"]` 在 `nodes.py` 中已无任何残留。

---

## 九、已知限制

1. **单进程调度**：执行队列在内存中，重启后 `running` 状态的 Run 需显式 resume。
2. **模型调用非 exactly-once**：见第四节。
3. **文件与 Checkpoint 非同一事务**：见第五节末。产物路径稳定、重放覆盖同一文件，
   但没有"临时文件 + 原子替换"，写文件中途被杀可能留下半写文件。
4. **SQL 只读仍是词法判定 + 只读连接**：`mode=ro` 能阻止写入，
   但 `utils/sql_guard.py` 的判定不是完整的 SQL 解析器；
   真正的隔离边界仍是 P2 的沙箱。
5. **工具仍同进程执行、无超时**（P2）。
6. **Checkpoint 文件随 Run 数增长**，尚无清理/归档策略。
7. **API 仅监听本机**：P2 的鉴权未实现前不对公网开放。
8. `data_sources` 允许任意本机路径（P1 语义），边界在 P2。
9. **`running` 的判定是启发式**：用"持有者进程是否还在 + 心跳是否过期"
   判断，只用于可观测性，不触发自动重启。心跳间隔为
   `STALE_HEARTBEAT_SECONDS / 6`。

**明确不宣称**：本阶段的并发与恢复验证都在单机、Mock 模型下完成，
不代表生产能力；也**不把测试期的资源限制称为完整安全沙箱**。
**Python 生成代码仍然没有可靠的安全隔离，禁止把系统暴露给不可信用户执行任意代码。**

---

## 十、结论对应的测试

上面每一条声明都可以在测试里找到对应断言，不存在"文档写了、代码没做"的情况。

| 声明 | 测试文件 |
|---|---|
| 跨进程恢复（新进程重启） | `test_cross_process_recovery.py`（10） |
| Checkpoint 可持久化、可重开 | `test_checkpoint_persistence.py`（4） |
| State 序列化契约 | `test_state_persistence_baseline.py`（7） |
| 人工审批与 interrupt/resume | `test_human_in_the_loop.py`（20） |
| Ledger started/completed/未确认 | `test_llm_crash_recovery.py`（15） |
| 文件与 Checkpoint 一致性边界 | `test_file_checkpoint_consistency.py`（8） |
| RunStore / 心跳 / 并发 resume | `test_run_lifecycle_consistency.py`（12） |
| SQL 只读真实链路 | `test_sql_readonly_chain.py`（32） |
| Run API 与输入校验 | `test_run_api.py`（24） |
| 表快照与 resolver | `test_resource_resolver.py`（17） |
| P0 行为无退化 | `test_workflow_mock.py`、`test_agents.py`、`test_conditions.py` 等 |
| Mock 模型 Token 不代表真实用量 | `test_modelgate_success.py`（7） |

### 关于 `docs/evidence/modelgate-success.json`

它是**运行取证快照**，每次执行 ModelGate 成功链路测试都会重写
（其中的 `run_id` 是随机的、`avg_latency_ms` 随机器负载变化）。
仓库里保留的是某个已通过版本的快照，因此跑完全量测试后该文件会显示为已修改——
这是预期行为，不是缺陷。需要刷新时：

```powershell
docker compose -p modelgate-docker-test `
  -f <modelgate>/docker-compose.test.yml `
  -f tests/docker/docker-compose.modelgate-success.yml up -d
.\.venv\Scripts\python.exe -m pytest tests/test_modelgate_success.py -q -s
```
