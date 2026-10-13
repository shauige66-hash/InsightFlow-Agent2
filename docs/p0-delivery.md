# P0 交付说明与 P1 前置分析

本文记录 P0 阶段做了什么、哪些指标可度量、以及 P1 引入 Checkpoint 前必须处理的状态问题。

---

## 一、P0 的目标与实际结果

P0 不新增业务功能，只解决四件事：

1. **测试基线** —— 原先 `tests/` 下 14 个文件没有一个 `def test_`，是导入即执行的打印脚本，
   其中 `test_tool_calling.py` 在导入阶段就调用真实模型。现在 `tests/` 只放真测试，
   手工脚本移到 `scripts/manual/`。
2. **工程隔离** —— V2 使用独立虚拟环境、独立输出目录；ModelGate 集成只打独立测试栈。
3. **LLM 解耦与追踪** —— Agent 不再各自构造客户端，改为按 run 注入，携带 run_id 与 agent_name。
4. **为 P1 铺路** —— 明确 state 中不能进 Checkpoint 的部分。

---

## 二、可度量的量（以及不可度量的量）

### 可度量

| 指标 | 含义 |
|---|---|
| LLM 调用次数（按 Agent） | 每个 Agent 实际发起的模型请求数 |
| 工具调用次数 | `execution_results` 条数 |
| 任务成功 / 失败数 | `analysis_plan` 与 `failed_tasks` |
| 结果语义 | `analysis_status`（completed / partial / no_tasks） |
| 端到端耗时 | 单次运行墙钟时间 |
| 产物 | 报告 Markdown / HTML / PDF 是否生成 |

### 这份基线的身份：P0 V2 Mock 基线，不是 V1 实测数据

`docs/baseline.json` 由 `scripts/run_baseline.py` 采集，固定三个场景：
单表聚合、多表关联、修复循环。

采集时的状态（已写入 JSON 的 `provenance` 字段）：

| 项 | 值 |
|---|---|
| commit | `64ccd6670edaff4c22826dca134cd462f61f6eb2`（V1 的 HEAD） |
| 工作树 | **有未提交改动（51 个文件）** |
| Mock 行为 | 每个 Agent 由固定脚本响应，不调用任何模型服务，不返回 usage |
| 输入 | `data/customers.csv`、`data/orders.csv`；无 knowledge sources |

**为什么不能称它为 V1 的性能数据**：采集发生在 P0 修改后的工作树上——
Agent 已支持注入模型、`nodes` 已改为显式注入、`workflow` 增加了空计划分支。
虽然被 Mock 固定住的三个场景（有心跳、有任务、无失败）在控制流上与原 V1 等价，
但这仍是"改后的代码 + Mock 模型"，不是原始 V1 在真实模型下的表现。

**因此本报告不宣称任何 V1 与 V2 的性能对比。** 这份数据的用途是
P0 与后续版本之间的**同口径回归基线**：同样的 Mock 脚本、同样的数据集，
用于发现控制流层面的退化（调用次数、循环次数、失败记录）。

耗时读数受机器负载影响明显——不同批次实测在 1.4s~2.1s 之间波动，
其中 PDF 生成（Playwright 启动浏览器）占绝大部分且与模型无关。
**只适合同一批次内的横向比较。**

采集过程未修改、也未读取原始 V1 工作目录。

### 不可度量（不得编造）

- **token 数与成本**：Mock 模型不返回 `usage`，因此记为 `unavailable` 而不是 0。
  真实数值需要接入真实模型或让 Mock 返回 usage，属于 P5 的评测范围。
- **真实模型上的任务成功率**：Mock 只验证控制流，不验证模型能力。
- **生产吞吐**：本轮没有做性能测试，也没有可对照的环境。

---

## 三、P1 的阻塞点：State 不能直接持久化

### 3.1 现状

`InsightFlowState` 里有两类值无法 JSON 序列化：

| 字段 | 问题 | 影响 |
|---|---|---|
| `tables: dict[str, TableResource]` | `TableResource.dataframe` 是 pandas DataFrame | Checkpoint 无法落盘 |
| `model_factory: ModelFactory` | 持有客户端缓存与 resolver | 同上；且跨进程恢复无意义 |
| `execution_results[].result` | 任意对象（可能是 DataFrame / ndarray / Timestamp） | 同上 |

`tests/test_state_contract.py` 用可执行断言锁定了这三点，P1 改造时这些测试会直接提示需要同步更新。

### 3.2 依赖 `state["tables"]` 的节点

| 节点 | 用途 |
|---|---|
| `loader_node` | 写入 `tables` |
| `context_node` | 遍历所有表构建 `data_context` |
| `router_node` | 按表的 `dataframe` / `source_type` 决定工具 |
| `tool_execution_node` | 把 `tables` 交给工具执行 |

这四个节点是 P1 改动的全部影响面。

### 3.3 建议的最小改造

引入 `TableStore`：按 `source_path` 缓存 `TableResource`，state 里只保留
可序列化的 `table_snapshot`（`name` / `source_type` / `source_path` / `database_table`）。
节点通过 store 按需取回资源；恢复时从快照重建。

代价：`loader_node` 之后仍需要一次真实的加载；收益：state 变成可序列化。

`model_factory` 更容易处理：恢复时按 `run_id` 重新构造即可，不需要进状态。
P1 应把它从 state 移到 `config["configurable"]`，这样也不会出现在 Checkpoint 里。

### 3.4 需要幂等保护的节点

| 节点 | 副作用 | 幂等键建议 |
|---|---|---|
| `artifact_node` | 写 3 个报告文件（含调用 Playwright 生成 PDF） | `(run_id, "artifact")` |
| `plot_execution_node` | 写 PNG | `(run_id, chart_id)` |
| `tool_execution_node` | 执行生成的 Python / SQL | `(run_id, task_id, retry_count)` |
| `plot_generation_node` / `plot_repair_node` | 额外的模型调用（有成本） | `(run_id, chart_id, attempt)` |

恢复时若不校验这些键，会出现重复写文件、重复调用模型（重复计费）、
以及 `execution_results` 重复追加。

注意 `execution_results` 目前是"追加"语义，且同一 `task_id` 可能出现多条
（首次失败 + 每次修复各一条），只在报告阶段由 `get_final_execution_results`
按 task_id 去重（last-wins）。恢复时若重放，需要区分"重放同一步"与"真正的新尝试"。

---

## 四、本轮修复的真实缺陷

每项都有失败→通过的测试证据。

| 缺陷 | 原行为 | 修复 | 证据 |
|---|---|---|---|
| 必填字段在重试范围外索引 | `result["code"]` 在 `invoke_json_with_retry` 之外，JSON 合法但缺字段时抛 `KeyError` 终止整条工作流 | 校验器移入重试内部，缺字段按可重试的格式问题处理 | `test_agents.py::TestExecutor::test_missing_code_field_is_retried` |
| 重试重发完全相同的 prompt | temperature=0 下确定性失败会重复失败 3 次 | 第二次携带上次失败原因（`retry_hint`） | `test_agents.py::TestPlanner::test_retry_prompt_includes_previous_failure` |
| 空计划导致工作流崩溃 | Planner 返回空任务列表时，`router_node` 取 `analysis_plan[0]` 抛 `IndexError` | 新增 `route_after_planner`，空计划直接收尾 | `test_conditions.py::TestRouteAfterPlanner` + `test_workflow_mock.py::TestBoundedLoops::test_planner_rejecting_produces_failure_record` |
| 图表列表为 null 被当作有效计划 | `{"charts": None}` 通过键存在性检查，返回 `None` 而非列表 | 校验类型而非仅存在性 | `test_agents.py::TestVisualizer::test_missing_charts_field_is_retried` |
| 工具参数校验异常直接往上抛 | 未知表名等参数错误抛 `ValueError`，绕过修复循环 | 新增 `tools/tool_registry.py` 统一返回错误字典 | `test_tools.py::TestPythonTool::test_unknown_table_raises_from_tool_but_registry_catches` |
| 报告生成无重试 | 唯一交付物反而最脆弱 | 增加空内容重试 | `agents/reporter.py`（`invoke_report_with_retry`） |
| `REFINE_PROMPT` 第 8–12 条重复且拼接错位 | 第 146 行两段内容挤在一行 | 删除重复块 | 代码审查 |

### 结果语义（`analysis_status`）

`failed_tasks` 为空**不等于**分析成功——空计划时工作流正常走完却没有任何产出。
因此新增一个显式字段：

| 取值 | 含义 |
|---|---|
| `completed` | 有计划且全部任务成功 |
| `partial` | 有计划但有任务最终失败（`status_reason` 给出失败数） |
| `no_tasks` | 计划为空，未进行分析（`status_reason` 说明原因） |
| `failed` | 工作流异常终止（异常直接抛出，结果字典不存在） |

同时写入 JSONL 的 `analysis_finished` 事件，便于事后审计而非只看返回值。

未修复（记录在已知限制，不属 P0 范围）：

- Python / SQL 工具同进程执行、无超时无资源上限（P2 沙箱）
- SQL 只读仅靠字符串前缀判断，连接以读写模式打开（P2）
- `router.py` 对未知表名抛 `KeyError` 而非 `ValueError`（异常类型不统一）
- 工具结果无截断，会无界增长进 prompt（P4）

---

## 五、与 ModelGate 的协同

- **接入方式**：`ChatOpenAI(base_url=.../v1)`，不改客户端代码，只改配置。
- **追踪头**：`X-Client-App: insightflow-agent2`、`X-Agent-Name`、`X-Run-Id`
  写在 client 的 `default_headers` 上。LangChain 的 `metadata` 只进本地回调、
  不会成为 HTTP 头，因此无法被 ModelGate 看见——这是实现时需要注意的点。
- **客户端重试关闭**（`max_retries=0`）：否则一次逻辑调用会被放大成多次真实请求，
  与 ModelGate 统计、业务层 JSON 修复循环叠加后口径失真。

### 两条链路都已验证

| 链路 | 测试文件 | 上游 | 验证内容 |
|---|---|---|---|
| 失败 | `tests/test_modelgate_integration.py` | `.invalid` 保留域名 | 连接错误被归类、仍按 run/agent 记账、未登记模型与错误密钥**不记账** |
| 成功 | `tests/test_modelgate_success.py` | 本机 Mock Provider | HTTP 200、有效响应体、token 提取与累加、同 run 两个 Agent 分别归属、列表接口可见 |

两条链路各自拉起并拆除独立的 ModelGate 测试栈（项目名 `modelgate-docker-test`、
库 `modelgate_docker_test`、端口 18000），互不依赖，也不接触业务实例。
成功链路的上游通过 `tests/docker/docker-compose.modelgate-success.yml`
覆盖为 `host.docker.internal:9099`——只覆盖环境变量，不修改 ModelGate 的任何文件。

Mock Provider 是 Agent2 自带的 `tests/support/openai_mock_provider.py`，
返回固定 usage（12 / 8）。**该数值仅用于验证网关的提取与累加，不代表真实用量。**

成功链路的取证结果写入 `docs/evidence/modelgate-success.json`
（测试栈会被拆除，因此必须在栈存活时取证）。

### 三层重试的关系（统计口径）

| 层 | 触发条件 | 是否重试 | 影响 |
|---|---|---|---|
| 业务层 JSON 修复 | 输出无法解析或缺必填字段 | 最多 3 次 | **产生多次真实模型请求** |
| ChatOpenAI 客户端 | 网络 / HTTP 错误 | **关闭** | 无 |
| ModelGate 上游 | 由网关的 `RETRY_*` 配置决定 | 默认关闭 | 若开启会再次放大 |

因此"一次 Agent 调用"最多对应 3 次 ModelGate 调用记录。P5 做调用次数对比时
必须显式声明这个口径。

---

## 六、隔离边界

| 边界 | 做法 |
|---|---|
| Git | V2 保留自己的历史与 remote，不指向 V1 仓库 |
| 虚拟环境 | V2 独立 `.venv`（Python 3.11.9） |
| 输出目录 | `outputs/`、`uploads/` 在 `.gitignore` 中；测试写入临时目录 |
| ModelGate 业务实例 | **从不使用**。集成用例只打独立测试栈（18000 / `modelgate_docker_test`） |
| 真实模型 API | 单元测试由 `tests/network_guard.py` 在 socket 层拦截；集成用例的测试栈上游是 `.invalid` 保留域名 |
| 密钥 | 测试用固定假密钥；V2 的 `.env` 只放 ModelGate client key，不含真实上游 key |
