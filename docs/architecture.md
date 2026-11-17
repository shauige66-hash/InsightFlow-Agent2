# InsightFlow-Agent2 Architecture

本文面向要改代码的工程师：先给分层与外部依赖，再给一次分析请求在图里的真实路径，
最后是有界循环、状态与持久化、人工中断。业务流程视角（谁用、解决什么问题）见
[overview.md](overview.md) 与 [demo/README.md](demo/README.md)。

## 分层

```
浏览器（零构建控制台 web/index.html）/ 脚本 / 飞书群机器人（可选，单向推送）
   │  HTTP（FastAPI）
   ▼
API 层          api.py · api_runs.py · api_conversations.py · core/auth.py
                输入校验、可选 Bearer→owner 鉴权、签名产物链接、状态码映射
   │
   ▼
Services 层     services/
                RunCoordinator（Run 生命周期）、分析、会话、记忆、知识、
                run_queue（PostgreSQL 队列模式）
   │
   ▼
LangGraph 工作流  workflow.py + nodes.py + conditions.py
                拓扑、路由、Checkpoint、HITL 中断、budget_stop 安全收尾
   │
   ├── Agent 层        agents/    Planner、Executor、Critic、Repair、Refine、
   │                              Visualizer、Plotter、PlotRepair、Reporter
   ├── Harness         harness/   模型调用与工具派发的唯一入口：
   │                              预算、账本、幂等、上下文选择与度量
   ├── 计划            planning/  计划校验、初始重规划、运行期计划修订、
   │                              可视化计划逐张校验（partition_visualization_plan）
   ├── 动态决策        agent_loop/  观察构造、动作决策、确定性策略校验（开关控制）
   ├── RAG             rag/       知识加载、切分、检索、证据、版本化、缓存
   ├── Tools/Sandbox/MCP  tools/ + sandbox/ + mcp_bridge/
   │                              容器化 Python/绘图（一次性容器、只读根、无网络）、
   │                              只读 SQL（direct 宿主进程 或 MCP stdio 子进程，
   │                              两种后端共用同一套防护）
   ├── Storage         storage/   会话、长期记忆、知识索引、数据库迁移
   │                              （local=SQLite / postgres=PostgreSQL+pgvector）
   └── utils/          账本、幂等、预算计数、重试、日志、产物
                       （HTML 白名单净化 + CSP、PDF 导出）

外部基础设施（除模型网关外均可选）：
  模型网关（DeepSeek 兼容 OpenAI 接口，经 core/llm.py 工厂构建）
  Docker Engine（Python/绘图沙箱容器，镜像 insightflow-sandbox:py311）
  PostgreSQL + pgvector、Redis（postgres 模式；Redis 只做检索缓存）
  浏览器（Playwright 驱动 PDF 导出；缺失时降级为 md+html，状态 degraded）
  飞书群机器人 webhook（状态卡片推送，未配置即整链路关闭）
```

职责边界是硬性的：LangGraph 只负责"下一步走哪个节点"；模型与工具的每一次真实
调用都经过 `harness/`，因此预算、账本与幂等只有一处实现；Agent 节点不写 SQL、
不碰 Redis。**Agent 层负责"想"（规划、生成、评审），Harness 负责"管"（每一次
调用怎么被治理）**——这是两个不同的关注点，不要混在一起描述。

RAG 与长期记忆也是两套东西：RAG（`rag/`）检索**本次 Run 传入的业务知识文档**，
产出带来源的证据；长期记忆（`storage/memories.py` + `services/memory_service.py`）
存**跨会话、经用户确认**的偏好与口径，写入默认是候选，确认后才注入上下文。

## 一次分析请求的路径

入口是 `POST /runs`（或会话消息 `POST /sessions/{id}/messages`），由
RunCoordinator 派发（inline 线程或 PostgreSQL 队列 + worker）：

```
loader              解析数据源（CSV/Excel/SQLite），写可序列化的表快照
context             生成数据上下文（schema / 行数 / 样本，按任务裁剪）
knowledge_retrieval 规划前的全局业务知识检索
planner             生成结构化计划 → 程序校验 → 有界重规划（≤3 次）
[approval_gate]     可选：计划审批中断（计划指纹 + 版本号；拒绝则终止）
router              按来源类型选 SQL / Python（含资源授权校验）
execution           任务级检索 → 生成执行代码（HARNESS.model_call）
tool_execution      HARNESS.run_tool（预算、幂等、只读边界、容器沙箱）
   ├─ 执行错误 → repair（≤3）→ tool_execution；仍失败 → failed_task 记录失败
   └─ 成功 → critic：结果是否回答了任务；质量问题 → refine（≤3）
[decide]            动态决策开启时：观察 → 动作决策 → 确定性策略校验
                    （continue/finish/repair/refine/retrieve_more/
                      replan_remaining/skip_task，各有硬上限）
[runtime_replan]    动态决策触发：只重写未执行任务，过与初始计划相同的校验；
                    开启计划审批时修订是新版计划（:r<n>），需重新审批
advance_task        推进到下一个任务（回到 router）
visualization_planner  图表规划：逐张校验（来源任务成功、类型 ∈ line/bar/scatter、
                    目标表匹配、≤3 张），不合法的拒绝并携带原因重规划（≤3 轮）；
                    部分合法即接受（状态 degraded），全部被拒记入 failed_charts
plot_generation → plot_execution → plot_repair（每张 ≤3）→ advance_chart
report              生成报告：正文 + 知识来源引用附录 + 图表附录 +
                    图表降级/失败说明（由证据与状态生成，不是模型复述）
[report_publish_gate] 可选：发布审批中断（拒绝则不产出交付物）
artifact            写 md / html / pdf（PDF 缺浏览器时降级），登记产物
budget_stop         任一环节预算耗尽 → 安全收尾，仍走发布闸与 artifact
```

固定工作流与动态决策的关系：**图结构不变**，`enable_dynamic_loop`（API 默认
`false`，控制台默认勾选）只决定 `tool_execution`/`critic` 之后是否经过 `decide`
的观察—决策—策略校验；关闭时走原有的有界修复/精修分支，行为与引入前一致。

用户交互侧（业务步骤的最后一环）：`GET /runs/{run_id}` 轮询状态、
`GET /runs/{run_id}/report.html` 看报告、`POST /runs/{run_id}/approval` 审批、
会话接口追问、飞书卡片通知（可选）。

失败与循环都是**有界**的：

| 循环 | 上限 | 位置 |
|---|---|---|
| 初始计划重写 | 3 次 | `nodes.MAX_PLAN_ATTEMPTS` |
| 执行修复 | 3 次 | `conditions.MAX_RETRIES` |
| 质量精修 | 3 次 | `conditions.MAX_REFINES` |
| 绘图修复 | 3 次/张 | `conditions.MAX_PLOT_RETRIES` |
| 可视化计划重规划 | 3 轮 | `nodes.py` visualization_planner（`range(3)`） |
| 动态决策 | 8 步 / 重规划 2 次 / 补充检索 2 次 / 重复动作 2 次 | `agent_loop.policy.DEFAULT_LIMITS` |
| 模型请求、工具调用、轮次、执行时间 | 预算总量 | `utils.agent_loop.AgentBudget`，耗尽进 `budget_stop` |

Checkpoint、预算、幂等是**内部治理机制**，不是用户直接操作的功能：用户能感知的
只有审批开关、运行状态、报告与（可选的）通知。

## 状态与持久化

`state.InsightFlowState` 是 Checkpoint 的契约：所有值必须可 JSON 编码。数据行不
在 state 里（按 `table_snapshot` 读回），模型客户端由运行期注入。LangGraph 会丢弃
未声明的键，因此 `failed_charts` 等诊断字段都在 `state.py` 显式声明。
`improvement_profile` / `improvement_profile_id`（见下节）同样在 state 里声明：
profile 快照随 Checkpoint 走，恢复的 Run 沿用原版本、中途换版本会被拒绝；对外
状态接口只公开 `improvement_profile_id`，不公开候选提示全文。

两种存储模式：

| | `local` | `postgres` |
|---|---|---|
| Run 登记 | SQLite | PostgreSQL `runs`（含 owner 列） |
| Checkpoint | `SqliteSaver` | `PostgresSaver` |
| 知识索引 | 进程内 | pgvector + hnsw |
| 会话 / 记忆 | SQLite | PostgreSQL |
| 队列执行 | 不支持（inline） | `run_jobs` + `FOR UPDATE SKIP LOCKED` |

Redis 只缓存查询向量与检索结果，不是任何状态的权威来源。
SQLite 与 PostgreSQL 的 Checkpoint 不互相迁移，切换模式后旧 Run 用原模式读取。

## 人工中断与产物访问

审批是显式开关，中断点有两个：

- `approval_gate`：计划执行前。记录 `plan_fingerprint` 与 `plan_revision`；
  运行期修订是新的计划版本，必须重新审批。
- `report_publish_gate`：报告落盘前。拒绝则不产出交付物。

中断期间没有任何副作用，恢复从被中断的节点继续；等待时间不计入执行预算。

开启鉴权后，报告与图表不以随机 run_id 代替授权：`GET /runs/{id}/artifact-link`
签发 5 分钟有效、绑定 Run 与 owner 的地址（`core/auth.py` HMAC-SHA256），首次访问
设置仅限该 Run 路径的 HttpOnly cookie 供正文图片加载。报告 HTML 经白名单净化
（`utils/html_safety.py`）并带 `default-src 'none'` CSP；PDF 渲染禁用 JavaScript
且拦截非 `file:`/`data:`/`about:` 请求。

## 受控自我改进（离线，可选）

`improvement/` + `eval/` 是一条**离线**闭环，不在分析请求路径上：真实重复评测 →
提示/经验候选搜索 → 独立验收 → 版本晋升与回退。方法与证据边界见
[improvement.md](improvement.md)，此处只记架构挂点：

- **候选是纯文本数据**：`CandidateProfile` 只含 `planner_guidance`、
  `critic_guidance`、`lessons`；不改评分器、安全规则、权限、预算或应用源码，
  不训练模型参数，不执行模型产生的代码。
- **运行期挂点只有两处**：`agents/planner.py` 与 `agents/critic.py` 把
  `improvement.profiles.current_guidance(node)` 作为模板输入值附加进提示（走既有
  上下文预算）；`services/analysis_service.py` 在**新 Run** 启动时解析一次 profile
  （显式传入 > 可信本地注册表 `INSIGHTFLOW_IMPROVEMENT_REGISTRY` > 空基线；
  这是宿主配置，绝不来自请求体或模型输出），已存在的 Checkpoint 沿用快照版本。
- **搜索后端两种**：内置 reflective（按开发集失败诊断生成完整候选 JSON）与官方
  GEPA SDK（`improvement/gepa_adapter.py`，`gepa==0.1.4`，可选依赖
  `requirements-improvement.txt`；SDK 缺失时明确报错，不用自制算法顶替）。
- **验收与晋升在宿主侧**：`eval/` 用独立 oracle 计算业务真值（Critic 的"通过"
  不算数）；`improvement/gates.py` 是候选无法修改的固定晋升条件；
  `improvement/registry.py` 用 `versions.sqlite3` 从逐条记录重算指标、事务内切换
  active 指针，支持回退。工作目录 `outputs.improvement_private/` 已被 Git 忽略。
- **入口**：`scripts/run_improvement.py`（baseline / improve / status / export /
  rollback）与 `scripts/run_eval.py --provider real`（评测专用凭据
  `INSIGHTFLOW_EVAL_REAL_*`，不与普通应用密钥混用）。

## 目录

```
agent_loop/   observation.py  policy.py  decision.py  controller.py
agents/       planner executor critic repair refine visualizer plotter plot_repair reporter
api*.py       api.py api_runs.py api_conversations.py
core/         runtime.py llm.py inference.py checkpoints.py config.py auth.py resource_access.py
harness/      runtime.py llm_gateway.py tool_dispatcher.py policy.py context_manager.py
planning/     validator.py orchestrator.py replan.py visualization.py
rag/          document_loader.py chunker.py retrieval.py evidence.py versioning.py cache.py
services/     analysis_service.py run_coordinator.py run_store*.py run_queue.py
              conversation_service.py memory_service.py knowledge_service.py
              run_notifier.py storage_factory.py
storage/      config.py models.py knowledge_models.py engine.py conversations.py
              memories.py knowledge.py
improvement/  cycle.py gates.py gepa_adapter.py profiles.py registry.py（离线自我改进）
eval/         real_runner.py real_tasks.py business_oracles.py（真实评测与独立评分）
integrations/ feishu.py（群机器人卡片，可选）
utils/        html_safety.py artifacts.py llm_ledger.py idempotency.py agent_loop.py …
web/          index.html（零构建控制台）
scripts/      serve_api.py run_worker.py check_model.py real_model_smoke.py
              real_model_regression.py prepare_retail_case.py run_eval.py
              run_improvement.py start_deploy.ps1 …
figures/      流程图（.mmd/.png/.svg + 说明 .md）
deploy/       compose.yaml（本地依赖）compose.app.yaml（整套编排，仅绑定回环地址）
```
