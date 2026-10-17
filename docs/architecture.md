# InsightFlow Architecture

## 分层

```
浏览器 / 脚本
   │  HTTP（FastAPI）
   ▼
api.py / api_runs.py / api_conversations.py     传输层：校验输入、映射状态码
   │
   ▼
services/                                      编排层：RunCoordinator、会话、记忆、知识
   │
   ▼
workflow.py + nodes.py + conditions.py         LangGraph：拓扑、路由、Checkpoint、HITL
   │
   ├── harness/          执行治理：模型调用、工具派发、预算、上下文选择与度量
   ├── planning/         计划校验、初始重规划、运行期计划修订
   ├── agent_loop/       运行期观察、动作决策、策略校验
   ├── rag/              知识加载、切分、检索、证据、缓存
   ├── storage/          会话、记忆、知识索引、数据库迁移
   ├── tools/ + mcp_bridge/ + sandbox/   实际执行（只读 SQL、容器化 Python/绘图）
   └── utils/            账本、幂等、预算计数、重试、日志、产物
```

职责边界是硬性的：LangGraph 只负责"下一步走哪个节点"；模型与工具的每一次真实
调用都经过 `harness/`，因此预算、账本与幂等只有一处实现；Agent 节点不写 SQL、
不碰 Redis。

## 一次分析请求的路径

```
POST /runs  或  POST /sessions/{id}/messages
  → loader          解析数据源，写可序列化的表快照
  → context         生成数据上下文（schema / 行数 / 样本）
  → knowledge       规划前的全局业务知识检索
  → planner         生成计划 → H3 校验 → 有界重规划 → 必要时人工审批
  → router          按来源类型选 SQL / Python（含资源授权校验）
  → execution       任务级检索 → 生成代码（HARNESS.model_call）
  → tool_execution  HARNESS.run_tool（预算、幂等、只读边界）
  → critic          结果是否回答了任务
  → decide          运行期观察 → 动作决策 → 策略校验（开关控制）
  → advance         下一个任务
  → visualization_planner → plot_generation → plot_execution
  → report          生成报告，附来源附录
  → artifact        写文件、登记产物
```

失败与循环都是**有界**的：

| 循环 | 上限 | 位置 |
|---|---|---|
| 初始计划重写 | 3 次 | `nodes.MAX_PLAN_ATTEMPTS` |
| 执行修复 | 3 次 | `conditions.MAX_RETRIES` |
| 质量精修 | 3 次 | `conditions.MAX_REFINES` |
| 绘图修复 | 3 次 | `conditions.MAX_PLOT_RETRIES` |
| 动态决策 | 8 步 / 重规划 2 次 / 补充检索 2 次 / 重复动作 2 次 | `agent_loop.policy.DEFAULT_LIMITS` |
| 模型请求、工具调用、轮次、执行时间 | 预算总量 | `utils.agent_loop.AgentBudget` |

## 状态与持久化

`state.InsightFlowState` 是 Checkpoint 的契约：所有值必须可 JSON 编码。数据行不
在 state 里（按 `table_snapshot` 读回），模型客户端由运行期注入。

两种存储模式：

| | `local` | `postgres` |
|---|---|---|
| Run 登记 | SQLite | PostgreSQL `runs` |
| Checkpoint | `SqliteSaver` | `PostgresSaver` |
| 知识索引 | 进程内 | pgvector + hnsw |
| 会话 / 记忆 | SQLite | PostgreSQL |

Redis 只缓存查询向量与检索结果，不是任何状态的权威来源。

## 人工中断

审批是显式开关，中断点有两个：

- `approval_gate`：计划执行前。记录 `plan_fingerprint` 与 `plan_revision`；
  运行期修订是新的计划版本，必须重新审批。
- `report_publish_gate`：报告落盘前。拒绝则不产出交付物。

中断期间没有任何副作用，恢复从被中断的节点继续；等待时间不计入执行预算。

## 目录

```
agent_loop/   observation.py  policy.py  decision.py  controller.py
agents/       planner executor critic repair refine visualizer plot_repair reporter
api*.py       api.py api_runs.py api_conversations.py
core/         runtime.py llm.py inference.py checkpoints.py config.py
harness/      runtime.py llm_gateway.py tool_dispatcher.py policy.py context_manager.py
planning/     validator.py orchestrator.py replan.py
rag/          document_loader.py chunker.py retrieval.py evidence.py versioning.py cache.py
services/     analysis_service.py run_coordinator.py run_store*.py conversation_service.py
              memory_service.py knowledge_service.py storage_factory.py
storage/      config.py models.py knowledge_models.py engine.py conversations.py
              memories.py knowledge.py
deploy/       compose.yaml（PostgreSQL + pgvector、Redis，仅绑定回环地址）
```
