# 受控的应用层自我改进

这是一版可运行的提示/经验优化闭环：真实任务评测、重复运行基线、开发反馈、候选搜索、独立验收、版本登记和回退。候选只能改变 Planner/Critic 的附加指引与结构化经验，不改评分器、基础提示中的安全规则、权限、预算或应用源码。它不训练模型参数，也不表示已证明递归改进能力持续提升。

## 执行与验收分开

- **候选侧**：`CandidateProfile` 只接受 `planner_guidance`、`critic_guidance`、`lessons`。经验含 agent、text、source_trials；没有 `verified=true`、评分器路径或可执行回调。
- **任务侧**：Planner/Critic 将附加内容作为模板输入值读取，经过现有上下文预算。Repair/Refine、模型调用、工具权限和执行预算继续由既有 Harness/agent_loop 管理。
- **评分侧**：宿主使用独立 CSV/Decimal oracle 校验真实答案，同时检查运行完成、图表约束和预算。Critic 的“通过”不决定评测通过。
- **验收侧**：同一实验内固定评分源码、数据、模型与预算指纹；候选选择依据验证集。锁定候选后才消费最终保留集，失败不在同一实验中继续试保留集。

这里的隔离建立在候选是纯文本数据、改进器没有宿主文件工具的前提上。评分器、版本库 API 是可信本地控制代码，不注册为模型工具或公开 HTTP 路由。本版没有执行模型产生的任意 Python/源码补丁，也不提供任意插件的操作系统级隔离。

## 当前任务范围

`eval/real_tasks.py` 为 development、validation、holdout 各生成两组不重合的客户/订单数据。任务是“累计订单金额严格大于 400 的客户，按地区汇总销售贡献”，覆盖边界值、分笔订单、负向调整和小数。最终答案由 `eval/business_oracles.py` 在宿主独立计算，不传给候选。

这是**一个业务任务的六组同领域合成数据变体**，用于跑通闭环，不等同于原模拟评测中的 42 项任务，更不证明跨领域泛化。默认每组重复三次，所有异常和失败都进入分母；报告同时包含完成率、业务正确率、每题多次全通过率、调用数和耗时。Token/美元成本拿不到时为 `null`，不会用调用数假装费用。

每次试验使用新的 run_id、输入目录、输出目录，没有共享 checkpoint、会话或任务记忆。输入文件在执行前后校验；评分源码与函数身份、预算和模型配置也会复核。输入或评分合同漂移会中断实验。模型名是服务商别名时，上游权重的实际变化不能仅靠配置指纹检测。

## 准备

在项目根目录运行；本机 Python 与依赖位于 D 盘。先启动 Docker Desktop，并确保项目的分析沙箱镜像已经构建。知识检索首次运行仍可能需要下载嵌入模型。

GEPA 是可选的正式 SDK，固定版本位于 `requirements-improvement.txt`。本次开发环境已安装 `gepa==0.1.4`。重建环境时可以使用已有 uv：

```powershell
D:\DevTools\bin\uv.exe pip install --python .venv\Scripts\python.exe -r requirements-improvement.txt
```

真实运行显式使用评测专用配置，不会自动采用普通应用密钥：

```powershell
$env:INSIGHTFLOW_EVAL_REAL_API_KEY = '<评测专用凭据>'
$env:INSIGHTFLOW_EVAL_REAL_MODEL = '<网关登记的模型名>'
$env:INSIGHTFLOW_EVAL_REAL_BASE_URL = 'http://127.0.0.1:8000/v1'
```

请求额度按模型调用次数限制，不是美元硬预算；实际金额仍应由 ModelGate 或服务商配额治理。不要把真实凭据写进候选配置、报告或命令示例文件。

## 先建立基线

已有评测命令现已接通真实运行，支持候选 JSON、重复次数和各任务预算：

```powershell
.venv\Scripts\python.exe scripts\run_eval.py --provider real --split development --repeats 3 --max-llm-calls 20 --max-seconds 300 --output outputs\real-eval
```

改进控制器的 baseline 同时跑开发集与验证集，保留集不用于常规基线浏览：

```powershell
.venv\Scripts\python.exe scripts\run_improvement.py baseline --repeats 3 --max-llm-calls 20 --max-seconds 300 --max-total-llm-calls 1000
```

`--max-total-llm-calls` 在派发每个评测批次之前，按任务数 × 重复数 × 单任务请求上限保守预留。提案与 GEPA 反思请求也先扣额度；真实请求可能远少于预留额度。额度不足会在新批次开始前失败，当前版本保持不变。时间限制复用现有运行预算和超时，不是一个可强制杀死任意宿主代码的通用执行器。

## 两种搜索后端

原生反思后端根据开发失败诊断提出完整候选 JSON；候选可携带有来源的经验条目。验证较优的版本参与下一轮诊断，既有指引与经验随版本传递：

```powershell
.venv\Scripts\python.exe scripts\run_improvement.py improve --backend reflective --rounds 1 --repeats 3 --max-llm-calls 20 --max-seconds 300 --max-total-llm-calls 1000
```

GEPA 后端使用官方 `gepa.optimize()`，通过开发轨迹反馈搜索 Planner/Critic 附加指引。评分器仍在外部，GEPA 的最佳候选必须再次通过相同的独立晋升规则：

```powershell
.venv\Scripts\python.exe scripts\run_improvement.py improve --backend gepa --max-metric-calls 8 --repeats 3 --max-llm-calls 20 --max-seconds 300 --max-total-llm-calls 2500
```

首版 GEPA 以每个小套件的聚合业务正确率为一个优化样本，不宣称已做到逐任务 Pareto 优化。`max-metric-calls` 约束套件评测调用，不是实际模型请求或美元数。GEPA 不修改经验手册；可以在后续 reflective 实验中积累经验，再独立验收。SDK 不存在时会明确报错，不用自制算法替代。

## 固定的晋升条件

1. 评分器、模型、预算、同一 split 的数据指纹一致，逐题重复试验覆盖完整。
2. 验证集逐题通过次数不下降，异常次数不增加。
3. 业务通过数严格增加；或业务通过数不变而总模型调用数至少减少一次。
4. 模型调用数不得超过基线的 1.25 倍。候选无法修改此策略。
5. 锁定后，最终保留集不出现逐题退化，仍符合成本约束。

候选反馈、注册的原始配置或伪造的 summary 都不是升级证明。版本库从逐条记录重新计算指标，要求基线/候选的验证与最终验收凭据齐全后，才在事务中切换 active 指针。并发实验基于旧 active 时会拒绝过期晋升。

新的经验必须引用可观测的开发试验，只有候选整体通过验收后才随版本启用。这证明的是**包含该经验的候选整体达标**，不是每一条经验的独立因果增益。回退会同时恢复旧提示和旧经验。每个命令行改进实验由宿主生成新的 holdout seed，降低重复验收同一组数据的风险；套件范围仍然很小。

## 使用、查看与回退版本

默认工作目录是 `outputs.improvement_private/`，包括 `registry/versions.sqlite3`、基线、逐次评测与 campaign 结果。该目录已被 Git 忽略，不随代码发布。

```powershell
.venv\Scripts\python.exe scripts\run_improvement.py status
.venv\Scripts\python.exe scripts\run_improvement.py export --output outputs\active-profile.json
.venv\Scripts\python.exe scripts\run_improvement.py rollback --target '<历史已验收的profile_id>'
```

命令行晋升仅更新显式版本库。要让现有 API 的**新 Run** 使用它，在启动 API 的终端配置可信路径：

```powershell
$env:INSIGHTFLOW_IMPROVEMENT_REGISTRY = (Join-Path (Get-Location) 'outputs.improvement_private\registry')
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8123
```

未配置时保持原有行为。调用 `run_analysis(improvement_profile=...)` 可显式选择实验版本，显式空配置用于基线。已运行或已中断的 Run 保存完整配置快照，恢复时沿用原版本；中途传不同配置会被拒绝。候选提示全文不通过状态接口公开。

## 本次验证与证据范围

开发验证使用脚本模型和注入运行器，不调用真实付费模型。新测试覆盖：放宽 Critic 不改变业务判分、恶意配置字段、花括号注入、恢复版本不漂移、输入/评分合同漂移、异常进入分母、错误报告拒绝、预算预扣、验证集过拟合、未知经验来源、版本晋升与回退，以及正式 GEPA SDK 的一次免费确定性搜索。

这些测试验证工程闭环与边界。真实模型基线、候选收益、金额成本和跨领域泛化仍需通过显式授权的真实评测记录确认。源码自修改保持关闭。

验收记录（2026-10-24）：148 项相关测试全部通过，运行耗时 48.38 秒；语法解析和本次跟踪文件的 diff 检查通过。正式 GEPA 0.1.4 已安装，并用确定性免费回调验证实际搜索流程。未调用真实付费模型。

复核组合包括 test_improvement_cycle、test_improvement_profiles、test_gepa_adapter、test_gepa_sdk、test_real_evaluation，以及既有 checkpoint、business_answers、harness、context_budget、workflow_mock 回归。Windows 受限环境下测试前应将进程 TEMP/TMP 指向项目内可写目录，并为 pytest 指定独立 --basetemp。
