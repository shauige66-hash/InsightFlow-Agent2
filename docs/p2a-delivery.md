# P2-A 交付说明：容器化执行沙箱

本文记录 P2-A 的执行沙箱架构、权限边界、错误分类设计，以及**尚未解决的限制**。

本阶段目标：把 Python 分析与 Matplotlib 绘图的执行从宿主进程移出，放进一次性、
受限的 Docker 容器，同时不改变工作流拓扑与 P1 的恢复语义。

验收结果：`388 passed / 0 failed / 0 skipped`，其中沙箱用例 74 项
（47 项实际启动容器、23 项故障模拟、4 项静态断言）。

---

## 零、执行链路速查

```
LangGraph 节点
  └─ tools/tool_registry.invoke_tool(tool, tables, args)
       ├─ "python" ─┐
       ├─ "plot"   ─┼─► tools/*.py（参数装配 + 结果归一化）
       └─ "sql"    ─┘        │
                             ▼
                    sandbox/executor.DockerSandboxExecutor
                      · require_ready()：daemon + 镜像存在性
                      · 进程级并发信号量（Python/Plot 共用）
                      · 写 /in（job.json + code.py + data/*.json）
                      · docker run（列表参数，非 shell）
                      · 宿主 wall-time 超时 → kill + 清理
                      · 校验 staging 产物 → 发布到 Run 路径
                             │
                             ▼
               一次性 Linux 容器（--rm）
                 sandbox/run.py → bootstrap 造作用域 → exec 模型代码
                             │
                             ▼
                 /scratch/result.json（结构化结果）
                             │
                             ▼
       SandboxOutcome → {success,result,error,error_kind} → LangGraph State
```

`exec` 只存在于 `sandbox/run.py`；`tools/` 下不存在任何同进程执行模型代码的路径，
由静态用例锁定。

---

## 一、挂载与权限模型（本阶段的安全边界）

安全边界落在**挂载权限**上，不是"作用域里缺哪些内置函数"。

| 挂载 | 权限 | 内容 |
|---|---|---|
| `/sandbox` | 只读 | 执行入口与共享模块 |
| `/in` | 只读 | `job.json`、`code.py`、本次用到的表 |
| `/out` | **可写** | 仅该次操作的专属 `staging/` |
| `/history` | 只读 | Run 已发布产物（可读不可改） |
| `/scratch` | 可写 | 结果 JSON |
| `/tmp` | tmpfs | 库缓存，`nosuid,nodev,noexec` |

### 为什么必须这样做

首版把整份 Run 目录可写挂载为 `/out`。实测可以这样篡改同一 Run 的历史产物：

```python
df.to_csv('/out/analysis_report.md')   # 覆盖已有报告，实测成功
df.to_csv('/out/evil_new.csv')         # 凭空新建文件，实测成功
```

**作用域里没有 `open` / `__import__` 并不能阻止这一点**——pandas 与 matplotlib
自己的方法就会写文件。这是本阶段最重要的一条教训：判断能力边界要看实际权限，
不能看"模型代码里有哪些符号"。

### 宿主发布校验

容器写 staging 后，宿主在发布前逐项校验：

1. 文件名合法（不含路径分隔符，由宿主生成、模型无法指定）；
2. `is_symlink()` —— **先于读取内容**，否则会顺着链接读到挂载之外；
3. 是普通文件、非空、体积在上限内；
4. PNG 魔数 + IHDR 宽高合法；
5. 目标若已是符号链接，先 `unlink`（否则 `os.replace` 会顺着链接写到目录外）；
6. 通过后 `os.replace` 落到 `<output_root>/<run_id>/<chart_id>.png`。

任何一项不过就拒绝，**不发布未经验证的产物**。绘图正常时返回的仍是宿主侧
稳定路径，P1 的产物幂等、哈希检查与 Checkpoint 恢复行为不变。

---

## 二、容器启动参数

参数集中在 `executor._container_args()`，**模型执行与诊断探测共用同一份**——
若探测用自己的参数集，它验证的就不是模型代码实际所处的运行时。

```
docker run --rm
  --name iflow-sbx-<run8>-<op8>          # 宿主生成的短哈希名
  --label insightflow.sandbox=1          # 孤儿清理只认这个标签
  --network none
  --read-only
  --cap-drop ALL
  --security-opt no-new-privileges       # 保留 Docker 默认 seccomp
  --user 65534:65534
  --memory 1024m --memory-swap 1024m
  --cpus 1.0
  --pids-limit 128
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=256m
  <上述五个挂载>
  --env MPLCONFIGDIR=/tmp/mpl  XDG_CACHE_HOME=/tmp
        PYTHONDONTWRITEBYTECODE=1  PYTHONUNBUFFERED=1  PYTHONHASHSEED=0
        PATH=/usr/local/bin:/usr/local/sbin:/usr/bin:/bin
  insightflow-sandbox:py311 python /sandbox/run.py /in/job.json
```

- **不传宿主环境变量**：只传显式列出的 6 个 `--env`，密钥与 ModelGate 配置不进入容器；
- **不挂 Docker socket、不挂宿主目录**、不使用 `shell=True`；
- **并发上限**放在**类级别**信号量（默认 2）：工具实例会按数据集指纹缓存出多个，
  实例级信号量无法真正封顶宿主容器数。拿不到配额即 fail-closed，有界等待。
- **孤儿清理**只按 `label=insightflow.sandbox=1` + 本 Run 名前缀匹配，
  不使用 `docker system prune`，不触碰其他 Docker 服务。

---

## 三、宿主/容器数据协议

**不使用 pickle**（跨版本不安全，且把反序列化风险交给沙箱），全部为带类型标注的 JSON。

- 输入：`/in/job.json` + `/in/code.py`（代码落成受控文件，不内联进 JSON、不经 shell）
  + `/in/data/*.json`（`encode_dataframe` 编码，只传本次用到的表）；
- 输出：`/scratch/result.json`，`{success, result, error, truncated, artifacts, diagnostics}`；
- 结果先过 Schema 与体积校验再进 State，避免畸形结构扩散到下游节点。

### 错误分类

| 类型 | 触发 | 是否进入 LLM 修复 |
|---|---|---|
| `unavailable` | daemon 不可用 / 镜像缺失 / 并发配额耗尽 / 缺运行期上下文 | **否** |
| `timeout` | 宿主 wall-time 到期 | **否** |
| `oom` | 退出码 137 | **否** |
| `execution_error` | 模型代码抛异常 / 非只读 SQL | 是（有界修复） |
| `invalid_result` | 结果无法 JSON 表达 / 超限 | 是 |
| `unsupported_tool` | 工具名不存在 | 是 |

`ENVIRONMENT_ERROR_KINDS = {unavailable, timeout, oom}`。分类**不靠解析错误文案**，
它作为结构化字段一路穿过：`SandboxOutcome.error_kind` → 工具契约 →
`state.execution_error_kind` → `conditions.route_after_execution` →
`TaskFailure.error_kind` 与 JSONL 日志。

**实测重试次数**（Mock LLM 调用计数）：

| 失败类型 | Repair Agent 调用次数 | 修复循环执行次数 |
|---|---|---|
| daemon 不可用 | 0 | 0 |
| 镜像不存在 | 0 | 0 |
| 执行超时 | 0 | 0 |
| OOM | 0 | 0 |
| 普通执行异常 | 1 | 3（`retry_count` 0→1→2→3） |

---

## 四、镜像与可复现性

| 项 | 值 |
|---|---|
| tag | `insightflow-sandbox:py311` |
| base | `python@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce` |
| base 平台 | `linux/amd64`（本地核验） |
| 容器内版本 | python 3.11.17 / pandas 3.0.6 / numpy 2.4.6 / matplotlib 3.11.2 / openpyxl 3.1.5 |

构建与运行分离：构建需要网络，运行期一律 `--network none`，因此镜像必须预构建，
运行期**禁止隐式拉取**。镜像缺失即 `unavailable`，不回退到同进程执行。

日常操作：

```bash
.venv/Scripts/python.exe sandbox/build_image.py          # 构建 + 记录 + CJK 验证
.venv/Scripts/python.exe -m pytest -q -m sandbox         # 沙箱用例
.venv/Scripts/python.exe scripts/sandbox_report.py       # 按实测口径分类统计
```

**CJK 字体**：matplotlib 在 Linux 上经 fontconfig 发现系统字体，因此镜像同时安装
`fontconfig` 与 `fonts-noto-cjk`（只装后者不够），并把 `Noto Sans CJK SC` 置于字体首选。
**不要设置 `MPL_IGNORE_SYSTEM_FONTS`**——那会让 matplotlib 忽略全部系统字体。
只读根会触发 `Fontconfig error: No writable cache directories`，用 `XDG_CACHE_HOME=/tmp` 解决。

---

## 五、尚未解决的限制（如实记录）

以下每一条都是本阶段**确认存在**、未解决的边界，不要在后续文档或对外说明中弱化。

### 5.1 隔离强度

1. **Docker Desktop 隔离不是绝对安全保证。** `--cap-drop ALL` +
   `no-new-privileges` + 默认 seccomp 显著收窄攻击面，但容器与宿主**共享内核**，
   内核漏洞仍可能逃逸。本机 `SecurityOptions` 不含 rootless，这是隔离强度的上限。
2. **资源限制只约束单容器。** `--memory` / `--cpus` / `--pids-limit` 不构成宿主级配额；
   并发容器仍可耗竭宿主资源。
3. **单进程容器信号量不是跨服务实例的宿主级资源配额。** 默认 2 的上限是
   **本进程内**的约束；多进程或多服务实例部署时，宿主上的容器总数不受此限制。

### 5.2 符号链接

4. **符号链接拒绝逻辑未在有创建权限的 Windows 环境做真实端到端验证。**
   当前环境宿主 `os.symlink` 报 `WinError 1314`（缺特权），容器内模型作用域不含 `os`，
   因此无法构造真实符号链接。已用两条替代验证：打桩 `is_symlink` 验证拒绝分支、
   用可创建的真实**硬链接**验证不会被链接形式绕过。**真实链接仍待具备权限的环境验证。**

### 5.3 构建可复现性

5. **Registry 不可访问，基础镜像 digest 的跨机器拉取与构建尚未验证。**
   本机 `registry-1.docker.io` 不可达（HTTP 000），无法与官方 manifest-list digest 比对。
   记录的是本地已核验的 `linux/amd64` 镜像 digest。换机器构建时若 registry 的
   manifest digest 与之不同，构建会**明确失败**（拉不到），不会静默换用别的镜像。
6. **Python wheel 哈希与 Debian 系统包版本尚未完全固定。**
   `requirements.lock` 固定了包版本，但 wheel 未按 sha256 固定；
   `fontconfig`、`fonts-noto-cjk` 只固定包名，未固定 Debian 版本。
   因此**不构成完全可复现构建**。
7. **Python 补丁版本不一致**：容器 3.11.17 / 宿主 3.11.9。
   测试以值相等为准，不比较序列化字节。

### 5.4 产物与运行期

8. **staging 文件在强杀期间可能不完整。** 进程在容器写 staging 时被杀会留下半个文件。
   由于宿主必须先通过完整校验才会发布，**未经验证的产物不会被发布**——
   但 staging 目录本身可能残留不完整文件。
9. **`noexec` 的能力边界。** 它阻止从 `/tmp` **直接执行文件**（实测 `PermissionError`），
   但**不能**阻止解释器执行 `/tmp` 里的脚本（实测 `python /tmp/x.py` 仍成功）。
   它只消除"落盘再执行"这一条路径，不构成"禁止执行代码"的保证。
10. **模型代码可见本次 Run 的数据。** 这是设计要求（代码要能分析数据），
    同一 Run 内模型代码可读取全部传入的表，数据敏感性仍然存在。
11. **`run_probe` 是测试诊断辅助**，复用生产参数集执行宿主脚本，不是对外能力。
    若将来接到外部输入，它会成为绕过作用域限制的通道。

### 5.5 范围外

12. **SQL 仍在宿主进程执行，隔离强度低于 Python/Plot。** 属于 P2-B 范围。
    仅有两层保护：`utils/sql_guard.ensure_read_only()`（词法，fail-closed）+
    `sqlite3.connect(f"file:{path}?mode=ro", uri=True)`。
    `set_authorizer`、`set_progress_handler`、`fetchmany` 行数上限、`ATTACH` 拒绝
    **均未实现**。

---

## 六、P2-B 高优先级阻塞项

### 6.1 Repair Agent 的 LLM 操作 ID 未包含真正的逻辑尝试序号

**这是 P2-B 的第一项开发任务，优先于 SQL 增强与大结果修复。**

现象（实测，`tests/test_error_classification.py` 锁定）：

- 修复循环最多推进三次（`execution_results` 的 `retry_count` 为 0、1、2、3）；
- **实际 Repair Agent 只请求模型一次**；
- 其余尝试复用第一次生成的修复代码，后两条错误记录完全相同。

原因：`utils/idempotency.llm_operation(run_id, node, subject)` 的 subject 只含
`task_id`，**不含尝试序号**；`repair_node` 传入的 subject 是 `task["task_id"]`，
因此第 2、3 次尝试算出的操作 ID 与第 1 次相同，`already_done` 直接复用旧结果。
`tool_execution` 的 `task_operation` 有 `attempt_N` 处理，`llm_operation` 没有。

P2-B 必须：

1. 区分**不同的逻辑 Repair 尝试**与**同一次尝试的 Checkpoint 重放**；
2. 为真正的新尝试生成**不同且稳定**的 LLM 操作 ID
   （稳定 = 恢复后重算得到同一个值，因此不能含时间戳或随机数）；
3. **保留 LLM Ledger 对未确认调用的人工审批机制**；
4. 测试连续两次不同错误反馈是否**真正触发两次新的模型请求**；
5. 测试同一次尝试在**跨进程恢复**时是否正确复用或进入未确认审批；
6. **不允许**通过随机 ID、关闭账本或取消幂等机制绕过问题。

### 6.2 其他 P2-B 事项

- **SQL 增强**：`set_authorizer`（拒绝写与 `SQLITE_ATTACH`）、`set_progress_handler`
  查询超时、`fetchmany` 行数/字节上限。三项在设计阶段已实测可行。
- **大结果外置后未正确恢复**：`utils/result_artifacts.load_result()` **无任何调用点**。
  实测 20000 行 DataFrame（849 KB）被外置后，`report_node` / `visualizer` 拿到的是
  `{"__kind__": "externalized", ...}` 摘要而非数据。当前数据集极小未触发，
  沙箱化后结果体积只会更大。
- 宿主 `requirements.txt` 仍未固定版本（沙箱已独立锁定）。
- `--cpus` 的实际限速效果未做基准测量（仅确认参数被接受）。
- 未做多 Run 并发打满沙箱配额的压力测试。

---

## 七、测试分类口径

分类**不靠人工标注**：`tests/conftest.py` 的 hook 观测
`sandbox.executor.containers_spawned()` 在每个用例期间的增量，
`scripts/sandbox_report.py` 汇总。因此报告不会与实际行为脱节。

| 类别 | 数量 | 含义 |
|---|---|---|
| A. 真实容器执行 | 47 | 该用例期间确实启动了容器 |
| B. 故障模拟 | 23 | 未启动容器，验证 fail-closed 与错误分类；**依赖 Docker 可用**（否则整体跳过） |
| C. 静态断言 | 4 | 源码、缓存键、参数结构，与 Docker 无关 |

**B 类不能算作"实际进入容器"。** 全量结果：`388 passed / 0 failed / 0 skipped`。
