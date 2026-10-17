"""Agent Evaluation & Reliability Benchmark。

模块划分：

- `tasks`：标准任务集合与**独立算出**的标准答案
- `scoring`：判分口径与指标（完成率与正确率分开）
- `runner`：统一执行器与报告输出
- `faults`：可靠性故障注入场景

**Mock 边界**：用 Mock LLM 跑出的结果只能证明工程链路的可靠性，
不能证明真实模型的自主分析能力。见 `runner.MOCK_SCOPE_NOTE`。
"""
