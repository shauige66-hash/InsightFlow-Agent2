# InsightFlow API 镜像。
#
# 只装 API 需要的依赖；沙箱镜像（sandbox/Dockerfile）是另一件事——分析代码在
# 那个容器里跑，本镜像只负责编排、存储与接口。
#
# 依赖里最重的是 sentence-transformers（会拉 torch）。这里先装 CPU 版 torch，
# 避免把 CUDA 运行时（数 GB）打进来。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 构建期依赖：psycopg / cryptography 的 wheel 一般够用，这里只补最小的系统库。
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl tini docker.io \
 && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch，再装其余依赖；分两步是为了让 requirements 变化时能复用这一层
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install "transformers>=4.40"

COPY requirements.txt ./
RUN pip install -r requirements.txt

# 应用代码
COPY . .

# 非 root 运行：上传与分析产物都写在这里
RUN useradd --create-home --uid 10001 insightflow \
 && mkdir -p /app/outputs /app/uploads \
 && chown -R insightflow:insightflow /app/outputs /app/uploads
USER insightflow

ENV INSIGHTFLOW_STORAGE=postgres \
    INSIGHTFLOW_HOST=0.0.0.0 \
    INSIGHTFLOW_PORT=8000 \
    INSIGHTFLOW_LOCAL_DB=/app/outputs/conversations.sqlite

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
