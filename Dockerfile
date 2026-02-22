# syntax=docker/dockerfile:1
# 构建阶段：安装依赖（可选多阶段时仅复制 venv，此处保持单阶段以简化）
ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION} AS builder

WORKDIR /app

# 仅复制依赖文件，利用 Docker 层缓存
COPY requirements.txt .

# 升级 pip 并安装依赖到临时目录（若后续做多阶段可复制 /install）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 'uvicorn[standard]'

# 运行阶段
ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION}

LABEL maintainer="zeabur"
LABEL language="python"
LABEL framework="fastapi"

# 使用非 root 用户运行
RUN addgroup --system --gid 1000 app && \
    adduser --system --uid 1000 --gid 1000 --no-create-home app

WORKDIR /app

# 从 builder 复制已安装的包（同阶段则直接在本镜像安装）
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 应用代码
COPY --chown=app:app . .

# 端口可通过环境变量覆盖（Zeabur/云平台常用）
ENV PORT=8080
EXPOSE ${PORT}

USER app

# 使用 PORT 环境变量，便于云平台动态端口
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

