FROM eclipse-temurin:8-jdk-jammy AS jdk8

FROM ghcr.io/astral-sh/uv:python3.11-trixie

# Install build/runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    curl \
    tini \
    xvfb \
    mesa-utils \
    libegl1 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=jdk8 $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md setup.py ./
COPY src ./src
COPY jarvisvla ./jarvisvla
COPY assets ./assets
COPY configs ./configs
COPY scripts ./scripts

RUN \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ARG DOWNLOAD_JARVIS_MODEL=0
ARG JARVIS_MODEL_ID=CraftJarvis/JarvisVLA-Qwen2-VL-7B
ENV VLA_MODEL_PATH=/models/JarvisVLA-Qwen2-VL-7B

RUN mkdir -p /models && \
    if [ "$DOWNLOAD_JARVIS_MODEL" = "1" ]; then \
      uv run python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${JARVIS_MODEL_ID}', local_dir='${VLA_MODEL_PATH}', local_dir_use_symlinks=False)"; \
    fi

COPY scripts/docker/entrypoint.sh /usr/local/bin/agentbeats-entrypoint.sh
RUN chmod +x /usr/local/bin/agentbeats-entrypoint.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 9009
EXPOSE 9020

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/agentbeats-entrypoint.sh"]
