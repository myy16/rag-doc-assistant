# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build arguments for improved pip reliability
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG PIP_TIMEOUT=120

# Explicitly clear proxy settings in build stage so Docker Desktop daemon proxy does not interfere
ENV http_proxy=""
ENV https_proxy=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY="localhost,127.0.0.1,::1,pypi.org,*.pypi.org,api.groq.com,*.groq.com"

RUN printf 'Acquire::http::Proxy "";\nAcquire::https::Proxy "";\n' > /etc/apt/apt.conf.d/99no-proxy
RUN cat > /etc/apt/sources.list <<'EOF'
deb https://deb.debian.org/debian trixie main contrib non-free
deb https://deb.debian.org/debian-security trixie-security main contrib non-free
deb https://deb.debian.org/debian trixie-updates main contrib non-free
EOF

# Install system deps needed by pdfplumber / antiword
RUN apt-get update -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false && apt-get install -y --no-install-recommends \
    antiword \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into a separate layer for caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch \
    && pip install --no-cache-dir \
       --retries 5 \
       --default-timeout ${PIP_TIMEOUT} \
       -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV http_proxy=""
ENV https_proxy=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV NO_PROXY="localhost,127.0.0.1,::1,pypi.org,*.pypi.org,api.groq.com,*.groq.com"

RUN printf 'Acquire::http::Proxy "";\nAcquire::https::Proxy "";\n' > /etc/apt/apt.conf.d/99no-proxy
RUN cat > /etc/apt/sources.list <<'EOF'
deb https://deb.debian.org/debian trixie main contrib non-free
deb https://deb.debian.org/debian-security trixie-security main contrib non-free
deb https://deb.debian.org/debian trixie-updates main contrib non-free
EOF

# Copy antiword binary from builder; no runtime apt install needed
COPY --from=builder /usr/bin/antiword /usr/bin/antiword

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Create required directories (volumes mount over these, ownership must be set first)
RUN mkdir -p uploads data/chroma

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]