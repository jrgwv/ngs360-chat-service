# Amazon ECR Public Python image (no Docker Hub rate limits), matching the
# convention used by the sibling ngs360-sql-langgraph-agent service.
FROM public.ecr.aws/docker/library/python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as an unprivileged user (fixed uid so orchestrators can enforce
# runAsNonRoot / match volume ownership).
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home \
       --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER 10001:10001

EXPOSE 3100

# --timeout 0 disables the worker request timeout: chat responses are
# long-lived SSE streams and must not be killed mid-stream (this is exactly the
# knob that fights the REST API's --timeout 120).
CMD ["gunicorn", "main:app", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:3100", \
     "--timeout", "0", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
