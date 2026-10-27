# Validates and scores submissions. Build with `kagent build-images`.
FROM python:3.11-slim

# Same data stack as the sandbox, pinned with hashes.
COPY docker/grader.lock.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG LOCK_SHA256=unknown
LABEL kagent.lock-sha256=$LOCK_SHA256

WORKDIR /app
COPY src/kagent/grader /app/grader

ENTRYPOINT ["python", "-m", "grader.main"]
