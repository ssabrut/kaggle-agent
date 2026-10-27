# Runs agent-written scripts. No grader code, no private labels, no secrets baked in.
# Build with `kagent build-images` (it passes LOCK_SHA256 so a stale image can be detected).
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Every package, direct or transitive, is pinned with hashes in sandbox.lock.txt. New packages
# arrive only through `kagent deps approve`, which recompiles that file.
COPY docker/sandbox.lock.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG LOCK_SHA256=unknown
LABEL kagent.lock-sha256=$LOCK_SHA256

WORKDIR /work
