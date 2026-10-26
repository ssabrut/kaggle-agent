# Runs agent-written scripts. No grader code, no private labels, no secrets baked in.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy pandas scikit-learn lightgbm

WORKDIR /work
