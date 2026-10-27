# Runs agent-written scripts. No grader code, no private labels, no secrets baked in.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep in sync with SANDBOX_VERSIONS in src/kagent/search/prompts.py (a test checks this).
# The prompt tells the model these versions so it avoids APIs removed since its training data.
RUN pip install --no-cache-dir numpy==2.4.6 pandas==3.1.0 scikit-learn==1.9.1 lightgbm==4.7.0

WORKDIR /work
