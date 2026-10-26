# Validates and scores submissions. Build from the repo root:
#   docker build -f docker/grader.Dockerfile -t kagent-grader:latest .
FROM python:3.11-slim

RUN pip install --no-cache-dir numpy pandas scikit-learn

WORKDIR /app
COPY src/kagent/grader /app/grader

ENTRYPOINT ["python", "-m", "grader.main"]
