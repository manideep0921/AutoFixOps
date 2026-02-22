FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8000

# Health check hits the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Pass ANTHROPIC_API_KEY via environment at runtime:
# docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 autofixops
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
