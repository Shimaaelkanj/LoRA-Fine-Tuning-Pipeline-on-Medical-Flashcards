
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ configs/
COPY src/ src/
COPY data/ data/
COPY deploy/adapters/ deploy/adapters/
COPY run_pipeline.py .
CMD ["python", "run_pipeline.py", "all"]