# Startup IQ web app (CPU). Build: docker build -t startupiq .   Run: docker run -p 8000:8000 startupiq
# The app needs data/processed (python -m startupiq.clean) and data/models (trained models) at build time;
# raw datasets are not redistributed (licences: devdocs/decisions/0002).
FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY docs/results/dl_weekly.json ./docs/results/dl_weekly.json
COPY data/processed ./data/processed
COPY data/models ./data/models
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[dl,api]"
EXPOSE 8000
CMD ["uvicorn", "startupiq.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
