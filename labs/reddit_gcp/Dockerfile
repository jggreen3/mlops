FROM python:3.11-slim

# Install deps
WORKDIR /
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model artefact
COPY redditApp.py .
COPY reddit_model_pipeline.joblib .

# Expose port and launch FastAPI using Uvicorn
EXPOSE 8000
CMD ["uvicorn", "redditApp:app", "--host", "0.0.0.0", "--port", "8000"]
