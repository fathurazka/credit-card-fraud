# Lightweight Docker image for Credit Card Fraud Detection API
# Size: ~150MB (vs ~1.5GB for MLflow's default image)

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    scikit-learn \
    joblib \
    numpy

# Copy model artifacts and server
COPY MLproject/preprocessor.joblib ./preprocessor.joblib
COPY MLproject/model.joblib ./model.joblib
COPY server.py ./server.py

# Environment variables
ENV MODEL_PATH=/app/model.joblib
ENV PREPROCESSOR_PATH=/app/preprocessor.joblib
ENV PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ping || exit 1

# Run server
CMD ["python", "server.py"]
