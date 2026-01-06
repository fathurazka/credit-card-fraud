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

# Run server (PORT is set by Railway)
CMD ["python", "server.py"]
