# ── fine_tuning_service/Dockerfile ──
# Use the shared base that already contains TF-CPU, NumPy, SciPy, Pillow, imageio
FROM deepanshu0903/tf-base:2.11.0-cpu

WORKDIR /app

# Install only service-specific Python dependencies (FastAPI/uvicorn, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
