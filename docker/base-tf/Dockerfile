# /docker/base-tf/Dockerfile
FROM python:3.9-slim

# Install system deps needed for building wheels (if any)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements-base.txt so that this layer can be cached
COPY requirements-base.txt /tmp/requirements-base.txt

# Install all “heavy” Python packages (no──cache-dir to avoid storing .whl)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-base.txt

# (Optionally) clean up apt caches again—already done above
# No application code here. This image only contains:
#   └─ python:3.9-slim + (tensorflow-cpu, numpy, scipy, Pillow, imageio)

