FROM python:3.9-slim-buster

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now use this image in your service Dockerfiles:
# FROM yourdockeruser/base-ml:latest
