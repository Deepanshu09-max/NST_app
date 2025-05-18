FROM python:3.9-slim-buster

# Install debugging tools
RUN apt-get update && apt-get install -y dnsutils iputils-ping curl grep && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
