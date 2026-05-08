FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY stream.py .

ENV FFMPEG_BIN=/usr/bin/ffmpeg
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-u", "stream.py"]
