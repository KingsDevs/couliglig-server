FROM python:3.10.12-slim AS builder

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
build-essential \
gfortran \
libgomp1 \
git \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
&& pip install --prefix=/install -r requirements.txt

FROM python:3.10.12-slim
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local
COPY . /app
WORKDIR /app