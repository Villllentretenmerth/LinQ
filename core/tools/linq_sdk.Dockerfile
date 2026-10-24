# syntax=docker/dockerfile:1
# Образ с SDK LinQ (tpu_framework): только Linux x86_64 + Python 3.12.11 — на Windows и в Jupyter хакатона
# (Python 3.11) его не поставить. Контекст сборки — папка tpu_framework из tpu_framework_<версия>.zip:
#
#   docker build -t team94-linq-sdk -f core/tools/linq_sdk.Dockerfile C:/путь/к/tpu_framework
#
# SDK требует torch==2.9.1 и torchvision==0.24.1; ставим их CPU-сборки заранее — иначе pip тянет ещё ~3 ГБ
# библиотек CUDA, которые для калибровки и компиляции не нужны. Кеш pip переживает пересборку.
# Дальше компиляция — core/tools/compile_tpu.py (команда в его описании).
FROM python:3.12.11
COPY packets /packets
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && \
    pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu && \
    cd /packets && pip install -r requirements.txt
ENV PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
