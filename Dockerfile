FROM python:3.11.4-slim-bullseye
USER root
WORKDIR /nnunet_pred_folder

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir /model && \
    mkdir -p /data && \
    mkdir -p /data/input && \
    mkdir -p /data/output && \
    mkdir -p utils

COPY utils/utils.py utils/utils.py
COPY utils/entrypoint.py utils/entrypoint.py
RUN chown root -R utils
ENTRYPOINT ["python", "utils/entrypoint.py","-o /data/output","-m /model"]