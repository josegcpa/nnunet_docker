FROM python:3.11.4-slim-bullseye
USER root
WORKDIR /nnunet_pred_folder

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir /model; mkdir -p /data; mkdir -p /data/input; mkdir -p /data/output

COPY utils utils
COPY assets assets
COPY entrypoint.sh .
ENTRYPOINT ["./entrypoint.sh","-o /data/output","-m /model"]