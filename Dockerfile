FROM python:3.11.4-slim-bullseye
USER root
WORKDIR /nnunet_pred_folder

# install environment
RUN pip install pip setuptools wheel
COPY requirements.txt ./
RUN pip install -r requirements.txt
# create accessory directories
RUN mkdir /model && \
    mkdir -p /data && \
    mkdir -p /data/input && \
    mkdir -p /data/output && \
    mkdir -p utils

COPY utils/utils.py utils/entrypoint.py utils/pydicom_seg_writers.py utils/
RUN chown root -R utils
ENTRYPOINT ["python", "utils/entrypoint.py","-o /data/output","-m /model"]