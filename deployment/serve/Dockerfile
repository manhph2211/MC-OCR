FROM pytorch/torchserve:0.1-cpu

COPY requirements.txt .
USER root
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 -y

USER model-server
RUN pip install --upgrade pip
RUN pip install -r /home/model-server/requirements.txt
RUN pip install --upgrade requests
COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8