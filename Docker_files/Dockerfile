FROM tensorflow/tensorflow:latest-gpu-jupyter  

MAINTAINER Wyctor Fogos da Rocha 

LABEL version="1.0"
LABEL description="Tensorflow-Gpu-Jupyter"

RUN pip install --upgrade pip

WORKDIR /Docker_files/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8888
