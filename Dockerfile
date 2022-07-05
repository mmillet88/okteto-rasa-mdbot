FROM python:3.7.7-stretch AS BASE

RUN apt-get update \
    && apt-get --assume-yes --no-install-recommends install \
        build-essential \
        curl \
        git \
        jq \
        libgomp1 \
        vim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

RUN pip install rasa
RUN pip install transformers
RUN pip install transformers sentencepiece datasets
RUN pip install -U sentence-transformers
RUN pip install torchvision
RUN pip install numpy


#Optional step

ADD config.yml config.yml
ADD domain.yml domain.yml
ADD credentials.yml credentials.yml
ADD endpoints.yml endpoints.yml