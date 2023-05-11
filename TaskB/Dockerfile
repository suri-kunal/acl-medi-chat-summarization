FROM python:3.8

WORKDIR /workspace

RUN git config --global user.email "kunal.suri.ml.experiments@gmail.com"
RUN git config --global user.name "Kunal Suri"

COPY requirements.txt .

RUN pip install -r requirements.txt
