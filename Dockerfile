FROM ubuntu:18.04
MAINTAINER keldendraduldorji@gmail.com

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install gunicorn -y

COPY requirements.txt requirements.txt
COPY predictioncontainer /opt/

RUN pip3 install -r requirements.txt
WORKDIR /opt/

CMD ['gunicorn','-b','0.0.0.0:8000','app:app','--workers=5']