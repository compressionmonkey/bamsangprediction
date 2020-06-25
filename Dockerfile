FROM ubuntu:18.04
MAINTAINER keldendraduldorji@gmail.com

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install gunicorn -y
RUN ls

COPY predictioncontainer /opt/
WORKDIR /opt/
