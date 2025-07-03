FROM ubuntu:16.04

ARG DCGM_VERSION=1.5.0

ELW LWIDIA_VISIBLE_DEVICES all
ELW LWIDIA_DRIVER_CAPABILITIES compute,utility

RUN apt-get update && apt-get install -y python python-pip
RUN pip install collectd prometheus_client

# pip requires a large number of dependencies, so a lot of space is saved by
# purging it

RUN apt-get remove -y --purge python-pip && apt-get autoremove -y

COPY datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb /tmp
RUN dpkg -i /tmp/*.deb && rm -f /tmp/*

COPY run-hostengine run-dcgm-client /bin/

EXPOSE 5555/tcp
EXPOSE 8000/tcp

CMD [ "run-hostengine" ]
