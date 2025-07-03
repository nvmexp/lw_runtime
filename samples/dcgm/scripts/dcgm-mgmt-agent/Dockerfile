#######################################
# LWCA Dockerfile
#######################################

# Start from a fresh ubuntu image ##
	FROM lwca/base_docker
COPY build/dcgm-mgmt-agent_1.0.deb /home/dcgm/
RUN dpkg -i /home/dcgm/dcgm-mgmt-agent_1.0.deb
RUN  chown -R dcgm /opt/
RUN echo "* * * * * python /opt/dcgm-mgmt-agent/node-api.py" > mycron
RUN crontab mycron; rm mycron
