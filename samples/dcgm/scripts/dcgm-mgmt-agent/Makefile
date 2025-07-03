VERSION=1.0

list:
	@echo;echo "List of LWCA dockers";\
	echo "====================";\
	sudo docker images|grep lwca;\
	echo ""

init:
	cp templates/mainDockerfile Dockerfile
	sudo docker build -t lwca/base_docker ./
	rm Dockerfile

## Use this to build the package
deb:
	@ tar -zcvf build/pkg.tar.gz pkg
	@ rm -f build/dcgm-mgmt-agent_$(VERSION).deb
	@ cp templates/template Dockerfile
	@ echo "COPY build/pkg.tar.gz /root/" >> Dockerfile
	@ echo "RUN cd /root/; tar -zxvf /root/pkg.tar.gz" >> Dockerfile
	@ echo "RUN cd /root/pkg/;sh run.sh" >> Dockerfile
	@ sudo docker build -t lwca/tmp_docker ./ > /dev/null
	@ rm build/pkg.tar.gz
	@ rm Dockerfile
	@ echo
	@ echo "Installing package..."
	@ echo "====================="
	@ sudo docker run -v $(PWD)/build/:/mnt lwca/tmp_docker /bin/bash -c 'cp /root/pkg/dcgm-mgmt-agent_$(VERSION).deb /mnt/'
	@ echo "  done... \n"

## Use this to test if the package has installed correct
install:
	@ cp templates/template ./Dockerfile
	@ echo "COPY build/dcgm-mgmt-agent_$(VERSION).deb /home/dcgm/" >> Dockerfile
	@ echo "RUN  chown -R dcgm /opt/" >> Dockerfile
	@ echo "USER dcgm" >> Dockerfile
	@ echo "WORKDIR /home/dcgm/" >> Dockerfile
	@ sudo docker build -t lwca/tmp_docker ./ > /dev/null
	@ rm Dockerfile
	@ echo Enter '$$ sudo dpkg -i dcgm-mgmt-agent_$(VERSION).deb   [passwd: dcgm]'
	@ sudo docker run -it lwca/tmp_docker

## Use this to test if the cron job runs correctly
cron: deb
	@ cp templates/template ./Dockerfile
	@ echo "COPY build/dcgm-mgmt-agent_$(VERSION).deb /home/dcgm/" >> Dockerfile
	@ echo "RUN dpkg -i /home/dcgm/dcgm-mgmt-agent_$(VERSION).deb" >> Dockerfile
	@ echo "RUN  chown -R dcgm /opt/" >> Dockerfile
	@ echo "RUN echo \"* * * * * python /opt/dcgm-mgmt-agent/node-api.py\" > mycron"  >> Dockerfile
	@ echo "RUN crontab mycron; rm mycron"  >> Dockerfile
	@ sudo docker build -t lwca/tmp_docker ./ > /dev/null	
	#@ rm Dockerfile
	#@ sudo docker run -it lwca/tmp_docker /bin/bash -c 'python /opt/dcgm-mgmt-agent/node-api.py'
	sudo docker run -it lwca/tmp_docker


.PHONY: clean
clean:
	rm build/*


# Use this to login to the shell:     sudo docker run -it lwca/tmp_docker
