#!/bin/bash

#Note: the protocol used below must match the one specified in dcgm_wsgi_nginx.conf. proxy_pass = HTTP. uwsgi_pass = UWSGI

#Start using the UWSGI protocol
PYTHONPATH=/usr/local/dcgm/bindings  /usr/local/bin/uwsgi --enable-threads --socket :1980 --wsgi-file /usr/share/dcgm_wsgi/dcgm_wsgi.py --logger syslog:dcgm_wsgi --daemonize2

#Start using the HTTP protocol
#PYTHONPATH=/usr/local/dcgm/bindings  /usr/local/bin/uwsgi --enable-threads --http :1980 --wsgi-file /usr/share/dcgm_wsgi/dcgm_wsgi.py --logger syslog:dcgm_wsgi --daemonize2