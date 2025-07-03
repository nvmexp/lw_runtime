#!/usr/bin/python

import os
import subprocess
import sys
import json
from pprint import pprint
from subprocess import call

## TODO: All the startup routines related to collectD should go here
def checkCollectDServices():
    # 1. Check if the services on live
    # 2. Start services if they are not live
    return;

## TODO: Services startup related to the master go here
def checkMasterServicesStatus():
    # 1. Check if the services on live
    # 2. Start services if they are not live
    return;

## TODO: Services startup related to the slaves go here
def checkSlaveServicesStatus():
    # 1. Check if the services on live
    # 2. Start services if they are not live
    return;

def queryDB(key):
    p = subprocess.Popen('python cosmosdb.py -j -k '+key, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT);
    try:
        jstr = (p.stdout.readlines())[0];
        jstr = jstr.strip();
        jsonstr=json.loads(jstr);
        return jsonstr[key]
    except:
        return "na"

def getNodeStatus():
    p = subprocess.Popen('python cosmosdb.py -j -a', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT);
    try:
        jstr = (p.stdout.readlines())[0];
        jstr = jstr.strip();
        jsonstr=json.loads(jstr);
        return jsonstr
    except:
        return "na"

def main():
    if ( queryDB("cloud_managed") != "yes"):
        return;

    checkCollectDServices();

    ## Now, lets check the node type & check the related services
    nodeType = queryDB("cluster_group");
    print nodeType;
    if (nodeType == "master"):
        checkMasterServicesStatus();
    elif (nodeType == "slave"):
        checkSlaveServicesStatus();
    else:
        return;

    # TODO: lwrl the cloud
    nodeStatus = getNodeStatus();
    if (nodeStatus != "na"):
        print "Send Status to cloud";
    # TODO "lwrl -H \"Content-Type: application/json\" -X POST -d " + nodeStatus + " http://localhost:3000/api/login"

### Main entry ###
if __name__ == "__main__":
    sys.exit(main())
