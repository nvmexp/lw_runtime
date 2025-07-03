#!/usr/bin/python

#--------------------------------------------------------
# At second, launch test command on terminal#2
# after checking breakpoint setting of fault-inj.py.
#
# usage: python3 fault-inj_client.py api_ret_client.json
#--------------------------------------------------------

import subprocess
import json
import os
import sys
import time

# json file
path = sys.argv[1]
# TODO: set IP addr of target device
#---------------------------------------------------
ip = "10.19.11.105"
# TODO: set your TEGRA_TOP
#---------------------------------------------------
TEGRA_TOP = "/home2/rel33/auto"
count = 0;
# TODO: Set test tool path (it's different per S/W module)
#---------------------------------------------------
t_path = "gpu/drv/drivers/lwsci/tests/lwsciipc"
#read api's json file
with open(path) as jf:
    try:
        tfile = json.load(jf)
    except ValueError as e:
        print(e)
        print("Invalid json. The exception raised is :- " )

# push test tool from host to target
# test tool path (from argv[2]), tool_name (from json)
def upload_test(tool_path, tool_name):
    print(tool_path+"\n")
    f1=open("pid2.sh","w")
    f1.write("#!/bin/bash\n")
    cmd=f"sshpass -p root scp {tool_path} root@{ip}:/tmp/{tool_name}\n"
    print(cmd)
    f1.write(cmd)
    f1.close()
    subprocess.call("chmod a+x pid2.sh",shell=True)
    log=subprocess.check_output("./pid2.sh", universal_newlines=True)

# run a process in bg in target and return its pid
def main():
    #go through each api in json file
    for j_component in tfile:
        for key,val in j_component.items():
            if key == "_comment" or key.startswith("#"):
                print(f"### {val}")
                continue
            if key == "TEST_TOOL":
                tool = val
                upload_test(f"{TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/lwpu/{t_path}-qnx_64/{tool}", tool)
                continue
            if key == "TEST_COMMAND":
                print(f"TEST_COMMAND is {val}")
                cmd = j_component["TEST_COMMAND"]
				# used qnx build
                subprocess.call(["/bin/bash", "-c", "source ${HOME}/p4/sw/tools/embedded/qnx/qnx700-ga6/qnxsdp-elw.sh"])

				# create pid3.sh to launch test command
                f1=open("pid3.sh","w")
                f1.write("#!/bin/bash\n")
                f1.write(f"sshpass -p root ssh root@{ip} /proc/boot/sh << EOF\n")
				# if iolauncher is required, use it instead of "on"
                #f1.write("on /tmp/"+path+"\n")
                #f1.write("export LD_LIBRARY_PATH=/tmp:$LD_LIBRARY_PATH\n")
                f1.write(cmd+"\n")
                f1.write("/tmp/sleep 2\n")
                f1.write("EOF\n")
                f1.close()

                subprocess.call("chmod a+x pid3.sh",shell=True)
                log=subprocess.check_output("./pid3.sh")
                #os.remove("pid3.sh")

                print(log.decode("utf-8"))
        # TODO: you might need to tweak this sleep time
        time.sleep(1)
        #time.sleep(2)

if __name__ =='__main__':
    main();


