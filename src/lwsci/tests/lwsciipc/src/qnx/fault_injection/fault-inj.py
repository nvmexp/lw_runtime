#!/usr/bin/python

#-------------------------------------------------
# At first, launch test command on terminal#1
#
# USAGE: python3 fault-inj.py api_ret.json
#-------------------------------------------------

import subprocess
import json
import os
import sys
import time
import ntpath

path = sys.argv[1]
# TODO: set IP addr of target device
#-------------------------------------------------
ip = "10.19.11.105"
# TODO: define TEGRA_TOP of your code tree and QNX target tool in P4
#-------------------------------------------------
TEGRA_TOP = "/home2/rel33/auto"
HOST_QNX_TARGET_TOOL = "${P4ROOT}/sw/tools/embedded/qnx/qnx700-ga6/target/qnx7/aarch64le"

# TODO: set test output log file of test tool (it can be different per S/W module)
#-------------------------------------------------
TEST_OUTPUT = "/tmp/LWSCIIPC_TEST"
# TODO: set addtional clean up command or comment it
#-------------------------------------------------
#MORE_CLEAN_CMD = "/tmp/slay -f {program}\n"
# TODO: set report filename
#-------------------------------------------------
REPORT_FILE = "REPORT.TXT"

#------------------------------------------------
# define MISC tools
#------------------------------------------------
# TODO: use "cat " or "/tmp/cat "
# In case of LwSciIpc, "/tmp/cat " is required since VSCD would be broken during test
#------------------------------------------------
cat = "/tmp/cat "
# TODO: use "rm " or "/tmp/rm "
# In case of LwSciIpc, "/tmp/rm " is required since VSCD would be broken during test
#------------------------------------------------
rm = "/tmp/rm -f "
# TODO: use "awk " or "/tmp/awk "
# In case of LwSciIpc, "/tmp/awk " is required since VSCD would be broken during test
#------------------------------------------------
awk = "/tmp/awk "
# Miscellany tools from QNX target tool
#------------------------------------------------
scp = f"scp {HOST_QNX_TARGET_TOOL}/"

#read api's json file
with open(path) as jf:
    try:
        tfile = json.load(jf)
    except ValueError as e:
        print(e)
        print("Invalid json. The exception raised is :- " )

with open("./api_ret_client.json") as jf:
    try:
        tfile1 = json.load(jf)
    except ValueError as e:
        print(e)
        print("Invalid json. The exception raised is :- " )

def find_test_id(test_id):
    #print("find_test_id: %d---->" % (test_id))
    for j_component1 in tfile1:
        for key,val in j_component1.items():
            #print("find_test_id: key(%s) val(%d)" % (key, val))
            if key == "_comment" or key.startswith("#"):
                continue
            if key == "TEST_TYPE":
                continue
            if key == "test_id":
                #print("test_id is %s" % (key))
                if val != test_id:
                    #print("test_id(%d) is differnt with %d" % (test_id, val))
                    break
                else:
                    #print("test_id(%d) is same with %d" % (test_id, val))
                    continue
            if key != "TEST_COMMAND":
                #print("key(%s) has val(%d)" % (key, val))
                return val

def cleanup_result():
    print("clean up previous result\n")

    f1=open("pid0.sh","w")
    f1.write("#!/bin/bash\n")
    f1.write(f"sshpass -p root ssh root@{ip} /proc/boot/sh <<EOF\n")
    # test output of test program
    f1.write(f"{rm} {TEST_OUTPUT}\n")
    # TODO: comment below command if you don't need
    #------------------------------------------------
    f1.write("/tmp/slay -f test_lwsciipc_perf\n")
    f1.write("/tmp/slay -f test_lwsciipc_read\n")
    f1.write("/tmp/slay -f test_lwsciipc_readm\n")
    f1.write("EOF\n")
    f1.close()

    subprocess.call("chmod a+x pid0.sh",shell=True)
    log=subprocess.check_output("./pid0.sh", universal_newlines=True)
    #os.remove("pid0.sh")

def slay_test_tool():
    print("slay test tool\n")

    f1=open("pid1.sh","w")
    f1.write("#!/bin/bash\n")
    f1.write(f"sshpass -p root ssh root@{ip} /proc/boot/sh <<EOF\n")
    f1.write("/tmp/slay -f test_lwsciipc_perf\n")
    f1.write("/tmp/slay -f test_lwsciipc_read\n")
    f1.write("/tmp/slay -f test_lwsciipc_readm\n")
    f1.write("EOF\n")
    f1.close()

    subprocess.call("chmod a+x pid1.sh",shell=True)
    log=subprocess.check_output("./pid1.sh", universal_newlines=True)
    #os.remove("pid1.sh")

# build standard flavor to generate tools
def push_misc_tools():
    print("push miscellany tools\n")
    f1=open("pid.sh","w")
    f1.write("#!/bin/bash\n")
    # TODO: add your misc tools
    #------------------------------------------------
    f1.write(f"sshpass -p root {scp}bin/cat       root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}bin/ls        root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}bin/pidin     root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}bin/ps        root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}bin/rm        root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}bin/slay      root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/awk   root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/cut   root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/find  root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/ldd   root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/scp   root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/sleep root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/tee   root@{ip}:/tmp\n")
    f1.write(f"sshpass -p root {scp}usr/bin/which root@{ip}:/tmp\n")
    f1.close()
    subprocess.call("chmod a+x pid.sh",shell=True)
    log=subprocess.check_output("./pid.sh", universal_newlines=True)

def validate_result():
    # wait for result
    time.sleep(1)
    report=open(REPORT_FILE, "w")
    # write logging
    f1=open("pidR.sh","w")
    f1.write("#!/bin/bash\n")
    f1.write(f"sshpass -p root ssh root@{ip} /proc/boot/sh <<EOF\n")
    # test output of test program
    f1.write(f"{cat} {TEST_OUTPUT} \n")
    f1.write("EOF\n")
    f1.close()
    subprocess.call("chmod a+x pidR.sh",shell=True)
    log=subprocess.check_output("./pidR.sh", universal_newlines=True)
    # write target logging of pidR.sh
    f2=open("target_log", "w")
    f2.write(log)
    f2.write("\n")
    f2.close()
    print(log)
    with open("target_log","r") as myfile:
        # reading each line
        for line in myfile:
            if line == "\n":
                break
            # reading each word
            cnt = 0
            for word in line.split():
                if cnt == 0:
                    test_id = word
                    cnt += 1;
                    continue;
                if cnt == 1:
                    api_name = word
                    cnt += 1;
                    continue
                result = int(word);
            #print("validate_result: %s %s %d\n" % (test_id, api_name, result))
            exp_ret = int(find_test_id(int(test_id)))
            if result == exp_ret:
                #print("result(%d) is ret(%d)" % (result, exp_ret))
                report.write(f"{test_id}\t{api_name}\tPASS\n")
            else:
                #print("result(%d) is not ret(%d)" % (result, exp_ret))
                report.write(f"{test_id}\t{api_name}\tFAIL\n")
    report.close()
    os.remove("target_log")
    print("\n-------------------------------------------------")
    print("TID\tTESTINFO\t\tRESULT")
    print("-------------------------------------------------")
    os.system(f"cat {REPORT_FILE}")
    print("-------------------------------------------------\n\n")

# breakpoint on function of library
def run_lib_test(key, val, file_to_load, tool_name, cmd_opt):
    subprocess.call(["/bin/bash", "-c", "source ${HOME}/p4/sw/tools/embedded/qnx/qnx700-ga6/qnxsdp-elw.sh"])
    api_name=key
    ret_val=val

    f=open("temp.txt","w")
    f.write("set logging on\n")
    f.write(f"target qnx {ip}:8000\n")
    f.write("set nto-cwd /tmp\n")
    # load symbol
    f.write(f"file {TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/lwpu/{file_to_load}\n")
    # upload tool
    f.write(f"upload {TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/lwpu/{file_to_load} /tmp/{tool_name}\n")
    # break main
    f.write("b main\n")
    # run test program
    f.write(f"r {cmd_opt}\n")
    f.write("i sharedlibrary\n")
    # set solib search path
    f.write(f"set solib-search-path {TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/systemimage/userspace/\n")
    # set breakpoint with library function
    f.write(f"b {api_name}\n")
    # continue
    f.write("c\n")
    # set return value
    f.write(f"return (long long int) {ret_val}\n")
    f.write("y\n")
    f.write("c\n")
    f.write(f"clear {api_name}\n")
    f.close()

    subprocess.call("${HOME}/p4/sw/tools/embedded/qnx/qnx700-ga6/host/linux/x86_64/usr/bin/ntoaarch64-gdb < temp.txt",shell=True)
    #os.remove("temp.txt")
    # TODO: you might need to tweak sleep time
    #------------------------------------------------
    time.sleep(1)

def attach_to_resmgr(key, val, file_to_load, resmgr_pid):
    subprocess.call(["/bin/bash", "-c", "source ${HOME}/p4/sw/tools/embedded/qnx/qnx700-ga6/qnxsdp-elw.sh"])
    api_name=key
    ret_val=val

    f=open("temp.txt","w")
    f.write("set logging on\n")
    f.write(f"target qnx {ip}:8000\n")
    f.write("set nto-cwd /tmp\n")
    f.write(f"set solib-search-path {TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/systemimage/userspace/\n")
    # target exlwtable
    f.write(f"file {TEGRA_TOP}/out/embedded-qnx-t186ref-debug-safety/lwpu/{file_to_load}\n")
    f.write(f"b {api_name}\n")
    # attach resmgr
    f.write(f"attach {resmgr_pid}\n")
    f.write("i b\n")
    # continue
    f.write("c\n")
    f.write(f"return (long long int) {ret_val}\n")
    #f.write("y\n")
    f.write(f"clear {api_name}\n")
    f.close()

    subprocess.call("${HOME}/p4/sw/tools/embedded/qnx/qnx700-ga6/host/linux/x86_64/usr/bin/ntoaarch64-gdb < temp.txt",shell=True)
    os.remove("temp.txt")
    # TODO: you might need to tweak sleep time
    #------------------------------------------------
    slay_test_tool()
    time.sleep(1)


#run a process in bg in target and return its pid
def main():
    # TODO: push addtional tools (from standard build) - comment it if you don't need
    #------------------------------------------------
    push_misc_tools()
    # clean up previous result
    cleanup_result()

    # go through each api in json file
    for j_component in tfile:
        for key,val in j_component.items():
            if key == "_comment" or key.startswith("#"):
                print(f"### {val}\n")
                continue
            if key == "TEST_TYPE":
                test_type = val
                print(f"TEST_TYPE: {test_type}\n")
                continue
            if key == "LOAD":
                filename = val
                print(f"LOAD: {filename}\n")
                binfile=ntpath.basename(filename)
                print(f"BINFILE: {binfile}")
                if test_type == "RESMGR":
                    # get pid of resmgr process
                    print("get pid of resmgr\n")
                    f1=open("pidp.sh","w")
                    f1.write("#!/bin/bash\n")
                    f1.write(f"sshpass -p root ssh root@{ip} /proc/boot/sh <<EOF\n")
                    # get pid of resmgr
                    cmd="/tmp/pidin -P " + binfile + " | " + awk + "'{print \$1}' | " + awk + "'NR==2'\n"
                    print(cmd)
                    f1.write(cmd)
                    f1.write("EOF\n")
                    f1.close()
                    subprocess.call("chmod a+x pidp.sh",shell=True)
                    log=subprocess.check_output("./pidp.sh", universal_newlines=True)
                    resmgr_pid = int(log)
                    print(f"resmgr pid: {resmgr_pid}\n")
                continue
            if key == "CMDOPT":
                cmdopt = val
                print(f"CMDOPT: {cmdopt}\n")
                continue
            if (test_type == "RESMGR"):
                # set brkpoint with api name, return val
                print(f"key:{key}, value:{val}\n")
                attach_to_resmgr(key, val, filename, resmgr_pid);
                test_type = ""
                binfile = ""
                test_tool = ""
                cmdopt = ""
            elif (test_type == "LIB"):
                # set brkpoint with api name, return val
                print(f"key:{key}, value:{val}\n")
                run_lib_test(key, val, filename, binfile, cmdopt);
                test_type = ""
                filename = ""
                binfile = ""
                cmdopt = ""
            else:
                continue;
    validate_result()
if __name__ =='__main__':
    main();

