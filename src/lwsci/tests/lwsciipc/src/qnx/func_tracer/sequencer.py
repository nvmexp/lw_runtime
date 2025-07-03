#
# Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
#

import io
import re
import json
import sys
import os, subprocess

P4ROOT = os.elwiron['P4ROOT']
TEGRA_TOP = os.elwiron['TEGRA_TOP']

print(P4ROOT)
print(TEGRA_TOP)

def fetchOffset(intAddr):
    jsonName = sys.argv[2]
    jsonFile = open(jsonName)
    jsonData = json.load(jsonFile)
    for module in jsonData['module']:
        baseAddr = int(module['base_address'], 16)
        endAddr = int(module['end_address'], 16)
        if intAddr >= baseAddr and intAddr < endAddr:
            offset = hex(intAddr - baseAddr)
            cmd = "${P4ROOT}/sw/tools/embedded/qnx/qnx700-ga6/host/linux/x86_64/usr/bin/aarch64-unknown-nto-qnx7.0.0-addr2line -e ${TEGRA_TOP}/" + module['binary_path'] + " -f " + offset
            process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            functionName = process.stdout.readline().decode()
            fileName = process.stdout.readline().decode()
            break
        else:
            print(str("---- ")+hex(intAddr))
    return functionName, fileName

lines = []
inFile = sys.argv[1]
tempFilename = inFile + "_temp"
tempFile = open(tempFilename, "w")
outFilename = inFile + "_out"
outFile = open(outFilename, "w")

with open(inFile, 'r') as inFile:
    for line in inFile:
        funcName = ''
        fileName = ''
        newLine = ''
        #Search for Resmgr Entry
        if (re.search("Resmgr Function", line)):
            hexAddr = line.split()[-1]
            intAddr = int(hexAddr.split(':')[-1].split('x')[-1], 16)
            funcName, fileName = fetchOffset(intAddr)
            newLine = line.rsplit(' ',1)[0] + " " + funcName.strip() + "\n"
            tempFile.write(newLine)
        elif (re.search("Library Function", line)):
            hexAddr = line.split()[-1]
            intAddr = int(hexAddr.split(':')[-1].split('x')[-1], 16)
            funcName, fileName = fetchOffset(intAddr)
            newLine = line.rsplit(' ',1)[0] + " " + funcName.strip() + "\n"
            tempFile.write(newLine)

tempFile.close()

with open(tempFilename, 'r') as inFile:
    for line in inFile:
        hexAddr = line.split('0x')[-1].split(']')[0]
        intAddr = int(hexAddr, 16)
        lines.append((intAddr, line))

#
# slog2info command list down the slog buffer per process
# in a way which is not as per the actual timing of the
# logs. To have function tracer slogs as per real timeline,
# sort the logs based cntvct_el0 counter values.
#
    lines.sort()
    sorted_lines = [x[1] for x in lines]
    for l in sorted_lines:
        outFile.write(l)

outFile.close()
os.remove(tempFilename)
