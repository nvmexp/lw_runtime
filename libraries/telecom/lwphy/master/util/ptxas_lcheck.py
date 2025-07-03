#!/usr/bin/elw python

# Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

# Run like this:
# make -j 20 2>&1 | ../util/ptxas_lcheck.py

import fileinput
import re
import subprocess

re_func_str  = r'ptxas info    : Function properties for (.*)'
re_local_str = r'    (\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads'

func_name = ''

kernels_with_lmem = []

for line in fileinput.input():
    line = line.rstrip()
    if not func_name:
        m = re.search(re_func_str, line)
        if m:
            func_name = m.group(1)
            #print('function: %s' % func_name)
        else:
            func_name = ''
    else:
        m = re.search(re_local_str, line)
        if m:
            if (int(m.group(1)) != 0) or (int(m.group(2)) != 0) or (int(m.group(3)) != 0):
                #print(line)
                cmd = 'c++filt %s' % func_name
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
                func_name_demangled = proc.stdout.readline().decode('ascii')
                kernels_with_lmem.append((func_name_demangled.rstrip(), line))
        else:
            func_name = ''
    print(line)

kernels_with_lmem_sorted = sorted(kernels_with_lmem, key=lambda t:t[0])

print('Kernels with local memory usage:')
for k in kernels_with_lmem_sorted:
    print(k[0])
    print(k[1])
        
