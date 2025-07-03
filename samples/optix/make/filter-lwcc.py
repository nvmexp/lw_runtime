#
#  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
#
#  LWPU Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from LWPU Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#
# filter-lwcc.py [lwcc-command]
#
# Filter LWCC output while still propagating the success/failure of LWCC to the caller.
#
# [lwcc-command] - Full LWCC command-line to run
#

# Use print() function when built as python2
from __future__ import print_function

import os
import re
import subprocess
import sys

errorFilters = [
    "ptxas .+, line .+; warning : Instruction '(shfl|vote)' without '.sync' is deprecated since PTX ISA version 6.0 and will be discontinued in a future PTX ISA version.*",
    "ptxas .+, line .+; warning : Instruction '(shfl|vote)' without '.sync' may produce unpredictable results on sm_70 and later architectures.*",
]

def filterOutput(output, file):
    global errorFilters
    lastLine = None
    for line in output.splitlines():
        filtered = False
        for f in errorFilters:
            if re.search(f, line.decode('utf-8')):
                filtered = True
                break
        if not filtered:
            if line == lastLine:
                continue
            lastLine = line
            print(line, file=file)

# On Linux, we might be ilwoking commands with environment variable prefixes.  We need
# to assimilate these into our environment before we can Popen the actual command.
cmd = sys.argv[1:]
if os.name == "posix":
    while '=' in cmd[0]:
        (name, value) = cmd[0].split("=")
        os.elwiron[name] = value
        cmd = cmd[1:]
    cmd = '"' + '" "'.join(cmd) + '"'
    useShell = True
else:
    useShell = False

lwcc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=useShell)
output = lwcc.communicate()
filterOutput(output[0], sys.stdout)
filterOutput(output[1], sys.stderr)
sys.exit(lwcc.returncode)
