#
#  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
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
import re
import sys

def isReleaseBranch(bldVerFile):
    file = open(bldVerFile, "r")

    for line in file:
        matches = re.search("#define\s+LW_BUILD_BRANCH_VERSION\s+\"rel/", line)
        if matches:
            return 1

    return 0

def branchVersionMajor(bldVerFile):
    file = open(bldVerFile, "r")

    for line in file:
        matches = re.search("#define\s+LW_BUILD_BRANCH\s+r([0-9]+)_", line)
        if matches:
            return int(matches.group(1))

    return 0

def main(bldVerFile):
    if isReleaseBranch(bldVerFile):
        version = branchVersionMajor(bldVerFile)
        if version == 0:
            return 1
    else:
        version = 999

    print("OPTIX_BRANCH_NUMERIC := " + str(version))
    return 0

main(sys.argv[1])
