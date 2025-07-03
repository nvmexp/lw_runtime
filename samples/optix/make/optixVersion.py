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

def main( versionField, headerFile, defineSymbol ):
    versionFile = open( headerFile, "r" )

    for line in versionFile:
        matches = re.search( defineSymbol + "\s+([0-9]+)\s+", line )
        if matches:
            version = int( matches.group( 1 ) )
            major = version/10000;
            minor = ( version - major*10000 )/100
            micro = ( version - major*10000 - minor*100 )
            if versionField.endswith( "MAJOR" ):
                print(versionField + " = " + str( major ))
            elif versionField.endswith( "MINOR" ):
                print(versionField + " = " + str( minor ))
            elif versionField.endswith( "MICRO" ):
                print(versionField + " = " + str( micro ))
            else:
                return 1
            return 0

    return 1

if len( sys.argv ) < 4:
    print("Usage: " + sys.argv[0] + " <versionField> <headerFile> <defineSymboL>")
    sys.exit( 1 )

sys.exit( main( sys.argv[1], sys.argv[2], sys.argv[3] ) )
