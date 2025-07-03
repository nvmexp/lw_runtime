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
# process-dependencies.py <output-file> <source-file>
#
# <output-file> - Path to write the dependency information
# <source-file> - The source file that is being compiled by LWCC
#
# Process LWCC dependency output to generate empty targets for dependent headers.  This
# script is dependent on the form of output emitted by LWCC for dependency generation.
#
# We don't emit an empty rule for the source file for symmetry with the output of gcc.
#

import os
import re
import sys

def usage():
    print("Usage: python " + sys.argv[0] + ": <output-file> <source-file>")

# Don't write out an empty rule for the source file
def writeDependency( file, sourceFile, dependency ):
    if dependency != sourceFile:
        file.write( dependency + ":\n")

def writeDependencies( file, sourceFile, lines ):
    file.write( "\n" )
    # Write the first dependency from the first line
    firstLine = lines[0].strip()
    matches = re.match( ".* : (.*) \\\\", firstLine )
    if not matches:
        print("Unexpected input in dependency file: " + firstLine)
        raise
    writeDependency( file, sourceFile, matches.group( 1 ) )
    # Write the remeaining dependencies
    for line in lines[1:]:
        writeDependency( file, sourceFile, line.strip()[0:-2] )

def main():
    if len( sys.argv ) != 3:
        usage()
        sys.exit( 1 )

    lines = sys.stdin.readlines()
    output = open( sys.argv[1], "wt" )
    sourceFile = sys.argv[2]
    output.writelines( lines )
    writeDependencies( output, sourceFile, lines )
    output.close()

main()
