#!/usr/bin/elw python2
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

# This script emulates cmake's configure_file() command minimally for the needs of configuring
# files in OptiX.
#
# Usage: $0 <list-vars|configure> [variable-definitions...] input-file
#
# The first argument selects the mode of operation for the script:
#   list-vars       Process the input file and produce a list of referenced variable names.
#   configure       Process the input file and substitute referenced variables from the supplied definitions.
#
# Variable definitions are NAME=VALUE pairs; the list may be empty.  A single input file
# is processed and the result is written to stdout.
#
# This script supports the #cmakedefine construct and the @ONLY variable substitution of
# cmake's configure_file() command, see https://cmake.org/cmake/help/v3.10/command/configure_file.html
#
# Any text in the input file matching @VAR@ is replaced with the variable value as
# supplied on the command-line.  If a variable is not defined, it will be replaced with nothing.
#
# Input file lines of the form "#cmakedefine VAR ..." will be replaced with either "#define VAR ..."
# or /* #undef VAR */ depending on whether VAR is set.
#

import sys, re, os

cmakeDefineRegEx = re.compile( '^\\s*#\\s*cmakedefine(?P<ZEROONE>(01)?)\\s+(?P<SYMBOL>[^\\s]+)(?P<REST>(\\s+.*)?)$' )
cmakeSymbolRegEx = re.compile( '@(?P<SYMBOL>[A-Za-z0-9_]+)@' )

variables = None
file = None

def isTrue( symbol ):
    # CMake considers '1', 'on', 'yes', 'true' and 'y' to be TRUE; otherwise FALSE.
    global variables
    return variables.get( symbol, '' ).lower() in ['1', 'on', 'yes', 'true', 'y']

def usage():
    sys.stderr.write( "Usage: " + sys.argv[0] + " <list-vars|configure> [variable-definitions...] input-file\n" )
    sys.exit( 1 )

def listVariables():
    global variables, file
    try:
        input = open( file, 'rt' )
    except:
        print( "Could not open {} for dump".format( file ) )

    for l in input.read().splitlines():
        left = 0
        # handle lines with cmakedefine
        match = cmakeDefineRegEx.match( l )
        if match:
            sys.stdout.write( '%s\n' % match.group( 'SYMBOL' ) )
            if match.group( 'ZEROONE' ):
                l = '' # no further processing
            else:
                l = match.group( 'REST' ) # process the rest of the line
        # handle rest, expand @SYMBOL@
        for m in cmakeSymbolRegEx.finditer( l ):
            sys.stdout.write( '%s\n' % m.group( 'SYMBOL' ) )
    input.close()


def configureFile():
    global variables, file
    try:
        input = open( file, 'rt' )
    except:
        print( "Could not open {} for configure".format( file ) )

    for l in input.read().splitlines():
        line = []
        left = 0
        # handle lines with cmakedefine
        match = cmakeDefineRegEx.match( l )
        if match:
            if match.group( 'ZEROONE' ):
                line.append( '#define %s %s' % (
                    match.group( 'SYMBOL' ),
                    {True : '1', False : '0'}[ isTrue( match.group( 'SYMBOL' ) ) ] ) )
                l = '' # no further processing
            else:
                if isTrue( match.group( 'SYMBOL' ) ):
                    line.append( '#define %s ' % match.group( 'SYMBOL' ) )
                    l = match.group( 'REST' ) # process the rest of the line
                else:
                    line.append( '/* #undef %s */' % match.group( 'SYMBOL' ) )
                    l = '' # no further processing
        # handle rest, expand @SYMBOL@
        for m in cmakeSymbolRegEx.finditer( l ):
            line.append( l[left:m.start()] )
            line.append( variables.get( m.group( 'SYMBOL' ), '' ) )
            left = m.end()
        line.append( l[left:] )
        line.append( '\n' )
        sys.stdout.write( ''.join( line ) )
    input.close()

if len( sys.argv ) < 2:
    usage()

if not sys.argv[1].lower() in [ 'list-vars', 'configure' ]:
    usage()

variables = dict( map( lambda v: v.split( '=', 1 ), sys.argv[2:len( sys.argv ) - 1] ) )
file = sys.argv[-1]
if not file.endswith( ".in" ):
    print >> sys.stderr, "Input file '" + file + "' is not a template."
    usage()

if sys.argv[1].lower() == 'list-vars':
    listVariables()
else:
    configureFile()
