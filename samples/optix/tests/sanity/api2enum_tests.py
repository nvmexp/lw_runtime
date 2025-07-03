from __future__ import print_function

#
#  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
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

import argparse
import copy
import os
import sys
import json
import clang.cindex
import collections
from pprint import pprint, pformat

options=None
json_tree=[]
out=None

def file_out_open(options):
    global out
    if options.output_filename != 'stdout':
        out = open(options.output_filename, 'w')
    else:
        out = sys.stdout
    return

def file_out_close():
    if out is not None and out != sys.stdout:
        out.close()
    return

def str_hook(obj):
    newobj={}
    for k,v in obj:
        newk = k.encode('utf-8') if isinstance(k,unicode) else k
        if isinstance(v, collections.Sequence) and not isinstance(v, unicode):
            newv = [ i.encode('utf-8') if isinstance(i,unicode) else i for i in v ]
        else:
            newv= v.encode('utf-8') if isinstance(v,unicode) else v
        newobj[newk]=newv
    return newobj

def parse_options():
    epilog_str='''
Examples:

OptiX 6 Enum AST:
    python ./tests/sanity/api2enum_tests.py --libclang /usr/lib/llvm-5.0/lib print-ast ./include/optix.h

OptiX 7 Enum AST:
    python ./tests/sanity/api2enum_tests.py --libclang /usr/lib/llvm-5.0/lib -D OPTIX_OPTIONAL_FEATURE_OPTIX7 print-ast ./include/optix.h

Create / Update OptiX 6 enum gtests:
    python ./tests/sanity/api2enum_tests.py --libclang /usr/lib/llvm-5.0/lib --output ./tests/Unit/enums/enums.cpp --gtest --gtest_main enums ./include/optix.h

Create / Update OptiX 7 enum gtests:
    python ./tests/sanity/api2enum_tests.py --libclang /usr/lib/llvm-5.0/lib --output ./tests/sanity/enums.cpp -D OPTIX_OPTIONAL_FEATURE_OPTIX7 --gtest enums ./include/optix.h

'''

    parser = argparse.ArgumentParser(
                        description='Parse headers and process Enum declarations for Unit tests',
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        epilog=epilog_str)
    parser.add_argument('COMMAND',help='Action to perform: print-ast, enums. Required')
    parser.add_argument('FILENAMES',nargs='+',help='File(s) to be processed.')
    parser.add_argument('--libclang',dest='libclang',help='Set the path to look for libclang')
    parser.add_argument('--name',dest='name',default='', help='Select only node with name NAME')
    parser.add_argument('-o','--output',dest='output_filename',default='stdout', help='Output to FILENAME. Default is stdout')
    parser.add_argument('--json',dest='json',default=False,action='store_true', help='Output JSON instead of text output')
    parser.add_argument('--gtest',dest='gtest',default=False,action='store_true', help='Output generated gtests instead of text output')
    parser.add_argument('--gtest_main',dest='gtest_main',default=False,action='store_true', help='Add a main function at the end for exelwting gtests')
    parser.add_argument('-D','--defines',dest='defines',default=[],action='append', help='Add extra defines to clang for processing')
 
    # Process command line
    options = parser.parse_args()

    if not options or options.COMMAND=='help':
        parser.print_help()
        exit(1)

    return options

def get_info(options, node, depth=0):
    text = node.spelling or node.displayname

    children = [get_info(options, c, depth+1)
                for c in node.get_children()]

    kind = str(node.kind)[str(node.kind).index('.')+1:]
    if not kind:
        kind = node.kind

    node_text = '{} {} {}'.format( ' '*2*depth, kind, text ) + os.linesep
    children_text = os.linesep.join(children)
    return node_text + children_text

def traverse_ast( options, node, select, process_node, depth=0 ):
    if select is None:
        process_node( node, depth )
    else:
        if select( node ):
            if options.name == '' or options.name == node.spelling:
                process_node( node )
                return
    
    for n in node.get_children():
        traverse_ast(options, n, select, process_node, depth+1 )
    return

def print_node( node, depth ):
    text=node.spelling or node.displayname
    kind = str(node.kind)[str(node.kind).index('.')+1:]
    if not kind:
        kind = node.kind
    print( '{}{}: {}'.format(' '*2*depth,kind,text), file=out )
    return

def process_typedef( node ):
    name = node.spelling or node.displayname
    decl=node.underlying_typedef_type.get_declaration()
    if decl.kind == clang.cindex.LwrsorKind.ENUM_DECL:
        process_enum( decl, name )
    return

def process_enum( node, typename=None ):
    name = node.spelling or node.displayname
    if not name and typename:
        name = typename
    # process enum values
    values = []
    for c in node.get_children():
        if c.kind == clang.cindex.LwrsorKind.ENUM_CONSTANT_DECL:
            t = { 'enum': name, 'value_name': c.spelling, 'actual_value': c.enum_value }
            values.append( t )

    if options.json:
        obj = { 'class': 'enum', 'name': name, 'type': str(node.kind) }
        json_tree.append(obj)
    elif options.gtest:
        prefix='Enum'
        print('class {} : public ::testing::Test {{}};'.format(prefix+name), file=out )
        print('', file=out )

        for v in values:
            print( 'TEST_F( {}, {} )'.format( prefix+name, v['value_name'] ), file=out )
            print('{', file=out )
            actual_value_format_str = '{}' if int(v['actual_value']) < 0 else '0x{:X}'   
            print(('     ASSERT_EQ( {}, ' + actual_value_format_str + ' );').format( v['value_name'],v['actual_value'] ), file=out )
            print('}', file=out )
            print('', file=out )
    else:
        print( '{}:'.format(name ), file=out )
        for v in values:
            print( '  {} = {}'.format( v['value_name'],v['actual_value'] ), file=out )
    return

###############################################################################
# Process command line commmands

# Main processing
options = parse_options()

# process clang config
if options.libclang:
    clang.cindex.Config.set_library_path( options.libclang )

file_out_open(options)

# select command
# note, we will look only at enums defined inside a typedef
selector = {
    'print-ast': None,
    'enums': lambda node: node.kind == clang.cindex.LwrsorKind.TYPEDEF_DECL
}

process = {
    'print-ast': print_node,
    'enums': process_typedef
}

# Process Input files
if options.gtest:
    print('''//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//
//
//

#include <optix.h>
#include <optix_stubs.h>

#include <gtest/gtest.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
 
//---------------------------------------------------------------------------
// This is automatically generated by the api2enum_tests.py script
//---------------------------------------------------------------------------

''', file=out )

for file in options.FILENAMES:
    if not options.json and not options.gtest:
        print( '-'*80, file=out)
    index = clang.cindex.Index.create()
    define_extras=''
    if options.defines:
        define_extras=['-D{}'.format(i) for i in options.defines]
    compile_options=[ '-x','c++','-std=c++11','-D__CODE_GENERATOR__' ]
    compile_options.extend(define_extras)
    tu = index.parse(file,compile_options)
    traverse_ast( options, tu.cursor, selector[ options.COMMAND ], process[ options.COMMAND ] )

# print footer
#print( '''
#int main( int argc, char** argv )
#{
#    testing::InitGoogleTest( &argc, argv );
#    return RUN_ALL_TESTS();
#}
#''', file=out )

# Output file and close
if options.json:
    json.dump( json_tree, out, indent=4, separators=(',',': ') )

if options.gtest_main:
    print('''
int main( int argc, char** argv )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
''' , file=out)

file_out_close()