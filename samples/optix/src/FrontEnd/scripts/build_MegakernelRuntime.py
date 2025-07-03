#!/usr/bin/elw python

import optixpy.util

import sys
import tempfile 
import subprocess
import os
import shutil

###############################################################################
#
# main
#
###############################################################################
# GOLDENROD: move this to someplace generic and consider making it
# versatile enough to be used for other builds - just taking a bunch
# of .ll and .cpp files.  It also needs to handle temporary files more
# gracefully.

import argparse
parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument('-b', '--bin-dir',
                    metavar='BIN-DIR',
                    default='',
                    help='Binary directory for LLVM tools' )
parser.add_argument('-s', '--src-dir',
                    metavar='SRC-DIR',
                    default='',
                    help='Source directory for MegakernelES' )
parser.add_argument('-c', '--common-src-dir',
                    metavar='COMMON-SRC-DIR',
                    default='',
                    help='Source directory for Common runtime' )

a = parser.parse_args( sys.argv[1:] )

llvm_as = os.path.join(a.bin_dir, 'llvm-as')
llvm_dis = os.path.join(a.bin_dir, 'llvm-dis')
llvm_link = os.path.join(a.bin_dir, 'llvm-link')
clang = os.path.join(a.bin_dir, 'clang')
opt = os.path.join(a.bin_dir, 'opt')
target = '--target=lwptx64-lwpu-lwca' # GOLDENROD: needs to be lwptx-lwpu-lwca on 64bit builds

print "Compiling Common C++ runtime ..."
subprocess.check_call( [clang, '-DOPTIX_BUILD_RUNTIME=1', '-o', 'CommonRuntime.ll', target, '-fno-unwind-tables', '-fno-sanitize-recover', '-fno-exceptions', '-O3', '-emit-llvm', '-S', os.path.join(a.common_src_dir, 'CommonRuntime.cpp')] )

#print "Assembling Common LLVM runtime ..."
#subprocess.check_call( [llvm_as, '-o', 'CommonRuntime_ll.bc', os.path.join(a.common_src_dir, 'CommonRuntime_ll.ll')] )

print "Compiling MegakernelES C++ runtime ..."
subprocess.check_call( [clang, '-DOPTIX_BUILD_RUNTIME=1', '-o', 'MegakernelRuntime.ll', target, '-fno-unwind-tables', '-fno-sanitize-recover', '-fno-exceptions', '-O3', '-emit-llvm', '-S', os.path.join(a.src_dir, 'MegakernelRuntime.cpp') ] )

print "Linking MegakernelES runtime ..."
subprocess.check_call( [llvm_link, '-o', 'MegakernelRuntime_linked.bc', 'MegakernelRuntime.ll', 'CommonRuntime.ll', os.path.join(a.common_src_dir, 'CommonRuntime_ll.ll') ] )

shutil.copy2( 'MegakernelRuntime_linked.bc', 'MegakernelRuntime_linked_opt.bc' )
#print "Optimizing MegakernelES runtime ..."
#subprocess.check_call( [opt, '-o', 'MegakernelRuntime_linked_opt.bc', '-std-compile-opts', '-always-inline', '-O3', 'MegakernelRuntime_linked.bc' ] )

# For debug only
print "Disassembling ..."
subprocess.check_call( [llvm_dis, '-o=MegakernelRuntime_linked_opt.ll', 'MegakernelRuntime_linked_opt.bc'] )

print 'Done.'
