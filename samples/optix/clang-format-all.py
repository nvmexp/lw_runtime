#!/usr/bin/python


#----------------------------------------------------------------------
# Configure directories and files here
includedTopLevelDirs = [
    'exp',
    'prodlib',
    'src',
    'tests',
    'tools',
]
excludedDirs = [
    'src/FrontEnd/PTX/PTXStitch/',
    'src/Util/jama/',
    'prodlib/bvhtools/',
    'tests/Unit/gmock',
    'tests/Unit/googletest-v1.10',
]
excludedFiles = [
]
excludedGlobs = [
    '*.bin.h',
    '*.bin.cpp',
]
#----------------------------------------------------------------------


import fnmatch
import os
import subprocess


def skipGlob(fullpath, file):
    for glob in excludedGlobs:
        if fnmatch.fnmatch(file, glob):
            if not args.quiet:
                print "SKIPPED " + fullpath + " : glob " + glob
            return True
    return False


def clangformat(dir, files, args):  
    if not args.quiet:
        print dir
    
    numSkipped = 0
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext not in [".cpp", ".h", ".lw", ".hpp", ".c"]:
            continue
        fullpath = os.path.join(dir,file)

        if skipGlob(fullpath, file):
            numSkipped += 1
            continue

        if not os.access(fullpath, os.W_OK):
            if not args.quiet:
                print "SKIPPED " + fullpath + " : write-protected"
            numSkipped += 1
            continue

        exclude = False
        for exFile in excludedFiles:
            exFile = os.path.normpath(exFile)
            if fullpath.endswith(exFile):
                exclude = True
                break
        if exclude:
            if not args.quiet:
                print "EXCLUDED " + fullpath
            continue

        print fullpath
        subprocess.call( [args.bin, "-i", fullpath] )

    return numSkipped
    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-b", "--bin",    help="Specify binary for clang-format.", default="")
parser.add_argument("-q", "--quiet",  help="Print only the files that are formatted.", action="store_true")
args = parser.parse_args()

if not args.bin: 
    # Figure out where clang-format lives. At least on Windows, having the clang
    # binaries in PATH confuses our OptiX cmake project, so we're just assuming
    # it's in the default location. Maybe this can be fixed in the cmake config?
    # I'm assuming that clang-format in PATH works for Linux/Mac -- if it doesn't,
    # then either do something similar as for Windows or let's fix cmake.
    if os.name == "nt":
        # Windows: assume clang-format in default location
        args.bin = "C:\\Program Files\\LLVM\\bin\\clang-format.exe"
    else:
        # Linux+Mac: assume clang-format is in PATH
        args.bin = "clang-format"

numSkipped = 0
for tldir in includedTopLevelDirs:
    for dir, dirs, files in os.walk(tldir):
        # Need to normalize both found path and searched path to let them match.
        # Otherwise if the directory contains a slash it won't work correctly
        # on Windows.
        dir = os.path.normpath(dir)
        exclude = False
        for exDir in excludedDirs:
            exDir = os.path.normpath(exDir)
            if dir.startswith(exDir):
                exclude = True
                break
        if exclude:
            if not args.quiet:
                print "EXCLUDED " + dir
            continue

        numSkipped += clangformat(dir, files, args)
if numSkipped > 0:
    print 'Skipped ' + str(numSkipped) + ' files.'
