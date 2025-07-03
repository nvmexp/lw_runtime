#!/usr/bin/elw python 

import shutil
import os
import os.path
import sys
import tarfile
import time

sourceDirNames = [
    'common',
    'common/protobuf',
    'common/transport',
    'dcgmi',
    'dcgmlib',
    'dcgmlib/src',
    'modules',
    'modules/config',
    'modules/diag',
    'modules/health',
    'modules/introspect',
    'modules/policy',
    'modules/vgpu'
]

copyExts = ['.c','.cpp','.h', '.proto']

#Files that we should exclude, even if they match the extension
excludeFiles = ['fabricmanager.proto', 'topology.proto', 'dcgm_module_fm_entry_point.h', 'dcgm_module_fm_internal.h' , 'dcgm_module_fm_structs_internal.h']

tgzFilename = "./dcgm_source.tar.gz"
licenseFilename = "./source_code_license.txt"

licenseText = open(licenseFilename).read()

outputTempFolder = "./dcgm_source"

def removeOutputFile():
    if os.path.isfile(tgzFilename):
        os.remove(tgzFilename)

def tarfileFilterFunction(tarInfo):
    tarInfo.mtime = time.time()
    tarInfo.uid = 0
    tarInfo.gid = 0
    tarInfo.uname = "lwpu"
    tarInfo.gname = "lwpu"
    return tarInfo

print("Cleaning up previous runs")
removeOutputFile()

#Recreate our temp folder
if os.path.isdir(outputTempFolder):
    shutil.rmtree(outputTempFolder)
os.mkdir(outputTempFolder)

tarFileObj = tarfile.open(tgzFilename, "w:gz")

sourceInputDir = '../'
for sourceDirName in sourceDirNames:

    print("Working on directory " + sourceDirName)

    createdDir = False

    filenames = os.listdir(sourceInputDir + sourceDirName)

    for filename in filenames:
        #Should we exclude this file from the archive?
        if filename in excludeFiles:
            print("EXCLUDED: " + filename)
            continue

        inputFilename = sourceInputDir + sourceDirName + '/' + filename
        keepFile = False
        for copyExt in copyExts:
            if inputFilename.endswith(copyExt):
                keepFile = True
                break
        
        if not keepFile:
            continue

        print("Keeping file " + inputFilename)

        if not createdDir:
            os.mkdir(outputTempFolder + '/' + sourceDirName)
            createdDir = True

        outputFilename = outputTempFolder + '/' + sourceDirName + '/' + filename

        outputFp = open(outputFilename, "wt")
        outputFp.write(licenseText)
        outputFp.write(open(inputFilename, "rt").read())
        outputFp.close()

        print("Wrote " + outputFilename)

#Write tar file
print("Writing " + tgzFilename)
tarFileObj.add(outputTempFolder, filter=tarfileFilterFunction)
print("Done")

