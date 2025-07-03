#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      malesiani
#
# Created:     16/04/2014
# Copyright:   (c) malesiani 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import os
import re

DIR = "./ptxInputs"

def printHeader(file):
    header = "// This file contains a list of lines of the following form:\n// [filename] [RT_PROGRAM name]\n//\n"
    file.write(header)

def unmangle(entry):
    matchObj = re.match( r'(_Z\d+)(.*)', entry )
    if not matchObj :
        return entry
    regroups = matchObj.groups()
    unmangled = regroups[1]
    unmangled = unmangled[:-1]
    return unmangled

def main():
    indexFile = open(DIR + "/all_ptx_index.txt", "w")
    printHeader(indexFile)
    # Enumerate each .ptx file
    for filename in os.listdir(DIR):
        if filename == "all_ptx_index.txt" :
            continue
        print "Processing "+filename+"..."
        # Read each line in the file
        entry_regex = re.compile(r'\s*.entry\s*([^\s\n()[]*)')
        with open(DIR+"/"+filename, "r") as fp:
            for line in fp:
                regexresult = entry_regex.search(str(line))
                if regexresult == None:
                    continue
                entry = regexresult.groups()
                if entry != []:
                    entry = entry[0]
                else:
                    continue
                #entry_unmangled = unmangle(entry)
                # Print it to index file
                indexFile.write(filename+" "+entry+"\n")
    indexFile.close
    pass

if __name__ == '__main__':
    main()
