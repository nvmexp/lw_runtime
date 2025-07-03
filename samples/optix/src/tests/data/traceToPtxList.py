#-------------------------------------------------------------------------------
# Created:     2015/11/05
#-------------------------------------------------------------------------------
import os
import re
import operator

#ODIR = "d:/optixmark/OptiXMark"
#TDIR = "pixar_rtp_dive_helmet_siggraph2015"
#TDIR = "3dexcite_stellar_sitz"
ODIR = "f:"
TDIR = "aventador_31.1.0_callable_lights"

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
    traceFile = open(ODIR + "/" + TDIR + "/trace.oac", "r")
    # rtProgramCreateFromPTXFile( 0x7f36f0c44800, /host/devel/suntory/scinst/fedora-gcc64-opt/lava/lwca/RtpInteractiveRenderer.ptx, rtpInteractiveRendererRayGen, 0x7f36f18ada38 )
    #entry_regex = re.compile(r'\s*rtProgramCreateFromPTX[.]*\(')#(\s+), (\s+), (\s+), (\s+)\)')
    entry_regex = re.compile('rtProgramCreateFromPTX.*\(\s+(.+),\s+(.+),\s+(.+),\s+(.+)\s+\)')
    file_regex = re.compile('\s*file\s*=\s*(.+)\s*')
    stuff = {}
    program_found = False
    name = ""
    for line in traceFile:
        if not program_found:
            regexresult = entry_regex.search(str(line))
            if regexresult == None:
                continue
            name = regexresult.groups()[2]
            program_found = True
            print name
        else:
            file = file_regex.search(str(line)).groups()[0]
            print file
            if file not in stuff:
                stuff[file] = [name]
            else:
                if name not in stuff[file]:
                    stuff[file].append(name)
            program_found = False
    # print stuff
    for file in sorted(stuff.keys()):
        a = file.split(".")
        print "cp \'%s/%s/%s\' ptxInputs/%s.%s.ptx" % (ODIR,TDIR,file,TDIR,a[2]) 
    for file,names in sorted(stuff.items(), key=operator.itemgetter(0)):
        a = file.split(".")
        for name in names:
            print "%s.%s.ptx %s" % (TDIR,a[2], name)
    traceFile.close
    pass

if __name__ == '__main__':
    main()
