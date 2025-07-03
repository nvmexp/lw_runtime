#Find all of the function symbols in the passed-in ptx file and append them as variables to 
#our bandwidth_calc_ptx_string.h

ptxFilename = 'bandwidth_calc.ptx'
outFilename = "bandwidth_calc_ptx_string.h"

ptxFp = open(ptxFilename, "rt")
outFp = open(outFilename, "at")

outFp.write("\n\n")

streamTriadFuncs = []

for line in ptxFp.readlines():
    if line.find(".entry") < 0:
        continue

    lineParts = line.split()
    funcName = lineParts[2][0:-1]

    if funcName.find("STREAM_Triad_cleanup") >= 0:
        outFp.write("const char *stream_triad_cleanup_func_name = \"%s\";\n" % funcName)
    elif funcName.find("STREAM_Triad") >= 0:
        streamTriadFuncs.append(funcName)
    elif funcName.find("set_array") >= 0:
        outFp.write("const char *set_array_func_name = \"%s\";\n" % funcName)
    else:
        print "WARNING: Unhandled function: %s" % funcName

if len(streamTriadFuncs) < 1:
    print "WARNING: No STREAM_Triad functions found."
else:
    outFp.write("#define NUM_STREAM_TRIADS %d\n" % len(streamTriadFuncs))
    outFp.write("const char *streamTriadFuncs[NUM_STREAM_TRIADS] = {\n")
    for i, funcName in enumerate(streamTriadFuncs):
        outLine = "    \"%s\"" % funcName
        if i < len(streamTriadFuncs) - 1:
            outLine += ","
        outLine += "\n"
        outFp.write(outLine)
    outFp.write("};\n")

