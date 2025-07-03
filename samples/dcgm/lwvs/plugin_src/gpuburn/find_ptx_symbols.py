# Find all of the function symbols in the passed-in ptx file and append them as variables to 
# the passed in output file
import sys

if len(sys.argv) < 3:
    print "USAGE: find_ptx_symbols.py <input.ptx> <output.h>\nThere must be two arguments supplied to this script"
    sys.exit(1)

ptxFilename = sys.argv[1]
outFilename = sys.argv[2]

ptxFp = open(ptxFilename, "rt")
outFp = open(outFilename, "at")

outFp.write("\n\n")

for line in ptxFp.readlines():
    if line.find(".entry") < 0:
        continue

    lineParts = line.split()
    funcName = lineParts[2][0:-1]

    outFp.write("const char *%s_func_name = \"%s\";\n" % (funcName, funcName))


