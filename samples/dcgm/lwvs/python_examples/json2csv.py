import json
import csv
import sys
import getopt
import string

inputfile = ''
outputfile = ''
keys = ''

def printUsage():
    print str(sys.argv[0]) + ' [-i <inputfile>] [-o <outputfile>] -k <keys (comma separated)>'

def parseArgs(argv):
    global inputfile
    global outputfile
    global keys
    try:
        opts, args = getopt.getopt(argv,"hi:o:k:",["ifile=","ofile=","keys="])
    except getopt.GetoptError:
        printUsage()
        sys.exit(2)
    keyArg = False
    for opt, arg in opts:
        if opt == '-h':
            printUsage()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-k", "--keys"):
            keys = arg
            keyArg = True
    if not keyArg:
        printUsage()
        sys.exit()

def cleanup():
    global jsonFile
    global outHandle

    if jsonFile is not sys.stdin:
        jsonFile.close()

    if outHandle is not sys.stdout:
        outHandle.close()

if __name__ == "__main__":
   parseArgs(sys.argv[1:])

jsonFile  = open(inputfile) if inputfile is not "" else sys.stdin
jsonData = json.load(jsonFile)

outHandle = open(outputfile, 'wb') if outputfile is not ""  else sys.stdout 
csvWriter = csv.writer(outHandle, quotechar='"', quoting=csv.QUOTE_ALL, delimiter=",")

keyList = keys.split(",")

gpusData = jsonData["gpus"]

header = ["GPU#", "time"]
for key in keyList:
    header.append(str(key))

csvWriter.writerow(header)

for gpu in gpusData:
    try:
        key = keyList[0]
        for i in range(len(gpusData[gpu][key])):
            row = [gpu]
            row.append(str(i))
            for key in keyList:
                entry = gpusData[gpu][key][i]
                row.append(str(entry["value"]))
            csvWriter.writerow(row)

    except KeyError:
        print 'Key \"' + key + '\" not found in JSON file.'

cleanup()


