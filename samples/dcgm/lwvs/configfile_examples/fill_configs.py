import fileinput 

whitelist = 'whitelist.txt'
tests = ['quick', 'long']
template = 'template.txt'

f = open(whitelist, 'r')
tuples = list()

for line in f:
    line = line.replace("\n", "")
    tuples.append(line)

f.close()

for test in tests:
    for tuple in tuples:
        splitTuple = tuple.split(", ")
        outFileName = splitTuple[0] + "_" + test + ".conf"
        outFileName = outFileName.replace(" ", "_")
        try:
            outFile = open(outFileName, 'w')
        except IOError as e:
            print "Unable to open %s for writing. Skipping." % outFileName
            continue

        for line in fileinput.input(template):
            if '%DEVICE%' in line:
                outFile.write (line.replace('%DEVICE%', splitTuple[0]))
            elif '%SETNAME%' in line:
                outFile.write (line.replace('%SETNAME%', "All " + splitTuple[0]))
            elif '%ID%' in line:
                outFile.write (line.replace('%ID%', splitTuple[1]))
            elif '%TEST%' in line:
                outFile.write (line.replace('%TEST%', test.capitalize()))
            else:
                outFile.write (line)
        outFile.close()


        
    
