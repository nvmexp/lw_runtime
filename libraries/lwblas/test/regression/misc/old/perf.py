import sys
import numpy as np

OKGREEN = '\033[92m'
FAIL = '\033[91m'
WARNING = '\033[93m'
ENDC = '\033[0m'

def getKey(tokens):
    first = 0
    while( first < len(tokens) and tokens[first].find(',') == -1 ):
        first+=1
    size = sorted(tokens[first+3].split(','))[1:]
    sizeStr = ""
    for s in size:
        sizeStr += str(s)+","
    key = tokens[first] + " "+ tokens[first+1] + " "+ tokens[first+2] + " "+ sizeStr #modes-c, modes-a, modes-b, extents
    return key

def perfRegression(oldf, newf, col):
    old = {}
    new = {}
    f = open(oldf)
    for l in f:

        tokens = l.split()
        key = getKey(tokens)

        if(l.lower().find("error") == -1):
            old[key] = float(tokens[col])
        else:
            old[key] = -1


    f = open(newf)
    speedups = []
    error = 0
    newWorking = 0
    for l in f:
        tokens = l.split()
        key = getKey(tokens)

        if(l.lower().find("error") == -1):
            flops = float(tokens[col])
        else:
            flops = -1

        if(flops == -1 and old[key] != -1):
            print "%sERROR%s not working any more:"%(FAIL,ENDC), l.rstrip()
            error+=1

        if(flops != -1 and old[key] == -1):
            newWorking += 1

        if(flops != -1 and old[key] != -1):
            speedup = flops / old[key]
            speedups.append(speedup)
            if(speedup < 0.8 and max(flops, old[key])>=1000):
                print "%sPERF WARNING%s: %.3fx "%(WARNING,ENDC,speedup), l.rstrip(), old[key]
            if(speedup > 1.2 ):
                print "%sGOOD PERF%s: %.3fx "%(OKGREEN,ENDC,speedup), l.rstrip(), old[key]

        new[key] = flops

    if(error > 0):
        print "%sBAD%s: %d testcase are no longer working!"%(FAIL,ENDC,error)
    if(newWorking > 0):
        print "%sGOOD%s: %d more TCs are working now."%(OKGREEN, ENDC, newWorking)

    for k in old:
        if(not new.has_key(k)):
            print "%sWARNING%s: new run has not exelwted %s"%(WARNING,ENDC,k)

    print "min: %.3f  avg: %.3f  max: %.3f"%(np.min(speedups), np.mean(speedups), max(speedups))

def analyzeAutotuning(filename):
    f = open(filename)
    speedups = []
    error = 0
    newWorking = 0

    flops = {}
    flopsHeuristic = {}
    flopsBlas = {}

    order = []

    for l in f:
        if( l.find("Warning") != -1 ):
            continue
        tokens = l.split()
        key = tokens[0][6:] + " " + tokens[1][6:] + " " + tokens[2][6:] + " " + tokens[3][7:]#c-ind a-ind b-ind size

        blocking = str(tokens[10:15]) #mc0,mc1,nc0,nc1,largek
        if( not flops.has_key(key) ): 
            order.append(key)
            flops[key] = {}

        isHeuristic = False
        if( int(tokens[10]) == -1):
            isHeuristic = True
        
        if( isHeuristic ):
            flopsHeuristic[key] = float(tokens[8]) #first is always heuristic

        flops[key][blocking] = float(tokens[8])
        flopsBlas[key] = float(tokens[16])


    bestBlockings = {}
    speedups = []
    speedupsBlas = []
    for tc in order:
        maxFlops = 0
        minFlops = 1e9
        avgFlops = 0
        bestBlocking = "NONE"
        for blocking in flops[tc]:
            avgFlops += flops[tc][blocking]
            if(flops[tc][blocking] > maxFlops):
                maxFlops = flops[tc][blocking]
                bestBlocking = blocking
            minFlops = min(minFlops, flops[tc][blocking])
        speedup = maxFlops/flopsHeuristic[tc]
        if(speedup > 2 ):
            print "%.2f"%speedup,blocking,maxFlops, tc
        speedups.append(speedup)
        if( flopsBlas[tc] > 0 ):
            speedupBlas = maxFlops/flopsBlas[tc]
            speedupsBlas.append(speedupBlas)
        if( bestBlockings.has_key(blocking) ):
            bestBlockings[blocking]+=1
        else:
            bestBlockings[blocking]=1
        #print tc, minFlops, avgFlops/len(flops[tc]), maxFlops, speedup, speedupBlas, flopsBlas[tc]
    print "min, avg, max seedup over heuristic: %.2fx %.2fx %.2fx"%(min(speedups),np.mean(np.array(speedups)), max(speedups))
    print "min, avg, max seedup over lwBlas: %.2fx %.2fx %.2fx"%(min(speedupsBlas),np.mean(np.array(speedupsBlas)), max(speedupsBlas))

    bestBlocking = "["
    for blocking in bestBlockings:
        b = map(lambda x: int(x),blocking.replace("'","").replace("]","").replace("[","").split(','))
        if(b[0] != 0):
            bestBlocking += str(tuple(b)) + ", "
        print b, bestBlockings[blocking]
    print bestBlocking + "]"

analyzeAutotuning(sys.argv[1])


if(len(sys.argv) < 3):
    print "Usage: old-file new-file"
    exit(0)

perfRegression(sys.argv[1],sys.argv[2],4)

