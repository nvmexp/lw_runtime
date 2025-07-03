import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def findAndSplit(keyword, line, splits):
    """ finds the keyword in line and splits 1) ',' then 2) '=' and asigns it to the returnd dict."""
    pos = line.find(keyword)
    if( pos == -1 ):
        return []
    pos += len(keyword)
    if( len(splits) == 0 ):
        return float(line[pos:].split()[0])

    tokens = line[pos:].split()[0].split(splits[0])
    if( len(splits) == 2 ):
        extent = {}
        for token in tokens:
            tk = token.split(splits[1])
            if(len(tk) < 2):
                continue
            extent[tk[0]] = int(tk[1])
        return extent
    else:
        #fix for 0-order tensors
        if(tokens[0].startswith("-")):
            return []
        else:
            return tokens

def getHashKey( extent, modeA, modeB, modeC ):
    extent_sorted = []
    for ex in extent:
        extent_sorted.append((ex,extent[ex]))
    extent_sorted.sort()
    key = "-extent"
    for ex in extent_sorted:
        key+= ex[0] + "="+ str(ex[1]) +","
    key += " -modeA"
    for a in modeA:
        key += a + ","
    key += " -modeB"
    for a in modeB:
        key += a + ","
    key += " -modeC"
    for a in modeC:
        key += a + ","
    return key

ALGO_NOT_APPLICABLE = -1000000

def getFlops( flops, algo ):
    """ flops is a dict as provided by readFlops() """
    retFlops = {}
    for key in flops:
        retFlops[key] = 0 
        for AL, FL in flops[key]:
            if( algo == "MAX" ):
                retFlops[key] = max(retFlops[key], FL)
            elif( algo < 0 ):
                if( AL == algo ):
                    retFlops[key] = FL
                    break
            else: #algo >=0
                if( AL >= 0 ): #search for maximum GETT
                    retFlops[key] = max(retFlops[key], FL)

    return retFlops
    
def intersect(l1, l2):
    l = []
    for a in l1:
        if( a in l2):
            l.append(a)
    return l
def product(modes , extent):
    total = 1
    for m in modes:
        total *= extent[m]
    return total

def readCounts(filename):
    f = open(filename)
    counts = {}
    for l in f:
        extent = findAndSplit("-extent", l, [',','='])
        modeA = findAndSplit("-modeA", l, [','])
        modeB = findAndSplit("-modeB", l, [','])
        modeC = findAndSplit("-modeC", l, [','])
        key = getHashKey( extent, modeA, modeB, modeC )
        counts[key] = int(l.split()[0])
    return counts

def readFlops(filename, algoId):
    flops = {}
    f = open(filename,"r")
    lwblasFlops = {}
    problemSize = {}
    for l in f:
        if l.find("GFLOPS") == -1:
            continue
        extent = findAndSplit("-extent", l, [',','='])
        if( len(extent) == 0 ):
            print(f"WARNING: not processed: {l}")
        modeA = findAndSplit("-modeA", l, [','])
        modeB = findAndSplit("-modeB", l, [','])
        modeC = findAndSplit("-modeC", l, [','])
        modeM = intersect(modeA, modeC)
        modeN = intersect(modeB, modeC)
        modeK = intersect(modeA, modeB)
        m = product(modeM, extent)
        n = product(modeN, extent)
        k = product(modeK, extent)
        flop = findAndSplit(algoId, l, [])
        lwblasFlop = findAndSplit("lwBLAS:", l, [])
        algo = findAndSplit("-algo", l, [])
        if( algo == [] ):
            algo = ALGO_NOT_APPLICABLE
        else:
            algo = int(algo)
        if( algo == -3 or algo == -5 ):
            continue #skip tensor cores
        key = getHashKey( extent, modeA, modeB, modeC )
        problemSize[key] = (m,n,k)
        lwblasFlops[key] = lwblasFlop
        if( key in flops ):
            flops[key].append((algo, flop))
        else:
            if( len(flops) >= 1000 ):
                break
            flops[key] = [ (algo, flop) ]
    return flops, lwblasFlops, problemSize

def plot( filename, x_axis, data, sortAccordingTo, xlabel, ylabel):

    testCaseOrder = data[sortAccordingTo][0]
    testCasesAsc = [ (key, testCaseOrder[key]) for key in testCaseOrder ]
    testCasesAsc.sort(key=lambda x: x[1]) # sort w.r.t. flops
    testCaseOrder = [ x[0] for x in testCasesAsc ]

    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_facecolor((248/256., 248/256., 248/256.))
    ax.set_ylabel(ylabel,fontsize=22)
    #plt.xlim((0,len(testCaseOrder)))

    ymax = 0

    ax.plot([0,1,2],[0,1,2])
    for (dat, label, marker, color) in data:
        # sort dat according to testCaseOrder
        flops = []
        for tc in testCaseOrder:
            flops.append(dat[tc])
        #print "#",label
        #print flops 

        ymax = max(max(flops), ymax)
        ax.plot(x_axis, flops, label=label, color=color, marker=marker, lw = 0, clip_on=False, zorder=10)

    plt.ylim((0,ymax*1.03))

    ax.set_xlabel(xlabel,fontsize=22)
    for item in ax.get_xticklabels():
        item.set_fontsize(22)
    for item in ax.get_yticklabels():
        item.set_fontsize(24)
    #ax.legend( loc =loc, numpoints = 1, markerscale=2., handletextpad=0.05)

    ldgn = ax.legend(loc='lower right', handletextpad=0.2, columnspacing=0.2,prop={'size': 14})
    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()


def quantifySpeedup(label, base, accel, ratio = 0.7):
    minSpeedup = 10000000
    maxSpeedup = 0
    avgSpeedup = 0
    notApplicable = 0
    count = 0
    for tc in base:
        try:
            if( accel[tc] > 0 ):
                speedup = accel[tc] / base[tc]
                #if (speedup < ratio ):
                #    print "./test/lwtensorTest", tc, "-Pad -Pbd -Pcd -Pcompd -fastVerify -Rcontraction", accel[tc], base[tc] 
                maxSpeedup = max( maxSpeedup, speedup)
                minSpeedup = min( minSpeedup, speedup)
                avgSpeedup += speedup
                count += 1
            else:
                print(f"../lwtensorTest {tc} -Pad -Pbd -Pcd -Pcompd -fastVerify -Rcontraction")#,accel[tc], base[tc], 
                notApplicable += 1
        except:
            pass#print "XX", accel[tc], base[tc], tc

    print(f"{label} Speedups: min: {minSpeedup:.2f}x avg: {avgSpeedup/count:.2f}x max: {maxSpeedup:.2f}x")
    if(notApplicable > 0 ):
        print(f"WARNING {label} was not applicable in {notApplicable} out of {len(base)} cases")

def getFlopsLWT(data, lwblasFlops, problemSize):
    counts = readCounts("peps_nx8_d4_chi8_counts")
    lwtensor_heur = getFlops(data, -1)
    lwtensor_max  = getFlops(data, "MAX")
    lwtensor_gett = getFlops(data, 1)
    quantifySpeedup("lwTENSOR(heur) vs lwBLAS", lwblasFlops, lwtensor_heur)
    quantifySpeedup("lwTENSOR(GETT) vs lwBLAS", lwblasFlops, lwtensor_gett)
    quantifySpeedup("lwTENSOR(max) vs lwBLAS", lwblasFlops, lwtensor_max)

    time = []
    for key in counts:
        gflopsTotal = 2. * np.prod(problemSize[key]) / 1e9
        time.append((key, gflopsTotal / lwtensor_max[key], counts[key]))
    time = sorted(time, key=lambda t:t[1]*t[2], reverse=True) #sort contractions w.r.t. their total time
    return lwtensor_heur, lwtensor_max, lwtensor_gett, time

def createPlots():
    lwtensorData_gv100, lwblasFlops_gv100, problemSize = readFlops("peps_nx8_d4_chi8_gv100.dat", "lwTENSOR:") # local GV100
    lwtensorData_ga100, lwblasFlops_ga100, problemSize = readFlops("peps_nx8_d4_chi8_ga100.dat", "lwTENSOR:") # local GA100

    lwtensor_gv100_heur, lwtensor_gv100_max, lwtensor_gv100_gett, time_gv100 = getFlopsLWT(lwtensorData_gv100, lwblasFlops_gv100, problemSize)
    lwtensor_ga100_heur, lwtensor_ga100_max, lwtensor_ga100_gett, time_ga100 = getFlopsLWT(lwtensorData_ga100, lwblasFlops_ga100, problemSize)

    totalTimeGV100 = 0
    for tc,t,c in time_gv100:
        totalTimeGV100 += t * c
        print("%s: %.2e sec; %dx exelwted; total time: %.2e"%( tc,t,c, t * c))
    totalTimeGA100 = 0
    for tc,t,c in time_ga100:
        totalTimeGA100 += t * c
    
    print("Speedup:",totalTimeGV100 / totalTimeGA100 )

    #speedup = []
    #for tc in lwtensor_gv100_max:
    #    print(tc, lwtensor_ga100_max[tc]/ lwtensor_gv100_max[tc])

    #data = [ (lwtensor_max, "lwTENSOR (GV100)", '^', "#31a354"),
    #         (lwblasFlops, "lwBLAS (GV100)", '^', "#fc8d59"),
    #         ]
    #plot(f"{filename}.pdf", x_axis, data, 0, "m = n = k", "GFLOPS/s")

createPlots()
