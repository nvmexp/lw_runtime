import os
import numpy as np
import math

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

def getFlopsLWT(data):
    """
    Returns lwtensor flops based on its algo: heur, gett, or the maximum of all algos
    """

    lwtensor_lwte = getFlops(data, -5)
    lwtensor_heur = getFlops(data, -1)
    lwtensor_ttgt = getFlops(data, -2)
    lwtensor_max  = getFlops(data, "MAX")
    lwtensor_gett = getFlops(data, -4)
    lwtensor_gett_max = getFlops(data, 1)

    return lwtensor_heur, lwtensor_max, lwtensor_gett, lwtensor_gett_max, lwtensor_ttgt, lwtensor_lwte

   
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

def readFlops(filename):
    if (not os.path.isfile(filename)):
        print("error: %s doesn't exit"%(filename))
        exit(-1)
    flops = {}
    f = open(filename,"r")
    memcpy = {}
    lwblasFlops = {}
    problemSize = {}

    tcOrder = []
    for l in f:
        if l.find("lwTENSOR:") == -1:
            continue
        extent = findAndSplit("-extent", l, [',','='])
        if( len(extent) == 0 ):
            print("WARNING: not processed:",l)
        operation = " -Rcontraction"
        if (l.find("-Relementwise") != -1):
            operation = " -Relementwise -permute"
        modeA = findAndSplit("-modeA", l, [','])
        strideA = findAndSplit("-strideA", l, [','])
        modeB = findAndSplit("-modeB", l, [','])
        strideB = findAndSplit("-strideB", l, [','])
        modeC = findAndSplit("-modeC", l, [','])
        strideC = findAndSplit("-strideC", l, [','])
        modeM = intersect(modeA, modeC)
        modeN = intersect(modeB, modeC)
        modeK = intersect(modeA, modeB)
        modeL = intersect(modeK, modeC)
        for mode in modeL:
            modeK.remove(mode);
            modeN.remove(mode);
            modeM.remove(mode);
        m = product(modeM, extent)
        n = product(modeN, extent)
        k = product(modeK, extent)
        L = product(modeL, extent)
        flop = findAndSplit("lwTENSOR:", l, [])
        lwblasFlop = findAndSplit("lwBLAS:", l, [])
        algo = findAndSplit("-algo", l, [])
        if( algo == [] ):
            algo = ALGO_NOT_APPLICABLE
        else:
            algo = int(algo)
        if(algo == 1000):
            continue #skip max algo
        key = getHashKey( extent, modeA, modeB, modeC, strideA, strideB, strideC )
        key += operation
        if l.find("memcpy:") != -1:
            memcpy[key] = findAndSplit("memcpy:", l, [])
        problemSize[key] = (m,n,k,L)
        if (lwblasFlop != 0):
            lwblasFlops[key] = lwblasFlop
        if( key in flops ):
            flops[key].append((algo, flop))
        else:
            flops[key] = [ (algo, flop) ]
        if( not key in tcOrder):
            tcOrder.append(key) # to keep track of the original order of the measurements

    lwtensor_heur, lwtensor_max, lwtensor_gett, lwtensor_gett_max, lwtensor_ttgt, lwtensor_lwte  = getFlopsLWT(flops)
    return lwtensor_heur, lwtensor_max, lwtensor_gett, lwtensor_gett_max, lwtensor_ttgt, lwtensor_lwte, lwblasFlops, problemSize, memcpy, tcOrder


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

def findAndSplit(keyword, line, splits = []):
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
            nonEmptyTokens = []
            for tok in tokens:
                if (tok != ''):
                    nonEmptyTokens.append(tok)
            return nonEmptyTokens 

def getHashKey( extent, modeA, modeB, modeC, strideA = [], strideB = [], strideC = []):
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
    if (len(modeB) > 0):
        key += " -modeB"
        for a in modeB:
            key += a + ","
    key += " -modeC"
    for a in modeC:
        key += a + ","
    key += " -strideA"
    for a in strideA:
        key += a + ","
    if (len(modeB) > 0):
        key += " -strideB"
        for a in strideB:
            key += a + ","
    key += " -strideC"
    for a in strideC:
        key += a + ","
    return key


def strToBool(val):
    """
    colwerts a string into a bool
    """
    val = val.lower()
    if (val == "true"):
        return 1
    elif (val == "false"):
        return 0
    else:
        print("Error: unknown string")
        exit(-1)

def lwdaToType(typeA):
    if(typeA == "LWDA_R_32I"):
        return "int32_t"
    if(typeA == "LWDA_R_32U"):
        return "uint32_t"
    if(typeA == "LWDA_R_8U"):
        return "uint8_t"
    if(typeA == "LWDA_R_8I"):
        return "int8_t"
    if(typeA == "LWDA_R_16F"):
        return "lwtlass::half_t"
    if(typeA == "LWDA_R_32F"):
        return "float"
    if(typeA == "LWDA_R_64F"):
      return "double"
    if(typeA == "LWDA_C_32F"):
      return "lwtlass::complex<float>"
    if(typeA == "LWDA_C_64F"):
        return "lwtlass::complex<double>"
    if(typeA == "LWDA_C_32F"):
      return "lwtlass::complex<float>"
    if(typeA == "LWDA_C_64F"):
        return "lwtlass::complex<double>"
    if(typeA == "LWDA_R_16BF"):
        return "lwtlass::bfloat16_t"
    return "x"

def typeToChar(typeA):
    if(typeA == "int32_t"):
        return "i"
    if(typeA == "uint32_t"):
        return "u"
    if(typeA == "uint8_t"):
        return "u8"
    if(typeA == "int8_t"):
        return "i8"
    if(typeA == "lwtlass::half_t" or typeA == "half"):
        return "h"
    if(typeA == "float"):
        return "s"
    if(typeA == "double"):
      return "d"
    if(typeA == "lwtlass::complex<tfloat32_t>"):
      return "r"
    if(typeA == "lwtlass::complex<float>" or typeA == "lwComplex"):
      return "c"
    if(typeA == "lwtlass::complex<double>" or typeA == "lwDoubleComplex"):
      return "z"
    if(typeA == "lwtlass::bfloat16_t" or typeA == "BFloat16"):
      return "b"
    if(typeA == "lwtlass::tfloat32_t"):
      return "t"
    return "x"

def typeToLWTENSOR(typeA):
    if(typeA == "int32_t"):
        return "LWTENSOR_COMPUTE_32I"
    if(typeA == "uint32_t"):
        return "LWTENSOR_COMPUTE_32U"
    if(typeA == "uint8_t"):
        return "LWTENSOR_COMPUTE_8U"
    if(typeA == "int8_t"):
        return "LWTENSOR_COMPUTE_8I"
    if(typeA == "lwtlass::half_t"):
        return "LWTENSOR_COMPUTE_16F"
    if(typeA == "float"):
        return "LWTENSOR_COMPUTE_32F"
    if(typeA == "double"):
        return "LWTENSOR_COMPUTE_64F"
    if(typeA == "lwtlass::complex<float>"):
        return "LWTENSOR_COMPUTE_32F"
    if(typeA == "lwtlass::complex<double>"):
        return "LWTENSOR_COMPUTE_64F"
    if(typeA == "lwtlass::bfloat16_t"):
        return "LWTENSOR_COMPUTE_16BF"
    if(typeA == "lwtlass::tfloat32_t"):
        return "LWTENSOR_COMPUTE_TF32"
    if(typeA == "lwtlass::complex<tfloat32_t>"):
        return "LWTENSOR_COMPUTE_TF32"
    return "x"

def typeToCompute(typeA):
    if(typeA == "int32_t"):
        return "LWTENSOR_COMPUTE_32I"
    if(typeA == "uint32_t"):
        return "LWTENSOR_COMPUTE_32U"
    if(typeA == "uint8_t"):
        return "LWTENSOR_COMPUTE_8U"
    if(typeA == "int8_t"):
        return "LWTENSOR_COMPUTE_8I"
    if(typeA == "lwtlass::half_t"):
        return "LWTENSOR_COMPUTE_16F"
    if(typeA == "float"):
        return "LWTENSOR_COMPUTE_32F"
    if(typeA == "double"):
        return "LWTENSOR_COMPUTE_64F"
    if(typeA == "lwtlass::complex<float>"):
        return "LWTENSOR_COMPUTE_32F"
    if(typeA == "lwtlass::complex<double>"):
        return "LWTENSOR_COMPUTE_64F"
    if(typeA == "lwtlass::complex<tfloat32_t>"):
        return "LWTENSOR_COMPUTE_TF32"
    if(typeA == "lwtlass::bfloat16_t"):
        return "LWTENSOR_COMPUTE_16BF"
    if(typeA == "lwtlass::tfloat32_t"):
        return "LWTENSOR_COMPUTE_TF32"
    return "x"


def typeToLWDA(typeA):
    if(typeA == "int32_t"):
        return "LWDA_R_32I"
    if(typeA == "uint32_t"):
        return "LWDA_R_32U"
    if(typeA == "uint8_t"):
        return "LWDA_R_8U"
    if(typeA == "int8_t"):
        return "LWDA_R_8I"
    if(typeA == "lwtlass::half_t"):
        return "LWDA_R_16F"
    if(typeA == "float"):
        return "LWDA_R_32F"
    if(typeA == "double"):
        return "LWDA_R_64F"
    if(typeA == "lwtlass::complex<float>"):
        return "LWDA_C_32F"
    if(typeA == "lwtlass::complex<double>"):
        return "LWDA_C_64F"
    if(typeA == "lwtlass::complex<tfloat32_t>"):
        return "LWDA_C_TF32"
    if(typeA == "lwtlass::bfloat16_t"):
        return "LWDA_R_16BF"
    if(typeA == "lwtlass::tfloat32_t"):
        return "LWDA_R_TF32"
    return "x"

def getDataTypeSize(typeA):
    if(typeA == "int32_t"):
        return 4
    if(typeA == "uint32_t"):
        return 4
    if(typeA == "uint8"):
        return 1
    if(typeA == "int8_t"):
        return 1
    if(typeA == "lwtlass::half_t"):
        return 2
    if(typeA == "float"):
        return 4
    if(typeA == "double"):
        return 8
    if(typeA == "lwtlass::complex<float>"):
        return 8
    if(typeA == "lwtlass::complex<tfloat32_t>"):
        return 8
    if(typeA == "lwtlass::complex<double>"):
        return 16
    if(typeA == "lwtlass::bfloat16_t"):
        return 2
    if(typeA == "lwtlass::tfloat32_t"):
        return 4
    return "x"

def typeToNative(typeA):
    if(typeA == "lwtlass::half_t"):
        return "__half"
    elif(typeA == "float"):
        return typeA
    elif(typeA == "double"):
        return typeA
    elif(typeA == "lwtlass::complex<float>"):
        return "float2"
    elif(typeA == "lwtlass::complex<double>"):
        return "double2"
    else:
        return "x"


def getVectorization(typeA):
    if(typeA == "lwtlass::half_t"):
        return 8
    elif(typeA == "float"):
        return 4
    elif(typeA == "double"):
        return 2
    elif(typeA == "lwtlass::complex<float>"):
        return 2
    elif(typeA == "lwtlass::complex<double>"):
        return 1
    else:
        return "x"

def readData(algo, filename):
    lwtensor_heur, lwtensor_max, lwtensor_gett, lwtensor_gett_max, lwtensor_ttgt, lwtensor_lwte, lwblas_ref, problemSize, memcpy, tcOrder = readFlops(filename)

    data = lwtensor_heur
    if (algo == "MAX"):
        data = lwtensor_max
    elif (algo == "GETT"):
        data = lwtensor_gett
    elif (algo == "GETT_MAX"):
        data = lwtensor_gett_max
    elif (algo == "TTGT"):
        data = lwtensor_ttgt
    elif (algo == "LWTE"):
        data = lwtensor_lwte
    elif (algo == "lwblas"):
        data = lwtensor_lwblas
    else:
        print("unknown algo")
        exit(-1)
    return data

def quantifySpeedup(x, perf):
    speedup = []
    for i in range(len(perf)):
        ref = perf[i][0]
        new = perf[i][1]
        sp = perf[i][2]
        if math.exp(sp) < 0.6:# or sp > 2:
            print(x[i], ref, new, math.exp(sp))
        speedup.append(sp)
    avg   = math.exp(np.median(np.array(speedup)))
    mymin = math.exp(min(speedup))
    mymax = math.exp(max(speedup))
    print("avg: %.2f median: %.2f min: %.2f max: %.2f"%(np.array(speedup).mean(),
        avg, mymin, mymax))
    return avg, mymin, mymax



