import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lwtensorUtil import *
import sys
import copy
import glob

benchmarksEW = ["ew300"]
benchmarksTC = ["tc300","largek"]
benchmarksRED = ["red300"]
gpus = ["gv100", "ga100"]
dataTypes = ["hhhh","hhhs","bbbs","bbbb","sssb","sssh","ssst","ssss","ddds","dddd","ccct","cccs","cccc","zzzz","zzzt","zzzs"]
allowedAlgo = ["MAX","GETT","GETT_MAX","TTGT","LWBLAS"]

if (len(sys.argv) < 5):
    print("This script compares the performance of all files --that have tags tag1,tag2,... with those files that share the same tages except that tag1, tag2,... are replaced with tagA,tagB,...-- in the <log folder>")
    print("Usage: python regression.py <log folder> tag1,tag2,... tagA,tagB,... ALGO mustHaveTag1,mustHaveTag2,... ")
    print("ALGO can be MAX,GETT,GETT_MAX,TTGT,LWBLAS")
    print("")
    print("Example: python regression.py ../data/ lwvm34;lwvm70 lwvm70;lwvm70,remat gv100")


def getBenchmark(tags):
    for tag in tags:
        if tag in benchmarksEW:
            return tag
        if tag in benchmarksTC:
            return tag
        if tag in benchmarksRED:
            return tag
    return "unknown benchmark"

def getDataType(tags):
    for tag in tags:
        if tag in dataTypes:
            return tag
    return "unknown type"

def getGPU(tags):
    for tag in tags:
        if tag in gpus:
            return tag
    return "unknown GPU"

def foundAll(tags1, tags2):
    for tag in tags1:
        if (not tag in tags2):
            return False
    return True

def allTagsMatch(tags1, tags2):
    if (len(tags1) != len(tags2)):
        return False
    for tag in tags1:
        if (not tag in tags2):
            return False
    return True

def intersect(l1,l2):
    intersection = []
    for x in l1:
        if x in l2:
            intersection.append(x)
    return intersection

def plot(flops, labels, outputFile, title, text):
    df = pd.DataFrame(flops, columns = labels)
    # sort w.r.t. ref
    df = df.sort_values(by=labels[0]).reset_index(drop=True).reset_index(drop=True).reset_index()

    ax = plt.gca()
    ax.scatter(df['index'].values, df[labels[0]].values, label = labels[0], s=8)
    ax.scatter(df['index'].values, df[labels[1]].values, label = labels[1], s=8)
    ax.set_ylabel("Performance (higher is better)")
    # sort w.r.t. speedup
    ax2 = ax.twinx()
    df = df.sort_values(by=labels[2]).reset_index(drop=True).reset_index(drop=True).reset_index()
    n = len(df['index'].values)
    ax2.scatter([i for i in range(n)], df[labels[2]].values, label = labels[2], color = 'red', s=8)
    ax2.plot([0,n],[0,0], '--')
    #ax2.set_ylim(0,1.4)
    ax2.set_ylabel("log(Speedup)")
    ax.legend(loc="upper left")
    plt.text(len(flops)*0.8, 0.05, text)
    plt.xlabel("#testcase")
    plt.title(title)
    plt.savefig(outputFile, bbox_inches='tight', transparent=False)
    plt.close()
    print(outputFile)

def getOperation(tags):
    for tag in tags:
        if tag in benchmarksEW:
            return "ew"
        if tag in benchmarksTC:
            return "tc"
        if tag in benchmarksRED:
            return "red"
    return "unknown"

tags = {}
files = glob.glob(sys.argv[1] + '/*.log')
for filename in files:
    tags[filename] = filename[:-4].split('/')[-1].split('_')

#compare = [(["lwvm34"],["lwvm70"]), (["lwvm70"], ["lwvm70","remat"])]
algo = sys.argv[4]
if (not algo in allowedAlgo):
    print("Selected algorithm in invalid")
    exit(0)

ref = sys.argv[2].split(',')
compareTo = sys.argv[3].split(',')
mustHaves = []
if (len(sys.argv) >= 6):
    mustHaves = sys.argv[5].split(',')
labelRef = "+".join(ref)
labelCompare = "+".join(compareTo)
labels = [labelRef, labelCompare, "speedup"]

perfAggregated = {"ew":[], "tc":[], "red":[]}
xAggregated = {"ew":[], "tc":[], "red":[]}
for filename in tags:
    if (not foundAll(mustHaves, tags[filename])):
        continue
    outputFile = "_".join(tags[filename]) + "_vs_" + "_".join(compareTo) + ".png"
    operation = getOperation(tags[filename])
    dataType = getDataType(tags[filename])
    gpu = getGPU(tags[filename])
    benchmark = getBenchmark(tags[filename])
    title = gpu + " @ " + dataType + " @ " + benchmark
    if (foundAll(ref, tags[filename])):

        # find all files that share all tags with the current file -- but with the ref tag replaced with all tags in compareTo
        searchTags = copy.deepcopy(tags[filename])
        for tag in ref:
            searchTags.remove(tag)
        for tag in compareTo:
            searchTags.append(tag)
        for f2 in tags:
            if (allTagsMatch(searchTags, tags[f2])):
                y1 = readData(algo, filename)
                y2 = readData(algo, f2)
                intersection = intersect(y1,y2)
                perf = []
                x = []
                for tc in intersection:
                    #speedup = math.log(y2[tc]/y1[tc])
                    speedup = math.log(y2[tc]/y1[tc])
                    perf.append([y1[tc], y2[tc], speedup])
                    #speedup = y2[tc]/y1[tc]#-1 if y2[tc] >= y1[tc] else -y1[tc]/y2[tc]+1
                    perfAggregated[operation].append([y1[tc], y2[tc], speedup])
                    xAggregated[operation].append(tc)
                    #if (y2[tc]/y1[tc] < 0.5):
                    #    print(tc, y2[tc], y1[tc])
                    x.append(tc)
                avg, min, max = quantifySpeedup(x, perf)
                text = "avg:%.2f\nmin:%.2f\nmax: %.2f"%(avg,min,max)
                plot(perf, labels, outputFile, title, text)
                break

for op in perfAggregated:
    if (len(perfAggregated[op]) > 0):
        outputFile = "_".join(ref) + "_vs_" + "_".join(compareTo) + "_%s.png"%(op)
        avg, min, max = quantifySpeedup(xAggregated[op], perfAggregated[op])
        text = "avg:%.2f\nmin:%.2f\nmax: %.2f"%(avg,min,max)
        plot(perfAggregated[op], labels, outputFile, op, text)



