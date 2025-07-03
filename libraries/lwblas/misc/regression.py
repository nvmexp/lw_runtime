"""
Compare multiple lwtensor log files with oneanother
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lwtensorUtil import *
import sys

def quantifySpeedup(data, labels):
    for i in range(1,len(data)):
        speedup = []
        for tc in data[0]:
            if (tc in data[i]):
                sp = data[0][tc] / data[i][tc]
                if sp < 0.6:# or sp > 2:
                    print(tc, data[0][tc], data[i][tc], sp)
                speedup.append(data[0][tc] / data[i][tc])
        print("flops[%s]/flops[%s]: avg: %.2f median: %.2f min: %.2f max: %.2f"%(labels[0], labels[i], np.array(speedup).mean(), np.median(np.array(speedup)), min(speedup), max(speedup)))


algo = "MAX"
outputFilename = sys.argv[1]

print("usage: <output>.png data1,label1 [data2,label2] ...")


data = []
labels = []

# read data from files
for i in range(2, len(sys.argv)):
    data.append(readData(algo, sys.argv[i].split(",")[0]))
    labels.append(sys.argv[i].split(",")[1])

quantifySpeedup(data, labels)

# intersect test cases
intersection = data[0].keys()
for i in range(1,len(data)):
    intersection = intersection & data[i].keys()

flops = []
x = []
for tc in intersection:
    flop = []
    for i in range(0,len(data)):
        flop.append(data[i][tc])
    flops.append(flop)
    x.append(tc)


df = pd.DataFrame(flops, columns = labels)
df = df.sort_values(by=labels[0]).reset_index(drop=True).reset_index(drop=True).reset_index()

ax = plt.gca()
for i in range(0,len(data)):
    plt.scatter(df['index'].values, df[labels[i]].values, label = labels[i])
ax.legend(loc="upper left")
plt.savefig(outputFilename, bbox_inches='tight', transparent=False)
plt.close()
