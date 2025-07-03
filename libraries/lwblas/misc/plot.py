"""
This script combines multiple inputfiles (.csv) into a unified plot
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

import myColors

parser = argparse.ArgumentParser(description='Create plot from csv files.')
parser.add_argument('--output', help="Output name of the generated plot.", type=str, required=True)
parser.add_argument('--sortBy', help="Sort data according to this label (must match a column of any of the csv files).", type=str, required=False)
parser.add_argument('--yLabel', help="y-axis label.", default = "GFlops/s", type=str, required=False)
parser.add_argument('--xLabel', help="x-axis label.", default = "Testcase", type=str, required=False)
parser.add_argument('--hideTicksX', help="hides all x-axis ticks", action='store_true', default = False, required=False)
parser.add_argument('--useGrid', help="Uses a y-grid", action='store_true', default = False, required=False)
parser.add_argument('--speedup', help="Denotes that this should become a speedup plot where the reference must be denoted by sortBy", action='store_true', default = False, required=False)
parser.add_argument('--yMax', help="maximal y-value", type=int, required=False)
parser.add_argument('--xLimitUpper', help="Limits the numer of test cases that are plotted\
        to the minimum numer of test cases among all columns (this is useful if tests are\
        failing).", action='store_true', default=False, required=False)
parser.add_argument('inputs', help="Output name of the generated plot.", type=str, nargs='+')
args = parser.parse_args()

specialColors = {
        "tblis (fp64)" : myColors.intel,
        }

# combine files into a single data fram
dfs = []
for fname in args.inputs:
    if (not os.path.isfile(fname)):
        print(f"error: {fname} doesn't exit")
        exit(-1)
    df = pd.read_csv(fname)
    dfs.append(df)

# plot
ax = plt.gca()
minNumTestCases = 100000000000
for df in dfs:
    for col in reversed(df.columns):
        minNumTestCases = min(minNumTestCases,len(df[col]))
ref = []
if (args.speedup):
    for df in dfs:
        if (args.sortBy in df.columns):
            ref = df[args.sortBy]
            break


idx = 0
idx_bar = 0
for df in dfs:
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    if (not args.speedup and args.sortBy):
        df = df.sort_values(by=[args.sortBy]).reset_index(drop=True)
    for col in reversed(df.columns):
        newDf = df[col]
        if (args.speedup and col == args.sortBy):
            continue
            
        elif (args.speedup and col != args.sortBy):
            df[col] = ref / newDf
            newDf = df[col]

        # limit number of results
        lim = len(df[col])
        if (args.xLimitUpper):
            lim = minNumTestCases

        # select color
        color = myColors.qualitative1[idx]
        if (col in specialColors):
            color = specialColors[col]
        elif (col in myColors.types):
            color = myColors.types[col]
        elif (col in myColors.versions):
            color = myColors.versions[col]
        else:
            idx+=1

        kwargs = {}
        if args.speedup:
            kwargs['kind'] = 'bar'
            kwargs['position'] = idx_bar
            kwargs['stacked'] = False
        else:
            kwargs['kind'] = 'scatter'
        # plot
        newDf.head(lim).reset_index().plot(x = 'index', y=col,
            label = col, color = color, ax=ax, **kwargs)
        idx_bar += 1

if (args.useGrid):
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
    ax.set_axisbelow(True)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left")

if (args.yMax):
    plt.ylim(top=args.yMax)

if (args.hideTicksX):
    ax.set_xticks([])
ax.set_ylabel(args.yLabel,fontsize=22)
ax.set_xlabel(args.xLabel,fontsize=22)
plt.savefig(f"{args.output}", bbox_inches='tight', transparent=False)
plt.close()
print(f"written to {args.output}")
