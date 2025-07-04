#!/usr/bin/elw python

import sys
import argparse
try:
    import seaborn as sns
    sns.set(style="dark")
except ImportError:
    pass
import matplotlib.pyplot as plt

from common_plot import parse_files, plot_loss

parser = argparse.ArgumentParser(description="Plot loss from log files.")
parser.add_argument('-v', dest='value_at_hover', action='store_true',
    help="Display plot values at cursor hover")
parser.add_argument('-s', dest='separate', action='store_true',
    help="plot each log separately, don't concatenate them")
parser.add_argument('log', nargs = '*', help = "list of log files.")
args = parser.parse_args()

data = parse_files(files=args.log, separate=args.separate)
plt = plot_loss(data=data, value_at_hover=args.value_at_hover)

plt.show()
