#!/usr/bin/elw python2.7

# testing submitting changes from git
# making a change from Perforce

# run with
# PYTHONPATH=<path_to_tegra_clocks>

from tegra_clocks import set_rate, adb_shell
import argparse
import sys

class Freqs:
    def __init__(self, cpu, gpu, emc=None):
        self.cpu = cpu
        self.gpu = gpu
        self.emc = emc

t124_defaults = Freqs(cpu=1938, gpu=540, emc=1600)
t210_defaults = Freqs(cpu=1912, gpu=537, emc=1600)

def error(msg):
    print(msg)
    sys.exit(1)

def default_clocks():
    board = adb_shell("cat /sys/devices/soc0/machine")[0].rstrip()
    if board == "loki_e":
        return t210_defaults
    if board == "jetson_e":
        return t210_defaults
    print ("Couldn't detect board: {0}".format(board))
    return

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, help='Set GPU clock rate (MHz)')
parser.add_argument('--cpu', type=int, help='Set CPU clock rate (MHz)')
parser.add_argument('--emc', type=int, help='Set EMC clock rate (MHz)')

args = parser.parse_args()

print "Stopping USSRD/PHS - this interferes with benchmarking"
adb_shell("stop ussrd")

freqs = default_clocks()

if not (args.cpu or args.gpu):
    print "Setting default clock rates"
else:
    if args.cpu is None:
        error("Must specify clock rate with --cpu")
    if args.gpu is None:
        error("Must specify clock rate with --gpu")
    freqs.cpu = args.cpu
    freqs.gpu = args.gpu
    if args.emc is not None:
        freqs.emc = args.emc

set_rate("cpu", freqs.cpu * 1000000)
set_rate("gbus", freqs.gpu * 1000000)
set_rate("emc", freqs.emc * 1000000)
