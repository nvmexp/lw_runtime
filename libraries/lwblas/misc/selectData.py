"""
This script selects a disired column form a lwtensorTest output file and stores it to a
.csv file
"""
from lwtensorUtil import *
import argparse
import os
parser = argparse.ArgumentParser(description='Select data from lwtensorTest logfile')

parser.add_argument('--logfile', help="lwtensorTest log file", type=str, required=True)
parser.add_argument('--output', help="output file", type=str, required=True)
parser.add_argument('--label', help="label used for storing the data in the csv file", type=str, required=True)
parser.add_argument('--algo', type=str, 
                    choices=['MAX','GETT','GETT_MAX','LWTE','lwblas'], default='MAX')

def dumpData(filename, algo, label, outputfile):
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

    fo = open(outputfile, "w+")
    fo.write(label + "\n")
    for tc in tcOrder:
        key = tc.replace(",",";").replace(" ","")
        fo.write(key+","+str(data[tc]) + "\n")
    fo.close()
    print(f"{outputfile} was created successfully")

args = parser.parse_args()
algo = args.algo
label = args.label
outputfile = args.output

dumpData(args.logfile, algo, label, outputfile)


