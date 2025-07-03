# Author: Paul Springer (springer@aices.rwth-aachen.de)

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def computeSpeedup(f1, f2, description):
    speedup = 0
    for i in range(len(f1)):
        sp = f2[i] / f1[i]
        if(sp < 0.2):
            print sp, description[i]
        speedup += sp
    print speedup / len(f1)

def main():

    lwtensor2 = readFlops("./smallk_2.dat", 22,[])
    lwtensor4 = readFlops("./smallk_4.dat", 22,[])
    lwtensor8 = readFlops("./smallk_8.dat", 22,[])
    lwtensor16 = readFlops("./smallk_16.dat", 22,[])
    lwtensor32 = readFlops("./smallk_32.dat", 22,[])
    x = range(16,257,16) + range(288,513,32)

    dat = [ ("2", lwtensor2),
            ("4", lwtensor4),
            ("8", lwtensor8),
            ("16", lwtensor16),
            ("32", lwtensor32)]
    plotSimple(dat,x, "gemv_small_k.png", "Bandwidth [GB/s]")
    lwtensor2 = readFlops("./smallk_2_lwec.dat", 22,[])
    lwtensor4 = readFlops("./smallk_4_lwec.dat", 22,[])
    lwtensor8 = readFlops("./smallk_8_lwec.dat", 22,[])
    lwtensor16 = readFlops("./smallk_16_lwec.dat", 22,[])
    lwtensor32 = readFlops("./smallk_32_lwec.dat", 22,[])
    x = range(16,257,16) + range(288,513,32)

    dat = [ ("2", lwtensor2),
            ("4", lwtensor4),
            ("8", lwtensor8),
            ("16", lwtensor16),
            ("32", lwtensor32)]
    plotSimple(dat,x, "smallk_lwec.pdf", "Bandwidth")


    sortTo = []
    f = open("./gemv_new.dat", "r")
    lines = []
    for l in f:
        lines.append(l)
    computeSpeedup(readFlops("./gemv_old.dat", 22,[]), readFlops("./gemv_new7.dat", 22,[]), lines)

    sortTo = readFlops("./gemv_new7.dat", 22,[])
    lwtensor_new = readFlops("./gemv_new7.dat", 22,sortTo)
    lwtensor_old = readFlops("./gemv_old.dat", 22,sortTo)
    

    memcpy = readFlops("./gemv_old.dat", 24,sortTo)
    dat = [ 
            ("lwTENSOR old (GV100)", 'v', 10, "#fe9929",lwtensor_old),
            ("lwTENSOR new (GV100)", '^', 10,"#31a354",lwtensor_new),
            #("memcpy (GV100)", 'v', 10, "#fe9929", memcpy),
            ]
    plot(dat, "gemv_fp32_v3.png", "upper left", "Bandwidth [GB/s]")


    sortTo = readFlops("./lwtensor_lwdnn_fp32.dat", 16,[])
    lwTensor = readFlops("./lwtensor_lwdnn_fp32.dat", 16, sortTo)
    hptt_1socket = readFlops("./hptt_lwdnn_1x_Platinum_8168.dat", 7, sortTo)
    hptt_1socket_beta0 = readFlops("./hptt_lwdnn_1x_Platinum_8168_beta0.dat", 7, sortTo)
    hptt_2socket = readFlops("./hptt_lwdnn_2x_Platinum_8168.dat", 7, sortTo)
    hptt_2socket_beta0 = readFlops("./hptt_lwdnn_2x_Platinum_8168_beta0.dat", 7, sortTo)
    memcpy = readFlops("./lwtensor_lwdnn_fp32.dat", 17, sortTo)
    print max(lwTensor), max(hptt_1socket), max(hptt_1socket_beta0), max(hptt_2socket), max(hptt_2socket_beta0)

    dat = [ 
            #("lwBlas (GV100)", 'v', "#fe9929",lwBlas),
            #("lwTensor - GETT (GV100)", '^', "#31a354",lwTensor),
            #("lwTensor - TTGT (GV100)", '^', "#bae4b3",lwTensor_ttgt),
            ("lwTensor (GV100)", '^', 10,"#31a354",lwTensor),
            ("memcpy (GV100)", 'v', 10, "#fe9929", memcpy),
            ("HPTT (1x Xeon Platinum 8168)", 'o', 9, "#a6bddb",hptt_1socket),
            ("HPTT (2x Xeon Platinum 8168)", 'o', 9, "#0071c5",hptt_2socket),
          ]
    plot(dat, "elementwise_fp32.png", "upper left", "Bandwidth [GB/s]")

    sortTo = readFlops("./lwtensor_lwdnn_fp32_local_64t.dat", 16,[])
    lwTensor_64t = readFlops("./lwtensor_lwdnn_fp32_local_64t.dat", 16, sortTo)
    lwTensor_32t = readFlops("./lwtensor_lwdnn_fp32_local_32t.dat", 16, sortTo)
    lwTensor_fp16_64t = readFlops("./lwtensor_lwdnn_fp16_local_64t.dat", 16, sortTo)
    lwTensor_fp16_32t = readFlops("./lwtensor_lwdnn_fp16_local_32t.dat", 16, sortTo)
    memcpy = readFlops("./lwtensor_lwdnn_fp16_local_64t.dat", 17, sortTo)
    print max(lwTensor), max(hptt_1socket), max(hptt_1socket_beta0), max(hptt_2socket), max(hptt_2socket_beta0)

    dat = [ 
            #("lwBlas (GV100)", 'v', "#fe9929",lwBlas),
            #("lwTensor - GETT (GV100)", '^', "#31a354",lwTensor),
            #("lwTensor - TTGT (GV100)", '^', "#bae4b3",lwTensor_ttgt),
            ("lwTensor - fp32@int32 (GV100)", '^', 10,"#31a354",lwTensor_32t),
            ("lwTensor - fp32@int64 (GV100)", '^', 10,"#0071c5",lwTensor_64t),
            ("lwTensor - fp16@int32 (GV100)", '^', 10,"#fecc5c",lwTensor_fp16_32t),
            ("lwTensor - fp16@int64 (GV100)", '^', 10,"#74a9cf",lwTensor_fp16_64t),
            ("memcpy (GV100)", 'v', 10, "#fe9929", memcpy),
          ]
    plot(dat, "elementwise_int64_vs_int32.pdf", "upper left", "Bandwidth [GB/s]")



    exit(0)
    #analyzeFlops(dat)

def plot(data, filename, loc, ylabel, xlim = []):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    if(len(xlim) == 2):
        plt.xlim(xlim)
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_facecolor((248/256., 248/256., 248/256.))
    ax.set_ylabel(ylabel,fontsize=22)

    for (label, marker, markersize, color, flops) in data:
        ax.plot(flops, label=label, color=color,marker=marker, markersize=markersize, markeredgewidth=0.0, lw = 0, clip_on=False, zorder=10)

    if(ylabel.find("eedup") != -1):
        ax.plot(ax.get_xlim(), [1, 1], color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('test case',fontsize=22)
    for item in ax.get_xticklabels():
        item.set_fontsize(22)
    for item in ax.get_yticklabels():
        item.set_fontsize(24)
    #ax.legend( loc =loc, numpoints = 1, markerscale=2., handletextpad=0.05)

    if( len(data) > 1 ):
        ldgn = ax.legend(loc='upper left', handletextpad=0.2, columnspacing=0.2,prop={'size': 14})
        #ldgn = ax.legend(loc='center', bbox_to_anchor=(0.5,1.04),ncol=len(data),
        #    handletextpad=0.2, columnspacing=0.2)#,prop={'size': 14})
    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()


def plotSimple(data,x, filename, ylabel):
    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(111)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_facecolor((248/256., 248/256., 248/256.))
    ax.set_ylabel(ylabel,fontsize=22)

    for (label, flops) in data:
        ax.plot(x, flops, label=label, markeredgewidth=0.0, lw = 1, clip_on=False, zorder=10)

    ax.set_xlabel('k',fontsize=22)
    for item in ax.get_xticklabels():
        item.set_fontsize(22)
    for item in ax.get_yticklabels():
        item.set_fontsize(24)

    ldgn = ax.legend(loc='upper left', handletextpad=0.2, columnspacing=0.2,prop={'size': 14})
    plt.savefig(filename, bbox_inches='tight', transparent=False)
    plt.close()

def analyzeFlops(data):
    print "".ljust(15) + "min".ljust(10) + "avg".ljust(10) + "max"
    for (label, marker, flops) in reversed(data):
        print label.ljust(15) + ("%.2f"%min(flops)).ljust(10) + ("%.2f"%np.mean(flops)).ljust(10) + "%.2f"%np.max(flops)

def readFlops(filename, column, sortAccordingTo):
    flops = []
    f = open(filename,"r")
    for l in f:
        value = l.split()[column]
        toFind = "memcpy:"
        if(value.find(toFind) != -1):
            value = value[len(toFind):]
        flops.append(float(value))

    if(len(sortAccordingTo) == len(flops)):
        flops = [f for _,f in sorted(zip(sortAccordingTo,flops))]
    return np.array(flops)

if __name__ == "__main__":
    main()
