import os
import sys
import subprocess

"""
This scripts detects if local memory is being used inside the fma-heavy main loop
"""

def analyzeMmaMainloop(lines, start, end, kernelName):
    """
    Analyzes generated sass code:
        * detect local memory usage
    """
    lmem_usage = 0
    for line in range(start, end):
        if (lines[line].find("LDL") != -1 or lines[line].find("STL") != -1):
            lmem_usage += 1
    print(f"lmem usage in {kernelName}: {lmem_usage}")

def analyzeKernel(lines, lwrrentLine):
    """
    Find beginning and end of mma mainloop
    """

    mma_ops = ["FMA", "HMMA", "DMMA"]

#	.sectionflags	@"SHF_BARRIERS=1"
#	.sectioninfo	@"SHI_REGISTERS=144"
#	.align	128
#.text.__lw_static_53__40_tensorContraction_lwte_sm70_ssss_cpp1_ii_545a3692__ZN4lwte11gett_kernelINS_5tupleIJiiiiiEEES2_NS1_IJiiiEEENS1_IJEEEfNS1_IJS2_S3_EEENS_6LayoutINS1_IJNS1_IJNS_8constantIiLi128EEEEEENS1_IJNS7_IiLi16EEEEEEEEENS1_IJNS1_IJNS7_IiLi1EEEEEENS1_IJNS7_IiLi129EEEEEEEEEEENS6_INS1_IJSA_SA_EEENS1_IJiiEEEEEfS5_NS6_INS1_IJNS1_IJNS7_IiLi64EEEEEESB_EEENS1_IJSE_NS1_IJNS7_IiLi65EEEEEEEEEEESL_fNS1_IJS2_S2_EEENS6_INS1_IJS9_SN_EEENS1_IJSE_S9_EEEEENS6_ISJ_NS1_IJSD_SA_EEEEEffEEvT_T0_T1_T2_PKT3_T4_S12_T5_T6_PKT7_T8_S12_T9_T10_PKT11_T12_S12_T13_T14_PS1F_T15_T16_:

    # find the first FMA (assuming this is within the main loop)
    fmaFound = False
    while lwrrentLine < len(lines) and not fmaFound:
        found = False
        for tok in mma_ops:
            if (lines[lwrrentLine].find(tok) != -1):
                #print("found", lwrrentLine, lines[lwrrentLine])
                fmaFound = True
                break
        lwrrentLine += 1
    lineFirstFMA = lwrrentLine

    if (not fmaFound):
        return lwrrentLine

    # find kernel name 
    kernelName = "UNKNOWN"
    kernelFound = False
    while lwrrentLine < len(lines) and lwrrentLine >= 0 and not kernelFound:
        if ((lines[lwrrentLine].find("sectionflags") != -1) or
                (lines[lwrrentLine].find("sectioninfo") != -1)): # found new section skim forward to kernel name
            while lwrrentLine < len(lines): #find .text.<kernelname>
                if (lines[lwrrentLine].startswith(".text.")):
                    kernelName = lines[lwrrentLine][len(".text."):].split()[0]
                    kernelFound = True
                    break
                lwrrentLine += 1
        lwrrentLine -= 1

    lwrrentLine = lineFirstFMA

    isContractionKernel = kernelName.lower().find("contraction") != -1

    # go back to prev label
    while (lwrrentLine >= 0 and lwrrentLine < len(lines) and lines[lwrrentLine].startswith(".L_")):
        lwrrentLine -= 1
    start = lwrrentLine

    # go forward to next jump
    while (lwrrentLine < len(lines) and lines[lwrrentLine].find("BRA") == -1):
        lwrrentLine += 1
    end = lwrrentLine

    # go to next kernel (to skip any FMAs that might be part of the epilogue)
    while lwrrentLine < len(lines) and lwrrentLine >= 0:
        if ((lines[lwrrentLine].find("sectionflags") != -1) or
            (lines[lwrrentLine].find("sectioninfo") != -1)): # found new section skim forward to kernel name
            break
        lwrrentLine += 1

    if (isContractionKernel):
        mainLoopFound = (end < len(lines)) and (start >= 0)

        if (mainLoopFound):
            analyzeMmaMainloop(lines, start, end, kernelName) 
        else:
            print("main loop not found: %s %d %d %d %d"%(filename, start, end, lwrrentLine, len(lines)))

    return lwrrentLine

def analyzeSASS(filename):

    asmFile = open(filename,"r")

    lines = []
    for l in asmFile:
        lines.append(l) 

    # loop over all kernels
    lwrrentLine = 0
    while (lwrrentLine < len(lines)):
        lwrrentLine = analyzeKernel(lines, lwrrentLine)

if( len(sys.argv) < 2):
    print("Usage: python detectLocalMemoryUsage.py <path to keep directory>")
    exit(-1)

for (dirpath, dirnames, filenames) in os.walk(sys.argv[1]):
    for fname in filenames:
        if fname.endswith(".lwbin") and (fname.lower().find("contraction") != -1):
            # create sass code
            filename = dirpath + "/" + fname
            print("Analyzing", filename)
            ret = os.system( "lwdisasm_internal %s > %s.sass"%(filename, filename))
            analyzeSASS(filename + ".sass")

