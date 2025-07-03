# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:16:14 2017

@author: yanxu

To colwert lwdnn API log to test command, use e.g.
python2 ./lwdnnLogToTestCmd.py ~/lwdnn_dev4/MobaXterm_workstation_v100_20171115_105959.txt
"""

import re
from helpers.Flags import Flags
import argparse

# gradually adding support to more finctions
supported_function_list = ["lwdnnColwolutionForward", "lwdnnColwolutionForward_v3",
                           "lwdnnActivationForward", "lwdnnPoolingBackward", "lwdnnActivationForward_v4",
                           "lwdnnPoolingForward", "lwdnnBatchNormalizationForwardTraining", 
                           "lwdnnSoftmaxForward", "lwdnnLRNCrossChannelForward", "lwdnnAddTensor",
                           "lwdnnColwolutionBackwardData", "lwdnnColwolutionBackwardFilter" ]

supported_flag_list = ["colw", "poolf", "poolb", "activationf", "dgrad", "wgrad",
                        "bnf","bnb","softmaxf","lrnf","add"]

unsupported_flag_list = []

lwdnlwersionRegex = re.compile("LwDNN \(v([0-9]{4})\) function", re.MULTILINE + re.DOTALL)
singleFuncCallRegex_legacy = re.compile("Function ([a-zA-Z0-9_]+)\(\) called:[ ]*[\r\n]+(.*?)time=[0-9T:\-\.]+?, pid=[0-9]+?, tid=[0-9]+?\.", re.MULTILINE + re.DOTALL)
singleFuncCallRegex_71xx = re.compile("LwDNN \(v71[0-9]{2}\) function ([a-zA-Z0-9_]+)\(\) called:[ ]*[\r\n]+(.*?)Time: [0-9Tdhmsincetar:\+\(\)\-\. ]+?[\r\n]+Process=[0-9]+?; Thread=[0-9]+?; Handle=[x0-9a-fA-FNUL]*?; StreamId=[x0-9a-fA-FniNULultSrm\(\) ]*?\.", re.MULTILINE + re.DOTALL)

singleVarRegex = re.compile("[ ]*([a-zA-Z0-9_]+?): type=([a-zA-Z0-9_ ]+?); val=([a-zA-Z0-9\[\]\(\)\.\-,_ ]+?);")
singlePtrRegex = re.compile("[ ]*([a-zA-Z0-9_]+?): location=([devhost]+?); addr=([a-zA-Z0-9\[\]\(\)\.,_ ]+?);")
NULLPtrRegex = re.compile("[ ]*([a-zA-Z0-9_]+?): .+?[;:] .*?NULL_PTR;")
singleStructRegex = re.compile("[ ]*([a-zA-Z0-9_]+?): type=([a-zA-Z0-9_ ]+?):")

lwdnnColwolutionBwdDataAlgo_colwert = { "LWDNN_COLWOLUTION_BWD_DATA_ALGO_0 (0)" : "0",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_1 (1)" : "1",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT (2)" : "2",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_FFT_TILING (3)" : "3",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD (4)" : "4",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED (5)" : "5",
                                        "LWDNN_COLWOLUTION_BWD_DATA_ALGO_COUNT (6)" : "6"}

lwdnnColwolutionFwdAlgo_colwert = { "LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_GEMM (0)" : "0",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (1)" : "1",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_GEMM (2)" : "2",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_DIRECT (3)" : "3",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_FFT (4)" : "4",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_FFT_TILING (5)" : "5",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD (6)" : "6",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_WINOGRAD_NONFUSED (7)" : "7",
                                    "LWDNN_COLWOLUTION_FWD_ALGO_COUNT (8)" : "8"}

lwdnnColwolutionBwdFilterAlgo_colwert = {   "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_0 (0)" : "0",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_1 (1)" : "1",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT (2)" : "2",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_3 (3)" : "3",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD (4)" : "4",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED (5)" : "5",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_FFT_TILING (6)" : "6",
                                            "LWDNN_COLWOLUTION_BWD_FILTER_ALGO_COUNT (7)" : "7"}

lwdnnPoolingMode_colwert = {    "LWDNN_POOLING_MAX (0)": "0",
                                "LWDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING (1)": "1",
                                "LWDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING (2)": "2",
                                "LWDNN_POOLING_MAX_DETERMINISTIC (3)": "3"}

# keep "seed" at the moment
lwdnnTestKeyWord = ["&&&&", "@@@@", "####", "^^^^", "Gflops", "PERF"]
nonDeterministicKeyWord = ["Time", "time", "pid", "Process", "tid", "Thread", "Handle", "handle", "StreamId", "addr", "plan"] # "Gflops", "^^^^"


def stripNonDeterministic(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        # line = line.lower()
        line = line.replace("I! ","")
        line = line.replace("i! ","")

        if ("^^^^" in line) or ("Gflops" in line) or ("PERF" in line):
            # need to skip, not replace, because second run with "-T1" will have different number of these lines
            continue
        elif any(keyWord in line for keyWord in nonDeterministicKeyWord):
            line="(removed)"
        if mode == 0:
            print line
        elif mode == 1:
            cleanLog += (line + "\n")
    
    if mode == 0:
        return ""
    elif mode == 1:
        return cleanLog

def stripNonLog(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        # line = line.lower()
        line = line.replace("I! ","")
        line = line.replace("i! ","")

        if any(keyWord in line for keyWord in lwdnnTestKeyWord):
            # need to skip, not replace, because second run with "-T1" will have different number of these lines
            continue
        elif line == "":
            continue
        elif any(keyWord in line for keyWord in nonDeterministicKeyWord):
            line="(removed)"
        if mode == 0:
            print line
        elif mode == 1:
            cleanLog += (line + "\n")
    
    if mode == 0:
        return ""
    elif mode == 1:
        return cleanLog

def stripLog(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        if any(keyWord in line for keyWord in ["I! ","i! "]):
            pass
        elif line == "":
            pass
        else:
            if mode == 0:
                print line
            elif mode == 1:
                cleanLog += (line + "\n")
    
    if mode == 1:
        return cleanLog


def getLwdnlwersion(logSegment):
    version = re.findall(lwdnlwersionRegex, logSegment)

    if not version:
        version = "alpha"
    else:
        version = version[0]
    
    return version

def logToTest_compareExelwtionPath( output1, output2):

    output1 = output1.replace("I! ","")
    output1 = output1.replace("i! ","")
    output2 = output2.replace("I! ","")
    output2 = output2.replace("i! ","")

    cleanLog1 = stripNonLog(output1, 1)
    cleanLog2 = stripNonLog(output2, 1)

    if cleanLog1 == cleanLog2:
        return True
    else:
        print "===========\ntestLog1=\n" + cleanLog1 + "===========\n"
        print "===========\ntestLog2=\n" + cleanLog2 + "===========\n"
        return False

def get_lwdnn_flags(func_name, params):

    flags = Flags()

    if(func_name in ["lwdnnColwolutionForward", "lwdnnColwolutionForward_v3"]):

        # Assume forward
        inImageName  = "xDesc"
        inFilterName = "wDesc"
        
        if("Forward" in func_name):
            flags["R"] = ("colw",)
        
        if(params["colwDesc"]["mode"]=="LWDNN_COLWOLUTION (0)"):
            pass
        elif(params["colwDesc"]["mode"]=="LWDNN_CROSS_CORRELATION (1)"):
            flags["x"] = ("",)

        if params["colwDesc"]["groupCount"] != "1":
            flags["groupCount"] = (params["colwDesc"]["groupCount"],)
        
        algo = lwdnnColwolutionFwdAlgo_colwert[params["algo"]]
        flags["algo"] = (algo,)

        nbDims = int(params[inImageName]["nbDims"])

        if nbDims == 5:
            flags["dim"] = ("3",)

        flags["dimA"] = (",".join(params[inImageName]["dimA"][:nbDims]),)
        flags["filtA"] = (",".join(params[inFilterName]["dimA"][:nbDims]),)
        flags["padA"] = (",".join(params["colwDesc"]["padA"][:(nbDims-2)]),)
        flags["colwStrideA"] = (",".join(params["colwDesc"]["strideA"][:(nbDims-2)]),)
        
        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return flags

    if(func_name in ["lwdnnColwolutionBackwardData", "lwdnnColwolutionBackwardFilter"]):

        if(func_name == "lwdnnColwolutionBackwardData"):
            flags["R"] = ("dgrad",)
            algo = lwdnnColwolutionBwdDataAlgo_colwert[params["algo"]]
            flags["algo"] = (algo,)
            inDataTensor = "dxDesc"
            filterTensor = "wDesc"
        elif(func_name == "lwdnnColwolutionBackwardFilter"):
            flags["R"] = ("wgrad",)
            algo = lwdnnColwolutionBwdFilterAlgo_colwert[params["algo"]]
            flags["algo"] = (algo,)
            inDataTensor = "xDesc"
            filterTensor = "dwDesc"
        
        if(params["colwDesc"]["mode"]=="LWDNN_COLWOLUTION (0)"):
            pass
        elif(params["colwDesc"]["mode"]=="LWDNN_CROSS_CORRELATION (1)"):
            flags["x"] = ("",)

        if params["colwDesc"]["groupCount"] != "1":
            flags["groupCount"] = (params["colwDesc"]["groupCount"],)

        nbDims = int(params[inDataTensor]["nbDims"])
        
        if nbDims == 5:
            flags["dim"] = ("3",)

        flags["dimA"] = (",".join(params[inDataTensor]["dimA"][:nbDims]),)
        flags["filtA"] = (",".join(params[filterTensor]["dimA"][:nbDims]),)
        flags["padA"] = (",".join(params["colwDesc"]["padA"][:(nbDims-2)]),)
        flags["colwStrideA"] = (",".join(params["colwDesc"]["strideA"][:(nbDims-2)]),)
    
        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)
        
        flags["b"] = ("",)
        return flags

    if(func_name in ["lwdnnActivationForward", "lwdnnActivationForward_v4"]):

        if("Forward" in func_name):
            flags["R"] = ("activationf",)
        elif("Backward" in func_name):
            flags["R"] = ("activationb",)

        nbDims = int(params["srcDesc"]["nbDims"])

        if nbDims == 4:
            flags["n"] = (params["srcDesc"]["dimA"][0],)
            flags["c"] = (params["srcDesc"]["dimA"][1],)
            flags["h"] = (params["srcDesc"]["dimA"][2],)
            flags["w"] = (params["srcDesc"]["dimA"][3],)
    
        else:
            flags["dimA"] = (",".join(params["srcDesc"]["dimA"][:nbDims]),)

        flags["activCoef"] = (params["activationDesc"]["coef"],)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return flags


    if(func_name in ["lwdnnPoolingForward", "lwdnnPoolingBackward"]):
        
        if("Forward" in func_name):
            flags["R"] = ("poolf",)
            get_size_from = "srcDesc"
        elif("Backward" in func_name):
            flags["R"] = ("poolb",)
            get_size_from = "destDesc"

        flags["mode"] = (lwdnnPoolingMode_colwert[params["poolingDesc"]["mode"]],)

        # pooling mode 1 don't use src desc
        if isinstance(params[get_size_from], dict):
            nbDims = int(params[get_size_from]["nbDims"])
        else:
            print "!!!! Warning " + get_size_from + " is NULL_PTR, expect to fail (why?)"
            return None

        if nbDims == 4:
            flags["n"] = (params[get_size_from]["dimA"][0],)
            flags["c"] = (params[get_size_from]["dimA"][1],)
            flags["h"] = (params[get_size_from]["dimA"][2],)
            flags["w"] = (params[get_size_from]["dimA"][3],)
            
            flags["pad_h"] = (params["poolingDesc"]["paddingA"][0],)
            flags["pad_w"] = (params["poolingDesc"]["paddingA"][1],)

            flags["u"] = (params["poolingDesc"]["strideA"][0],)
            flags["v"] = (params["poolingDesc"]["strideA"][1],)

            flags["win_h"] = (params["poolingDesc"]["windowDimA"][0],)
            flags["win_w"] = (params["poolingDesc"]["windowDimA"][1],)

        else:
            flags["dimA"] = (",".join(params[get_size_from]["dimA"][:nbDims]),)
            flags["filtA"] = (",".join(params["poolingDesc"]["windowDimA"][:(nbDims-2)]),)
            flags["padA"] = (",".join(params["poolingDesc"]["paddingA"][:(nbDims-2)]),)
            flags["colwStrideA"] = (",".join(params["poolingDesc"]["strideA"][:(nbDims-2)]),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)

        return flags

    if(func_name in ["lwdnnBatchNormalizationForwardTraining"]):
        
        if(params["mode"] == "LWDNN_BATCHNORM_SPATIAL"):
            algo = "1"
        elif(params["mode"] == "LWDNN_BATCHNORM_PER_ACTIVATION"):
            algo = "0"
            
        if("Forward" in func_name):
            flags["R"] = ("bnf",)
        elif("Backward" in func_name):
            flags["R"] = ("bnb",)

        nbDims = int(params["xDesc"]["nbDims"])

        if nbDims == 4:
            flags["n"] = (params["xDesc"]["dimA"][0],)
            flags["c"] = (params["xDesc"]["dimA"][1],)
            flags["h"] = (params["xDesc"]["dimA"][2],)
            flags["w"] = (params["xDesc"]["dimA"][3],)

        else:
            flags["dimA"] = (",".join(params["xDesc"]["dimA"][:nbDims]),)

        flags["algo"] = (algo)

        if("Backward" in func_name):
            if(params["dy"] == params["dx"]):
                flags["inplace"] = ("",)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return flags
        
    if(func_name in ["lwdnnSoftmaxForward"]):
    
        if("Forward" in func_name):
            flags["R"] = ("softmaxf",)
            inTensorName = "srcDesc"
            
        if("Backward" in func_name):
            flags["R"] = ("softmaxb",)
            inTensorName = "dxDesc"
        
        nbDims = int(params[inTensorName]["nbDims"])

        if nbDims == 4:
            flags["n"] = (params[inTensorName]["dimA"][0],)
            flags["c"] = (params[inTensorName]["dimA"][1],)
            flags["h"] = (params[inTensorName]["dimA"][2],)
            flags["w"] = (params[inTensorName]["dimA"][3],)
        else:
            flags["dimA"] = (",".join(params[inTensorName]["dimA"][:nbDims]),)

        mode_colwert = {"LWDNN_SOFTMAX_MODE_INSTANCE (0)": 0,
                        "LWDNN_SOFTMAX_MODE_CHANNEL (1)": 1}
        
        int_mode = mode_colwert[params["mode"]]
        
        mode_colwert = {"LWDNN_SOFTMAX_FAST (0)" : 0,
                        "LWDNN_SOFTMAX_ACLWRATE (1)" : 1,
                        "LWDNN_SOFTMAX_LOG (2)" : 2}
        int_algo = mode_colwert[params["algorithm"]]
        
        lwdnn_mode = str( int_algo*2 + int_mode )
        
        flags["mode"] = (lwdnn_mode,)
        
        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)
        
        flags["b"] = ("",)
        return flags
        
    if(func_name in ["lwdnnLRNCrossChannelForward"]):
        flags["R"] = ("lrnf",)

        nbDims = int(params["xDesc"]["nbDims"])
        
        if nbDims == 4:
            flags["n"] = (params["xDesc"]["dimA"][0],)
            flags["c"] = (params["xDesc"]["dimA"][1],)
            flags["h"] = (params["xDesc"]["dimA"][2],)
            flags["w"] = (params["xDesc"]["dimA"][3],)
        else:
            flags["dimA"] = (",".join(params["xDesc"]["dimA"][:nbDims]),)
        
        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return flags

    if(func_name in ["lwdnnAddTensor"]):
        flags["R"] = ("add",)

        nbDims = int(params["srcDestDesc"]["nbDims"])
        
        if nbDims == 4:
            flags["n"] = (params["srcDestDesc"]["dimA"][0],)
            flags["c"] = (params["srcDestDesc"]["dimA"][1],)
            flags["h"] = (params["srcDestDesc"]["dimA"][2],)
            flags["w"] = (params["srcDestDesc"]["dimA"][3],)
        else:
            flags["dimA"] = (",".join(params["srcDestDesc"]["dimA"][:nbDims]),)

        mode = "?"
        # check tensor descriptors to know what kind of addition we are performing
        if (params["biasDesc"] == params["srcDestDesc"]):
            mode = "3" #LWDNN_ADD_FULL_TENSOR
        
        elif (params["biasDesc"] == params["srcDestDesc"] and (params["biasDesc"]["dimA"][0] == "1") and (params["biasDesc"]["dimA"][1] == params["srcDestDesc"]["dimA"][1])):
            # Add same value to all images
            mode = "1" #LWDNN_ADD_FEATURE_MAP
        
        elif (params["biasDesc"] == params["srcDestDesc"] and (params["biasDesc"]["dimA"][0] == "1") and (params["biasDesc"]["dimA"][1] == "1")):
            # Add pixels to all images, channels
            mode = "0" #LWDNN_ADD_IMAGE
        
        elif (params["srcDestDesc"]["dimA"][1] == params["biasDesc"]["dimA"][1]):
            nbElems = 1
            for n in range(nbDims):
                if n != 1:
                    nbElems *= int(params["biasDesc"]["dimA"][n])
            if nbElems == 1:
                # add channel to all images and pixels
                mode = "2" #LWDNN_ADD_SAME_C
        
        if mode == "?":
            print "!!!! Wrning: mode is not set properly"
        flags["mode"] = (mode,)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return flags

    return None



def parseVariable(paramList):
    if paramList.__len__() == 1:
        param = paramList[0]
        if "NULL_PTR" in param:
            name = re.findall(NULLPtrRegex, param)
            if name != []:
                return name[0], "NULL_PTR"
            else:
                print "!!!! warning: NULLPtrRegex no match in"
                print param
                return "unknown", 0
        elif "lwdnnHandle_t" in param:
            # don't care handle value
            return "handle", 0
        elif "addr=" in param:
            var = re.findall(singlePtrRegex, param)
            if var != []:
                return var[0][0], var[0][2]
            else:
                print "!!!! warning: singlePtrRegex no match in"
                print param
                return "unknown", 0
        elif "val=" in param:
            var = re.findall(singleVarRegex, param)
            if var != []:
                name = var[0][0]
                val = var[0][2]
                if "[" in val and "]" in val:
                    array = val.replace('[','')
                    array = array.replace(']','')
                    array = array.replace(' ','')
                    array = array.split(',')
                    return name, array
                else:
                    return name, val
            else:
                print "!!!! warning: singleVarRegex no match in"
                print param
                return "unknown", 0
        else:
            print "!!!! warning: parse single line variable no match in"
            print param
            return "unknown", 0

    else:
        totalLines = paramList.__len__()
        struct_dict = {}
        structSig = re.findall(singleStructRegex, paramList[0])

        if structSig != []:
            struct_name = structSig[0][0]
            indentLv = list()
            for lineNum in range(totalLines):
                line = paramList[lineNum]
                indentLv.insert( lineNum, (len(line) - len(line.lstrip()))/4.0)
            
            topIndentLv = indentLv[1]
            varLineStart = 1
            varLineEnd = 1
            for lineNum in range(1,totalLines):
                lwrrIndentLv = indentLv[lineNum]
            
                if ( lwrrIndentLv == topIndentLv ):
                    varLineStart = lineNum
                    varLineEnd = lineNum
                elif ( lwrrIndentLv > topIndentLv ):
                    varLineEnd = lineNum
    
                if( lineNum == totalLines-1 ):
                    name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
                    struct_dict[name] = val
                elif ( indentLv[lineNum+1] == topIndentLv ):
                    name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
                    struct_dict[name] = val
            
            return struct_name, struct_dict
        else:
            return "unknownStructName", {"unknown":"unknown"}
        


def parseFuncCall(singleFuncCall):
    funcCall = {}
    funcCall["Function_name"] = singleFuncCall[0]
    if singleFuncCall[1] == '':
        funcCall["Function_params"] = {"unknown":"unknown"}
        return funcCall
    params = {}
    paramList = singleFuncCall[1].rstrip()
    paramList = paramList.split("\n")
    indentLv = list()
    for lineNum in range(len(paramList)):
        line = paramList[lineNum]
        indentLv.insert( lineNum, (len(line) - len(line.lstrip()))/4.0)
    
    totalLines = len(paramList)
    varLineStart = 0
    varLineEnd = 0
    topIndentLv = indentLv[0]
    
    varLineStart = 0
    varLineEnd = 0
    for lineNum in range(0,totalLines):
        lwrrIndentLv = indentLv[lineNum]
    
        if ( lwrrIndentLv == topIndentLv ):
            varLineStart = lineNum
            varLineEnd = lineNum
        elif ( lwrrIndentLv > topIndentLv ):
            varLineEnd = lineNum

        if( lineNum == totalLines-1 ):
            name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
            params[name] = val
        elif ( indentLv[lineNum+1] == topIndentLv ):
            name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
            params[name] = val
    
    funcCall["Function_params"] = params
    return funcCall


def logToTest_checkFlagSupport(flags):
    
    if (flags["R"][0] not in supported_flag_list):
        return False
    elif "b" not in flags:
        # skip randomized ones
        return False
    elif "N" in flags:
        # skip commands that have multiple tests
        if flags["N"][0] != "1":
            return False
    else:
        for key in flags:
            if key in unsupported_flag_list:
                return False
        return True

def logToTest_generate(test_results, flags):
    # flags i.e. the correct answer is passed in just in case we need to follow flags like "-d0" or "-T1" so UID matches
    # but it is not used right now, since excelwtion path comparison is immune to difference in "-d0" or "-T1"

    cleanoutput = test_results.output.replace("I! ","")
    cleanoutput = cleanoutput.replace("i! ","")

    version = re.findall(lwdnlwersionRegex, cleanoutput)
    if not version:
        version = "alpha"
    else:
        version = version[0]

    if version == "alpha":
        singleFuncCallRegex = singleFuncCallRegex_legacy
    elif version == "7100" or version == "7101":
        singleFuncCallRegex = singleFuncCallRegex_71xx
    else:
        print "!!!! Warning version" + version + " is not supported. Default: try v7100 format."
        singleFuncCallRegex = singleFuncCallRegex_71xx

    funcCallList = re.findall(singleFuncCallRegex, cleanoutput)

    funcCallDictList = list()
    for singleFuncCall in funcCallList:
        funcName = singleFuncCall[0]
        if funcName in supported_function_list:
            funcCall = parseFuncCall(singleFuncCall)
            funcCallDictList.append(dict(funcCall))
    
    #always only look at the last supported function call
    if len(funcCallDictList) == 0:
        print "!!!! Warning, no supported function detected!"
        return "error"
    elif len(funcCallDictList) > 1:
        print "!!!! Warning, multiple supported function detected, looking at the last one!"
        print [funcCallDictList[n]["Function_name"] for n in range(len(funcCallDictList))]
        singleFuncCall = funcCallDictList[-1]
    else:
        singleFuncCall = funcCallDictList[0]
    
    gen_flags = get_lwdnn_flags(funcCall["Function_name"], funcCall["Function_params"])
    
    if gen_flags is not None:
        return gen_flags
    else:
        print "!!!! Error, no flags output"
        return "error"

def flagsToLayer(flags):
    layerComponents = []
    for key in flags:
        if key == "R":
            layerComponents.append("R:" + flags[key][0])
        else:
            layerComponents.append(key+':'+flags[key][0])
    
    return (" * ").join(layerComponents)


def main():

    def make_help(s, has_choices=False):
        result = s + " [default: %(default)s]"
        if(has_choices):
            result += " [choices: %(choices)s]"
        return result

    arg_format = lambda prog: argparse.HelpFormatter(prog,max_help_position=100, width=160)
    parser = argparse.ArgumentParser(description='LwDNN Log File Utility', formatter_class=arg_format)
    parser._optionals.title = "Mutually Exclusive Options"

    mode_args = parser.add_mutually_exclusive_group(required=True)
    
    mode_args.add_argument('-stripNondeterministic', action='store_const', const=True, default=False, 
        help=make_help("Option 1: Remove all nondeterminstic log content (e.g. time, pid, tid, addresses) for cleaner diff."))

    mode_args.add_argument('-stripLog', action='store_const', const=True, default=False, 
        help=make_help("Option 2: Remove all API log from input file."))

    mode_args.add_argument('-genTestCmd', action='store_const', const=True, default=False, 
        help=make_help("Option 3: Regenerate lwdnnTest command from log."))

    mode_args.add_argument('-genRunCmd', action='store_const', const=True, default=False, 
        help=make_help("Option 4: Regenerate lwdnn_run.py command from log."))
    
    gen_args = parser.add_argument_group('General Options')
    
    gen_args.add_argument('-inputFile', metavar='"path"', dest='filePath', default=None , required=True,
        help=make_help("Required: Specify path to input log file"))

    gen_args.add_argument('-noRepeat', action='store_const', const=True, default=False,
        help=make_help("Avoid generating repeated tests in \"-genTestCmd\" and \"-genRunCmd\" mode"))

    args = parser.parse_args()

    print "Reading file " + args.filePath

    if args.filePath is not None:
        with open(args.filePath, "r") as in_file:
            fileContent = in_file.read()

    if args.stripNondeterministic == True:
        stripNonDeterministic(fileContent, mode=0)
        return

    elif args.stripLog == True:
        stripLog(fileContent, mode=0)
        return
    
    fileContent = fileContent.replace("I! ","")
    fileContent = fileContent.replace("i! ","")

    version = re.findall(lwdnlwersionRegex, fileContent)
    if not version:
        version = "alpha"
    else:
        version = version[0]

    if version == "alpha":
        singleFuncCallRegex = singleFuncCallRegex_legacy
    elif version == "7100" or version == "7101":
        singleFuncCallRegex = singleFuncCallRegex_71xx
    else:
        print "!!!! Warning version" + version + " is not supported. Default: try v7100 format."
        singleFuncCallRegex = singleFuncCallRegex_71xx

    funcCallList = re.findall(singleFuncCallRegex, fileContent)

    # print funcCallList

    funcCallDictList = list()
    for singleFuncCall in funcCallList:
        funcName = singleFuncCall[0]
        if funcName in supported_function_list:
            funcCall = parseFuncCall(singleFuncCall)
            funcCallDictList.append(dict(funcCall))
    
    allowRepeat = not args.noRepeat

    if args.genTestCmd == True:
        listoflwdnnTest = []
        for singleFuncCall in funcCallDictList:
            flags = get_lwdnn_flags(singleFuncCall["Function_name"], singleFuncCall["Function_params"])
            if flags is not None:
                testCmd = "lwdnnTest " + str(flags)
                if testCmd not in listoflwdnnTest or allowRepeat:
                    listoflwdnnTest.append(testCmd)
                else:
                    pass # print "Found repeated..."
            else:
                print "!!!! Error, no flags output for this function:"
                print singleFuncCall
        print ("\n").join(listoflwdnnTest)

    elif args.genRunCmd == True:
        listoflwdnnRun = []
        for singleFuncCall in funcCallDictList:
            flags = get_lwdnn_flags(singleFuncCall["Function_name"], singleFuncCall["Function_params"])
            if flags is not None:
                layerCmd = flagsToLayer(flags)
                if layerCmd not in listoflwdnnRun or allowRepeat:
                    listoflwdnnRun.append(layerCmd)
                else:
                    pass # print "Found repeated..."
            else:
                print "!!!! Error, no flags output for this function:"
                print singleFuncCall
        
        print ("\n").join(listoflwdnnRun)
        
if __name__== "__main__":
    import sys
    main()

