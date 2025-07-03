import copy
import glob
import os
import sys
import random
import argparse

from lwtensorUtil import *
from candidateLwtlass import *

supportedTypes = {}
#typeA, typeB, typeC, typeCompute, typeScalar
supportedTypes["hhhs"] = ("lwtlass::half_t", "lwtlass::half_t", "lwtlass::half_t", "float", "float","_tc")
supportedTypes["sssh"] = ("float", "float", "float", "float", "float","_tc")
supportedTypes["ssss"] = ("float", "float", "float", "float", "float","")
supportedTypes["dddd"] = ("double", "double", "double", "double", "double","")
supportedTypes["dddt"] = ("double", "double", "double", "float", "double","_tc_tf32")
supportedTypes["ddds"] = ("double", "double", "double", "float", "double","")
supportedTypes["cccc"] = ("lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>","")
supportedTypes["ccct"] = ("lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>", "lwtlass::complex<float>","_tc_tf32")
supportedTypes["zzzc"] = ("lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<float>", "lwtlass::complex<double>","")
supportedTypes["dzzz"] = ("double", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>","")
supportedTypes["zdzz"] = ("lwtlass::complex<double>", "double", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>","")
supportedTypes["zzzz"] = ("lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>", "lwtlass::complex<double>","")
supportedTypes["sssb"] = ("float", "float", "float", "float", "float","_tc_bf16")
supportedTypes["ssst"] = ("float", "float", "float", "float", "float","_tc_tf32")
supportedTypes["bbbs"] = ("lwtlass::bfloat16_t", "lwtlass::bfloat16_t", "lwtlass::bfloat16_t", "float", "float","_tc_bf16")

heuristicDnnWeightsSuffix = { dataType : t.upper() for t, dataType in supportedTypes.items() }

typesUsingSassFeatures = set([
    (70, supportedTypes["ssss"]),
    (70, supportedTypes["hhhs"]),
    (70, supportedTypes["dddd"]),
    (70, supportedTypes["ddds"])
    ])


def handleArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", type=str)
    parser.add_argument("--use-sass-features", action="store_true")
    parser.add_argument("--dont-fallback", action="store_true")
    parser.add_argument("--include-kernels", type=str)
    return parser.parse_args()


def range2(max):
    i = 1
    while i <= max:
        yield i
        i *= 2


def range2x2(max):
    for i in range2(max):
        for j in range2(max):
            yield i, j


def enumerateKernels(tb_max, k_max, warp_max):
    for tb_n, tb_m in range2x2(tb_max):
        for k in range2(k_max):
            for warp_n, warp_m in range2x2(warp_max):
                if warp_n > tb_n: continue
                if warp_m > tb_m: continue
                yield tb_n, tb_m, k, warp_n, warp_m


def compute_smem_max(sm):
    result = 48 * 1024
    if sm >= 70:
        result = 64 * 1024
    if sm >= 80:
        result = 100 * 1024
    return result


def compute_instr_shape(sm, name):
    result = (1, 1, 1)
    if sm >= 70:
        if name in ['hhhs', 'sssh']:
            result = (8, 8, 4)
    if sm >= 80:
        if name in ['hhhs', 'bbbs', 'sssh', 'sssb', 'ssst', 'ccct']:
            result = (16, 8, 8)
        if name in ['dddd', 'zzzz']:
            result = (8, 8, 4)
    return result


def compute_smem_elem_size(sm, gmem_size, op_input_size):
    if sm >= 80:
        return gmem_size
    else:
        return op_input_size


def compute_gmem_size(s):
    lut = {
        'h': 2,
        'b': 2,
        's': 4,
        'd': 8,
        'c': 8,
        'z': 16
    }
    return lut[s]


def compute_acc_size(name):
    lut = {
        'hhhs': 4,
        'bbbs': 4,
        'sssh': 4,
        'sssb': 4
    }
    return lut.get(name, compute_input_size(name))


def compute_input_size(name):
    lut = {
        'hhhs': 2,
        'bbbs': 2,
        'sssh': 2,
        'sssb': 2,
        'ssst': 4,
        'ssss': 4,
        'dddt': 4,
        'ddds': 4,
        'dddd': 8,
        'ccct': 8,
        'cccc': 8,
        'zzzc': 8,
        'zzzz': 16,
        'zdzz': 16,
        'dzzz': 16
    }
    return lut[name]


def compute_min_sm(name):
    lut = {
        'hhhs': 70,
        'bbbs': 80,
        'sssh': 70,
        'sssb': 80,
        'ssst': 80,
        'ssss': 0,
        'dddt': 80,
        'ddds': 0,
        'dddd': 0,
        'ccct': 80,
        'cccc': 0,
        'zzzc': 0,
        'zzzz': 0,
        'zdzz': 0,
        'dzzz': 0
    }
    return lut[name]


def compute_min_ai(instr_shape):
    flops = instr_shape[0] * instr_shape[1] * instr_shape[2]
    bw = instr_shape[0] * instr_shape[1]
    bw += instr_shape[0] * instr_shape[2]
    bw += instr_shape[1] * instr_shape[2]
    return 5 # * flops / bw


def generate_kernels(sm, name, vecSupported):
    if sm < compute_min_sm(name):
        return []
    elem_input_size = compute_input_size(name)
    elem_acc = compute_acc_size(name)
    instr_shape = compute_instr_shape(sm, name)
    elem_gmem_a = compute_gmem_size(name[0])
    elem_gmem_b = compute_gmem_size(name[1])
    elem_smem_a = compute_smem_elem_size(sm, elem_gmem_a, elem_input_size)
    elem_smem_b = compute_smem_elem_size(sm, elem_gmem_b, elem_input_size)
    SMEM_MIN = 1 * 1024
    SMEM_MAX = compute_smem_max(sm)
    REG_SIZE = 4
    REG_MAX = 256
    TB_REG_MAX = 64 * 1024
    WARP_SIZE = 32
    TB_THREAD_MAX = min(1024, TB_REG_MAX / REG_MAX)
    TB_WARP_MAX = TB_THREAD_MAX / WARP_SIZE
    AI_MIN = compute_min_ai(instr_shape)
    #print(AI_MIN)
    result = []
    possibleCandidates = list(enumerateKernels(256, 32, 64))
    random.seed(0)
    random.shuffle(possibleCandidates) # randomly permute s.t. we keep random kernels for fallback vectorization

    target_vec = 128 // (elem_gmem_a * 8)
    target_vectors = [] # possible vectorization values
    target_vectors.append(1) #fallback
    if (target_vec != 1):
        target_vectors.append(target_vec)
    if (target_vec > 2): #another high-perf fallback
        target_vectors.append(2)
    if (elem_gmem_a != elem_gmem_b):
        target_vectors = [1]

    numFallbacks = {} # keeps track of number of fallback kernels for a given vectorization
    for vec in target_vectors:
        vec_a = vec
        vec_b = vec
        if( sm == 80 and (name == 'hhhs' or name == 'bbbs') and (vec_a < 2 or vec_b < 2)):
            continue # not supported we need vectorization (due to ldgsts)
        for tb_n, tb_m, tb_k, warp_n, warp_m in possibleCandidates:
            n_warps = tb_n * tb_m // (warp_n * warp_m)
            if n_warps > TB_WARP_MAX: continue
            if tb_n * tb_k / vec_a < n_warps * WARP_SIZE: continue # LDG limit
            if tb_m * tb_k / vec_b < n_warps * WARP_SIZE: continue # LDG limit
            if warp_n < instr_shape[0]: continue
            if warp_m < instr_shape[1]: continue
            if tb_k < instr_shape[2]: continue
            if tb_k > 8 * instr_shape[2]: continue
            if min(tb_n, tb_m) * 2 < max(tb_n, tb_m): continue # tb skew limit
            if min(warp_n, warp_m) * 2 < max(warp_n, warp_m): continue # warp skew limit
            if warp_n // instr_shape[0] * elem_smem_a * 8 < 32: continue # LDS
            if warp_m // instr_shape[1] * elem_smem_b * 8 < 32: continue # LDS
            if (not vecSupported) and (vec_a > 1 or vec_b > 1):
                continue # vectorization is not yet supported for SIMT iterators
            if sm == 80 and instr_shape == (8, 8, 4):
                if tb_n <= 32: continue
                if tb_m <= 32: continue
                if warp_n < 16: continue
                if warp_m < 16: continue
                if tb_k < 16: continue
                if n_warps == 1: continue
            if sm == 80 and instr_shape == (16, 8, 8):
                if tb_n <= 32: continue
                if tb_m <= 32: continue
                if warp_n < 32: continue
                if warp_m < 32: continue
                if tb_k < 32: continue
                if n_warps == 1: continue
            if sm in [70, 75]:
                if instr_shape == (8, 8, 4):
                    if tb_n <= 32: continue
                    if tb_m <= 32: continue
                    if warp_n < 32: continue
                    if warp_m < 32: continue
                    if tb_k < 32: continue
                    if n_warps == 1: continue
                elif instr_shape == (1, 1, 1):
                    if tb_n >= 256: continue
                    if tb_m >= 256: continue
                    if tb_m <= 32 and tb_n <= 32: continue
                    if warp_m <= 16 and warp_n <= 16: continue
            if ((tb_m >= 256 or tb_n >= 256) and vec == 1):
                continue # This is a crude way to avoid using too many predicates
            smem = elem_smem_a * tb_n * tb_k + elem_smem_b * tb_m * tb_k
            if smem < SMEM_MIN: continue
            if smem > SMEM_MAX: continue
            ai = tb_n * tb_m * tb_k / (tb_n * tb_k + tb_m * tb_k + tb_m * tb_n)
            if ai < AI_MIN: continue
            #limit the number of fallback kernels
            if (vecSupported and vec == 1 and target_vec != 1):
                if (not 1 in numFallbacks):
                    numFallbacks[1] = 0
                if (numFallbacks[1] > 4):
                    continue
                else:
                    numFallbacks[1] += 1
            if (vecSupported and vec == 2 and target_vec != 2):
                if (not 2in numFallbacks):
                    numFallbacks[2] = 0
                if (numFallbacks[2] > 8):
                    continue
                else:
                    numFallbacks[2] += 1
            acc = warp_n * warp_m // WARP_SIZE * elem_acc // REG_SIZE
            load_a = tb_n * tb_k * elem_gmem_a // WARP_SIZE // n_warps / REG_SIZE
            load_b = tb_m * tb_k * elem_gmem_b // WARP_SIZE // n_warps / REG_SIZE
            # if acc > REG_MAX: continue
            #print(tb_n, tb_m, tb_k, warp_n, warp_m, vec_a, vec_b, ai, acc, load_a, load_b)
            result.append(((tb_n, tb_m, tb_k), (warp_n, warp_m, tb_k), instr_shape, (vec_a, vec_b)))
    print(sm, name, len(result))
    return result


def targetForGenerate(l, sm, type_name):
    if sm != 80: return
    # return # ATTN: Uncomment to allow for kernel generation
    if not l: return
    if type_name not in ['ddds', 'dddd', 'sssh', 'ssss', 'ssst', 'bbbs', 'hhhs', 'ccct', 'cccc', 'zzzz', 'zzzc']: return
    # ELW = 'LWTENSOR_GENERATE_TARGET'
    # if ELW not in os.elwiron: return
    # step = os.elwiron[ELW]
    # step = int(step)
    step = 0
    template = l[-1]
    l.clear()
    for transA in ['true', 'false']:
        for transB in ['true', 'false']:
            vecSupported = ((not template.isSimt()) or transA == 'false' and transB == 'true') and not (type_name == 'dddd' and sm == 80)
            g = generate_kernels(sm, type_name, vecSupported)
            for tb, warp, instr, vec in g[step*50:step*50+50]:
                template_copy = copy.deepcopy(template)
                if( type_name == "hhhs"):
                    template_copy.shapeK = [32,1]
                else:
                    template_copy.shapeK = [8,1]
                template_copy.threadblockShape = list(tb)
                template_copy.warpShape = list(warp)
                template_copy.instructionShape = list(instr)
                template_copy.elementsPerAccessA = vec[0]
                template_copy.elementsPerAccessB = vec[1]
                template_copy.transformA = UnaryTransform.IDENTITY
                template_copy.transformB = UnaryTransform.IDENTITY
                template_copy.stridedLoadsA = "false"
                template_copy.stridedLoadsB = "false"
                template_copy.transA = transA
                template_copy.transB = transB
                if any([template_copy.getTemplateArguments() == elem.getTemplateArguments() for elem in l]): continue
                print(template_copy.encode(transA, transB))
                l.append(template_copy)

def generateCandidatesNew(typesString, dataTypes, cc, candidates, use_toggle_heuristic=False):
    if (cc, dataTypes) in typesUsingSassFeatures:
        use_toggle_heuristic = True
    code  = ""
    code += "static CandidateContainerTyped<\n"
    code += "    Traits, ContractionDescriptorInternal, CandidateInfoLwtlass,"
    if (use_toggle_heuristic):
        code += "\n#if defined(__x86_64__) || defined(_MSC_VER) // can be removed once DNN heuristic is enabled for ARM and IBM as well\n"
        code += f" HeuristicToggle<ContractionWeights{heuristicDnnWeightsSuffix[dataTypes]}>\n"
        code += "#else\n"
    code += " HeuristicSimple\n"
    if (use_toggle_heuristic):
        code += "#endif\n"
    definition = "CandidateContainer<ContractionDescriptorInternal>* getContractionContainer_lwtlass_sm%d_%s()"%(cc, typesString)
    encodings = []
    for idx, candidate in enumerate(candidates):
        code += ",CandidateLwtlass<%s>\n" % (candidate.getTemplateArguments(use_sass_features=use_toggle_heuristic))
    code += ">candidatesLwtlass_sm%d_%s;\n" % (cc, typesString)
    code += definition
    code += "\n{\n"
    code += "   return &candidatesLwtlass_sm%d_%s;\n" % (cc, typesString)
    code += "}\n"
    return code, encodings

def generateDispatchNew(dataTypes, cc, allowedCandidates, alreadyDispatched, traits, use_sass_features=False):
    code  = ""

    if (cc, dataTypes) in typesUsingSassFeatures:
        use_sass_features = True

    for idx, candidate in enumerate(allowedCandidates):
                template_args = candidate.getTemplateArguments(use_sass_features=use_sass_features)
                kernel_name = candidate.getKernelName()
                launch_bounds = ""
                minCTAs = candidate.getLaunchBoundsMinCTAs()
                if minCTAs:
                    launch_bounds = "__launch_bounds__(%d, %d) " % (candidate.getNumThreads(), minCTAs)
                code += "\nnamespace LWTENSOR_NAMESPACE {\n"
                code += "  struct KernelParam_%s { typename ::LWTENSOR_NAMESPACE::CandidateLwtlass<%s>::Contraction::Gett::GettKernel::Params params; };\n" % (kernel_name, template_args)
                code += "  %s__global__ void contraction_kernel(KernelParam_%s params) { \n" % (launch_bounds, kernel_name)
                code += "    lwtlass::KernelSpecializationDevice<typename ::LWTENSOR_NAMESPACE::CandidateLwtlass<%s>::Contraction::Gett::GettKernel>(params.params);\n" % (template_args)
                code += "  }\n"
                code += "}\n"
                key = template_args + "," + traits
                if key not in alreadyDispatched:
                    if (candidate.useNewIterator()):
                        code += "\n\nnamespace lwtlass { namespace contraction { namespace device {\n"
                    else:
                        code += "\n\nnamespace lwtlass { namespace contraction { namespace device {\n"
                    code += "\n  template<>\n  void dispatch_lwtlass_kernel<typename ::LWTENSOR_NAMESPACE::CandidateLwtlass<%s>::Contraction::Gett::GettKernel>(int threadblockCount, int threadCount, int smem_size, lwdaStream_t stream,\n" % (template_args)
                    code += " const ::LWTENSOR_NAMESPACE::CandidateLwtlass<%s>::Contraction::Gett::GettKernel::Params &params) {\n"%(template_args)
                    code += "    LWTENSOR_NAMESPACE::contraction_kernel<<<threadblockCount, threadCount, smem_size, stream>>>(\n      LWTENSOR_NAMESPACE::KernelParam_%s {params});\n" % (kernel_name)
                    code += "  }\n"
                    code += "\n  template<>\n  void* lookup_lwtlass_kernel<typename ::LWTENSOR_NAMESPACE::CandidateLwtlass<%s>::Contraction::Gett::GettKernel>() {\n"%(template_args)
                    code += "  return reinterpret_cast<void*>(static_cast<void (*)(::LWTENSOR_NAMESPACE::KernelParam_%s)>(::LWTENSOR_NAMESPACE::contraction_kernel));" % (kernel_name)
                    code += "  }\n"
                    code += "} } }\n"
                    alreadyDispatched.add(key)
                else:
                    print("already dispatched:",kernel_name, template_args)

    return code

def genMaxAlogs(allowedCandidates):
    code  = "extern \"C\" EXPORT_SYMBOL"
    code += "\n"
    code += "lwtensorStatus_t lwtensorContractionMaxAlgos(int32_t *maxNumAlgos) {\n"
    code += "\n"

    maxNumAlgos = 0
    for cc in allowedCandidates:
        for dataTypes in allowedCandidates[cc]:
            maxNumAlgos = max( maxNumAlgos, len(allowedCandidates[cc][dataTypes]))

    code += "    if (maxNumAlgos == nullptr)\n"
    code += "    {\n"
    code += "        return LWTENSOR_STATUS_ILWALID_VALUE;\n"
    code += "    }\n"
    code += "\n"
    code += "    *maxNumAlgos = %d;\n"%maxNumAlgos
    code += "\n"
    code += "    return LWTENSOR_STATUS_SUCCESS;\n"
    code += "}\n"
    return code

def generateDispatcherGeneric(allowedCandidates, typeOrder):
    code  = "/* This file is AUTO-GENERATED by generateContraction.py */\n"
    code += "#include <lwtensor/internal/types.h>\n"
    code += "#include <lwtensor/internal/export.h>\n"
    code += "#include<lwtensor/internal/computeEngine.h>\n"
    code += "#include<lwtensor/internal/defines.h>\n"
    code += "#include<lwtensor/internal/heuristicsLwtlass.h>\n"
    code += "namespace LWTENSOR_NAMESPACE\n{\n"

    for cc in sorted(allowedCandidates):
        for dataTypes in sorted(allowedCandidates[cc]):
            if (len(allowedCandidates[cc][dataTypes]) <= 0):
                continue
            typeA = dataTypes[0]
            typeB = dataTypes[1]
            typeC = dataTypes[2]
            typeComp = dataTypes[3]
            typesString = typeToChar(dataTypes[0]) + typeToChar(dataTypes[1]) + typeToChar(dataTypes[2]) + typeToChar(dataTypes[3]) + dataTypes[5]

            code += "  CandidateContainer<ContractionDescriptorInternal>* getContractionContainer_lwtlass_sm%d_%s();\n"%(cc, typesString)
    code += "\n"
    declaration = "ComputeEngineBase<ContractionDescriptorInternal>* getContractionEngineLwtlass()"

    initializerList = ""

    kNumContainersLWDA11= 0
    kNumContainers = 0

    ccFuture = 61 # the minimal compute arch is the one for which we supply ptx code (i.e., the one that is used for future dispatch)
    isFirst = True
    #for cc in allowedCandidates:
    computeCapabilities = sorted(allowedCandidates)
    computeCapabilities.remove(ccFuture)
    computeCapabilities.append(ccFuture) # move to last
    for cc in computeCapabilities:
        for dataTypes in sorted(allowedCandidates[cc], key=lambda d: typeOrder.index(d)):
            if (len(allowedCandidates[cc][dataTypes]) <= 0):
                continue
            typeA = dataTypes[0]
            typeB = dataTypes[1]
            typeC = dataTypes[2]

            typeScalar = dataTypes[4]
            typesString = typeToChar(dataTypes[0]) + typeToChar(dataTypes[1]) + typeToChar(dataTypes[2]) + typeToChar(dataTypes[3]) + dataTypes[5]
            containerVariableName = "lwtlass_sm%d_%s"%(cc, typesString)
            if cc >= 80:
                initializerList += '#if LWTENSOR_LWDA_VERSION_MAJOR >= 11\n'
            comma = ","
            if isFirst:
                comma = ""
                isFirst = False
            initializerList += "   %sgetContractionContainer_lwtlass_sm%d_%s()\n"%(comma, cc, typesString)
            if cc >= 80:
                initializerList += '#endif\n'
            else:
                kNumContainers += 1
            kNumContainersLWDA11 += 1
    code += "\n"

    code += '#if LWTENSOR_LWDA_VERSION_MAJOR >= 11\n'
    code += 'static const int kNumContainers = %d;\n'%(kNumContainersLWDA11)
    code += '#else\n'
    code += 'static const int kNumContainers = %d;\n'%(kNumContainers)
    code += '#endif\n'

    code += "static ComputeEngine<kNumContainers, ContractionDescriptorInternal> contractionEngineLwtlass({\n"
    code += initializerList
    code += "});\n\n"

    code += declaration + "\n{\n"
    code += "   contractionEngineLwtlass.init();\n"
    code += "   return &contractionEngineLwtlass;\n"
    code += "}\n"

    code += "} // end of namespace\n"

    code += "\n#pragma GCC diagnostic pop\n"

    code += genMaxAlogs(allowedCandidates)

    return code

def updateFile(name, content):
    old_content = None
    try:
        with open(name, "r") as f:
            old_content = f.read()
    except:
        pass
    if content == old_content:
        return
    with open(name, "w") as f:
        f.write(content)

def genFilesNew(candidates, use_sass_features):
    alreadyDispatched = set()
    for cc in sorted(candidates):
        for dataTypes in sorted(candidates[cc]):
            typeA      = dataTypes[0]
            typeB      = dataTypes[1]
            typeC      = dataTypes[2]
            typeComp   = dataTypes[3]
            typeScalar = dataTypes[4]
            typesString = typeToChar(typeA) + typeToChar(typeB) + typeToChar(typeC) + typeToChar(typeComp) + dataTypes[5]
            if (len(candidates[cc][dataTypes]) == 0):
                updateFile("./src/tensorContraction_sm%d_%s.lw"%(cc,typesString), "")
                continue
            print("./src/tensorContraction_sm%d_%s.lw"%(cc,typesString))
            code  = "/************************************\n"
            code += " * This file is AUTO-GENERATED by generateContractionLwtlass.py\n"
            code += "************************************/\n"
            code += "#pragma GCC diagnostic push\n"
            code += "#pragma GCC diagnostic ignored \"-Wunused-parameter\"\n"
            code += "#include<lwtensor/internal/candidateContainer.h>\n"
            if (candidates[cc][dataTypes][0].useNewIterator()):
                code += "#include<lwtlass/contraction/device/gett.h>\n"
                code += "#include<lwtensor/internal/candidateLwtlass.h>\n"
            else:
                code += "#include<lwtlass/contraction/device/gett.h>\n"
                code += "#include<lwtensor/internal/candidateLwtlass.h>\n"
            code += "#include<lwtensor/internal/defines.h>\n"
            code += "#include<lwtensor/internal/heuristicsLwtlass.h>\n"
            if use_sass_features or (cc, dataTypes) in typesUsingSassFeatures:
                code += f"#include<lwtensor/internal/dnnContractionWeights{heuristicDnnWeightsSuffix[dataTypes]}.h>\n"
            minTypeComp = candidates[cc][dataTypes][0].getMinComputeType(True)
            opClass = candidates[cc][dataTypes][0].opClass
            mathTag = candidates[cc][dataTypes][0].MathOperatorTag
            for idx, candidate in enumerate(candidates[cc][dataTypes]):
                if (opClass != candidate.opClass or mathTag != candidate.MathOperatorTag):
                    print(cc,dataTypes,opClass , candidate.opClass,mathTag , candidate.MathOperatorTag)
                    print("ERROR: opClass and mathOperatorTag within the same container must match")
                    exit(-1)
            traits = "%s, %s, %s, %s, %s, %s, %s, %s, %d, %d, %d"%(typeA, typeB, typeC, typeScalar, typeComp, minTypeComp, opClass, mathTag, cc, min(binaryCompatibility[cc]), max(binaryCompatibility[cc]))
            code += "using Traits = LWTENSOR_NAMESPACE::ContractionTraits<%s>;\n"%(traits)
            code += "namespace LWTENSOR_NAMESPACE {\n"
            tcode , encodings = generateCandidatesNew(typesString, dataTypes, cc, candidates[cc][dataTypes], use_sass_features)
            code += tcode + "} // end namespace\n"
            code += "#ifdef LWTLASS_CONTRACTION_KERNEL_RENAME\n"
            code += generateDispatchNew(dataTypes, cc, candidates[cc][dataTypes], alreadyDispatched, traits, use_sass_features)
            code += "#endif\n"
            updateFile("./src/tensorContraction_sm%d_%s.lw"%(cc,typesString), code)

def decodeFile(filename, use_sass_features):
    candidates = []
    fi = open(filename, "r")
    for encoding in fi:
        if (encoding.startswith("#")): # skip commented kernels
            continue
        can = CandidateLwtlass(encoding, use_sass_features)
        if (can.isValid()):
            candidates.append(can)
    return candidates


def main(kernels, use_sass_features, dont_fallback, include_kernels):

    typeOrder = ["hhhs", "bbbs", "sssh", "sssb", "ssst", "ssss", "dddt", "ddds", "dddd", "ccct", "cccc", "zzzc", "zzzz", "zdzz", "dzzz"]
    typeOrder = [supportedTypes[t] for t in typeOrder]

    #return genOld(typeOrder)

    candidates = {}
    if kernels is None:
        files = glob.glob('./misc/kernels/fallback/tc.*.kernels')
    else:
        files = [kernels]

    for filename in files:
        fallbackKernels = []
        fullfilled = set([None])
        if '/fallback/' in filename:
            if os.path.exists(filename.replace('fallback/', '')):
                if not dont_fallback:
                    fallbackKernels = decodeFile(filename, use_sass_features)
                filename = filename.replace('fallback/', '')

        candidatesTmp = decodeFile(filename, use_sass_features) # those are the kernels that are selected in a data-driven way
        if (len(candidatesTmp) > 0):
            # add all data driven kernels
            for candidate in candidatesTmp:
                cc = candidate.ccTarget
                dataTypes = candidate.dataTypes
                if (not cc in candidates):
                    candidates[cc] = {}
                if (not dataTypes in candidates[cc]):
                    candidates[cc][dataTypes] = []
                candidates[cc][dataTypes].append(candidate)
                fullfilled.add(candidate.getFallbackCode())

            if (dont_fallback):
                continue
            # Use the first data driven kernel to generate all required fallbacks
            # Rationale: This is the best kernel (on average)
            cc = candidatesTmp[0].ccTarget
            dataTypes = candidatesTmp[0].dataTypes
            encoding = candidatesTmp[0].encode(candidatesTmp[0].transA, candidatesTmp[0].transB)

            for stridedB in ["true","false"]:
                if candidatesTmp[0].useNewIterator() and stridedB == "true":
                    continue
                for transA in ["true","false"]:
                    for transB in ["true","false"]:
                        candidate = CandidateLwtlass(encoding, use_sass_features)
                        candidate.elementsPerAccessA = 1
                        candidate.elementsPerAccessB = 1
                        candidate.elementsPerAccessC = 1
                        if (dataTypes[0] == 'lwtlass::bfloat16_t'): # vectorization of one is not supported for bfloat
                            candidate.elementsPerAccessA = 2
                            candidate.elementsPerAccessB = 2
                            candidate.elementsPerAccessC = 2
                        candidate.transA = transA
                        candidate.transB = transB
                        candidate.stridedLoadsB = stridedB
                        fbc = candidate.getFallbackCode()
                        if fbc not in fullfilled:
                            fullfilled.add(fbc)
                            candidates[cc][dataTypes].append(candidate)

        else:
            for candidate in fallbackKernels:
                cc = candidate.ccTarget
                dataTypes = candidate.dataTypes
                if (not cc in candidates):
                    candidates[cc] = {}
                if (not dataTypes in candidates[cc]):
                    candidates[cc][dataTypes] = []
                fbc = candidate.getFallbackCode()
                if fbc not in fullfilled:
                    fullfilled.add(fbc)
                    candidates[cc][dataTypes].append(candidate)

    for sm in candidates:
        for types in candidates[sm]:
            if include_kernels is None:
                matched = True
            else:
                matched = False
                for elem in include_kernels.split(','):
                    if "_" in elem:
                        include_sm, include_type = elem.split('_')
                        if sm == int(include_sm) and types == supportedTypes[include_type]:
                            matched = True
                    else:
                        include_sm = elem
                        if sm == int(include_sm):
                            matched = True
            if not matched:
                candidates[sm][types] = []

    genFilesNew(candidates, use_sass_features)

    if kernels is None:
        dispatcher = generateDispatcherGeneric(candidates, typeOrder)
        updateFile("./src/tensorContraction_lwtlass_auto.lw", dispatcher)


if __name__ == "__main__":
    args = handleArguments()
    main(args.kernels, args.use_sass_features, args.dont_fallback, args.include_kernels)
