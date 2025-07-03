from lwtensorUtil import *
from enum import Enum

from functools import reduce

class UnaryTransform(Enum):
    IDENTITY = 1
    CONJUGATE = 2
    def __str__(self):
        if ( self.name == 'IDENTITY' ):
            return "typename lwtlass::transform::thread::UnaryTransform::Identity"
        elif ( self.name == 'CONJUGATE'):
            return "typename lwtlass::transform::thread::UnaryTransform::Conjugate"
        else:
            print("Unknown operator.")
            exit(-1)
            
binaryCompatibility = {}
binaryCompatibility[61] = {60, 10000}
binaryCompatibility[70] = {70, 72}
binaryCompatibility[75] = {75}
binaryCompatibility[80] = {80, 89}

typeLookup = {'lwtlass::half_t': 'half',
              'float': 'float',
              'double': 'double',
              'lwtlass::complex<float>': 'complex_float',
              'lwtlass::complex<double>': 'complex_double',
              'lwtlass::bfloat16_t': 'bf16',
              'lwtlass::tfloat32_t': 'tf32'}
charToType = {'h':'lwtlass::half_t',
        's':'float',
        'd' :'double',
        'c' : 'lwtlass::complex<float>',
        'z' : 'lwtlass::complex<double>',
        'b' : 'lwtlass::bfloat16_t',
        't' :'lwtlass::tfloat32_t'}
opLookup = {'lwtlass::arch::OpClassTensorOp': 'tensor', 'lwtlass::arch::OpClassSimt': 'simt'}
opToInt = {'lwtlass::arch::OpClassTensorOp': 0, 'lwtlass::arch::OpClassSimt': 1}
archLookup = {'lwtlass::arch::Sm50': 'simt', 'lwtlass::arch::Sm70': 'tc8', 'lwtlass::arch::Sm75': 'tc16', 'lwtlass::arch::Sm80': 'tcd'}
archToInt = {'lwtlass::arch::Sm50': 0, 'lwtlass::arch::Sm70': 1, 'lwtlass::arch::Sm75': 2, 'lwtlass::arch::Sm80': 3}
fmaLookup = {'lwtlass::arch::OpMultiplyAdd' : 'fma', 'lwtlass::arch::OpMultiplyAddComplex' : 'cfma', 'lwtlass::arch::OpMultiplyAddFastF16' : 'half', 'lwtlass::arch::OpMultiplyAddFastBF16' : 'bfloat', }
fmaToInt = {'lwtlass::arch::OpMultiplyAdd' : 0, 'lwtlass::arch::OpMultiplyAddComplex' : 1, 'lwtlass::arch::OpMultiplyAddFastF16' : 2, 'lwtlass::arch::OpMultiplyAddFastBF16' : 3, }
transformLookup = {UnaryTransform.IDENTITY : 'iden', UnaryTransform.CONJUGATE : 'conj'}
transformToInt = {UnaryTransform.IDENTITY : 1, UnaryTransform.CONJUGATE : 9} # WARNING: IT has to be ensured that this mapping matches that of lwtensorOperator_t

def valueToKey(lookup, val):
    for key, v in lookup.items():
        if (v == val):
            return key

def intToBool(val):
    if (val == 1):
        return "true"
    elif (val == 0):
        return "false"
    else:
        print("intToBool() failed")
        exit(0)

def toIntTuple(shape):
    ret = "lwtlass::contraction::IntTuple<"
    for s in shape:
        ret += "lwtlass::contraction::Int<%d>, "%s
    return ret[:-2] + ">"

FEATURES = [
                                                    #A          B          C          D          Comp       scalar     tcpostfix
        dict(name="dataTypes",              default=("unknown", "unknown", "unknown", "unknown", "unknown", "unknown", ""), enable="base"),
        dict(name="ccTarget",               default="unknown",  enable="base"),
        dict(name="threadblockShape",       default="unknown",  enable="base"),
        dict(name="shapeK",                 default="unknown",  enable="base"),
        dict(name="warpShape",              default="unknown",  enable="base"),
        dict(name="instructionShape",       default="unknown",  enable="base"),
        dict(name="elementsPerAccessA",     default="unknown",  enable="base"),
        dict(name="transformA",             default="unknown",  enable="base"),
        dict(name="stridedLoadsA",          default="unknown",  enable="base"),
        dict(name="blockedModesM",          default=2,          enable="base"),
        dict(name="elementsPerAccessB",     default="unknown",  enable="base"),
        dict(name="transformB",             default="unknown",  enable="base"),
        dict(name="stridedLoadsB",          default="unknown",  enable="base"),
        dict(name="blockedModesN",          default=2,          enable="base"),
        dict(name="elementsPerAccessC",     default="unknown",  enable="base"),
        dict(name="opClass",                default="unknown",  enable="base"),
        dict(name="archTag",                default="unknown",  enable="base"),
        dict(name="MathOperatorTag",        default="unknown",  enable="base"),
        dict(name="kMaxContractedModes",    default=8,          enable="base"),
        dict(name="transA",                 default="unknown",  enable="base"),
        dict(name="transB",                 default="unknown",  enable="base"),
        dict(name="launchboundsMinCTAs",    default=None,       enable="base"),

        dict(name="loc_mem",                default=0,  enable="sass"),
        dict(name="wait_schedule",          default=0,  enable="sass"),
        dict(name="avg_lds",                default=0,  enable="sass"),
        dict(name="avg_ldg",                default=0,  enable="sass"),
        dict(name="avg_anti",               default=0,  enable="sass"),
    ]

class CandidateLwtlass:
    def __init__(self, encoding, use_sass_features):

        # workaround, because custom __setattr__
        features = {}
        for feat in FEATURES:
            features[feat["name"]] = feat["default"]

        super().__setattr__("_features", features)

        if use_sass_features:
            self.enable = ["base", "sass"]
        else:
            self.enable = ["base"]

        self.decode(encoding)
        #update tensorcore postfix
        dt = self.dataTypes
        self.dataTypes = (dt[0], dt[1], dt[2], dt[3], dt[4], self.getTensorCorePostfix())

        if (not self.isValid()):
            print("ERROR: candidate is invalid.")
            self.print()
            exit(-1)

    def useNewIterator(self):
        return True
        if self.ccTarget == 80 and self.dataTypes[0] == "double" and self.dataTypes[1] == "double" and self.dataTypes[2] == "double":
            return False
        else:
            return True

    def __getattr__(self, attr):
        if attr != "_features" and attr in self._features:
            return self._features[attr]
        return super().__getattr__(attr)

    def __setattr__(self, attr, val):
        if attr in self._features:
            self._features[attr] = val
        else:
            super().__setattr__(attr, val)

    def print(self):
        print([self._features[feat["name"]] for feat in FEATURES])

    def kShapeIsValid(self):
        if(reduce(lambda x,y:x*y,self.shapeK) != self.threadblockShape[2]):
            print("Error: Invalid k-blocking")
            return False
        return True 

    def isValid(self):
        # checks if all features are set
        for feat in FEATURES:
            if feat["enable"] in self.enable:
                if self._features[feat["name"]] == "unknown":
                    return False
        return True

    def getMinComputeType(self, codeGenFlag = False):
        typeComp = self.getComputeType()
        if ('lwtlass::arch::OpClassTensorOp' == self.opClass):
            if (self.ccTarget >= 80):
                if (self.dataTypes[0] == 'lwtlass::bfloat16_t'):
                    if(codeGenFlag):
                        typeComp = "float"
                    else:
                        typeComp = "lwtlass::bfloat16_t"
                elif(self.dataTypes[0] == "float" and self.MathOperatorTag == "lwtlass::arch::OpMultiplyAddFastBF16"):
                    typeComp = "lwtlass::bfloat16_t"
                elif (self.dataTypes[0] == "lwtlass::half_t"):
                    if(codeGenFlag):
                        typeComp = "float"
                    else:
                        typeComp = "lwtlass::half_t"
                elif(self.dataTypes[0] == "float" and self.MathOperatorTag == "lwtlass::arch::OpMultiplyAddFastF16"):
                    typeComp = "lwtlass::half_t"
                elif(typeComp != "double" and typeComp != "lwtlass::complex<double>"):
                    typeComp = "lwtlass::tfloat32_t"
            else:
                if(self.dataTypes[0] == "lwtlass::half_t"):
                    typeComp = "float"
                else:
                    typeComp = "lwtlass::half_t"
        return typeComp

    def getComputeType(self):
        return self.dataTypes[3]

    def getTensorCorePostfix(self):
        """
        tensorCorePostfix could be _tc, _tc_bf16, _tc_tf32
        """
        tensorCorePostfix= ""
        
        if (self.ccTarget < 80):
            if ('lwtlass::arch::OpClassTensorOp' == self.opClass):
                tensorCorePostfix += "_tc"
        else:
            if (self.getMinComputeType() == "lwtlass::bfloat16_t"):
                tensorCorePostfix += "_tc_bf16"
            elif (self.getMinComputeType() == "lwtlass::half_t"):
                tensorCorePostfix += "_tc"
            elif(self.getMinComputeType() == "lwtlass::tfloat32_t"):
                tensorCorePostfix += "_tc_tf32"

        return tensorCorePostfix

    def isSimt(self):
        return (self.opClass.find("Simt") != -1);

    def getNumThreads(self):
        return 32 * self.threadblockShape[0] / self.warpShape[0] * self.threadblockShape[1] / self.warpShape[1]

    def getScalarType(self):
        return self.dataTypes[4]

    def getTypeString(self):
        ret = ""
        ret += valueToKey(charToType, self.dataTypes[0])
        ret += valueToKey(charToType, self.dataTypes[1])
        ret += valueToKey(charToType, self.dataTypes[2])
        ret += valueToKey(charToType, self.dataTypes[3])
        ret += self.getTensorCorePostfix()
        return ret

    def getKernelName(self, ta = "dummy", tb = "dummy"):
        typeScalar = self.getScalarType()
        transA = self.transA
        if (ta != "dummy"):
            transA = ta
        transB = self.transB
        if (tb != "dummy"):
            transB = tb
        args = "" 
        args += "%s_%s_%d_%d_%s_%s_"%(typeLookup[self.dataTypes[0]], transformLookup[self.transformA], self.elementsPerAccessA, self.blockedModesM, self.transA, self.stridedLoadsA)
        args += "%s_%s_%d_%d_%s_%s_"%(typeLookup[self.dataTypes[1]], transformLookup[self.transformB], self.elementsPerAccessB, self.blockedModesN, self.transB, self.stridedLoadsB)
        args += "%s_%d_"%(typeLookup[self.dataTypes[2]], self.elementsPerAccessC)
        args += "%s_%s_"%(typeLookup[typeScalar], typeLookup[self.dataTypes[3]])
        args += "tb_%d_%d_%d_"%(self.threadblockShape[0],self.threadblockShape[1],self.threadblockShape[2])
        args += "k_"
        for k in self.shapeK:
            args += "%d_"%k
        args += "warp_%d_%d_%d_"%(self.warpShape[0],self.warpShape[1],self.warpShape[2])
        #args += "ins_%d_%d_%d_"%(self.instructionShape[0],self.instructionShape[1],self.instructionShape[2])
        args += "%s_%s_%d"%(opLookup[self.opClass], archLookup[self.archTag], self.getNumThreads())
        args += "_%s_%s"%(fmaLookup[self.MathOperatorTag], self.ccTarget)
        return args

    def encode(self, transA, transB):
        """
        - this information is used by the selectKernels.py script
        - this function should match candidateLwtlass.h::info()
        """
        ret = ""
        ret += "tb:%d,%d,%d;"%(self.threadblockShape[0],self.threadblockShape[1],self.threadblockShape[2])
        ret += "k:"
        for k in self.shapeK:
            ret += "%d,"%k
        ret = ret[:-1] + ";" #remove last comma
        ret += "w:%d,%d,%d;"%(self.warpShape[0],self.warpShape[1],self.warpShape[2])
        ret += "is:%d,%d,%d;"%(self.instructionShape[0],self.instructionShape[1],self.instructionShape[2])
        ret += "a:%d,%d,%d;"%(self.elementsPerAccessA, self.elementsPerAccessB, self.elementsPerAccessC)
        ret += "s:%d,%d;"%(strToBool(self.stridedLoadsA), strToBool(self.stridedLoadsB))
        ret += "t:%d,%d;"%(strToBool(transA), strToBool(transB))
        ret += "bf:%d,%d;"%(self.blockedModesM, self.blockedModesN)
        ret += "op:%d,%d;"%(transformToInt[self.transformA], transformToInt[self.transformB])
        ret += "cc:%d;"%(self.ccTarget)
        ret += "ar:%d;"%(archToInt[self.archTag])
        ret += "oc:%d;"%(opToInt[self.opClass])
        ret += "fm:%d;"%(fmaToInt[self.MathOperatorTag])
        ret += "tp:%c,%c,%c,%c,%c;"%(typeToChar(self.dataTypes[0]), typeToChar(self.dataTypes[1]), typeToChar(self.dataTypes[2]), typeToChar(self.getScalarType()), typeToChar(self.dataTypes[3]))
        if self.launchboundsMinCTAs is not None:
            ret += "lbmc:%d;" % self.launchboundsMinCTAs
        if ("sass" in self.enable):
            ret += "lmem:%s;wa:%s;ls:%s;lg:%s;la:%s;"%(self.loc_mem, self.wait_schedule, self.avg_lds, self.avg_ldg, self.avg_anti)
        return ret

    def decode(self, encoding):
        """
        Initializes this candidate based on the provided encoding (which has to match that
        provided by encode()
        """
        for tok in encoding.split(';'):
            if (tok.startswith("tb:")):
                tb = findAndSplit("tb:", tok, [','])
                self.threadblockShape = [0,0,0]
                self.threadblockShape[0] = int(tb[0])
                self.threadblockShape[1] = int(tb[1])
                self.threadblockShape[2] = int(tb[2])
            if (tok.startswith("w:")):
                w = findAndSplit("w:", tok, [','])
                self.warpShape = [0,0,0]
                self.warpShape[0] = int(w[0])
                self.warpShape[1] = int(w[1])
                self.warpShape[2] = int(w[2])
            if (tok.startswith("is:")):
                val = findAndSplit("is:", tok, [','])
                self.instructionShape = [0,0,0]
                self.instructionShape[0] = int(val[0])
                self.instructionShape[1] = int(val[1])
                self.instructionShape[2] = int(val[2])
            if (tok.startswith("a:")):
                a = findAndSplit("a:", tok, [','])
                self.elementsPerAccessA = int(a[0])
                self.elementsPerAccessB = int(a[1])
                self.elementsPerAccessC = int(a[2])
            if (tok.startswith("s:")):
                s = findAndSplit("s:", tok, [','])
                self.stridedLoadsA = intToBool(int(s[0]))
                self.stridedLoadsB = intToBool(int(s[1]))
            if (tok.startswith("t:")):
                t = findAndSplit("t:", tok, [','])
                self.transA = intToBool(int(t[0]))
                self.transB = intToBool(int(t[1]))
            if (tok.startswith("op:")):
                ops = findAndSplit("op:", tok, [','])
                self.transformA = valueToKey(transformToInt, int(ops[0]))
                self.transformB = valueToKey(transformToInt, int(ops[1]))
            if (tok.startswith("ar:")):
                val = findAndSplit("ar:", tok)
                self.archTag = valueToKey(archToInt, int(val))
            if (tok.startswith("oc:")):
                val = findAndSplit("oc:", tok)
                self.opClass = valueToKey(opToInt, int(val))
            if (tok.startswith("fm:")):
                val = findAndSplit("fm:", tok)
                self.MathOperatorTag = valueToKey(fmaToInt, int(val))
            if (tok.startswith("k:")):
                shapeK = findAndSplit("k:", tok,[','])
                self.shapeK = [int(i) for i in shapeK]
            if (tok.startswith("cc:")):
                self.ccTarget= int(findAndSplit("cc:", tok, [','])[0])
            if (tok.startswith("tp:")):
                types = findAndSplit("tp:", tok, [','])
                self.dataTypes = (charToType[types[0]],
                                  charToType[types[1]],
                                  charToType[types[2]],
                                  charToType[types[4]],
                                  charToType[types[3]],
                                  "") #postfix is not necesarrily known at this point
            if (tok.startswith("bf:")):
                self.blockedModesM, self.blockedModesN = map(int, findAndSplit("bf:", tok, [","]))
            if (self.shapeK == "unknown"):
                self.shapeK = [self.threadblockShape[2]] + [1 for i in range(7)]
            if (tok.startswith("lmem:")):
                self.loc_mem = int(findAndSplit("lmem:", tok))
            if (tok.startswith("wa:")):
                self.wait_schedule = int(findAndSplit("wa:", tok))
            if (tok.startswith("ls:")):
                self.avg_lds = int(findAndSplit("ls:", tok))
            if (tok.startswith("lg:")):
                self.avg_ldg = int(findAndSplit("lg:", tok))
            if (tok.startswith("la:")):
                self.avg_anti = int(findAndSplit("la:", tok))
            if tok.startswith("lbmc:"):
                self.launchboundsMinCTAs = int(findAndSplit("lbmc:", tok))


    def getTemplateArguments(self, ta = "dummy", tb = "dummy", use_sass_features = False):
        typeScalar = self.getScalarType()
        transA = self.transA
        if (ta != "dummy"):
            transA = ta
        transB = self.transB
        if (tb != "dummy"):
            transB = tb
        """ This must match the constructor of lwtensor/include/internal/types.h """
        args = "Traits,\n"
        args += "    %s, %d, %d, /*transA*/%s, %s,\n"%(str(self.transformA), self.elementsPerAccessA, self.blockedModesM, transA, self.stridedLoadsA)
        args += "    %s, %d, %d, /*transB*/%s, %s,\n"%(str(self.transformB), self.elementsPerAccessB, self.blockedModesN, transB, self.stridedLoadsB)
        args += "    %d,\n"%(self.elementsPerAccessC)
        args += "    lwtlass::gemm::GemmShape<%d, %d, %d>,\n"%(self.threadblockShape[0],self.threadblockShape[1],self.threadblockShape[2])
        args += "    %s,\n"%(toIntTuple(self.shapeK))
        args += "    lwtlass::gemm::GemmShape<%d, %d, %d>,\n"%(self.warpShape[0],self.warpShape[1],self.warpShape[2])
        args += "    lwtlass::gemm::GemmShape<%d, %d, %d>,\n"%(self.instructionShape[0],self.instructionShape[1],self.instructionShape[2])
        args += "    %s, %d"%(self.archTag, self.getNumThreads())

        if ("sass" in self.enable or use_sass_features):
            args += ", %s, %s, %s, %s, %s"%(self.loc_mem, self.wait_schedule, self.avg_lds, self.avg_ldg, self.avg_anti)
        return args

    def getFallbackCode(self):
        if self.elementsPerAccessA != 1: return None
        if self.elementsPerAccessB != 1: return None
        if self.elementsPerAccessC != 1: return None
        return (self.transA, self.transB, self.stridedLoadsA, self.stridedLoadsB)
        # kernel_name = ""
        # kernel_name += "kernel:"
        # kernel_name += "tb:%d,%d,%d;" % self.threadblockShape
        # kernel_name += "w:%d,%d,%d;" % self.warpShape
        # kernel_name += "a:%d,%d,%d;" % (self.elementsPerAccessA, self.elementsPerAccessB, self.elementsPerAccessC)
        # kernel_name += "s:%d,%d;" % (int(self.stridedLoadsA), int(self.stridedLoadsB))
        # return kernel_name

    def getLaunchBoundsMinCTAs(self):
        return self.launchboundsMinCTAs

