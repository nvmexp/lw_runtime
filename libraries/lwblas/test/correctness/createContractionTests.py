"""
This script allows users to generate a collection of semi-random testcases.

A users has to specify orthogonal test features (e.g., alpha !=0, beta != 0).
These features are then organized into orthogonalFeatures and randomFeatures. All features
that are part of the orthogonalFeatures list will be combined with each other (i.e., every
combination will be tested) where as those featrues that are part of randomFeatures are
only used to randomly perturb the exising tests.

Moreover, each random feature must specify its probability. This way one can favor certain
featrues (or conditions) over others.

Finally, the number of generated testcases can be easily extended by changing the value of
'numTestMultiplier'.


HWO TO:
    - Normal Users: 1) define your orthogonal properties, 2) define random properties 3) call createTest()

    - Developers: 1) Instantiate your own TestCase() class that can handle all user-defined features

"""

import copy
import random

random.seed(0)

mustIncludeTests = [
        # TODO: enable this test once EW doesn't propagate NaNs: "./lwtensorTest -Rcontraction  -extent1=2,2=3,3=2,4=2,  -modeA3,1, -modeB2, -modeC2,1,3 -Pas -Pbs -Pcs -Pcomps  -alpha1  -beta0 -strideC2,6,12,24  -pCn",
        # https://lwbugs/200703470
        "./lwtensorTest -Rcontraction -modeAa,b,c,d,e,f,g,h,i,j,k,l,0,m,n,o,p,r,s -modeB0,q -modeCa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s -extenta=2,b=2,c=2,d=2,e=2,f=2,g=2,h=2,i=2,j=2,k=2,l=2,0=2,m=2,n=2,o=2,p=2,0=2,q=2,r=2,s=2",
        "./lwtensorTest -modeC1,0,3 -alignmentReqA16 -modeA1,0,2,3 -strideA1,5,20,100, -alignmentReqB16 -modeB2,3 -strideB1,4, -useD -alignmentReqD16 -extent0=2,1=2,2=2,3=2, -strideC1,4,16, -alignmentReqC16 -Pah -Pbh -Pch -Pcomph -alpha1.100000 -beta2000000.250000 -Rcontraction  -algo-4  -pCl",
        "./lwtensorTest -modeC0,1 -modeA2 -strideA1, -modeB0,2,1 -strideB1,2,8, -extent0=2,1=2,2=2, -strideC1,5, -opAB3 -opABC3 -opReduce3 -opA1 -opB1 -opC1 -Pah -Pbh -Pch -Pcomph -alpha1.100000 -beta2.200000 -gamma0.000000 -Rcontraction",
        "./lwtensorTest -Rcontraction  -extent1=2,2=3,3=2,4=2,  -modeA4,3,1, -modeB4,2, -modeC2,1,3 -Pas -Pbs -Pcs -Pcomps  -alpha0  -beta2.20 -strideC2,6,12,24 -pCn",
        "./lwtensorTest -Rcontraction  -extent1=2,2=3,3=2,4=2,  -modeA4,3,1, -modeB4,2, -modeC2,1,3 -Pas -Pbs -Pcs -Pcomps  -alpha0  -beta2.20 -strideC2,6,12,24 -pDn -useD",
        "./lwtensorTest -Rcontraction  -extent1=2,2=3,3=2,4=2,  -modeA4,3,1, -modeB4,2, -modeC2,1,3 -Pas -Pbs -Pcs -Pcomps  -alpha0  -beta2.20 -strideC2,6,12,24 -pAn -pBn",
        "./lwtensorTest -modeC2,0,1,4 -modeA0,3,4 -modeB3,2,1,4 -extent0=16,1=22,2=10,3=24,4=16, -opAB3 -opABC3 -opA1 -opB1 -opC1 -Pah -Pbh -Pch -Pcomph -alpha1.100000 -beta2.200000 -gamma0.000000 -Rcontraction",
        "./lwtensorTest -modeAb,v,x,d,y,h,z,w -modeBz,y,x,v,k -modeCb,k,d,h,w -extentb=2,k=512,v=256,x=2,y=2,z=2,h=64,w=64,d=32 -Rcontraction -fastVerify -Pas -Pbs -Pcs -Pcomps -algo-2",
        "./lwtensorTest -modeC2,6,0,1,4,5,3 -alignmentReqA16 -alignmentA128 -modeA2,11,0,10,1,3,9,7,4,5,8 -strideA1,2,4,12,24,72,216,432,864,1728,3456, -alignmentReqB16 -alignmentB128 -modeB7,11,10,9,6,8 -strideB1,2,4,8,16,32, -extent0=3,1=3,2=2,3=3,4=2,5=2,6=2,7=2,8=3,9=2,10=2,11=2, -strideC1,2,4,12,36,72,144, -alignmentReqC16 -alignmentC128 -opAB3 -opABC3 -opReduce3 -opA9 -opB1 -opC1 -Paz -Pbz -Pcz -Pcompd -alpha1.100000 -beta2.200000 -gamma0.000000 -Rcontraction -alignmentReqD16 -alignmentD128",
        "./lwtensorTest -modeC1,0 -alignmentReqA128 -alignmentA128 -modeA3,5,2,4,7,0,6,8,9 -strideA1,3,6,18,36,72,216,432,864, -alignmentReqB128 -alignmentB128 -modeB5,4,6,7,8,3,2,1,9 -strideB1,2,4,8,16,32,96,288,864, -useD -alignmentReqD128 -alignmentD128 -extent0=3,1=3,2=3,3=3,4=2,5=2,6=2,7=2,8=2,9=3, -strideC1,3, -alignmentReqC128 -alignmentC128 -opAB3 -opABC3 -opReduce3 -opA1 -opB1 -opC1 -Pad -Pbd -Pcd -Pcompd -alpha1.100000 -beta0.000000 -gamma0.000000 -Rcontraction",
        "./lwtensorTest -modeC0,1 -alignmentReqA128 -alignmentA512 -modeA3,5,2,6,0,4 -strideA1,3,9,27,81,243, -alignmentReqB128 -alignmentB512 -modeB2,1,5,3,4,6 -strideB1,3,9,27,81,243, -useD -alignmentReqD128 -alignmentD512 -extent0=3,1=3,2=3,3=3,4=3,5=3,6=3, -strideC1,3, -alignmentReqC128 -alignmentC512 -opAB3 -opABC3 -opReduce3 -opA1 -opB1 -opC1 -Pad -Pbd -Pcd -Pcomps -alpha1.100000 -beta0.000000 -gamma0.000000 -Rcontraction",
        "./lwtensorTest -modeC0,1 -alignmentReqA128 -alignmentA128 -modeA0,5,8,4,2,3,9,7,6 -strideA1,3,6,12,36,72,216,648,1944, -alignmentReqB128 -alignmentB128 -modeB5,1,4,8,9,3,7,2,6 -strideB1,2,6,18,36,108,324,972,1944, -useD -alignmentReqD128 -alignmentD128 -extent0=3,1=3,2=2,3=3,4=3,5=2,6=2,7=3,8=2,9=3, -strideC1,3, -alignmentReqC128 -alignmentC128 -opAB3 -opABC3 -opReduce3 -opA1 -opB1 -opC1 -Pad -Pbd -Pcd -Pcompd -alpha1.100000 -beta2.200000 -gamma0.000000 -Rcontraction",
        "./lwtensorTest -modeAb,v,x,d,y,h,z,w -modeBz,y,x,v,k -modeCb,k,d,h,w -extentb=2,k=512,v=256,x=2,y=2,z=2,h=64,w=64,d=32 -Rcontraction -fastVerify -Pas -Pbs -Pcs -Pcomps -algo-2",
        "./lwtensorTest -numRuns2 -Rcontraction  -extent0=3,1=4,2=6,3=5, -modeA1,3,0, -modeB2,3, -modeC1,0,2, -Pas -Pbs -Pcs -Pcomps  -alpha1.0  -beta1.1 -useD",
        "./lwtensorTest -numRuns2 -modeC0,3 -modeA1,2,3 -modeB2,1,0,3 -extent0=21,1=16,2=24,3=30, -strideC1,23, -alpha1.100000 -beta1.1 -gamma0.000000 -Rcontraction -useD",
        "./lwtensorTest -numRuns1 -Rcontraction -extent5=3,9=2,11=144,  -modeB11,5,9, -modeA11,9,5, -modeC -Pas -Pbs -Pcs -Pcomps -alpha1.0 -beta0",
        "./lwtensorTest -numRuns1 -modeCm,n -modeAm,k,c -modeBn,c,k -extentm=2,n=2,k=2,c=2 -strideB1,2,2147483650 -Rcontraction",
        "./lwtensorTest -numRuns1 -Rcontraction -extent0=2,1=2 -modeA0,1 -modeB1 -modeC0 -strideC3 -Paz -Pbd -Pcz -Pcompd",
        # large-k:
        "./lwtensorTest -algo-101 -Rcontraction -modeAk,m,c -modeBa,c,k -modeCm,a -extentm=64,a=64,k=50304,c=2 -fastVerify -beta0",
        "./lwtensorTest -algo-101 -Rcontraction -modeAm,k,c -modeBa,c,k -modeCm,a -extentm=256,a=256,k=45056,c=2 -fastVerify -beta20.3",
        "./lwtensorTest -algo-101 -Rcontraction -modeAm,k,c -modeBa,c,k -modeCm,a -extentm=256,a=256,k=31232,c=2 -fastVerify -beta0",
        "./lwtensorTest -algo-101 -Rcontraction -modeAk,m,c -modeBa,c,k -modeCm,a -extentm=64,a=256,k=36352,c=2 -fastVerify -beta1.3",
        "./lwtensorTest -algo-101 -Rcontraction -modeAm,k,c -modeBa,c,k -modeCm,a -extentm=256,a=64,k=49152,c=2 -fastVerify -beta0",
        "./lwtensorTest -algo-101 -Rcontraction -modeAk,m,c -modeBa,c,k -modeCm,a -extentm=32,a=256,k=46080,c=2 -fastVerify -beta1.2",
        "./lwtensorTest -algo-101 -Rcontraction -modeAm,k,c -modeBa,c,k -modeCm,a -extentm=128,a=128,k=29440,c=2 -fastVerify -beta0",
        "./lwtensorTest -algo-101 -Rcontraction -modeAk,m,c -modeBa,c,k -modeCm,a -extentm=256,a=64,k=48128,c=2 -fastVerify -beta0",
        ]

dataTypeCombinations = [
                "-Pah -Pbh -Pch -Pcomph",
                "-Pas -Pbs -Pcs -Pcomps",
                "-Pad -Pbd -Pcd -Pcompd",
                "-Pac -Pbc -Pcc -Pcomps",
                "-Paz -Pbz -Pcz -Pcompd",

                # mixed
                "-Pas -Pbs -Pcs -Pcomph",
                "-Pad -Pbd -Pcd -Pcomps",
                "-Paz -Pbz -Pcz -Pcomps",

                # complex-real
                "-Pad -Pbz -Pcz -Pcompd",

                # Ampere
                "-Pas -Pbs -Pcs -Pcompb",
                "-Pas -Pbs -Pcs -Pcompt",
                "-Pab -Pbb -Pcb -Pcompb",
                "-Pac -Pbc -Pcc -Pcompt",
                ]

# You can specify the probabilities per feature separately
unaryOperators = [
   # operator                #id (must be same as in types.h)  #relative probability
   [("LWTENSOR_OP_IDENTITY" , 1),  8], # e.g., identity is 8x as likely as SQRT
   [("LWTENSOR_OP_SQRT"     , 2),  1],
   [("LWTENSOR_OP_RELU"     , 8),  2],
   [("LWTENSOR_OP_CONJ"     , 9),  2],
   [("LWTENSOR_OP_RCP"      , 10), 1],
   ]
binaryOperators = [
   [("LWTENSOR_OP_ADD"      , 3), 4], # e.g., ADD is 4x as likely as MIN
   [("LWTENSOR_OP_MUL"      , 5), 2],
   [("LWTENSOR_OP_MAX"      , 6), 1],
   [("LWTENSOR_OP_MIN"      , 7), 1],
   ]

isValidUnaryOperators = { # this should match lwtensor's isValidUnaryOperator()
        'z' : [1, 9],
        'c' : [1, 9],
        's' : [1],# 2, 8, 10],
        'd' : [1],# 2, 8, 10],
        'h' : [1],# 2, 8, 10],
        'i' : [1],# 8],
        'j' : [1],# 8],
        'k' : [1],# 8],
        'u' : [1],# 8],
        'b' : [1],# 8],
        't' : [1],# 8],
        }

# - All combination of orthogonal features will be generated
# - Each test case will be randomly modified via the randomFeatures
# - All (of orthogonalFeatures) parameters are set in the order as they appear (this is important since extent must be set after permA, B,C)


# Default
orthogonalDefaultAlgo = [
        ("dataTypes", dataTypeCombinations),
        ("numModesM", [0,1,2]),
        ("numModesN", [0,1,2]),
        ("numModesK", [0,1,2]),
        ("numModesL", [0,1]),
        ("extent", [2, 'rand']), # all extents equal to two, or random
        ]

randomDefaultAlgo = [      # pairs of (value, rel. probability)
        ("alpha" , [(1.1, 12), (0, 1)]),
        ("beta"  , [(2000000.2, 1),(2.2, 7), (0, 2)]),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # ---[ If set: a random extent will be set to this specific number. ]
        #("specificExtent" , [(1,1), (2,5), (100,1), (1000,1)]),
        ("useD" , [(True, 1),(False, 1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("alignment"   , [(128, 100), (16, 5), (512, 5)]),
        ("alignmentReq", [(128, 1), (16, 1), (512, 1)]),
        #("opC"   , unaryOperators),
        ]

orthogonalDefaultAlgo2 = [
        ("dataTypes", dataTypeCombinations),
        ("extent", ['rand3']), # random 2-3
        ]
randomDefaultAlgo2 = [      # pairs of (value, rel. probability)
        ("numModesM", [(0,1),(1,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1)]),
        ("numModesN", [(0,1),(1,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1)]),
        ("numModesK", [(0,1),(1,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1)]),
        ("numModesL", [(0,10),(1,1)]),
        ("useD" , [(True, 1),(False, 1)]),
        ("alpha" , [(1.1, 12)]),
        ("beta"  , [(2000000.2, 1),(2.2, 7), (0, 2)]),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # ---[ If set: a random extent will be set to this specific number. ]
        #("specificExtent" , [(1,1), (2,5), (100,1), (1000,1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("alignment"   , [(128, 100), (16, 5), (512, 5)]),
        ("alignmentReq", [(128, 1), (16, 1), (512, 1)]),
        #("opC"   , unaryOperators),
        ]
orthogonalDefaultAlgoL3 = [
        ("dataTypes", dataTypeCombinations),
        ("extent", ['randLarge']), # random large extents
        ("fastVerify", [True])
        ]


maxContractedExtent = {
        'z' : 60000,
        'c' : 30000,
        's' : 30000,
        'd' : 6000,
        'h' : 800,
        'i' : 1000,
        'b' : 800,
        't' : 30000,
        }

maxCounts = {}
maxCounts["isStrided"] = 50
maxCounts["alpha0"] = 20

totalCounts = {}
for l in maxCounts:
    totalCounts[l] = 0

###################### DON'T EDIT ANYTHING BELOW THIS LINE ############################
#(unless you are a developer)


class TestCase:
    def __init__(self):
        self.numModesM = 0
        self.numModesN = 0
        self.numModesK = 0
        self.numModesL = 0
        self.randomLargeExtent = False
        self.randomExtent = False
        self.randomExtent3 = False
        self.fixedExtent = 1
        self.modeA = []
        self.modeB = []
        self.modeC = []
        self.strideA = 0
        self.strideB = 0
        self.strideC = 0
        self.extent = {}
        self.dataTypes = "-Pas -Pbs -Pcs -Pcomps"
        self.opA = 1 # identity
        self.opB = 1
        self.opC = 1
        self.useD = False
        self.opOut = 1
        self.alpha = 1
        self.beta = 1
        self.fastVerify = False

        self.specificExtent = -1 # Assign a random mode with this extent

        self.hasExtentOne = 0 # if set: a random mode will have extent 1
        self.padStride = 0    # if set: a strides will be randomly padded
        self.dropModeA = 0    # if set: a random mode of A will be removed
        self.dropModeB = 0    # if set: a random mode of B will be removed

        self.alignment = 128
        self.alignmentReq = 128

    def getModeCommandLine(self, perm, ABC):
        ret = "-mode%s"%ABC
        for i in perm:
            ret += str(i) + ","
        return ret

    def getTypeA(self):
        return self.dataTypes.split()[0][-1]

    def getTypeB(self):
        return self.dataTypes.split()[1][-1]

    def getTypeC(self):
        return self.dataTypes.split()[2][-1]

    def getComputeType(self):
        return self.dataTypes.split()[3][-1]

    def setFeature(self, key, feature):
        if( key == "fastVerify" ):
            self.fastVerify = True
        elif( key == "useD" ):
            self.useD = feature
        elif( key == "numModesM" ):
            self.numModesM = feature
        elif( key == "numModesN" ):
            self.numModesN = feature
        elif( key == "numModesK" ):
            self.numModesK = feature
        elif( key == "numModesL" ):
            self.numModesL = feature
        elif( key == "dataTypes" ):
            self.dataTypes = " " + feature
        elif( key == "alpha" ):
            self.alpha = feature
        elif( key == "beta" ):
            self.beta = feature
        elif( key == "opA" ):
            if( feature[1] in isValidUnaryOperators[self.getTypeA()] ):
                self.opA = feature[1]
            else:
                self.opA = random.choice(isValidUnaryOperators[self.getTypeA()])
        elif( key == "opB" ):
            if( feature[1] in isValidUnaryOperators[self.getTypeB()] ):
                self.opB = feature[1]
            else:
                self.opB = random.choice(isValidUnaryOperators[self.getTypeB()])
        elif( key == "opC" ):
            if( feature[1] in isValidUnaryOperators[self.getTypeC()] ):
                self.opC = feature[1]
            else:
                self.opC = random.choice(isValidUnaryOperators[self.getTypeC()])
        elif( key == "opOut" ):
            if( feature[1] in isValidUnaryOperators[self.getTypeC()] ):
                self.opOut = feature[1]
            else:
                self.opOut = random.choice(isValidUnaryOperators[self.getTypeC()])
        elif( key == "padStride" ):
            self.padStride = feature
        elif( key == "specificExtent" ):
            self.specificExtent = feature
        elif( key == "hasExtentOne" ):
            self.hasExtentOne = feature
        elif( key == "dropModeA" ):
            self.dropModeA = feature
        elif( key == "dropModeB" ):
            self.dropModeB = feature
        elif( key == "extent" ):
            if( feature == "rand" ):
                self.randomExtent = True
            elif( feature == "rand3" ):
                self.randomExtent3 = True
            elif( feature == "randLarge" ):
                self.randomLargeExtent = True
            else:
                self.fixedExtent = feature
        elif( key == "alignment" ):
            self.alignment = feature
        elif( key == "alignmentReq" ):
            self.alignmentReq = feature
        else:
            print("ERROR: UNKNOWN KEY")
            exit(-1)

    def generateModeABC(self):
        nmodes = self.numModesM + self.numModesN + self.numModesK + self.numModesL
        minExtent = 7
        maxExtent = 100
        if( self.randomLargeExtent ):
            if nmodes <= 2:
                maxExtent = 1024
            elif nmodes <= 3:
                maxExtent = 768
            elif nmodes <= 4:
                maxExtent = 92
            elif nmodes == 5:
                maxExtent = 40
            elif nmodes == 6:
                maxExtent = 28
            elif nmodes == 7:
                maxExtent = 16
            elif nmodes >= 8:
                maxExtent = 10
        elif( self.randomExtent ):
            if nmodes <= 4:
                maxExtent = 41
            elif nmodes == 5:
                maxExtent = 31
            elif nmodes == 6:
                maxExtent = 21
            elif nmodes == 7:
                maxExtent = 12
            elif nmodes >= 8:
                maxExtent = 10
        elif( self.randomExtent3 ):
            minExtent = 2
            maxExtent = 3
        else:
            minExtent = self.fixedExtent
            maxExtent = self.fixedExtent
        self.modeM = []
        self.modeN = []
        self.modeK = []
        self.modeL = []
        firstMode = 0
        if ( self.numModesM ) :
            self.modeM = list(range( firstMode, firstMode + self.numModesM ))
            firstMode = firstMode + self.numModesM
            for mode in self.modeM:
                self.extent[mode] = random.randint(minExtent, maxExtent)
        if ( self.numModesN ) :
            self.modeN = list(range( firstMode, firstMode + self.numModesN ))
            firstMode = firstMode + self.numModesN
            for mode in self.modeN:
                self.extent[mode] = random.randint(minExtent, maxExtent)
        if ( self.numModesK ) :
            self.modeK = list(range( firstMode, firstMode + self.numModesK ))
            firstMode = firstMode + self.numModesK
            for mode in self.modeK:
                self.extent[mode] = random.randint(minExtent, maxExtent)
        if ( self.numModesL ) :
            self.modeL = list(range( firstMode, firstMode + self.numModesL ))
            firstMode = firstMode + self.numModesL
            for mode in self.modeL:
                self.extent[mode] = random.randint(minExtent, maxExtent)
        self.modeA = self.modeA + self.modeM
        self.modeA = self.modeA + self.modeK
        self.modeB = self.modeB + self.modeN
        self.modeB = self.modeB + self.modeK
        self.modeC = self.modeC + self.modeM
        self.modeC = self.modeC + self.modeN
        random.shuffle( self.modeA )
        random.shuffle( self.modeB )
        random.shuffle( self.modeC )
        self.modeA = self.modeA + self.modeL
        self.modeB = self.modeB + self.modeL
        self.modeC = self.modeC + self.modeL

    def isStrided(self):
        if( (len( self.modeA ) > 0 and self.modeA[0] in self.modeL) or
            (len( self.modeB ) > 0 and self.modeB[0] in self.modeL) or
            (len( self.modeC ) > 0 and self.modeC[0] in self.modeL) ):
            return True
        else:
            return False

    def getNumElementsA(self):
        numElements = 1
        for m in self.modeA:
            numElements *= self.extent[m]
        return numElements

    def getNumElementsB(self):
        numElements = 1
        for m in self.modeB:
            numElements *= self.extent[m]
        return numElements

    def getNumElementsC(self):
        numElements = 1
        for m in self.modeC:
            numElements *= self.extent[m]
        return numElements


    def isValid(self):
        if len( self.modeA ) == 0 or len( self.modeA ) > 12 :
            return False
        if len( self.modeB ) == 0 or len( self.modeB ) > 12 :
            return False
        if len( self.modeC ) > 12 :
            return False
        if( self.getNumElementsA() > 24 * 1024* 1024):
            return False
        if( self.getNumElementsB() > 24 * 1024* 1024):
            return False
        if( self.getNumElementsC() > 24 * 1024* 1024):
            return False

        # skip tests for which the contracted dimension is too large
        if ( self.numModesK ) :
            totalExtentK = 1
            for mode in self.modeK:
                totalExtentK *= self.extent[mode]
            if( totalExtentK > maxContractedExtent[self.getComputeType()] ):
                return False

        if( self.isStrided() ):
            if( totalCounts["isStrided"] < maxCounts["isStrided"] ):
                totalCounts["isStrided"] = totalCounts["isStrided"] + 1
            else:
                return False
        if( self.alpha == 0 ):
            if( totalCounts["alpha0"] < maxCounts["alpha0"] ):
                totalCounts["alpha0"] = totalCounts["alpha0"] + 1
            else:
                return False

        if self.alignment < self.alignmentReq:
            return False

        return True

    def computeStride(self, modes, ABC):
        stride = 1 #TODO adopt for vectorization
        ret = " -stride%s"%ABC
        for mode in modes:
            ret += str(stride)+","
            padding = 0
            if( self.padStride ):
                padding = random.randint(0,4)
            stride *= (self.extent[mode] + padding)

        return ret

    def __str__(self):
        ret = "./lwtensorTest -numRuns1 -Rcontraction "

        nmodes = len(self.modeC)


        if self.specificExtent != -1 :
            # ---[ Set a random mode with the specific extent. ]
            self.extent[self.modeC[random.randint(0,nmodes -1)]] = self.specificExtent

        if self.hasExtentOne :
            self.extent[self.modeC[random.randint(0,nmodes -1)]] = 1 #set a random mode to one

        if self.dropModeA and len(self.modeA) > 0:
            del self.modeA[random.randint(0,len(self.modeA)-1)]
        if self.dropModeB and len(self.modeB) > 0:
            del self.modeB[random.randint(0,len(self.modeB)-1)]

        ret += " -extent"
        for mode, ext in self.extent.items():
            ret += str(mode)+"="+str(ext)+","

        # swap AB at random
        if(random.random() <= 0.5):
            Achar = "A"
            Bchar = "B"
        else:
            Achar = "B"
            Bchar = "A"

        ret += self.computeStride(self.modeA, Achar)
        ret += self.computeStride(self.modeB, Bchar)
        ret += self.computeStride(self.modeC, "C")
        ret += " " + self.getModeCommandLine(self.modeA, Achar)
        ret += " " + self.getModeCommandLine(self.modeB, Bchar)
        ret += " " + self.getModeCommandLine(self.modeC, "C")
        if( self.dataTypes == "-Pad -Pbz -Pcz -Pcompz" and random.random() <= 0.5 ): # swap A and B by chance
            self.dataTypes = "-Paz -Pbd -Pcz -Pcompz"
        ret += self.dataTypes + " "
        ret += " -alpha%.2f "%(self.alpha)
        ret += " -beta%.2f "%(self.beta)
        ret += " -opA%d "%(self.opA)
        ret += " -opB%d "%(self.opB)
        ret += " -opC%d "%(self.opC)
        ret += " -alignmentA%d "%(self.alignment)
        ret += " -alignmentB%d "%(self.alignment)
        ret += " -alignmentC%d "%(self.alignment)
        ret += " -alignmentD%d "%(self.alignment)
        ret += " -alignmentReqA%d "%(self.alignmentReq)
        ret += " -alignmentReqB%d "%(self.alignmentReq)
        ret += " -alignmentReqC%d "%(self.alignmentReq)
        ret += " -alignmentReqD%d "%(self.alignmentReq)
        if(self.fastVerify):
            ret += " -fastVerify "
        if( self.useD ):
            ret += " -useD "
        #ret += " -opOut%d "%(self.opOut)
        return ret

# creates tests based on the orthogonal and random features defined above (in a relwrsive fashion)
def createTest( orthFeatures, randomFeatures, test, allTests, numTestMultiplier ):
    if( len(orthFeatures) == 0 ): # stop relwrsion
        for i in range(numTestMultiplier):
            newTest = copy.deepcopy(test)
            # perturb all random features randomly
            for key, possibleValues in randomFeatures:

                # make sure that the values are selected w.r.t. their probabilities
                # compute forward-scan of probabilities
                scanProbabilities = [possibleValues[0][1]]
                for (val, prob) in possibleValues[1:]:
                    scanProbabilities.append(scanProbabilities[-1] + prob)
                rand = random.random() * scanProbabilities[-1];

                selectedVal = "NONE"
                for j in range(len(possibleValues)):
                    if( rand <= scanProbabilities[j] ):
                        selectedVal = possibleValues[j][0]
                        break
                assert( selectedVal != "NONE" )

                newTest.setFeature( key, selectedVal )

            newTest.generateModeABC()
            if( newTest.isValid() ):
                allTests.append(newTest)
        return


    orthCopy = copy.deepcopy(orthFeatures)
    key, values = orthCopy[0] # pick an arbitrary key
    del orthCopy[0]

    for value in values:
        newTest = copy.deepcopy(test)
        newTest.setFeature(key, value)
        createTest( orthCopy, randomFeatures, newTest, allTests, numTestMultiplier )


######## create TESTS #############
allTests = []
numTestMultiplier = 11 # increase this value to generate more tests
createTest( orthogonalDefaultAlgo, randomDefaultAlgo, TestCase(), allTests, numTestMultiplier  )
numTestMultiplier = 1000 # increase this value to generate more tests
createTest( orthogonalDefaultAlgo2, randomDefaultAlgo2, TestCase(), allTests, numTestMultiplier  )
testsL3 = []
createTest( orthogonalDefaultAlgoL3, randomDefaultAlgo2, TestCase(), testsL3, numTestMultiplier *8  )
print(len(testsL3))

######## separate tests into L0, L1, L2 #############
numTests = len(allTests)
numTestsL0 = min(700, numTests)
numTestsL1 = min(3000, numTests)

# write L0
randomIdx = list(range(numTests))
random.shuffle( randomIdx )
l0File = open("lwtensorContractionL0.sh", "w+")
content = ""
for idx in range(numTestsL0):
    content += str(allTests[randomIdx[idx]]) + "\n"
for test in mustIncludeTests:
    content += test + "\n"
l0File.write(content)


# write L1
random.shuffle( randomIdx )
l1File = open("lwtensorContractionL1.sh", "w+")
content = ""
for idx in range(numTestsL1):
    assert(randomIdx[idx] < len(allTests))
    content += str(allTests[randomIdx[idx]]) + "\n"
l1File.write(content)

# write L2 (all remaining tests)
l2File = open("lwtensorContractionL2.sh", "w+")
content = ""
for idx in range(numTestsL1, len(allTests)):
    assert(randomIdx[idx] < len(allTests))
    content += str(allTests[randomIdx[idx]]) + "\n"
l2File.write(content)

# write L3
l2File = open("lwtensorContractionL3.sh", "w+")
content = ""
for test in testsL3:
    content += str(test) + "\n"
l2File.write(content)










