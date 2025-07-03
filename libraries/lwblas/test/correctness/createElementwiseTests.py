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
        # https://lwbugs/200704581
        "./lwtensorTest -Relementwise -modeAa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s -modeBq -modeCa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s -extenta=2,b=2,c=2,d=2,e=2,f=2,g=2,h=2,i=2,j=2,k=2,l=2,0=2,m=2,n=2,o=2,p=2,0=2,q=2,r=2,s=2",
        "./lwtensorTest -modeC2 -modeA2 -modeB2 -extent0=2,1=2,2=2,3=2, -opC10 -Pas -Pbs -Pcs -Pcomps -alpha1 -beta0 -Relementwise -pCl",
        "./lwtensorTest -modeCc,h -modeAh,c -extentc=4,h=4 -strideC2,8 -gamma0 -alpha1 -Relementwise",
        "./lwtensorTest -modeC0,1,2 -modeA0,2,1 -strideA1,41,1107, -modeB1,2,0 -strideB1,24,648, -extent0=41,1=24,2=27, -strideC1,41,984, -Pas -Pbs -Pcs -Pcomps -alpha1.100000 -beta2.200000 -gamma3.300000 -Relementwise"
        ]

dataTypeCombinations = [
                "-Pah -Pbh -Pch -Pcomph",
                "-Pas -Pbs -Pcs -Pcomps",
                "-Pad -Pbd -Pcd -Pcompd",
                "-Pac -Pbc -Pcc -Pcomps",
                "-Paz -Pbz -Pcz -Pcompd",

                "-Pah -Pbh -Pch -Pcomps",
                "-Pas -Pbs -Pch -Pcomps",
                "-Pad -Pbd -Pcs -Pcompd",
                "-Paz -Pbz -Pcc -Pcompd",
                
                #Ampere
                "-Pab -Pbb -Pcb -Pcompb",
                "-Pab -Pbb -Pcb -Pcomps",
                ]

perm2D = [[0,1],[1,0]]
perm3D = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
perm4D = [[0,1,2,3],[0,3,2,1],[3,2,1,0],[1,0,2,3]]


# You can specify the probabilities per feature separately
unaryOperators = [
   # operator                #id (must be same as in types.h)  #relative probability
   [("LWTENSOR_OP_IDENTITY" , 1), 20], # e.g., identity is 20x as likely as SQRT
   [("LWTENSOR_OP_SQRT"     , 2),  1],
   [("LWTENSOR_OP_RELU"     , 8),  2],
   [("LWTENSOR_OP_CONJ"     , 9),  2],
   [("LWTENSOR_OP_RCP"      , 10), 1],
   [("LWTENSOR_OP_SIGMOID"  , 11), 1],
   [("LWTENSOR_OP_TANH"     , 12), 1],
   [("LWTENSOR_OP_EXP"      , 22), 1],
   [("LWTENSOR_OP_LOG"      , 23), 1],
   [("LWTENSOR_OP_ABS"      , 24), 1],
   [("LWTENSOR_OP_NEG"      , 25), 1],
   [("LWTENSOR_OP_SIN"      , 26), 1],
   [("LWTENSOR_OP_COS"      , 27), 1],
   [("LWTENSOR_OP_TAN"      , 28), 1],
   [("LWTENSOR_OP_SINH"     , 29), 1],
   [("LWTENSOR_OP_COSH"     , 30), 1],
   [("LWTENSOR_OP_ASIN"     , 31), 1],
   [("LWTENSOR_OP_ACOS"     , 32), 1],
   [("LWTENSOR_OP_ATAN"     , 33), 1],
   [("LWTENSOR_OP_ASINH"    , 34), 1],
   [("LWTENSOR_OP_ACOSH"    , 35), 1],
   [("LWTENSOR_OP_ACOSH"    , 35), 1],
   [("LWTENSOR_OP_ATANH"    , 36), 1],
   [("LWTENSOR_OP_CEIL"     , 37), 1],
   [("LWTENSOR_OP_FLOOR"    , 38), 1],
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
        's' : [1, 2, 8, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        'd' : [1, 2, 8, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        'h' : [1, 2, 8, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        'b' : [1, 2, 8, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        'i' : [1, 8],
        'j' : [1, 8],
        'k' : [1, 8],
        'u' : [1, 8],
        }

def isValidBinaryOperators(typeCompute):
    isComplex = typeCompute == 'z' or typeCompute == 'c'
    # this should match lwtensor's isValidBinaryOperator()
    if( isComplex ):
        return [3, 5]
    else:
        return [3, 5, 6, 7]

# - All combination of orthogonal features will be generated
# - Each test case will be randomly modified via the randomFeatures
# - All (of orthogonalFeatures) parameters are set in the order as they appear (this is important since extent must be set after permA, B,C)

numTestMultiplier = 1 # increase this value to generate more tests

# 1D
orthogonalFeatures1D = [
        ("dataTypes", dataTypeCombinations),
        ("permA", [[0]]),
        ("permB", [[0]]),
        ("permC", [[0]]),
        ("extent", [1, 2]), # all extents equal to two, or random
        ("opAB"  , binaryOperators),
        ("opABC" , binaryOperators),
        ]
randomFeatures1D = [      # pairse of (value, rel. probability)
        ("alpha" , [(1.1, 12), (0, 1)]),
        ("beta"  , [(2.2, 6), (0, 1)]),
        ("gamma" , [(3.3, 6), (0, 1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("opC"   , unaryOperators),
        ("padStride"    , [(0,8), (1,1)]), # if set: a strides will be randomly padded
        # not padding (i.e., 0) is 12x more likely than padding
        ("dropModeA"    , [(0,9) , (1,1)]), # if set: a random mode of A will be removed
        ("dropModeB"    , [(0,9) , (1,1)]), # if set: a random mode of B will be removed
        ("permute"      , [(True, 1), (False, 9)])
        ]

# 2D
orthogonalFeatures2D = [
        ("dataTypes", dataTypeCombinations),
        ("permA", perm2D),
        ("permB", perm2D),
        ("permC", perm2D),
        ("extent", [1, 2]), # all extents equal to two, or random
        ("opAB"  , binaryOperators),
        ("opABC" , binaryOperators),
        ]
# select these features randomly
randomFeatures2D = [      # pairse of (value, rel. probability)
        ("alpha" , [(1.1, 12), (0, 1)]),
        ("beta"  , [(2.2, 6), (0, 1)]),
        ("gamma" , [(3.3, 6), (0, 1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("opC"   , unaryOperators),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # not padding (i.e., 0) is 12x more likely than padding
        ("hasExtentOne" , [(0,16) , (1,1)]), # if set: a random mode will have extent 1
        ("dropModeA"    , [(0,9) , (1,1)]), # if set: a random mode of A will be removed
        ("dropModeB"    , [(0,9) , (1,1)]), # if set: a random mode of B will be removed
        ("permute"      , [(True, 1), (False, 9)])
        ]

# 3D
orthogonalFeatures3D = [
        ("dataTypes", dataTypeCombinations),
        ("permA", perm3D),
        ("permB", perm3D),
        ("permC", perm3D),
        ("extent", [2, "rand"]), # all extents equal to two, or random
        ]
randomFeatures3D = [      # pairse of (value, rel. probability)
        ("alpha" , [(1.1, 12), (0, 1)]),
        ("beta"  , [(2.2, 6), (0, 1)]),
        ("gamma" , [(3.3, 6), (0, 1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("opC"   , unaryOperators),
        ("opAB"  , binaryOperators),
        ("opABC" , binaryOperators),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # not padding (i.e., 0) is 12x more likely than padding
        ("hasExtentOne" , [(0,16) , (1,1)]), # if set: a random mode will have extent 1
        ("dropModeA"    , [(0,9) , (1,1)]), # if set: a random mode of A will be removed
        ("dropModeB"    , [(0,9) , (1,1)]), # if set: a random mode of B will be removed
        ("permute"      , [(True, 1), (False, 9)])
        ]

# 4D
orthogonalFeatures4D = [
                       ("dataTypes", dataTypeCombinations),
                       ("permA", perm4D),
                       ("permB", perm4D),
                       ("permC", perm4D),
                       ("extent", [2, "rand"]), # all extents equal to two, or random
                       ]
randomFeatures4D = [      # pairse of (value, rel. probability)
        ("alpha" , [(1.1, 12), (0, 1)]),
        ("beta"  , [(2.2, 6), (0, 1)]),
        ("gamma" , [(3.3, 6), (0, 1)]),
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("opC"   , unaryOperators),
        ("opAB"  , binaryOperators),
        ("opABC" , binaryOperators),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # not padding (i.e., 0) is 12x more likely than padding
        ("hasExtentOne" , [(0,16) , (1,1)]), # if set: a random mode will have extent 1
        ("dropModeA"    , [(0,9) , (1,1)]), # if set: a random mode of A will be removed
        ("dropModeB"    , [(0,9) , (1,1)]), # if set: a random mode of B will be removed
        ("permute"      , [(True, 1), (False, 9)])
        ]

###################### DON'T EDIT ANYTHING BELOW THIS LINE ############################
#(unless you are a developer)


class TestCase:
    def __init__(self):
        self.modeA = 0
        self.modeB = 0
        self.modeC = 0
        self.strideA = 0
        self.strideB = 0
        self.strideC = 0
        self.extent = {}
        self.dataTypes = "-Pas -Pbs -Pcs -Pcomps"
        self.opA = 1 # identity
        self.opB = 1
        self.opC = 1
        self.opAB = 3 # add
        self.opABC = 3
        self.alpha = 1
        self.beta = 1
        self.gamma = 1

        self.hasExtentOne = 0 # if set: a random mode will have extent 1
        self.padStride = 0    # if set: a strides will be randomly padded
        self.dropModeA = 0    # if set: a random mode of A will be removed
        self.dropModeB = 0    # if set: a random mode of B will be removed

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
        assert self.dataTypes[-2] == 'p' # ensure that -Pcomp is the last argument
        return self.dataTypes[-1]

    def setFeature(self, key, feature):
        if( key == "permA" ):
            self.modeA = feature
        elif( key == "permB" ):
            self.modeB = feature
        elif( key == "permC" ):
            self.modeC = feature
        elif( key == "dataTypes" ):
            self.dataTypes = " " + feature
        elif( key == "alpha" ):
            self.alpha = feature
        elif( key == "beta" ):
            self.beta = feature
        elif( key == "gamma" ):
            self.gamma = feature
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
        elif( key == "opAB" ):
            if( feature[1] in isValidBinaryOperators(self.getTypeC()) ):
                self.opAB = feature[1]
            else:
                self.opAB = random.choice(isValidBinaryOperators(self.getTypeC()))
        elif( key == "opABC" ):
            if( feature[1] in isValidBinaryOperators(self.getTypeC()) ):
                self.opABC = feature[1]
            else:
                self.opABC = random.choice(isValidBinaryOperators(self.getTypeC()))
        elif( key == "padStride" ):
            self.padStride = feature
        elif( key == "hasExtentOne" ):
            self.hasExtentOne = feature
        elif( key == "dropModeA" ):
            self.dropModeA = feature
        elif( key == "permute" ):
            self.permute = feature
        elif( key == "dropModeB" ):
            self.dropModeB = feature
        elif( key == "extent" ):
            nmodes = len(self.modeC)
            minExtent = 7
            maxExtent = 100
            if( feature == "rand" ):
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
            else:
                minExtent = feature
                maxExtent = feature

            for mode in self.modeC:
                self.extent[mode] = random.randint(minExtent, maxExtent)
        else:
            print "ERROR: UNKNOWN KEY"
            exit(-1)

    def isValid(self):
        for mode in self.modeA:
            if( not (mode in self.modeC) ):
                return False
        for mode in self.modeB:
            if( not (mode in self.modeC) ):
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
        ret = "./lwtensorTest -numRuns2 -Relementwise "

        nmodes = len(self.modeC)

        if self.hasExtentOne :
            self.extent[self.modeC[random.randint(0,nmodes -1)]] = 1 #set a random mode to one

        if self.dropModeA and len(self.modeA) > 0:
            del self.modeA[random.randint(0,len(self.modeA)-1)]
        if self.dropModeB and len(self.modeB) > 0:
            del self.modeB[random.randint(0,len(self.modeB)-1)]

        ret += " -extent"
        for mode in self.modeC:
            ret += str(mode)+"="+str(self.extent[mode])+","

        ret += self.computeStride(self.modeA, "A")
        if not self.permute:
            ret += self.computeStride(self.modeB, "B")
        ret += self.computeStride(self.modeC, "C")
        
        # TODO this doesn't belong here, but this case is not supported
        if( self.alpha == self.beta == self.gamma  == 0.0 ):
            rand = random.randint(0,2)
            if( rand == 0 ):
                self.alpha = 1.0
            elif( rand == 1 ):
                self.beta = 1.0
            else:
                self.gamma = 1.0

        if self.permute:
            self.gamma = 0.0
            if self.alpha == 0.0:
                self.alpha = 1.0
            self.opAB = 3
            self.opABC = 3
            ret += " -permute"

        ret += " " + self.getModeCommandLine(self.modeA, "A") 
        if not self.permute:
            ret += " " + self.getModeCommandLine(self.modeB, "B") 
        ret += " " + self.getModeCommandLine(self.modeC, "C") 
        ret += self.dataTypes + " "
        ret += " -alpha%.2f "%(self.alpha)
        ret += " -beta%.2f "%(self.beta)
        ret += " -gamma%.2f "%(self.gamma)
        ret += " -opA%d "%(self.opA)
        ret += " -opB%d "%(self.opB)
        ret += " -opC%d "%(self.opC)
        ret += " -opAB%d "%(self.opAB)
        ret += " -opABC%d "%(self.opABC)
        return ret


# creates tests based on the orthogonal and random features defined above (in a relwrsive fashion)
def createTest( orthFeatures, randomFeatures, test, allTests ):
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

            if( newTest.isValid() ):
                allTests.append(newTest)
        return


    orthCopy = copy.deepcopy(orthFeatures)
    key, values = orthCopy[0] # pick an arbitrary key
    del orthCopy[0]

    for value in values:
        newTest = copy.deepcopy(test)
        newTest.setFeature(key, value)
        createTest( orthCopy, randomFeatures, newTest, allTests )


######## create TESTS #############
allTests = []
createTest( orthogonalFeatures1D, randomFeatures1D, TestCase(), allTests )
createTest( orthogonalFeatures2D, randomFeatures2D, TestCase(), allTests )
createTest( orthogonalFeatures3D, randomFeatures3D, TestCase(), allTests )
createTest( orthogonalFeatures4D, randomFeatures4D, TestCase(), allTests )

######## separate tests into L0, L1, L2 #############
numTests = len(allTests)
numTestsL0 = min(4000, numTests)
numTestsL1 = min(10000, numTests)

# write L0
randomIdx = range(numTests)
random.shuffle( randomIdx )
l0File = open("correctness/lwtensorElementwiseL0.sh", "w+")
content = ""
for idx in range(numTestsL0):
    content += str(allTests[randomIdx[idx]]) + "\n"
for test in mustIncludeTests:
    content += test + "\n"
l0File.write(content)

# write L1
random.shuffle( randomIdx )
l1File = open("correctness/lwtensorElementwiseL1.sh", "w+")
content = ""
for idx in range(numTestsL1):
    assert(randomIdx[idx] < len(allTests))
    content += str(allTests[randomIdx[idx]]) + "\n"
l1File.write(content)

# write L2
random.shuffle( randomIdx )
l2File = open("correctness/lwtensorElementwiseL2.sh", "w+")
content = ""
for test in allTests:
    content += str(test) + "\n"
l2File.write(content)














