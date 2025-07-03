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
        "./lwtensorTest -Rreduction -modeAa,b,c,d,e,f,g,h,i,j,k,l,m,0,n,o,p,r,s -modeCa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s -extenta=2,b=2,c=2,d=2,e=2,f=2,g=2,h=2,i=2,j=2,k=2,l=2,0=2,m=2,n=2,o=2,p=2,0=2,q=2,r=2,s=2",
        "./lwtensorTest -Rreduction -modeAa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s -modeCa,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,r,s -extenta=2,b=2,c=2,d=2,e=2,f=2,g=2,h=2,i=2,j=2,k=2,l=2,0=2,m=2,n=2,o=2,p=2,0=2,q=2,r=2,s=2",
        "./lwtensorTest -Rreduction  -extent1=2,2=3,3=2,4=2,  -modeA4,3,1, -modeC2,1,3 -Pas -Pcs -Pcomps  -alpha0  -gamma2.20",
        "./lwtensorTest -modeC -modeA0 -modeB0 -extent0=30 -alpha1.1 -beta2.200000 -gamma2.200000 -Pac -Pbc -Pcc -Pcomps -Rreduction"
        "./lwtensorTest -Rreduction -modeCn -modeAm,n -extentn=100000,m=1025",
        "./lwtensorTest -modeC1,0 -modeA0,2,4,3,1 -strideA1,9,297,7722,108108, -extent0=8,1=14,2=30,3=14,4=26, -strideC1,18, -alignmentReqC128 -alignmentC128 -opAB3 -opABC3 -opReduce5 -opA1 -opB1 -opC9 -Paz -Pbz -Pcz -Pcompd -alpha0.000000 -beta2.200000 -gamma2.200000 -Rreduction -pDn",
        "./lwtensorTest -modeC -modeA1 -extent1=16  -Paz -Pbz -Pcz -Pcompd -alpha0.00000 -beta1.00000 -gamma1.00000 -Rreduction"
        ]

dataTypeCombinations = [
                "-Pab -Pbb -Pcb -Pcompb",
                "-Pab -Pbb -Pcb -Pcomps",
                "-Pah -Pbh -Pch -Pcomph",
                "-Pah -Pbh -Pch -Pcomps",
                "-Pas -Pbs -Pcs -Pcomps",
                "-Pad -Pbd -Pcd -Pcompd",
                "-Pac -Pbc -Pcc -Pcomps",
                "-Paz -Pbz -Pcz -Pcompd",
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
   [("LWTENSOR_OP_ADD"      , 3), 20], # e.g., ADD is 10x as likely as MIN
   [("LWTENSOR_OP_MUL"      , 5), 1],
   [("LWTENSOR_OP_MAX"      , 6), 2],
   [("LWTENSOR_OP_MIN"      , 7), 2],
   ]

isValidUnaryOperators = { # this should match lwtensor's isValidUnaryOperator()
        'z' : [1, 9],
        'c' : [1, 9],
        's' : [1, 2, 8, 10],
        'd' : [1, 2, 8, 10],
        'h' : [1, 2, 8, 10],
        'b' : [1, 2, 8, 10],
        'i' : [1],# 8],
        'j' : [1],# 8],
        'k' : [1],# 8],
        'u' : [1],# 8],
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

numTestMultiplier = 40 # increase this value to generate more tests


# Default
orthogonalDefaultAlgo = [
        ("dataTypes", dataTypeCombinations),
        ("numModesM", [0,1,2,3]),
        ("numModesN", [0]),
        ("numModesK", [0,1,2,3]),
        ("numModesL", [0]),
        ("extent", [2, 'rand']), # all extents equal to two, or random
        # ("extent", [1, 2, 'rand']), # all extents equal to two, or random
        #("opOut"  , unaryOperators),
        ]

randomDefaultAlgo = [      # pairse of (value, rel. probability)
        ("alpha" , [(2.0, 12), (0, 1)]),
        ("beta"  , [(20000.2, 1),(2.5, 6), (0, 1)]),
        ("padStride"    , [(0,12), (1,1)]), # if set: a strides will be randomly padded
        # ---[ If set: a random extent will be set to this specific number. ]
        #("specificExtent" , [(1,1), (2,5), (100,1), (1000,1)]), 
        ("opA"   , unaryOperators),
        ("opB"   , unaryOperators),
        ("opC"   , unaryOperators),
        ("opReduce", binaryOperators),
        ]

def maxContractedExtent(typeCompute):
    if( typeCompute == 'd' ):
        return 60000
    elif( typeCompute == 's' ):
        return 30000
    elif( typeCompute == 'h' or typeCompute == 'b' ):
        return 800
    elif( typeCompute == 'i' or typeCompute == 'j' or typeCompute == 'u' ):
        return 1000
    else:
        print("ERROR: UNKNOWN compute type")
        exit(-1);

def maxContractedExtentMul(typeCompute):  #limits for multiplication (to avoid over or underflow)
    if( typeCompute == 'd' ):
        return 60
    elif( typeCompute == 's' ):
        return 30
    elif( typeCompute == 'h' or typeCompute == 'b' ):
        return 10 
    elif( typeCompute == 'i' or typeCompute == 'j' or typeCompute == 'u' ):
        return 10
    else:
        print("ERROR: UNKNOWN compute type")
        exit(-1);

###################### DON'T EDIT ANYTHING BELOW THIS LINE ############################
#(unless you are a developer)


class TestCase:
    def __init__(self):
        self.numModesM = 0
        self.numModesN = 0
        self.numModesK = 0
        self.numModesL = 0
        self.randomExtent = False
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
        self.opReduce = 3
        self.opOut = 1
        self.alpha = 1
        self.beta = 1

        self.specificExtent = -1 # Assign a random mode with this extent
        
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
        if( key == "numModesM" ):
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
        elif( key == "opReduce" ):
            if( feature[1] in isValidBinaryOperators(self.getTypeC()) ):
                self.opReduce = feature[1]
            else:
                self.opReduce = random.choice(isValidBinaryOperators(self.getTypeC()))
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
            else:
                self.fixedExtent = feature
        else:
            print("ERROR: UNKNOWN KEY")
            exit(-1)

    def generateModeABC(self):
        nmodes = self.numModesM + self.numModesN + self.numModesK + self.numModesL
        minExtent = 7
        maxExtent = 100
        if( self.randomExtent ):
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
            minExtent = self.fixedExtent
            maxExtent = self.fixedExtent
        self.modeM = [] 
        self.modeN = [] 
        self.modeK = [] 
        self.modeL = [] 
        firstMode = 0
        extent1Prob = 0.05
        caseWithContractedProb = 0.2
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
        # Try producing 20% cases with reductions that have a contracted mode.
        if ( random.random() > caseWithContractedProb ):
            self.modeA = self.modeA + self.modeK
        # Set one of extents to 1 randomly, prob < 5%
        if ( random.random() < extent1Prob and len(self.modeA) > 0 ):
            self.extent[self.modeA[random.randint(0, len(self.modeA)-1)]] = 1
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

    def isValid(self):
        if( self.alpha == 0 and self.beta == 0):
            return False
        if len( self.modeA ) == 0 or len( self.modeA ) > 8 :
            return False
        if len( self.modeB ) == 0 or len( self.modeB ) > 8 :
            return False
        if len( self.modeC ) > 8 :
            return False
        # check if all extents from modeA == 1
        extentA = {}
        for mode in self.modeA:
            extentA[mode] = self.extent[mode]
        if sum(extentA.values()) == len( self.modeA ):
            return False

        # skip tests for which the contracted dimension is too large
        if ( self.numModesK ) :
            totalExtentK = 1
            for mode in self.modeK:
                totalExtentK *= self.extent[mode]
            if( totalExtentK > maxContractedExtent(self.getComputeType()) ):
                return False
            if( self.opReduce == 5 and totalExtentK > maxContractedExtentMul(self.getComputeType())):
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
        ret = "./lwtensorTest -Rreduction "

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

        ret += self.computeStride(self.modeA, "A")
        ret += self.computeStride(self.modeB, "B")
        ret += self.computeStride(self.modeC, "C")

        ret += " " + self.getModeCommandLine(self.modeA, "A") 
        ret += " " + self.getModeCommandLine(self.modeB, "B") 
        ret += " " + self.getModeCommandLine(self.modeC, "C") 
        ret += self.dataTypes + " "
        ret += " -alpha%.2f "%(self.alpha)
        ret += " -gamma%.2f "%(self.beta)
        ret += " -beta%.2f "%(self.beta)
        ret += " -opA%d "%(self.opA)
        ret += " -opB%d "%(self.opB)
        ret += " -opC%d "%(self.opC)
        ret += " -opReduce%d "%(self.opReduce)
        #ret += " -opOut%d "%(self.opOut)
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
        createTest( orthCopy, randomFeatures, newTest, allTests )


######## create TESTS #############
allTests = []
createTest( orthogonalDefaultAlgo, randomDefaultAlgo, TestCase(), allTests )
#createTest( orthogonalDefault2D, randomDefault2D, TestCase(), allTests )
#createTest( orthogonalFeatures1D, randomFeatures1D, TestCase(), allTests )
#createTest( orthogonalFeatures2D, randomFeatures2D, TestCase(), allTests )
#createTest( orthogonalFeatures3D, randomFeatures3D, TestCase(), allTests )
#createTest( orthogonalFeatures4D, randomFeatures4D, TestCase(), allTests )

######## separate tests into L0, L1, L2 #############
numTests = len(allTests)
numTestsL0 = min(1000, numTests)
numTestsL1 = min(3000, numTests)

# write L0
randomIdx = list(range(numTests))
random.shuffle( randomIdx )
l0File = open("lwtensorReductionL0.sh", "w+")
content = ""
for idx in range(numTestsL0):
    content += str(allTests[randomIdx[idx]]) + "\n"
for test in mustIncludeTests:
    content += test + "\n"
l0File.write(content)
l0File.close()

# write L1
random.shuffle( randomIdx )
l1File = open("lwtensorReductionL1.sh", "w+")
content = ""
for idx in range(numTestsL1):
    assert(randomIdx[idx] < len(allTests))
    content += str(allTests[randomIdx[idx]]) + "\n"
l1File.write(content)
l1File.close()


# write L2 (all remaining tests)
l2File = open("lwtensorReductionL2.sh", "w+")
content = ""
for idx in range(numTestsL1, len(allTests)):
    assert(randomIdx[idx] < len(allTests))
    content += str(allTests[randomIdx[idx]]) + "\n"
l2File.write(content)
l2File.close()



