import numpy as np
import copy
import random

random.seed(0)


# - All combination of orthogonal features will be generated
# - Each test case will be randomly modified via the randomFeatures
# - All (of orthogonalFeatures) parameters are set in the order as they appear (this is important since extent must be set after permA, B,C)

numTestMultiplier = 2 # increase this value to generate more tests

def roundUp(x , y):
    if( x % y == 0 ): return x
    return x + y - x%y

class TestCase:
    def __init__(self, isReduction):
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
        self.dataTypes = " -Pas -Pbs -Pcs -Pcomps"
        self.opA = 1 # identity
        self.opB = 1
        self.opC = 1
        self.opOut = 1
        self.alpha = 1
        self.beta = 1
        self.isReduction = isReduction
    def setDataTypes(self, dataTypes ):
        if( dataTypes == "hhhs" ):
            self.dataTypes = " -Pah -Pbh -Pch -Pcomps"
        elif( dataTypes == "ssss" ):
            self.dataTypes = " -Pas -Pbs -Pcs -Pcomps"
        elif( dataTypes == "dddd" ):
            self.dataTypes = " -Pad -Pbd -Pcd -Pcompd"
        else:
            print("Data type unknown")
            exit(0)

    def getModeCommandLine(self, perm, ABC):
        ret = "-mode%s"%ABC
        for i in perm:
            ret += str(i) + ","
        return ret

    def roundUp(x, to):
        if (x%to == 0):
            return x
        else:
            return x + to - x%to;

    def generateModeABC(self, numElementsA):
        nmodes = self.numModesM + self.numModesN + self.numModesK + self.numModesL
        self.modeM = [] 
        self.modeN = [] 
        self.modeK = [] 
        self.modeL = [] 
        firstMode = 0

        meanExtent = numElementsA**(1.0/nmodes)

        makeMultiple = random.choice([4,8])

        if ( self.numModesM ) :
            self.modeM = list(range( firstMode, firstMode + self.numModesM ))
            firstMode = firstMode + self.numModesM
            for mode in self.modeM:
                self.extent[mode] = roundUp(np.absolute(int(np.random.normal(meanExtent, meanExtent/2))), makeMultiple)
        if ((not self.isReduction) and self.numModesN ) :
            self.modeN = list(range( firstMode, firstMode + self.numModesN ))
            firstMode = firstMode + self.numModesN
            for mode in self.modeN:
                self.extent[mode] = roundUp(np.absolute(int(np.random.normal(meanExtent, meanExtent/2))), makeMultiple)
        if ( self.numModesK ) :
            self.modeK = list(range( firstMode, firstMode + self.numModesK ))
            for mode in self.modeK:
                self.extent[mode] = roundUp(np.absolute(int(np.random.normal(meanExtent, meanExtent/2))), makeMultiple)
            firstMode = firstMode + self.numModesK
        if ( self.numModesL ) :
            self.modeL = list(range( firstMode, firstMode + self.numModesL ))
            firstMode = firstMode + self.numModesL
            for mode in self.modeL:
                self.extent[mode] = roundUp(np.absolute(int(np.random.normal(meanExtent, meanExtent/2))), makeMultiple)
        self.modeA = self.modeA + self.modeM
        self.modeA = self.modeA + self.modeK
        self.modeB = self.modeB + self.modeN
        self.modeB = self.modeB + self.modeK
        self.modeC = self.modeC + self.modeM
        self.modeC = self.modeC + self.modeN
        random.shuffle( self.modeA )
        random.shuffle( self.modeB )
        random.shuffle( self.modeC )
        if( random.random() < 0.7 ): # round stride-1 extent such that vectorization is not under represented
            self.extent[self.modeA[0]] = roundUp(self.extent[self.modeA[0]], 4)
        self.modeA = self.modeA + self.modeL
        self.modeB = self.modeB + self.modeL
        self.modeC = self.modeC + self.modeL

    def isValid(self, minNumElements, maxNumElements, useLargeK):
        if len( self.modeA ) == 0 or len( self.modeA ) > 8 :
            return False
        if (not self.isReduction):
            if len( self.modeB ) == 0 or len( self.modeB ) > 8 :
                return False
        if len( self.modeC ) > 8 :
            return False
        if (useLargeK):
            totalK = 1
            for k in self.modeK:
                totalK *= self.extent[k]
            if (totalK < 8000):
                return False
        numElementsA = 1
        for mode in self.modeA:
            numElementsA *= self.extent[mode]
        if(numElementsA < minNumElements or numElementsA > maxNumElements ):
            return False
        return True

    def __str__(self):
        ret = "./lwtensorTest -numRuns3 -Rcontraction -algo-1 -fastVerify "
        if (self.isReduction):
            ret = "./lwtensorTest -numRuns3 -Rreduction -algo-1 -fastVerify "

        nmodes = len(self.modeC)

        ret += " -extent"
        for mode, ext in self.extent.items():
            ret += str(mode)+"="+str(ext)+","

        ret += " " + self.getModeCommandLine(self.modeA, "A")
        if (not self.isReduction):
            ret += " " + self.getModeCommandLine(self.modeB, "B")
        ret += " " + self.getModeCommandLine(self.modeC, "C")
        ret += self.dataTypes + " "
        ret += " -alpha%.2f "%(self.alpha)
        ret += " -beta%.2f "%(self.beta)
        ret += " -opA%d "%(self.opA)
        if (not self.isReduction):
            ret += " -opB%d "%(self.opB)
        ret += " -opC%d "%(self.opC)
        return ret

def generateTest( dataTypes, minNumElements, maxNumElements, isReduction ):
    numElementsA = random.randint(minNumElements, maxNumElements)
    numModes = random.randint(2,4)
    numModesM = random.randint(0,numModes)
    numModesK = numModes - numModesM
    allowedBeta = [0, 4.5]
    test = TestCase(isReduction)
    test.beta = allowedBeta[random.randint(0,1)]
    test.numModesM = numModesM
    test.numModesK = numModesK
    test.generateModeABC(numElementsA)
    test.setDataTypes(dataTypes)
    return test

# creates tests based on the orthogonal and random features defined above (in a relwrsive fashion)
def createTest( dataTypes, useLargeK, isReduction ):
    minNumElements = 1024**2
    maxNumElements = 100 * minNumElements
    tests = []
    numTests = 0
    while( numTests < 600 ):
        test = generateTest(dataTypes, minNumElements, maxNumElements, isReduction)
        if( test.isValid(minNumElements, maxNumElements, useLargeK) ):
            tests.append( test )
            numTests += 1
    return tests

######## create TESTS #############

tests = createTest('hhhs', False, False)

# write L0
l0File = open("gemvBenchmark.sh", "w+")
content = ""
for test in tests:
    content += str(test) + "\n"
l0File.write(content)

largeK = createTest('hhhs', True, False)

# write L0
l0File = open("gemv_largeK.sh", "w+")
content = ""
for test in largeK:
    content += str(test) + "\n"
l0File.write(content)

largeK = createTest('hhhs', True, True)

# write L0
l0File = open("reduction_largeK.sh", "w+")
content = ""
for test in largeK:
    content += str(test) + "\n"
l0File.write(content)








