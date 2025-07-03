"""
This file generates numTests "nice-and-square" tensor permutations for which all
dimensions are powers of two.
The intention is that we should obtain close to SOL on these test cases
"""
import random
import numpy as np

numTestCases = 1000
numGenerated = 0
precision = "-Pas -Pbs -Pcs -Pcomps"

content = ""

while(numGenerated < numTestCases):
    dim = random.choice([3,4,5,6]) 
    modeC = list(range(dim))
    modeA = list(range(dim))
    random.shuffle(modeA)
    if( modeC == modeA ):
        continue
    extents = [random.choice([4,8,16,32,64,128,256,512]) for i in range(dim)]
    totalExtent = np.prod(extents)
    if (totalExtent < 512**2 or totalExtent > 1024*1024*32):
        continue # too small or too big
    #if (modeA[0] != modeC[0] and extents[0] * extents[modeA[0]] < 64 or # TODO remove this constraint once we have multi-dim blocking
    #    modeA[0] == modeC[0] and extents[0] < 16):
    #    continue

    strModeA = ",".join(map(str,modeA))
    strModeC = ",".join(map(str,modeC))
    strExtent = ""
    for i in range(dim):
        strExtent += "%d=%d,"%(i, extents[i])

    content += f"./lwtensorTest {precision} -Relementwise -alpha1.1 -beta0 -permute -modeA{strModeA} -modeC{strModeC} -extent{strExtent}\n"
    
    numGenerated += 1

f = open("easy_ew.sh","w+")
f.write(content)
f.close()
