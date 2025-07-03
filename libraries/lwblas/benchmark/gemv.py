import random
from random import shuffle


for targetSizeMB in [100, 300]:
    count = 0
    while count < 200:
        nmodeA = random.randint(2,6)
        nmodeC = nmodeA - 1
        modeA = range(nmodeA)
        modeC = range(nmodeA)
        contractedMode = modeA[random.randint(0,nmodeA-1)]
        modeC.remove(contractedMode)
        shuffle(modeC)
        modeAstr = ""
        for i in modeA:
            modeAstr += str(i) + ","
        modeCstr = ""
        for i in modeC:
            modeCstr += str(i) + ","
        totalExtent = 4
        sizeATarget = (targetSizeMB * 1e6)**(1./nmodeA) # bytes
        extentStr = ""
        for i in modeA:
            extent = random.randint(int(0.5 * sizeATarget), int(1.2 * sizeATarget))
            totalExtent *= extent
            extentStr += "%d=%d,"%(i, extent)

        if( totalExtent >= 2*1024**3 ):
            continue
        count += 1

        print "./lwtensorTest -Rcontraction -modeC%s -modeA%s -modeB%d -extent%s -Pad -Pbd -Pcd -Pcompd"%(modeCstr, modeAstr, contractedMode, extentStr)
