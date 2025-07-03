/***************************** CHEETAH SYS State Rotuines ********************\
*                                                                           *
* Module: TEGRASYS124.C                                                     *
*         CHEETAH System T124 Definitions.                                    *
\***************************************************************************/

#include "tegrasys.h"

/** 
 * @brief parses relocation table and fills out pTegraRelocTable
 * 
 * @return LW_OK if successful, error otherwise
 */
U032 tegrasysParseRelocationTable(PTEGRASYS pTegraSys, LwU64 *rawTable, PDEVICE_LIST devicesToInit, LwU32 numDevIds)
{
    struct
    {
        LwU64 id;
        LwU64 start;
        LwU64 size;
    } *pReloc = (void*) &rawTable[1];

    LwU32              numDevices = 0;
    PDEVICE_RELOCATION pTegraRelocTable;
    U032               i, devIdx;

    pTegraSys->numDevices = 0;

    // First, count the number of devices
    for (i = 0; pReloc[i].id; i++)
    {
        U032 j;
        U032 devId = ((U032)pReloc[i].id) >>16;

        for (j = 0; j < numDevIds; j++)
        {
            if (devicesToInit[j].devId == devId)
            {
                numDevices++;
                break;
            }
        }
    }

    pTegraRelocTable = (PDEVICE_RELOCATION)malloc(numDevices * sizeof(DEVICE_RELOCATION));
    if (pTegraRelocTable == NULL)
    {
        assert(0);
        return LW_ERR_GENERIC;
    }

    pTegraSys->numDevices = numDevices;
    pTegraSys->pRelocationTable = pTegraRelocTable;
    devIdx = 0;

    for (i = 0; pReloc[i].id; i++)
    {
        U032 j, indexInstCnt, len;
        U032 devId = ((U032)pReloc[i].id) >>16;

        // Check if this is a device we care about
        for (j = 0; j < numDevIds; j++)
        {
            if (devicesToInit[j].devId == devId)
                break;
        }

        if (j == numDevIds)
            continue;

        len = (U032) strlen(devicesToInit[j].devName);
        pTegraRelocTable[devIdx].devName = (char *)malloc(sizeof(char) * (len + 1));
        if (pTegraRelocTable[devIdx].devName == NULL)
        {
            assert(0);
            return LW_ERR_GENERIC;
        }
        strcpy(pTegraRelocTable[devIdx].devName, devicesToInit[j].devName);
        pTegraRelocTable[devIdx].devName[len] = '\0';
        pTegraRelocTable[devIdx].devId   = devId;
        pTegraRelocTable[devIdx].devInst = 0;
        pTegraRelocTable[devIdx].verMaj  = (pReloc[i].id >> 12) & 0xf;
        pTegraRelocTable[devIdx].verMin  = (pReloc[i].id >>  8) & 0xf;
        //
        // Don't care about these yet, but listing for completeness
        // powerGroup = (pReloc[i].id >> 4) & 0xf;
        // barNum = pReloc[i].id & 0xf;
        //
        pTegraRelocTable[devIdx].start = pReloc[i].start;
        pTegraRelocTable[devIdx].end   = pReloc[i].start + pReloc[i].size - 1;

        // XXX remove once GPU accesses are correct phys addr
        //assert(pTegraRelocTable[devIdx].start >= DRF_SIZE(LW_RSPACE));

        // Find device instance by searching back for any prior matching devices
        if (devIdx > 0)
        {
            // Search prior devices
            indexInstCnt = devIdx;
            do 
            {
                indexInstCnt--;
                if (pTegraRelocTable[indexInstCnt].devId == devId)
                {
                    pTegraRelocTable[devIdx].devInst = pTegraRelocTable[indexInstCnt].devInst+1;
                    break;
                }
            }
            while (indexInstCnt > 0);
        }

        devIdx++;

        assert(devIdx <= numDevices);
    }

    // Next in the table are interrupts, we don't need them yet.

    return LW_OK;
}

/**
 * @brief retrieves relocation info for a specified device
 *
 * @param[in] devName  Device Name string to retrieve the LW_DEVID_* value
 * @param[in] devIndex Device index to use if multiple devices of the same kind are present
 *
 * @return a DEVICE_RELOCATION pointer, NULL if device not found
 */
PDEVICE_RELOCATION tegrasysGetDeviceReloc(PTEGRASYS pTegraSys,  const char * const devName, U032 devIndex)
{
    PDEVICE_RELOCATION pRelocationTable = pTegraSys->pRelocationTable;
    U032               i, numInst = 0;

    for (i = 0; i < pTegraSys->numDevices; i++)
    {
        if (!strncmp(devName, pRelocationTable[i].devName, strlen(devName)))
        {
            if (numInst == devIndex)
            {
                return &pRelocationTable[i];
            }
            numInst++;
        }
    }

    return NULL;
}

/**
 * @brief  - Lists all the available cheetah Devices
 *
 * @return void
 */
void tegrasysListAllDevs(PTEGRASYS pTegraSys)
{
    PDEVICE_RELOCATION pRelocationTable = pTegraSys->pRelocationTable;
    U032 j;

    dprintf("Here is the list of all available CheetAh Devices:\n");
    for (j = 0; j < pTegraSys->numDevices; j++)
    {
        dprintf("%2d. %s(%d)\n", j+1, pRelocationTable[j].devName, pRelocationTable[j].devInst);
    }
}
