/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//*****************************************************
//
// lwwatch extension
// deviceinfo.c
//
//*****************************************************

#include "chip.h"
#include "deviceinfo.h"
#include "os.h"
#include "fifo.h"

DeviceInfo deviceInfo;

LW_STATUS deviceInfoAlloc(void)
{
    LwLength deviceInfoRowsAllocSize = pFifo[indexGpu].fifoGetDeviceInfoNumRows() *
                                       sizeof(*deviceInfo.pRows);
    LwLength deviceInfoEnginesAllocSize = pFifo[indexGpu].fifoGetDeviceInfoMaxDevices() *
                                          sizeof(*deviceInfo.pEngines);
    if (NULL == deviceInfo.pRows)
    {
        deviceInfo.pRows = malloc(deviceInfoRowsAllocSize);
        if (NULL == deviceInfo.pRows)
        {
            dprintf("**ERROR: %s: Failed to allocate memory\n", __FUNCTION__);
            return LW_ERR_NO_MEMORY;
        }
        memset(deviceInfo.pRows, 0, deviceInfoRowsAllocSize);

        deviceInfo.pEngines = malloc(deviceInfoEnginesAllocSize);
        if (NULL == deviceInfo.pEngines)
        {
            free(deviceInfo.pRows);
            deviceInfo.pRows = NULL;
            dprintf("**ERROR: %s: Failed to allocate memory\n", __FUNCTION__);
            return LW_ERR_NO_MEMORY;
        }
        memset(deviceInfo.pEngines, 0, deviceInfoEnginesAllocSize);
    }
    return LW_OK;
}

void deviceInfoDump(void)
{
    LwU32 i;
    LwU32 idxPbdma;
    if (LW_OK == pFifo[indexGpu].fifoGetDeviceInfo())
    {
        for (i = 0; i < pFifo[indexGpu].fifoGetDeviceInfoNumRows(); i++)
        {
            if (!deviceInfo.pRows[i].bValid)
                continue;
            dprintf("lw: DEVICE_INFO(%02d) = 0x%08x (%s 0x%08x %s)\n",
                    i, deviceInfo.pRows[i].value,
                    (deviceInfo.pRows[i].bInChain ? " chain" : "!chain"),
                    deviceInfo.pRows[i].data,
                    ((deviceInfo.pRows[i].type == NULL) ? "null" : deviceInfo.pRows[i].type));
        }

        dprintf("\nDecoded Device Info Table\n\n");
        dprintf("NumRows        MaxRowsPerDevice        MaxDevices       Version\n");
        if (deviceInfo.cfg.version >= 2)
        {
            dprintf("%-5d          %-5d                   %-5d            ",
                deviceInfo.cfg.numRows, deviceInfo.cfg.maxRowsPerDevice,
                deviceInfo.cfg.maxDevices);
        }
        else
        {
            dprintf("-----          -----                   -----            ");
        }
        dprintf("%-3d\n", deviceInfo.cfg.version);
        dprintf("\n");
        if (deviceInfo.cfg.version >= 2)
        {
            dprintf("Idx Name      Id   Type Inst Runlist Fault ID Reset NPdma  ");
        }
        else
        {
            dprintf("Idx Name      Id   Type Inst Runlist Intr Fault ID Reset NPbdma ");
        }

        for (i = 0; i < pFifo[indexGpu].fifoGetPbdmaConfigSize(); i++)
        {
            printf("Pbdma%d Pb%d Fault ID ", i, i);
        }
        dprintf("RLPribase RLEngId ChramPriBase\n");

        for (i = 0; i < deviceInfo.enginesCount; i++)
        {
            dprintf("%3d %-9s ", i, deviceInfo.pEngines[i].engineName);

            if (deviceInfo.pEngines[i].bHostEng)
            {
                dprintf("0x%02x ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_FIFO_TAG]);
            }
            else
            {
                dprintf("---- ");
            }
            dprintf("0x%02x ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_ENGINE_TYPE]);
            dprintf("0x%02x ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INST_ID]);

            if (deviceInfo.pEngines[i].bHostEng)
            {
                dprintf("0x%02x    ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST]);
            }
            else
            {
                dprintf("----    ");
            }

            if (deviceInfo.cfg.version < 2)
            {
                dprintf("0x%02x ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_INTR]);
            }
            else
            {
                dprintf("---- (see -intr command for interrupts) ");
            }
            
            if ((deviceInfo.cfg.version >= 2) && deviceInfo.pEngines[i].bHostEng) {
                dprintf("0x%02x     ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_MMU_FAULT_ID]) ;
            } else {
                dprintf("----     ");
            }
            dprintf("0x%02x  ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RESET]);

            if (deviceInfo.pEngines[i].bHostEng)
            {
                dprintf("%02d     ", deviceInfo.pEngines[i].numPbdmas);
                for (idxPbdma = 0; idxPbdma < deviceInfo.pEngines[i].numPbdmas; idxPbdma++)
                {
                    dprintf("0x%02x   ", deviceInfo.pEngines[i].pPbdmaIds[idxPbdma]);
                    if (deviceInfo.pEngines[i].pPbdmaFaultIds) {
                        dprintf("0x%02x         ", deviceInfo.pEngines[i].pPbdmaFaultIds[idxPbdma]);
                    } else {
                        dprintf("----         ");
                    }
                }
                for (;idxPbdma < pFifo[indexGpu].fifoGetPbdmaConfigSize(); idxPbdma++)
                {
                    dprintf("----   ----         ");
                }
                if (deviceInfo.cfg.version >= 2) {
                    dprintf("0x%06x  ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST_PRI_BASE]);
                    dprintf("0x%02x    ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_RUNLIST_ENGINE_ID]);
                    dprintf("0x%06x     ", deviceInfo.pEngines[i].engineData[ENGINE_INFO_TYPE_CHRAM_PRI_BASE]);
                }
                else {
                    dprintf("--------  ");
                    dprintf("----    ");
                    dprintf("--------     ");
                }
            }
            else
            {
                dprintf("----   ");
                for (idxPbdma = 0;idxPbdma < pFifo[indexGpu].fifoGetPbdmaConfigSize(); idxPbdma++)
                {
                    dprintf("----   ----         ");
                }
                dprintf("--------  ");
                dprintf("----    ");
                dprintf("--------     ");
            }

            dprintf("\n");
        }
        dprintf("NOTE: - in the decoded table means no valid data for that part of an engine\n");
    }
}
