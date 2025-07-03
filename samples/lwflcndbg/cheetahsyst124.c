/***************************** CHEETAH SYS State Rotuines ********************\
*                                                                           *
* Module: TEGRASYS124.C                                                     *
*         CHEETAH System T124 Definitions.                                    *
\***************************************************************************/

#include "tegrasys.h"
#include "t12x/t124/project_relocation_table.h"
#include "t12x/t124/dev_arclk_rst.h"
#include "t12x/t124/dev_bus.h"

/** 
 * @brief parses relocation table and fills out pTegraRelocTable
 * 
 * @return LW_OK if successful, error otherwise
 */
U032 tegrasysInit_T124(PTEGRASYS pTegraSys)
{
    PDEVICE_RELOCATION pDev = NULL;
    DEVICE_LIST devicesToInit[] = 
    {
        { LW_DEVID_GPU, "GPU" },
        { LW_DEVID_HOST1X, "HOST1X" },
        { LW_DEVID_DISPLAY, "DISPLAY" },
        { LW_DEVID_VI, "VI" },
        { LW_DEVID_HDMI, "HDMI" },
        { LW_DEVID_SOR, "SOR" },
        { LW_DEVID_DPAUX, "DPAUX" },
        { LW_DEVID_SHR_SEM, "SEM" },
        { LW_DEVID_CAR, "CAR" },
        { LW_DEVID_FLOW, "FLOW" },
        { LW_DEVID_GPIO, "GPIO" },
        { LW_DEVID_VECTOR, "VECTOR" },
        { LW_DEVID_MISC, "MISC" },
        { LW_DEVID_MC, "MC" },
        { LW_DEVID_FUSE, "FUSE" },
        { LW_DEVID_EMEM, "EMEM" },
        { LW_DEVID_MSENC, "MSENC" },
        { LW_DEVID_PMIF, "PMIF" },
        { LW_DEVID_TSEC, "SEC" },
        { LW_DEVID_DVFS, "DVFS" },
        { LW_DEVID_DSI, "DSI" },
        { LW_DEVID_ICTLR, "ICTLR" },
        { LW_DEVID_VIC, "VIC" }
    };

    LwU64 rawTable[] = 
    {
        LW_RELOCATION_TABLE_INIT,
    };

    LwU32 numDevices = sizeof(devicesToInit)/sizeof(devicesToInit[0]);
    U032  status     = LW_ERR_GENERIC;

    status = tegrasysParseRelocationTable(pTegraSys, rawTable, devicesToInit, numDevices);

    pDev   = tegrasysGetDeviceReloc(&TegraSysObj[indexGpu], "GPU", 0);
    if (pDev)
    {
        // We have a valid GPU device. Set BAR0 as the GPU register base.
        lwBar0 = (U032) pDev->start;
        lwBar1 = lwBar0 + DRF_SIZE(LW_RSPACE);
    }

    return status;
}

/** 
 * @brief Check if device is powered on
 * 
 * @return TRUE if device is powered on
 */
BOOL tegrasysDeviceOn_T124(PTEGRASYS pTegraSys, const char * const devName, LwU32 devInst)
{
    LwU32   i, numInst = 0;
    LwU32   devId = (~0);
    BOOL    clockRunning = TRUE;
    BOOL    inReset = FALSE;
    BOOL    clampEnabled = FALSE;

    // Look up the device ID/instance
    for (i = 0; i < pTegraSys->numDevices; i++)
    {
        if (!strncmp(devName, pTegraSys->pRelocationTable[i].devName, strlen(devName)))
        {
            if (numInst == devInst)
            {
                devId = pTegraSys->pRelocationTable[i].devId;
                break;
            }
            numInst++;
        }
    }

    switch (devId)
    {
        case LW_DEVID_GPU:
            clockRunning = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _CLK_OUT_ENB_X, _CLK_ENB_GPU,_ENABLE);
            inReset      = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _RST_DEVICES_X, _SWR_GPU_RST,_ENABLE);
            //clampEnabled = DEV_FLD_TEST_DRF("???", 0, _PAPBDEV_PMC, _GPU_RG_CNTRL, _RAIL_CLAMP, _ENABLE);
            break;

        case LW_DEVID_DPAUX:
            clockRunning = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _CLK_OUT_ENB_X, _CLK_ENB_DPAUX,_ENABLE);
            inReset      = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _RST_DEVICES_X, _SWR_DPAUX_RST,_ENABLE);
            break;

        case LW_DEVID_VIC:
            clockRunning = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _CLK_OUT_ENB_X, _CLK_ENB_VIC,_ENABLE);
            inReset      = DEV_FLD_TEST_DRF("CAR", 0, _PCLK_RST_CONTROLLER, _RST_DEVICES_X, _SWR_VIC_RST,_ENABLE);
            break;

        default:
            // Just assume the device is powered on and accessible
            clockRunning = TRUE;
            inReset = FALSE;
            clampEnabled = FALSE;
            break;
    }

    // Return TRUE if device is powered on and accessible
    return ((clockRunning==TRUE) && (inReset==FALSE) && (clampEnabled==FALSE));
}
