/***************************** SOC BRDG State Rotuines *********************\
*                                                                           *
* Module: SOCBRDGT124.C                                                     *
*         SOC Bridge T124 Definitions.                                      *
\***************************************************************************/

#include "socbrdg.h"

#include "t12x/t124/dev_bus.h"
#include "t12x/t124/dev_fai_cfg.h"
#include "t12x/t124/dev_fai_bar0.h"

#include "tegrasys.h"

/**
 * FUNCTION DECLARATIONS
 */

/**
 * Global definitions
 */
struct OBJSOCBRDG *pSocBrdgLw;

/** 
 * @brief Initializes the SOC bridge
 *
 * @return LW_OK if successful, error otherwise
 */
LW_STATUS socbrdgInit_T124()
{
    LwU32 i;
    LwU64 winLoc;
    LwU32 numWindows = pSocbrdg[indexGpu].socbrdgLwGetNumWindows();

    pSocBrdgLw = (struct OBJSOCBRDG *)malloc(sizeof(struct OBJSOCBRDG));
    assert(pSocBrdgLw);
    memset(pSocBrdgLw, 0x0, sizeof(struct OBJSOCBRDG));

    // for rev 2 of peatrans, window address is shifted 8 bits to allow 40b addressing
    if((lwClassCodeRevId & DRF_MASK(LW_FAI_CFG_REV_CC_REVISION_ID)) == LW_FAI_CFG_REV_CC_REVISION_ID_2)
    {
        pSocBrdgLw->winAddrShift = 8;
        pSocBrdgLw->winAddrMask = (1 << pSocBrdgLw->winAddrShift) - 1;
    }

    pSocBrdgLw->regPhysAddr = lwBar0;
    pSocBrdgLw->winPhysAddr = lwBar1 & DRF_SHIFTMASK(LW_FAI_CFG_BAR_1_BASE_ADDRESS);

    winLoc = pSocBrdgLw->winPhysAddr;

    pSocBrdgLw->windowInfo = (SOCBRDG_WINDOW_INFO *)malloc(numWindows * sizeof(SOCBRDG_WINDOW_INFO));
    assert(pSocBrdgLw->windowInfo);

    for (i = 0; i < numWindows; i++)
    {
        pSocBrdgLw->windowInfo[i].windowLoc = winLoc;
        pSocBrdgLw->windowInfo[i].windowSize = pSocbrdg[indexGpu].socbrdgLwGetWindowSize(i);
        pSocBrdgLw->windowInfo[i].windowTarget =
            (LwU64)RD_PHYS32(lwBar0 + (LW_FAI_BAR1_WINDOW0 + i*4)) << pSocBrdgLw->winAddrShift;

        winLoc += pSocBrdgLw->windowInfo[i].windowSize;
    }

    pSocbrdg[indexGpu].socbrdgSaveState();

    return LW_OK;
}

/** 
 * @brief Destroy SOCBRDG.  Undoes what socbrdgInit did.
 * 
 * @return LW_OK if successful, error otherwise
 */
LW_STATUS socbrdgDestroy_T124()
{
    if (pSocBrdgLw->windowInfo)
    {
        free(pSocBrdgLw->windowInfo);
        pSocBrdgLw->windowInfo = NULL;
    }

    if (pSocBrdgLw)
    {
        free(pSocBrdgLw);
        pSocBrdgLw = NULL;
    }

    return LW_OK;
}

/** 
 * @brief Sets the target address for the specified window
 * 
 * @param[in] idx    index of window
 * @param[in] target target address to set for the window
 *
 * @returns Nothing
 */
void socbrdgLwSetWindow_T124(LwU32 idx, LwU64 target)
{
    assert(idx < 8);

    target = target >> pSocBrdgLw->winAddrShift;
    WR_PHYS32(lwBar0 + (LW_FAI_BAR1_WINDOW0 + idx*4), ((LwU32)((LwU64)target & 0xffffffff)));
    pSocBrdgLw->windowInfo[idx].windowTarget = target << pSocBrdgLw->winAddrShift;
}

/** 
 * @brief Gets the target address for the specified window
 * 
 * @param[in] idx    index of window
 *
 * @returns The target address
 */
LwU64 socbrdgLwGetWindow_T124(LwU32 idx)
{
    return (LwU64)RD_PHYS32(lwBar0 + (LW_FAI_BAR1_WINDOW0 + idx*4)) << pSocBrdgLw->winAddrShift;
}

/** 
 * @brief Returns the size of a specified window
 * 
 * @param[in] idx index of window
 *
 * @returns size in bytes of requested window
 */
LwU64 socbrdgLwGetWindowSize_T124(LwU32 idx)
{
    assert(idx < 8);
    //return (LwU64)RD_PHYS32(lwBar0 + LW_FAI_BAR1_SIZE_REMAP_WINDOWS);
    return 128*1024*1024;
}

/** 
 * @brief Retrieves Physical address of the requested device register
 *
 * Moves the SOCBRDG_WINDOW_REG window if necessary to point at the requested
 * location and returns the correct PHYS address.
 * 
 * @param target 
 *
 * @return PHYS address to requested device register
 */
LwU64 socbrdgLwGetTunnelAddress_T124(LwU64 target)
{
    PSOCBRDG_WINDOW_INFO pWindow = &pSocBrdgLw->windowInfo[SOCBRDG_WINDOW_REG];
    LwU64 winLoc = pWindow->windowLoc;
    LwU64 windowTarget = 0;

    if ((pWindow->windowTarget > target) ||
        (pWindow->windowTarget + pWindow->windowSize <= target))
    {
        socbrdgLwSetWindow_T124(SOCBRDG_WINDOW_REG, target);
        windowTarget = target;
    }

    return winLoc + (target - pWindow->windowTarget);
}

/** 
 * @brief Retrieve the number of available windows
 * 
 * @return number of windows
 */
LwU32 socbrdgLwGetNumWindows_T124()
{
    return SOCBRDG_WINDOW_ILWALID;
    //return RD_PHYS32(lwBar0 + LW_FAI_BAR1_NUM_REMAP_WINDOWS);
}

/** 
 * @brief Saves the state of the SOCBRDG windows before the extension exelwtes
 * 
 * @return void
 */
void socbrdgSaveState_T124()
{
    pSocBrdgLw->oldSocBrdgREGWindow = pSocbrdg[indexGpu].socbrdgLwGetWindow(SOCBRDG_WINDOW_REG);
    pSocBrdgLw->oldSocBrdgBAR1Window = pSocbrdg[indexGpu].socbrdgLwGetWindow(SOCBRDG_WINDOW_BAR1);
}

/** 
 * @brief Restores the state of the SOCBRDG windows after the extension is exelwted
 * 
 * @return void
 */
void socbrdgRestoreState_T124()
{
    pSocbrdg[indexGpu].socbrdgLwSetWindow(SOCBRDG_WINDOW_REG, pSocBrdgLw->oldSocBrdgREGWindow);
    pSocbrdg[indexGpu].socbrdgLwSetWindow(SOCBRDG_WINDOW_BAR1, pSocBrdgLw->oldSocBrdgBAR1Window);
}
