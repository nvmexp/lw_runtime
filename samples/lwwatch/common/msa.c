/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2002 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/***************************************************************************\
*                                                                           *
* Module: msa.C                                                             *
*   This file implements Main Stream Attribute (MSA) interface for LWWATCH. *
*                                                                           *
*   For all the fields in MSA except Mvid/Lwid which is dynamically         *
*   callwlated by HW, it provides a set of privilege registers which can    *
*   programmed by SW at anytime. For the raster configuration such as       *
*   Htotal/Hstart/Hwidth/Vtotal/Vstart/Vheight/HSP/VSP/HSW/VSW, each has a  *
*   corresponding field-wise mask to indicate that its privilege register   *
*   will be used to override the HW encoded value or not. For the MISC0 and *
*   MISC1 and 3 reserved bytes, they have bit-wise mask to indicate which   *
*   bit will be used to override the HW encoded value. For whole MSA, it    *
*   has an Enable bit to indicate software programmed MSA enable or disable *
*   and a Trigger/Pending/Done bit to make sure whole software programmed   *
*   MSA taking effect simultaneously.                                       *
*                                                                           *
*   Amit Kumar <asinghal@lwpu.com>  10.08.2011                            *
*                                                                           *
\***************************************************************************/

//
// includes
//

#include "msa.h"
#include "disp/v02_02/dev_disp.h"


typedef struct
{
    LwU8         miscMask[2];
    LwBool       bRasterTotalHorizontal;
    LwBool       bRasterTotalVertical;
    LwBool       bActiveStartHorizontal;
    LwBool       bActiveStartVertical;
    LwBool       bSurfaceTotalHorizontal;
    LwBool       bSurfaceTotalVertical;
    LwBool       bSyncWidthHorizontal;
    LwBool       bSyncPolarityHorizontal;
    LwBool       bSyncHeightVertical;
    LwBool       bSyncPolarityVertical;
    LwBool       bReservedEnable[3];
} MsaPropertiesMask;

typedef struct
{
    LwU8         misc[2];
    LwU16        rasterTotalHorizontal;
    LwU16        rasterTotalVertical;
    LwU16        activeStartHorizontal;
    LwU16        activeStartVertical;
    LwU16        surfaceTotalHorizontal;
    LwU16        surfaceTotalVertical;
    LwU16        syncWidthHorizontal;
    LwU16        syncPolarityHorizontal;
    LwU16        syncHeightVertical;
    LwU16        syncPolarityVertical;
    LwU8         reserved[3];
} MsaPropertiesValues;

/*!
 * @brief Sets MSA properties and enable or disable individually with provided mask
 *
 *
 * @param[in]   SfId                sfId on which SW MSA values need to be set
 * @param[in]   bEnableMSA          LW_TRUE if MSA needs to be enabled
 * @param[in]   bStereoPhaseIlwerse LW_TRUE if in Ilwerse Mode
 * @param[in]   pFeatureMask        pointer to mask for each MSA property
 * @param[in]   pFeatureValues      pointer to values for each MSA property
 *
 */
static void
setMsaProperties
(
    unsigned   sfId,
    LwBool   bEnableMSA,
    LwBool   bStereoPhaseIlwerse,
    MsaPropertiesMask   *pFeatureMask,
    MsaPropertiesValues *pFeatureValues
)
{
    LwU32       regValue = 0x0;
    LwU32       mask     = 0x0;

    osPerfDelay(1000);    //Delay of 1 ms to wait for any previous transaction, if any
    if(!GPU_FLD_TEST_DRIF_DEF(_PDISP, _SF_DP_MSA, sfId, _CNTL, _DONE))
    {
        dprintf("\n\n*****Engine is busy so Can't write MSA Values now.\n");
        return;
    }

    //Set properties values accordingly
    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_MISC(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_MISC, _MISC0,
        (pFeatureValues->misc[0]), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_MISC, _MISC1,
        (pFeatureValues->misc[1]), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_MISC(sfId), regValue);

    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_RASTER_TOTAL(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_RASTER_TOTAL, _HORIZONTAL,
        (pFeatureValues->rasterTotalHorizontal), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_RASTER_TOTAL, _VERTICAL,
        (pFeatureValues->rasterTotalVertical), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_RASTER_TOTAL(sfId), regValue);

    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_ACTIVE_START(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_ACTIVE_START, _HORIZONTAL,
        (pFeatureValues->activeStartHorizontal), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_ACTIVE_START, _VERTICAL,
        (pFeatureValues->activeStartVertical), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_ACTIVE_START(sfId), regValue);

    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_SURFACE_TOTAL(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SURFACE_TOTAL, _HORIZONTAL,
        (pFeatureValues->surfaceTotalHorizontal), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SURFACE_TOTAL, _VERTICAL,
        (pFeatureValues->surfaceTotalHorizontal), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_SURFACE_TOTAL(sfId), regValue);

    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_SYNC(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SYNC, _WIDTH_HORIZONTAL,
        (pFeatureValues->syncWidthHorizontal), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SYNC, _POLARITY_HORIZONTAL,
        (pFeatureValues->syncPolarityHorizontal), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SYNC, _HEIGHT_VERTICAL,
        (pFeatureValues->syncHeightVertical), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_SYNC_POLARITY, _VERTICAL,
        (pFeatureValues->syncPolarityVertical), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_SYNC(sfId), regValue);

    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_RESERVED(sfId));
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_RESERVED, _FIRST_BYTE,
        (pFeatureValues->reserved[0]), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_RESERVED, _SECOND_BYTE,
        (pFeatureValues->reserved[1]), regValue);
    regValue = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_RESERVED, _THIRD_BYTE,
        (pFeatureValues->reserved[2]), regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_RESERVED(sfId), regValue);

    //Set properties mask accordingly
    mask = GPU_REG_RD32(LW_PDISP_SF_DP_MSA_MASK(sfId));
    mask = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_MASK, _MISC0,
        (pFeatureMask->miscMask[0]), mask);
    mask = FLD_SET_DRF_NUM(_PDISP, _SF_DP_MSA_MASK, _MISC1,
        (pFeatureMask->miscMask[1]), mask);

    if(LW_TRUE == pFeatureMask->bRasterTotalHorizontal)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _RASTER_HORIZONTAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _RASTER_HORIZONTAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bRasterTotalVertical)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _RASTER_VERTICAL, _ENABLE,
                           mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _RASTER_VERTICAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bActiveStartHorizontal)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _ACTIVE_START_HORIZONTAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _ACTIVE_START_HORIZONTAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bActiveStartVertical)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _ACTIVE_START_VERTICAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _ACTIVE_START_VERTICAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSurfaceTotalHorizontal)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SURFACE_HORIZONTAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SURFACE_HORIZONTAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSurfaceTotalVertical)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SURFACE_VERTICAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SURFACE_VERTICAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSyncWidthHorizontal)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_WIDTH_HORIZONTAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_WIDTH_HORIZONTAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSyncPolarityHorizontal)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_POLARITY_HORIZONTAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_POLARITY_HORIZONTAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSyncHeightVertical)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_HEIGHT_VERTICAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_HEIGHT_VERTICAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bSyncPolarityVertical)
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_POLARITY_VERTICAL,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SYNC_POLARITY_VERTICAL,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bReservedEnable[0])
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _FIRST_RESERVED_BYTE,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _FIRST_RESERVED_BYTE,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bReservedEnable[1])
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SECOND_RESERVED_BYTE,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _SECOND_RESERVED_BYTE,
                           _DISABLE, mask);

    if(LW_TRUE == pFeatureMask->bReservedEnable[2])
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _THIRD_RESERVED_BYTE,
                           _ENABLE, mask);
    else
        mask = FLD_SET_DRF(_PDISP, _SF_DP_MSA_MASK, _THIRD_RESERVED_BYTE,
                           _DISABLE, mask);

    GPU_REG_WR32(LW_PDISP_SF_DP_MSA_MASK(sfId), mask);

    //Enable or Disable MSA
    regValue = GPU_REG_RD32(LW_PDISP_SF_DP_MSA(sfId));

    if(LW_TRUE == bStereoPhaseIlwerse)
        regValue = FLD_SET_DRF(_PDISP, _SF_DP_MSA, _STEREO_PHASE_ILWERSE,
                               _YES, regValue);
    else
        regValue = FLD_SET_DRF(_PDISP, _SF_DP_MSA, _STEREO_PHASE_ILWERSE, _NO,
                               regValue);

    if(LW_TRUE == bEnableMSA)
        regValue = FLD_SET_DRF(_PDISP, _SF_DP_MSA, _ENABLE, _YES, regValue);
    else
        regValue = FLD_SET_DRF(_PDISP, _SF_DP_MSA, _ENABLE, _NO, regValue);

    regValue = FLD_SET_DRF(_PDISP, _SF_DP_MSA, _CNTL, _TRIGGER, regValue);
    GPU_REG_WR32(LW_PDISP_SF_DP_MSA(sfId), regValue);

    osPerfDelay(1000);    //Delay of 1 ms to wait for operation completion
    if(!GPU_FLD_TEST_DRIF_DEF(_PDISP, _SF_DP_MSA, sfId, _CNTL, _DONE))
    {
        dprintf("\n\n*****Engine is busy so Can't write MSA Values now.\n");
        return;
    }

}

/*!
 * @brief Get MSA Debug properies values which are actually set on hw
 *
 * @param[in]   SfId                sfId from which HW MSA values need to be extracted
 * @param[out]  pFeatureValues      Will contain values returned by Hw debug regs
 *
 */
static void
getMsaHwProperties
(
    LwU8    sfId,
    MsaPropertiesValues *pFeatureValues
)
{
    pFeatureValues->misc[0] = (LwU8)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_MISC, sfId, _MISC0);
    pFeatureValues->misc[1] = (LwU8)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_MISC, sfId, _MISC1);

    pFeatureValues->rasterTotalHorizontal = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_RASTER_TOTAL, sfId, _HORIZONTAL);
    pFeatureValues->rasterTotalVertical = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_RASTER_TOTAL, sfId, _VERTICAL);

    pFeatureValues->activeStartHorizontal = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_ACTIVE_START, sfId, _HORIZONTAL);
    pFeatureValues->activeStartVertical = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_ACTIVE_START, sfId, _VERTICAL);

    pFeatureValues->surfaceTotalHorizontal = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SURFACE_TOTAL, sfId, _HORIZONTAL);
    pFeatureValues->surfaceTotalVertical = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SURFACE_TOTAL, sfId, _VERTICAL);

    pFeatureValues->syncWidthHorizontal = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SYNC, sfId, _WIDTH_HORIZONTAL);
    pFeatureValues->syncPolarityHorizontal = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SYNC, sfId, _POLARITY_HORIZONTAL);

    pFeatureValues->syncHeightVertical = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SYNC, sfId, _HEIGHT_VERTICAL);
    pFeatureValues->syncPolarityVertical = (LwU16)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_SYNC, sfId, _POLARITY_VERTICAL);

    pFeatureValues->reserved[0] = (LwU8)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_RESERVED, sfId, _FIRST_BYTE);
    pFeatureValues->reserved[1] = (LwU8)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_RESERVED, sfId, _SECOND_BYTE);
    pFeatureValues->reserved[2] = (LwU8)
        GPU_REG_IDX_RD_DRF(_PDISP, _SF_DP_MSA_DEBUG_RESERVED, sfId, _THIRD_BYTE);
}


/*!
 * @brief To print MSA values & Mask configuration
 *
 * @param[in]  bEnableMsa          True to enable MSA & false to disable it
 * @param[in]  pDebugValues        MSA values which are lwrrently being written on HW
 * @param[in]  pValues             MSA Values which user wants to program
 * @param[in]  pMask               Individual MSA attribute mask to enable or disable
 *
 */
static void printMsaProperites(
LwBool bEnableMsa,
MsaPropertiesValues *pDebugValues,
MsaPropertiesValues *pValues,
MsaPropertiesMask   *pMask
)
{
    dprintf("\nMSA Properties Values being written on HW");
    dprintf("\n======================================================");
    dprintf("\n Misc[2]:                %08x    %08x",
                            pDebugValues->misc[0], pDebugValues->misc[1]);
    dprintf("\n RasterTotalHorizontal:  %08x",  pDebugValues->rasterTotalHorizontal);
    dprintf("\n RasterTotalVertical:    %08x",  pDebugValues->rasterTotalVertical);
    dprintf("\n ActiveStartHorizontal:  %08x",  pDebugValues->activeStartHorizontal);
    dprintf("\n ActiveStartVertical:    %08x",  pDebugValues->activeStartVertical);
    dprintf("\n SurfaceTotalHorizontal: %08x",  pDebugValues->surfaceTotalHorizontal);
    dprintf("\n SurfaceTotalVertical:   %08x",  pDebugValues->surfaceTotalVertical);
    dprintf("\n SyncWidthHorizontal:    %08x",  pDebugValues->syncWidthHorizontal);
    dprintf("\n SyncPolarityHorizontal: %08x",  pDebugValues->syncPolarityHorizontal);
    dprintf("\n SyncHeightVertical:     %08x",  pDebugValues->syncHeightVertical);
    dprintf("\n SyncPolarityVertical:   %08x",  pDebugValues->syncPolarityVertical);
    dprintf("\n Reserved[3]:            %08x    %08x    %08x",
            pDebugValues->reserved[0], pDebugValues->reserved[1], pDebugValues->reserved[2]);


    dprintf("\n\nMSA Properties Values you are suppose it write on HW");
    dprintf("\n======================================================");
    dprintf("\n Enable MSA = [%01x]\n", bEnableMsa);
    dprintf("\n Misc[2]:                %08x [%01x]    %08x [%01x]",
            pValues->misc[0], pMask->miscMask[0], pValues->misc[1], pMask->miscMask[1]);
    dprintf("\n RasterTotalHorizontal:  %08x [%01x]",  pValues->rasterTotalHorizontal, pMask->bRasterTotalHorizontal);
    dprintf("\n RasterTotalVertical:    %08x [%01x]",  pValues->rasterTotalVertical, pMask->bRasterTotalVertical);
    dprintf("\n ActiveStartHorizontal:  %08x [%01x]",  pValues->activeStartHorizontal, pMask->bActiveStartHorizontal);
    dprintf("\n ActiveStartVertical:    %08x [%01x]",  pValues->activeStartVertical, pMask->bActiveStartVertical);
    dprintf("\n SurfaceTotalHorizontal: %08x [%01x]",  pValues->surfaceTotalHorizontal, pMask->bSurfaceTotalHorizontal);
    dprintf("\n SurfaceTotalVertical:   %08x [%01x]",  pValues->surfaceTotalVertical, pMask->bSurfaceTotalVertical);
    dprintf("\n SyncWidthHorizontal:    %08x [%01x]",  pValues->syncWidthHorizontal, pMask->bSyncWidthHorizontal);
    dprintf("\n SyncPolarityHorizontal: %08x [%01x]",  pValues->syncPolarityHorizontal, pMask->bSyncPolarityHorizontal);
    dprintf("\n SyncHeightVertical:     %08x [%01x]",  pValues->syncHeightVertical, pMask->bSyncHeightVertical);
    dprintf("\n SyncPolarityVertical:   %08x [%01x]",  pValues->syncPolarityVertical, pMask->bSyncPolarityVertical);
    dprintf("\n Reserved[3]:            %08x [%01x]    %08x [%01x]    %08x [%01x]",
            pValues->reserved[0], pMask->bReservedEnable[0], pValues->reserved[1], pMask->bReservedEnable[1], pValues->reserved[2], pMask->bReservedEnable[2]);
}

/*!
 * @brief Prints all options available in MSA menu
 *
 */
static void printhelp()
{
    dprintf("lw: msa help file\n");
    dprintf("USAGE: <COMMAND> [ARGS]\n\n");

    dprintf("COMMAND    ARGUMENTS       DESCRIPTION\n");
    dprintf("~~~~~~~    ~~~~~~~~~       ~~~~~~~~~~~\n\n");

    dprintf("   q                       Quit the msa interface.\n");
    dprintf("   ?                       Prints this help file.\n");
    dprintf("   sf       <sfID>         Changes the current sf ID to enable MSA on.\n");
    dprintf("   w                       Write the updated MSA values on registers.\n");
    dprintf("   p                       Print current MSA values being written on HW.\n");

    dprintf("\nEnter 0 for Disable & 1 for Enable for following:\n");
    dprintf("   bMsa     <0/1>           Enable or Disable MSA \n");
    dprintf("   bMisc0  <0/1>           [Misc 0th byte]\n");
    dprintf("   bMisc1  <0/1>           [Misc 1th byte]\n");
    dprintf("   bRTH    <0/1>           [RasterTotalHorizontal]\n");
    dprintf("   bRTV    <0/1>           [RasterTotalVertical]\n");
    dprintf("   bASH    <0/1>           [ActiveStartHorizontal]\n");
    dprintf("   bASV    <0/1>           [ActiveStartVertical]\n");
    dprintf("   bSTH    <0/1>           [SurfaceTotalHorizontal]\n");
    dprintf("   bSTV    <0/1>           [SurfaceTotalVertical]\n");
    dprintf("   bSWH    <0/1>           [SyncWidthHorizontal]\n");
    dprintf("   bSPH    <0/1>           [SyncPolarityHorizontal]\n");
    dprintf("   bSHV    <0/1>           [SyncHeightVertical]\n");
    dprintf("   bSPV    <0/1>           [SyncPolarityVertical]\n");
    dprintf("   bRes0   <0/1>           [Reserved 0th byte]\n");
    dprintf("   bRes1   <0/1>           [Reserved 1th byte]\n");
    dprintf("   bRes2   <0/1>           [Reserved 2th byte]\n");

    dprintf("\nEnter property value to write on HW:\n");
    dprintf("   vMisc0  <value>         [Misc 0th byte]\n");
    dprintf("   vMisc1  <value>         [Misc 1th byte]\n");
    dprintf("   vRTH    <value>         [RasterTotalHorizontal]\n");
    dprintf("   vRTV    <value>         [RasterTotalVertical]\n");
    dprintf("   vASH    <value>         [ActiveStartHorizontal]\n");
    dprintf("   vASV    <value>         [ActiveStartVertical]\n");
    dprintf("   vSTH    <value>         [SurfaceTotalHorizontal]\n");
    dprintf("   vSTV    <value>         [SurfaceTotalVertical]\n");
    dprintf("   vSWH    <value>         [SyncWidthHorizontal]\n");
    dprintf("   vSPH    <value>         [SyncPolarityHorizontal]\n");
    dprintf("   vSHV    <value>         [SyncHeightVertical]\n");
    dprintf("   vSPV    <value>         [SyncPolarityVertical]\n");
    dprintf("   vRes0   <value>         [Reserved 0th byte]\n");
    dprintf("   vRes1   <value>         [Reserved 1th byte]\n");
    dprintf("   vRes2   <value>         [Reserved 2th byte]\n");
}

/*!
 * @brief Controls MSA menu & its operation
 *
 */

void msaMenu()
{
    LwU8 done=0;
    char input1024[1024];
    char *command = input1024;
    char *argument = input1024;
    LwU8 sfId = 0;
    LwBool   bEnableMSA = LW_FALSE;
    LwBool   bStereoPhaseIlwerse = LW_FALSE;
    MsaPropertiesMask   featureMask;
    MsaPropertiesValues featureValues, featureDebugValues;

    memset(&featureMask, 0, sizeof(featureMask));
    memset(&featureValues, 0, sizeof(featureValues));
    memset(&featureDebugValues, 0, sizeof(featureDebugValues));

    dprintf("lw: Starting i2c Menu. (Type '?' for help)\n");

    while (!done)
    {
        LwU64 argVal = 0;
        dprintf("\n");
        memset(input1024, 0, sizeof(input1024));

        dprintf("current sf: %u\n", sfId);

        if (osGetInputLine((LwU8 *)"msa> ", (LwU8 *)input1024, sizeof(input1024)))
        {
            //true if argument is present & separated by ' '
            if(NULL != (argument = strchr(input1024, ' ')))
            {
                argument[0] = '\0';
                argument++;
                skipDelims(&argument, GENERIC_DELIMS);
                GetExpressionEx(argument, &argVal, &argument);
            }
            struppr(command);

            if (0 == strcmp(command, "Q"))
            {
                dprintf("Exiting user I2C interface.\n");
                done = 1;
            }
            else if (0 == strcmp(command, "SF"))
            {
                sfId = (LwU8)argVal;
            }
            else if (0 == strcmp(command, "P"))
            {
                getMsaHwProperties(sfId, &featureDebugValues);
                printMsaProperites(bEnableMSA, &featureDebugValues, &featureValues, &featureMask);
            }
            else if (0 == strcmp(command, "W"))
            {
                setMsaProperties(sfId, bEnableMSA, bStereoPhaseIlwerse, &featureMask, &featureValues);
            }
            else if (0 == strcmp(command, "BMSA"))
            {
                bEnableMSA = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BMISC0"))
            {
                featureMask.miscMask[0] = (1 == argVal) ? 0xFF : 0x00;
            }
            else if (0 == strcmp(command, "BMISC1"))
            {
                featureMask.miscMask[1] = (1 == argVal) ? 0xFF : 0x00;
            }
            else if (0 == strcmp(command, "BRTH"))
            {
                featureMask.bRasterTotalHorizontal = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BRTV"))
            {
                featureMask.bRasterTotalVertical = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BASH"))
            {
                featureMask.bActiveStartHorizontal = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BASV"))
            {
                featureMask.bActiveStartVertical = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSTH"))
            {
                featureMask.bSurfaceTotalHorizontal = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSTV"))
            {
                featureMask.bSurfaceTotalVertical = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSWH"))
            {
                featureMask.bSyncWidthHorizontal = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSPH"))
            {
                featureMask.bSyncPolarityHorizontal = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSHV"))
            {
                featureMask.bSyncHeightVertical = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BSPV"))
            {
                featureMask.bSyncPolarityVertical = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BRES0"))
            {
                featureMask.bReservedEnable[0] = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BRES1"))
            {
                featureMask.bReservedEnable[1] = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "BRES2"))
            {
                featureMask.bReservedEnable[2] = (1 == argVal) ? LW_TRUE : LW_FALSE;
            }
            else if (0 == strcmp(command, "VMISC0"))
            {
                featureValues.misc[0] = (LwU8)argVal;
            }
            else if (0 == strcmp(command, "VMISC1"))
            {
                featureValues.misc[1] = (LwU8)argVal;
            }
            else if (0 == strcmp(command, "VRTH"))
            {
                featureValues.rasterTotalHorizontal = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VRTV"))
            {
                featureValues.rasterTotalVertical = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VASH"))
            {
                featureValues.activeStartHorizontal = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VASV"))
            {
                featureValues.activeStartVertical = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSTH"))
            {
                featureValues.surfaceTotalHorizontal = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSTV"))
            {
                featureValues.surfaceTotalVertical = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSWH"))
            {
                featureValues.syncWidthHorizontal = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSPH"))
            {
                featureValues.syncPolarityHorizontal = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSHV"))
            {
                featureValues.syncHeightVertical = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VSPV"))
            {
                featureValues.syncPolarityVertical = (LwU16)argVal;
            }
            else if (0 == strcmp(command, "VRES0"))
            {
                featureValues.reserved[0] = (LwU8)argVal;
            }
            else if (0 == strcmp(command, "VRES1"))
            {
                featureValues.reserved[1] = (LwU8)argVal;
            }
            else if (0 == strcmp(command, "VRES2"))
            {
                featureValues.reserved[2] = (LwU8)argVal;
            }

            else
            {
                if (0 != strcmp(command, "?"))
                    dprintf("*** Unknown command!  Printing help...\n\n");
                printhelp();
            }


        }
    }

 }



