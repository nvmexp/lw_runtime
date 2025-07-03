
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgm107.c
//
//*****************************************************

//
// includes
//
#include "maxwell/gm107/dev_ltc.h"
#include "maxwell/gm107/dev_fuse.h"
#include "maxwell/gm107/hwproject.h"
#include "fb.h"
#include "sig.h"
#include "g_fb_private.h"

//
// Flags taken from RM for fbL2IlwalidateEvict
//
#define FERMI_FB_L2_ILW_EV_FLAGS_ALL            0x00000001
#define FERMI_FB_L2_ILW_EV_FLAGS_FIRST          0x00000002
#define FERMI_FB_L2_ILW_EV_FLAGS_LAST           0x00000004
#define FERMI_FB_L2_ILW_EV_FLAGS_NORMAL         0x00000008
#define FERMI_FB_L2_ILW_EV_FLAGS_CLEAN          0x00000010

#define DELAY    1000   //1000 us delay between polls
#define POLL_MAX 1000    //num of polls till timeout (1 sec)

LW_STATUS fbL2State_GM107()
{
    LwU32 i, j, k;
    LwU32 val;
    LwU32 valOrig;
    LwU32 val2;
    LwU32 numActiveLTCs  = pFb[indexGpu].fbGetActiveLTCCountLwW();
    LwU32 numLTSPerLTC   = pFb[indexGpu].fbGetLTSPerLTCCountLwW();
    LwU32 numLinesPerLTS = pFb[indexGpu].fbGetLinesPerLTSCountLwW();
    LW_STATUS status     = LW_OK;

    dprintf("lw: Reading out general status:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        for ( j = 0; j < numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_G_STATUS +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_G_STATUS: 0x%x\n", i, j, val);
        }
    }
    dprintf("lw:\n");

    dprintf("lw: Reading out IQ status registers:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        for (j = 0; j < numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            valOrig = val;

            for (k = 0; k < 2; k++ )
            {
                //
                // The value of _STATUS_SELECT controls which information is
                // reported in the _XBAR_STATUS.
                //
                val &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
                val |=  DRF_NUM(_PLTCG, _LTC0_LTS0_IQ_CFG_0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                            (LW_LTC_PRI_STRIDE * i) +
                            (LW_LTS_PRI_STRIDE * j),
                             val);
                dprintf("lw:\tWrote %d to _STATUS_SELECT\n", k);
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_IQ_CFG_0: 0x%x\n",
                        i, j, val);

                val2 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_XBAR_STATUS_0 +
                                   (LW_LTC_PRI_STRIDE * i) +
                                   (LW_LTS_PRI_STRIDE * j));
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_IQ_XBAR_STATUS_0: 0x%x\n",
                        i, j, val2);
            }

            // Restore the original value.
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                        (LW_LTC_PRI_STRIDE * i) +
                        (LW_LTS_PRI_STRIDE * j),
                         valOrig);
        }
    }
    dprintf("lw:\n");

    dprintf("lw: Reading out other _STATUS registers:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        for (j = 0; j < numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_ROP_STATUS_0 +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_IQ_ROP_STATUS_0: 0x%x\n",
                    i, j, val);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_CBC_STATUS +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_CBC_STATUS: 0x%x\n",
                    i, j, val);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_STATUS +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_TSTG_STATUS: 0x%x\n",
                    i, j, val);
        }
    }
    dprintf("lw:\n");

    dprintf("lw: Reading out other _TSTG_SRV_STATUS registers:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        for (j = 0; j < numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            valOrig = val;

            for (k = 0; k < 4; k++ )
            {
                //
                // The value of _STATUS_SELECT controls which information is
                // reported in the _XBAR_STATUS.
                //
                val &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
                val |=  DRF_NUM(_PLTCG, _LTC0_LTS0_IQ_CFG_0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                            (LW_LTC_PRI_STRIDE * i) +
                            (LW_LTS_PRI_STRIDE * j),
                             val);
                dprintf("lw:\tWrote %d to _STATUS_SELECT\n", k);
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_IQ_CFG_0: 0x%x\n",
                        i, j, val);


                val2 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_SRV_STATUS_0 +
                                   (LW_LTC_PRI_STRIDE * i) +
                                   (LW_LTS_PRI_STRIDE * j));
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_TSTG_SRV_STATUS_0: 0x%x\n",
                        i, j, val2);
            }

            // Restore the original value.
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 +
                        (LW_LTC_PRI_STRIDE * i) +
                        (LW_LTS_PRI_STRIDE * j),
                         valOrig);
        }
    }
    dprintf("lw:\n");

    dprintf("lw: Reading out other _DSTG_STATUS registers:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        for (j = 0; j < numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            valOrig = val;

            for (k = 0; k < 25; k++ )
            {
                //
                // The value of _STATUS_SELECT selects which threads or
                // sequencers is the _DSTG_STATUS.
                //
                val &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_DSTG_CFG0_STATUS_SELECT);
                val |=  DRF_NUM(_PLTCG, _LTC0_LTS0_DSTG_CFG0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 +
                            (LW_LTC_PRI_STRIDE * i) +
                            (LW_LTS_PRI_STRIDE * j),
                            val);
                dprintf("lw:\tWrote %d to _STATUS_SELECT\n", k);
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_DSTG_CFG0: 0x%x\n",
                        i, j, val);

                val2 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_DSTG_STATUS_0 +
                                   (LW_LTC_PRI_STRIDE * i) +
                                   (LW_LTS_PRI_STRIDE * j));
                dprintf("lw:\tLW_PLTCG_LTC%d_LTS%d_DSTG_STATUS_0: 0x%x\n",
                        i, j, val2);
            }

            // Restore the original value.
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 +
                        (LW_LTC_PRI_STRIDE * i) +
                        (LW_LTS_PRI_STRIDE * j),
                         valOrig);
        }
    }
    dprintf("lw:\n");

    dprintf("lw:\tReading out _ROP_OUTRUN_STATUS registers:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        val = GPU_REG_RD32(LW_PLTCG_LTC0_MISC_ROP_OUTRUN_STATUS_0 +
                          (LW_LTC_PRI_STRIDE * i));
        dprintf("lw:\tLW_PLTCG_LTC%d_MISC_ROP_OUTRUN_STATUS_0: 0x%x\n",
                i, val);
    }
    dprintf("lw:\n");

    status = pFb[indexGpu].fbL2StateForCacheLines(numActiveLTCs,
                                                  numLTSPerLTC,
                                                  numLinesPerLTS);

    return status;
}

LW_STATUS fbL2StateForCacheLines_GM107
(
    LwU32 numActiveLTCs,
    LwU32 numLTSPerLTC,
    LwU32 numLinesPerLTS
)
{
    LwU32 i, j, k, l;
    LwU32 val;
    LwU32 valOrig;
    LwU32 val2;

    dprintf("lw:\tReading out cache tag stage:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        dprintf("lw:\tLTC %d\n", i);

        for (j = 0; j < numLTSPerLTC; j++)
        {
            dprintf("lw:\t\tLTS %d\n", j);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            valOrig = val;

            // Loops through cache lines...
            for (k = 0; k < numLinesPerLTS; k++ )
            {
                //
                // The value of _ADDRESS controls which L2 cache line tag and
                // state to access.
                //
                val &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX_ADDRESS);
                val |=  DRF_NUM(_PLTCG, _LTC0_LTS0_TSTG_CST_RDI_INDEX, _ADDRESS, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                            (LW_LTC_PRI_STRIDE * i) +
                            (LW_LTS_PRI_STRIDE * j),
                            val);

                dprintf("lw:\t\t\tLine %d\n", k);
                dprintf("lw:\t\t\t\t");
                for (l = 0; l < 3; l++)
                {
                    val2 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_DATA(l) +
                                       (LW_LTC_PRI_STRIDE * i) +
                                       (LW_LTS_PRI_STRIDE * j));
                    dprintf(" [%d]:0x%x", l, val2);
                }
                dprintf("\n");
            }

            // Restore the original value.
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                        (LW_LTC_PRI_STRIDE * i) +
                        (LW_LTS_PRI_STRIDE * j),
                         valOrig);
        }
    }

    return LW_OK;
}

/*!
 *  Force fb L2 ilwalidate Timeout is in msec.
 *
 */
LW_STATUS fbL2IlwalEvict_GM107(LwU32 scaleTime)
{
    LwU32       ltcCounter;
    LwU32       ltcOffset;
    LwU32       cmgmt0;
    LwU32       cmgmt1;
    BOOL        done;
    LwU32       poll;
    LwU32       evictValue = 0;
    LwU32       clealwalue = 0;
    LwU32       numActiveLTCs = pFb[indexGpu].fbGetActiveLTCCountLwW();
    LwU32       maxPolls = scaleTime*POLL_MAX;

    //defaulting with ALL and CLEAN evict
    LwU32       flags = FERMI_FB_L2_ILW_EV_FLAGS_ALL | FERMI_FB_L2_ILW_EV_FLAGS_CLEAN ;

    clealwalue = DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN, _PENDING);
    evictValue = DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE, _PENDING);

    if (flags & FERMI_FB_L2_ILW_EV_FLAGS_ALL)
    {
        evictValue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_LAST_CLASS, _TRUE) |
                      DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_NORMAL_CLASS, _TRUE) |
                      DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_FIRST_CLASS, _TRUE);
        clealwalue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_LAST_CLASS, _TRUE) |
                      DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_NORMAL_CLASS, _TRUE) |
                      DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_FIRST_CLASS, _TRUE);
    }

    if (flags & FERMI_FB_L2_ILW_EV_FLAGS_LAST)
    {
        evictValue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_LAST_CLASS, _TRUE);
        clealwalue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_LAST_CLASS, _TRUE);
    }

    if (flags & FERMI_FB_L2_ILW_EV_FLAGS_NORMAL)
    {
        evictValue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_NORMAL_CLASS, _TRUE);
        clealwalue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_NORMAL_CLASS, _TRUE);
    }

    if (flags & FERMI_FB_L2_ILW_EV_FLAGS_FIRST)
    {
        evictValue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE_EVICT_FIRST_CLASS, _TRUE);
        clealwalue |= DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN_EVICT_FIRST_CLASS, _TRUE);
    }

    if (flags & FERMI_FB_L2_ILW_EV_FLAGS_CLEAN)
    {
        dprintf("lw: Exelwting the CLEAN stage with data=0x%08x.\n", clealwalue);

        // L2 clean evict
        GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_CMGMT_1, clealwalue);

        // poll on CLEAN_NOT_PENDING for each fbps
        for (ltcCounter = 0; ltcCounter < numActiveLTCs; ltcCounter++)
        {
            // get offset for this fbps
            ltcOffset = LW_LTC_PRI_STRIDE * ltcCounter;

            done = FALSE;
            poll = 0;
            while (! done)
            {
                cmgmt1 = GPU_REG_RD32(LW_PLTCG_LTC0_LTSS_TSTG_CMGMT_1 + ltcOffset);

                if (DRF_VAL(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN, cmgmt1) ==
                    DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_1, _CLEAN, _NOT_PENDING))
                {
                    done = TRUE;
                    break;
                }

                if (poll++ == maxPolls)
                {
                    dprintf("lw: Clean timed out for %d msec\n", maxPolls*(DELAY/1000));
                    return LW_ERR_GENERIC;
                }
                osPerfDelay(DELAY);
            }
        }
    }

    dprintf("lw: Exelwting the ILWALIDATE stage with data=0x%08x.\n", evictValue);

    // L2 evict ilwalidate
    GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_CMGMT_0, evictValue);

    // poll on ILWALIDATE_NOT_PENDING for each fbps
    for (ltcCounter = 0; ltcCounter < numActiveLTCs; ltcCounter++)
    {
        // get offset for this fbps
        ltcOffset = LW_LTC_PRI_STRIDE * ltcCounter;

        done = FALSE;
        poll = 0;
        while (! done)
        {
            cmgmt0 = GPU_REG_RD32(LW_PLTCG_LTC0_LTSS_TSTG_CMGMT_0 + ltcOffset);

            if (DRF_VAL(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE, cmgmt0) == DRF_DEF(_PLTCG, _LTCS_LTSS_TSTG_CMGMT_0, _ILWALIDATE, _NOT_PENDING))
            {
                done = TRUE;
                break;
            }

            if (poll++ == maxPolls)
            {
                dprintf("lw: Ilwalidate timed out for %d msec\n", maxPolls*(DELAY/1000));
                return LW_ERR_GENERIC;
            }
            osPerfDelay(DELAY);
        }
    }
    return LW_OK;
}

//-----------------------------------------------------
// fbReadL2ColorZBCindex_GM107( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadL2ColorZBCindex_GM107(LwU32 index)
{
    LwU32 dataZBC[4] = {0, 0, 0, 0};
    LwU32 i;
    // Read and print the L2 Table data

    // First write the intrested index in the L2 register

    GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_DSTG_ZBC_INDEX, DRF_NUM(_PLTCG, _LTCS_LTSS_DSTG_ZBC_INDEX, _ADDRESS, index ));
    dprintf("lw: Color Information in L2 Table at Index  0x%x\n", index);

    for (i = 0; i < LW_PLTCG_LTCS_LTSS_DSTG_ZBC_COLOR_CLEAR_VALUE__SIZE_1; i++)
    {
        dataZBC[i] = DRF_VAL(_PLTCG, _LTCS_LTSS_DSTG_ZBC_COLOR_CLEAR_VALUE, _FIELD, GPU_REG_RD32(LW_PLTCG_LTCS_LTSS_DSTG_ZBC_COLOR_CLEAR_VALUE(i)));
        dprintf("lw: API Color Clear Value 0x%x is 0x%x\n", i, dataZBC[i]);
    }
    return LW_OK;
}

//-----------------------------------------------------
// fbReadL2DepthZBCindex_GM107( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadL2DepthZBCindex_GM107(LwU32 index)
{
    LwU32 dataZBC = 0;
    // Read and print the L2 Table data

    // First write the intrested index in the L2 register

    GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_DSTG_ZBC_INDEX, DRF_NUM(_PLTCG, _LTCS_LTSS_DSTG_ZBC_INDEX, _ADDRESS, index ));
    dprintf("lw: Depth Information in L2 Table at Index  0x%x\n", index);
    dataZBC = DRF_VAL(_PLTCG, _LTCS_LTSS_DSTG_ZBC_DEPTH_CLEAR_VALUE, _FIELD, GPU_REG_RD32(LW_PLTCG_LTCS_LTSS_DSTG_ZBC_DEPTH_CLEAR_VALUE));
    dprintf("lw: API Depth Clear Value is 0x%x\n", dataZBC);

    return LW_OK;
}

/*!
 * @brief Gets the lines per LTS count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The lines per LTS count.
 */
LwU32
fbGetLinesPerLTSCountLwW_GM107()
{
    return LW_SCAL_LITTER_NUM_LTC_LTS_SETS * LW_SCAL_LITTER_NUM_LTC_LTS_WAYS;
}

LwU32
fbGetActiveLtcMaskforFbp_GM107(LwU32 fbpIdx)
{
    return LWBIT32(pFb[indexGpu].fbGetActiveLTCCountLwW() - 1);
}

LwU32
fbGetActiveLtsMaskForLTC_GM107(LwU32 ltcIdx)
{
    return LWBIT32(pFb[indexGpu].fbGetLTSPerLTCCountLwW()) - 1;
}