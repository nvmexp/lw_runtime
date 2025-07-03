
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgk104.c
//
//*****************************************************

//
// includes
//
#include "kepler/gk104/dev_ltc.h"
#include "kepler/gk104/dev_pri_ringmaster.h"
#include "kepler/gk104/dev_fbpa.h"
#include "kepler/gk104/hwproject.h"
#include "kepler/gk104/dev_fb.h"
#include "kepler/gk104/dev_bus.h"
#include "kepler/gk104/dev_ram.h"
#include "kepler/gk104/dev_graphics_nobundle.h"
#include "fb.h"
#include "sig.h"
#include "chip.h"
#include "vmem.h"
#include "priv.h"
#include "gpuanalyze.h"
#include "sig.h"
#include "g_fb_private.h"

#ifdef WIN32
#include <windows.h>
#endif

LW_STATUS fbL2State_GK104()
{
    LwU32   i, j, k, line;
    LwU32   data32, val;
    LwU32   numActiveLTCs = pFb[indexGpu].fbGetActiveLTCCountLwW();
    LwU32   numLTSPerLTC  = pFb[indexGpu].fbGetLTSPerLTCCountLwW();

    dprintf("lw: Reading out general status: \n");
    for (i=0; i<numActiveLTCs; i++)
    {
        for (j=0; j<numLTSPerLTC; j++)
        {
            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_G_STATUS + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw:  LW_PLTCG_LTC%d_LTS%d_G_STATUS : 0x%x\n", i, j, data32);
        }
    }

    dprintf("lw:  \n");
    dprintf("lw: Reading out IQ status registers: \n");
    for (i=0; i<numActiveLTCs; i++)
    {
        for (j=0; j<numLTSPerLTC; j++)
        {
            for (k=0; k<=1; k++) // number of statuses needing a read, no define for this
            {
                // writing 0..1
                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                data32 &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
                data32 |= DRF_NUM(_PLTCG_LTC0, _LTS0_IQ_CFG_0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j, data32);

                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: Written %d to _STATUS_SELECT\n", k);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_CFG_0 : 0x%x\n", i, j, data32);

                val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_XBAR_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_XBAR_STATUS_0 : 0x%x\n", i, j, val);
            }
        }
    }

    dprintf("lw:  \n");
    dprintf("lw: Reading out other _STATUS\n");
    for (i=0; i<numActiveLTCs; i++)
    {
        for (j=0; j<numLTSPerLTC; j++)
        {
            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_ROP_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_ROP_STATUS_0 : 0x%x\n", i, j, val);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_CBC_STATUS + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_CBC_STATUS : 0x%x\n", i, j, val);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_STATUS + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_TSTG_STATUS : 0x%x\n", i, j, val);
        }
    }

    dprintf("lw:  \n");
    dprintf("lw: Reading out other _TSTG_SRV_STATUS_0\n");
    for (i=0; i<numActiveLTCs; i++)
    {
        for (j=0; j<numLTSPerLTC; j++)
        {
            for (k=0; k<=3; k++) // number of statuses needing a read, no define for this
            {
                // writing 0 ... 3
                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                data32 &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
                data32 |= DRF_NUM(_PLTCG_LTC0, _LTS0_IQ_CFG_0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j, data32);

                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: Written %d to _STATUS_SELECT\n", k);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_CFG_0 : 0x%x\n", i, j, data32);

                val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_SRV_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_TSTG_SRV_STATUS_0 : 0x%x\n", i, j, val);
            }
        }
    }

    dprintf("lw:  \n");
    dprintf("lw: Reading out other _DSTG_STATUS_0\n");
    for (i=0; i<numActiveLTCs; i++)
    {
        for (j=0; j<numLTSPerLTC; j++)
        {
            for (k=0; k<=24; k++) // number of status registers, no define for this, written in manuals
            {
                // writing 0 ... 24
                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                data32 &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_DSTG_CFG0_STATUS_SELECT);
                data32 |= DRF_NUM(_PLTCG_LTC0, _LTS0_DSTG_CFG0, _STATUS_SELECT, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j, data32);

                data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_DSTG_CFG0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: Written %d to _STATUS_SELECT\n", k);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_DSTG_CFG0 : 0x%x\n", i, j, data32);

                val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_DSTG_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
                dprintf("lw: LW_PLTCG_LTC%d_LTS%d_DSTG_STATUS_0 : 0x%x\n", i, j, val);
            }
        }
    }

    dprintf("lw:  \n");
    dprintf("lw: Reading out _ROP_OUTRUN_STATUS_0\n");
    for (i=0; i<numActiveLTCs; i++)
    {
        val = GPU_REG_RD32(LW_PLTCG_LTC0_MISC_ROP_OUTRUN_STATUS_0 + LW_LTC_PRI_STRIDE*i);
        dprintf("lw: LW_PLTCG_LTC%d_MISC_ROP_OUTRUN_STATUS_0 : 0x%x\n", i, val);
    }


    dprintf("lw:  \n");
    dprintf("lw: Reading out cache tag stage\n");
    for (i=0; i<numActiveLTCs; i++)           //LTC
    {
        dprintf("lw: Partition %d\n", i);
        for (j=0; j<numLTSPerLTC; j++)        //LTS
        {
            dprintf("lw:     Slice %d\n", j);
            for (line=0; line<550; line++)       //cache line, no define for this, written in manuals
            {
                //setup address
                val = line & DRF_SHIFTMASK(LW_PLTCG_LTCS_LTSS_TSTG_CST_RDI_INDEX_ADDRESS);
                GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_CST_RDI_INDEX, val);
                dprintf("lw:         Line %d\n", line);
                dprintf("lw:             ");
                for (k=0; k<3; k++)
                {
                    data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_DATA(k) + i*LW_LTC_PRI_STRIDE + j*LW_LTS_PRI_STRIDE);
                    dprintf(" [%d]:0x%x", k, data32);
                }
                dprintf("\n");
            }
        }
    }
    return LW_OK;
}

#define VIDMEMBIT   0
#define SYSMEMBIT   1

LW_STATUS fbIsMemReq_GK104(LwU32 nFbp)
{
    LwU32 i, j, val, data32;
    char* access[2] = { "YES", "NO" };
    char* statusVid;
    char* statusSys;
    LwU32 numLTCSlices = pFb[indexGpu].fbGetLTSPerLTCCountLwW();

    dprintf("lw: Checking for pending mem requests \n");
    for (i=0; i<nFbp; i++)
    {
        dprintf("lw: \n");
        dprintf("lw: Partition %d\n", i);
        for (j=0; j<numLTCSlices; j++)
        {
            dprintf("lw: \n");
            dprintf("lw:    Slice %d\n", j);
            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_STATUS + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw:         _LTC%d_LTS%d_TSTG_STATUS: 0x%x\n", i, j, data32);

            if (!(data32 & DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_TSTG_STATUS_CACHELINES_PINNED)))
            {
                dprintf("lw:         _LTS0_TSTG_STATUS_CACHELINES_PINNED 0\n");
            }

            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            data32 &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
            data32 |= DRF_NUM(_PLTCG_LTC0, _LTS0_IQ_CFG_0, _STATUS_SELECT, 3);
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j, data32);

            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: Written 3 to _STATUS_SELECT\n");
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_CFG_0 : 0x%x\n", i, j, data32);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_SRV_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_TSTG_SRV_STATUS_0 : 0x%x\n", i, j, val);
            val = DRF_VAL(_PLTCG_LTC0_LTS0, _TSTG_SRV_STATUS_0, _STATE, val);

            //if bit is 0 ; we have pending access
            statusVid = (val & BIT(VIDMEMBIT))? access[1]: access[0];  //[1] for NO, [0] for YES
            statusSys = (val & BIT(SYSMEMBIT))? access[1]: access[0];
            dprintf("lw:         VID: %s\n", statusVid);
            dprintf("lw:         SYS: %s\n", statusSys);
        }
    }
    return LW_OK;
}


/*!
 *  Callwlates the total fb RAM
 *  Total = #fb partitions * RAM per fbpa
 *
 *  @return total fb memory in MB.
 */
LwU32 fbGetMemSizeMb_GK104( void )
{
    LwU32   data32 = GPU_REG_RD_DRF(_PFB_FBPA, _CSTATUS, _RAMAMOUNT);
    LwU32   fbps = GPU_REG_RD_DRF(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_FBP, _COUNT);
    return (fbps * data32);
}


LwU32 fbGetActiveFbpCount_GK104( void )
{
    return GPU_REG_RD_DRF(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_FBP, _COUNT);
}

/*!
 * @brief Gets the LTS per LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The LTS per LTC count.
 */
LwU32
fbGetLTSPerLTCCountLwW_GK104( void )
{
    return LW_SCAL_LITTER_NUM_LTC_SLICES;
}


#if LINKSASS
#pragma message( "Enabling SASS support in gvdiss" )

// Windows version
#if defined(_AMD_64_)
#define _M_X64  1
#endif

#include "sass3lib.h"

LW_STATUS fbDisassembleVirtual_GK104(LwU32 chId, LwU64 va, LwU32 length, LwU32 shaderType)
{
    LW_STATUS  status = LW_OK;
    LwU8* pBuffer = NULL;
    VMemSpace vMemSpace;
    VMEM_INPUT_TYPE Id;

    HMODULE hSassLib = 0;
    SASS3LIB_SETSASSSMVERSION           pSassSetVer = NULL;
    SASS3LIB_DISASSEMBLEUCODE           pSassDis = NULL;
    SASS3LIB_GETSASSDISASSEMBLY         pSassDisGet = NULL;
    SASS3LIB_FREESASSDISASSEMBLY        pSassDisFree = NULL;
    SASS3LIB_GETSASSERRSTRING           pSassDisGetErrString = NULL;
    SASS3LIB_GETSASSPARSEERRORMESSAGE   pSassDisGetErrMsg = NULL;

    hSassLib = LoadLibrary(SASS3LIB_NAME);
    if (hSassLib)
    {
        pSassSetVer          = (SASS3LIB_SETSASSSMVERSION)          GetProcAddress(hSassLib, SASS3LIB_SETSASSSMVERSION_IMPORT);
        pSassDis             = (SASS3LIB_DISASSEMBLEUCODE)          GetProcAddress(hSassLib, SASS3LIB_DISASSEMBLEUCODE_IMPORT);
        pSassDisGet          = (SASS3LIB_GETSASSDISASSEMBLY)        GetProcAddress(hSassLib, SASS3LIB_GETSASSDISASSEMBLY_IMPORT);
        pSassDisFree         = (SASS3LIB_FREESASSDISASSEMBLY)       GetProcAddress(hSassLib, SASS3LIB_FREESASSDISASSEMBLY_IMPORT);
        pSassDisGetErrString = (SASS3LIB_GETSASSERRSTRING)          GetProcAddress(hSassLib, SASS3LIB_GETSASSERRSTRING_IMPORT);
        pSassDisGetErrMsg    = (SASS3LIB_GETSASSPARSEERRORMESSAGE)  GetProcAddress(hSassLib, SASS3LIB_GETSASSPARSEERRORMESSAGE_IMPORT);
    }

    if (pSassSetVer == NULL || pSassDis == NULL || pSassDisGet == NULL || pSassDisFree == NULL || pSassDisGetErrString == NULL || pSassDisGetErrMsg == NULL)
    {
        if (pSassSetVer == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_SETSASSSMVERSION_IMPORT);
        if (pSassDis == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_DISASSEMBLEUCODE_IMPORT);
        if (pSassDisGet == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_GETSASSDISASSEMBLY_IMPORT);
        if (pSassDisFree == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_FREESASSDISASSEMBLY_IMPORT);
        if (pSassDisGetErrString == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_GETSASSERRSTRING_IMPORT);
        if (pSassDisGetErrMsg == NULL)
            dprintf("lw: %s - Couldn't locate %s, aborting\n", __FUNCTION__, SASS3LIB_GETSASSPARSEERRORMESSAGE_IMPORT);
        return LW_ERR_GENERIC;
    }

    pBuffer = malloc(length);
    if (!pBuffer)
    {
        dprintf("lw: %s - failed to allocate 0x%08x bytes for ucode output\n", __FUNCTION__, length);
        return LW_ERR_GENERIC;
    }

    memset(&Id, 0, sizeof(Id));
    Id.ch.chId = chId;

    if (vmemGet(&vMemSpace, VMEM_TYPE_CHANNEL, &Id) != LW_OK)
    {
        dprintf("lw: %s: Could not get a VMEM Space for ChId 0x%x.\n",
                __FUNCTION__, chId);
        status = LW_ERR_GENERIC;
    }

    if (status == LW_OK)
    {
        status = pVmem[indexGpu].vmemRead(&vMemSpace, va, length, pBuffer);
    }

    // disassemble uCode
    if (status == LW_OK)
    {
        LwU32 ret = SASS3LIB_ERR_OK;

        // Setting SM version for sass3lib.dll
        // reference:
        // https://wiki.lwpu.com/engwiki/index.php/Sass3lib.dll#SASS3LIB_API_LwU32___cdecl_sass3lib_setSassSMVersion.28LwU32_sm_major_version.2C_LwU32_sm_minor_version.29.3B
        //
        // default to use Fermi
        (*pSassSetVer)(NULL, 2, 3);

        ret = (*pSassDis)(NULL, length, pBuffer);
        if (ret != SASS3LIB_ERR_OK)
        {
            status = LW_ERR_GENERIC;
        }

        // Get the disassembled string
        if (ret == SASS3LIB_ERR_OK)
        {
            char **ppProg = (*pSassDisGet)(NULL);
            LwU32 i ;

            if (ppProg != NULL)
            {
                for (i = 0; ppProg[i]; i++)
                {
                    dprintf("%s\n", ppProg[i]);
                }
            }
            else
            {
                ret    = SASS3LIB_ERR_OUT_OF_MEMORY;
                status = LW_ERR_GENERIC;
            }
        }

        // dprintf the string
        if (ret != SASS3LIB_ERR_OK)
        {
            const char *parserErr = (*pSassDisGetErrString)(NULL);
            dprintf("lw: %s: FERMISASS returns error %d - %s\n", __FUNCTION__, ret, (*pSassDisGetErrMsg)(NULL));
            if (parserErr[0])
            {
                dprintf("lw: FERMISASS parser error: %s\n", parserErr);
            }
        }

        // free any sasslib resource
        (*pSassDisFree)(NULL);
    }

    free(pBuffer);
    return status;
}

#else

LW_STATUS fbDisassembleVirtual_GK104(LwU32 chId, LwU64 va, LwU32 length, LwU32 shaderType)
{
    dprintf("lw: %s - lwwatch build with LINKSASS=0.  Extension not supported\n", __FUNCTION__);

    return LW_OK;
}

#endif

#define STARTOF_ZBC_CLEAR_TABLE    1
#define SIZEOF_ZBC_CLEAR_TABLE   16

//-----------------------------------------------------
// fbReadDSColorZBCindex_GK104( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadDSColorZBCindex_GK104(LwU32 index)
{
    LwU32 dataZBC[4] = {0, 0, 0, 0};
    LwU32 format = 0;
    // Read and print the DS Table data

    // First write the intrested index in the DS register

    GPU_REG_WR32(LW_PGRAPH_PRI_DS_ZBC_TBL_INDEX, DRF_NUM(_PGRAPH, _PRI_DS_ZBC_TBL_INDEX, _VAL, index ));
    dprintf("lw: Color Information in DS Table at Index  0x%x\n", index);


    // Trigger the Read operation
    GPU_REG_WR32(LW_PGRAPH_PRI_DS_ZBC_TBL_LD,
                       (DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _SELECT, LW_PGRAPH_PRI_DS_ZBC_TBL_LD_SELECT_C )     |
                        DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _ACTION, LW_PGRAPH_PRI_DS_ZBC_TBL_LD_ACTION_READ ) |
                        DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _TRIGGER, 1 )) );

    dataZBC[0] = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_COLOR_R, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_COLOR_R));
    dataZBC[1] = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_COLOR_G, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_COLOR_G));
    dataZBC[2] = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_COLOR_B, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_COLOR_B));
    dataZBC[3] = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_COLOR_A, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_COLOR_A));
    format     = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_COLOR_FMT, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_COLOR_FMT));

     dprintf("lw: R Color Value 0x%x\n", dataZBC[0]);
     dprintf("lw: G Color Value 0x%x\n", dataZBC[1]);
     dprintf("lw: B Color Value 0x%x\n", dataZBC[2]);
     dprintf("lw: A Color Value 0x%x\n", dataZBC[3]);
     dprintf("lw: FB Color Format Value is 0x%x\n", format);
    return LW_OK;
}

//-----------------------------------------------------
// fbReadDSDepthZBCindex_GK104( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadDSDepthZBCindex_GK104(LwU32 index)
{
    LwU32 dataZBC = 0;
    LwU32 format  = 0;
    // Read and print the DS Table data

    // First write the intrested index in the DS register

    GPU_REG_WR32(LW_PGRAPH_PRI_DS_ZBC_TBL_INDEX, DRF_NUM(_PGRAPH, _PRI_DS_ZBC_TBL_INDEX, _VAL, index ));
    dprintf("lw: Depth Information in DS Table at Index  0x%x\n", index);

    // Trigger the Read operation
    GPU_REG_WR32(LW_PGRAPH_PRI_DS_ZBC_TBL_LD,
                       (DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _SELECT, LW_PGRAPH_PRI_DS_ZBC_TBL_LD_SELECT_Z )    |
                        DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _ACTION, LW_PGRAPH_PRI_DS_ZBC_TBL_LD_ACTION_READ ) |
                        DRF_NUM(_PGRAPH,_PRI_DS_ZBC_TBL_LD, _TRIGGER, 1 )) );

    dataZBC = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_Z, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_Z));
    format  = DRF_VAL(_PGRAPH, _PRI_DS_ZBC_Z_FMT, _VAL, GPU_REG_RD32(LW_PGRAPH_PRI_DS_ZBC_Z_FMT));

     dprintf("lw: Depth Value 0x%x\n", dataZBC);
     dprintf("lw: FB Depth Format Value is 0x%x\n", format);
    return LW_OK;
}

//-----------------------------------------------------
// fbReadL2ColorZBCindex_GK104( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadL2ColorZBCindex_GK104(LwU32 index)
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
// fbReadL2ColorZBCindex_GK104( LwU32 index )
//
//-----------------------------------------------------
LW_STATUS fbReadL2DepthZBCindex_GK104(LwU32 index)
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

//-----------------------------------------------------
// fbReadDSColorZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadDSColorZBCtable_GK104(void)
{
    LwU32 index;
    // Read and print the DS Table data for all indices.

    for (index =1; index < SIZEOF_ZBC_CLEAR_TABLE; index++)
    {
        pFb[indexGpu].fbReadDSColorZBCindex(index);
    }
    return LW_OK;
}

//-----------------------------------------------------
// fbReadDSDepthZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadDSDepthZBCtable_GK104(void)
{
    LwU32 index;
    // Read and print the DS Table data for all indices.

    for (index =1; index < SIZEOF_ZBC_CLEAR_TABLE; index++)
    {
        pFb[indexGpu].fbReadDSDepthZBCindex(index);
    }
    return LW_OK;
}

//-----------------------------------------------------
// fbReadL2ColorZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadL2ColorZBCtable_GK104(void)
{
    LwU32 index;
    // Read and print the DS Table data for all indices.

    for (index =1; index < SIZEOF_ZBC_CLEAR_TABLE; index++)
    {
        pFb[indexGpu].fbReadL2ColorZBCindex(index);
    }
    return LW_OK;
}

//-----------------------------------------------------
// fbReadDSDepthZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadL2DepthZBCtable_GK104(void)
{
    LwU32 index;
    // Read and print the DS Table data for all indices.

    for (index =1; index < SIZEOF_ZBC_CLEAR_TABLE; index++)
    {
        pFb[indexGpu].fbReadL2DepthZBCindex(index);
    }
    return LW_OK;
}


// EXPOSED ROUTINES/WRAPPERS TO THE USER
//
//-----------------------------------------------------
// fbReadColorZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadColorZBCtable_GK104(void)
{
    fbReadDSColorZBCtable_GK104();
    fbReadL2ColorZBCtable_GK104();
    return LW_OK;
}

//-----------------------------------------------------
// fbReadDepthZBCtable_GK104( )
//
//-----------------------------------------------------
LW_STATUS fbReadDepthZBCtable_GK104(void)
{
    fbReadDSDepthZBCtable_GK104();
    fbReadL2DepthZBCtable_GK104();
    return LW_OK;
}

//-----------------------------------------------------
// fbReadColorZBCindex_GK104(LwU32 index)
//
//-----------------------------------------------------
LW_STATUS fbReadColorZBCindex_GK104(LwU32 index)
{
    pFb[indexGpu].fbReadDSColorZBCindex(index);
    pFb[indexGpu].fbReadL2ColorZBCindex(index);
    return LW_OK;
}

//-----------------------------------------------------
// fbReadDepthZBCindex_GK104(LwU32 index)
//
//-----------------------------------------------------
LW_STATUS fbReadDepthZBCindex_GK104(LwU32 index)
{
    pFb[indexGpu].fbReadDSDepthZBCindex(index);
    pFb[indexGpu].fbReadL2DepthZBCindex(index);
    return LW_OK;
}

//-----------------------------------------------------
// fbReadZBC_GK104(LwU32 index)
//
//-----------------------------------------------------
LW_STATUS fbReadZBC_GK104(LwU32 index)
{
    if (index >= SIZEOF_ZBC_CLEAR_TABLE )
    {

        dprintf("lw: Invalid Index Supplied. Index should be from 1 to 15\n");
    }
    else if (index == 0)
    {
        // Dump the whole ZBC Color & Depth DS/L2 Table data if index supplied is 0.
        fbReadColorZBCtable_GK104();
        fbReadDepthZBCtable_GK104();
    }
    else
    {
        // Dump the whole ZBC Color & Depth DS/L2 Table data for given index.
        fbReadColorZBCindex_GK104(index);
        fbReadDepthZBCindex_GK104(index);
    }

    return LW_OK;
}

#define SOME_ADDR       0xC
#define FBPSTRIDE(n)    (n > 2 ? 0x100 : 0x200)
#define PA_PER_FBP(i)   (SOME_ADDR + i * FBPSTRIDE(i))
#define N_FB_READS      3

/*!
 *  Checks fb state
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS fbTest_GK104( void )
{
    LwU32 i,j;
    LwU32 prevValue = 0;
    LwU32 value = 0xdeadbeef;
    LwU32 testValue[N_FB_READS];
    LW_STATUS status = LW_OK;
    LW_STATUS retVal = LW_OK;
    LwU32 readStatus = TRUE;
    LwU32 addr = 0;
    LwU32 partitions = pFb[indexGpu].fbGetActiveFbpCount();

    // do R/W test on active partitions
    if (partitions == 0)
    {
        dprintf("lw: Error. Incorrect number of FB partitions\n");
        addUnitErr("\t Incorrect number of FB partitions: 0x%08x\n", partitions);
        return LW_ERR_GENERIC;
    }

    dprintf("lw: Partitions:0x%02x\n", partitions);

    dprintf("lw: Start bypassing L2 ...\n");
    pFb[indexGpu].fbL2BypassEnable(TRUE);

    for (i = 0; i < partitions; i++)
    {
        addr = PA_PER_FBP(i);
        dprintf("lw: Testing partition %u @ 0x%08x\n", i, addr);

        status = pFb[indexGpu].fbRead(addr, &prevValue, 4);

        if (status == LW_ERR_GENERIC)
        {
           retVal = LW_ERR_GENERIC;
           dprintf("lw: Partition %u FAILED read test\n", i);
           addUnitErr("\t Partition %u FAILED read test\n", i);
           continue;
        }

        //now do repeated reads at the same address

        pFb[indexGpu].fbWrite(addr, &value, 4);

        status = LW_OK;
        for (j=0;j<N_FB_READS;j++)
        {
            status &= pFb[indexGpu].fbRead(addr, &testValue[j], 4);
        }

        pFb[indexGpu].fbWrite(addr, &prevValue, 4);

        if (status == LW_ERR_GENERIC)
        {
           retVal = LW_ERR_GENERIC;
           dprintf("lw: Partition %u FAILED read test\n", i);
           addUnitErr("\t Partition %u FAILED read test\n", i);
           continue;
        }

        if (verboseLevel > 1)
        {
            dprintf("lw: Value expected: 0x%08x\n", value);
            dprintf("lw: Values read:    ");
            for (j=0;j<N_FB_READS;j++)
            {
                dprintf(" 0x%08x\t", testValue[j]);
            }
            dprintf("\n");
        }

        for (j=0;j<N_FB_READS;j++)
        {
            readStatus = readStatus && (testValue[j] == value);
        }

        if ((readStatus == TRUE) && (status == LW_OK))
        {
            dprintf("lw: Partition %u passed test\n", i);
            retVal = LW_OK;
        }
        else
        {
            dprintf("lw: Partition %u FAILED write test\n", i);
            addUnitErr("\t Partition %u FAILED read test\n", i);
            retVal = LW_ERR_GENERIC;
        }
    }

    dprintf("lw: Stop bypassing L2 ...\n");
    pFb[indexGpu].fbL2BypassEnable(FALSE);

    return retVal;
}


/*!
 *  Checks TLB for stuck up ilwalidates
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS fbTestTLB_GK104( void )
{
    LW_STATUS status = LW_OK;
    LwU32 ctrlData = GPU_REG_RD32(LW_PFB_PRI_MMU_CTRL);
    LwU32 errorBit = DRF_VAL(_PFB, _PRI_MMU_CTRL, _PRI_FIFO_ERROR, ctrlData);
    LwU32 fifoSpace = DRF_VAL(_PFB, _PRI_MMU_CTRL, _PRI_FIFO_SPACE, ctrlData);

    dprintf("lw: available PRI input fifo space : %d.\n", fifoSpace);

    if (errorBit == LW_PFB_PRI_MMU_CTRL_PRI_FIFO_ERROR_TRUE)
    {
        dprintf("lw: Error : ILWALIDATE request to PRI input fifo was written when it was full.\n");
        addUnitErr("\t LW_PFB_PRI_MMU_CTRL showed _PRI_FIFO_ERROR_TRUE\n");
        status = LW_ERR_GENERIC;
    }
    return status;
}

/*!
 *  Checks sysmem state
 *
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS fbTestSysmem_GK104()
{
    //search for a non ilwasive and quick way to check sysmem accesses
    //returning OK for now

    dprintf("lw: Skipping sysmem test for now \n");
    return LW_ERR_NOT_SUPPORTED;
}

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

/*!
 *  Force fb L2 ilwalidate Timeout is in msec.
 *
 */
LW_STATUS fbL2IlwalEvict_GK104(LwU32 scaleTime)
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

setup_writes_t * fbGetPMEnableWrites_GK104( void )
{
    static setup_writes_t PmEnableWrites_GK104[] =
    {
        { "LW_PFB_FBPA_PM_ENABLE"  , LW_PFB_FBPA_PM, 0x00000001, 0x00000001},
        { "LW_PFB_FBPA_PM_SELECT"  , LW_PFB_FBPA_PM, 0x00000110, 0x000003f0},
        { "LW_PFB_FBPA_PM_ENABLE"  , LW_PFB_FBPA_PM, 0x00000001, 0x00000001},
        { "LW_PFB_FBPA_PM_SELECT"  , LW_PFB_FBPA_PM, 0x00000110, 0x000003f0},
        {NULL}
    };

    return(PmEnableWrites_GK104);
}

static setup_writes_t FbSetupWrites[] =
{
    { "TRIGGERCNT"  , 0x001a008c, 0x00000000, 0xffffffff},
    { "SAMPLECNT"  , 0x001a0090, 0x00000000, 0xffffffff},
    { "THRESHOLD"  , 0x001a0098, 0x00000000, 0xffffffff},
    { "CONTROL"  , 0x001a009c, 0x00000001, 0xffffffff},
    { "ENGINE_SEL"  , 0x001a006c, 0x0000006d, 0xffffffff},
    { "TRIG1_SEL"  , 0x001a0048, 0x0000006f, 0xffffffff},
    { "TRIG1_OP"  , 0x001a004c, 0x0000ffff, 0xffffffff},
    { "SAMPLE_SEL"  , 0x001a0058, 0x0000006f, 0xffffffff},
    { "SAMPLE_OP"  , 0x001a005c, 0x00000000, 0xffffffff},
    { "CLRFLAG_OP"  , 0x001a0064, 0x00000000, 0xffffffff},
    { "SETFLAG_OP"  , 0x001a0064, 0x00000000, 0xffffffff},
    { "EVENT_SEL"  , 0x001a0050, 0x0000100f, 0xffffffff},
    { "EVENT_OP"  , 0x001a0054, 0x0000eeee, 0xffffffff},
    { "TRIG0_SEL"  , 0x001a0040, 0x0000006f, 0xffffffff},
    { "TRIG0_OP"  , 0x001a0044, 0x0000ffff, 0xffffffff},
    { "CLAMP_CYA_CONTROL"  , 0x001a0100, 0x00000000, 0xffffffff},
    { "STARTEXPERIMENT"  , 0x001a00e0, 0x00000001, 0xffffffff},
    {NULL}
};

#define FBP_STRIDE 0x1000
#define REG_FBP(reg,n) (reg + n*FBP_STRIDE)

#ifndef LW_PERF_PMMFBP_CONTROL
#define LW_PERF_PMMFBP_CONTROL                  0x001a009c
#define LW_PERF_PMMFBP_CONTROL_STATE            29:28
#define LW_PERF_PMMFBP_CONTROL_STATE_CAPTURE    0x03
#define LW_PERF_PMMFBP_CONTROL_STATE_IDLE       0x00
#endif //LW_PERF_PMMFBP_CONTROL

static BOOL checkCounterSetup(LwU32 nFbp)
{
    LwU32 i;
    LwU32 val;
    BOOL status = TRUE;
    setup_writes_t* pPmWrites = pFb[indexGpu].fbGetPMEnableWrites();

    dprintf("lw: Checking counter setup\n");

    for (i=0; i<nFbp; i++)
    {
        val = GPU_REG_RD32(REG_FBP(LW_PERF_PMMFBP_CONTROL,i));
        if (DRF_VAL(_PERF, _PMMFBP_CONTROL, _STATE, val) == LW_PERF_PMMFBP_CONTROL_STATE_CAPTURE)
        {
            dprintf("lw: FBP %d CAPTURE\n", i);
            status &= TRUE;
            continue;
        }

        if (DRF_VAL(_PERF, _PMMFBP_CONTROL, _STATE, val) == LW_PERF_PMMFBP_CONTROL_STATE_IDLE)
        {
            dprintf("lw: ERROR: FBP %d IDLE\n", i);
            status &= FALSE;
        }
    }

    if (pPmWrites != NULL)
    {
        dprintf("lw: Checking _PFB_FBPA_PM_ENABLE \n");

        // checking _ENABLE
        val = RegBitRead(pPmWrites[0].addr, 0);
        if (val == pPmWrites[0].value)
        {
            dprintf("lw: FBP_PFB_FBPA_PM_ENABLE 0x%x\n", val);
            status &= TRUE;
        }
        else
        {
            dprintf("lw: ERROR: FBP not _PFB_FBPA_PM_ENABLE 0x%x\n", val);
            status &= FALSE;
        }
    }

    return status;
}

static LW_STATUS setupCounter(LwU32 nFbp)
{
    LwU32 i;
    setup_writes_t* pWrites;
    setup_writes_t* pPmWrites;
    LwU32 reg;

    dprintf("lw: Setting up .. \n");
    dprintf("lw: Reg           Addr         Value         Mask .. \n");

    pPmWrites = pFb[indexGpu].fbGetPMEnableWrites();
    if (pPmWrites != NULL)
    {
        dprintf("lw: Setting up broadcast writes\n");
        for (; pPmWrites->name != NULL; pPmWrites++)
        {
            dprintf("lw: %s 0x%x 0x%x 0x%x\n", pPmWrites->name,
                    pPmWrites->addr, pPmWrites->value, pPmWrites->mask);
            RegWrite(pPmWrites->addr, pPmWrites->value, pPmWrites->mask);
        }
    }

    for (i=0; i<nFbp; i++)
    {
        dprintf("lw: Setting up unicasts FBP%d\n", i);
        for (pWrites = FbSetupWrites; pWrites->name != NULL; pWrites++)
        {
            reg = REG_FBP(pWrites->addr,i);
            dprintf("lw: %s 0x%x 0x%x 0x%x\n", pWrites->name,
                    reg, pWrites->value, pWrites->mask);
            RegWrite(reg, pWrites->value, pWrites->mask);
        }
    }

    return LW_OK;
}

static LwU32 getAccessCount(LwU32 nFbp)
{
    LwU32 i;
    LwU32 count = 0;
    LwU32 gCount = 0;
    //fb_read_or_write_subp0_fbp0.event_count 0x001a0080
    LwU32 reg = 0x001a0080;

    dprintf("lw: getting access counts\n");
    for (i=0; i<nFbp; i++)
    {
        count = GPU_REG_RD32(REG_FBP(reg,i));
        dprintf("lw:  FBP%d    Count: 0x%x\n", i, count);
        gCount += count;
    }

    return gCount;
}

LW_STATUS fbMonitorAccess_GK104(LwU32 nFbp, BOOL bReadback)
{
    BOOL bSetupDone = FALSE;
    LwU32 counter = 0;
    LW_STATUS status = LW_OK;

    if (bReadback)
    {
        bSetupDone = checkCounterSetup(nFbp);
        if (bSetupDone)
        {
            counter = getAccessCount(nFbp);
        }
        else
        {
            dprintf("lw: ERROR setup the experiment first\n");
            dprintf("lw: dumping counters anyway\n");
            counter = getAccessCount(nFbp);
            return LW_ERR_GENERIC;
        }
    }
    else
    {
        status = setupCounter(nFbp);
        if (status == LW_ERR_GENERIC)
        {
            dprintf("lw: Counter setup could not be done\n");
            return status;
        }
    }

    return status;
}

void fbGetEccInfo_GK104(BOOL bFullPrint)
{
    BOOL bFbpaEnabled = GPU_REG_RD_DRF(_PFB, _FBPA_ECC_CONTROL, _MASTER_EN);
    BOOL bLtcEnabled = GPU_REG_RD_DRF(_PLTCG, _LTCS_LTSS_DSTG_CFG0, _ECC);
    BOOL bL1cEnabled = GPU_REG_RD_DRF(_PGRAPH, _PRI_GPCS_TPCS_L1C_ECC_CSR, _ENABLE);
    BOOL bSmEnabled = GPU_REG_RD_DRF(_PGRAPH, _PRI_GPCS_TPCS_SM_LRF_ECC_CONTROL, _MASTER_EN);

    dprintf("lw: ECC summary:\n");
    dprintf("lw:  + FBPA enabled:       %d\n", bFbpaEnabled);
    dprintf("lw:  + LTC enabled:        %d\n", bLtcEnabled);
    dprintf("lw:  + L1C enabled:        %d\n", bL1cEnabled);
    dprintf("lw:  + SM enabled:         %d\n", bSmEnabled);

    if ( bFullPrint )
    {
        dprintf("\nlw: ECC details:\n");
        
        dprintf("lw: LW_PFB_FBPA_0_CAP0:\n");
        priv_dump("LW_PFB_FBPA_0_CAP0");

        dprintf("lw: LW_PFB_FBPA_ECC_CONTROL:\n");
        priv_dump("LW_PFB_FBPA_ECC_CONTROL");

        dprintf("lw: LW_PFB_FBPA_0_ECC_SEC_COUNT:\n");
        priv_dump("LW_PFB_FBPA_0_ECC_SEC_COUNT");

        dprintf("lw: LW_PFB_FBPA_0_ECC_DED_COUNT:\n");
        priv_dump("LW_PFB_FBPA_0_ECC_DED_COUNT");

        dprintf("lw: LW_PLTCG_LTCS_LTSS_DSTG_CFG0:\n");
        priv_dump("LW_PLTCG_LTCS_LTSS_DSTG_CFG0");

        dprintf("lw: LW_PGRAPH_PRI_GPCS_TPCS_L1C_ECC_CSR:\n");
        priv_dump("LW_PGRAPH_PRI_GPCS_TPCS_L1C_ECC_CSR");

        dprintf("lw: LW_PGRAPH_PRI_GPCS_TPCS_SM_LRF_ECC_CONTROL:\n");
        priv_dump("LW_PGRAPH_PRI_GPCS_TPCS_SM_LRF_ECC_CONTROL");
    }
}


void fbL2BypassEnable_GK104(BOOL bEnable)
{
    LwU32 regVal = GPU_REG_RD32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2);

    if(bEnable)
    {
        regVal = FLD_SET_DRF(_PLTCG, _LTCS_LTSS_TSTG_SET_MGMT_2, _L2_BYPASS_MODE, _ENABLED, regVal);

        GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2, regVal);
    }
    else
    {
        regVal = FLD_SET_DRF(_PLTCG, _LTCS_LTSS_TSTG_SET_MGMT_2, _L2_BYPASS_MODE, _DISABLED, regVal);

        GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2, regVal);
    }
}

/*!
 * @brief Gets the active LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The active LTC count.
 */
LwU32 fbGetActiveLTCCountLwW_GK104( void )
{
    // There is a 1:1 correspondence of LTCs and FBPs.
    return pFb[indexGpu].fbGetActiveFbpCount();
}

LwU32 fbGetFBIOBroadcastDDRMode_GK104( void )
{
    return GPU_REG_RD_DRF(_PFB_FBPA, _FBIO_BROADCAST, _DDR_MODE);
}

LwU32 fbGetBAR0WindowRegAddress_GK104( void )
{
     return LW_PBUS_BAR0_WINDOW;
}

void fbSetBAR0WindowBase_GK104(LwU32 baseOffset)
{
    GPU_REG_WR32(LW_PBUS_BAR0_WINDOW,
                 DRF_NUM(_PBUS, _BAR0_WINDOW, _BASE, baseOffset) |
                 DRF_DEF(_PBUS, _BAR0_WINDOW, _TARGET, _VID_MEM));
}

/*!
 * @brief Read/write from/to FB memory.
 *
 * If is_write == 0, reads from FB memory at the given address for the given
 * number of bytes and stores it in the buffer.
 * If is_write != 0, writes to FB memory at the given address for the given
 * number of bytes with the contents of buffer.
 * Note that function doesn't align offset by DWORD.
 *
 * @param[in] offset        LwU64 address in FB memory to read/write.
 * @param[in, out] buffer   void * pointer to buffer,
 *                          function reads/writes into/from this buffer.
 * @param[in] length        LwU32 number of bytes to read/write.
 * @param[in] is_write      LwU32 0 for read, otherwise, write
 *
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
LW_STATUS fbReadWrite_GK104(LwU64 offset, void* buffer, LwU32 length, LwU32 is_write)
{
    LwU32  old_window;
    LwU64  lwrrent_window;
    LwU64  lwrrent_window_offset;
    LwU64  lwrrent_offset;
    char*  current;
    LwU32  bytes_left;
    LwU32  lwrrent_window_bytes_left;
    LwU32  bar0_offset;
    LwU32  bar0WindowRegAddr;
    LwU32  transfer_length;
    const  LwU32 block_size = 0x1000;
    const  LwU32 window_size = 0x100000;
    LW_STATUS  status = LW_OK;
    LwU32 i;

    if (buffer == NULL)
    {
        return LW_ERR_GENERIC;
    }

    if (lwMode == MODE_DUMP)
    {
        return fbReadWrite_DUMP(offset, buffer, length, is_write);
    }

    lwrrent_offset = offset;
    current = (char*)buffer;
    bytes_left = length;

    bar0WindowRegAddr = pFb[indexGpu].fbGetBAR0WindowRegAddress();
    // Store BAR0 window
    old_window = GPU_REG_RD32(bar0WindowRegAddr);

    // Loop through all windows (64kb alignment, 1MB window size)
    while (bytes_left > 0)
    {
        // Setup BAR0 window
        lwrrent_window = lwrrent_offset >> 16;
        lwrrent_window_offset = lwrrent_window << 16;
        
        pFb[indexGpu].fbSetBAR0WindowBase((LwU32)lwrrent_window);

        lwrrent_window_bytes_left = (LwU32) min(lwrrent_window_offset + window_size - lwrrent_offset, bytes_left);
        // Loop through writing up to a block_size at a time
        while (lwrrent_window_bytes_left > 0)
        {
            // Check for early exit
            if (osCheckControlC())
            {
                status = LW_ERR_GENERIC;
                break;
            }

            transfer_length = min(block_size, lwrrent_window_bytes_left);
            // Callwlate BAR0 offset
            bar0_offset = (LwU32) LW_PRAMIN_DATA008(lwrrent_offset - lwrrent_window_offset);
            if (is_write)
            {
                // Write FB memory through BAR0
                // Since VF can't access FB from BAR0, GPU_REG* go to plugin 
                for (i = 0; i < transfer_length/4; i++)
                {
                    GPU_REG_WR32(bar0_offset + (i * 4), *((LwU32 *)(current) + i));
                }
            }
            else
            {
                // Read FB memory through BAR0 
                // Since VF can't access FB from BAR0, GPU_REG* go to plugin 
                for (i = 0; i < transfer_length/4; i++)
                {
                    *((LwU32 *)(current) + i) = GPU_REG_RD32(bar0_offset + (i * 4));
                }
            }
            // Update tracking
            lwrrent_offset += transfer_length;
            current += transfer_length;
            lwrrent_window_bytes_left -= transfer_length;
            bytes_left -= transfer_length;
        }
    }

    // Restore BAR0 window
    GPU_REG_WR32(bar0WindowRegAddr, old_window);

    return status;
}

/*!
 * @brief Read from FB memory.
 *
 * Function reads from FB memory at the given address for the given
 * number of bytes and stores it in the buffer.
 * Note that function doesn't align offset by DWORD.
 *
 * @param[in] offset        LwU64 address in FB memory to read.
 * @param[out] buffer       void * pointer to buffer,
 *                          function reads into this buffer.
 * @param[in] length        LwU32 number of bytes to read.
 *
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
LW_STATUS fbRead_GK104(LwU64 offset, void* buffer, LwU32 length)
{
    return fbReadWrite_GK104(offset, buffer, length, 0);
}

/*!
 * @brief Write to FB memory.
 *
 * Function writes to FB memory at the given address for the given
 * number of bytes with the contents of buffer.
 * Note that function doesn't align offset by DWORD.
 *
 * @param[in] offset        LwU64 address in FB memory to write.
 * @param[in] buffer        void * pointer to buffer,
 *                          function writes from this buffer.
 * @param[in] length        LwU32 number of bytes to write.
 *
 * @return LW_OK on success, LW_ERR_GENERIC on failure.
 */
LW_STATUS fbWrite_GK104(LwU64 offset, void* buffer, LwU32 length)
{
    return fbReadWrite_GK104(offset, buffer, length, 1);
}
