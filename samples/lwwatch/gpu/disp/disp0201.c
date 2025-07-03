/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// disp0201.c - Disp V02_01 display routines 
// 
//*****************************************************

#include <stdio.h>
#include <string.h>
#include <ctype.h>

#include "inc/disp.h"
#include "inc/chip.h"
#include "inc/fb.h"
#include "dpaux.h"
#include "clk.h"
#include "print.h"
#include "methodParse.h"
#include "class_mthd/mthd_type.h"
#include "disp/v02_01/dev_disp.h"
#include "disp/v02_01/disp0201.h"
#include "kepler/gk104/dev_trim.h"
#include "kepler/gk104/dev_pmgr.h"

#include "class/cl917c.h"
#include "class/cl917d.h"
#include "class/cl917e.h"
#include "class/cl917C_variables.h"
#include "class/cl917D_variables.h"
#include "class/cl917E_variables.h"

#include "g_disp_private.h"     // (rmconfig)  implementation prototypes
#include "g_dpaux_private.h"
#include "g_fb_private.h"

extern LwU32 classHeaderNum[CHNTYPE_OVLY + 1];

static AUXPORT link2AuxPort[PADLINK_MAX] =
{
    AUXPORT_NONE,
    AUXPORT_NONE,
    AUXPORT_0,
    AUXPORT_1,
    AUXPORT_2,
    AUXPORT_3,
    AUXPORT_NONE
};

/*!
 * @brief Returns aux port by specified link.
 *
 * @param[in]  index        specified Link.
 *
 * @returns  aux port
 */
LwU32 dispGetAuxPortByLink_v02_01(LwU32 index)
{
    if (index < PADLINK_MAX)
        return link2AuxPort[index];
    else
        return AUXPORT_NONE;
}

LwU32
dispGetNumSfs_v02_01(void)
{
    return LW_PDISP_CLK_REM_SF__SIZE_1;
}

LwU32
dispGetNumAuxPorts_v02_01(void)
{
    return LW_PMGR_DP_AUXCTL__SIZE_1;
}

// Search through the file at scPath to find all the classes, output to dest, seperated by separater
void dispGetAllClasses_v02_01(char *dest, char separator, char *scPath)
{
    char name[16];
    char line[512];
    // keep place for line
    char *linePlace;
    FILE *scFile;
    size_t strSize;
    
    scFile = fopen(scPath, "r");
    if (!scFile)
    {
        dprintf("File specified by %s does not exist!!!\n", scPath);
        return;
    }
    while (fgets(line, 512, scFile) != NULL) 
    {
        linePlace = line;
        if (strstr(linePlace, "_SC_") != NULL)
        {
            // jump over "#define LW_PDISP_"
            strSize = strlen("#define LW_PDISP_");
            linePlace = linePlace + strSize; 
            memset(name, 0, sizeof(name));
            strncpy(name, linePlace, 4);
            if (strstr(dest, name) == NULL)
            {
                sprintf(dest, "%sLW%s%c", dest, name, separator);
            }
        }
    }
    fclose(scFile);
}

void
initializeDisp_v02_01(char *chipName)
{
    char *tmpScPath;
    char classNames[256];
    char scPath[256];
    char dispManualPath[256];

    memset(scPath, 0, sizeof(scPath));
    tmpScPath = getelw("LWW_MANUAL_SDK");
    if (tmpScPath == NULL)
    {
        dprintf("lw: Please set your LWW_MANUAL_SDK environment variable to point to your "
                INC_DIR_EXAMPLE " directory\n");
        return;
    }

    strcpy(scPath, tmpScPath);

    strcat(scPath, DIR_SLASH);


    if(!GetDispManualsDir(dispManualPath))
    {
        dprintf("lw:%s(): Failed to initialise for current chip",
            __FUNCTION__);
        return;
    }

    strcat(scPath, dispManualPath);

    strcat(scPath, DIR_SLASH "dev_disp.h");

    memset(classNames, 0, sizeof(classNames));
    dispGetAllClasses_v02_01((char *)(classNames), ';', scPath);

    initializeClassHeaderNum(classNames, classHeaderNum);
}

// Returns Channel State Descriptor
static int
dispGetChanDesc_v02_01
(
    char           *name,
    LwU32           headNum,
    ChanDesc_t    **dchnst
)
{
    LwU32 i;
    LwU32 chanNum = 0;
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    // Default is core
    if (!name)
    {       
        name = "core";     
        chanNum = 0;
        headNum = 0;
    }
    else if (!strcmp(name, "core"))
    {
        chanNum = 0;
        headNum = 0;
    }
    else
    {
        for (i = 0; i < numDispChannels; i++)
        {
            if (!strcmp(dispChanState_v02_01[i].name, name) && 
                (headNum == dispChanState_v02_01[i].headNum))
            {
                chanNum = i;
                break;
            }
        }

        if (i == numDispChannels)
        {
            return -1 ;
        }
    }

    *dchnst = &dispChanState_v02_01[chanNum];    
    return chanNum;
}

LwU32
dispGetMaxChan_v02_01(void)
{
    return LW_PDISP_CHANNELS;
}

//
// Prints channel state.
//
void
dispPrintChanState_v02_01
(
    LwU32 chanNum
)
{
    LwU32 pending = 0;
    ChanDesc_t *chnst;
    LwU32 chnctl, val;

    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return;

    chnst = &dispChanState_v02_01[chanNum];

    chnctl = val = GPU_REG_RD32(chnst->base);
    val = GET_BITS(val,chnst->highbit, chnst->lowbit);
    if (val > chnst->numstate)
    {
        dprintf("invalid state value %x\n", val);
        return;
    }

    dprintf("%2d \t%s\t%2d  ", chanNum, chnst->name, chnst->headNum);
    dprintf("\t%13s", (chnst->cap & DISP_STATE) ?  DCHN_GET_DESC_V02_01(chanNum,val): "N/A"); 

    if (chnst->cap & DISP_SPVSR)
    {
        int i, numpend = 0, idx = 0;

        val = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_DISPATCH);

        if (FLD_TEST_DRF(_PDISP, _DSI_RM_INTR_DISPATCH, _SUPERVISOR_VBIOS, _PENDING, val))
        {
            val = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_SV);
            for (i = 0; i < LW_PDISP_DSI_RM_INTR_SV_SUPERVISOR__SIZE_1; i++)
            {
                if (FLD_IDX_TEST_DRF(_PDISP, _DSI_RM_INTR_SV, _SUPERVISOR, i,  _PENDING, val)) {
                   idx = i + 1; 
                   numpend += 1; 
                }
            }
        }
        if (numpend ==  1)
        {
            dprintf("\t%3s#%d PENDING", "", idx); 
        }
        else if (numpend ==  0)
        {
            dprintf("\t%13s","NOT PENDING");
        }
        else
        {
            dprintf("\t %10s  ","ERROR!!!");
        }
    }
    else
    {
        dprintf("\t%9s   ","N/A");
    }

    if (chnst->cap & DISP_EXCPT)
    {
        val = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_DISPATCH);

        // Check if exception is pending in any disp channel.
        if (FLD_TEST_DRF(_PDISP, _DSI_RM_INTR_DISPATCH, _EXCEPTION, _PENDING, val))
        {
            val = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_EXCEPTION);
            //
            // Check if exception if pending in this channel
            // Likely we can just check this register directly but let's
            // keep the approach of walking the tree for now
            //
            if (FLD_IDX_TEST_DRF(_PDISP, _DSI_RM_INTR_EXCEPTION, _CHN, chanNum, _PENDING, val))
                pending = 1;
        }

        if (pending)
        {
            dprintf("\t%8s", "  PENDING  ");
        }
        else
        {
            dprintf("\t%12s", "TBD");
        }
    }
    else
    {
        dprintf("\t%5s  ","N/A");
    }
    dprintf("\t0x%08x\n", chnctl);
}


//
// Prints channel number
//
LwS32
dispGetChanNum_v02_01
(
    char   *chanName,
    LwU32   headNum
)
{
    ChanDesc_t *chnst;
    LwS32 chanNum;

    if ((chanNum = dispGetChanDesc_v02_01(chanName, headNum, &chnst)) == -1)
    {
        return -1;
    }
    else
    {
        return chanNum; 
    } 
}

// 
// Prints Channel Name
//
void
dispPrintChanName_v02_01
(
    LwU32 chanNum
)
{
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels) {
        dprintf("<channelNumber> should be in the range 0 .. %d\n", numDispChannels - 1 );
        return;
    }

    if (dispChanState_v02_01[chanNum].cap & DISP_STATE)
        dprintf("ChannelName : %s, Head : %d\n",
                dispChanState_v02_01[chanNum].name, dispChanState_v02_01[chanNum].headNum);
}

LwU32
dispGetNumOrs_v02_01(LWOR orType)
{
    switch (orType)
    {
        case LW_OR_SOR:
            return LW_PDISP_SORS;
        case LW_OR_PIOR:
            return LW_PDISP_PIORS;
        case LW_OR_DAC:
            return LW_PDISP_DACS;
        default:
            dprintf("Error");
            return 0;
    }
}

/*!
 *  Checks whether the given resource exists
 *
 *  @param[in]  orType      DAC/SOR/PIOR  
 *  @param[in]  index       resource index 
 *
 *  @return   BOOL TRUE/FALSE
 */
BOOL
dispResourceExists_v02_01(LWOR orType, LwU32 index)
{
    LwU32 orHwCap  = 0;
    LwU32 orCap    = GPU_REG_RD32(LW_PDISP_CLK_REM_SYS_CAP);

    if (index >= pDisp[indexGpu].dispGetNumOrs(orType))
    {
        dprintf("lw: %s Illegal OR Index: %d\n", __FUNCTION__, index);
        return FALSE;
    }

    switch (orType)
    {
        case LW_OR_DAC:
            orHwCap = GPU_REG_RD32(LW_PDISP_DAC_CAP(index));
            return (DRF_IDX_VAL(_PDISP, _CLK_REM_SYS_CAP, _DAC_EXISTS, index, orCap) == 
                     LW_PDISP_CLK_REM_SYS_CAP_DAC_EXISTS_YES);

        case LW_OR_SOR:
            orHwCap = GPU_REG_RD32(LW_PDISP_SOR_CAP(index));
            return (DRF_IDX_VAL(_PDISP, _CLK_REM_SYS_CAP, _SOR_EXISTS, index, orCap) ==
                     LW_PDISP_CLK_REM_SYS_CAP_SOR_EXISTS_YES);
            
        case LW_OR_PIOR:
            orHwCap = GPU_REG_RD32(LW_PDISP_PIOR_CAP(index));
            return (DRF_IDX_VAL(_PDISP, _CLK_REM_SYS_CAP, _PIOR_EXISTS, index, orCap) == 
                     LW_PDISP_CLK_REM_SYS_CAP_PIOR_EXISTS_YES);

        default:
            dprintf(" %s Invalid OR type : %d ", __FUNCTION__, orType);
            return FALSE;
    }
}

LwS32
dispGetChanDescriptor_v02_01(LwU32 chanNum, void **desc)
{
    ChanDesc_t **desc_t;
    LwU32 numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    desc_t = (ChanDesc_t **)desc;

    if (chanNum >= numDispChannels)
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return -1;
    }
    if (desc_t)
    {
        *desc_t = &dispChanState_v02_01[chanNum];
    }
    else
    {
        dprintf("ERROR: null pointer to descriptor\n");
        return -1;
    }
    return 0;
}

void
dispPrintScanoutOwner_v02_01(void)
{
    LwU32 scanoutOwnerArmed, scanoutOwnerActiv;
    LwU32 data32;
    LwU32 i;

    for (i = 0; i < pDisp[indexGpu].dispGetNumHeads(); ++i)
    {     
        data32 = GPU_REG_RD32(LW_PDISP_DSI_CORE_HEAD_STATE(i));
        scanoutOwnerArmed = DRF_VAL(_PDISP, _DSI_CORE_HEAD_STATE, _SHOWING_ARMED, data32);
        scanoutOwnerActiv = DRF_VAL(_PDISP, _DSI_CORE_HEAD_STATE, _SHOWING_ACTIVE, data32);      
        
        if (scanoutOwnerActiv == LW_PDISP_DSI_CORE_HEAD_STATE_SHOWING_ACTIVE_DRIVER)
        {
            dprintf("Scanout owner for head%d (ACTIV) = DRIVER\n", i);
        }
        else
        {
            dprintf("Scanout owner for head%d (ACTIV) = VBIOS\n", i);
        }


        if (scanoutOwnerArmed == LW_PDISP_DSI_CORE_HEAD_STATE_SHOWING_ARMED_DRIVER)
        {
            dprintf("Scanout owner for head%d (ARMED) = DRIVER\n", i);
        }
        else
        {
            dprintf("Scanout owner for head%d (ARMED) =  VBIOS\n", i);
        }
    }
}

/*!
 *  Get Channel type and Head num for the given channel num 
 *
 *
 *  @param[in]   chanNum     channel number  
 *  @param[out]  pHeadNum    head num for that channel 
 *
 *  @return   channel num. negative when illegal.
 */
LwS32 dispGetChanType_v02_01(LwU32 chanNum, LwU32* pHeadNum)
{
    if (chanNum >= pDisp[indexGpu].dispGetMaxChan())
        return -1;

    if (pHeadNum)
    {
        *pHeadNum = dispChanState_v02_01[chanNum].headNum;
    }

    return dispChanState_v02_01[chanNum].id;
}

void dispPrintExceptPending_v02_01(LwU32 headNum)
{
    LwU32   temp32 = 0;
    LwU32   chanNum;

    temp32 = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_DISPATCH);

    // Check if exception is pending in any disp channel.
    if (FLD_TEST_DRF(_PDISP, _DSI_RM_INTR_DISPATCH, _EXCEPTION, _NOT_PENDING, temp32))
        return;

    temp32 = GPU_REG_RD32(LW_PDISP_DSI_RM_INTR_EXCEPTION);

    chanNum = 0;
    if (FLD_IDX_TEST_DRF(_PDISP, _DSI_RM_INTR_EXCEPTION, _CHN, chanNum, _PENDING, temp32))
    {
        dprintf("Exception pending in %s[%d]\n", dispChanState_v02_01[chanNum].name, headNum);
    }

    chanNum = pDisp[indexGpu].dispGetChanNum("base", headNum);
    if (FLD_IDX_TEST_DRF(_PDISP, _DSI_RM_INTR_EXCEPTION, _CHN, chanNum, _PENDING, temp32))
    {
        dprintf("Exception pending in %s[%d]\n", dispChanState_v02_01[chanNum].name, headNum);
    }

    chanNum = pDisp[indexGpu].dispGetChanNum("ovly", headNum);
    if (FLD_IDX_TEST_DRF(_PDISP, _DSI_RM_INTR_EXCEPTION, _CHN, chanNum, _PENDING, temp32))
    {
        dprintf("Exception pending in %s[%d]\n", dispChanState_v02_01[chanNum].name, headNum);
    }
}

LwU32 dispGetDmiMemaccOffset_v02_01()
{
    return LW_PDISP_DMI_MEMACC;
}


LW_STATUS dispGetChnAndPbCtlRegOffsets_v02_01
(
    LwU32           headNum,
    LwU32           channelNum,
    LwU32           channelClass,
    LwU32*           pChnCtl,
    PBCTLOFFSET*    pPbCtl  
)
{
    if((pChnCtl == NULL) && (pPbCtl == NULL))
    {
        return LW_ERR_GENERIC;
    }

    if(pChnCtl) 
    {
        switch (channelClass)
        {
            case CHNTYPE_CORE:
                *pChnCtl = LW_PDISP_CHNCTL_CORE(headNum);
            break;
    
            case CHNTYPE_BASE:
                *pChnCtl = LW_PDISP_CHNCTL_BASE(headNum);       
            break;
    
            case CHNTYPE_OVLY:
                *pChnCtl = LW_PDISP_CHNCTL_OVLY(headNum);
            break;
    
            default:
                dprintf("lw : Illegal channel type. "
                    "Use core, base or ovly. Aborting\n");
                return LW_ERR_GENERIC;
        }
    }

    if (pPbCtl)
    {
        if (channelNum >= LW_PDISP_PBCTL0__SIZE_1)
        {
            return LW_ERR_GENERIC;
        }

        pPbCtl->PbCtlOffset[0] = LW_PDISP_PBCTL0(channelNum);
        pPbCtl->PbCtlOffset[1] = LW_PDISP_PBCTL1(channelNum);
        pPbCtl->PbCtlOffset[2] = LW_PDISP_PBCTL2(channelNum);
    }
    return LW_OK;
}

LwU32 dispReadOrSetControlArm_v02_01(LWOR orType, LwU32 idx)
{
    switch (orType)
    {
        case LW_OR_SOR:
            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(
                            LW_PDISP_907D_CHN_CORE) + 
                            LW917D_SOR_SET_CONTROL(idx));
        case LW_OR_DAC:
            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(
                            LW_PDISP_907D_CHN_CORE) + 
                            LW917D_DAC_SET_CONTROL(idx));
        case LW_OR_PIOR:
            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(
                            LW_PDISP_907D_CHN_CORE) + 
                            LW917D_PIOR_SET_CONTROL(idx));
        default:
            return 0;
    }
}

ORPROTOCOL dispGetOrProtocol_v02_01(LWOR orType, LwU32 protocolValue)
{
    switch (orType)
    {
        case LW_OR_SOR:
        {
            switch (protocolValue)
            {
                case LW917D_SOR_SET_CONTROL_PROTOCOL_LVDS_LWSTOM:
                    return sorProtocol_LvdsLwstom;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_A:
                    return sorProtocol_SingleTmdsA;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_SINGLE_TMDS_B:
                    return sorProtocol_SingleTmdsB;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_DUAL_TMDS:
                    return sorProtocol_DualTmds;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_DP_A:
                    return sorProtocol_DpA;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_DP_B:
                    return sorProtocol_DpB;
                case LW917D_SOR_SET_CONTROL_PROTOCOL_LWSTOM:
                    return sorProtocol_Lwstom;
            }
            break;
        }
        case LW_OR_PIOR:
        {
            switch (protocolValue)
            {
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_EXT_TMDS_ENC:
                    return piorProtocol_ExtTmdsEnc;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_EXT_TV_ENC:
                    return piorProtocol_ExtTvEnc;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_SD_ENC:
                    return piorProtocol_ExtSdiSdEnc;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_EXT_SDI_HD_ENC:
                    return piorProtocol_ExtSdiHdEnc;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_OUT:
                    return piorProtocol_DistRenderOut;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_IN:
                    return piorProtocol_DistRenderIn;
                case LW917D_PIOR_SET_CONTROL_PROTOCOL_DIST_RENDER_INOUT:
                    return piorProtocol_DistRenderInout;
            }
            break;
        }
        case LW_OR_DAC:
        {
            switch (protocolValue)
            {
                case LW917D_DAC_SET_CONTROL_PROTOCOL_RGB_CRT:
                    return dacProtocol_RgbCrt;
                case LW917D_DAC_SET_CONTROL_PROTOCOL_YUV_CRT:
                    return dacProtocol_YuvCrt;
            }
            break;
        }
    }
    return protocolError;
}

void
dispGetPiorSeqCtlPwrAndBlankRegs_v02_01
(
    LwU32 piorIndex,
    LwU32 *pSeqCtlReg,
    LwU32 *pPwrReg,
    LwU32 *pBlankReg
)
{
    if (pSeqCtlReg)
        *pSeqCtlReg = LW_PDISP_PIOR_SEQ_CTL(piorIndex);

    if (pPwrReg)
        *pPwrReg    = LW_PDISP_PIOR_PWR(piorIndex);

    if (pBlankReg)
        *pBlankReg  = LW_PDISP_PIOR_BLANK(piorIndex);
}

void dispHeadOrConnAsciiData_v02_01 
(
    LWOR orType,
    LwU32 *headDisplayIds,
    LwU32 *ownerMasks
)
{
    LwU32 orNum=0, data32=0, ownerMask=0, head, displayCode;    

    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(orType); orNum++) 
    {
        if (pDisp[indexGpu].dispResourceExists(orType, orNum) != TRUE)
        {
            continue;
        }

        switch(orType)
        {
                case LW_OR_SOR:
                        data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_SOR_SET_CONTROL(orNum));
                        ownerMask = DRF_VAL(917D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
                        break;
                case LW_OR_DAC:
                        data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_DAC_SET_CONTROL(orNum));
                        ownerMask = DRF_VAL(917D, _DAC_SET_CONTROL, _OWNER_MASK, data32);
                        break;
                case LW_OR_PIOR:
                        data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_PIOR_SET_CONTROL(orNum));
                        ownerMask = DRF_VAL(917D, _PIOR_SET_CONTROL, _OWNER_MASK, data32);
                        break;
        }

        ownerMasks[orNum]=ownerMask;
        if (!ownerMask)
        {
                continue;
        }
        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
        {
            if (BIT(head) & ownerMask)
            {
                data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_HEAD_SET_DISPLAY_ID(head, 0));
                displayCode = DRF_VAL(917D, _HEAD_SET_DISPLAY_ID, _CODE, data32);
                headDisplayIds[head]=displayCode;
            }
        }
    }
}


void dispHeadSorConnection_v02_01(void)
{
    LwU32       orNum, data32, head, ownerMask;
    LwS32       numSpaces;
    ORPROTOCOL  orProtocol;
    char        *protocolString;
    char        *orString = dispGetORString(LW_OR_SOR);
    BOOL        bAtLeastOneHeadPrinted;

    protocolString = (char *)malloc(256 * sizeof(char));

    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); orNum++) 
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_SOR, orNum) != TRUE)
        {
            continue;
        }
        
        //dispPrintOwnerProtocol_LW50(LW_OR_SOR, orNum);
        data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_SOR_SET_CONTROL(orNum));
        ownerMask = DRF_VAL(917D, _SOR_SET_CONTROL, _OWNER_MASK, data32);
        if (!ownerMask)
        {
            dprintf("%s%d    NONE        N/A                 ", orString, orNum);
        }
        else
        {
            bAtLeastOneHeadPrinted = FALSE;

            orProtocol = pDisp[indexGpu].dispGetOrProtocol(LW_OR_SOR, DRF_VAL(917D, _SOR_SET_CONTROL, _PROTOCOL, data32));
            sprintf(protocolString, "%s", dispGetStringForOrProtocol(LW_OR_SOR, orProtocol));

            // Check extra information in case protocol was DP & add that to the data to be printed
            if ((orProtocol == sorProtocol_DpA) || (orProtocol == sorProtocol_DpB))
            {
                // Read DP_LINKCTL data & add appropriate sting to protocol
                pDisp[indexGpu].dispReadDpLinkCtl(orNum,
                                                  ((orProtocol == sorProtocol_DpA) ? 0 : 1),
                                                  protocolString);

            }

            dprintf("%s%d    HEAD", orString, orNum);

            numSpaces = 7;
            // If more that one owner is there, we need to print brackets
            if (ownerMask & (ownerMask  - 1))
            {    
                dprintf("(");
                --numSpaces;
            }
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (BIT(head) & ownerMask)
                {
                    if (bAtLeastOneHeadPrinted)
                    {
                        dprintf("|");
                        --numSpaces;
                    }
                    bAtLeastOneHeadPrinted = TRUE;
                    dprintf("%d", head);
                    --numSpaces;
                }
            }
            // If more that one owner is there, we need to print brackets
            if (ownerMask & (ownerMask  - 1))
            {    
                dprintf(")");
                --numSpaces;
            }
            while (numSpaces > 0)
            {
                dprintf(" ");
                numSpaces--;
            }
            dprintf(" %-20s", protocolString);
        }

        data32 = GPU_REG_RD32(LW_PDISP_SOR_PWR(orNum));
        if(DRF_VAL(_PDISP, _SOR_PWR, _MODE, data32) == LW_PDISP_SOR_PWR_MODE_SAFE)
        {
            dprintf("SAFE     %-40s", (DRF_VAL(_PDISP, _SOR_PWR, _SAFE_STATE, data32) == LW_PDISP_SOR_PWR_SAFE_STATE_PU)? "PU" : "PD");
        }
        else
        {
            dprintf("NORMAL   %-40s", (DRF_VAL(_PDISP, _SOR_PWR, _NORMAL_STATE, data32) == LW_PDISP_SOR_PWR_NORMAL_STATE_PU)? "PU" : "PD");
        }

        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
        {
            if (head > 0)
            {
                dprintf("/");
            }
            if (BIT(head) & ownerMask)
            {
                data32 = GPU_REG_RD32(LW_PDISP_SF_BLANK(head));
                if (DRF_VAL(_PDISP, _SF_BLANK, _STATUS, data32) == LW_PDISP_SF_BLANK_STATUS_BLANKED)
                {
                    dprintf("YES%s", (DRF_VAL(_PDISP, _SF_BLANK, _OVERRIDE, data32) == LW_PDISP_SF_BLANK_OVERRIDE_TRUE)? " (because of override)" : "");
                }
                else
                {
                    dprintf("NO");
                }
            }
            else
            {
                dprintf("NA"); // If a head is not attached, we say Not Applicable.
            }
        }
        dprintf("\n");
    }
}

// 
// Analyze the blank display
//
LwS32 dispAnalyzeBlank_v02_01
(
    LwS32 headNum
)
{
    LwS32 status              = 0; 
    LwU32 data32              = 0; 
    LwU32 i                   = 0; 
    LwU32 numSinks            = 0;
    LwU32 temp32              = 0; 
    LwU32 temp                = 0;
    BOOL nullCoreBaseCtxDmas = TRUE;
    LwS32 owner               = 0;
    LwU32 chType              = 0;

    if ((headNum >= (LwS32)pDisp[indexGpu].dispGetNumHeads()) || (headNum <= -1))
    {
        headNum = 0;   
        dprintf("Wrong head!!! hence Setting Head = %d\n\n", headNum);
    }    

    //
    // Check the DispOwner
    //
    dprintf("---------------------------------------------------------------------\n");
    dprintf("Display Owner:            %s\n", ((owner = pDisp[indexGpu].dispDispOwner()) == 0)? "Driver         (Looks OK)": "Vbios          (Did you expect vbios mode?)");

    // If Display owner is VBIOS and VGA's base status is INVALID, then head will be blanked
    if(owner != 0)
    {
        data32 = GPU_REG_RD_DRF(_PDISP, _VGA_BASE, _STATUS);        
        dprintf("VGA base status: ");
        if(data32 == LW_PDISP_VGA_BASE_STATUS_ILWALID)
                dprintf("INVALID\n");
        else
                dprintf("VALID\n");
    }

    //
    // Status of the Core, Base and Ovly channel
    //
    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_CORE, 0, _STATE);
    dprintf("Core Channel State:       %s\n", dCoreState[data32]);

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_BASE, headNum, _STATE);
    if (data32 != LW_PDISP_CHNCTL_BASE_STATE_DEALLOC)
    {
        dprintf("Base Channel State:       %s\n", dBaseState[data32]);
    }

    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_OVLY, headNum, _STATE);
    if (data32 != LW_PDISP_CHNCTL_OVLY_STATE_DEALLOC)
    {
        dprintf("Ovly Channel State:       %s\n", dOvlyState[data32]);
    }

    //
    // Is an exception pending in core or base channels thats 
    // preventing the channels from proceeding
    //
    pDisp[indexGpu].dispPrintExceptPending(headNum);

    
    // DMI_MEMACC status. It is fetching or stopped?  
    data32 = GPU_REG_RD32(pDisp[indexGpu].dispGetDmiMemaccOffset());    
    
    temp32 = DRF_IDX_VAL(_PDISP, _DMI_MEMACC_HEAD, _STATUS, (LwU32)headNum, data32);
    dprintf("Dmi Fetch Status:         %s\n",
            (temp32 == LW_PDISP_DMI_MEMACC_HEAD_STATUS_FETCHING) ? "Fetching       (Looks OK)" : "Stopped        (Did you expect head to be not fetching data?)");

    temp  = DRF_IDX_VAL(_PDISP, _DMI_MEMACC_HEAD, _SETTING_NEW, (LwU32)headNum, data32);            
    if (temp == LW_PDISP_DMI_MEMACC_HEAD_SETTING_NEW_PENDING)
    {
        dprintf("A new setting for DMI_MEMACC is pending\n");

        temp  = DRF_IDX_VAL(_PDISP, _DMI_MEMACC_HEAD, _REQUEST, (LwU32)headNum, data32);

        if ( ((temp32 == LW_PDISP_DMI_MEMACC_HEAD_STATUS_FETCHING) &&
             ( temp   != LW_PDISP_DMI_MEMACC_HEAD_REQUEST_FETCH))  ||
             ((temp32 != LW_PDISP_DMI_MEMACC_HEAD_STATUS_FETCHING) &&
             ( temp   == LW_PDISP_DMI_MEMACC_HEAD_REQUEST_FETCH)) )
        {
            dprintf("Software's request to DMI: %s\n",
                    (temp == LW_PDISP_DMI_MEMACC_HEAD0_REQUEST_FETCH) ? "Fetch" : "Stop");
            dprintf("Software's request doesn't match the current status\n");
        }
    }
    

    //
    // Iso ctx dmas in base and core channels. 
    //
    data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_CORE, 0, _ALLOCATION);
    if (data32 == LW_PDISP_CHNCTL_CORE_ALLOCATION_ALLOCATE)
    {
        dprintf("Iso Ctx dmas in core:\n");
        chType = CHNTYPE_CORE;
        for (i = 0; i < pDisp[indexGpu].dispGetNumSurfPerHead(&chType); i++)
        {
            data32 = pDisp[indexGpu].dispReadSurfCtxDmaHandle(&chType, headNum, i);
            if (!i) // because of zaphod
            {
                dprintf("    Surf%d:                0x%08x", i, data32);
                if (data32)
                {
                    nullCoreBaseCtxDmas = FALSE;
                    dprintf("     (Looks OK)\n");
                }
                else
                {
                    dprintf("     (Did you expect a NULL iso ctx dma?)\n");
                }
            }                               
            else if (data32)
            {
                dprintf("    Surf%d:                0x%08x", i, data32);
                dprintf("         (This is bad since zaphod is not yet supported)\n");
            }
         }

        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_BASE, headNum, _ALLOCATION);
        if (data32 == LW_PDISP_CHNCTL_BASE_ALLOCATION_ALLOCATE)
        {
            dprintf("Iso Ctx dmas in base:\n");
            chType = CHNTYPE_BASE;
            for (i = 0; i < pDisp[indexGpu].dispGetNumSurfPerHead(&chType); i++)
            {
                data32 = pDisp[indexGpu].dispReadSurfCtxDmaHandle(&chType, headNum, i);
                if (i == 0)
                {
                    dprintf("    Surf%d:                0x%08x", i, data32);
                    if (data32)
                    {
                        nullCoreBaseCtxDmas = FALSE;
                        dprintf("     (Looks OK)\n");
                    }
                    else
                    {
                        dprintf("     (Did you expect a NULL iso ctx dma?)\n");
                    }
                }
                else if (data32)
                {
                    dprintf("    Surf%d:                0x%08x\n", i, data32);
                }
            }
        }

        if(nullCoreBaseCtxDmas == TRUE)
        {
            data32 = GPU_REG_IDX_RD_DRF(_PDISP, _CHNCTL_OVLY, headNum, _ALLOCATION); 
            if (data32 == LW_PDISP_CHNCTL_OVLY_ALLOCATION_ALLOCATE)
            {
                dprintf("Iso Ctx dmas in ovly:\n");
                chType = CHNTYPE_OVLY;
                for(i = 0; i < pDisp[indexGpu].dispGetNumSurfPerHead(&chType); i++)
                {
                    data32 = pDisp[indexGpu].dispReadSurfCtxDmaHandle(&chType, headNum, i);
                    if (i == 0)
                    {
                        dprintf("    Surf%d:                0x%08x", i, data32);
                        if (data32)
                        {
                            dprintf("     (Looks OK)\n");
                        }
                        else
                        {
                            dprintf("     (Did you expect a NULL iso ctx dma? All the channels have NULL iso ctx dmas)\n");
                        }
                    }
                    else if (data32)
                    {
                        dprintf("    Surf%d:                0x%08x\n", i, data32);
                    }

                }

            }
        }
    }

    dprintf("ORs owned by the head:    ");

    //
    // Does this head own any OR at all?
    //
    pDisp[indexGpu].dispUpdateNumSinks(headNum, LW_OR_DAC, &numSinks);
    pDisp[indexGpu].dispUpdateNumSinks(headNum, LW_OR_SOR, &numSinks);
    pDisp[indexGpu].dispUpdateNumSinks(headNum, LW_OR_PIOR, &numSinks);

    if (numSinks == 0)
    {
        dprintf("NONE           (Did you expect no ORs to be owned by this head?)");
    }
    else
    {
        if (numSinks <= 2)
        {
            for (i = 2 - numSinks; i > 0; --i)
            {
                dprintf("      ");
            }
            dprintf("   (Looks OK)");
        }
        else
        {
            dprintf("   (Looks OK). Alignment/formatting will be bad");
        }
    }
    dprintf("\n");

    // Check if DD blanked the head using SetGetHeadBlankingCtrl or LW917D_SET_GET_HEAD_BLANKING_CTRL
    data32 = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(LW_PDISP_907D_CHN_CORE) + 
                      LW917D_HEAD_SET_GET_BLANKING_CTRL(headNum));
    dprintf("Is head blanked by DD? %s",(DRF_VAL(917D, _HEAD_SET_GET_BLANKING_CTRL, _BLANK, data32) == LW917D_HEAD_SET_GET_BLANKING_CTRL_BLANK_ENABLE)?"   Yes":"   No");

    dprintf("\n");
    dprintf("---------------------------------------------------------------------\n");

    return status;
}

// Analyze PRE state machine to debug display hang.

void dispAnalyzeHangPreSm_v02_01(void)
{
    LwU32 data32     = 0;
    LwU32 i          = 0;
    LwU32 numOfHeads = 0;
    dprintf("\n");
    dprintf("lw: Analyzing PRE State machine in all the heads\n");

    numOfHeads = pDisp[indexGpu].dispGetNumHeads();
    for(i = 0; i < numOfHeads; i++)
    {
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _DSI_CORE_HEAD_UPD_STATE, i, _PRE);        
        dprintf("lw:     DSI CORE UPDATE STATE PRE in head %d: 0x%08x\n",i,data32); 
        dprintf("lw:     DSI CORE UPDATE PRE WAITING: ");
        switch(data32)
        {
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_SNOOZE:
                dprintf(" to send the snooze command to the OR\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SNOOZE:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SNOOZE)\n");        
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_SAFE:
                dprintf(" to send the safe power command to the OR\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SAFE:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SAFE)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_UPD1:
                dprintf(" to send the update to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_UPD1)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_UPD1:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_UPD1)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_LV1:
                dprintf(" for 1 loadv.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_POLL1:
                dprintf(" for a response to the request sent in POLL1.\n");        
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_SLEEP:
                dprintf(" to send the sleep command to the OR.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SLEEP:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_SLEEP)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_UPD2:
                dprintf(" to send the update to the OR. \n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_SEND_UPD2)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_UPD2:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_UPD2)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_LV2:
                dprintf(" for a loadv to activate the sleep bundle.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_POLL2:
                dprintf(" waits for a response to the request sent in POLL2.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_DETACH:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_DETACH)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_PRE_WAIT_POLL3:
                dprintf(" for the response to the request sent in the POLL3 state\n");        
                break;
            default:
                dprintf(" for NONE\n");
        }

    }
}

// Analyze CMGR state machine to debug display hang.
void dispAnalyzeHangCmgrSm_v02_01(void)
{
    LwU32 data32 = 0;
    dprintf("\n");
    dprintf("lw: Analyzing CMGR State machine \n");
    data32 = GPU_REG_RD_DRF(_PDISP,_DSI_CORE_UPD_STATE,_CMGR);        
    dprintf("lw:     DISP CORE UPD STATE CMGR: 0x%08x\n",data32);
    dprintf("lw:     DISP CORE UPD CMGR WAITING: ");
    switch(data32)
    {
        case  LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_FOR_ACK_PLLRESET_ENABLE:
            dprintf(" for a response from the DCI for the pll reset.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_SAFE:
            dprintf(" for a response from the DCI that the clocks have been switched to safe.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_SAFE_SETTLE:
            dprintf(" for 7us for the switch to safe clock to settle.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_BYPASS:
            dprintf(" for ack from the DCI for the Bypass command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_BYPASS_SETTLE:
            dprintf(" for 108 dispclk cycles for the vpll bypass to settle.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_DISABLE:
            dprintf(" for an ack from the DCI for the disable command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ON:
            dprintf(" for the desired clocks to have turned on\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_COEFF:
            dprintf(" for a response from the DCI for the coefficients.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_ENABLE:
            dprintf(" for an ack from the DCI for vpll enable.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_VPLL_LOCK_SETTLE:
            dprintf(" for LW_PDISP_DSI_PLL_LOCK_DLY microseconds for the vpll to settle.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_UNBYPASS:
            dprintf(" for an ack from the DCI for the vpll unbypass command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_UNBYPASS_SETTLE:
            dprintf(" for 108 dispclk cycles for the vpll unbypass.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_SET_OWNER:
            dprintf(" for LW_PDISP_DSI_PLL_LOCK_DLY microseconds for the clock to switch.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_FOR_ACK_PLLRESET_DISABLE:
            dprintf(" for a response from the DCI to the pll reset disable command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_MACROPLL_SETTLE:
            dprintf(" for LW_PDISP_DSI_VPLL_MACROPLL_LOCK_DLY microseconds for the PLL in the SOR macros to settle.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_UNSAFE:
            dprintf(" for a response from the DCI for the Unsafe command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_UNSAFE_SETTLE:
            dprintf(" for 7us for the Unsafe to settle.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_FAST:
            dprintf(" for a response from the DCI for the fast command.\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_CMGR_WAIT_4_ACK_OF_NORM:
            dprintf(" for a response from the DCI for the normal command.\n");
            break;
        default:
            dprintf(" for NONE\n");
    }
}

// Analyze POST state machine to debug display hang.
void dispAnalyzeHangPostSm_v02_01(void)
{
    LwU32 data32     = 0;
    LwU32 i          = 0;
    LwU32 numOfHeads = 0;
    dprintf("\n");
    dprintf("lw: Analyzing POST State machine in all the heads\n");

    numOfHeads = pDisp[indexGpu].dispGetNumHeads();
    for(i = 0; i < numOfHeads; i++)
    {
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _DSI_CORE_HEAD_UPD_STATE, i, _POST);        
        dprintf("lw:     DSI CORE UPDATE STATE POST in head %d: 0x%08x\n",i,data32); 
        dprintf("lw:     DSI CORE UPDATE POST WAITING: ");
        switch(data32)
        {
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_ATTACH:
                dprintf(" for an ack from the bundle bus arbiter that the attach command has been sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_ATTACH)\n"); 
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_POLL1:
                dprintf(" for a response from the request sent in POLL1.\n"); 
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSLEEP:
                dprintf(" to send the unsleep command to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSLEEP)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSLEEP:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSLEEP)\n"); 
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSAFE:
                dprintf(" to send the unsafe command to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSAFE)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSAFE:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSAFE)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UPD2:
                dprintf(" to send the update to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UPD2)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UPD2:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UPD2)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_LV2:
                dprintf(" for a loadv to activate the unsleep bundle.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_LV2)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_POLL2:
                dprintf(" for a response to the request sent in POLL2.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_BACK2MAIN:
                dprintf(" in this state until after SV3.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSNOOZE:
                dprintf(" to send the unsnooze command to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UNSNOOZE)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSNOOZE:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UNSNOOZE)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UPD3:
                dprintf(" to send the update to the OR.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_SEND_UPD3)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UPD3:
                dprintf(" for a response from the bundle arbiter saying that the bundle was sent.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_UPD3)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_LV3:
                dprintf(" for a loadv to activate the unsnooze bundle.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_LV3)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_POLL3:
                dprintf(" for a response to the request sent in POLL3.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_4_LV_ONLY:
                dprintf(" for a loadv.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_POST_WAIT_4_LV_ONLY)\n");
                break;
            default:
                dprintf(" for NONE\n");
        }
    }
}

// Analyze USUB state machine to debug display hang.
void dispAnalyzeHangUsubSm_v02_01(void)
{
    LwU32 data32     = 0;
    LwU32 i          = 0;
    LwU32 numOfHeads = 0;
    dprintf("\n");
    dprintf("lw: Analyzing USUB State machine in all the heads\n");

    numOfHeads = pDisp[indexGpu].dispGetNumHeads();
    for(i = 0; i < numOfHeads; i++)
    {
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _DSI_CORE_HEAD_UPD_STATE, i, _USUB);        
        dprintf("lw:     DSI CORE UPDATE STATE USUB in head %d: 0x%08x\n",i,data32); 
        dprintf("lw:     DSI CORE UPDATE USUB WAITING: ");
        switch(data32)
        {
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_WAITPOLL:
                dprintf(" to send a request to the OR Polling mechanism to make sure there is room in the OR's bundle fifo.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_WAIT4BASE:
                dprintf(" if interlocked with base, wait for it to be ready to update.\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_BWAIT:
                dprintf(" for the update bundle ack from the bundle bus arbiter. \n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_BWAIT)\n");
                break;
            case LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_OWAIT:
                dprintf(" for the update bundle ack from the bundle bus arbiter.\n");
                dprintf("lw:     (State: LW_PDISP_DSI_CORE_HEAD_UPD_STATE_USUB_OWAIT)\n");
                break;
            default:
                dprintf(" for NONE\n");
        }
    }
}

//Analyze display hang

void dispAnalyzeHang_v02_01(void)
{
    LwU32 data32 = 0, i=0;
    LwU32 numOfHeads = 0;
    numOfHeads = pDisp[indexGpu].dispGetNumHeads();
    
    dprintf("===========Analyze Display Hang Start===========\n");
    dprintf("\n");
    dprintf("lw: Analyzing RG Underflow state\n");
    for(i = 0; i < numOfHeads; i++)
    {
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, i, _ENABLE);        
        dprintf("lw:     HEAD %d ",i);
        if(data32 == LW_PDISP_RG_UNDERFLOW_ENABLE_DISABLE)
        {
            dprintf("Underflow reporting disabled\n");
            continue;
        }
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, i, _UNDERFLOWED);        
        if(data32 == LW_PDISP_RG_UNDERFLOW_UNDERFLOWED_YES)
            dprintf("Underflowed. ");
        else
            dprintf("Not Underflowed. ");
        data32 = GPU_REG_IDX_RD_DRF(_PDISP, _RG_UNDERFLOW, i, _MODE);        
        if(data32 == LW_PDISP_RG_UNDERFLOW_MODE_RED)
            dprintf("Mode set to RED");
        else
            dprintf("Mode set to REPEAT");
        dprintf("\n");
    }            
    dprintf("\n");
    data32 = GPU_REG_RD_DRF(_PDISP,_DSI_CORE_UPD_STATE,_MAIN);
    dprintf("lw: Analyzing Main State machine: central control state machine controlling core updates\n");
    dprintf("lw:     DSI CORE UPDATE STATE: 0x%08x\n",data32); 
    dprintf("lw:     DSI CORE UPDATE WAITING: ");
    switch (data32)
    {
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_TO_BEGIN:
            dprintf(" to begin\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM1:
            dprintf(" for the RM to write LW_PDISP_SUPERVISOR_RESTART.\n");
            dprintf("lw:     (State: LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM1)\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_FAST:
            dprintf(" for the switch to fast dispclk\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_PRE:
            dprintf(" for the PRE state machine to complete.\n");
            pDisp[indexGpu].dispAnalyzeHangPreSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_PRECLK:
            dprintf(" for the CMGR to finish disabling clocks or switching them to safe mode.");
            pDisp[indexGpu].dispAnalyzeHangCmgrSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM2:
            dprintf(" for the RM to write LW_PDISP_SUPERVISOR_RESTART.\n"); 
            dprintf("lw:     (State: LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM2)\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_USUB:
            dprintf(" for the USUB state machines to finish.\n");
            pDisp[indexGpu].dispAnalyzeHangUsubSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_POSTCLK:
            dprintf(" for the CMGR state machine to finish.\n");
            pDisp[indexGpu].dispAnalyzeHangCmgrSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_POST1:
            dprintf(" for the POST state machines to finish.\n");
            dprintf("lw:     (State: LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_POST1)\n");
            pDisp[indexGpu].dispAnalyzeHangPostSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM3:
            dprintf(" for the RM to write LW_PDISP_SUPERVISOR_RESTART.\n"); 
            dprintf("lw:     (State: LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_RM3)\n");
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_POST2:
            dprintf(" for the POST state machines to finish.\n"); 
            dprintf("lw:     (State: LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_POST2 \n");
            pDisp[indexGpu].dispAnalyzeHangPostSm();
            break;
        case LW_PDISP_DSI_CORE_UPD_STATE_MAIN_WAIT_4_NORMAL:
            dprintf(" for dispclk to resume to normal speed.\n");
            break;
        default:
            dprintf(" for NONE\n");

    }
    dprintf("\n");
    dprintf("===========Analyze Display Hang End=============\n");
}


/*!
 *  Finds out owner and protocol for given OR
 *
 *
 *  @param[in]  orType      Type of OR
 *  @param[in]  orNum       Or index
 *
 *  @param[out] pOrOwner    HEAD owner
 *  @param[out] pOrProtocol OR   Protocol
 *
 *  @return   void
 */
void
dispReadOrOwnerAndProtocol_v02_01
(
    LWOR orType,
    LwU32 orNum,
    HEAD *pOrOwner,
    ORPROTOCOL *pOrProtocol
)
{
    LwU32       readOrData = 0;
    LwU32       value = 0;
    HEAD        orOwner = HEAD_UNSPECIFIED;

    if (!pOrOwner && !pOrProtocol)
    {
        dprintf("Bad args to %s. Both pOrOwner and pOrProtocol are NULL. Bailing out early\n", __FUNCTION__);
        return;
    }

    readOrData = pDisp[indexGpu].dispReadOrSetControlArm(orType, orNum);

    switch (orType)
    {
        case LW_OR_SOR:
        {
            value = DRF_VAL(917D, _SOR_SET_CONTROL, _OWNER_MASK, readOrData);

            switch (value)
            {
                case LW917D_SOR_SET_CONTROL_OWNER_MASK_NONE:
                    break; 
                case LW917D_SOR_SET_CONTROL_OWNER_MASK_HEAD0:
                    orOwner = HEAD0;
                    break; 
                case LW917D_SOR_SET_CONTROL_OWNER_MASK_HEAD1:
                    orOwner = HEAD1;
                    break; 
                case LW917D_SOR_SET_CONTROL_OWNER_MASK_HEAD2:
                    orOwner = HEAD2;
                    break; 
                case LW917D_SOR_SET_CONTROL_OWNER_MASK_HEAD3:
                    orOwner = HEAD3;
                    break;
                default:
                    dprintf("Bad owner%d for SOR%d\n", value, orNum);
                    break;
            }

            value = DRF_VAL(917D, _SOR_SET_CONTROL, _PROTOCOL, readOrData);
            
            break;
        }
        case LW_OR_PIOR:
        {
            value = DRF_VAL(917D, _PIOR_SET_CONTROL, _OWNER_MASK, readOrData);

            switch (value)
            {
                case LW917D_PIOR_SET_CONTROL_OWNER_MASK_NONE:
                    break; 
                case LW917D_PIOR_SET_CONTROL_OWNER_MASK_HEAD0:
                    orOwner = HEAD0;
                    break; 
                case LW917D_PIOR_SET_CONTROL_OWNER_MASK_HEAD1:
                    orOwner = HEAD1;
                    break; 
                case LW917D_PIOR_SET_CONTROL_OWNER_MASK_HEAD2:
                    orOwner = HEAD2;
                    break; 
                case LW917D_PIOR_SET_CONTROL_OWNER_MASK_HEAD3:
                    orOwner = HEAD3;
                    break;
                default:
                    dprintf("Bad owner%d for PIOR%d\n", value, orNum);
                    break;
            }

            value = DRF_VAL(917D, _PIOR_SET_CONTROL, _PROTOCOL, readOrData);
            
            break;
        }

        case LW_OR_DAC:
        {
            value = DRF_VAL(917D, _DAC_SET_CONTROL, _OWNER_MASK, readOrData);

            switch (value)
            {
                case LW917D_DAC_SET_CONTROL_OWNER_MASK_NONE:
                    break; 
                case LW917D_DAC_SET_CONTROL_OWNER_MASK_HEAD0:
                    orOwner = HEAD0;
                    break; 
                case LW917D_DAC_SET_CONTROL_OWNER_MASK_HEAD1:
                    orOwner = HEAD1;
                    break; 
                case LW917D_DAC_SET_CONTROL_OWNER_MASK_HEAD2:
                    orOwner = HEAD2;
                    break; 
                case LW917D_DAC_SET_CONTROL_OWNER_MASK_HEAD3:
                    orOwner = HEAD3;
                    break;
                default:
                    dprintf("Bad owner%d for DAC%d\n", value, orNum);
                    break;
            }

            value = DRF_VAL(917D, _DAC_SET_CONTROL, _PROTOCOL, readOrData);
            
            break;
        }
        default:
            dprintf("Error\n");
    }

    if (pOrOwner)
    {
        *pOrOwner = orOwner;
    }

    if (pOrProtocol)
    {
        *pOrProtocol = pDisp[indexGpu].dispGetOrProtocol(orType, value);
        if (*pOrProtocol == protocolError)
            dprintf("Bad protocol%d for %s%d\n", value, dispGetORString(orType), orNum);
    }
}

//
// Returns the debug mode
// If -1 is returned, chanNum is out of range
// If -2 is returned, debugMode is not available
//

LwS32 dispGetDebugMode_v02_01
(
    LwU32 chanNum
)
{
    ChanDesc_t *chnst;
    LwU32       val;
    LwU32       numDispChannels = pDisp[indexGpu].dispGetMaxChan();
    if (chanNum >= numDispChannels) 
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return -1;
    }
    chnst=&dispChanState_v02_01[chanNum];
    if (! (chnst->cap & DISP_DEBUG) )
    {
        return -2;
    }
    val = GPU_REG_RD32(LW_PDISP_DSI_DEBUG_CTL(chanNum));
    return (FLD_TEST_DRF(_PDISP, _DSI_DEBUG_CTL, _MODE, _ENABLE, val)) ? TRUE: FALSE;
}


void  dispSetDebugMode_v02_01
(
    LwU32 chanNum, 
    BOOL set
)
{
    ChanDesc_t *chnst;
    LwU32       val;
    LwU32       numDispChannels = pDisp[indexGpu].dispGetMaxChan();

    if (chanNum >= numDispChannels)
    {
        dprintf("chanNum should be less than %d\n", numDispChannels);
        return ;
    }
    chnst = &dispChanState_v02_01[chanNum];
    // Check for cap.
    if (! (chnst->cap & DISP_DEBUG) )
    {
        dprintf("DebugMode is not available for channel %d\n", chanNum);
        return ;
    }
    val = GPU_REG_RD32(LW_PDISP_DSI_DEBUG_CTL(chanNum));
    if (set != FLD_TEST_DRF(_PDISP, _DSI_DEBUG_CTL, _MODE, _ENABLE, val)) 
    {
        if (set)
                val = FLD_SET_DRF(_PDISP, _DSI_DEBUG_CTL, _MODE, _ENABLE, val);
        else
                val = FLD_SET_DRF(_PDISP, _DSI_DEBUG_CTL, _MODE, _DISABLE, val);

        GPU_REG_WR32(LW_PDISP_DSI_DEBUG_CTL(chanNum), val);
        val = GPU_REG_RD32(LW_PDISP_DSI_DEBUG_CTL(chanNum));
        if (set != FLD_TEST_DRF(_PDISP, _DSI_DEBUG_CTL, _MODE, _ENABLE, val))
                dprintf("Failed to set the debug mode..(0x%08x)\n",val);
    }
}


void dispPrintChalwars_v02_01
(
    LwS32 chanNum,
    BOOL printHeadless
)
{
    ChnType chanId;
    LwU32 k;
    LwU32 headNum;

    chanId = dispChanState_v02_01[chanNum].id;
    headNum = dispChanState_v02_01[chanNum].headNum;

    switch (chanId)
    {
        case CHNTYPE_CORE: 
            for (headNum = 0; headNum < 4 /*pDisp[indexGpu].dispGetNumHeads()*/; ++headNum)
            {
                dprintf("---------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                     ARM      |     ARM OFF\n", headNum);
                dprintf("---------------------------------------------------------------------------------\n");
 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_VIEWPORT_VISIBLE_HEIGHT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_RASTER_WIDTH_ACTIVE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_CORE_OUTPUT_PASSTHROUGH_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_CORE_OUTPUT_PASSTHROUGH, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USING_BASE_LUT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_OFFSET_ISO, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_HEAD_CONNECTED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_HEAD_CONNECTED_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_CORE_BASE_ACTIVE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_RASTER_HEIGHT_ACTIVE_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_VIEWPORT_OUT_FULLY_VISIBLE_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_CORE_BASE_ACTIVE_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USING_OUTPUT_LUT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_VIEWPORT_OUT_FULLY_VISIBLE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_OFFSET_LWRSOR, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_BASE_LUT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_OUTPUT_LUT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_RASTER_HEIGHT_ACTIVE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_WILL_BE_SCALING, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_BASE_BASE_ACTIVE_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USING_SURFACE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_WAS_SCALING, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_GLOBAL_BASE_BASE_ACTIVE, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USING_SCALER, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_CONTEXT_DMA_LUT, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USING_LWRSOR, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_CONTEXT_DMA_LWRSOR, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_HWV_NEW_CONTEXT_DMA_ISO, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_USE_OF_SCALER_CHANGED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
                DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917D_ECV_RASTER_WIDTH_ACTIVE_ARMED, headNum, LW_PDISP_907D_SC_CORE_VARIABLES); 
            }

            if( printHeadless == TRUE )
            {
                dprintf("---------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                     ARM      |     ARM OFF\n");
                dprintf("----------------------------------------------------------------------------------\n");
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_BLOCK_HEIGHT_STATE, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_TMP_DELTA, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_LINE_STORE_SIZE, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_FILTER_MODE444, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_VIEWPORT_MAX_Y, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_PIXEL_SHIFT, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_MODE_SWITCH_CHECKS_REQUIRED, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_LAST_GOB_LINE, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_HWV_NEW_OFFSET_NOTIFIER, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_HWV_NEW_CONTEXT_DMA_NOTIFIER, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_HEIGHT_GOBS, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_TMP_OFFSET, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_TMP_SHORT, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_BUFFER_SIZE_GOBS, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_VIEWPORT_MIN_Y, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_WIDTH_BYTES, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_BLOCK_HEIGHT_MASK, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_STATE_ERR_CODE, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_LWRSOR_SIZE_GOBS, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_HEIGHT_LINES, LW_PDISP_907D_SC_CORE_VARIABLES);
                DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917D_ECV_LAST_LINE, LW_PDISP_907D_SC_CORE_VARIABLES);
            }

            break;

        case CHNTYPE_BASE:            
            dprintf("-----------------------------------------------------------------------------------\n");
            dprintf("BASE CHANNEL HEAD %u                                     ARM      |     ARM OFF\n", headNum);
            dprintf("-----------------------------------------------------------------------------------\n");

            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_PIXEL_SHIFT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_USING_BASE_LUT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_BUFFER_SIZE_GOBS, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_HEAD_CONNECTED, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_WIDTH_BYTES, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_UPDATE_TIMESTAMP_HI, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_OFFSET_NOTIFIER, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_BASE_BASE_ACTIVE_ARMED, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_BASE_BASE_ACTIVE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_HEIGHT_LINES, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_TIMESTAMP_ORIGIN_HI, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_CONTEXT_DMA_SEMAPHORE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_CONTEXT_DMA_NOTIFIER, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_HEAD_CONNECTED_ARMED, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_OFFSET_SEMAPHORE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_UPDATE_TIMESTAMP_LO, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_CONTEXT_DMA_LUT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_TIMESTAMP_ORIGIN_LO, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_SURFACE1NEW_OFFSET0, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_SURFACE1NEW_OFFSET1, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_LAST_LINE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_TMP_OFFSET, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_FORCE_INTERRUPT_RM_BASE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_CHECK_SUB_SURFACE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_USING_OUTPUT_LUT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_BLOCK_HEIGHT_STATE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_PIXEL_DEPTH, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_USING_SURFACE0, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_USING_SURFACE1, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_BASE_LUT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_HEIGHT_GOBS, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_OUTPUT_LUT, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_SURFACE0NEW_OFFSET0, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_SURFACE0NEW_OFFSET1, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_BLOCK_HEIGHT_MASK, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_LAST_GOB_LINE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_CORE_OUTPUT_PASSTHROUGH_ARMED, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_ECV_GLOBAL_CORE_OUTPUT_PASSTHROUGH, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917C_HWV_NEW_VALUE_SEMAPHORE, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));

            if(printHeadless == TRUE)
            {
                for(k = 0; k < LW917C_HWV_NEW_CONTEXT_DMA_ISO__SIZE_1; k++)
                {
                    DISP_PRINT_SC_VAR_SINGLE_IDX_V02_01(LW917C_HWV_NEW_CONTEXT_DMA_ISO, k, LW_PDISP_907C_SC_BASE_VARIABLES(headNum));
                }
            }
            break;

        case CHNTYPE_OVLY:
            dprintf("-----------------------------------------------------------------------------------\n");
            dprintf("OVLY CHANNEL HEAD %u                                     ARM      |     ARM OFF\n", headNum);
            dprintf("-----------------------------------------------------------------------------------\n");

            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_GLOBAL_HEAD_CONNECTED_ARMED, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_FORCE_INTERRUPT_RM_OVERLAY, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_TMP0, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_TMP1, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_PIXEL_SHIFT, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_BUFFER_SIZE_GOBS, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_BLOCK_HEIGHT, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_WIDTH_BYTES, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_TMP_OFFSET, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_UPDATE_TIMESTAMP_HI, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_OFFSET_ISO(0), LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_GLOBAL_BASE_BASE_ACTIVE_ARMED, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_GLOBAL_BASE_BASE_ACTIVE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_OVERLAY_LUT, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_TIMESTAMP_ORIGIN_HI, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_CONTEXT_DMA_SEMAPHORE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_OFFSET_SEMAPHORE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_UPDATE_TIMESTAMP_LO, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_CONTEXT_DMA_LUT, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_BLOCK_HEIGHT_MASK, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_TIMESTAMP_ORIGIN_LO, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_LAST_GOB_LINE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_OVERLAY_ACTIVE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_VALUE_SEMAPHORE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_LAST_LINE, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_CONTEXT_DMA_ISO(0), LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_GLOBAL_HEAD_CONNECTED, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_OFFSET_NOTIFIER, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_HEIGHT_LINES, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_PIXEL_DEPTH, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_HWV_NEW_CONTEXT_DMA_NOTIFIER, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            DISP_PRINT_SC_VAR_NON_IDX_V02_01(LW917E_ECV_HEIGHT_GOBS, LW_PDISP_907E_SC_OVERLAY_VARIABLES(headNum));
            break;

        default:
            break;
    }
}

// Note: Add dummy params for prototype consistency
void dispAnalyzeInterrupts_v02_01(LwU32 dummy1, LwU32 dummy2, LwU32 dummy3, LwU32 dummy4)
{
    LwU32 data32, tgt, pen, ebe, eve, en0,rmk, dmk, pmk, hind=0;
    LwU32 rip, pip, dip, rpe, dpe, ppe;
    int ind = 0;
    data32=tgt=pen=ebe=0;    
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    dprintf("%s   %40s   %10s   %10s   %10s\n","NAME","PENDING?","ENABLED?","TARGET","SANITY CHECK");
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_AWAKEN, _AWAKEN_CHN, "AWAKEN_CHN_", 0);
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_EXCEPTION, _EXCEPTION_CHN, "EXCEPTION_CHN_", 0);
    dprintf("----------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_OR, _OR_SOR, "OR_SOR_", 0);
    DIN_ANL_IDX(_OR, _OR_PIOR, "OR_PIOR_", 0);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    DIN_ANL_IDX(_SV, _SV_SUPERVISOR, "SV_SUPERVISOR", 1); 
    DIN_ANL(_SV, _SV_VBIOS_RELEASE, "SV_VBIOS_RELEASE");
    DIN_ANL(_SV, _SV_VBIOS_ATTENTION, "SV_VBIOS_ATTENTION");

    #ifdef LW_CHIP_DISP_ZPW_ENABLE
    DIN_ANL(_SV, _SV_ZPW_DISABLE, "SV_ZPW_DISABLE");
    DIN_ANL(_SV, _SV_CORE_UPDATE_ARRIVE, "SV_CORE_UPDATE_ARRIVE");
    DIN_ANL(_SV, _SV_CORE_UPDATE_DONE, "SV_CORE_UPDATE_DONE");
    DIN_ANL_IDX(_SV, _SV_BASE_UPDATE_ARRIVE, "SV_BASE_UPDATE_ARRIVE_", 0);
    DIN_ANL_IDX(_SV, _SV_BASE_UPDATE_DONE, "SV_BASE_UPDATE_DONE_", 0);
    DIN_ANL_IDX(_SV, _SV_OVLY_UPDATE_ARRIVE, "SV_OVLY_UPDATE_ARRIVE_", 0);
    DIN_ANL_IDX(_SV, _SV_OVLY_UPDATE_DONE, "SV_OVLY_UPDATE_DONE_", 0);
    #endif
    //DIN_ANL(_SV, _SV_PMU_DIRECT, "SV_PMU_DIRECT");
    DIN_POPU(_SV);
    pen = (DRF_VAL(_PDISP, _DSI_EVENT, _SV_PMU_DIRECT, eve) == LW_PDISP_DSI_EVENT_SV_PMU_DIRECT_PENDING);
    rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,_SV_PMU_DIRECT,rip)==LW_PDISP_DSI_RM_INTR_SV_PMU_DIRECT_PENDING);
    //ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,_SV_PMU_DIRECT,pip)==LW_PDISP_DSI_PMU_INTR_SV_PMU_DIRECT_PENDING);
    ppe=0;
    dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,_SV_PMU_DIRECT,dip)==LW_PDISP_DSI_DPU_INTR_SV_PMU_DIRECT_PENDING);
    ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,_SV_PMU_DIRECT,en0)==LW_PDISP_DSI_RM_INTR_EN0_SV_PMU_DIRECT_ENABLE);
    if(DRF_VAL(_PDISP, _DSI_RM_INTR_MSK, _SV_PMU_DIRECT, rmk))
        tgt = INTR_TGT_RM;
    else if(DRF_VAL(_PDISP, _DSI_DPU_INTR_MSK, _SV_PMU_DIRECT, dmk))
        tgt = INTR_TGT_DPU;
    else
        tgt = INTR_TGT_NONE;
    DIN_PRINT("SV_PMU_DIRECT",ebe,pen,tgt);
    dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));

    //DIN_ANL(_SV, _SV_DPU_DIRECT, "SV_DPU_DIRECT");
    DIN_POPU(_SV);
    pen = (DRF_VAL(_PDISP, _DSI_EVENT, _SV_DPU_DIRECT, eve) == LW_PDISP_DSI_EVENT_SV_DPU_DIRECT_PENDING);
    rpe=(DRF_VAL(_PDISP,_DSI_RM_INTR,_SV_DPU_DIRECT,rip)==LW_PDISP_DSI_RM_INTR_SV_DPU_DIRECT_PENDING);
    ppe=(DRF_VAL(_PDISP,_DSI_PMU_INTR,_SV_DPU_DIRECT,pip)==LW_PDISP_DSI_PMU_INTR_SV_DPU_DIRECT_PENDING);
    //dpe=(DRF_VAL(_PDISP,_DSI_DPU_INTR,_SV_DPU_DIRECT,dip)==LW_PDISP_DSI_DPU_INTR_SV_DPU_DIRECT__PENDING);
    dpe=0;
    ebe=(DRF_VAL(_PDISP,_DSI_RM_INTR_EN0,_SV_DPU_DIRECT,en0)==LW_PDISP_DSI_RM_INTR_EN0_SV_DPU_DIRECT_ENABLE);
    if(DRF_VAL(_PDISP, _DSI_RM_INTR_MSK, _SV_DPU_DIRECT, rmk))
        tgt = INTR_TGT_RM;
    else if(DRF_VAL(_PDISP, _DSI_PMU_INTR_MSK, _SV_DPU_DIRECT, pmk))
        tgt = INTR_TGT_PMU;
    else
        tgt = INTR_TGT_NONE;
    DIN_PRINT("SV_DPU_DIRECT",ebe,pen,tgt);
    dprintf("%s\n",DIN_SANITY(pen, tgt, rpe, ppe, dpe));

    DIN_ANL(_SV, _SV_TIMEOUT, "SV_TIMEOUT");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    while(hind < LW_PDISP_DSI_EVENT_HEAD__SIZE_1)
    {
        if(!(DRF_VAL(_PDISP, _DSI_SYS_CAP, _HEAD_EXISTS(hind), GPU_REG_RD32(LW_PDISP_DSI_SYS_CAP)) ==
               LW_PDISP_DSI_SYS_CAP_HEAD_EXISTS_YES))
        {
               dprintf("HEAD%d doesn't exist\n",hind);
               ++hind;
               continue;
        }
        dprintf("----------------------------------------------------HEAD%d-----------------------------------------------\n",hind);
        DIN_ANL(_HEAD(hind), _HEAD_VBLANK, "HEAD_VBLANK");
        DIN_ANL(_HEAD(hind), _HEAD_HBLANK, "HEAD_HBLANK");

        #ifdef LW_CHIP_DISP_PBUF_LARGE_LATENCY_BUFFER
        DIN_ANL(_HEAD(hind), _HEAD_PBUF_UFLOW, "HEAD_PBUF_UFLOW");
        DIN_ANL(_HEAD(hind), _HEAD_PBUF_UNRECOVERABLE_UFLOW, "HEAD_PBUF_UNRECOVERABLE_UFLOW");
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_RG_UNDERFLOW, "HEAD_RG_UNDERFLOW");

        #ifdef LW_CHIP_DISP_LWDPS_1_5
        DIN_ANL(_HEAD(hind), _HEAD_LWDDS_STATISTIC_COUNTERS_MSB_SET, "HEAD_LWDDS_STATISTIC_COUNTERS_MSB_SET");
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER, "HEAD_LWDDS_STATISTIC_GATHER",0);
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER_UPPER_BOUND, "HEAD_LWDDS_STATISTIC_GATHER_UPPER_BOUND",0);
        DIN_ANL_IDX(_HEAD(hind), _HEAD_LWDDS_STATISTIC_GATHER_LOWER_BOUND, "HEAD_LWDDS_STATISTIC_GATHER_LOWER_BOUND",0);
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_SD3_BUCKET_WALK_DONE, "HEAD_SD3_BUCKET_WALK_DONE");
        DIN_ANL(_HEAD(hind), _HEAD_RG_VBLANK, "HEAD_RG_VBLANK");
        #ifdef LW_CHIP_DISP_ZPW_ENABLE
        DIN_ANL(_HEAD(hind), _HEAD_RG_ZPW_CRC_ERROR, "HEAD_RG_ZPW_CRC_ERROR");
        #endif
        DIN_ANL(_HEAD(hind), _HEAD_PMU_DMI_LINE, "HEAD_PMU_DMI_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_PMU_RG_LINE, "HEAD_PMU_RG_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_RM_DMI_LINE, "HEAD_RM_DMI_LINE");
        DIN_ANL(_HEAD(hind), _HEAD_RM_RG_LINE, "HEAD_RM_RG_LINE");
        dprintf("--------------------------------------------------------------------------------------------------------\n");
        ++hind;
    }
}
                           
typedef struct
{
    LwU32 exists;
    LwU32 active;
    struct sc
    {
        LwU32 rasterWidth;
        LwU32 rasterHeight;
        LwU32 rasterSyncEndX;
        LwU32 rasterSyncEndY;
        LwU32 rasterBlankEndX;
        LwU32 rasterBlankEndY;
        LwU32 rasterBlankStartX;
        LwU32 rasterBlankStartY;
        LwU32 rasterVertBlank2Ystart;
        LwU32 rasterVertBlank2Yend;
        LwU32 pclkHz;
        LwU32 buffWidth;
        LwU32 buffHeight;
        struct vp
        {
            LwU32 vpHeight;
            LwU32 vpWidth;
            LwU32 vpPosX;
            LwU32 vpPosY;
        }vp[2];
        LwU32 hactive;
        LwU32 hfporch;
        LwU32 hbporch;
        LwU32 hsync;
        LwU32 vactive;
        LwU32 vfporch;
        LwU32 vbporch;
        LwU32 vsync;
    }armd, assy;
}dTimings;

#define GETARM(r,f,idx,chan) DRF_VAL(917D, _HEAD, r##f, GPU_REG_RD32(                       \
                             LW_UDISP_DSI_CHN_ARMED_BASEADR(chan)+LW917D_HEAD##r(idx)))
#define GETASY(r,f,idx,chan) DRF_VAL(917D, _HEAD, r##f, GPU_REG_RD32(                       \
                             LW_UDISP_DSI_CHN_ASSY_BASEADR(chan)+LW917D_HEAD##r(idx)))
#define PRINTREC(name, var)  dprintf("%-22s |",name);                                     \
                             for(head=0;head<numHead;head++)                     \
                               {if(!pMyTimings[head].exists)continue;                    \
                                dprintf("%10d, %-10d|",pMyTimings[head].armd.var,        \
                                pMyTimings[head].assy.var);}                             \
                             dprintf("\n");
#define PRINTRECN(na,v,x,y)  dprintf("%-20s",na);                                       \
                             for(head=0;head<numHead;head++)                     \
                             {if(!pMyTimings[head].exists)continue;                      \
                              dprintf("(%d,%d)    %-7s(%d,%d)%-15s",pMyTimings[head].    \
                                            armd.v##x,pMyTimings[head].armd.v##y,        \
                                            "|",pMyTimings[head].assy.v##x, pMyTimings    \
                                             [head].assy.v##y,"");}                     \
                             dprintf("\n");
void dispTimings_v02_01(void)
{
    LwU32 head=0, numHead=0;
    dTimings *pMyTimings;

    numHead = pDisp[indexGpu].dispGetNumHeads();
    pMyTimings = (dTimings*)malloc(sizeof(dTimings) * numHead);

    if (pMyTimings == NULL)
    {
        dprintf("lw: %s - malloc failed!\n", __FUNCTION__);
        return;
    }

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Display Timings                                                 \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    dprintf("%-22s |","Name (Armed , Assy)");
    for (head = 0; head < numHead; ++head)    
    {
        if(!(DRF_VAL(_PDISP, _DSI_SYS_CAP, _HEAD_EXISTS(head), GPU_REG_RD32(LW_PDISP_DSI_SYS_CAP)) ==
               LW_PDISP_DSI_SYS_CAP_HEAD_EXISTS_YES))
        {
               pMyTimings[head].exists = 0;
               continue;
        }
        dprintf("%s%d","HEAD",head);
        pMyTimings[head].exists = 1;
        pMyTimings[head].active = DRF_VAL(_PDISP, _DSI_CORE_HEAD_STATE, _OPERATING_MODE,
                                              GPU_REG_RD32(LW_PDISP_DSI_CORE_HEAD_STATE(head)));
        if(pMyTimings[head].active)
            dprintf("%-19s"," (Active)       | ");
        else
            dprintf("%-19s"," (Not Active)   | ");

        //Raster size
        pMyTimings[head].armd.rasterWidth = GETARM(_SET_RASTER_SIZE, _WIDTH, head, 0);
        pMyTimings[head].armd.rasterHeight = GETARM(_SET_RASTER_SIZE, _HEIGHT, head, 0);
        pMyTimings[head].assy.rasterWidth = GETASY(_SET_RASTER_SIZE, _WIDTH, head, 0);
        pMyTimings[head].assy.rasterHeight = GETASY(_SET_RASTER_SIZE, _HEIGHT, head, 0);

        //Raster Sync End
        pMyTimings[head].armd.rasterSyncEndX = GETARM(_SET_RASTER_SYNC_END, _X, head, 0);
        pMyTimings[head].armd.rasterSyncEndY = GETARM(_SET_RASTER_SYNC_END, _Y, head, 0);
        pMyTimings[head].assy.rasterSyncEndX = GETASY(_SET_RASTER_SYNC_END, _X, head, 0);
        pMyTimings[head].assy.rasterSyncEndY = GETASY(_SET_RASTER_SYNC_END, _Y, head, 0);

        //Raster Blank End
        pMyTimings[head].armd.rasterBlankEndX = GETARM(_SET_RASTER_BLANK_END, _X, head, 0);
        pMyTimings[head].armd.rasterBlankEndY = GETARM(_SET_RASTER_BLANK_END, _Y, head, 0);
        pMyTimings[head].assy.rasterBlankEndX = GETASY(_SET_RASTER_BLANK_END, _X, head, 0);
        pMyTimings[head].assy.rasterBlankEndY = GETASY(_SET_RASTER_BLANK_END, _Y, head, 0);
                
        //Raster Blank Start 
        pMyTimings[head].armd.rasterBlankStartX = GETARM(_SET_RASTER_BLANK_START, _X, head, 0);
        pMyTimings[head].armd.rasterBlankStartY = GETARM(_SET_RASTER_BLANK_START, _Y, head, 0);
        pMyTimings[head].assy.rasterBlankStartX = GETASY(_SET_RASTER_BLANK_START, _X, head, 0);
        pMyTimings[head].assy.rasterBlankStartY = GETASY(_SET_RASTER_BLANK_START, _Y, head, 0);

        //Raster Vert Blank 
        pMyTimings[head].armd.rasterVertBlank2Ystart=GETARM(_SET_RASTER_VERT_BLANK2,_YSTART, head, 0);
        pMyTimings[head].armd.rasterVertBlank2Yend = GETARM(_SET_RASTER_VERT_BLANK2, _YEND, head, 0);
        pMyTimings[head].assy.rasterVertBlank2Ystart=GETASY(_SET_RASTER_VERT_BLANK2,_YSTART, head, 0);
        pMyTimings[head].assy.rasterVertBlank2Yend = GETASY(_SET_RASTER_VERT_BLANK2, _YEND, head, 0);

        //Pclk
        pMyTimings[head].armd.pclkHz = GETARM(_SET_PIXEL_CLOCK_FREQUENCY, _HERTZ, head, 0); 
        pMyTimings[head].assy.pclkHz = GETASY(_SET_PIXEL_CLOCK_FREQUENCY, _HERTZ, head, 0); 
 
        //Buff size
        pMyTimings[head].armd.buffWidth = GETARM(_SET_SIZE, _WIDTH, head, 0);
        pMyTimings[head].armd.buffHeight = GETARM(_SET_SIZE, _HEIGHT, head, 0);
        pMyTimings[head].assy.buffWidth = GETASY(_SET_SIZE, _WIDTH, head, 0);
        pMyTimings[head].assy.buffHeight = GETASY(_SET_SIZE, _HEIGHT, head, 0);

        //Vp[0]=vpin, vp[1]=vpout
      
        pMyTimings[head].armd.vp[0].vpHeight = GETARM(_SET_VIEWPORT_SIZE_IN, _HEIGHT, head, 0);
        pMyTimings[head].armd.vp[0].vpWidth = GETARM(_SET_VIEWPORT_SIZE_IN, _WIDTH, head, 0);
        pMyTimings[head].armd.vp[0].vpPosX = GETARM(_SET_VIEWPORT_POINT_IN, _X, head, 0);
        pMyTimings[head].armd.vp[0].vpPosY = GETARM(_SET_VIEWPORT_POINT_IN, _Y, head, 0);

        pMyTimings[head].assy.vp[0].vpHeight = GETASY(_SET_VIEWPORT_SIZE_IN, _HEIGHT, head, 0);
        pMyTimings[head].assy.vp[0].vpWidth = GETASY(_SET_VIEWPORT_SIZE_IN, _WIDTH, head, 0);
        pMyTimings[head].assy.vp[0].vpPosX = GETASY(_SET_VIEWPORT_POINT_IN, _X, head, 0);
        pMyTimings[head].assy.vp[0].vpPosY = GETASY(_SET_VIEWPORT_POINT_IN, _Y, head, 0);

        pMyTimings[head].armd.vp[1].vpHeight = GETARM(_SET_VIEWPORT_SIZE_OUT, _HEIGHT, head, 0);
        pMyTimings[head].armd.vp[1].vpWidth = GETARM(_SET_VIEWPORT_SIZE_OUT, _WIDTH, head, 0);
        pMyTimings[head].armd.vp[1].vpPosX = GETARM(_SET_VIEWPORT_POINT_OUT_ADJUST, _X, head, 0);
        pMyTimings[head].armd.vp[1].vpPosY = GETARM(_SET_VIEWPORT_POINT_OUT_ADJUST, _Y, head, 0);

        pMyTimings[head].assy.vp[1].vpHeight = GETASY(_SET_VIEWPORT_SIZE_OUT, _HEIGHT, head, 0);
        pMyTimings[head].assy.vp[1].vpWidth = GETASY(_SET_VIEWPORT_SIZE_OUT, _WIDTH, head, 0);
        pMyTimings[head].assy.vp[1].vpPosX = GETASY(_SET_VIEWPORT_POINT_OUT_ADJUST, _X, head, 0);
        pMyTimings[head].assy.vp[1].vpPosY = GETASY(_SET_VIEWPORT_POINT_OUT_ADJUST, _Y, head, 0);
        
        pMyTimings[head].armd.hfporch = pMyTimings[head].armd.rasterWidth - 1 - pMyTimings[head].armd.rasterBlankStartX;
        pMyTimings[head].armd.hbporch = pMyTimings[head].armd.rasterBlankEndX - pMyTimings[head].armd.rasterSyncEndX; 
        pMyTimings[head].armd.hactive = pMyTimings[head].armd.rasterBlankStartX - pMyTimings[head].armd.rasterBlankEndX;
        pMyTimings[head].armd.hsync   = pMyTimings[head].armd.rasterSyncEndX+1;
        pMyTimings[head].assy.hfporch = pMyTimings[head].assy.rasterWidth - 1 -pMyTimings[head].assy.rasterBlankStartX;
        pMyTimings[head].assy.hbporch = pMyTimings[head].assy.rasterBlankEndX - pMyTimings[head].assy.rasterSyncEndX; 
        pMyTimings[head].assy.hactive = pMyTimings[head].assy.rasterBlankStartX - pMyTimings[head].assy.rasterBlankEndX;
        pMyTimings[head].assy.hsync   = pMyTimings[head].assy.rasterSyncEndX+1;

        pMyTimings[head].armd.vfporch = pMyTimings[head].armd.rasterHeight -1 - pMyTimings[head].armd.rasterBlankStartY;
        pMyTimings[head].armd.vbporch = pMyTimings[head].armd.rasterBlankEndY - pMyTimings[head].armd.rasterSyncEndY; 
        pMyTimings[head].armd.vactive = pMyTimings[head].armd.rasterBlankStartY - pMyTimings[head].armd.rasterBlankEndY;
        pMyTimings[head].armd.vsync   = pMyTimings[head].armd.rasterSyncEndY+1;
        pMyTimings[head].assy.vfporch = pMyTimings[head].assy.rasterHeight - 1 -pMyTimings[head].assy.rasterBlankStartY;
        pMyTimings[head].assy.vbporch = pMyTimings[head].assy.rasterBlankEndY - pMyTimings[head].assy.rasterSyncEndY; 
        pMyTimings[head].assy.vactive = pMyTimings[head].assy.rasterBlankStartY - pMyTimings[head].assy.rasterBlankEndY;
        pMyTimings[head].assy.vsync   = pMyTimings[head].assy.rasterSyncEndY+1;
    }
    dprintf("\n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("Hvisible",hactive);
    PRINTREC("HFront Porch",hfporch);
    PRINTREC("HBack Porch",hbporch);
    PRINTREC("HSync",hsync);
    PRINTREC("Vvisible",vactive);
    PRINTREC("VFront Porch",vfporch);
    PRINTREC("VBack Porch",vbporch);
    PRINTREC("VSync",vsync);
    PRINTREC("Pixel Clock (Hz)",pclkHz);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("Buffer Height",buffHeight);
    PRINTREC("Buffer Width",buffWidth);
    PRINTREC("VPIN Height",vp[0].vpHeight);
    PRINTREC("VPIN Width",vp[0].vpWidth);
    PRINTREC("VPIN PosX",vp[0].vpPosX);
    PRINTREC("VPIN PosY",vp[0].vpPosY);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    PRINTREC("VPOUT Height",vp[1].vpHeight);
    PRINTREC("VPOUT Width",vp[1].vpWidth);
    PRINTREC("VPOUT PosX",vp[1].vpPosX);
    PRINTREC("VPOUT PosY",vp[1].vpPosY);
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("\n");

    free(pMyTimings);
}


void dispCtxDmaDescription_v02_01(LwU32 handle, LwS32 chanNum, BOOL searchAllHandles)
{
    LwU32 dsiInstMem0;
    LwU32 i, clientID, instance, objMemOffset;
    HASH_TABLE_ENTRY htEntry[(LW_UDISP_HASH_LIMIT - LW_UDISP_HASH_BASE + 1)/sizeof(HASH_TABLE_ENTRY)]; 
    DMAOBJECT dmaObj;
    PhysAddr dsiInstMemAddr, dispHTAddr, objMemStart, tmpAddr;
    BOOL isPhysLwm = FALSE;
    BOOL searchAllChannels = (chanNum  == -1) ? TRUE : FALSE;
    BOOL handleFound = FALSE;
    LW_STATUS status;

    // Read the base address of the display instance memory
    dsiInstMem0 = GPU_REG_RD32(LW_PDISP_DSI_INST_MEM0);

    // Check the status bit to know if the loaded address is valid.
    if (FLD_TEST_DRF(_PDISP, _DSI_INST_MEM0, _STATUS, _ILWALID, dsiInstMem0))
    {
        dprintf("Handle couldn't be found in the Hash Table because instance memory in invalid.\n");
        return;
    }

    // obtain the starting address of display instance memory.
    dsiInstMemAddr = DRF_VAL(_PDISP, _DSI_INST_MEM0, _ADDR, dsiInstMem0);
    dsiInstMemAddr <<= 16;

    // obtain the base address of the display hash table
    dispHTAddr = dsiInstMemAddr + LW_UDISP_HASH_BASE;

    if (FLD_TEST_DRF(_PDISP, _DSI_INST_MEM0, _TARGET, _PHYS_LWM, dsiInstMem0))
    {
        isPhysLwm = TRUE;
    }

    if (isPhysLwm)
        status = pFb[indexGpu].fbRead(dispHTAddr, &htEntry, sizeof(htEntry));
    else
        status = readSystem(dispHTAddr, &htEntry, sizeof(htEntry));

    if (status != LW_OK)
    {
        dprintf("Failed to read hash table memory.\n");
        return;
    }

    for (i = 0 ; i < ((LW_UDISP_HASH_LIMIT - LW_UDISP_HASH_BASE + 1)/sizeof(HASH_TABLE_ENTRY)) ; i++)
    {
        instance = DRF_VAL(_UDISP, _HASH_TBL, _INSTANCE, htEntry[i].data);
        if (instance == LW_UDISP_HASH_TBL_INSTANCE_ILWALID)
        {
            continue;
        }

        if (searchAllChannels || (chanNum == (LwS32)(DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data))))
        {
            chanNum = DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data);
            if (((LwU32)chanNum) >= pDisp[indexGpu].dispGetMaxChan())
            {
                continue;
            }

            //
            // Channels where we can't use debug port to send methods
            // don't have any ctx dmas
            // XXXDISP: Fix this assumption in future.
            //
            if (!(dispChanState_v02_01[chanNum].cap & DISP_DEBUG))
            {
                continue;
            }

            if (searchAllHandles || (handle == DRF_VAL(_UDISP, _HASH_TBL, _HANDLE, htEntry[i].handle)))
            {
                handle = DRF_VAL(_UDISP, _HASH_TBL, _HANDLE, htEntry[i].handle);
                chanNum = DRF_VAL(_UDISP, _HASH_TBL, _CHN, htEntry[i].data);
                clientID = DRF_VAL(_UDISP, _HASH_TBL, _CLIENT_ID, htEntry[i].data);

                if (handle == 0)
                {
                    continue;
                }
                handleFound = TRUE;

                dprintf("===============================================================================================\n");
                dprintf("%-30s: " PhysAddr_FMT " in %s\n", "DSI_INST_MEM0", dsiInstMemAddr, isPhysLwm ? "VIDMEM" : "SYSMEM");
                dprintf("%-30s: 0x%x\n", "Handle", handle);
                dprintf("%-30s: %s(%d)\n", "Channel", dispChanState_v02_01[chanNum].name, dispChanState_v02_01[chanNum].headNum);
                dprintf("%-30s: 0x%x\n", "Client ID", clientID);

                // All objects are allocated in chunks of 32 bytes, the offset read is multiplied by 32 to get the actual OBJ_MEM offset.
                objMemOffset = instance*32;
                dprintf("%-30s: 0x%x bytes\n", "Offset from base of HT", objMemOffset);
                objMemStart = dispHTAddr + objMemOffset;
                dprintf("%-30s: " PhysAddr_FMT " in %s\n", "Object Address", objMemStart, isPhysLwm ? "VIDMEM" : "SYSMEM");

                if (isPhysLwm)
                    status = pFb[indexGpu].fbRead(objMemStart, &dmaObj, sizeof(dmaObj));
                else
                    status = readSystem(objMemStart, &dmaObj, sizeof(dmaObj));

                if (status != LW_OK)
                {
                    dprintf("Failed to read hash table entry.\n");
                    return;
                }

                switch (DRF_VAL(_DMA, _TARGET, _NODE, dmaObj.classNum))
                {
                    case LW_DMA_TARGET_NODE_VIRTUAL:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "VIRTUAL");
                        break;
                    case LW_DMA_TARGET_NODE_PHYSICAL_LWM:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_LWM");
                        break;
                    case LW_DMA_TARGET_NODE_PHYSICAL_PCI:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_PCI");
                        break;
                    case LW_DMA_TARGET_NODE_PHYSICAL_PCI_COHERENT:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "PHYSICAL_PCI_COHERENT");
                        break;
                    default:
                        dprintf("%-30s: %s\n", "LW_DMA_TARGET_NODE", "UNKNOWN");
                }

                switch (DR_VAL(_DMA, _ACCESS, dmaObj.classNum))
                {
                    case LW_DMA_ACCESS_FROM_PTE:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "FROM_PTE");
                        break;
                    case LW_DMA_ACCESS_READ_ONLY:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "READ_ONLY");
                        break;
                    case LW_DMA_ACCESS_READ_AND_WRITE:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "READ_AND_WRITE");
                        break;
                    default:
                        dprintf("%-30s: %s\n", "LW_DMA_ACCESS", "UNKNOWN");
                }

                switch (DR_VAL(_DMA, _ENCRYPTED, dmaObj.classNum))
                {
                    case LW_DMA_ENCRYPTED_FALSE:
                        dprintf("%-30s: %s\n", "LW_DMA_ENCRYPTED", "FALSE");
                        break;
                    case LW_DMA_ENCRYPTED_TRUE:
                        dprintf("%-30s: %s\n", "LW_DMA_ENCRYPTED", "TRUE");
                        break;
                    case LW_DMA_ENCRYPTED_FROM_PTE:
                        dprintf("%-30s: %s\n", "LW_DMA_ENCRYPTED", "FROM_PTE");
                        break;
                    default:
                        dprintf("%-30s: %s\n", "LW_DMA_ENCRYPTED", "UNKNOWN");
                }

                dprintf("%-30s: %s\n", "LW_DMA_PAGE_SIZE", (DR_VAL(_DMA, _PAGE_SIZE, dmaObj.classNum) == LW_DMA_PAGE_SIZE_SMALL)? "SMALL" : "BIG");
                dprintf("%-30s: 0x%x (See LW_MMU_PTE_KIND* for decoding)\n", "LW_DMA_KIND", DR_VAL(_DMA, _KIND, dmaObj.classNum));

                tmpAddr = DRF_VAL(_DMA, _ADDRESS,     _BASE,  dmaObj.limitLo);
                dprintf("%-30s: " PhysAddr_FMT " bytes\n", "LW_DMA_ADDRESS_BASE",     tmpAddr << 8);

                tmpAddr = DRF_VAL(_DMA, _ADDRESS,     _LIMIT, dmaObj.adjust);
                dprintf("%-30s: " PhysAddr_FMT " bytes\n", "LW_DMA_ADDRESS_LIMIT",    tmpAddr  << 8);
                dprintf("%-30s: 0x%x\n",       "LW_DMA_TAGS_BASE",        DRF_VAL(_DMA, _TAGS,        _BASE,  dmaObj.limitHi));
                dprintf("%-30s: 0x%x\n",       "LW_DMA_TAGS_SIZE",        DRF_VAL(_DMA, _TAGS,        _SIZE,  dmaObj.tags));
                dprintf("%-30s: 0x%x\n",       "LW_DMA_TAGS_OFFSET_PHYS", DRF_VAL(_DMA, _TAGS_OFFSET, _PHYS,  dmaObj.partStride));
            }
        }
    }

    if ( (handleFound == FALSE) && (searchAllHandles == FALSE) )
    {
        if (searchAllChannels)
        {
            dprintf("Queried handle 0x%x could not be found in the Hash Table!\n", handle);
        }
        else
        {
            dprintf("Queried handle 0x%x could not be found in channel number %d in the Hash Table!\n", handle, chanNum);
        }
    }
}

/*!
 * @brief dispDumpSLIConfig - Function to dump SLI Config Data. 
 * It dumps SLI register values & print results related to GPU 
 * configuration 
 *  
 * @param[in] LwU32      Verbose - Switch to enable extra information 
 *                                  logging. 
 */
void dispDumpSliConfig_v02_01
(
    LwU32 verbose
)
{
    LwU32               head = 0, numHead = 0;
    LwU32               pior = 0, numPior = 0;
    LwU32               pin = 0,  numPin = LW_PDISP_DSI_LOCK_PIN_CAPA_LOCK_PIN_USAGE__SIZE_1;
    DSLI_DATA           *pDsliData;
    DSLI_PIOR_DATA      *pDsliPiorData;
    DSLI_PRINT_PARAM    *pDsliPrintData;
    
    // Find out total number of heads
    numHead = pDisp[indexGpu].dispGetNumHeads();

    // Find number of Piors
    numPior = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

    pDsliData = (DSLI_DATA*)malloc(sizeof(DSLI_DATA) * numHead);
    pDsliPiorData = (DSLI_PIOR_DATA*)malloc(sizeof(DSLI_PIOR_DATA));
    pDsliPrintData = (DSLI_PRINT_PARAM*)malloc(sizeof(DSLI_PRINT_PARAM) * numHead);

    if ((pDsliData == NULL) || (pDsliPiorData == NULL) || (pDsliPrintData == NULL))
    {
        dprintf("lw: %s - malloc failed!\n", __FUNCTION__);
        dprintf("lw: Failed Pointers : DSLI_DATA-%p, DSLI_PIOR_DATA-%p, DSLI_PRINT_PARAM-%p\n", \
                pDsliData, pDsliPiorData, pDsliPrintData);
        free(pDsliData);
        free(pDsliPiorData);
        free(pDsliPrintData);
        return;
    }

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Display SLI Configuration                                                 \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    for (head = 0; head < numHead; ++head)
    {
        // Make all heads inactive initially
        pDsliData[head].DsliHeadActive = FALSE;
    
        // Initialize the print parameters
        dispInitializeSliData(&pDsliPrintData[head]);
        
        // Function to get all SLI configuration
        pDisp[indexGpu].dispGetSliData(head, pDsliData);
    }

    pDsliPiorData->DsliCap = GPU_REG_RD32(LW_PDISP_DSI_LOCK_PIN_CAPA);

    for (pior = 0; pior < numPior; ++pior)
    {
       pDisp[indexGpu].dispGetPiorData(pior, pDsliData, pDsliPiorData);
    }

    for (pin = 0; pin < numPin; ++pin)
    {
        pDsliPiorData->DsliCapLockPinUsage[pin] = DRF_IDX_VAL(_PDISP, _DSI_LOCK_PIN_CAPA, _LOCK_PIN_USAGE, (LwU32)pin, pDsliPiorData->DsliCap);
    }
    
    // Get the clock data
    pDisp[indexGpu].dispGetClockData(pDsliData);

    // Call the print function to print configuration results 
    dispPrintSliData(numHead, numPior, numPin, pDsliData, pDsliPiorData, pDsliPrintData, verbose);

    free(pDsliData);
    free(pDsliPiorData);
    free(pDsliPrintData);
}

/*!
 * @brief Helper function to retrun SLI Data.
 * 
 * @param[in] head       HEAD number
 * @param[in] pDsliData  DSLI_DATA pointer - For filling Data in
 * datastructure
 */
void dispGetSliData_v02_01
(
    LwU32 head, 
    DSLI_DATA *pDsliData
)
{
    pDsliData[head].DsliRgDistRndr = GPU_REG_RD32(LW_PDISP_RG_DIST_RNDR(head));
    pDsliData[head].DsliRgDistRndrSyncAdv = DRF_VAL(_PDISP, _RG_DIST_RNDR, _SYNC_ADVANCE, pDsliData[head].DsliRgDistRndr);
    pDsliData[head].DsliRgFlipLock = GPU_REG_RD32(LW_PDISP_RG_FLIPLOCK(head));
    pDsliData[head].DsliRgStatus = GPU_REG_RD32(LW_PDISP_RG_STATUS(head));
    pDsliData[head].DsliRgStatusLocked = DRF_VAL(_PDISP, _RG_STATUS, _LOCKED, pDsliData[head].DsliRgStatus);
    pDsliData[head].DsliRgStatusFlipLocked = DRF_VAL(_PDISP, _RG_STATUS, _FLIPLOCKED, pDsliData[head].DsliRgStatus);
    pDsliData[head].DsliClkRemVpllExtRef = GPU_REG_RD32(LW_PDISP_CLK_REM_VPLL_EXT_REF_CONFIG(head));
    pDsliData[head].DsliClkDriverSrc = DRF_VAL(_PDISP, _CLK_REM_VPLL_EXT_REF_CONFIG, _SRC, pDsliData[head].DsliClkRemVpllExtRef);
    pDsliData[head].DsliHeadSetCntrl = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) + LW917D_HEAD_SET_CONTROL(head));
    pDsliData[head].DsliHeadSetSlaveLockMode = DRF_VAL(917D, _HEAD_SET_CONTROL, _SLAVE_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockMode = DRF_VAL(917D, _HEAD_SET_CONTROL, _MASTER_LOCK_MODE, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetSlaveLockPin = DRF_VAL(917D, _HEAD_SET_CONTROL, _SLAVE_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
    pDsliData[head].DsliHeadSetMasterLockPin = DRF_VAL(917D, _HEAD_SET_CONTROL, _MASTER_LOCK_PIN, pDsliData[head].DsliHeadSetCntrl);
}

/*!
 * @brief Helper function to return Clock Data for SLI.
 * 
 * @param[in] pDsliData     DSLI_DATA pointer - For filling Data in
 *                          datastructure
 */
void dispGetClockData_v02_01
(
    DSLI_DATA *pDsliData
)
{
    LwU32 head = 0, numHead = 0;

    // Find out total number of heads
    numHead = pDisp[indexGpu].dispGetNumHeads();

    for (head = 0; head < numHead; ++head)
    {
        pDsliData[head].DsliPvTrimSysVClkRefSwitch = GPU_REG_RD32(LW_PVTRIM_SYS_VCLK_REF_SWITCH(head));
        pDisp[indexGpu].dispGetClockRegBnkInfo(head, pDsliData);
        pDsliData[head].DsliQualStatus = DRF_VAL(_PVTRIM, _SYS_VCLK_REF_SWITCH, _EXT_REFCLK, \
                                                          pDsliData[head].DsliPvTrimSysVClkRefSwitch);
    }
}

/*This is added to resolve register collision without increasing redundancy of code*/
/*!
 * @brief Helper function to return Clock register banking data
 * 
 * @param[in] head          HEAD number
 * @param[in] pDsliData     DSLI_DATA pointer - For filling Data in
 *                          datastructure
 */
void dispGetClockRegBnkInfo_v02_01
(
    LwU32 head,
    DSLI_DATA *pDsliData
)
{
    pDsliData[head].DsliVclkRefSwitchFinalSel = DRF_VAL(_PVTRIM, _SYS_VCLK_REF_SWITCH, _FINALSEL, \
                                                          pDsliData[head].DsliPvTrimSysVClkRefSwitch);
    pDsliData[head].DsliSlowClk = DRF_VAL(_PVTRIM, _SYS_VCLK_REF_SWITCH, _SLOWCLK, \
                                                          pDsliData[head].DsliPvTrimSysVClkRefSwitch);
    pDsliData[head].DsliMisCclk = DRF_VAL(_PVTRIM, _SYS_VCLK_REF_SWITCH, _MISCCLK, \
                                                          pDsliData[head].DsliPvTrimSysVClkRefSwitch);
}

/*!
 * @brief Helper function to return PIOR Data for SLI.
 *  
 * @param[in] pior              PIOR number 
 * @param[in] pDsliData         DSLI_DATA pointer - For filling Data in
 * datastructure 
 * @param[in] pDsliPiorData     DSLI_PIOR_DATA pointer - For filling Data in
 *                              datastructure
 */
void dispGetPiorData_v02_01
(
    LwU32 pior, DSLI_DATA *pDsliData, 
    DSLI_PIOR_DATA *pDsliPiorData
)
{
    HEAD                orOwner;
    ORPROTOCOL          orProtocol;
    char                *orStr = dispGetORString(LW_OR_PIOR);

    pDsliPiorData->DsliPiorDro[pior] = GPU_REG_RD32(LW_PDISP_PIOR_DRO(pior));
    pDisp[indexGpu].dispReadOrOwnerAndProtocol(LW_OR_PIOR, pior, &orOwner, &orProtocol);

    if (orOwner >= HEAD(pDisp[indexGpu].dispGetNumHeads()))
    {
        dprintf("\n");
        dprintf("lw: %s ERROR: Bad owner %d for %s_%d\n", __FUNCTION__, orOwner, orStr, pior);
        return;
    }

    // Get the Owner head number & save it
    pDsliPiorData->DsliVgaPiorCntrlOwner[pior] = ((orOwner == HEAD_UNSPECIFIED ) ? OR_OWNER_UNDEFINED : HEAD_IDX(orOwner));
    
    if (!(pDsliPiorData->DsliVgaPiorCntrlOwner[pior] == OR_OWNER_UNDEFINED))
    {
        // Set the Head as active, if returned owner is valid (Head 0/1)
        pDsliData[HEAD_IDX(orOwner)].DsliHeadActive = TRUE;
    }
    
    // Get the Owner Protocol & Save it
    pDsliPiorData->DsliVgaPiorCntrlProtocol[pior] = dispGetStringForOrProtocol(LW_OR_PIOR, orProtocol);
}

/*!
 * @brief dispPrintClkData - Function to print SLI-OR config data, 
 * used by DSLI. It prints SLI register values for configuration 
 *
 *  @param[in]  LwU32               head            Head Number
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM    *pDsliPrintData  Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Verbose switch
 */

void dispPrintClkData_v02_01
(
    LwU32               head,
    DSLI_DATA           *pDsliData,
    DSLI_PRINT_PARAM    *pDsliPrintData,
    LwU32               verbose
)
{
    switch(pDsliData[head].DsliVclkRefSwitchFinalSel)
    {
        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_SLOWCLK:
            switch(pDsliData[head].DsliSlowClk)
            {
                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_XTAL_IN:
                    pDsliPrintData[head].refClkForVpll = "XTAL";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_XTALS_IN:
                    pDsliPrintData[head].refClkForVpll = "XTALS";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_SWCLK:
                    pDsliPrintData[head].refClkForVpll = "SWCLK";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_SLOWCLK_XTAL4X:
                    pDsliPrintData[head].refClkForVpll = "4X-XTAL";
                    break;
            }
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_TESTORJTAGCLK:
            pDsliPrintData[head].refClkForVpll = "Test-Jtag";
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_MISCCLK:
            switch(pDsliData[head].DsliMisCclk)
            {
                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCCLK_PEX_REFCLK_FAST:
                    pDsliPrintData[head].refClkForVpll = "PEX-REF";
                    break;

                case LW_PVTRIM_SYS_VCLK_REF_SWITCH_MISCCLK_EXT_REFCLK:
                    switch(pDsliData[head].DsliClkDriverSrc)
                    {
                        case EXT_REFCLK_SRCA:
                            pDsliPrintData[head].refClkForVpll = "EXT-Ref-Clock-A";
                            break;

                        case EXT_REFCLK_SRCB:
                            pDsliPrintData[head].refClkForVpll = "EXT-Ref-Clock-B";
                            break;
                    }
                    break;
            }
            break;

        case LW_PVTRIM_SYS_VCLK_REF_SWITCH_FINALSEL_ONESRCCLK:
            pDsliPrintData[head].refClkForVpll = "ONESRC";
            break;
    }
}

/*!
 * @brief dispPrintHeadData - Function to print SLI-HEAD config data, 
 * used by DSLI. It prints SLI register values for configuration 
 *
 *  @param[in]  LwU32               numHead         Number of Heads
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM    *pDsliPrintData Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Verbose switch
 */
void dispPrintHeadData_v02_01
(
    LwU32               numHead,
    DSLI_DATA           *pDsliData,
    DSLI_PRINT_PARAM    *pDsliPrintData,
    LwU32               verbose
)

{
    LwU32 head  = 0;
    
    for (head = 0; head < numHead; ++head)
    {
        if (pDsliData[head].DsliHeadActive)
        {
            pDsliPrintData[head].headStatus = "Active";
        }

        PRINTLOCKMODE(DsliHeadSetSlaveLockMode, pDsliData, pDsliPrintData, slaveLock)
        PRINTLOCKPIN(DsliHeadSetSlaveLockPin, pDsliData, pDsliPrintData, slaveLockPin)
        PRINTLOCKMODE(DsliHeadSetMasterLockMode, pDsliData, pDsliPrintData, masterLock)
        PRINTLOCKPIN(DsliHeadSetMasterLockPin, pDsliData, pDsliPrintData, masterLockPin)

        switch(pDsliData[head].DsliRgStatusLocked)
        {
            case LW_PDISP_RG_STATUS_LOCKED_TRUE:
                pDsliPrintData[head].scanLockStatus = "Locked";
                break;

            default:
                pDsliPrintData[head].scanLockStatus = "N/A";
                break;
        }

        switch(pDsliData[head].DsliRgStatusFlipLocked)
        {
            case LW_PDISP_RG_STATUS_FLIPLOCKED_TRUE:
                pDsliPrintData[head].flipLock = "Enabled";
                pDsliPrintData[head].flipLockStatus = "Locked";
                break;

            default:
                pDsliPrintData[head].flipLock = "disabled";
                pDsliPrintData[head].flipLockStatus = "N/A";

        }

        pDsliPrintData[head].syncAdvance = pDsliData[head].DsliRgDistRndrSyncAdv;

        // Print clock data for active Head
        pDisp[indexGpu].dispPrintClkData(head, pDsliData, pDsliPrintData, verbose);
    }

    dispPrintSliStatus (numHead, pDsliData, pDsliPrintData, verbose);

    dprintf("--------------------------------------------------------------------------------------------------------\n");
}

/*!
 * @brief dispPrintPinData - Function to print SLI-PIN config data, 
 * used by DSLI. It prints SLI register values for configuration 
 *
 *  @param[in]  LwU32           numPin          Number of Pins
 *  @param[in]  DSLI_PIOR_DATA  *pDsliPiorData  Pointer to PIOR
 *                                              datastructure
 */

void dispPrintPinData_v02_01
(
    LwU32           numPin,
    DSLI_PIOR_DATA  *pDsliPiorData
)
{
    LwU32 pin = 0;
  
    dprintf("                                  LOCK-PIN-CAPABILITIES                                                 \n");
    dprintf("========================================================================================================\n");
    dprintf("PIN#       Lock Usage      Value                       \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    for (pin = 0; pin < numPin; ++pin)
    {
        dprintf("%s-%d", "PIN", pin);
        
        switch (pDsliPiorData->DsliCapLockPinUsage[pin])
        {
            case LW_PDISP_DSI_LOCK_PIN_CAPA_LOCK_PIN0_USAGE_SCAN_LOCK:
                 PRINTPINTABLE("Scan-Lock", pDsliPiorData->DsliCapLockPinUsage[pin]);
                 break;

            case LW_PDISP_DSI_LOCK_PIN_CAPA_LOCK_PIN0_USAGE_FLIP_LOCK:
                 PRINTPINTABLE("Flip-Lock", pDsliPiorData->DsliCapLockPinUsage[pin]);
                 break;

            case LW_PDISP_DSI_LOCK_PIN_CAPA_LOCK_PIN0_USAGE_UNAVAILABLE:
                 PRINTPINTABLE("Unavailable", pDsliPiorData->DsliCapLockPinUsage[pin]);
                 break;

            case LW_PDISP_DSI_LOCK_PIN_CAPA_LOCK_PIN0_USAGE_STEREO:
                 PRINTPINTABLE("Stereo", pDsliPiorData->DsliCapLockPinUsage[pin]);
                 break;
        }
    }

    dprintf("--------------------------------------------------------------------------------------------------------\n");
}

/*!
 *  Function to print configuration of specified displayport.
 *
 *  @param[in]  port        Specified AUX port.
 *  @param[in]  sorIndex    Specified SOR.
 *  @param[in]  dpIndex     Specified sublink.
 */
void dispDisplayPortInfo_v02_01
(
    LwU32 port,
    LwU32 sorIndex,
    LwU32 dpIndex
)
{
    LwU32 reg, headMask;

    if (pDpaux[indexGpu].dpauxGetHpdStatus(port))
    {
        dprintf("================================================================================\n");
        dprintf("%-55s: %s\n\n", "LW_PMGR_DP_AUXSTAT_HPD_STATUS", "PLUGGED");
    }
    else
    {
        dprintf("ERROR: %s: DP not plugged in. Bailing out early\n",
            __FUNCTION__);
        return;
    }

    // Get right SOR & sublink index.
    if (dpIndex == LW_MAX_SUBLINK)
    {
        LwU32 i, j;
        LwU32 link[LW_MAX_SUBLINK];

        for (i = 0; i < LW_MAX_SUBLINK; i++)
            link[i] = PADLINK_NONE;

        for (i = 0; i < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); i++)
        {
            if (pDisp[indexGpu].dispGetLinkBySor(i, link))
            {
                for (j = 0; j < LW_MAX_SUBLINK; j++)
                {
                    if (pDisp[indexGpu].dispGetAuxPortByLink(link[j]) == port)
                    {
                        sorIndex = i;
                        dpIndex = j;
                        break;
                    }
                }
                if (dpIndex != LW_MAX_SUBLINK)
                    break;
            }
        }
        if (dpIndex == LW_MAX_SUBLINK)
        {
            dprintf("ERROR: %s: Can't get corrusponding SOR & sublink.\n",
                __FUNCTION__);
            return;
        }
    }

    dprintf("Tx:\n");
    dprintf("%-55s: %d\n", "sorIndex", sorIndex);
    dprintf("%-55s: %d\n", "dpIndex", dpIndex);

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_ENHANCEDFRAME",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _ENHANCEDFRAME, reg) ?
        "ENABLED" : "DISABLED");

    switch(DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _LANECOUNT, reg))
    {
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ZERO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ZERO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_ONE:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "ONE");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_TWO:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "TWO");
            break;
        case LW_PDISP_SOR_DP_LINKCTL_LANECOUNT_FOUR:
            dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_LANECOUNT",
                "FOUR");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_SOR_DP_LINKCTL_LANECOUNT value.\n",
                __FUNCTION__);
    }

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_LINKCTL_FORMAT_MODE",
        DRF_VAL(_PDISP, _SOR_DP_LINKCTL, _FORMAT_MODE, reg) ?
        "MULTI_STREAM" : "SINGLE_STREAM");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_PADCTL(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_0",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_0, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_1",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_1, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_2",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_2, reg) ?
        "NO" : "YES");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_PADCTL_PD_TXD_3",
        DRF_VAL(_PDISP, _SOR_DP_PADCTL, _PD_TXD_3, reg) ?
        "NO" : "YES");

    reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR(sorIndex));

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_XITION:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DIV",
                "XITION");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE",
                "SAFE");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_1");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_2");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_PLL_REF_DIV",
                "BY_4");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_PLL_REF_DIV value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE_BYPASS, reg))
    {
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_NONE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "NONE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_NORMAL:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_NORMAL");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_SAFE:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "DP_SAFE");
            break;
        case LW_PDISP_CLK_REM_SOR_MODE_BYPASS_FEEDBACK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_MODE_BYPASS",
                "FEEDBACK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_MODE_BYPASS value.\n",
                __FUNCTION__);
    }

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _DP_LINK_SPEED, reg))
    {
        case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_1_62GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED",
                    "1_62GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_2_70GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED",
                    "2_70GHZ");
            break;
        case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_5_40GHZ:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED",
                    "5_40GHZ");
            break;
        default:
            dprintf("WARNING: %-55s: %x\n",
                "LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED",
                DRF_VAL(_PDISP, _CLK_REM_SOR, _DP_LINK_SPEED, reg));
            break;
    }

    dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_STATE",
        DRF_VAL(_PDISP, _CLK_REM_SOR, _STATE, reg) ? "ENABLE" : "DISABLE");

    switch(DRF_VAL(_PDISP, _CLK_REM_SOR, _CLK_SOURCE, reg))
    {
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_PCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_PCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_SINGLE_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "SINGLE_DPCLK");
            break;
        case LW_PDISP_CLK_REM_SOR_CLK_SOURCE_DIFF_DPCLK:
            dprintf("%-55s: %s\n", "LW_PDISP_CLK_REM_SOR_CLK_SOURCE",
                "DIFF_DPCLK");
            break;
        default:
            dprintf("ERROR: %s: Invalid LW_PDISP_CLK_REM_SOR_CLK_SOURCE value.\n",
                __FUNCTION__);
    }

    reg = GPU_REG_RD32(LW_PDISP_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_CAP_DP_INTERFACE",
        DRF_VAL(_PDISP, _SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_DSI_SOR_CAP(sorIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_A",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_A, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_B",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_B, reg) ? "TRUE" : "FALSE");

    dprintf("%-55s: %s\n", "LW_PDISP_DSI_SOR_CAP_DP_INTERLACE",
        DRF_VAL(_PDISP, _DSI_SOR_CAP, _DP_INTERLACE, reg) ? "TRUE" : "FALSE");

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_MSB(sorIndex));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_VALUE",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _VALUE, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_MSB_REPEATER",
        DRF_VAL(_PDISP, _SOR_DP_HDCP_BKSV_MSB, _REPEATER, reg));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_BKSV_LSB_VALUE",
        GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_BKSV_LSB(sorIndex)));

    dprintf("%-55s: %x\n", "LW_PDISP_SOR_DP_HDCP_RI",
            GPU_REG_RD32(LW_PDISP_SOR_DP_HDCP_RI(sorIndex)));

    reg = GPU_REG_RD32(LW_PDISP_SOR_DP_SPARE(sorIndex, dpIndex));

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_SEQ_ENABLE",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _SEQ_ENABLE, reg) ? "YES" : "NO");

    dprintf("%-55s: %s\n", "LW_PDISP_SOR_DP_SPARE_PANEL",
        DRF_VAL(_PDISP, _SOR_DP_SPARE, _PANEL, reg) ? "INTERNAL" : "EXTERNAL");

    reg = GPU_REG_RD32(LW_PDISP_SOR_TEST(sorIndex));
    dprintf("%-55s: ", "LW_PDISP_SOR_TEST");
    headMask = DRF_VAL(_PDISP, _SOR_TEST, _OWNER_MASK, reg);
    if (headMask)
    {
        BOOL bHeadOnce;
        LwU32 i;

        // Print attached HEADs
        bHeadOnce = FALSE;
        for (i = 0; i < pDisp[indexGpu].dispGetNumHeads(); i++)
        {
            if (headMask & (1 << i))
            {
                if (bHeadOnce)
                    dprintf("|%d", i);
                else
                    dprintf("HEAD %d", i);

                bHeadOnce = TRUE;
            }
        }

        if (bHeadOnce)
            dprintf("\n");
        else
            dprintf("ERROR: head index can't be recognized\n");

        // Print relevant SF registers
        for (i = 0; i < pDisp[indexGpu].dispGetNumSfs(); i++)
        {
            LwU32 reg = GPU_REG_RD32(LW_PDISP_SF_TEST(i));

            if (DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, reg) & headMask)
            {
                dprintf("%-55s: %d\n", "sfIndex", i);

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_LINKCTL(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_LINKCTL_FORMAT_MODE",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _FORMAT_MODE,
                    reg) ? "MULTI_STREAM" : "SINGLE_STREAM");

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_LINKCTL_LANECOUNT",
                    DRF_VAL(_PDISP, _SF_DP_LINKCTL, _LANECOUNT, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_FLUSH(i));

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_ENABLE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _ENABLE,
                    reg) ? "YES" : "NO");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_MODE",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _MODE,
                    reg) ? "IMMEDIATE" : "LOADV");

                dprintf("%-55s: %s\n", "LW_PDISP_SF_DP_FLUSH_CNTL",
                    DRF_VAL(_PDISP, _SF_DP_FLUSH, _CNTL,
                    reg) ? "PENDING" : "DONE");

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_CTL(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_START",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_CTL_LENGTH",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_START_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _START_ACTIVE, reg));

                dprintf("%-55s: %d\n",
                    "LW_PDISP_SF_DP_STREAM_CTL_LENGTH_ACTIVE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_CTL, _LENGTH_ACTIVE, reg));

                reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_BW(i));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_ALLOCATED",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _ALLOCATED, reg));

                dprintf("%-55s: %d\n", "LW_PDISP_SF_DP_STREAM_BW_TIMESLICE",
                    DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _TIMESLICE, reg));
            }
        }
    }

    dispPrintDpRxInfo(port);
}

// Enumerate displayport pipe information, including source and sink.
void
dispDisplayPortEnum_v02_01(void)
{
    DPINFO_SF  *sf = 0;
    DPINFO_SOR *sor = 0;
    LwU32       numHead, numSor, numSf, reg, *headDisplayId = 0;
    LwU8        i, j, k;

    numHead = pDisp[indexGpu].dispGetNumHeads();
    if (numHead)
    {
        headDisplayId = (LwU32*)malloc(sizeof(LwU32) * numHead);
        if (headDisplayId == NULL)
        {
            dprintf("Failed to allocate memory for HEADs");
            return;
        }
        memset((void*)headDisplayId, 0, sizeof(LwU32) * numHead);
    }
    else
    {
        dprintf("No Head to enumerate.\n");
    }

    numSor = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
    if (numSor)
    {
        sor = (DPINFO_SOR*)malloc(sizeof(DPINFO_SOR) * numSor);
        if (sor == NULL)
        {
            dprintf("Failed to allocate memory for SOR_CFG");
            if (numHead && headDisplayId)
                free(headDisplayId);
            return;
        }
        memset((void*)sor, 0, sizeof(DPINFO_SOR) * numSor);
    }
    else
    {
        dprintf("No SOR to enumerate.\n");
    }

    numSf = pDisp[indexGpu].dispGetNumSfs();
    if (numSf)
    {
        sf = (DPINFO_SF*)malloc(sizeof(DPINFO_SF) * numSf);
        if (sor == NULL)
        {
            dprintf("Failed to allocate memory for SF_CFG");
            if (numHead && headDisplayId)
                free(headDisplayId);
            if (numSor && sor)
                free(sor);
            return;
        }
        memset((void*)sf, 0, sizeof(DPINFO_SF) * numSf);
    }
    else
    {
        dprintf("No SF to enumerate.\n");
    }

    for (i = 0; i < numHead; i++)
    {
        headDisplayId[i] = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) +
                                        LW917D_HEAD_SET_DISPLAY_ID(i, 0));
    }

    for (i = 0; i < numSf; i++)
    {
        // Exit if no displayId assigned.
        reg = GPU_REG_RD32(LW_PDISP_SF_TEST(i));
        if (!(sf[i].headMask = DRF_VAL(_PDISP, _SF_TEST, _OWNER_MASK, reg)))
            continue;

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_LINKCTL(i));
        sf[i].bDpEnabled  = FLD_TEST_DRF(_PDISP, _SF_DP_LINKCTL, _ENABLE, _YES,
                                         reg);
        sf[i].bMstEnabled = FLD_TEST_DRF(_PDISP, _SF_DP_LINKCTL, _FORMAT_MODE,
                                         _MULTI_STREAM, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_FLUSH(i));
        sf[i].bFlushEnabled = FLD_TEST_DRF(_PDISP, _SF_DP_FLUSH, _ENABLE, _YES,
                                           reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_CTL(i));
        sf[i].timeSlotStart  = DRF_VAL(_PDISP, _SF_DP_STREAM_CTL,
                                       _START_ACTIVE, reg);
        sf[i].timeSlotLength = DRF_VAL(_PDISP, _SF_DP_STREAM_CTL,
                                       _LENGTH_ACTIVE, reg);

        reg = GPU_REG_RD32(LW_PDISP_SF_DP_STREAM_BW(i));
        sf[i].pbn = DRF_VAL(_PDISP, _SF_DP_STREAM_BW, _ALLOCATED, reg);
    }

    for (i = 0; i < numSor; i++)
    {
        // SOR info
        sor[i].bExist = pDisp[indexGpu].dispResourceExists(LW_OR_SOR, i);
        if (!sor[i].bExist)
        {
            continue;
        }

        reg = GPU_REG_RD32(LW_PDISP_SOR_TEST(i));
        sor[i].headMask = DRF_VAL(_PDISP, _SOR_TEST, _OWNER_MASK, reg);

        reg = GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(0) +
                           LW917D_SOR_SET_CONTROL(i));
        sor[i].protocol = DRF_VAL(917D, _SOR_SET_CONTROL, _PROTOCOL, reg);

        sor[i].bDpActive[PRIMARY]   = FALSE;
        sor[i].bDpActive[SECONDARY] = FALSE;

        if (!pDisp[indexGpu].dispGetLinkBySor(i, sor[i].link))
        {
            dprintf("failed to get links to SOR%d\n", i);
            break;
        }

        for (j = 0; j < LW_MAX_SUBLINK; j++)
        {
            LwU32 link;

            reg = GPU_REG_RD32(LW_PDISP_SOR_DP_LINKCTL(i, j));
            sor[i].bDpActive[j] = FLD_TEST_DRF(_PDISP, _SOR_DP_LINKCTL,
                                               _ENABLE, _YES, reg);

            link = sor[i].link[j];
            sor[i].auxPort[j] = pDisp[indexGpu].dispGetAuxPortByLink(link);
        }
    }

    for (i = 0; i < numHead; i++)
    {
        if (!headDisplayId[i])
            continue;

        for (j = 0; j < numSf; j++)
        {
            if (sf[j].headMask & (1 << i))
            {
                sf[j].displayId = headDisplayId[i];
                break;
            }
        }
    }

    dprintf("Tx:\n"
            "---------------------------------------------------------------------------------------------------------\n"
            "HEAD  DISPLAYID  PROTOCOL         MODE  PBN     TIMESLOT(START:LENGTH)  FLUSH  SOR:SUBLINK  LINK  AUXPORT\n"
            "---------------------------------------------------------------------------------------------------------\n");
    for (i = 0; i < numHead; i++)
    {
        for (j = 0; j < numSf; j++)
        {
            if (sf[j].headMask != (1 << i))
                continue;

            // Exit if paired head&sf is not DP mode.
            if (!sf[j].bDpEnabled)
                break;

            dprintf("%-6d%-11x", i, headDisplayId[i]);
            for (k = 0; k < numSor; k++)
            {
                if (sor[k].bExist && (sor[k].headMask & (1 << i)))
                {
                    // Head = i, SF index = j, SOR = k
                    if (sf[j].bMstEnabled)
                    {
                        dprintf("%-17s%-6s%-8d%2d:%-21d%-7s",
                            dispGetStringForOrProtocol(LW_OR_SOR,
                            sor[k].protocol), "MST", sf[j].pbn,
                            sf[j].timeSlotStart, sf[j].timeSlotLength,
                            sf[j].bFlushEnabled ? "YES" : "NO");
                    }
                    else
                    {
                        dprintf("%-17s%-6s%-8s%-24s%-7s",
                            dispGetStringForOrProtocol(LW_OR_SOR,
                            sor[k].protocol), "SST", "NA", "NA",
                            sf[j].bFlushEnabled ? "YES" : "NO");
                    }

                    if (sor[k].bDpActive[PRIMARY])
                    {
                        dprintf("%d:Primary    %-6c%d\n", k,
                            (char)('A' + sor[k].link[PRIMARY]),
                            sor[k].auxPort[PRIMARY]);
                    }
                    else if (sor[k].bDpActive[SECONDARY])
                    {
                        dprintf("%d:Secondary  %-6c%d\n", k,
                            (char)('A' + sor[k].link[SECONDARY]),
                            sor[k].auxPort[SECONDARY]);
                    }

                    // Print second row if both sublinks are on.
                    if (sor[k].bDpActive[PRIMARY] &&
                        sor[k].bDpActive[SECONDARY])
                    {
                        dprintf("%80d:Secondary  %-6c%d\n", k,
                                (char)('A' + sor[k].link[SECONDARY]),
                                sor[k].auxPort[SECONDARY]);
                    }
                    break;
                }
            }
        }
    }

    dispPrintDpRxEnum();

    if (numHead && headDisplayId)
        free(headDisplayId);
 
    if (numSor && sor)
        free(sor);
 
    if (numSf && sf)
        free(sf);
}

/**
 * @brief Read PixelClk settings.
 *
 * @returns void
 */
void
dispReadPixelClkSettings_v02_01(void)
{
    LwU32 regVal;
    LwU32 VPLL;
    LwU32 rgMode;
    LwU32 rgDiv;
    LwU32 idx;

    dprintf("lw: All PixelClk settings\n");

    for (idx = 0; idx < pDisp[indexGpu].dispGetNumHeads(); idx++)
    {
        // Read RG settings
        regVal = GPU_REG_RD32(LW_PDISP_CLK_REM_RG(idx));

        // Check if the RG is enabled
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_RG, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   RG%d[ RG_STATE: DISABLE ]\n", idx);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
            continue;
        }

        // Check which RG mode is selected
        rgMode = DRF_VAL(_PDISP, _CLK_REM_RG, _MODE, regVal);
        if (rgMode == LW_PDISP_CLK_REM_RG_MODE_XITION)
        {
            dprintf("lw:   RG%d[ RG_MODE: XITION ]\n", idx);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
        }
        else if (rgMode == LW_PDISP_CLK_REM_RG_MODE_SAFE)
        {
            dprintf("lw:   RG%d[ RG_MODE: SAFE ]\n", idx);
            dprintf("lw: RG%d_PCLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (rgMode == LW_PDISP_CLK_REM_RG_MODE_NORMAL)
        {
            // Read the VPLL settings
            VPLL = pClk[indexGpu].clkGetVClkFreqKHz(idx) / 1000;
            dprintf("lw:   VPLL%d = %4d MHz\n", idx, VPLL);

            // Read the RG_DIV settings
            switch (DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal))
            {
                case LW_PDISP_CLK_REM_RG_DIV_BY_1:
                    rgDiv = 1;
                    dprintf("lw:   RG%d[ RG_DIV: BY_1 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_2:
                    rgDiv = 2;
                    dprintf("lw:   RG%d[ RG_DIV: BY_2 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_3:
                    rgDiv = 3;
                    dprintf("lw:   RG%d[ RG_DIV: BY_3 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_4:
                    rgDiv = 4;
                    dprintf("lw:   RG%d[ RG_DIV: BY_4 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_6:
                    rgDiv = 6;
                    dprintf("lw:   RG%d[ RG_DIV: BY_6 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_8:
                    rgDiv = 8;
                    dprintf("lw:   RG%d[ RG_DIV: BY_8 ]\n", idx);
                    break;

                case LW_PDISP_CLK_REM_RG_DIV_BY_16:
                    rgDiv = 16;
                    dprintf("lw:   RG%d[ RG_DIV: BY_16 ]\n", idx);
                    break;

                default:
                    rgDiv = DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal) + 1;
                    dprintf("lw:   RG%d[ RG_DIV: invalid enum (%d) ]\n",
                            idx, DRF_VAL(_PDISP, _CLK_REM_RG, _DIV, regVal));
                    break;
            }
            dprintf("lw: RG%d_PCLK = %4d MHz\n\n", idx, (VPLL / rgDiv));
        }
        else
        {
            dprintf("lw:   RG%d[ RG_MODE: invalid enum (%d) ]\n", idx, rgMode);
            dprintf("lw: RG%d_PCLK = N/A\n\n", idx);
        }
    }
}

/**
 * @brief Read SorClk settings.
 *
 * @returns void
 */
void
dispReadSorClkSettings_v02_01(void)
{
    LwU32 regVal;
    LwU32 VPLL;
    LwU32 idx;
    LwU32 headNum;
    LwU32 sorMode;
    LwU32 sorDiv;
    LwU32 sorPllRefDiv;
    LwU32 sorModeBypass;
    LwU32 sorLinkSpeed;
    LwU32 sorClk;

    dprintf("lw: All SorClk settings\n");

    for (idx = 0; idx < pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR); idx++)
    {
        // Read SOR settings
        regVal = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR(idx));

        // Check if the SOR is enabled
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR, _STATE, _DISABLE, regVal))
        {
            dprintf("lw:   SOR%d[ SOR_STATE: DISABLE ]\n", idx);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
            continue;
        }

        // Check which SOR mode is selected
        sorMode = DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE, regVal);
        if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_XITION)
        {
            dprintf("lw:   SOR%d[ SOR_MODE: XITION ]\n", idx);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
        }
        else if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_SAFE)
        {
            dprintf("lw:   SOR%d[ SOR_MODE: SAFE ]\n", idx);
            dprintf("lw: SOR%d_CLK = %4d MHz\n\n",
                    idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
        }
        else if (sorMode == LW_PDISP_CLK_REM_SOR_MODE_NORMAL)
        {
            headNum = DRF_VAL(_PDISP, _CLK_REM_SOR, _HEAD, regVal);
            if (headNum == LW_PDISP_CLK_REM_SOR_HEAD_NONE)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: NONE ]\n", idx);
                continue;
            }
            if (headNum > LW_PDISP_CLK_REM_SOR_HEAD_3)
            {
                dprintf("lw:   SOR%d[ SOR_HEAD: invalid enum (%d) ]\n", idx, headNum);
                continue;
            }

            sorLinkSpeed = DRF_VAL(_PDISP, _CLK_REM_SOR, _DP_LINK_SPEED, regVal);
            sorModeBypass = DRF_VAL(_PDISP, _CLK_REM_SOR, _MODE_BYPASS, regVal);

            if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_NONE)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: NONE ]\n", idx);

                // sorClk = (VPLL freq / SOR_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal))
                {
                    case LW_PDISP_CLK_REM_SOR_DIV_BY_1:
                        sorDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_2:
                        sorDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_4:
                        sorDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_8:
                        sorDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DIV_BY_16:
                        sorDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorDiv = 2 << DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal);
                        dprintf("lw:   SOR%d[ SOR_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _CLK_REM_SOR, _DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_FEEDBACK)
            {
                VPLL = pClk[indexGpu].clkGetVClkFreqKHz(headNum) / 1000;
                dprintf("lw:   VPLL%d[ %4d MHz ]\n", headNum, VPLL);
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: FEEDBACK ]\n", idx);

                // sorClk = (VPLL freq / SOR_PLL_REF_DIV) * LINK_SPEED / 10
                switch (DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal))
                {
                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_1:
                        sorPllRefDiv = 1;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_1 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_2:
                        sorPllRefDiv = 2;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_2 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_4:
                        sorPllRefDiv = 4;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_4 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_8:
                        sorPllRefDiv = 8;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_8 ]\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_PLL_REF_DIV_BY_16:
                        sorPllRefDiv = 16;
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: BY_16 ]\n", idx);
                        break;

                    default:
                        sorPllRefDiv = 2 << DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal);
                        dprintf("lw:   SOR%d[ SOR_PLL_REF_DIV: invalid enum (%d) ]\n",
                                idx, DRF_VAL(_PDISP, _CLK_REM_SOR, _PLL_REF_DIV, regVal));
                        break;
                }
                sorClk = (VPLL / sorPllRefDiv) * (sorLinkSpeed / 10);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n", idx, sorClk);
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_NORMAL)
            {
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: DP_NORMAL ]\n", idx);

                // sorClk uses DP pad macro feedback clock
                switch (sorLinkSpeed)
                {
                    case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_1_62GHZ:
                        dprintf("lw: SOR%d_CLK = 162 MHz\n\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_2_70GHZ:
                        dprintf("lw: SOR%d_CLK = 270 MHz\n\n", idx);
                        break;

                    case LW_PDISP_CLK_REM_SOR_DP_LINK_SPEED_5_40GHZ:
                        dprintf("lw: SOR%d_CLK = 540 MHz\n\n", idx);
                        break;

                    default:
                        dprintf("lw:   SOR%d[ SOR_LINK_SPEED: invalid enum (%d) ]\n",
                                idx, sorLinkSpeed);
                        dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
                        break;
                }
            }
            else if (sorModeBypass == LW_PDISP_CLK_REM_SOR_MODE_BYPASS_DP_SAFE)
            {
                // sorClk is Xtal safe clock
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: DP_SAFE ]\n", idx);
                dprintf("lw: SOR%d_CLK = %4d MHz\n\n",
                        idx, pClk[indexGpu].clkReadCrystalFreqKHz() / 1000);
            }
            else
            {
                dprintf("lw:   SOR%d[ SOR_MODE_BYPASS: invalid enum (%d) ]\n",
                        idx, sorModeBypass);
                dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
            }
        }
        else
        {
            dprintf("lw:   SOR%d[ SOR_MODE: invalid enum (%d) ]\n", idx, sorMode);
            dprintf("lw: SOR%d_CLK = N/A\n\n", idx);
        }
    }
}



// Note: display v02_01 state is also stored in dispChanState_v02_01[]. As display v02_01 
// supports different number of heads (from 2 to 4), we require the chanNum parameter of functions
// like dispPrintChanState_v02_01(chanNum) be obtained via dispGetChanNum_v02_01().


LwU32
dispGetNumHeads_v02_01(void)
{
    LwU32 val;

    val = GPU_REG_RD32(LW_PDISP_CLK_REM_MISC_CONFIGA);
    return DRF_VAL(_PDISP, _CLK_REM_MISC_CONFIGA, _NUM_HEADS, val);
}

// Print the ARM and ASSY values for a given EVO channel.
void
dispPrintChanMethodState_v02_01
(
    LwU32 chanNum,
    BOOL printHeadless,
    BOOL printRegsWithoutEquivMethod,
    LwS32 coreHead,
    LwS32 coreWin
)
{
    ChnType chanId;
    LwU32 headNum, head, surf, k;
    LwU32 arm, assy;
    LwU32 i = 0;
    char classString[32];
    char commandString[52];
    GetClassNum(classString);         

    chanId = dispChanState_v02_01[chanNum].id;
    headNum = dispChanState_v02_01[chanNum].headNum;

#ifndef LW917C_SET_CONTEXT_DMAS_ISO__SIZE_1
#define LW917C_SET_CONTEXT_DMAS_ISO__SIZE_1                         4
#endif
#ifndef LW917C_SURFACE_SET_OFFSET__SIZE_1
#define LW917C_SURFACE_SET_OFFSET__SIZE_1                           2
#endif
#ifndef LW917C_SURFACE_SET_OFFSET__SIZE_2
#define LW917C_SURFACE_SET_OFFSET__SIZE_2                           2
#endif
#ifndef LW917C_SURFACE_SET_SIZE__SIZE_1
#define LW917C_SURFACE_SET_SIZE__SIZE_1                             2
#endif
#ifndef LW917D_HEAD_SET_DISPLAY_ID__SIZE_1
#define LW917D_HEAD_SET_DISPLAY_ID__SIZE_1                          2
#endif

    switch(chanId)
    {
        case CHNTYPE_CORE: // Core channel - 917D
            for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
            {
                if (coreHead >= 0 && head != coreHead)
                    continue;

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                dprintf("----------------------------------------------------------------------------------------------\n");
                //
                // The following list is based off of //sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/class/cl917d.h#13
                // Note that it's implicit that the above comment applies only to core channel (d in 917d implies core)
                //
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PRESENT_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTROL_OUTPUT_RESOURCE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_LOCK_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_OVERSCAN_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_RASTER_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_RASTER_SYNC_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_RASTER_BLANK_END, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_RASTER_BLANK_START, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_RASTER_VERT_BLANK2, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_LOCK_CHAIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_DEFAULT_BASE_COLOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_LEGACY_CRC_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTEXT_DMA_CRC, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_BASE_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_BASE_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_OUTPUT_LUT_LO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_OUTPUT_LUT_HI, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PIXEL_CLOCK_FREQUENCY, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PIXEL_CLOCK_CONFIGURATION, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTEXT_DMA_LUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_OFFSET, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_SIZE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_STORAGE, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PARAMS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTEXT_DMAS_ISO, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTROL_LWRSOR, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PRESENT_CONTROL_LWRSOR, head, chanNum); // new registers in 917D 
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917D_HEAD_SET_OFFSETS_LWRSOR, head, 0, chanNum); // 917D has stereo lwrsors
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917D_HEAD_SET_OFFSETS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 0, chanNum);
                DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917D_HEAD_SET_CONTEXT_DMAS_LWRSOR, head, 1, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_DITHER_CONTROL, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CONTROL_OUTPUT_SCALER, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PROCAMP, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_POINT_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_SIZE_IN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_POINT_OUT_ADJUST, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_SIZE_OUT, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_SIZE_OUT_MIN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VIEWPORT_SIZE_OUT_MAX, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_BASE_CHANNEL_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_OVERLAY_USAGE_BOUNDS, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_PROCESSING, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_COLWERSION_RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_COLWERSION_GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_COLWERSION_BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_RED2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_GRN2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_BLU2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_CONSTANT2RED, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_RED2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_GRN2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_BLU2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_CONSTANT2GRN, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_RED2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_GRN2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_BLU2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_CSC_CONSTANT2BLU, head, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_HDMI_CTRL, head, chanNum); // new registers in 917D 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_VACTIVE_SPACE_COLOR, head, chanNum); 

                for (k = 0; k < LW917D_HEAD_SET_DISPLAY_ID__SIZE_1; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917D_HEAD_SET_DISPLAY_ID, head, k, chanNum);
                }

                // It seems the following registers need not be printed
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_SW_METHOD_PLACEHOLDER_A, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_SW_METHOD_PLACEHOLDER_B, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_SW_METHOD_PLACEHOLDER_C, head, chanNum); 
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_HEAD_SET_SW_METHOD_PLACEHOLDER_D, head, chanNum); 
            }

            if (printHeadless == TRUE)
            {
                LwU32 numDacs = pDisp[indexGpu].dispGetNumOrs(LW_OR_DAC);
                LwU32 numSors = pDisp[indexGpu].dispGetNumOrs(LW_OR_SOR);
                LwU32 numPiors = pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR);

                dprintf("----------------------------------------------------------------------------------------------\n");
                dprintf("CORE CHANNEL HEADLESS                                 ASY    |    ARM     | ASY-ARM Mismatch\n");
                dprintf("----------------------------------------------------------------------------------------------\n");
                for (k = 0; k < numDacs; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_DAC_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_DAC_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numSors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_SOR_SET_CONTROL,           k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_SOR_SET_LWSTOM_REASON,     k, chanNum);
                }
                for (k = 0; k < numPiors; ++k)
                {
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_PIOR_SET_CONTROL,          k, chanNum);
                    DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917D_PIOR_SET_LWSTOM_REASON,    k, chanNum);
                }
                DISP_PRINT_SC_NON_IDX_V02_01(LW917D_SET_CONTEXT_DMA_NOTIFIER, chanNum);
                DISP_PRINT_SC_NON_IDX_V02_01(LW917D_SET_NOTIFIER_CONTROL, chanNum);
            }

            if (printRegsWithoutEquivMethod == TRUE)
            {
                for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
                {
                    dprintf("----------------------------------------------------------------------------------------------\n");
                    dprintf("CORE CHANNEL HEAD %u (SC w/o equiv method)             ASY    |    ARM     | ASY-ARM Mismatch\n", head);
                    dprintf("----------------------------------------------------------------------------------------------\n");
                }

                if (printHeadless == TRUE)
                {
                    dprintf("----------------------------------------------------------------------------------------------\n");
                    dprintf("CORE CHANNEL HEADLESS (SC w/o equiv method)           ASY    |    ARM     | ASY-ARM Mismatch\n");
                    dprintf("----------------------------------------------------------------------------------------------\n");
                }
            }
            break;

        case CHNTYPE_BASE: // Base channel - 917C
            dprintf("----------------------------------------------------------------------------------------------\n");
            dprintf("BASE CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", headNum);
            dprintf("----------------------------------------------------------------------------------------------\n");
            for (k = 0; k < LW917C_SET_CONTEXT_DMAS_ISO__SIZE_1; ++k)
            {
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917C_SET_CONTEXT_DMAS_ISO, k, chanNum);
            }

            for (surf = 0; surf < LW917C_SURFACE_SET_OFFSET__SIZE_1; ++surf)
            {
                for (k = 0; k < LW917C_SURFACE_SET_OFFSET__SIZE_2; ++k)
                {
                    DISP_PRINT_SC_DOUBLE_IDX_V02_01(LW917C_SURFACE_SET_OFFSET, surf, k, chanNum);
                }
            }
            for (surf = 0; surf < LW917C_SURFACE_SET_SIZE__SIZE_1; ++surf)
            {
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917C_SURFACE_SET_SIZE, surf, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917C_SURFACE_SET_STORAGE, surf, chanNum);
                DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917C_SURFACE_SET_PARAMS, surf, chanNum);
            }
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CONTEXT_DMA_LUT, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CONTEXT_DMA_NOTIFIER, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CONTEXT_DMA_SEMAPHORE, chanNum);

            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_DIST_RENDER_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_DIST_RENDER_EXTEND_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_DIST_RENDER_INHIBIT_FLIP_REGION, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_DIST_RENDER_CONFIG, chanNum);

            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_BASE_LUT_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_BASE_LUT_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_OUTPUT_LUT_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_OUTPUT_LUT_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_NOTIFIER_CONTROL, chanNum);

            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_PROCESSING, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_COLWERSION_RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_COLWERSION_GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_COLWERSION_BLU, chanNum);

            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_PRESENT_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_SEMAPHORE_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_SEMAPHORE_ACQUIRE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_SEMAPHORE_RELEASE, chanNum);

            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_RED2RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_GRN2RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_BLU2RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_CONSTANT2RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_RED2GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_GRN2GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_BLU2GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_CONSTANT2GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_RED2BLU, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_GRN2BLU, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_BLU2BLU, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917C_SET_CSC_CONSTANT2BLU, chanNum);

            break;

        case CHNTYPE_OVLY: // Ovly channel - 917E
            dprintf("----------------------------------------------------------------------------------------------\n");
            dprintf("OVLY CHANNEL HEAD %u                                   ASY    |    ARM     | ASY-ARM Mismatch\n", headNum);
            dprintf("----------------------------------------------------------------------------------------------\n");
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_COMPOSITION_CONTROL, chanNum);
            DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917E_SET_CONTEXT_DMAS_ISO, 0, chanNum); // 917E has stereo overlay
            DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917E_SET_CONTEXT_DMAS_ISO, 1, chanNum); 
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_CONTEXT_DMA_NOTIFIER, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_CONTEXT_DMA_SEMAPHORE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_CONTEXT_DMA_LUT, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_NOTIFIER_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_PRESENT_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_POINT_IN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_SEMAPHORE_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_SEMAPHORE_ACQUIRE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_SEMAPHORE_RELEASE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_SIZE_IN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_SIZE_OUT, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_TIMESTAMP_ORIGIN_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_TIMESTAMP_ORIGIN_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_UPDATE_TIMESTAMP_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_UPDATE_TIMESTAMP_HI, chanNum);
            DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917E_SURFACE_SET_OFFSET, 0, chanNum);
            DISP_PRINT_SC_SINGLE_IDX_V02_01(LW917E_SURFACE_SET_OFFSET, 1, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SURFACE_SET_SIZE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SURFACE_SET_STORAGE, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SURFACE_SET_PARAMS, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_OVERLAY_LUT_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_OVERLAY_LUT_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_COMPOSITION_CONTROL, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_KEY_COLOR_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_KEY_COLOR_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_KEY_MASK_LO, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_KEY_MASK_HI, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_PROCESSING, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_COLWERSION_RED, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_COLWERSION_GRN, chanNum);
            DISP_PRINT_SC_NON_IDX_V02_01(LW917E_SET_COLWERSION_BLU, chanNum);

            break;

        case CHNTYPE_OVIM: // Ovim channel - 917B
            // Nothing to print.
            break;

        case CHNTYPE_LWRS: // Lwrs channel - 917A
            // Nothing to print.
            break;

        default:
            dprintf("EVO channel %u not supported.\n", chanNum);
    }
}

LwU32 dispGetNumSurfPerHead_v02_01
(
    void *pType
)
{
    switch (*((ChnType*)pType))
    {
        case CHNTYPE_CORE:
            return LW917D_CORE_SURFACE_PER_HEAD;
        case CHNTYPE_BASE:
            return LW917C_BASE_SURFACE_PER_HEAD;
        case CHNTYPE_OVLY:
            return LW917E_OVLY_SURFACE_PER_HEAD; 
        default:
            return 0;
    }
}

LwU32 dispReadSurfCtxDmaHandle_v02_01
(
    void *pType,
    LwU32 headNum,
    LwU32 nSurf
)
{
    LwS32 chanNum;

    switch (*((ChnType*)pType))
    {
        case CHNTYPE_OVLY:
            chanNum = dispGetChanNum_v02_01("ovly", headNum);
            if (chanNum < 0)
            {
                return 0;
            }

            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + 
                            LW917E_SET_CONTEXT_DMAS_ISO(nSurf)); 

        case CHNTYPE_CORE:
            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(
                                LW_PDISP_907D_CHN_CORE) +
                            LW917D_HEAD_SET_CONTEXT_DMAS_ISO(headNum));
        case CHNTYPE_BASE:
            chanNum = dispGetChanNum_v02_01("base", headNum);
            if (chanNum < 0)
            {
                return 0;
            }
            return GPU_REG_RD32(LW_UDISP_DSI_CHN_ARMED_BASEADR(chanNum) + 
                            LW917C_SET_CONTEXT_DMAS_ISO(nSurf)); 

        default:
            return 0;
    }
}

/*!
 *  Function to get links driven by specified SOR.
 *
 *  @param[in]  sorIndex    Index of SOR.
 *  @param[out] pLinks      Links driven by specified SOR.
 */
BOOL dispGetLinkBySor_v02_01
(
    LwU32 sorIndx,
    LwU32 *pLinks
)
{
    PADLINK sorLinkMatrix[LW_PDISP_MAX_SOR][LW_MAX_SUBLINK] =
    {
        {PADLINK_A, PADLINK_B},
        {PADLINK_C, PADLINK_NONE},
        {PADLINK_D, PADLINK_NONE},
        {PADLINK_E, PADLINK_F}
    };
    LwU32 reg;

    if (sorIndx >= LW_PDISP_MAX_SOR || pLinks == NULL)
        return FALSE;

    // Update mapping if split SOR.
    if (sorIndx == 0 || sorIndx == 3)
    {
        reg = GPU_REG_RD32(LW_PDISP_CLK_REM_SOR_CTRL(0));
        if (FLD_TEST_DRF(_PDISP, _CLK_REM_SOR_CTRL, _BACKEND, _SOR3, reg))
        {
            sorLinkMatrix[0][PRIMARY]   = PADLINK_NONE;
            sorLinkMatrix[0][SECONDARY] = sorLinkMatrix[3][SECONDARY];
            sorLinkMatrix[3][SECONDARY] = PADLINK_NONE;
        }
    }

    pLinks[PRIMARY]   = sorLinkMatrix[sorIndx][PRIMARY];
    pLinks[SECONDARY] = sorLinkMatrix[sorIndx][SECONDARY];
    return TRUE;
}

/*!
 * @brief Function to print SLI registers.
 *
 *  @param[in]  LwU32               numHead         Number of Heads
 *  @param[in]  LwU32               numPior         Number of PIORs
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  data structure
 *  @param[in]  DSLI_PIOR_DATA      *pDsliPiorData  Pointer to DSLI_PIOR
 *                                                  data structure
 */
void dispPrintSliRegisters_v02_01
(
    LwU32           numHead,
    LwU32           numPior,
    DSLI_DATA      *pDsliData,
    DSLI_PIOR_DATA *pDsliPiorData
)
{
    LwU32 head = 0;
    LwU32 pior = 0;

    dprintf("--------------------------------------------------------------------------------------------------------\n");
    dprintf("                                        Register Information                                            \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");

    dprintf("%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("HEAD-%d       |", head);
    }
    dprintf("\n%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("------       |");
    }
    dprintf("\n");

    PRINTCONFIGHEAD("LW_PDISP_RG_DIST_RNDR", pDsliData, DsliRgDistRndr);
    PRINTCONFIGHEAD("LW_PDISP_RG_DIST_RNDR_SYNC_ADVANCE", pDsliData, DsliRgDistRndrSyncAdv);
    PRINTCONFIGHEAD("LW_PDISP_RG_FLIPLOCK", pDsliData, DsliRgFlipLock);
    PRINTCONFIGHEAD("LW_PDISP_RG_STATUS", pDsliData, DsliRgStatus);
    PRINTCONFIGHEAD("LW_PDISP_CLK_REM_VPLL_EXT_REF_CONFIG", pDsliData, DsliClkRemVpllExtRef);
    PRINTCONFIGHEAD("LW917D_HEAD_SET_CONTROL", pDsliData, DsliHeadSetCntrl);
    PRINTCONFIGHEAD("LW_PVTRIM_SYS_VCLK_REF_SWITCH", pDsliData, DsliPvTrimSysVClkRefSwitch);

    dprintf("%40s |", "");
    for (head = 0; head < numHead; ++head)
    {
        dprintf("------       |");
    }
    dprintf("\n\n");

    dprintf("%48s       |", "|PIOR-0");
    dprintf("%-12s |", "PIOR-1");
    dprintf("%-12s |\n", "PIOR-2");
    dprintf("%48s       |", "|------");
    dprintf("%-12s |", "------");
    dprintf("%-12s |\n", "------");
    PRINTCONFIGPIOR("LW_PDISP_PIOR_DRO");
    dprintf("%48s       |", "|------");
    dprintf("%-12s |", "------");
    dprintf("%-12s |\n\n", "------");

    PRINTCONFIG("LW_PDISP_DSI_CAPA", pDsliPiorData->DsliCap);
}


//-----------------------------------------------------
// dispGetRemVpllCfgSize_v02_01
//
//-----------------------------------------------------
LwU32 dispGetRemVpllCfgSize_v02_01()
{
    return LW_PDISP_CLK_REM_VPLL_CFG__SIZE_1;
}


LwS32 dispInjectMethod_v02_01 (LwU32 chanNum, LwU32 offset, LwU32 data )
{
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();
}

/*!
 * mthdName format - MTHD_NAME_FIELD@IDX@IDX
 */
LwS32 dispMatchMethod_v02_01(LwU32 chanNum, LwU32 headNum, char * mthdName, LwU32 *offset, LwU32 *hbit, LwU32 *lbit, LwU32 *sc)
{
    PRINT_LWWATCH_NOT_IMPLEMENTED_MESSAGE_AND_RETURN0();
}


#define DISP_v0201_NUM_DMA_CHANNELS LW_PDISP_DSI_DEBUG_CTL__SIZE_1

#define DISP_PUSH_BUFFER_SIZE 4096

static void dispPrintOwnerProtocol_v02_01(LWOR orType,LwU32 orNum);
static void printDispPbParsed_v02_01(PhysAddr baseAddr, LwU32 OffsetDwords, LwU32* buffer, LwU32 numDwords, LwU32 classNum, LwU32 getoffset, LwU32 putoffset);
static void getPbData_v02_01(PhysAddr physOffset, LwU32 numDwords, char * buffer, LwU32 mem_Target);

//
// Global variables.
//
mthds_t *mthd[DISP_v0201_NUM_DMA_CHANNELS];
int mthd_num[DISP_v0201_NUM_DMA_CHANNELS];
static int mthdInitialized = 0;

LwU32 classHeaderNum[CHNTYPE_OVLY + 1];

#define     PHYSICAL_ADDR   40

//
// Prints the dump Push Buffer
//
LW_STATUS  dispDumpPB_v02_01
(
    LwU32 chanNum,
    LwS32 headNum,
    LwS32 numDwords,
    LwS32 OffsetDwords,
    LwU32 printParsed
)
{
    LW_STATUS   status = 0, data32 = 0;
    PhysAddr    physOffset = 0;
    LwU32       flagAllocated = 0, flagConnected = 0;
    LwU32       memTarget;
    LwU32       chnCtlOffset;
    PBCTLOFFSET pbCtlOffset;
    LwS32       channelClass = pDisp[indexGpu].dispGetChanType(chanNum, NULL);
    char*       buffer;
    LwU32       classNum;
    LwU32       getoffset = 0, putoffset = 0;

    status = pDisp[indexGpu].dispGetChnAndPbCtlRegOffsets(headNum,
                                                          chanNum,
                                                          channelClass,
                                                          &chnCtlOffset,
                                                          &pbCtlOffset);
    if (status == LW_ERR_GENERIC)
    {
            return status;
    }

    data32 = GPU_REG_RD32(chnCtlOffset);
    switch (channelClass)
    {
        case CHNTYPE_CORE:
            flagAllocated = DRF_VAL(_PDISP, _CHNCTL, _CORE_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _CHNCTL, _CORE_CONNECTION, data32);

            dprintf("lw: LW_PDISP_CHNCTL_CORE_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_CHNCTL_CORE_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        case CHNTYPE_BASE:
            flagAllocated = DRF_VAL(_PDISP, _CHNCTL, _BASE_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _CHNCTL, _BASE_CONNECTION, data32);

            dprintf("lw: LW_PDISP_CHNCTL_BASE_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_CHNCTL_BASE_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        case CHNTYPE_OVLY:
            flagAllocated = DRF_VAL(_PDISP, _CHNCTL, _OVLY_ALLOCATION, data32);
            flagConnected = DRF_VAL(_PDISP, _CHNCTL, _OVLY_CONNECTION, data32);

            dprintf("lw: LW_PDISP_CHNCTL_OVLY_ALLOCATION\t      %s\n",
                    flagAllocated ? "ALLOCATE" : "DEALLOCATE");

            dprintf("lw: LW_PDISP_CHNCTL_OVLY_CONNECT\t      %s\n",
                    flagConnected ? "CONNECT" : "DISCONNECT");
        break;

        default:
            dprintf("Error : Channel is not DMA channel. Use core, base or ovly\n\n");
    }

    //
    // if channel is not allocated or connected then return
    //
    if (!(flagConnected && flagAllocated))
    {
        dprintf("lw: Channel is not connected to any PB. Skipping the dump of push buffer.\n\n");
        return status;
    }

    //
    // Read the PB physical address by shifting 4 bit right and
    // then shift 12 bit left to get 40 bit physical address
    //
    data32 = GPU_REG_RD32(pbCtlOffset.PbCtlOffset[0]);

    memTarget = DRF_VAL(_PDISP, _PBCTL0, _PUSHBUFFER_TARGET, data32);

    physOffset = ((PhysAddr)DRF_VAL(_PDISP, _PBCTL0, _PUSHBUFFER_START_ADDR, data32)) <<
        (PHYSICAL_ADDR - DRF_SIZE(LW_PDISP_PBCTL0_PUSHBUFFER_START_ADDR));

    buffer = (char *)malloc((size_t)(DISP_PUSH_BUFFER_SIZE));
    getPbData_v02_01(physOffset, (DISP_PUSH_BUFFER_SIZE / 4), buffer, memTarget);
    classNum = classHeaderNum[channelClass];

    dprintf( "lw: LW_PDISP_PBCTL0_PUSHBUFFER_TARGET(%d) :", chanNum);
    dprintf( (memTarget == LW_PDISP_PBCTL0_PUSHBUFFER_TARGET_PHYS_LWM) ? "VIDEO MEMORY\n" : "SYSTEM MEMORY\n");
    dprintf( "lw: LW_PDISP_PBCTL0(%d) \t\t            0x%x\n\n", chanNum, data32);

    putoffset = GPU_REG_RD32(LW_UDISP_DSI_PUT(chanNum));
    getoffset = GPU_REG_RD32(LW_UDISP_DSI_GET(chanNum));

    // printing out get and put pointers
    dprintf("lw: GET POINTER OFFSET: 0x%08x\n",getoffset);
    dprintf("lw: PUT POINTER OFFSET: 0x%08x\n",putoffset);

    if (!printParsed)
    {
        printBuffer((buffer + (OffsetDwords * 4)), (LwU32)(numDwords * 4), (physOffset + (OffsetDwords * 4)), 4);
    }
    else
    {
        printDispPbParsed_v02_01((LwU32)physOffset, OffsetDwords, (LwU32 *)buffer, (LwU32)numDwords, classNum, getoffset, putoffset);
    }

    free(buffer);
    return status;
}

LW_STATUS
dispUpdateNumSinks_v02_01
(
    LwU32 headNum,
    LWOR orType,
    LwU32* pNumSinks
)
{
    LwU32   i;
    LwU32   numOrs = pDisp[indexGpu].dispGetNumOrs(orType);
    char*   orStr = dispGetORString(orType);
    HEAD    orOwner;

    if (!pNumSinks)
    {
        return LW_ERR_GENERIC;
    }

    for (i = 0; i < numOrs; i++)
    {
        pDisp[indexGpu].dispReadOrOwnerAndProtocol(orType, i, &orOwner, NULL);
        if (orOwner >= HEAD(pDisp[indexGpu].dispGetNumHeads()))
        {
            dprintf("\n");
            dprintf("lw: %s ERROR: Bad owner %d for %s%d\n", __FUNCTION__, orOwner, orStr, i);
            return LW_ERR_GENERIC;
        }

        if (orOwner != HEAD_UNSPECIFIED)
        {
            if (orOwner == HEAD(headNum))
            {
                ++(*pNumSinks);
                dprintf("%s%d  ", orStr, i);
            }
        }
    }
    return LW_OK;
}

//
// returns the DispOwner
//
LwS32 dispDispOwner_v02_01(void)
{
    LwU32 val;
    val = GPU_REG_RD32(LW_PDISP_VGA_CR_REG58);
    val = DRF_VAL(_PDISP, _VGA_CR_REG58, _SET_DISP_OWNER, val);

    if (val == LW_PDISP_VGA_CR_REG58_DISP_OWNER_DRIVER)
        return 0;
    else if (val == LW_PDISP_VGA_CR_REG58_DISP_OWNER_VBIOS)
        return 1;
    else
        return -1;
}

static void dispPrintOwnerProtocol_v02_01(LWOR orType,LwU32 orNum)
{
    HEAD    orOwner;
    ORPROTOCOL orProtocol;
    char *protocolString;
    char *orString = dispGetORString(orType);

    pDisp[indexGpu].dispReadOrOwnerAndProtocol(orType, orNum, &orOwner, &orProtocol);
    if (orOwner >= HEAD(pDisp[indexGpu].dispGetNumHeads()))
    {
        dprintf("Error: Invalid head                     ");
    }
    else
    {
        if (orOwner == HEAD_UNSPECIFIED)
        {
            dprintf("%s%d    NONE        N/A                 ", orString, orNum);
        }
        else
        {
            protocolString = dispGetStringForOrProtocol(orType, orProtocol);
            dprintf("%s%d    HEAD%d       %-20s", orString, orNum, HEAD_IDX(orOwner), protocolString);
        }
    }
}

void dispHeadDacConnection_v02_01(void)
{
    LwU32 i, data32;

    for (i = 0; i < pDisp[indexGpu].dispGetNumOrs(LW_OR_DAC); i++)
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_DAC, i) != TRUE)
        {
            continue;
        }
        dispPrintOwnerProtocol_v02_01(LW_OR_DAC, i);

        data32 = GPU_REG_RD32(LW_PDISP_DAC_PWR(i));
        if(DRF_VAL(_PDISP, _DAC_PWR, _MODE, data32) == LW_PDISP_DAC_PWR_MODE_SAFE)
        {
            dprintf("SAFE             ");
            switch(DRF_VAL(_PDISP, _DAC_PWR, _SAFE_HSYNC, data32))
            {
                case LW_PDISP_DAC_PWR_SAFE_HSYNC_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_HSYNC_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_HSYNC_LO:
                    dprintf("%-8s", "LO,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }
            switch(DRF_VAL(_PDISP, _DAC_PWR, _SAFE_VSYNC, data32))
            {
                case LW_PDISP_DAC_PWR_SAFE_VSYNC_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_VSYNC_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_VSYNC_LO:
                    dprintf("%-8s", "LO,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }
            switch(DRF_VAL(_PDISP, _DAC_PWR, _SAFE_DATA, data32))
            {
                case LW_PDISP_DAC_PWR_SAFE_DATA_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_DATA_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                case LW_PDISP_DAC_PWR_SAFE_DATA_LO:
                    dprintf("%-8s", "LO,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }
            dprintf("%-8s", (DRF_VAL(_PDISP, _DAC_PWR, _SAFE_PWR, data32) == LW_PDISP_DAC_PWR_SAFE_PWR_OFF)? "OFF" : "ON");
        }
        else
        {
            dprintf("NORMAL           ");
            switch(DRF_VAL(_PDISP, _DAC_PWR, _NORMAL_HSYNC, data32))
            {
                case LW_PDISP_DAC_PWR_NORMAL_HSYNC_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_HSYNC_LO:
                    dprintf("%-8s", "LO,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_HSYNC_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }

            switch(DRF_VAL(_PDISP, _DAC_PWR, _NORMAL_VSYNC, data32))
            {
                case LW_PDISP_DAC_PWR_NORMAL_VSYNC_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_VSYNC_LO:
                    dprintf("%-8s", "LO,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_VSYNC_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }

            switch(DRF_VAL(_PDISP, _DAC_PWR, _NORMAL_DATA, data32))
            {
                case LW_PDISP_DAC_PWR_NORMAL_DATA_HI:
                    dprintf("%-8s", "HI,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_DATA_LO:
                    dprintf("%-8s", "LO,");
                break;
                case LW_PDISP_DAC_PWR_NORMAL_DATA_ENABLE:
                    dprintf("%-8s", "ENABLE,");
                break;
                default:
                    dprintf("Error: Invalid Data,");
                break;
            }
            dprintf("%-8s", (DRF_VAL(_PDISP, _DAC_PWR, _NORMAL_PWR, data32) == LW_PDISP_DAC_PWR_NORMAL_PWR_OFF)? "OFF" : "ON");
        }

        data32 = GPU_REG_RD32(LW_PDISP_DAC_BLANK(i));
        if(DRF_VAL(_PDISP, _DAC_BLANK, _STATUS, data32) == LW_PDISP_DAC_BLANK_STATUS_BLANKED)
        {
            dprintf("YES%s", (DRF_VAL(_PDISP, _DAC_BLANK, _OVERRIDE, data32) == LW_PDISP_DAC_BLANK_OVERRIDE_TRUE)? " (because of override)" : "");
        }
        else
        {
            dprintf("NO");
        }

        dprintf("\n");
    }
}

void dispHeadPiorConnection_v02_01(void)
{
    LwU32 i, data32;
    LwU32 PiorPwrOffset, PiorBlankOffset;

    for (i = 0; i < pDisp[indexGpu].dispGetNumOrs(LW_OR_PIOR); i++)
    {
        if (pDisp[indexGpu].dispResourceExists(LW_OR_PIOR, i) != TRUE)
        {
            continue;
        }
        dispPrintOwnerProtocol_v02_01(LW_OR_PIOR, i);

        pDisp[indexGpu].dispGetPiorSeqCtlPwrAndBlankRegs(i, NULL, &PiorPwrOffset, &PiorBlankOffset);

        data32 = GPU_REG_RD32(PiorPwrOffset);
        if(DRF_VAL(_PDISP, _PIOR_PWR, _MODE, data32) == LW_PDISP_PIOR_PWR_MODE_SAFE)
        {
            dprintf("SAFE     %-40s", (DRF_VAL(_PDISP, _PIOR_PWR, _SAFE_STATE, data32) == LW_PDISP_PIOR_PWR_SAFE_STATE_PU)? "PU" : "PD");
        }
        else
        {
            dprintf("NORMAL   %-40s", (DRF_VAL(_PDISP, _PIOR_PWR, _NORMAL_STATE, data32) == LW_PDISP_PIOR_PWR_NORMAL_STATE_PU)? "PU" : "PD");
        }
        data32 = GPU_REG_RD32(PiorBlankOffset);
        if(DRF_VAL(_PDISP, _PIOR_BLANK, _STATUS, data32) == LW_PDISP_PIOR_BLANK_STATUS_BLANKED)
        {
            dprintf("YES%s",  (DRF_VAL(_PDISP, _PIOR_BLANK, _OVERRIDE, data32) == LW_PDISP_PIOR_BLANK_OVERRIDE_TRUE)? " (because of override)" : "");
        }
        else
        {
            dprintf("NO");
        }
        dprintf("\n");
    }
}

void dispHeadORConnection_v02_01(void)
{
    CHECK_INIT(MODE_LIVE);

    dprintf("=================================================================================================\n");
    dprintf("OR#     OWNER       PROTOCOL            MODE     STATE   HSYNC   VSYNC   DATA    PWR     BLANKED?\n");
    dprintf("-------------------------------------------------------------------------------------------------\n");

    dispHeadDacConnection_v02_01();
    pDisp[indexGpu].dispHeadSorConnection();
    dispHeadPiorConnection_v02_01();

    dprintf("=================================================================================================\n");
}


// Helper
static int getIdxMatch(int chanNum, char *name)
{
    int i;
    for (i = 0 ; i < mthd_num[chanNum] ; i++) {
        if (!strcmp(name,mthd[chanNum][i].name))
            return i;
    }
    return -1;
}
static int getIdxFldMatch(int chanNum, char *name, int i)
{
    int fi;
    for (fi = 0 ; fi < mthd[chanNum][i].num_fld ; fi++) {
        if (!strcmp(name,mthd[chanNum][i].fld[fi].name))
            return fi;
    }
    return -1;
}

//
// Parses and prints out the Display Push Buffer
//
static void printDispPbParsed_v02_01
(
    PhysAddr baseAddr,
    LwU32 OffsetDwords,
    LwU32 *buffer,
    LwU32 numDwords,
    LwU32 classNum,
    LwU32 getoffset,
    LwU32 putoffset)
{
    LwU32 i;
    LwU32 pbOffset, subdeviceMaskValue;
    PhysAddr addr, get, put;
    LwU32 * bufferstart;

    // 2 dws at least
    if (numDwords < 2)
    {
        numDwords = 8;
    }

    if (buffer == NULL)
    {
        dprintf("lw: Push buffer empty!\n");
        return;
    }

    if (!isValidClassHeader(classNum))
    {
        dprintf("lw: WARNING - Class Header file does not exist for \""
                CLASS_PATH_LOCAL "\"\n", classNum);
        goto EXIT;
    }

    addr = baseAddr + OffsetDwords *4;
    bufferstart = buffer + OffsetDwords;
    get = baseAddr + getoffset;
    put = baseAddr + putoffset;

    // Traverse the buffer by DWORDS
    for (i = 0; i < numDwords; i++)
    {
        LwU32 lwrMethHdr = bufferstart[i];
        LwU32 methCount, methAddr;
        LwU32 opcode;

        dprintf("\n" LwU40_FMT ": %08x\t", addr, lwrMethHdr);

        // The Get and Put pointers could be on a header
        if (addr == put || addr == get)
        {
            dprintf("lw: METHOD HEADER ADDR: " LwU40_FMT, addr);
            if (addr == get) dprintf(" <- _DISP_DMA_GET");
            if (addr == put)
            {
                dprintf(" <- _DISP_DMA_PUT\n");
                dprintf("lw: Parsing ends here\n");
                goto EXIT;
            }
            dprintf("\n");
        }

        // Increment current offset
        addr += 4;

        opcode  = DRF_VAL(_UDISP, _DMA, _OPCODE, lwrMethHdr);

        // JUMP is always to offset 0
        if (opcode == LW_UDISP_DMA_OPCODE_JUMP)
        {
            pbOffset = DRF_VAL(_UDISP, _DMA, _JUMP_OFFSET, lwrMethHdr);
            dprintf("lw: LW_UDISP_DMA_OPCODE_JUMP: OFFSET = 0x%08x\n", pbOffset * 4);
            printDispPbParsed_v02_01(baseAddr, pbOffset / 4, buffer, numDwords - (i+1), classNum, getoffset, putoffset);
            goto EXIT;
        }

        // SET SUBDEVICE MASK
        else if (opcode == LW_UDISP_DMA_OPCODE_SET_SUBDEVICE_MASK)
        {
            subdeviceMaskValue = DRF_VAL(_UDISP, _DMA, _SET_SUBDEVICE_MASK_VALUE, lwrMethHdr);
            dprintf("lw: LW_UDISP_SET_SUBDEVICE_MASK_VALUE: SUBDEVICE MASK VALUE = %x\n", subdeviceMaskValue);
        }

        // Method Header
        else if ((opcode == LW_UDISP_DMA_OPCODE_NONINC_METHOD ||
                  opcode == LW_UDISP_DMA_OPCODE_METHOD))
        {
            methCount = DRF_VAL(_UDISP, _DMA, _METHOD_COUNT, lwrMethHdr);
            methAddr = (DRF_VAL(_UDISP, _DMA, _METHOD_OFFSET, lwrMethHdr) * 4) & 0x1FFC;

            // When the methCount is zero, this is effectively a NOP
            if (!methCount)
            {
                dprintf("lw: 0x0000: NO_OPERATION\n");
            }

            // Parse each method to a data value (multiple data values for one header).
            for (methCount += i; (i+1) <= methCount && (i+1) < numDwords; i++)
            {
                LwU32 data = bufferstart[i + 1];

                dprintf("%08x:", data);

                if (!parseClassHeader(classNum, methAddr, data))
                {
                    dprintf("lw: 0x%04x: DATA: 0x%08x ", methAddr, data);
                }

                if (addr == get) dprintf(" <- _DISP_DMA_GET");
                if (addr == put)
                {
                    dprintf(" <- _DISP_DMA_PUT\n");
                    dprintf("lw: Parsing ends here\n");
                    goto EXIT;
                }

                // Increment current offset
                addr += 4;

                dprintf("\n");

                // Incrementing method
                if (opcode == LW_UDISP_DMA_OPCODE_METHOD)
                {
                    methAddr += 4;
                }
            }
        }
        else
        {
            // need to print out address also here!
            dprintf("lw: INVALID METHOD HEADER: 0x%08x at " LwU40_FMT "\n", lwrMethHdr, addr);
            goto EXIT;
        }
    }

EXIT:
    dprintf("\n");
}

//
// fills given buffer according to the target memory type and offset specified
//
static void getPbData_v02_01
(
    PhysAddr physOffset,
    LwU32 numDwords,
    char *buffer,
    LwU32 mem_Target
)
{
    LwU32 i;
    LwU32 * ptr;

    if ( mem_Target == LW_PDISP_PBCTL0_PUSHBUFFER_TARGET_PHYS_LWM)
    {
        if (buffer)
        {
            if (pFb[indexGpu].fbRead(physOffset, buffer, (LwU32)(numDwords*4)) == LW_ERR_GENERIC)
            {
                dprintf( "lw: ERROR READING VIDEO MEMORY\n");
            }
        }
    }
    else
    {
        if (buffer)
        {
            ptr = (LwU32 *) buffer;
            for (i = 0; i < numDwords; i++)
            {
                *ptr = RD_PHYS32(physOffset + (i * 4));
                ptr ++;
            }
        }
    }
}

void dispDumpChannelState_v02_01(char *chName, LwS32 headNum, LwS32 winNum, BOOL printHeadless, BOOL printRegsWithoutEquivMethod)
{
    LwU32 minChan = 0, maxChan = 0, channel;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        if ( chName != NULL )
        {
            LwS32 chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum);
            if (chNum != -1)
            {
                minChan = chNum;
                maxChan = chNum + 1;
            }
        }

        if (minChan == maxChan)
        {
            minChan = 0;
            maxChan = pDisp[indexGpu].dispGetMaxChan();
        }

        for (channel = minChan; channel < maxChan; channel++)
        {
            if (channel != 0)
            {
                LwU32 chanHead;
                if (pDisp[indexGpu].dispGetChanType(channel, &chanHead) != -1)
                {
                    if (headNum >= 0 && chanHead != headNum)
                        continue;
                }
            }
            pDisp[indexGpu].dispPrintChanMethodState(channel, printHeadless,
                                                     printRegsWithoutEquivMethod,
                                                     headNum, -1 /* window */);
        }
    }
    MGPU_LOOP_END;
}

/**
 * @brief Read dispClk settings.
 *
 * @returns void
 */
void
dispReadDispClkSettings_v02_01()
{
    dprintf("lw: All DispClk settings\n");
    dprintf("lw: DispClk = %4d MHz\n\n", pClk[indexGpu].clkGetDispclkFreqKHz() / 1000);
}


void dispDumpGetDebugMode_v02_01(char *chName, LwS32 headNum, LwU32 dArgc)
{
    ChanDesc_t *desc;
    LwS32 chNum = 0;
    LwS32 ret;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        dprintf("Ch#  Name  Head#    DBG MODE\n");
        dprintf("-----------------------------\n");
        if ( dArgc > 0 )
        {
            if ( (chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
            {
                int i, numheads = 1;

                if (pDisp[indexGpu].dispGetChanDescriptor(chNum, (void**)&desc))
                {
                    dprintf("error in dispGetChanDescriptor\n");
                    return;
                }
                if (!(desc->cap & DISP_DEBUG))
                {
                    dprintf("DebugMode is not available for channel %d\n", chNum);
                    return;
                }

                // if dArgc == 1, print all heads
                if (dArgc == 1)
                {
                    numheads = desc->numHeads;
                }

                for (i = 0; i < numheads ; i++)
                {
                    ret=pDisp[indexGpu].dispGetDebugMode(chNum + i);
                    dprintf("#%d   %s   %d       %8s\n",chNum + i,  chName, i, ret ? "ENABLED":"DISABLED");
                }
            }
            else
                dprintf("lw: Usage: !lw.dgetdbgmode [chName] [-h<hd>]\n");
        }
        else
        {
            LwU32 i, k;
            k = pDisp[indexGpu].dispGetMaxChan();
            for (i = 0; i < k; i++)
            {
                pDisp[indexGpu].dispGetChanDescriptor(i, (void**)&desc);
                if (!(desc->cap & DISP_DEBUG))
                    continue;
                ret=pDisp[indexGpu].dispGetDebugMode(i);
                if (ret >= 0)
                    dprintf("#%d   %s   %d       %8s\n",i,  desc->name, desc->headNum, ret ? "ENABLED":"DISABLED");
            }
        }
    }
    MGPU_LOOP_END;
}

void dispDumpSetDebugMode_v02_01(char *chName, LwS32 headNum, LwU32 debugMode)
{
    LwS32 chNum = 0;

    CHECK_INIT(MODE_LIVE);

    MGPU_LOOP_START;
    {
        if ((chNum = pDisp[indexGpu].dispGetChanNum(chName, headNum)) != -1)
            if (debugMode != pDisp[indexGpu].dispGetDebugMode(chNum))
                pDisp[indexGpu].dispSetDebugMode(chNum,  debugMode);
    }
    MGPU_LOOP_END;
}

LwU32 dispGetChannelStateCacheValue_v02_01(LwU32 chNum, BOOL isArmed, LwU32 offset)
{
    LwU32 headNum = dispChanState_v02_01[chNum].headNum;
    LwU32 baseAddr  = isArmed ? LW_UDISP_DSI_CHN_ARMED_BASEADR(headNum) : LW_UDISP_DSI_CHN_ASSY_BASEADR(headNum);
    return GPU_REG_RD32(baseAddr + offset);
}
