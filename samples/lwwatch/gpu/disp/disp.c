/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2006-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// disp.c
//
//*****************************************************

#include "disp.h"
#include "dpaux.h"
#include "displayport.h"
#include "ctrl/ctrl0073/ctrl0073dp.h"


char *dispGetStringForOrProtocol(LwU32 orType, ORPROTOCOL orProtocol)
{
    switch (orType)
    {
        case LW_OR_SOR:
            switch (orProtocol)
            {
                case sorProtocol_LvdsLwstom:
                    return "LVDS_LWSTOM"; 

                case sorProtocol_SingleTmdsA:
                    return "SINGLE_TMDS_A";

                case sorProtocol_SingleTmdsB:
                    return "SINGLE_TMDS_B";

                case sorProtocol_SingleTmdsAB:
                    return "SINGLE_TMDS_AB";

                case sorProtocol_DualSingleTmds:
                    return "DUAL_SINGLE_TMDS";

                case sorProtocol_DualTmds:
                    return "DUAL_TMDS";

                case sorProtocol_SdiOut:
                    return "SDI_OUT";

                case sorProtocol_DdiOut:
                    return "DDI_OUT";

                case sorProtocol_DpA:
                    return "DP_A";

                case sorProtocol_DpB:
                    return "DP_B";

                case sorProtocol_HdmiFrl:
                    return "HDMI_FRL";

                case sorProtocol_Lwstom:
                    return "CUSTOM";

                default:
                    return "ERROR - this protocol is not supported";
            }

        case LW_OR_PIOR:
            switch (orProtocol)
            {
                case piorProtocol_ExtTmdsEnc:
                    return "EXT_TMDS_ENC";

                case piorProtocol_ExtTvEnc:
                    return "EXT_TV_ENC"; 

                case piorProtocol_ExtSdiSdEnc:
                    return "EXT_SDI_SD_ENC"; 

                case piorProtocol_ExtSdiHdEnc:
                    return "EXT_SDI_HD_ENC"; 

                case piorProtocol_DistRenderOut:
                    return "DIST_RENDER_OUT"; 

                case piorProtocol_DistRenderIn:
                    return "DIST_RENDER_IN"; 

                case piorProtocol_DistRenderInout:
                    return "DIST_RENDER_INOUT"; 

                default:
                    return "ERROR - this protocol is not supported";
            }

        case LW_OR_DAC:
            switch (orProtocol)
            {
                case dacProtocol_RgbCrt:
                    return "RGB_CRT";

                case dacProtocol_CpstNtscM:
                    return "CPST_NTSC_M";

                case dacProtocol_CpstNtscJ:
                    return "CPST_NTSC_J";

                case dacProtocol_CpstPalBdghi:
                    return "CPST_PAL_BDGHI";

                case dacProtocol_CpstPalM:
                    return "CPST_PAL_M";

                case dacProtocol_CpstPalN:
                    return "CPST_PAL_N";

                case dacProtocol_CpstPalCn:
                    return "CPST_PAL_CN";

                case dacProtocol_CompNtscM:
                    return  "COMP_NTSC_M";

                case dacProtocol_CompNtscJ:
                    return "COMP_NTSC_J";

                case dacProtocol_CompPalBdghi:
                    return "COMP_PAL_BDGHI";

                case dacProtocol_CompPalM:
                    return "COMP_PAL_M";

                case dacProtocol_CompPalN:
                    return "COMP_PAL_N";

                case dacProtocol_CompPalCn:
                    return "COMP_PAL_CN";

                case dacProtocol_Comp480p60:
                    return "COMP_480P_60";

                case dacProtocol_Comp576p50:
                    return "COMP_576P_50";

                case dacProtocol_Comp720p50:
                    return "COMP_720P_50";

                case dacProtocol_Comp720p60:
                    return "COMP_720P_60";

                case dacProtocol_Comp1080i50:
                    return "COMP_1080I_50";

                case dacProtocol_Comp1080i60:
                    return "COMP_1080I_60";

                case dacProtocol_YuvCrt:
                    return "YUV_CRT";

                case dacProtocol_Lwstom:
                    return "CUSTOM";

                default:
                    return "ERROR - this protocol is not supported";
        }

        default:
            return "ERROR";
    }
}

char* dispGetORString(LwU32 orType)
{
    switch (orType)
    {
        case LW_OR_SOR:
            return "SOR";

        case LW_OR_PIOR:
            return "PIOR";

        case LW_OR_DAC:
            return "DAC";

        default:
            return " Bad OR type ";
    }
}


const char* dispGetPadLinkString(PADLINK padLink)
{
    switch (padLink)
    {
        case PADLINK_A:
            return "PADLINK_A";

        case PADLINK_B:
            return "PADLINK_B";

        case PADLINK_C:
            return "PADLINK_C";

        case PADLINK_D:
            return "PADLINK_D";

        case PADLINK_E:
            return "PADLINK_E";

        case PADLINK_F:
            return "PADLINK_F";

        case PADLINK_G:
            return "PADLINK_G";

        case PADLINK_NONE:
            return "None";

        default:
            return "Error";
    }
}

/*!
 *  Default Lwwatch stub spews out "Unsupported chip"
 *  Use this instead. 

 *  @param[in]  orType      OR type (DAC/SOR/PIOR)
 *  @param[in]  index       OR index 
 *
 *  @return   TRUE/FALSE
 */
BOOL dispResourceExists_STUB(
    LWOR          orType,
    LwU32         index
)
{
    return TRUE;
}

#define PRINTLOOP(ch,emptySpaces,oclwrences) do{loopVar = oclwrences; if(emptySpaces) \
                 dprintf("%*c",emptySpaces,' '); while(loopVar--){dprintf("%c",ch);}}while(0);
#define PRINTNLOOP(ch1,ch2,emptySpaces,space,oclwrences) do{loopVar = oclwrences; \
                  while(loopVar--){ if(emptySpaces)dprintf("%*c",emptySpaces,' '); dprintf("%c%*c\n",ch1,space,ch2);}}while(0);

void dispDrawAsciiOrConnDiagram
(
    LWOR orType,
    char *orString,
    LwU32 *ownerMasks,
    LwU32 *headDisplayIds
)
{
    LwU32 orNum=0, ownerMask, head, displayCode;    
    LwU32 loopVar=0, initEmptySpaces=8, numOfORSpaces=8, distBtOROD=8, distBtORHD=2;


    for (orNum = 0; orNum < pDisp[indexGpu].dispGetNumOrs(orType); orNum++) 
    {
        ownerMask = ownerMasks[orNum];
        if ((pDisp[indexGpu].dispResourceExists(orType, orNum) != TRUE) || (!ownerMask))
        {
            continue;
        }
        PRINTLOOP('-', initEmptySpaces, numOfORSpaces);
        dprintf("%s%d",orString,orNum);
        PRINTLOOP('-', 0, distBtOROD+1);
        dprintf("\n");

        for (head = 0; head < pDisp[indexGpu].dispGetNumHeads(); ++head)
        {
            if (BIT(head) & ownerMask)
            {
                displayCode = headDisplayIds[head];
                PRINTNLOOP('|', '|', initEmptySpaces, numOfORSpaces+distBtOROD+4, distBtORHD);        
                dprintf("%*c", initEmptySpaces+1, '|');
                PRINTLOOP('-', 0, distBtORHD+1);
                dprintf("%s%d","HD",head);
                if(displayCode)
                {
                    PRINTLOOP('.', 0, numOfORSpaces+distBtOROD-3);        
                    dprintf("DI-%d\n",displayCode);
                }
                else
                {
                    dprintf("%*c\n",numOfORSpaces+distBtOROD-2,'*');
                }
            }   
        }
        dprintf("\n");
    }
}

void dispDrawAsciiOrConn
(
    asciiOrConnData asciiData
)
{
    dprintf("\n                            *** ASCII DIAGRAMS ***\n");
    dprintf("-------------------------------------------------------------------------------------------------\n");
    dispDrawAsciiOrConnDiagram(LW_OR_SOR, "SOR", asciiData.sorOwnerMasks, asciiData.headDisplayIds);
    dispDrawAsciiOrConnDiagram(LW_OR_DAC, "DAC", asciiData.dacOwnerMasks, asciiData.headDisplayIds);
    dispDrawAsciiOrConnDiagram(LW_OR_PIOR, "PIOR", asciiData.piorOwnerMasks, asciiData.headDisplayIds);
    dprintf("=================================================================================================\n");
}


void dispHeadORConnectionAscii(void)
{
    asciiOrConnData asciiData;
    memset(&asciiData, 0x0, sizeof(asciiData));
    pDisp[indexGpu].dispHeadOrConnAsciiData(LW_OR_SOR, asciiData.headDisplayIds, asciiData.sorOwnerMasks);
    pDisp[indexGpu].dispHeadOrConnAsciiData(LW_OR_DAC, asciiData.headDisplayIds, asciiData.dacOwnerMasks);
    pDisp[indexGpu].dispHeadOrConnAsciiData(LW_OR_PIOR, asciiData.headDisplayIds, asciiData.piorOwnerMasks);
    dispDrawAsciiOrConn(asciiData);
}

/*!
 * @brief Helper function to Initialize SLI Data.
 * 
 * @param[in] pDsliPrintData    DSLI_PRINT_PARAM pointer - For filling Data in
 * datastructure
 */
void dispInitializeSliData
(
    DSLI_PRINT_PARAM *pDsliPrintData
)
{
    // Set all parameter to default value
    pDsliPrintData->headStatus = "In-Active";
    pDsliPrintData->slaveLock = "N/A";
    pDsliPrintData->slaveLockPin = "-N/A";
    pDsliPrintData->masterLock = "N/A";
    pDsliPrintData->masterLockPin = "-N/A";
    pDsliPrintData->scanLockStatus = "N/A";
    pDsliPrintData->flipLock = "N/A";
    pDsliPrintData->flipLockStatus = "N/A";
    pDsliPrintData->syncAdvance = 0x0;
    pDsliPrintData->refClkForVpll = "XTAL";
}


/*!
 * @brief dispPrintSliData - Function to print SLI config data, 
 * used by DSLI. It prints SLI register values for configuration 
 * 
 *  @param[in]  LwU32               numHead         Number of Heads
 *  @param[in]  LwU32               numPior         Number of Piors
 *  @param[in]  LwU32               numPin          Number of Pins
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  datastructure
 * 
 *  @param[in]  DSLI_PIOR_DATA      *pDsliPiorData  Pointer to PIOR
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM    *pDsliPrintData  Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Display all Values
 *                                                  if required
 */

void dispPrintSliData
(
    LwU32               numHead,
    LwU32               numPior,
    LwU32               numPin,
    DSLI_DATA           *pDsliData,
    DSLI_PIOR_DATA      *pDsliPiorData,
    DSLI_PRINT_PARAM    *pDsliPrintData,
    LwU32               verbose
)
{
    LwU32 head = 0;

    dprintf("                                          RESULTS                                                       \n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    
    dprintf("========================================================================================================\n");
    dprintf("%-40s", "Parameter");
    
    for(head = 0; head < numHead; ++head)
    {
        dprintf("Head-%d               ", head);
    }
    dprintf("\n");
    dprintf("--------------------------------------------------------------------------------------------------------\n");
    
    // Print the data for each head in tabular format
    pDisp[indexGpu].dispPrintHeadData(numHead, pDsliData, pDsliPrintData, verbose);

    // Print the PIN config info
    pDisp[indexGpu].dispPrintPinData(numPin, pDsliPiorData);
    
    // Print the PIOR config info
    pDisp[indexGpu].dispHeadORConnection();

    pDisp[indexGpu].dispPrintSliRegisters(numHead, numPior, pDsliData, pDsliPiorData);

    dprintf("--------------------------------------------------------------------------------------------------------\n");
}

/*!
 * @brief dispPrintSliStatus - Function to print SLI-Data on screen.
 * 
 * 
 *  @param[in]  LwU32               numHead         Number of Heads
 *  @param[in]  DSLI_DATA           *pDsliData      Pointer to DSLI
 *                                                  datastructure
 *  @param[in]  DSLI_PRINT_PARAM    *pDsliPrintData  Pointer to print
 *                                                  Param datastructure
 *  @param[in]  LwU32               verbose         Verbose switch
 */
void dispPrintSliStatus
(
    LwU32               numHead,
    DSLI_DATA           *pDsliData,
    DSLI_PRINT_PARAM    *pDsliPrintData,
    LwU32               verbose
)
{
    LwU32 head = 0;

    PRINTHEADTABLE("SLI Head Status", pDsliData, pDsliPrintData, headStatus, verbose, DsliHeadActive);
    PRINTLOCKTABLE("Slave-Lock-Setup", pDsliData, pDsliPrintData, slaveLock, slaveLockPin, verbose, DsliHeadSetSlaveLockMode);
    PRINTLOCKTABLE("Master-Lock-Setup", pDsliData, pDsliPrintData, masterLock, masterLockPin, verbose, DsliHeadSetMasterLockMode);
    PRINTHEADTABLE("Scan-Lock Status of RG", pDsliData, pDsliPrintData, scanLockStatus, verbose, DsliRgStatusLocked);
    PRINTHEADTABLE("Flip-Lock", pDsliData, pDsliPrintData, flipLock, verbose, DsliRgStatusFlipLocked);
    PRINTHEADTABLE("Flip Lock Status of RG", pDsliData, pDsliPrintData, flipLockStatus, verbose, DsliRgStatusFlipLocked);
    PRINTSYNCVAL("Sync Advance (RG)", pDsliData, pDsliPrintData, syncAdvance, verbose, DsliRgDistRndrSyncAdv);
    PRINTHEADTABLE("Ref Clk Src for VPLL", pDsliData, pDsliPrintData, refClkForVpll, verbose, DsliClkDriverSrc);
}

/*!
 * @brief dispFindHeaderNum - Function to populate the classHeaderNum array.
 * 
 * 
 *  @param[out] LwU32               *header         Pointer to return class name
 *  @param[in]  char                *classNames     Pointer to class names buffer
 *  @param[in]  char                separator       Separator character
 *  @param[in]  int                 *chan_template  Template array to find the match
 *  @param[in]  char                n               number of template
 */
LwU32 dispFindHeaderNum(LwU32 *header, char *classNames, char separator, char *chan_template, int n)
{
    char lwr_class[32];

    assert(n < 32);

    if (findDisplayClass(classNames, separator, chan_template, n, lwr_class)==0)
    {
        dprintf("\nLwrrent Class not found for template \"%s\"\n", chan_template);
        return 0;
    }
    else
    {
        sscanf((lwr_class+2), "%x", header);
    }
    return 1;
}

/*!
 * @brief findDisplayClass - Function to find the current display class.
 * 
 * 
 *  @param[in]  char                *classNames     Pointer to class names buffer
 *  @param[in]  char                separator       Separator character
 *  @param[in]  char                *chan_template  Template array to find the match
 *  @param[in]  int                 n               number of template
 *  @param[out] char                *lwr_class      Pointer to return class name buffer
 */
LwU32 findDisplayClass(char *classNames, char separator, char *chan_template, int n, char *lwr_class)
{
    int counter = 0;
    int found = 0;
    char classes[256];
    char *tmp;
 
    memcpy(classes, classNames, 256);

    if (classes[0] == '\0' || chan_template == NULL || chan_template[0] == '\0')
    {
        char *tmpScPath = getelw("LWW_MANUAL_SDK");
        dprintf("lw: Invalid class names or class template name, they can NOT be NULL\n");
        dprintf("lw: LWW_MANUAL_SDK=\"%s\"\n",tmpScPath?tmpScPath:"");
        dprintf("lw: [classes[0]=0x%X, chan_template=0x%p, chan_template[0]=0x%X]\n",
                classes[0], chan_template,
                ((chan_template) ? chan_template[0] : '?'));
        return 0;
    }
    
    tmp = strtok(classes, ";");

    memset(lwr_class, 0, n + 1);
    while (1)
    {
        counter = 0;
        found = 1;
        while (counter < n)
        {
            if (*(chan_template+counter) == '*')
            {
                *(lwr_class+counter) = *(tmp+counter);
                counter++;
            }
            else if (*(chan_template+counter) == *(tmp+counter) || 
                    toupper((int)(*(chan_template+counter))) == toupper((int)(*(tmp+counter))))
            {
                *(lwr_class+counter) = *(tmp+counter);
                counter++;
            }
            else
            {
                found = 0;
                break;
            }
        }
        if (found == 1)
        {
            break;
        }
        tmp = strtok(NULL, ";");
        if (tmp == NULL)
        {
            dprintf("No %s class found in state cache file\n", chan_template);
            return 0;
        }
    }
    return 1;
}

/*!
 * @brief findDisplayClass - Function to initialize the classHeaderNum array from 
 *                           which the appropriate class header is found (used for dispDumpPb_LW50)
 * 
 *  @param[in]  char                *classNames         Pointer to classNames buffer
 *  @param[Out] LwU32               classHeaderNum[]    Array to hold the matching class names
 */
void initializeClassHeaderNum(char *classNames, LwU32 classHeaderNum[])
{
    LwU32 classheaderC, classheaderD, classheaderE;

    if (dispFindHeaderNum(&classheaderC, classNames, ';', "LW***C", 6) != 0 &&
        dispFindHeaderNum(&classheaderD, classNames, ';', "LW***D", 6) != 0 &&
        dispFindHeaderNum(&classheaderE, classNames, ';', "LW***E", 6) != 0 )
    {
        classHeaderNum[0] = classheaderD;
        classHeaderNum[1] = classheaderC;
        classHeaderNum[2] = classheaderE;
    }
    else
    {
        dprintf("Class Headers not initialized\n");
    }
}

void dispPrintDpRxInfo
(
    LwU32 port
)
{
    LwS32 dpcd;
    LwS8 dpcdVer = 0;

    dprintf("\nRx:\n");

    // DPCD revision
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_REV);
    if (dpcd != -1)
    {
        dpcdVer = (LwU8)dpcd;

        dprintf("%-55s: %x\n", "LW_DPCD_REV_MAJOR",
            DRF_VAL(_DPCD, _REV, _MAJOR, dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_REV_MINOR",
            DRF_VAL(_DPCD, _REV, _MINOR, dpcd));
    }

    // Power state
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_SET_POWER);
    if (dpcd != -1)
    {
        char *pPwrState;

        switch (DRF_VAL(_DPCD, _SET_POWER, _VAL, dpcd))
        {
            case LW_DPCD_SET_POWER_VAL_D0_NORMAL:
                pPwrState = "D0";
                break;
            case LW_DPCD_SET_POWER_VAL_D3_PWRDWN:
                pPwrState = "D3_PWRWN";
                break;
            case LW_DPCD_SET_POWER_VAL_D3_AUX_ON:
                pPwrState = "D3_AUX_ON";
                break;
            default:
                pPwrState = "UNKNOWN";
                break;
        }
        dprintf("%-55s: %s\n", "LW_DPCD_SET_POWER", pPwrState);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_SINK_COUNT);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_SINK_COUNT_VAL",
                LW_DPCD_SINK_COUNT_VAL(dpcd));

        dprintf("%-55s: %s\n", "LW_DPCD_SINK_COUNT_CP_READY",
            DRF_VAL(_DPCD, _SINK_COUNT, _CP_READY, dpcd) ? "YES" : "NO");
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port,
                                        LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n",
            "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_REMOTE_CTRL",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _REMOTE_CTRL,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n",
            "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_AUTO_TEST",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _AUTO_TEST,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_CP",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _CP,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n",
            "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_MCCS_IRQ",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _MCCS_IRQ,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n",
            "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_DOWN_REP_MSG",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _DOWN_REP_MSG_RDY,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n",
            "LW_DPCD_DEVICE_SERVICE_IRQ_VECTOR_SINK_SPECIFIC",
            DRF_VAL(_DPCD, _DEVICE_SERVICE_IRQ_VECTOR, _SINK_SPECIFIC_IRQ,
            dpcd) ? "YES" : "NO");
    }

    // Training pattern set
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_TRAINING_PATTERN_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_PATTERN_SET_TPS",
            DRF_VAL(_DPCD, _TRAINING_PATTERN_SET, _TPS, dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_PATTERN_SET_LQPS",
            DRF_VAL(_DPCD, _TRAINING_PATTERN_SET, _LQPS, dpcd));

        dprintf("%-55s: %x\n",
            "LW_DPCD_TRAINING_PATTERN_SET_RECOVERED_CLOCK_OUT_EN",
            DRF_VAL(_DPCD, _TRAINING_PATTERN_SET,
            _RECOVERED_CLOCK_OUT_EN, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_PATTERN_SET_SCRAMBLING_DISABLED",
            DRF_VAL(_DPCD, _TRAINING_PATTERN_SET, _SCRAMBLING_DISABLED, dpcd) ?
            "TRUE" : "FALSE");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_TRAINING_PATTERN_SET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_TRAINING_LANE0_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE0_SET_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE0_SET_DRIVE_LWRRENT_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT_MAX_REACHED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE0_SET_PREEMPHASIS",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE0_SET_PREEMPHASIS_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS_MAX_REACHED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE0_SET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_TRAINING_LANE1_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE1_SET_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE1_SET_DRIVE_LWRRENT_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT_MAX_REACHED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE1_SET_PREEMPHASIS",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE1_SET_PREEMPHASIS_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS_MAX_REACHED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE1_SET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_TRAINING_LANE2_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE2_SET_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE2_SET_DRIVE_LWRRENT_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT_MAX_REACHED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE2_SET_PREEMPHASIS",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE2_SET_PREEMPHASIS_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS_MAX_REACHED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE2_SET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_TRAINING_LANE3_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE3_SET_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE3_SET_DRIVE_LWRRENT_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _DRIVE_LWRRENT_MAX_REACHED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE3_SET_PREEMPHASIS",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS, dpcd));

        dprintf("%-55s: %s\n",
            "LW_DPCD_TRAINING_LANE3_SET_PREEMPHASIS_MAX_REACHED",
            DRF_VAL(_DPCD, _TRAINING_LANEX_SET, _PREEMPHASIS_MAX_REACHED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE3_SET read failed\n",
            __FUNCTION__);
    }

    // Show current adjust request for lanes
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LANE0_1_ADJUST_REQ);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n",
            "LW_DPCD_LANE0_1_ADJUST_REQ_LANE0_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEX, _DRIVE_LWRRENT,
            dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_LANE0_1_ADJUST_REQ_LANE0_PREEMPHASIS",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEX, _PREEMPHASIS,
            dpcd));

        dprintf("%-55s: %x\n",
            "LW_DPCD_LANE0_1_ADJUST_REQ_LANE1_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEXPLUS1, _DRIVE_LWRRENT,
            dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_LANE0_1_ADJUST_REQ_LANE1_PREEMPHASIS",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEXPLUS1, _PREEMPHASIS,
            dpcd));
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LANE0_1_ADJUST_REQ read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LANE2_3_ADJUST_REQ);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n",
            "LW_DPCD_LANE2_3_ADJUST_REQ_LANE2_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEX, _DRIVE_LWRRENT,
            dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_LANE2_3_ADJUST_REQ_LANE2_PREEMPHASIS",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEX, _PREEMPHASIS,
            dpcd));

        dprintf("%-55s: %x\n",
            "LW_DPCD_LANE2_3_ADJUST_REQ_LANE3_DRIVE_LWRRENT",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEXPLUS1, _DRIVE_LWRRENT,
            dpcd));

        dprintf("%-55s: %x\n", "LW_DPCD_LANE2_3_ADJUST_REQ_LANE3_PREEMPHASIS",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_ADJUST_REQ_LANEXPLUS1, _PREEMPHASIS,
            dpcd));
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LANE2_3_ADJUST_REQ read failed\n",
            __FUNCTION__);
    }

    // POST_LWRSOR_2 relevant.
    if (dpcdVer >= DPCD_VERSION_12)
    {
        dpcd = pDpaux[indexGpu].dpauxChRead(port,
                                            LW_DPCD_TRAINING_LANE0_1_SET2);
        if (dpcd != -1)
        {
            dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE0_1_SET2_LANE0",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEX_SET2,
                _POST_LWRSOR2, dpcd));

            dprintf("%-55s: %s\n",
                "LW_DPCD_TRAINING_LANE0_1_SET2_LANE0_MAX_REACHED",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEX_SET2,
                _POST_LWRSOR2_MAX_REACHED, dpcd) ? "YES" : "NO");

            dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE0_1_SET2_LANE1",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEXPLUS1_SET2,
                _POST_LWRSOR2, dpcd));

            dprintf("%-55s: %s\n",
                "LW_DPCD_TRAINING_LANE0_1_SET2_LANE1_MAX_REACHED",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEXPLUS1_SET2,
                _POST_LWRSOR2_MAX_REACHED, dpcd) ? "YES" : "NO");
        }
        else
        {
            dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE0_1_SET2 read failed\n",
                __FUNCTION__);
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(port,
                                            LW_DPCD_TRAINING_LANE2_3_SET2);
        if (dpcd != -1)
        {
            dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE2_3_SET2_LANE2",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEX_SET2,
                _POST_LWRSOR2, dpcd));

            dprintf("%-55s: %s\n",
                "LW_DPCD_TRAINING_LANE2_3_SET2_LANE2_MAX_REACHED",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEX_SET2,
                _POST_LWRSOR2_MAX_REACHED, dpcd) ? "YES" : "NO");

            dprintf("%-55s: %x\n", "LW_DPCD_TRAINING_LANE2_3_SET2_LANE3",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEXPLUS1_SET2,
                        _POST_LWRSOR2, dpcd));

            dprintf("%-55s: %s\n",
                "LW_DPCD_TRAINING_LANE2_3_SET2_LANE3_MAX_REACHED",
                DRF_VAL(_DPCD, _LANEX_XPLUS1_TRAINING_LANEXPLUS1_SET2,
                _POST_LWRSOR2_MAX_REACHED, dpcd) ? "YES" : "NO");
        }
        else
        {
            dprintf("ERROR: %s: LW_DPCD_TRAINING_LANE2_3_SET2 read failed\n",
                __FUNCTION__);
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(port,
                                            LW_DPCD_ADJUST_REQ_POST_LWRSOR2);
        if (dpcd != -1)
        {
            dprintf("%-55s: %x\n", "LW_DPCD_ADJUST_REQ_POST_LWRSOR2_LANE0",
                DRF_VAL(_DPCD, _ADJUST_REQ_POST_LWRSOR2, _LANE0, dpcd));

            dprintf("%-55s: %x\n", "LW_DPCD_ADJUST_REQ_POST_LWRSOR2_LANE1",
                DRF_VAL(_DPCD, _ADJUST_REQ_POST_LWRSOR2, _LANE1, dpcd));

            dprintf("%-55s: %x\n", "LW_DPCD_ADJUST_REQ_POST_LWRSOR2_LANE2",
                DRF_VAL(_DPCD, _ADJUST_REQ_POST_LWRSOR2, _LANE2, dpcd));

            dprintf("%-55s: %x\n", "LW_DPCD_ADJUST_REQ_POST_LWRSOR2_LANE3",
                DRF_VAL(_DPCD, _ADJUST_REQ_POST_LWRSOR2, _LANE3, dpcd));
        }
        else
        {
            dprintf("ERROR: %s: LW_DPCD_ADJUST_REQ_POST_LWRSOR2 read failed\n",
                __FUNCTION__);
        }
    }

    // Enhanced Framing & Lanecount
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LANE_COUNT_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n", "LW_DPCD_LANE_COUNT_SET_ENHANCEDFRAMING",
            DRF_VAL(_DPCD, _LANE_COUNT_SET, _ENHANCEDFRAMING, dpcd) ?
            "TRUE" : "FALSE");

        if (DRF_VAL(_DPCD, _LANE_COUNT_SET, _LANE, dpcd))
        {
            dprintf("%-55s: %x\n", "LW_DPCD_LANE_COUNT_SET_LANE",
                DRF_VAL(_DPCD, _LANE_COUNT_SET, _LANE, dpcd));
        }
        else
        {
            dprintf("Error: %s: Bad LW_DPCD_LANE_COUNT_SET_LANE value.\n",
                __FUNCTION__);
        }
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LANE_COUNT_SET read failed\n",
            __FUNCTION__);
    }

    // Link Bandwidth
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LINK_BANDWIDTH_SET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_LINK_BANDWIDTH_SET_VAL",
            DRF_VAL(_DPCD, _LINK_BANDWIDTH_SET, _VAL, dpcd));
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LINK_BANDWIDTH_SET read failed\n",
            __FUNCTION__);
    }

    // Lane Status
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LANE0_1_STATUS);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n", "LW_DPCD_LANE0_STATUS_CR_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _CR_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE0_STATUS_CHN_EQ_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _CHN_EQ_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE0_STATUS_SYMBOL_LOCKED",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _SYMBOL_LOCKED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE1_STATUS_CR_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _CR_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE1_STATUS_CHN_EQ_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _CHN_EQ_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE1_STATUS_SYMBOL_LOCKED",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _SYMBOL_LOCKED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LANE0_1_STATUS read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_LANE2_3_STATUS);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n", "LW_DPCD_LANE2_STATUS_CR_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _CR_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE2_STATUS_CHN_EQ_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _CHN_EQ_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE2_STATUS_SYMBOL_LOCKED",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEX, _SYMBOL_LOCKED,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE3_STATUS_CR_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _CR_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE3_STATUS_CHN_EQ_DONE",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _CHN_EQ_DONE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_LANE3_STATUS_SYMBOL_LOCKED",
            DRF_VAL(_DPCD, _LANEX_XPLUS1_STATUS_LANEXPLUS1, _SYMBOL_LOCKED,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_LANE2_3_STATUS read failed\n",
            __FUNCTION__);
    }

    // MST relevant.
    if (dpcdVer >= DPCD_VERSION_12)
    {
        LwU32 i;
        dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_MSTM);
        if (dpcd != -1)
        {
            dprintf("%-55s: %s\n", "LW_DPCD_MSTM_CAP",
                DRF_VAL(_DPCD, _MSTM, _CAP, dpcd) ? "YES" : "NO");
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_MSTM_CTRL);
        if (dpcd != -1)
        {
            dprintf("%-55s: %s\n", "LW_DPCD_MSTM_CTRL_EN",
                DRF_VAL(_DPCD, _MSTM_CTRL, _EN, dpcd) ? "YES" : "NO");

            dprintf("%-55s: %s\n", "LW_DPCD_MSTM_CTRL_UP_REQ_EN",
                DRF_VAL(_DPCD, _MSTM_CTRL, _UP_REQ_EN,
                dpcd) ? "YES" : "NO");

            dprintf("%-55s: %s\n", "LW_DPCD_MSTM_CTRL_UPSTREAM_IS_SRC",
                DRF_VAL(_DPCD, _MSTM_CTRL, _UPSTREAM_IS_SRC,
                dpcd) ? "YES" : "NO");
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(port,
            LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS);
        if (dpcd != -1)
        {
            dprintf("%-55s: %s\n",
                "LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS_UPDATED",
                DRF_VAL(_DPCD, _PAYLOAD_TABLE_UPDATE_STATUS, _UPDATED,
                dpcd) ? "YES" : "NO");

            dprintf("%-55s: %s\n",
                "LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS_ACT_HANDLED",
                DRF_VAL(_DPCD, _PAYLOAD_TABLE_UPDATE_STATUS, _ACT_HANDLED,
                dpcd) ? "YES" : "NO");
        }

        dprintf("%-55s:\n", "LW_DPCD_VC_PAYLOAD_ID_SLOT");
        dprintf(" 1 ~ 31:   ");
        for (i = 0; i < LW_DPCD_VC_PAYLOAD_ID_SLOT__SIZE; i++)
        {
            if (i == (LW_DPCD_VC_PAYLOAD_ID_SLOT__SIZE / 2))
            {
                dprintf("\n32 ~ 63: ");
            }

            dpcd = pDpaux[indexGpu].dpauxChRead(port,
                                            LW_DPCD_VC_PAYLOAD_ID_SLOT(i));
            if (dpcd != -1)
            {
                dprintf("%x ", dpcd);
            }
            else
            {
                dprintf("ERROR: LW_DPCD_VC_PAYLOAD_ID_SLOT_%d read failed", i);
                break;
            }
        }
        dprintf("\n");
    }

    // HDCP regs
    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_BKSV_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_HDCP_BKSV_OFFSET", (LwU8) dpcd);
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_BKSV_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_RPRIME_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_HDCP_RPRIME_OFFSET", (LwU8) dpcd);
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_RPRIME_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_AKSV_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_HDCP_AKSV_OFFSET", (LwU8) dpcd);
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_AKSV_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_AN_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_HDCP_AN_OFFSET", (LwU8) dpcd);
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_AN_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_VPRIME_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %x\n", "LW_DPCD_HDCP_VPRIME_OFFSET", (LwU8) dpcd);
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_VPRIME_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_BCAPS_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n", "LW_DPCD_HDCP_BCAPS_OFFSET_HDCP_CAPABLE",
            DRF_VAL(_DPCD, _HDCP_BCAPS_OFFSET, _HDCP_CAPABLE,
            dpcd) ? "YES" : "NO");

        dprintf("%-55s: %s\n", "LW_DPCD_HDCP_BCAPS_OFFSET_HDCP_REPEATER",
            DRF_VAL(_DPCD, _HDCP_BCAPS_OFFSET, _HDCP_REPEATER,
            dpcd) ? "YES" : "NO");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_BCAPS_OFFSET read failed\n",
            __FUNCTION__);
    }

    dpcd = pDpaux[indexGpu].dpauxChRead(port, LW_DPCD_HDCP_BSTATUS_OFFSET);
    if (dpcd != -1)
    {
        dprintf("%-55s: %s\n", "LW_DPCD_HDCP_BSTATUS_READY",
            DRF_VAL(_DPCD, _HDCP_BSTATUS, _READY, dpcd) ? "TRUE" : "FALSE");

        dprintf("%-55s: %s\n", "LW_DPCD_HDCP_BSTATUS_RPRIME_AVAILABLE",
            DRF_VAL(_DPCD, _HDCP_BSTATUS, _RPRIME_AVAILABLE,
            dpcd) ? "TRUE" : "FALSE");

        dprintf("%-55s: %s\n", "LW_DPCD_HDCP_BSTATUS_LINK_INTEGRITY_FAILURE",
            DRF_VAL(_DPCD, _HDCP_BSTATUS, _LINK_INTEGRITY_FAILURE,
            dpcd) ? "TRUE" : "FALSE");
    }
    else
    {
        dprintf("ERROR: %s: LW_DPCD_HDCP_BSTATUS_OFFSET read failed\n",
            __FUNCTION__);
    }

    dprintf("================================================================================\n");
}

void dispPrintDpRxEnum(void)
{
    LwS32 numAuxPort;
    LwS32 dpcd, ver, lanes, rate, lane0_1, lane2_3;
    BOOL bMstCap, bMstEn;
    char *pPwrState;
    LwU8 i, j;

    dprintf("Rx:\n"
            "-----------------------------------------------------------------------------------------------------------------\n"
            "AUXPORT  HPD    DPCD_VER  MST_CAP  POWER_STATE  MODE:RATE@LANES  LANE_STATUS      TIME_SLOT(STATUS&STREAM_LENGTH)\n"
            "-----------------------------------------------------------------------------------------------------------------\n");

    numAuxPort = pDisp[indexGpu].dispGetNumAuxPorts();
    for (i = 0; i < numAuxPort; i++)
    {
        if (!pDpaux[indexGpu].dpauxGetHpdStatus(i))
        {
            dprintf("%-9dLOW\n", i);
            continue;
        }

        if (!pDpaux[indexGpu].dpauxIsPadPowerUpForPort(i))
        {
            dprintf("%-9dHIGH - pad off\n", i);
            continue;
        }

        if (!pDpaux[indexGpu].dpauxHybridAuxInDpMode(i))
        {
            dprintf("%-9dHIGH - I2C mode\n", i);
            continue;
        }

        ver = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_REV);
        if (ver == -1)
        {
            dprintf("auxport%d LW_DPCD_REV(0x%x) read failed\n", i, LW_DPCD_REV);
            break;
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_SET_POWER);
        if (ver == -1)
        {
            dprintf("auxport%d LW_DPCD_SET_POWER(0x%x) read failed\n", i,
                LW_DPCD_SET_POWER);
            break;
        }
        switch (DRF_VAL(_DPCD, _SET_POWER, _VAL, dpcd))
        {
            case LW_DPCD_SET_POWER_VAL_D0_NORMAL:
                pPwrState = "D0";
                break;
            case LW_DPCD_SET_POWER_VAL_D3_PWRDWN:
                pPwrState = "D3_PWRWN";
                break;
            case LW_DPCD_SET_POWER_VAL_D3_AUX_ON:
                pPwrState = "D3_AUX_ON";
                break;
            default:
                pPwrState = "UNKNOWN";
                break;
        }

        dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_LINK_BANDWIDTH_SET);
        if (dpcd == -1)
        {
            dprintf("auxport%d LW_DPCD_LINK_BANDWIDTH_SET(0x%x) read failed\n",
                i, LW_DPCD_LINK_BANDWIDTH_SET);
            break;
        }
        // Get rate to 10Mbps units.
        rate = dpcd * 27;

        dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_LANE_COUNT_SET);
        if (dpcd == -1)
        {
            dprintf("auxport%d LW_DPCD_LANE_COUNT_SET(0x%x) read failed\n",
                i, LW_DPCD_LANE_COUNT_SET);
            break;
        }
        lanes = DRF_VAL(_DPCD, _LANE_COUNT_SET, _LANE, dpcd);

        dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_MSTM);
        if (dpcd == -1)
        {
            dprintf("auxport%d LW_DPCD_MSTM(0x%x) read failed\n",
                i, LW_DPCD_MSTM);
            break;
        }
        bMstCap = FLD_TEST_DRF(_DPCD, _MSTM, _CAP, _YES, dpcd);

        dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_MSTM_CTRL);
        if (dpcd == -1)
        {
            dprintf("auxport%d LW_DPCD_MSTM_CTRL(0x%x) read failed\n",
                i, LW_DPCD_MSTM_CTRL);
            break;
        }
        bMstEn = FLD_TEST_DRF(_DPCD, _MSTM_CTRL, _EN, _YES, dpcd);

        lane0_1 = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_LANE0_1_STATUS);
        if (lane0_1 == -1)
        {
            dprintf("auxport%d LW_DPCD_LANE0_1_STATUS(0x%x) read failed\n",
                i, LW_DPCD_MSTM);
            break;
        }

        lane2_3 = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_LANE2_3_STATUS);
        if (lane2_3 == -1)
        {
            dprintf("auxport%d LW_DPCD_LANE2_3_STATUS(0x%x) read failed\n",
                i, LW_DPCD_LANE2_3_STATUS);
            break;
        }

        dprintf("%-9dHIGH   %d.%-8d%-9s%-13s%s:%d.%-2dGbps@%-4d%x:%x:%x:%-11x",
            i, DRF_VAL(_DPCD, _REV, _MAJOR, ver),
            DRF_VAL(_DPCD, _REV, _MINOR, ver),
            (bMstCap ? "YES" : "NO"), pPwrState, (bMstEn ? "MST" : "SST"),
            (rate / 100), (rate % 100), lanes,
            lane0_1 & 0xf, (lane0_1 >> 4) & 0xf,
            lane2_3 & 0xf, (lane2_3 >> 4) & 0xf);

        // Print out PAYLOAD table for MST mode
        if (bMstEn)
        {
            LwS32 streamId;
            LwU8 streamLength;
            BOOL streamValid, streamValidOnce;

            dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS);
            if (dpcd == -1)
            {
                dprintf("auxport%d LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS(0x%x) read failed\n",
                    i, LW_DPCD_PAYLOAD_TABLE_UPDATE_STATUS);
                break;
            }
            dprintf("UPDATED:%s ACT_HANDLED:%s",
                DRF_VAL(_DPCD, _PAYLOAD_TABLE_UPDATE_STATUS, _UPDATED,
                dpcd) ? "YES" : "NO",
                DRF_VAL(_DPCD, _PAYLOAD_TABLE_UPDATE_STATUS, _ACT_HANDLED,
                dpcd) ? "YES" : "NO");

            streamId = 0;
            streamLength = 0;
            streamValid = FALSE;
            streamValidOnce = FALSE;
            for (j = 0; j < LW_DPCD_VC_PAYLOAD_ID_SLOT__SIZE; j++)
            {
                dpcd = pDpaux[indexGpu].dpauxChRead(i, LW_DPCD_VC_PAYLOAD_ID_SLOT(j));
                if (dpcd == -1)
                {
                    dprintf("\nauxport%d LW_DPCD_LANE_COUNT_SET_X(0x%x) read failed\n",
                        i, LW_DPCD_VC_PAYLOAD_ID_SLOT(j));
                }

                if (streamValid == FALSE && dpcd)
                {
                    streamId = dpcd;
                    streamLength = 1;
                    streamValid = TRUE;
                    streamValidOnce = TRUE;
                }
                else if (streamValid == TRUE && streamId == dpcd)
                {
                    streamLength++;
                }

                if (streamValid == TRUE && 
                    (j == LW_DPCD_VC_PAYLOAD_ID_SLOT__SIZE - 1 ||
                    dpcd == 0 || streamId != dpcd))
                {
                    dprintf("\n%82sstreamId:%d length=%d", "", streamId,
                        streamLength);
                    if (dpcd == 0)
                    {
                        streamId = 0;
                        streamLength = 0;
                        streamValid = FALSE;
                    }
                    else if (streamId != dpcd)
                    {
                        streamId = dpcd;
                        streamLength = 1;
                        if (j == LW_DPCD_VC_PAYLOAD_ID_SLOT__SIZE - 1)
                        {
                            dprintf("\n%82sstreamId:%d length=%d", "",
                                streamId, streamLength);
                        }
                    }
                }
            }
            if (!streamValidOnce)
            {
                dprintf("\n%82s(EMPTY!)", "");
            }
        }
        else
        {
            dprintf("NA");
        }
        dprintf("\n");
    }
}

//Helper function for dispAnalyzeInterrupts_v03_00
char *ynfunc(LwU32 val)
{
    if (val != 0)
        return "YES";
    else
        return "NO";
}

//Sanity Test for dispAnalyzeInterrupts
const char *santest(LwU32 evtPen, LwU32 rm_intrPen, LwU32 pmu_intrPen, LwU32 gsp_intrPen,
               LwU32 ie, LwU32 rm_im, LwU32 pmu_im, LwU32 gsp_im)
{
    if ((rm_im + pmu_im + gsp_im) > 1)
    return "FAIL: More than one prim target";
    else if ((rm_intrPen + pmu_intrPen + gsp_intrPen) >  1)
    return "FAIL: intr pending in multiple T";
    else if (evtPen & (!(rm_intrPen|pmu_intrPen|gsp_intrPen)))
    return "FAIL: Pending evt but no intr";
    else if ((!evtPen) & (rm_intrPen|pmu_intrPen|gsp_intrPen))
    return "FAIL: Pending intr but no evt";
    else if (!(rm_im|pmu_im|gsp_im))
    return "WARNING: No primary target set";
    else if (rm_im&(!ie))
    return "WARNING: T=RM & INTR_EN not set";
    else
    return "PASS";
}

/*!
 * @brief printDpAuxlog - Parse and print out content of DPAUXPACKETs 
 * 
 *  @param[in]  char     *buffer        pointer to DPAUXPACKET
 *  @param[in]  LwU32     entries       amount of DPAUXPACKET
 */
void printDpAuxlog(char *buffer, LwU32 entries)
{
    PDPAUXPACKET packet = (PDPAUXPACKET)buffer;
    LwU32 i, j, ackCount, start, end;
    char *msg;

    if (buffer == NULL || entries == 0)
    {
        dprintf("%s - invalid parameters\n", __FUNCTION__);
        return;
    }

    dprintf("No.       TimeStamp(us)  Port   Address  RequestSize  RequestType  ReplyType  Duration(us)  AckSize  Contents\n");
    dprintf("-------------------------------------------------------------------------------------------------------------\n");

    i = 0;
    do
    {
        dprintf("%-8d", packet[i].auxCount);
        dprintf("%14u   ", packet[i].auxRequestTimeStamp);
        dprintf("%3d    ", packet[i].auxOutPort);

        switch (DRF_VAL(_DP_AUXLOGGER, _EVENT, _TYPE, packet[i].auxEvents))
        {
            case LW_DP_AUXLOGGER_EVENT_TYPE_AUX:
                dprintf("0x%05x  ", packet[i].auxPortAddress);
                dprintf("%6d       ", packet[i].auxMessageReqSize);

                switch (DRF_VAL(_DP_AUXLOGGER, _REQUEST, _TYPE, packet[i].auxEvents))
                {
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_I2CWR:
                        msg = "I2CWR";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_I2CREQWSTAT:
                        msg = "I2CREQWSTAT";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_MOTWR:
                        msg = "MOTWR";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_MOTREQWSTAT:
                        msg = "MOTREQWSTAT";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_AUXWR:
                        msg = "AUXWR";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_I2CRD:
                        msg = "I2CRD";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_MOTRD:
                        msg = "MOTRD";
                        break;
                    case LW_DP_AUXLOGGER_REQUEST_TYPE_AUXRD:
                        msg = "AUXRD";
                        break;
                    default:
                        msg = "UNKNOWN";
                        break;
                }
                dprintf("  %-11s", msg);

                switch (DRF_VAL(_DP_AUXLOGGER, _REPLY, _TYPE, packet[i].auxEvents))
                {
                    case LW_DP_AUXLOGGER_REPLY_TYPE_SB_ACK:
                        msg = "SB_ACK";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_RETRY:
                        msg = "RETRY";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_TIMEOUT:
                        msg = "TIMEOUT";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_DEFER:
                        msg = "DEFER";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_DEFER_TO:
                        msg = "DEFER_TO";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_ACK:
                        msg = "ACK";
                        break;
                    case LW_DP_AUXLOGGER_REPLY_TYPE_ERROR:
                        msg = "SB_ACK";
                        break;
                    default:
                        msg = "UNKNOWN";
                        break;
                }
                dprintf("  %-9s", msg);

                // Print out consumed time
                start = packet[i].auxRequestTimeStamp;
                end   = packet[i].auxReplyTimeStamp;
                dprintf("   %-11u", end >= start ?
                        (end - start) : (LW_U32_MAX - start + end + 1));

                // Print out ack data size
                ackCount = packet[i].auxMessageReplySize;
                dprintf("   %-6d", ackCount);

                // Print out ack data content
                if (ackCount)
                {
                    if (ackCount > DP_MAX_MSG_SIZE)
                        ackCount = DP_MAX_MSG_SIZE;

                    for (j = 0; j < ackCount; j++)
                    {
                        dprintf("%02x ", packet[i].auxPacket[j]);
                    }
                }
                break;

            case LW_DP_AUXLOGGER_EVENT_TYPE_HOT_PLUG:
                dprintf("%23s %-10s", "", "HOTPLUG");
                break;

            case LW_DP_AUXLOGGER_EVENT_TYPE_HOT_UNPLUG:
                dprintf("%23s %-10s", "", "UNPLUG");
                break;

            case LW_DP_AUXLOGGER_EVENT_TYPE_IRQ:
                dprintf("%23s  %-10s", "", "IRQ");
                break;

            default:
                break;
        }
        dprintf("\n");
        i++;
        entries--;
    } while (entries);
}
