/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "chip.h"
#include "sig.h"
#include "gr.h"

#include "g_sig_private.h"     // (rmconfig)  implementation prototypes

#ifndef SIGDUMP_ENABLE

void sigInitInstanceInfo_GM204(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigPrintLegend_GM204(FILE* fp)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

#else // SIGDUMP_ENABLE

//Initialize the Instance limits based on fs config
void sigInitInstanceInfo_GM204(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
    LwU32 i;
    for (i = 0; i < NUMINSTANCES; i++)
    {
        pInfo[i].instanceName = sigGetInstanceName(i);
        switch(i)
        {
            case gpc:
                pInfo[i].chipletLimit = ngpc-1;   //gpc chiplet limit
                pInfo[i].instanceLimit = 3;       //four instances
                pInfo[i].bValid = TRUE;
                break;

            case gpc_tpc:
                pInfo[i].chipletLimit = ngpc-1;   //gpc chiplet limit
                pInfo[i].instanceLimit = 15;      //sixteen instances
                pInfo[i].bValid = TRUE;
                break;

            case gpc_ppc:
                pInfo[i].chipletLimit = ngpc-1;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case fbp:
                pInfo[i].chipletLimit = nfbp-1;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case sys:
                pInfo[i].chipletLimit = 0;
                pInfo[i].instanceLimit = 0;       //only one instance
                pInfo[i].bValid = TRUE;
                break;

            case sys_mxbar_cs_daisy:
                pInfo[i].chipletLimit = 0;        //only one chiplet
                pInfo[i].instanceLimit = 8;       // 9 instances
                pInfo[i].bValid = TRUE;
                break;

            case sys_wxbar_cs_daisy:
                pInfo[i].chipletLimit = 0;        //only one chiplet
                pInfo[i].instanceLimit = 4;       // 5 instances
                pInfo[i].bValid = TRUE;
                break;

            default:
                pInfo[i].chipletLimit = 0;
                pInfo[i].instanceLimit = 0;
                pInfo[i].bValid = FALSE;        //dont include this stray
        }
    }
}

void sigPrintLegend_GM204(FILE* fp)
{
    fprintf(fp, " ****** GM204 Sigdump LEGEND ****** \n");
    fprintf(fp, " There are 5 columns (source, chiplet, instance, signal, value).\n");
    fprintf(fp, "   'source' - It represents hierarchy at which the PM signal exists.\n");
    fprintf(fp, "       fbp     => in FBP.\n");
    fprintf(fp, "       gpc     => in GPC(outside TPCs).\n"); 
    fprintf(fp, "       gpc_tpc => in GPC(inside TPCs).\n"); 
    fprintf(fp, "       gpc_ppc => in GPC(inside PPCs).\n"); 
    fprintf(fp, "       sys     => in SYS.\n"); 
    fprintf(fp, "       sys_mxbar_cs_daisy/unrolled => in MXBAR.\n"); 
    fprintf(fp, "       sys_wxbar_cs_daisy/unrolled => in WXBAR.\n"); 
    fprintf(fp, "   'chiplet' - It represents the instance # of the chiplet in which the signal exists.\n");
    fprintf(fp, "       Chiplet numbers are virtualized on floorswept chips.\n");
    fprintf(fp, "       For 'source' value\n");
    fprintf(fp, "           fbp     => fbp# : 0..%u based on FBP floorsweeping\n",
        pGr[indexGpu].grGetMaxFbp());
    fprintf(fp, "           gpc     => gpc# : 0..%u based on GPC floorsweeping\n",
        pGr[indexGpu].grGetMaxGpc());
    fprintf(fp, "           gpc_tpc => gpc# : 0..%u based on GPC/TPC floorsweeping\n",
        pGr[indexGpu].grGetMaxTpcPerGpc());
    fprintf(fp, "           sys, sys_mxbar_cs_daisy, sys_wxbar_cs_daisy     => 0 always\n");
    fprintf(fp, "   'instance' - It represents the instance # of the unit in the chiplet in which the signal exists.\n");
    fprintf(fp, "       For 'source' value\n");
    fprintf(fp, "           fbp, gpc, gpc_ppc, sys     => 0 always\n");
    fprintf(fp, "           gpc_tpc => tpc# : 0..15 based on TPC floorsweeping, virtualized when floorswept.\n");
    fprintf(fp, "           sys_mxbar_cs_daisy => 0..8 (numGPC + numFBP + numSYS), independent of floorsweeping, not virtualized.\n");
    fprintf(fp, "           sys_wxbar_cs_daisy => 0..4  (numGPC + numSYS), independent of floorsweeping, not virtualized.\n");
    fprintf(fp, "                xbar Details with David Tang\n");
    fprintf(fp, "   'signal' -  Name of the PM signal.\n");
    fprintf(fp, "       Each row represents a PM signal, which can be a 1 bit wire or a bus.\n");
    fprintf(fp, "   'value' -   Value of the PM signal.\n");
    fprintf(fp, " ****************************** \n");
    fprintf(fp, "\n");
}

#endif
