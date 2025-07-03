/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All
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

#define RM_ASSERT(x)
#define DBG_BREAKPOINT()

#ifndef SIGDUMP_ENABLE

void sigInitInstanceInfo_GP100(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

void sigPrintLegend_GP100(FILE* fp)
{
   dprintf("lw: This function needs to be built with SIGDUMP enabled\n");
   dprintf("lw: Add /DSIGDUMP_ENABLE to the C_DEFINES in sources\n");
}

#else // SIGDUMP_ENABLE

//Initialize the Instance limits based on fs config
void sigInitInstanceInfo_GP100(InstanceInfo* pInfo, LwU32 ngpc, LwU32 ntpc, LwU32 nfbp, LwU32 option)
{
    dprintf("lw: SIGDUMP support needs to be added for this chip.\n");
    RM_ASSERT(0);
}

void sigPrintLegend_GP100(FILE* fp)
{
    dprintf("lw: SIGDUMP support needs to be added for this chip.\n");
    fprintf(fp, "lw: SIGDUMP support needs to be added for this chip.\n");
    RM_ASSERT(0);
}

#endif
