
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// PRIV ring routines
//
//*****************************************************

#include "hal.h"
#include "g_priv_hal.h"
#include "exts.h"
#include "kepler/gk104/dev_pri_ringstation_sys.h"


//-----------------------------------------------------
// privDumpPriHistoryBuffer  - dump of the priv history buffer
//-----------------------------------------------------
void privDumpPriHistoryBuffer_GK104()
{

    LwU32 index;
    LwU32 readInd;
    LwU32 read0, subID, privAddr;
    LwU32 read1;
    LwU32 read2, privLevel, localOrd, transType, be, wrapCnt;

    dprintf("lw: PRIV History Buffer Dump\n");
    dprintf("lw: %-7s %-22s %-15s %-30s\n", "Index", "READ0", "READ1", "READ2");
    dprintf("lw: %-7s %-22s %-15s %-40s\n", "-----", "-----------------", "----------", "-----------------------------------------------------");
    dprintf("lw: %-7s %-7s %-14s %-15s %-12s %-6s %-11s %-15s %-10s\n",
             "", "SUBID", "ADDR", "DATA", "WRAP_COUNT", "BE", "TYPE", "LOCAL_ORDERING", "PRIV_LEVEL");

    // Write the index and wait for value before reading
    for (index = 0; index <= 15; index++)
    {
        readInd = GPU_REG_RD32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_CTRL);
        readInd = FLD_SET_DRF_NUM(_PPRIV, _SYS_PRI_HISTORY_BUFFER_CTRL, _INDEX, index, readInd);

        GPU_REG_WR32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_CTRL, readInd);
        do
        {
            readInd = GPU_REG_RD32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_CTRL);
        } while (DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_CTRL, _INDEX, readInd) != index);

        read0 = GPU_REG_RD32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_READ0);
        subID = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ0, _SUBID, read0);
        privAddr = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ0, _PRIV_ADDR, read0);

        read1 = GPU_REG_RD32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_READ1);

        read2 = GPU_REG_RD32(LW_PPRIV_SYS_PRI_HISTORY_BUFFER_READ2);
        privLevel = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ2, _PRIV_LEVEL, read2);
        localOrd = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ2, _LOCAL_ORDERING, read2);
        transType = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ2, _LOCAL_TRANSACTION_TYPE , read2);
        be = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ2, _BE , read2);
        wrapCnt = DRF_VAL(_PPRIV, _SYS_PRI_HISTORY_BUFFER_READ2, _WRAP_COUNT , read2);

        dprintf("lw: %-7d 0x%02x    0x%07x      0x%08x      0x%02x         0x%02x   %-11s %-15s 0x%02x \n",
                index, subID, privAddr, read1, wrapCnt, be,
                ( (transType == 0) ? "READ" : ((transType == 1) ? "WRITE" : "WRITE_ACK") ),
                ( (localOrd == 0) ? "DISABLED" : "ENABLED"), privLevel );
    }
}
