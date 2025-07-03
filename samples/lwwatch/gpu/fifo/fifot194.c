/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2004-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// includes
//
#include "t19x/t194/dev_arhost1x.h"
#include "t19x/t194/dev_arhost1x_thost.h"

#include "fifo.h"

#include "g_fifo_private.h"  // (rmconfig) implementation prototypes

#include "utils/lwmacro.h"

#define LW_HOST1X_CHANNELS         63

#define PRINT_HOST1X_COMMON_REG(_reg_)                                  \
    dprintf("lw: LW_PHOST1X_THOST_COMMON%-40s 0x%08x\n",                \
            _LW_STRINGIFY(_reg_),                                       \
            DEV_REG_RD32(                                               \
                LW_PHOST1X_THOST_COMMON##_reg_,                         \
                "HOST1X",                                               \
                0 ))

#define PRINT_HOST1X_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU_REG(cluster_id, i)                     \
    dprintf("lw: LW_PHOST1X_THOST_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU%u(%-2u)%-6s 0x%08x\n",    \
            cluster_id,                                                                         \
            i,                                                                                  \
            "",                                                                                 \
            DEV_REG_RD32(                                                                       \
                LW_PHOST1X_THOST_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU##cluster_id(i),            \
                "HOST1X",                                                                       \
                0 ))

#define PRINT_HOST1X_CHANNEL_REG(_chid_, _reg_)                                                 \
    dprintf("lw: LW_PHOST1X_THOST_CHANNEL_CH%u%-*s 0x%08x\n",                                   \
            _chid_,                                                                             \
            (_chid_ < 10) ? 35 : (_chid_ < 100) ? 34 : (_chid_ < 1000) ? 33 : 32,               \
            LW_STRINGIFY(_reg_),                                                                \
            DEV_REG_RD32(                                                                       \
                ( (_chid_ * LW_PHOST1X_CHANNEL_MAP_SIZE_BYTES) +                                \
                  LW_PHOST1X_THOST_CHANNEL_CH0##_reg_ ),                                        \
                "HOST1X",                                                                       \
                0 ))

//-----------------------------------------------------
// dumpHost1xChannel_T194
//
//-----------------------------------------------------
void dumpHost1xChannel_T194(LwU32 chid)
{
    if (chid > LW_HOST1X_CHANNELS)
    {
        dprintf("lw: %s Invalid channel: 0x%x\n", __FUNCTION__, chid);
        return;
    }

    dprintf("lw: Dumping arhost1x_thost.h State for Channel: 0x%x\n", chid);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDFIFO_STAT);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMASTART);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMASTART_HI);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAPUT);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAPUT_HI);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAGET);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAGET_HI);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAEND);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMAEND_HI);
    PRINT_HOST1X_CHANNEL_REG(chid, _DMACTRL);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDFIFO_STAT);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDFIFO_RDATA);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDP_OFFSET);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDP_CLASS);
    PRINT_HOST1X_CHANNEL_REG(chid, _CHANNELSTAT);
    PRINT_HOST1X_CHANNEL_REG(chid, _DROP_ILLEGAL_OPCODES);
    PRINT_HOST1X_CHANNEL_REG(chid, _GATHER_PARSE_DISABLED);
    PRINT_HOST1X_CHANNEL_REG(chid, _CMDPROC_STOP);
    PRINT_HOST1X_CHANNEL_REG(chid, _TEARDOWN);
    PRINT_HOST1X_CHANNEL_REG(chid, _SYNCPT_PAYLOAD);
    PRINT_HOST1X_CHANNEL_REG(chid, _ILLEGAL_ACCESS_INTR);
    PRINT_HOST1X_CHANNEL_REG(chid, _ILLEGAL_ACCESS_INTRMASK);
    PRINT_HOST1X_CHANNEL_REG(chid, _SMMU_STREAMID);
    PRINT_HOST1X_CHANNEL_REG(chid, _MLOCK_BUSY_TIMEOUT);
    PRINT_HOST1X_CHANNEL_REG(chid, _RSB_NS);
    PRINT_HOST1X_CHANNEL_REG(chid, _CHANNEL_SPARE);
}

//-----------------------------------------------------
// fifoGetInfo_T194
//
//-----------------------------------------------------
LW_STATUS fifoGetInfo_T194(void)
{
    LwU32 chid;
    LwU32 i;

    PRINT_HOST1X_COMMON_REG(_INTRSTATUS_0);
    PRINT_HOST1X_COMMON_REG(_THOST_INTRSTATUS_0);

    dprintf("\n");

    for (i = 0;
         i < LW_PHOST1X_THOST_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU0__SIZE;
         ++i) {
        PRINT_HOST1X_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU_REG(0, i);
        PRINT_HOST1X_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU_REG(1, i);
        PRINT_HOST1X_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU_REG(2, i);
        PRINT_HOST1X_COMMON_SYNCPT_THRESH_INTRSTATUS_CPU_REG(3, i);
    }

    dprintf("\n");

    // Dump the state for each host1x chid
    for (chid = 0; chid < LW_HOST1X_CHANNELS; chid++)
    {
        dumpHost1xChannel_T194(chid);
        dprintf("\n");
    }

    // Dump the GPU fifo registers
    fifoGetInfo_GK104();
    return LW_OK;
}
