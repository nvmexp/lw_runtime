/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SV10_H_
#define _SV10_H_

#include "lwlink.h"
#include "lwCpuUuid.h"
#include "g_lwconfig.h"

#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "pmgr_lwswitch.h"
#include "rom_lwswitch.h"

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

#include "lwswitch/svnp01/dev_lws_master.h"

//#define DISABLE_CLOCK_INIT

// DLPL size is 32 KB
#define LWSWITCH_LWLINK_DLPL_SIZE_SV10                 0x8000

//
// Per link register access routines
//

#define LWSWITCH_IS_LINK_ENG_VALID_SV10(_d, _eng, _linknum)  \
    (                                                            \
        (_linknum < LWSWITCH_NUM_LINKS_SV10) &&              \
        (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->link[_linknum].eng ## _eng != NULL) &&    \
        (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->link[_linknum].eng ## _eng->valid)        \
    )

//
// LINK_* MMIO wrappers are used to reference per-link engine instances
//

#define LWSWITCH_LINK_OFFSET_SV10(_d, _physlinknum, _eng, _dev, _reg) \
    (                                                               \
        LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID_SV10(_d, _eng, _physlinknum))\
        ,                                                           \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: LINK_OFFSET link[%d] %s: %s,%s (+%04x)\n",         \
            __FUNCTION__,                                           \
            _physlinknum,                                           \
            #_eng, #_dev, #_reg, LW ## _dev ## _reg)                \
        ,                                                           \
        LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->link[_physlinknum].eng ##_eng->uc_addr + LW##_dev##_reg    \
    )

#define LWSWITCH_LINK_WR32_SV10(_d, _physlinknum, _eng, _dev, _reg, _data)  \
    LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID_SV10(_d, _eng, _physlinknum)); \
    LWSWITCH_PRINT(_d, MMIO,                                        \
        "%s: LINK_WR link[%d] %s: %s,%s (+%04x) 0x%08x\n",          \
        __FUNCTION__,                                               \
        _physlinknum,                                               \
        #_eng, #_dev, #_reg, LW ## _dev ## _reg, _data);            \
    lwswitch_reg_write_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->link[_physlinknum].eng##_eng->uc_addr + LW##_dev##_reg, _data); \
    ((void)(_d))

#define LWSWITCH_LINK_RD32_SV10(_d, _physlinknum, _eng, _dev, _reg) \
    (                                                               \
        LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID_SV10(_d, _eng, _physlinknum))\
        ,                                                           \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: LINK_RD link[%d] %s: %s,%s (+%04x)\n",             \
            __FUNCTION__,                                           \
            _physlinknum,                                           \
            #_eng, #_dev, #_reg, LW ## _dev ## _reg)                \
        ,                                                           \
        lwswitch_reg_read_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->link[_physlinknum].eng ##_eng->uc_addr + LW##_dev##_reg)    \
    );                                                              \
    ((void)(_d))

//
// LWSWITCH_ENG_* MMIO wrappers are to be used for top level discovered
// devices like SAW, FUSE, PMGR, XVE, etc.
//

#define LWSWITCH_ENG_WR32_SV10(_d, _eng, _bcast, _engidx, _blwc, _dev, _reg, _data)   \
    {                                                             \
        LWSWITCH_PRINT(_d, MMIO,                                  \
            "%s: MEM_WR %s[%d] %s: %s,%s (+%04x) 0x%08x\n",       \
            __FUNCTION__,                                         \
            #_eng#_bcast, _engidx,                                \
            #_blwc, #_dev, #_reg, LW ## _dev ## _reg, _data);     \
        if (_engidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng##_bcast) \
        {                                                         \
            if (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng##_bcast[_engidx].valid)   \
            {                                                     \
                LWSWITCH_OFF_WR32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng##_bcast[_engidx]._blwc##_addr + LW ## _dev ## _reg, _data); \
            }                                                     \
            else                                                  \
            {                                                     \
                LWSWITCH_PRINT(_d, MMIO,                          \
                    "%s: %s[%d] marked invalid (disabled)\n",     \
                    __FUNCTION__,                                 \
                    #_eng#_bcast, _engidx);                       \
            }                                                     \
        }                                                         \
        else                                                      \
        {                                                         \
            LWSWITCH_PRINT(_d, MMIO,                              \
                "%s: %s[%d] out of range 0..%d\n",                \
                __FUNCTION__,                                     \
                #_eng#_bcast, _engidx, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng##_bcast);  \
        }                                                         \
    }

#define LWSWITCH_ENG_RD32_SV10(_d, _eng, _engidx, _dev, _reg) \
    (                                                             \
        LWSWITCH_PRINT(_d, MMIO,                                  \
            "%s: MEM_RD %s[%d]: %s,%s (+%04x)\n",                 \
            __FUNCTION__,                                         \
            #_eng, _engidx,                                       \
            #_dev, #_reg, LW ## _dev ## _reg)                     \
    ,                                                             \
        (                                                         \
            ((_engidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng) &&               \
             (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng[_engidx].valid))            \
        ?                                                         \
            lwswitch_reg_read_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng[_engidx].uc_addr + LW ## _dev ## _reg) \
        :                                                         \
            0xBADFBADF                                            \
        )                                                         \
    );                                                            \
    ((void)(_d))

//
// LWSWITCH_SUBENG_* MMIO wrappers are to be used for second level discovered
// devices within SIOCTRL, NPG, and SWX.
//

#define LWSWITCH_SUBENG_OFF_WR32_SV10(_d, _eng, _bcast, _engidx, _subeng, _mcast, _subengidx, _blwc, _off, _data)   \
    {                                                             \
        LWSWITCH_PRINT(_d, MMIO,                                  \
            "%s: MEM_WR %s[%d] %s[%d] %s: %s (+%04x) 0x%08x\n",   \
            __FUNCTION__,                                         \
            #_eng#_bcast, _engidx, #_subeng#_mcast, _subengidx,   \
            #_blwc, #_off, _off, _data);                          \
        if (_engidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng##_bcast) \
        {                                                         \
            if (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng##_bcast[_engidx].valid)   \
            {                                                     \
                if (_subengidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng##_bcast[_engidx].num##_subeng##_mcast)                 \
                {                                                 \
                    if (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng##_bcast[_engidx].subeng##_subeng##_mcast[_subengidx].valid)     \
                    {                                             \
                        LWSWITCH_OFF_WR32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng##_bcast[_engidx].subeng##_subeng##_mcast[_subengidx]._blwc##_addr + _off, _data); \
                    }                                             \
                    else                                          \
                    {                                             \
                        LWSWITCH_PRINT(_d, MMIO,                  \
                            "%s: %s[%d].%s[%d] marked invalid (disabled)\n", \
                            __FUNCTION__,                         \
                            #_eng#_bcast, _engidx, #_subeng#_mcast, _subengidx); \
                    }                                             \
                }                                                 \
                else                                              \
                {                                                 \
                    LWSWITCH_PRINT(_d, MMIO,                      \
                        "%s: %s[%d].%s[%d] out of range 0..%d\n", \
                        __FUNCTION__,                             \
                        #_eng#_bcast, _engidx, #_subeng#_mcast, _subengidx, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng##_bcast[_engidx].num##_subeng##_mcast);        \
                }                                                 \
            }                                                     \
            else                                                  \
            {                                                     \
                LWSWITCH_PRINT(_d, MMIO,                          \
                    "%s: %s[%d] marked invalid (disabled)\n",     \
                    __FUNCTION__,                                 \
                    #_eng#_bcast, _engidx);                       \
            }                                                     \
        }                                                         \
        else                                                      \
        {                                                         \
            LWSWITCH_PRINT(_d, MMIO,                              \
                "%s: %s[%d] out of range 0..%d\n",                \
                __FUNCTION__,                                     \
                #_eng#_bcast, _engidx, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng##_bcast);  \
        }                                                         \
    }

#define LWSWITCH_SUBENG_WR32_SV10(_d, _eng, _bcast, _engidx, _subeng, _mcast, _subengidx, _blwc, _dev, _reg, _data)   \
    LWSWITCH_SUBENG_OFF_WR32_SV10(_d, _eng, _bcast, _engidx, _subeng, _mcast, _subengidx, _blwc, LW ## _dev ## _reg, _data)

#define LWSWITCH_SUBENG_OFF_RD32_SV10(_d, _eng, _engidx, _subeng, _subengidx, _off)   \
    (                                                           \
        LWSWITCH_PRINT(_d, MMIO,                                \
            "%s: MEM_RD %s[%d] %s[%d]: %s (+%04x)\n",           \
            __FUNCTION__,                                       \
            #_eng, _engidx, #_subeng, _subengidx,               \
            #_off, _off)                                        \
    ,                                                           \
        (                                                       \
            ((_engidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->num##_eng) &&  \
             (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->eng##_eng[_engidx].valid) && \
             (_subengidx < LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng[_engidx].num##_subeng) && \
             (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng[_engidx].subeng##_subeng[_subengidx].valid)) \
        ?                                                       \
            lwswitch_reg_read_32(_d, LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng[_engidx].subeng##_subeng[_subengidx].uc_addr + _off) \
        :                                                       \
            0xBADFBADF                                          \
        )                                                       \
    );                                                          \
    ((void)(_d))

#define LWSWITCH_SUBENG_RD32_SV10(_d, _eng, _engidx, _subeng, _subengidx, _dev, _reg)   \
    LWSWITCH_SUBENG_OFF_RD32_SV10(_d, _eng, _engidx, _subeng, _subengidx, LW ## _dev ## _reg)

#define LWSWITCH_BCAST_OFF_WR32_SV10(_d, _eng, _subeng, _mcast, _off, _data)    \
    {                                                                               \
        LwU32 idx_mmio;                                                             \
                                                                                    \
        for (idx_mmio = 0; idx_mmio < NUM_##_eng##_ENGINE_SV10; idx_mmio++)              \
        {                                                                           \
            if (LWSWITCH_GET_CHIP_DEVICE_SV10(_d)->subeng##_eng[idx_mmio].subeng##_subeng##_mcast[0].valid) \
            {                                                                       \
                LWSWITCH_SUBENG_OFF_WR32_SV10(_d, _eng, , idx_mmio, _subeng, _mcast, 0, uc, _off, _data); \
            }                                                                       \
        }                                                                           \
    }

#define LWSWITCH_BCAST_WR32_SV10(_d, _eng, _subeng, _mcast, _dev, _reg, _data)                \
    LWSWITCH_BCAST_OFF_WR32_SV10(_d, _eng, _subeng, _mcast, LW##_dev##_reg, _data)

//
// Simplified wrappers built upon the verbose engine and subengine MMIO wrappers
//

#define LWSWITCH_NPG_RD32_SV10(_d, _engidx, _dev, _reg)                   \
    LWSWITCH_SUBENG_RD32_SV10(_d, NPG, _engidx, NPG, 0,  _dev, _reg)

#define LWSWITCH_NPG_WR32_SV10(_d, _engidx, _dev, _reg, _data)            \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPG, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_NPG_BC_WR32_SV10(_d, _engidx, _dev, _reg, _data)         \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPG, , 0, bc, _dev, _reg, _data)

#define LWSWITCH_NPG_BCAST_WR32_SV10(_d, _dev, _reg, _data)               \
    LWSWITCH_BCAST_WR32_SV10(_d, NPG, NPG, , _dev, _reg, _data)

#define LWSWITCH_NPGPERF_WR32_SV10(_d, _engidx, _dev, _reg, _data)        \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPG_PERFMON, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_NPGPERF_BC_WR32_SV10(_d, _engidx, _dev, _reg, _data)     \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPG_PERFMON, , 0, bc, _dev, _reg, _data)

#define LWSWITCH_NPGPERF_BCAST_WR32_SV10(_d, _dev, _reg, _data)           \
    LWSWITCH_BCAST_WR32_SV10(_d, NPG, NPG_PERFMON, , _dev, _reg, _data)

#define LWSWITCH_SWX_RD32_SV10(_d, _engidx, _dev, _reg)                   \
    LWSWITCH_SUBENG_RD32_SV10(_d, SWX, _engidx, SWX, 0, _dev, _reg)

#define LWSWITCH_SWX_BCAST_WR32_SV10(_d, _dev, _reg, _data)               \
    LWSWITCH_BCAST_WR32_SV10(_d, SWX, SWX, , _dev, _reg, _data)

#define LWSWITCH_AFS_RD32_SV10(_d, _engidx, _subengidx, _dev, _reg)       \
    LWSWITCH_SUBENG_RD32_SV10(_d, SWX, _engidx, AFS, _subengidx, _dev, _reg)

#define LWSWITCH_AFS_WR32_SV10(_d, _engidx, _subengidx, _dev, _reg, _data) \
    LWSWITCH_SUBENG_WR32_SV10(_d, SWX, , _engidx, AFS, , _subengidx, uc, _dev, _reg, _data)

#define LWSWITCH_AFS_MC_BCAST_WR32_SV10(_d, _dev, _reg, _data)            \
    LWSWITCH_BCAST_WR32_SV10(_d, SWX, AFS, _MULTICAST, _dev, _reg, _data)

#define LWSWITCH_SAW_RD32_SV10(_d, _dev, _reg)                            \
    LWSWITCH_ENG_RD32_SV10(_d, SAW, 0, _dev, _reg)

#define LWSWITCH_SAW_WR32_SV10(_d, _dev, _reg, _data)                     \
    LWSWITCH_ENG_WR32_SV10(_d, SAW, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_SIOCTRL_RD32_SV10(_d, _engidx, _dev, _reg)                \
    LWSWITCH_SUBENG_RD32_SV10(_d, SIOCTRL, _engidx, SIOCTRL, 0, _dev, _reg)

#define LWSWITCH_SIOCTRL_OFF_RD32_SV10(_d, _engidx, _off)                  \
    LWSWITCH_SUBENG_OFF_RD32_SV10(_d, SIOCTRL, _engidx, SIOCTRL, 0, _off)

#define LWSWITCH_SIOCTRL_WR32_SV10(_d, _engidx, _dev, _reg, _data)         \
    LWSWITCH_SUBENG_WR32_SV10(_d, SIOCTRL, , _engidx, SIOCTRL, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_SIOCTRL_OFF_WR32_SV10(_d, _engidx, _off, _data)           \
    LWSWITCH_SUBENG_OFF_WR32_SV10(_d, SIOCTRL, , _engidx, SIOCTRL, , 0, uc, _off, _data)

#define LWSWITCH_SIOCTRL_BCAST_WR32_SV10(_d, _dev, _reg, _data)            \
    LWSWITCH_BCAST_WR32_SV10(_d, SIOCTRL, SIOCTRL, , _dev, _reg, _data)

#define LWSWITCH_LWLIPT_RD32_SV10(_d, _engidx, _dev, _reg)                 \
    LWSWITCH_SUBENG_RD32_SV10(_d, SIOCTRL, _engidx, LWLIPT, 0, _dev, _reg)

#define LWSWITCH_LWLIPT_OFF_RD32_SV10(_d, _engidx, _off)                   \
    LWSWITCH_SUBENG_OFF_RD32_SV10(_d, SIOCTRL, _engidx, LWLIPT, 0, _off)

#define LWSWITCH_LWLIPT_WR32_SV10(_d, _engidx, _dev, _reg, _data)          \
    LWSWITCH_SUBENG_WR32_SV10(_d, SIOCTRL, , _engidx, LWLIPT, , 0, uc, _dev, _reg, _data)

#define LWSWITCH_LWLIPT_OFF_WR32_SV10(_d, _engidx, _off, _data)            \
    LWSWITCH_SUBENG_OFF_WR32_SV10(_d, SIOCTRL, , _engidx, LWLIPT, , 0, uc, _off, _data)

#define LWSWITCH_LWLIPT_BCAST_WR32_SV10(_d, _dev, _reg, _data)             \
    LWSWITCH_BCAST_WR32_SV10(_d, SIOCTRL, LWLIPT, , _dev, _reg, _data)

#define LWSWITCH_LWLTLC_RD32_SV10(_d, _engidx, _subengidx, _dev, _reg)     \
    LWSWITCH_SUBENG_RD32_SV10(_d, SIOCTRL, _engidx, LWLTLC, _subengidx, _dev, _reg)

#define LWSWITCH_LWLTLC_WR32_SV10(_d, _engidx, _subengidx, _dev, _reg, _data)          \
    LWSWITCH_SUBENG_WR32_SV10(_d, SIOCTRL, , _engidx, LWLTLC, , _subengidx, uc, _dev, _reg, _data)

#define LWSWITCH_LWLTLC_MCAST_WR32_SV10(_d, _engidx, _dev, _reg, _data)    \
    LWSWITCH_SUBENG_WR32_SV10(_d, SIOCTRL, , _engidx, LWLTLC, _MULTICAST, 0, uc, _dev, _reg, _data)

#define LWSWITCH_LWLTLC_BCAST_WR32_SV10(_d, _dev, _reg, _data)             \
    LWSWITCH_BCAST_WR32_SV10(_d, SIOCTRL, LWLTLC, _MULTICAST, _dev, _reg, _data)

#define LWSWITCH_NPORT_RD32_SV10(_d, _engidx, _subengidx, _dev, _reg) \
    LWSWITCH_SUBENG_RD32_SV10(_d, NPG, _engidx, NPORT, _subengidx, _dev, _reg)

#define LWSWITCH_NPORT_WR32_SV10(_d, _engidx, _subengidx, _dev, _reg, _data) \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPORT, , _subengidx, uc, _dev, _reg, _data)

#define LWSWITCH_NPORT_MC_WR32_SV10(_d, _engidx, _dev, _reg, _data)       \
    LWSWITCH_SUBENG_WR32_SV10(_d, NPG, , _engidx, NPORT, _MULTICAST, 0, bc, _dev, _reg, _data)

#define LWSWITCH_NPORT_MC_BCAST_WR32_SV10(_d, _dev, _reg, _data)          \
    LWSWITCH_BCAST_WR32_SV10(_d, NPG, NPORT, _MULTICAST, _dev, _reg, _data)

#define LW_NPORT_PORTSTAT_SV10(_block, _reg, _idx)   (LW_NPORT_PORTSTAT ## _block ## _reg ## _0 + _idx*(LW_NPORT_PORTSTAT ## _block ## _reg ## _1 - LW_NPORT_PORTSTAT ## _block ## _reg ## _0))

#define LWSWITCH_CLK_LWLINK_RD32_SV10(_d, _reg, _idx)                   \
    LWSWITCH_REG_RD32(_d, _PCLOCK, _LWSW_LWLINK##_reg(_idx))

#define LWSWITCH_CLK_LWLINK_WR32_SV10(_d, _reg, _idx, _data)            \
    if (IS_RTLSIM(_d) || IS_FMODEL(_d))                                 \
    {                                                                   \
        LWSWITCH_PRINT(_d, MMIO,                                        \
        "%s: Skip write LW_PCLOCK_LWSW_LWLINK%d %s (0x%06x) on FSF\n",  \
            __FUNCTION__,                                               \
            _idx, #_reg,                                                \
            LW_PCLOCK_LWSW_LWLINK##_reg(_idx));                         \
    }                                                                   \
    else                                                                \
    {                                                                   \
        LWSWITCH_REG_WR32(_d, _PCLOCK, _LWSW_LWLINK##_reg(_idx), _data);     \
    }
//
// Device discovery table
//

#define NUM_PTOP_ENGINE_SV10                 1
#define NUM_SIOCTRL_ENGINE_SV10              9
#define NUM_SIOCTRL_BCAST_ENGINE_SV10        2
#define NUM_NPG_ENGINE_SV10                  5
#define NUM_NPG_BCAST_ENGINE_SV10            2
#define NUM_SWX_ENGINE_SV10                  2
#define NUM_SWX_BCAST_ENGINE_SV10            1
#define NUM_CLKS_ENGINE_SV10                 1
#define NUM_FUSE_ENGINE_SV10                 1
#define NUM_JTAG_ENGINE_SV10                 1
#define NUM_PMGR_ENGINE_SV10                 1
#define NUM_SAW_ENGINE_SV10                  1
#define NUM_XP3G_ENGINE_SV10                 1
#define NUM_XVE_ENGINE_SV10                  1
#define NUM_ROM_ENGINE_SV10                  1
#define NUM_EXTDEV_ENGINE_SV10               1
#define NUM_PRIVMAIN_ENGINE_SV10             1
#define NUM_PRIVLOC_ENGINE_SV10              3

#define NUM_PTOP_ENG_INSTANCES_SV10              NUM_PTOP_ENGINE_SV10
#define NUM_SIOCTRL_ENG_INSTANCES_SV10           NUM_SIOCTRL_ENGINE_SV10
#define NUM_SIOCTRL_BCAST_ENG_INSTANCES_SV10     NUM_SIOCTRL_BCAST_ENGINE_SV10
#define NUM_NPG_ENG_INSTANCES_SV10               NUM_NPG_ENGINE_SV10
#define NUM_NPG_BCAST_ENG_INSTANCES_SV10         NUM_NPG_BCAST_ENGINE_SV10
#define NUM_SWX_ENG_INSTANCES_SV10               NUM_SWX_ENGINE_SV10
#define NUM_SWX_BCAST_ENG_INSTANCES_SV10         NUM_SWX_BCAST_ENGINE_SV10
#define NUM_CLKS_ENG_INSTANCES_SV10              NUM_CLKS_ENGINE_SV10
#define NUM_FUSE_ENG_INSTANCES_SV10              NUM_FUSE_ENGINE_SV10
#define NUM_JTAG_ENG_INSTANCES_SV10              NUM_JTAG_ENGINE_SV10
#define NUM_PMGR_ENG_INSTANCES_SV10              NUM_PMGR_ENGINE_SV10
#define NUM_SAW_ENG_INSTANCES_SV10               NUM_SAW_ENGINE_SV10
#define NUM_XP3G_ENG_INSTANCES_SV10              NUM_XP3G_ENGINE_SV10
#define NUM_XVE_ENG_INSTANCES_SV10               NUM_XVE_ENGINE_SV10
#define NUM_ROM_ENG_INSTANCES_SV10               NUM_ROM_ENGINE_SV10
#define NUM_EXTDEV_ENG_INSTANCES_SV10            NUM_EXTDEV_ENGINE_SV10
#define NUM_PRIVMAIN_ENG_INSTANCES_SV10          NUM_PRIVMAIN_ENGINE_SV10
#define NUM_PRIVLOC_ENG_INSTANCES_SV10           NUM_PRIVLOC_ENGINE_SV10

typedef struct
{
    LwBool valid;
    LwU32 engine;
    LwU32 instance;
    LwU32 version;
    LwU32 uc_addr;
    LwU32 bc_addr;
    LwU32 reset_addr;
    LwU32 reset_bit;
    LwU32 intr_addr;
    LwU32 intr_bit;
    LwU32 cluster;
    LwU32 cluster_id;
    LwU32 discovery;                // Used for debugging
    LwU32 initialized;
} ENGINE_DESCRIPTOR_TYPE_SV10;

#define NUM_SIOCTRL_INSTANCES_SV10               1
#define NUM_LWLTLC_INSTANCES_SV10                2
#define NUM_DLPL_INSTANCES_SV10                  2
#define NUM_TX_PERFMON_INSTANCES_SV10            2
#define NUM_RX_PERFMON_INSTANCES_SV10            2
#define NUM_MINION_INSTANCES_SV10                1
#define NUM_LWLIPT_INSTANCES_SV10                1
#define NUM_DLPL_MULTICAST_INSTANCES_SV10        1
#define NUM_LWLTLC_MULTICAST_INSTANCES_SV10      1
#define NUM_SIOCTRL_PERFMON_INSTANCES_SV10       1
#define NUM_LWLIPT_SYS_PERFMON_INSTANCES_SV10    1
#define NUM_TX_PERFMON_MULTICAST_INSTANCES_SV10  1
#define NUM_RX_PERFMON_MULTICAST_INSTANCES_SV10  1

#define LWSWITCH_DECLARE_ENGINE_SV10(engine)              \
    LwU32 num##engine;                      \
    ENGINE_DESCRIPTOR_TYPE_SV10  eng##engine[NUM_##engine##_ENG_INSTANCES_SV10];

#define DECLARE_SUBENGINE_SV10(engine)           \
    LwU32 num##engine;                      \
    ENGINE_DESCRIPTOR_TYPE_SV10  subeng##engine[NUM_##engine##_INSTANCES_SV10];

typedef struct
{
    LwU32   master;
    LwU32   master_id;
    LwU32   num_tx;
    LwU32   num_rx;
} DLPL_INFO_TYPE_SV10;

typedef struct
{
    DECLARE_SUBENGINE_SV10(MINION)
    DECLARE_SUBENGINE_SV10(LWLIPT)
    DECLARE_SUBENGINE_SV10(LWLTLC)
    DECLARE_SUBENGINE_SV10(DLPL_MULTICAST)
    DECLARE_SUBENGINE_SV10(LWLTLC_MULTICAST)
    DECLARE_SUBENGINE_SV10(DLPL)
    DECLARE_SUBENGINE_SV10(SIOCTRL)
    DECLARE_SUBENGINE_SV10(SIOCTRL_PERFMON)
    DECLARE_SUBENGINE_SV10(LWLIPT_SYS_PERFMON)
    DECLARE_SUBENGINE_SV10(TX_PERFMON_MULTICAST)
    DECLARE_SUBENGINE_SV10(RX_PERFMON_MULTICAST)
    DECLARE_SUBENGINE_SV10(TX_PERFMON)
    DECLARE_SUBENGINE_SV10(RX_PERFMON)

    DLPL_INFO_TYPE_SV10 dlpl_info[NUM_DLPL_INSTANCES_SV10];
    DLPL_INFO_TYPE_SV10 dlpl_info_multicast;
} LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10;

//
// NPG subengines
//
#define NUM_NPG_SUBENGINE_SV10                       (NUM_NPG_ENGINE_SV10+NUM_NPG_BCAST_ENGINE_SV10)
#define NUM_NPORT_SUBENGINE_SV10                     (NUM_NPG_SUBENGINE_SV10*NUM_NPORT_INSTANCES_SV10)
#define NUM_NPORT_MULTICAST_SUBENGINE_SV10           (NUM_NPG_SUBENGINE_SV10)
#define NUM_NPG_PERFMON_SUBENGINE_SV10               (NUM_NPG_SUBENGINE_SV10)
#define NUM_NPORT_PERFMON_SUBENGINE_SV10             (NUM_NPORT_SUBENGINE_SV10)
#define NUM_NPORT_PERFMON_MULTICAST_SUBENGINE_SV10   (NUM_NPG_SUBENGINE_SV10)

#define NUM_NPG_INSTANCES_SV10                       1
#define NUM_NPORT_INSTANCES_SV10                     4
#define NUM_NPORT_MULTICAST_INSTANCES_SV10           1
#define NUM_NPG_PERFMON_INSTANCES_SV10               NUM_NPG_INSTANCES_SV10
#define NUM_NPORT_PERFMON_INSTANCES_SV10             NUM_NPORT_INSTANCES_SV10
#define NUM_NPORT_PERFMON_MULTICAST_INSTANCES_SV10   NUM_NPORT_MULTICAST_INSTANCES_SV10

typedef struct
{
    DECLARE_SUBENGINE_SV10(NPG)
    DECLARE_SUBENGINE_SV10(NPORT)
    DECLARE_SUBENGINE_SV10(NPORT_MULTICAST)
    DECLARE_SUBENGINE_SV10(NPG_PERFMON)
    DECLARE_SUBENGINE_SV10(NPORT_PERFMON)
    DECLARE_SUBENGINE_SV10(NPORT_PERFMON_MULTICAST)
} NPG_SUBENGINE_DESCRIPTOR_TYPE_SV10;

//
// SWX subengines
//
#define NUM_SWX_SUBENGINE_SV10                   (NUM_SWX_ENGINE_SV10+NUM_SWX_BCAST_ENGINE_SV10)
#define NUM_AFS_SUBENGINE_SV10                   (NUM_SIOCTRL_ENGINE_SV10*NUM_SWX_SUBENGINE_SV10)
#define NUM_AFS_MULTICAST_SUBENGINE_SV10         NUM_SWX_SUBENGINE_SV10
#define NUM_SWX_PERFMON_SUBENGINE_SV10           NUM_SWX_SUBENGINE_SV10
#define NUM_AFS_PERFMON_SUBENGINE_SV10           NUM_AFS_SUBENGINE_SV10
#define NUM_AFS_PERFMON_MULTICAST_SUBENGINE_SV10 NUM_SWX_SUBENGINE_SV10

#define NUM_SWX_INSTANCES_SV10                   1
#define NUM_AFS_INSTANCES_SV10                   NUM_SIOCTRL_ENGINE_SV10
#define NUM_AFS_MULTICAST_INSTANCES_SV10         1
#define NUM_SWX_PERFMON_INSTANCES_SV10           NUM_SWX_INSTANCES_SV10
#define NUM_AFS_PERFMON_INSTANCES_SV10           NUM_AFS_INSTANCES_SV10
#define NUM_AFS_PERFMON_MULTICAST_INSTANCES_SV10 NUM_AFS_MULTICAST_INSTANCES_SV10

typedef struct
{
    DECLARE_SUBENGINE_SV10(SWX)
    DECLARE_SUBENGINE_SV10(AFS)
    DECLARE_SUBENGINE_SV10(AFS_MULTICAST)
    DECLARE_SUBENGINE_SV10(SWX_PERFMON)
    DECLARE_SUBENGINE_SV10(AFS_PERFMON)
    DECLARE_SUBENGINE_SV10(AFS_PERFMON_MULTICAST)
} SWX_SUBENGINE_DESCRIPTOR_TYPE_SV10;

typedef struct
{
    LwBool WAR_Bug_200241882_AFS_interrupt_bits;
} LWSWITCH_OVERRIDE_TYPE_SV10;

//
// Background tasks
//      1) latency bins logging
//      2) thermal warn slowdown
//      3) thermal & voltage logging
//      4) crumbstore scrub
//

#define LWSWITCH_NUM_LINKS_SV10          (NUM_SIOCTRL_ENGINE_SV10*NUM_DLPL_INSTANCES_SV10)
#define LWSWITCH_NUM_LANES_SV10          8
#define LWSWITCH_NUM_VCS_SV10            8

#define LWSWITCH_NUM_LINKS_PER_LWLIPT_SV10 (LWSWITCH_NUM_LINKS_SV10/NUM_SIOCTRL_ENGINE_SV10)

#define LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_SV10(_physlinknum) \
    ((_physlinknum)%LWSWITCH_NUM_LINKS_PER_LWLIPT_SV10)
//
// Ingress request/response mapping
//

typedef struct
{
    LwU32   ingress_reqresmapdata0;
    LwU32   ingress_reqresmapdata1;
    LwU32   ingress_reqresmapdata2;
    LwU32   ingress_reqresmapdata3;
} INGRESS_REQUEST_RESPONSE_ENTRY_SV10;


//
// Ganged link mappings
//

typedef struct
{
    LwU32   regtabledata0;
} ROUTE_GANG_ENTRY_SV10;

//
// Per-link information
//

typedef struct
{
    // TRUE if at least some parts of this link are instantiated below
    LwBool                      valid;

    // NPG
    ENGINE_DESCRIPTOR_TYPE_SV10          *engNPORT;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engNPORT_PERFMON;

    // SIOCTRL
    ENGINE_DESCRIPTOR_TYPE_SV10          *engDLPL;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engLWLTLC;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engTX_PERFMON;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engRX_PERFMON;

    // SIOCTRL Shared
    ENGINE_DESCRIPTOR_TYPE_SV10          *engSIOCTRL;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engMINION;
    ENGINE_DESCRIPTOR_TYPE_SV10          *engLWLIPT;

    INGRESS_REQUEST_RESPONSE_ENTRY_SV10  *ingress_req_table;
    INGRESS_REQUEST_RESPONSE_ENTRY_SV10  *ingress_res_table;
    ROUTE_GANG_ENTRY_SV10                *ganged_link_table;

    LwU32                       link_clock_khz;

    LwBool                      fatal_error_oclwrred;
    LwBool                      ingress_packet_latched;
    LwBool                      egress_packet_latched;

    // Internal diagnostics
    LwBool                      nea;    // Near end analog
    LwBool                      ned;    // Near end digital
} LWSWITCH_LINK_TYPE_SV10;

//
// Latency counters
//

typedef struct
{
    LwU64 count;
    LwU64 low;
    LwU64 medium;
    LwU64 high;
    LwU64 panic;
}
LWSWITCH_LATENCY_BINS_SV10;

typedef struct
{
    LwU32                   count;
    LwU64                   start_time_nsec;
    LwU64                   last_read_time_nsec;
    LWSWITCH_LATENCY_BINS_SV10  lwrr_latency[LWSWITCH_NUM_LINKS_SV10];
    LWSWITCH_LATENCY_BINS_SV10  last_latency[LWSWITCH_NUM_LINKS_SV10];
    LWSWITCH_LATENCY_BINS_SV10  aclwm_latency[LWSWITCH_NUM_LINKS_SV10];
}
LWSWITCH_LATENCY_VC_SV10;

// VCs DNGRD(1), ATR(2), ATSD(3), and PROBE(4) are not relevant on Intel fabrics
#define LWSWITCH_NUM_VCS_SV10    8

typedef struct
{
    LwU32                   sample_interval_msec;       // Set via LWSWITCH_SET_LATENCY_BINS
    LWSWITCH_LATENCY_VC_SV10 latency[LWSWITCH_NUM_VCS_SV10];
}
LWSWITCH_LATENCY_STATS_SV10;

//
// Thermal alert monitoring
//

typedef struct
{
    LwBool  low_power_mode;
    LwU32   event_id;
    LwU64   time_elapsed_nsec;
} LWSWITCH_THERMAL_ALERT_ENTRY_SV10;

//
// Thermal
//

typedef struct LWSWITCH_THERMAL_INFO_SV10
{
    LwU32   idx_i2c_dev_voltage;

    LWSWITCH_TDIODE_INFO_TYPE   tdiode_center;
    LWSWITCH_TDIODE_INFO_TYPE   tdiode_east;
    LWSWITCH_TDIODE_INFO_TYPE   tdiode_west;
} LWSWITCH_THERMAL_INFO_TYPE_SV10;

typedef struct
{
    LwBool  low_power_mode;
    LwU64   time_last_change_nsec;
    LwU32   event_count;
    LWSWITCH_GPIO_INFO *gpio_thermal_alert;
} LWSWITCH_THERMAL_ALERT_SV10;

//
// Per-chip device information
//
typedef struct
{
    // Latency statistics
    LWSWITCH_LATENCY_STATS_SV10         *latency_stats;

    LWSWITCH_OVERRIDE_TYPE_SV10         overrides;

    // Thermal
    struct LWSWITCH_THERMAL_INFO_SV10   thermal;

    // Thermal alert
    LWSWITCH_THERMAL_ALERT_SV10         thermal_alert;

    LWSWITCH_DECLARE_ENGINE_SV10(PTOP)
    LWSWITCH_DECLARE_ENGINE_SV10(SIOCTRL)
    LWSWITCH_DECLARE_ENGINE_SV10(SIOCTRL_BCAST)
    LWSWITCH_DECLARE_ENGINE_SV10(NPG)
    LWSWITCH_DECLARE_ENGINE_SV10(NPG_BCAST)
    LWSWITCH_DECLARE_ENGINE_SV10(SWX)
    LWSWITCH_DECLARE_ENGINE_SV10(SWX_BCAST)
    LWSWITCH_DECLARE_ENGINE_SV10(CLKS)
    LWSWITCH_DECLARE_ENGINE_SV10(FUSE)
    LWSWITCH_DECLARE_ENGINE_SV10(JTAG)
    LWSWITCH_DECLARE_ENGINE_SV10(PMGR)
    LWSWITCH_DECLARE_ENGINE_SV10(SAW)
    LWSWITCH_DECLARE_ENGINE_SV10(XP3G)
    LWSWITCH_DECLARE_ENGINE_SV10(XVE)
    LWSWITCH_DECLARE_ENGINE_SV10(ROM)
    LWSWITCH_DECLARE_ENGINE_SV10(EXTDEV)
    LWSWITCH_DECLARE_ENGINE_SV10(PRIVMAIN)
    LWSWITCH_DECLARE_ENGINE_SV10(PRIVLOC)

    LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10    subengSIOCTRL[NUM_SIOCTRL_ENGINE_SV10];
    LWLINK_SUBENGINE_DESCRIPTOR_TYPE_SV10    subengSIOCTRL_BCAST[NUM_SIOCTRL_BCAST_ENGINE_SV10];

    NPG_SUBENGINE_DESCRIPTOR_TYPE_SV10       subengNPG[NUM_NPG_ENGINE_SV10];
    NPG_SUBENGINE_DESCRIPTOR_TYPE_SV10       subengNPG_BCAST[NUM_NPG_BCAST_ENGINE_SV10];

    SWX_SUBENGINE_DESCRIPTOR_TYPE_SV10       subengSWX[NUM_SWX_ENGINE_SV10];
    SWX_SUBENGINE_DESCRIPTOR_TYPE_SV10       subengSWX_BCAST[NUM_SWX_BCAST_ENGINE_SV10];

    LWSWITCH_LINK_TYPE_SV10                  link[LWSWITCH_NUM_LINKS_SV10];

    LwBool                                   timer_initialized;

    // GPIO
    LWSWITCH_GPIO_INFO                       *gpio_pin;
    LwU32                                    gpio_pin_size;

    // Interrupts
    LwU32                               intr_enable_legacy;
    LwU32                               intr_enable_uncorr;
    LwU32                               intr_enable_corr;

    //
    // Book-keep interrupt masks to restore them after reset.
    // Note: There is no need to book-keep interrupt masks for LWLink units like
    // DL, MINION, TLC etc. because LWLink init routines would setup them.
    //
    struct
    {
        LwU32 tstate;
        LwU32 fstate;
        LwU32 route;
        LwU32 ingress;
        LwU32 egress;
        LwU32 nport_uc;
        LwU32 nport_c;
        LwU32 lwlipt_uc;
        LwU32 lwlipt_c;
        LwU32 clkcross;
    } intr_mask;

} sv10_device;

#define LWSWITCH_GET_CHIP_DEVICE_SV10(_device)                      \
    (                                                               \
        ((_device)->chip_id == LW_PSMC_BOOT_42_CHIP_ID_SVNP01) ?    \
            ((sv10_device *) _device->chip_device) :                \
            NULL                                                    \
    )

//
// Flush any posted writes with a read.  On silicon we should be careful that writes
// are committed.  On RTL simulation extra flushes are inserted to reduce the time
// the CPU "hung" and dumps an NMI warning.
//
#define LWSWITCH_FLUSH_MMIO(device) ((void)lwswitch_reg_read_32(device, LW_PPRIV_SYS_PRI_FENCE))

#define LWSWITCH_SIM_FLUSH_MMIO(device)             \
    if (IS_RTLSIM(device))                          \
        LWSWITCH_FLUSH_MMIO(device);

//
// Internal function declarations
//
LwlStatus lwswitch_device_discovery_sv10(lwswitch_device *device, LwU32 discovery_offset);
void lwswitch_filter_discovery_sv10(lwswitch_device *device);
LwlStatus lwswitch_process_discovery_sv10(lwswitch_device *device);
lwswitch_device *lwswitch_get_device_by_pci_info_sv10(lwlink_pci_info *info);
LwlStatus lwswitch_ring_master_cmd_sv10(lwswitch_device *device, LwU32 cmd);
void lwswitch_initialize_interrupt_tree_sv10(lwswitch_device *device);
void lwswitch_lib_enable_interrupts_sv10(lwswitch_device *device);
void lwswitch_lib_disable_interrupts_sv10(lwswitch_device *device);
LwlStatus lwswitch_lib_service_interrupts_sv10(lwswitch_device *device);
LwlStatus lwswitch_lib_check_interrupts_sv10(lwswitch_device *device);
void lwswitch_set_ganged_link_table_sv10(lwswitch_device *device, LwU32 port, LwU32 firstIndex, LwU32 numEntries);
LwlStatus lwswitch_pmgr_init_config_sv10(lwswitch_device *device);
LwlStatus lwswitch_minion_service_falcon_interrupts_sv10(lwswitch_device *device, LwU32 instance);
LwlStatus lwswitch_ctrl_i2c_indexed_sv10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams);
LwlStatus lwswitch_ctrl_i2c_get_port_info_sv10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams);
LwlStatus lwswitch_corelib_add_link_sv10(lwlink_link *link);
LwlStatus lwswitch_corelib_remove_link_sv10(lwlink_link *link);
LwlStatus lwswitch_corelib_set_dl_link_mode_sv10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_dl_link_mode_sv10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_set_tl_link_mode_sv10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tl_link_mode_sv10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_set_tx_mode_sv10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tx_mode_sv10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_mode_sv10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_rx_mode_sv10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_detect_sv10(lwlink_link *link, LwU32 flags);
LwlStatus lwswitch_corelib_get_rx_detect_sv10(lwlink_link *link);
LwlStatus lwswitch_corelib_write_discovery_token_sv10(lwlink_link *link, LwU64 token);
LwlStatus lwswitch_corelib_read_discovery_token_sv10(lwlink_link *link, LwU64 *token);
void      lwswitch_corelib_training_complete_sv10(lwlink_link *link);
void      lwswitch_init_dlpl_interrupts_sv10(lwlink_link *link);
void      lwswitch_save_lwlink_seed_data_from_minion_to_inforom_sv10(lwswitch_device *device, LwU32 linkId);
void      lwswitch_store_seed_data_from_inforom_to_corelib_sv10(lwswitch_device *device);
LwlStatus lwswitch_get_link_public_id_sv10(lwswitch_device *device, LwU32 linkId, LwU32 *publicId);
LwlStatus lwswitch_get_link_local_idx_sv10(lwswitch_device *device, LwU32 linkId, LwU32 *localLinkIdx);
LwlStatus lwswitch_set_training_error_info_sv10(lwswitch_device *device,
                                                LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams);
LwlStatus lwswitch_ctrl_get_fatal_error_scope_sv10(lwswitch_device *device, LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams);
void      lwswitch_init_scratch_sv10(lwswitch_device *device);
void      *lwswitch_alloc_chipdevice_sv10(lwswitch_device *device);
LwlStatus lwswitch_init_nport_sv10(lwswitch_device *device);
LwlStatus lwswitch_get_soe_ucode_binaries_sv10(lwswitch_device *device, const LwU32 **soe_ucode_data, const LwU32 **soe_ucode_header);

LwlStatus lwswitch_set_training_mode_sv10(lwswitch_device *device);
LwU32     lwswitch_get_sublink_width_sv10(lwswitch_device *device,LwU32 linkNumber);
void lwswitch_corelib_get_uphy_load_sv10(lwlink_link *link, LwBool *bUnlocked);

LwlStatus lwswitch_poll_sublink_state_sv10(lwswitch_device *device, lwlink_link *link);
void      lwswitch_setup_link_loopback_mode_sv10(lwswitch_device *device, LwU32 linkNumber);
void lwswitch_reset_persistent_link_hw_state_sv10(lwswitch_device *device, LwU32 linkNumber);
void lwswitch_store_topology_information_sv10(lwswitch_device *device, lwlink_link *link);
void lwswitch_init_lpwr_regs_sv10(lwlink_link *link);
LwBool lwswitch_i2c_is_device_access_allowed_sv10(lwswitch_device *device, LwU32 port, LwU8 addr, LwBool bIsRead);
LwlStatus lwswitch_parse_bios_image_sv10(lwswitch_device *device);
LwBool    lwswitch_is_link_in_reset_sv10(lwswitch_device *device, lwlink_link *link);
void      lwswitch_init_buffer_ready_sv10(lwswitch_device *device, lwlink_link *link, LwBool bNportBufferReady);
LwlStatus lwswitch_ctrl_get_lwlink_lp_counters_sv10(lwswitch_device *device, LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params);
void      lwswitch_apply_recal_settings_sv10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_service_lwldl_fatal_link_sv10(lwswitch_device *device, LwU32 lwliptInstance, LwU32 link);
LwlStatus lwswitch_ctrl_inband_send_data_sv10(lwswitch_device *device, LWSWITCH_INBAND_SEND_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_read_data_sv10(lwswitch_device *device, LWSWITCH_INBAND_READ_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_flush_data_sv10(lwswitch_device *device, LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_pending_data_stats_sv10(lwswitch_device *device, LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p);
LwlStatus lwswitch_service_minion_link_sv10(lwswitch_device *device, LwU32 lwliptInstance);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwBool    lwswitch_is_cci_supported_sv10(lwswitch_device *device);
LwlStatus lwswitch_get_board_id_sv10(lwswitch_device *device, LwU16 *boardId);
void      lwswitch_fetch_active_repeater_mask_sv10(lwswitch_device *device);
LwU64     lwswitch_get_active_repeater_mask_sv10(lwswitch_device *device);
LwlStatus lwswitch_is_link_in_repeater_mode_sv10(lwswitch_device *device, LwU32 link_id, LwBool *isRepeaterMode);
LwlStatus lwswitch_corelib_set_optical_infinite_mode_sv10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_enable_optical_maintenance_sv10(lwlink_link *link, LwBool bTx);
LwBool    lwswitch_cci_is_optical_link_sv10(lwswitch_device *device, LwU32 linkNumber);
LwlStatus lwswitch_init_cci_sv10(lwswitch_device *device);

LwlStatus lwswitch_corelib_set_optical_iobist_sv10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_set_optical_pretrain_sv10(lwlink_link *link, LwBool bTx, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_pretrain_sv10(lwlink_link *link, LwBool bTx, LwBool *bSuccess);
LwlStatus lwswitch_corelib_init_optical_links_sv10(lwlink_link *link);
LwlStatus lwswitch_corelib_set_optical_force_eq_sv10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_eom_status_sv10(lwlink_link *link, LwBool *bEomLow);
LwlStatus lwswitch_ctrl_i2c_get_dev_info_sv10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams);

LwlStatus lwswitch_ctrl_set_mc_rid_table_sv10(lwswitch_device *device, LWSWITCH_SET_MC_RID_TABLE_PARAMS *p);
LwlStatus lwswitch_ctrl_get_mc_rid_table_sv10(lwswitch_device *device, LWSWITCH_GET_MC_RID_TABLE_PARAMS *p);

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus lwswitch_launch_ALI_sv10(lwswitch_device *device);
#endif
#endif //_SV10_H_
