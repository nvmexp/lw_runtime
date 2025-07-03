/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LR10_H_
#define _LR10_H_

#include "lwlink.h"
#include "lwCpuUuid.h"
#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "pmgr_lwswitch.h"
#include "rom_lwswitch.h"

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

#include "lwswitch/lr10/dev_lws_master.h"

//
// Re-direction to use new common link access wrappers
//

#define LWSWITCH_IS_LINK_ENG_VALID_LR10(_d, _eng, _linknum)  \
    LWSWITCH_IS_LINK_ENG_VALID(_d, _linknum, _eng)

#define LWSWITCH_LINK_OFFSET_LR10(_d, _physlinknum, _eng, _dev, _reg) \
    LWSWITCH_LINK_OFFSET(_d, _physlinknum, _eng, _dev, _reg)

#define LWSWITCH_LINK_WR32_LR10(_d, _physlinknum, _eng, _dev, _reg, _data)  \
    LWSWITCH_LINK_WR32(_d, _physlinknum, _eng, _dev, _reg, _data)

#define LWSWITCH_LINK_RD32_LR10(_d, _physlinknum, _eng, _dev, _reg) \
    LWSWITCH_LINK_RD32(_d, _physlinknum, _eng, _dev, _reg)

#define LWSWITCH_LINK_WR32_IDX_LR10(_d, _physlinknum, _eng, _dev, _reg, _idx, _data)  \
    LWSWITCH_LINK_WR32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx, _data)

#define LWSWITCH_LINK_RD32_IDX_LR10(_d, _physlinknum, _eng, _dev, _reg, _idx) \
    LWSWITCH_LINK_RD32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx)

//
// LWSWITCH_ENG_* MMIO wrappers are to be used for top level discovered
// devices like SAW, FUSE, PMGR, XVE, etc.
//

#define LWSWITCH_ENG_WR32_LR10(_d, _eng, _bcast, _engidx, _dev, _reg, _data) \
    LWSWITCH_ENG_WR32(_d, _eng, _bcast, _engidx, _dev, _reg, _data)

#define LWSWITCH_ENG_RD32_LR10(_d, _eng, _engidx, _dev, _reg)           \
    LWSWITCH_ENG_RD32(_d, _eng, , _engidx, _dev, _reg)

#define LWSWITCH_ENG_WR32_IDX_LR10(_d, _eng, _bcast, _engidx, _dev, _reg, _idx, _data) \
    LWSWITCH_ENG_WR32_IDX(_d, _eng, _bcast, _engidx, _dev, _reg, _idx, _data)

#define LWSWITCH_BCAST_WR32_LR10(_d, _eng, _dev, _reg, _data)           \
    LWSWITCH_ENG_WR32_LR10(_d, _eng, _BCAST, 0, _dev, _reg, _data)

#define LWSWITCH_BCAST_RD32_LR10(_d, _eng, _dev, _reg)           \
    LWSWITCH_ENG_RD32(_d, _eng, _BCAST, 0, bc, _dev, _reg)

#define LWSWITCH_CLK_LWLINK_RD32_LR10(_d, _reg, _idx)                   \
    LWSWITCH_REG_RD32(_d, _PCLOCK, _LWSW_LWLINK##_reg(_idx))

#define LWSWITCH_CLK_LWLINK_WR32_LR10(_d, _reg, _idx, _data)            \
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

#define LWSWITCH_ENG_VALID_LR10(_d, _eng, _engidx)                      \
    (                                                                   \
        ((_engidx < NUM_##_eng##_ENGINE_LR10) &&                        \
        (LWSWITCH_GET_CHIP_DEVICE_LR10(_d)->eng##_eng[_engidx].valid)) ? \
        LW_TRUE : LW_FALSE                                              \
    )

#define LWSWITCH_SAW_RD32_LR10(_d, _dev, _reg)                          \
    LWSWITCH_ENG_RD32_LR10(_d, SAW, 0, _dev, _reg)

#define LWSWITCH_SAW_WR32_LR10(_d, _dev, _reg, _data)                   \
    LWSWITCH_ENG_WR32_LR10(_d, SAW, , 0, _dev, _reg, _data)

#define LWSWITCH_NPG_RD32_LR10(_d, _engidx, _dev, _reg)                 \
    LWSWITCH_ENG_RD32_LR10(_d, NPG, _engidx, _dev, _reg)

#define LWSWITCH_NPG_WR32_LR10(_d, _engidx, _dev, _reg, _data)          \
    LWSWITCH_ENG_WR32_LR10(_d, NPG, , _engidx, _dev, _reg, _data)

#define LWSWITCH_NPGPERF_WR32_LR10(_d, _engidx, _dev, _reg, _data)      \
    LWSWITCH_ENG_WR32_LR10(_d, NPG_PERFMON, , _engidx, _dev, _reg, _data)

#define LWSWITCH_NPORT_RD32_LR10(_d, _engidx, _dev, _reg)               \
    LWSWITCH_ENG_RD32_LR10(_d, NPORT, _engidx, _dev, _reg)

#define LWSWITCH_NPORT_WR32_LR10(_d, _engidx, _dev, _reg, _data)        \
    LWSWITCH_ENG_WR32_LR10(_d, NPORT, , _engidx, _dev, _reg, _data)

#define LWSWITCH_NPORT_MC_BCAST_WR32_LR10(_d, _dev, _reg, _data)        \
    LWSWITCH_BCAST_WR32_LR10(_d, NPORT, _dev, _reg, _data)

#define LWSWITCH_LWLIPT_RD32_LR10(_d, _engidx, _dev, _reg)                 \
    LWSWITCH_ENG_RD32_LR10(_d, LWLIPT, _engidx, _dev, _reg)

#define LWSWITCH_LWLIPT_WR32_LR10(_d, _engidx, _dev, _reg, _data)          \
    LWSWITCH_ENG_WR32_LR10(_d, LWLIPT, , _engidx, _dev, _reg, _data)

typedef struct
{
    LwBool valid;
    LwU32 initialized;
    LwU32 version;
    LwU32 disc_type;
    union
    {
        struct
        {
            LwU32 cluster;
            LwU32 cluster_id;
            LwU32 discovery;                // Used for top level only
        } top;
        struct
        {
            LwU32 uc_addr;
        } uc;
        struct
        {
            LwU32 bc_addr;
            LwU32 mc_addr[3];
        } bc;
    } info;
} ENGINE_DESCRIPTOR_TYPE_LR10;

#define NUM_PTOP_ENGINE_LR10                    1
#define NUM_CLKS_ENGINE_LR10                    1
#define NUM_FUSE_ENGINE_LR10                    1
#define NUM_JTAG_ENGINE_LR10                    1
#define NUM_PMGR_ENGINE_LR10                    1
#define NUM_SAW_ENGINE_LR10                     1
#define NUM_XP3G_ENGINE_LR10                    1
#define NUM_XVE_ENGINE_LR10                     1
#define NUM_ROM_ENGINE_LR10                     1
#define NUM_EXTDEV_ENGINE_LR10                  1
#define NUM_PRIVMAIN_ENGINE_LR10                1
#define NUM_PRIVLOC_ENGINE_LR10                 10
#define NUM_PTIMER_ENGINE_LR10                  1
#define NUM_SOE_ENGINE_LR10                     1
#define NUM_SMR_ENGINE_LR10                     2
#define NUM_I2C_ENGINE_LR10                     1
#define NUM_SE_ENGINE_LR10                      1
#define NUM_THERM_ENGINE_LR10                   1

#define NUM_NPG_ENGINE_LR10                     9
#define NUM_NPG_BCAST_ENGINE_LR10               1
#define NUM_NPG_PERFMON_ENGINE_LR10             9
#define NUM_NPG_PERFMON_BCAST_ENGINE_LR10       1
#define NUM_NPORT_ENGINE_LR10                   36
#define NUM_NPORT_BCAST_ENGINE_LR10             4
#define NUM_NPORT_MULTICAST_ENGINE_LR10         9
#define NUM_NPORT_MULTICAST_BCAST_ENGINE_LR10   1
#define NUM_NPORT_PERFMON_ENGINE_LR10           36
#define NUM_NPORT_PERFMON_BCAST_ENGINE_LR10     4
#define NUM_NPORT_PERFMON_MULTICAST_ENGINE_LR10 9
#define NUM_NPORT_PERFMON_MULTICAST_BCAST_ENGINE_LR10 1

#define NUM_NXBAR_ENGINE_LR10                   4
#define NUM_NXBAR_BCAST_ENGINE_LR10             1
#define NUM_NXBAR_PERFMON_ENGINE_LR10           4
#define NUM_NXBAR_PERFMON_BCAST_ENGINE_LR10     1
#define NUM_TILE_ENGINE_LR10                    16
#define NUM_TILE_BCAST_ENGINE_LR10              4
#define NUM_TILE_MULTICAST_ENGINE_LR10          4
#define NUM_TILE_MULTICAST_BCAST_ENGINE_LR10    1
#define NUM_TILE_PERFMON_ENGINE_LR10            16
#define NUM_TILE_PERFMON_BCAST_ENGINE_LR10      4
#define NUM_TILE_PERFMON_MULTICAST_ENGINE_LR10  4
#define NUM_TILE_PERFMON_MULTICAST_BCAST_ENGINE_LR10 1

//
// Tile Column consists of 4 Tile blocks and 9 Tileout blocks.
// There are 4 Tile Columns, one per each NXBAR.

#define NUM_NXBAR_TCS_LR10   NUM_NXBAR_ENGINE_LR10
#define NUM_NXBAR_TILEOUTS_PER_TC_LR10      9
#define NUM_NXBAR_TILES_PER_TC_LR10         4

#define TILE_TO_LINK(_device, _tc, _tile)                     \
    (                                                         \
        LWSWITCH_ASSERT((_tc < NUM_NXBAR_TCS_LR10))        \
    ,                                                         \
        LWSWITCH_ASSERT((_tile < NUM_NXBAR_TILES_PER_TC_LR10))  \
    ,                                                         \
        ((_tc) *  NUM_NXBAR_TILES_PER_TC_LR10 + (_tile))      \
    )

#define LW_NXBAR_TC_TILEOUT_ERR_FATAL_INTR_EN(i)  (LW_NXBAR_TC_TILEOUT0_ERR_FATAL_INTR_EN +  \
    i * (LW_NXBAR_TC_TILEOUT1_ERR_FATAL_INTR_EN - LW_NXBAR_TC_TILEOUT0_ERR_FATAL_INTR_EN))

#define  LW_NXBAR_TC_TILEOUT_ERR_STATUS(i)  (LW_NXBAR_TC_TILEOUT0_ERR_STATUS +  \
    i * (LW_NXBAR_TC_TILEOUT1_ERR_STATUS - LW_NXBAR_TC_TILEOUT0_ERR_STATUS))

#define LW_NXBAR_TC_TILEOUT_ERR_FIRST(i)  (LW_NXBAR_TC_TILEOUT0_ERR_FIRST +  \
    i * (LW_NXBAR_TC_TILEOUT1_ERR_FIRST - LW_NXBAR_TC_TILEOUT0_ERR_FIRST))

#define LW_NXBAR_TC_TILEOUT_ERR_CYA(i)  (LW_NXBAR_TC_TILEOUT0_ERR_CYA +  \
    i * (LW_NXBAR_TC_TILEOUT1_ERR_CYA - LW_NXBAR_TC_TILEOUT0_ERR_CYA))

#define LWSWITCH_NXBAR_RD32_LR10(_d, _engidx, _dev, _reg)  \
    LWSWITCH_ENG_RD32_LR10(_d, NXBAR, _engidx, _dev, _reg)

#define LWSWITCH_NXBAR_WR32_LR10(_d, _engidx, _dev, _reg, _data)  \
    LWSWITCH_ENG_WR32_LR10(_d, NXBAR, , _engidx, _dev, _reg, _data)

#define LWSWITCH_TILE_RD32_LR10(_d, _engidx, _dev, _reg)  \
    LWSWITCH_ENG_RD32_LR10(_d, TILE, _engidx, _dev, _reg)

#define LWSWITCH_TILE_WR32_LR10(_d, _engidx, _dev, _reg, _data)  \
    LWSWITCH_ENG_WR32_LR10(_d, TILE, , _engidx, _dev, _reg, _data)


#define LW_PPRIV_PRT_PRT_PRIV_ERROR_ADR(i) (LW_PPRIV_PRT_PRT0_PRIV_ERROR_ADR + \
    i * (LW_PPRIV_PRT_PRT1_PRIV_ERROR_ADR - LW_PPRIV_PRT_PRT0_PRIV_ERROR_ADR))

#define LW_PPRIV_PRT_PRT_PRIV_ERROR_WRDAT(i) (LW_PPRIV_PRT_PRT0_PRIV_ERROR_WRDAT + \
    i * (LW_PPRIV_PRT_PRT1_PRIV_ERROR_WRDAT - LW_PPRIV_PRT_PRT0_PRIV_ERROR_WRDAT))

#define LW_PPRIV_PRT_PRT_PRIV_ERROR_INFO(i) (LW_PPRIV_PRT_PRT0_PRIV_ERROR_INFO + \
    i * (LW_PPRIV_PRT_PRT1_PRIV_ERROR_INFO - LW_PPRIV_PRT_PRT0_PRIV_ERROR_INFO))

#define LW_PPRIV_PRT_PRT_PRIV_ERROR_CODE(i) (LW_PPRIV_PRT_PRT0_PRIV_ERROR_CODE + \
    i * (LW_PPRIV_PRT_PRT1_PRIV_ERROR_CODE - LW_PPRIV_PRT_PRT0_PRIV_ERROR_CODE))

#define NUM_LWLW_ENGINE_LR10                            9
#define NUM_LWLW_BCAST_ENGINE_LR10                      1
#define NUM_LWLW_PERFMON_ENGINE_LR10                    9
#define NUM_LWLW_PERFMON_BCAST_ENGINE_LR10              1
#define NUM_MINION_ENGINE_LR10                          9
#define NUM_MINION_BCAST_ENGINE_LR10                    1
#define NUM_LWLIPT_ENGINE_LR10                          9
#define NUM_LWLIPT_BCAST_ENGINE_LR10                    1
#define NUM_LWLIPT_SYS_PERFMON_ENGINE_LR10              9
#define NUM_LWLIPT_SYS_PERFMON_BCAST_ENGINE_LR10        1
#define NUM_LWLTLC_ENGINE_LR10                          36
#define NUM_LWLTLC_BCAST_ENGINE_LR10                    4
#define NUM_LWLTLC_MULTICAST_ENGINE_LR10                9
#define NUM_LWLTLC_MULTICAST_BCAST_ENGINE_LR10          1
#define NUM_TX_PERFMON_ENGINE_LR10                      36
#define NUM_TX_PERFMON_BCAST_ENGINE_LR10                4
#define NUM_TX_PERFMON_MULTICAST_ENGINE_LR10            9
#define NUM_TX_PERFMON_MULTICAST_BCAST_ENGINE_LR10      1
#define NUM_RX_PERFMON_ENGINE_LR10                      36
#define NUM_RX_PERFMON_BCAST_ENGINE_LR10                4
#define NUM_RX_PERFMON_MULTICAST_ENGINE_LR10            9
#define NUM_RX_PERFMON_MULTICAST_BCAST_ENGINE_LR10      1
#define NUM_PLL_ENGINE_LR10                             9
#define NUM_PLL_BCAST_ENGINE_LR10                       1
#define NUM_LWLDL_ENGINE_LR10                           36
#define NUM_LWLDL_BCAST_ENGINE_LR10                     4
#define NUM_LWLDL_MULTICAST_ENGINE_LR10                 9
#define NUM_LWLDL_MULTICAST_BCAST_ENGINE_LR10           1
#define NUM_LWLIPT_LNK_ENGINE_LR10                      36
#define NUM_LWLIPT_LNK_BCAST_ENGINE_LR10                4
#define NUM_LWLIPT_LNK_MULTICAST_ENGINE_LR10            9
#define NUM_LWLIPT_LNK_MULTICAST_BCAST_ENGINE_LR10      1
#define NUM_SYS_PERFMON_ENGINE_LR10                     36
#define NUM_SYS_PERFMON_BCAST_ENGINE_LR10               4
#define NUM_SYS_PERFMON_MULTICAST_ENGINE_LR10           9
#define NUM_SYS_PERFMON_MULTICAST_BCAST_ENGINE_LR10     1
#define LWSWITCH_NUM_PRIV_PRT_LR10                      9


#define LWSWITCH_NPORT_PER_NPG          (NUM_NPORT_ENGINE_LR10/NUM_NPG_ENGINE_LR10)
#define NPORT_TO_LINK(_device, _npg, _nport)                 \
    (                                                        \
        LWSWITCH_ASSERT((_npg < NUM_NPG_ENGINE_LR10))     \
    ,                                                        \
        LWSWITCH_ASSERT((_nport < LWSWITCH_NPORT_PER_NPG))\
    ,                                                        \
        ((_npg) * LWSWITCH_NPORT_PER_NPG + (_nport))         \
    )
#define LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(_lwlipt_idx)     \
    (LWBIT64(LWSWITCH_LINKS_PER_LWLIPT) - 1) << (_lwlipt_idx * LWSWITCH_LINKS_PER_LWLIPT);

#define LWSWITCH_NUM_LINKS_LR10         (NUM_NPORT_ENGINE_LR10)
#define LWSWITCH_NUM_LANES_LR10         4

#define LWSWITCH_LINKS_PER_LWLW         (LWSWITCH_NUM_LINKS_LR10/NUM_LWLW_ENGINE_LR10)
#define LWSWITCH_LINKS_PER_MINION       (LWSWITCH_NUM_LINKS_LR10/NUM_MINION_ENGINE_LR10)
#define LWSWITCH_LINKS_PER_LWLIPT       (LWSWITCH_NUM_LINKS_LR10/NUM_LWLIPT_ENGINE_LR10)
#define LWSWITCH_LINKS_PER_NPG          (LWSWITCH_NUM_LINKS_LR10/NUM_NPG_ENGINE_LR10)

#define LWSWITCH_DECLARE_ENGINE_UC_LR10(_engine)                                \
    ENGINE_DESCRIPTOR_TYPE_LR10  eng##_engine[NUM_##_engine##_ENGINE_LR10];

#define LWSWITCH_DECLARE_ENGINE_LR10(_engine)                                   \
    ENGINE_DESCRIPTOR_TYPE_LR10  eng##_engine[NUM_##_engine##_ENGINE_LR10];   \
    ENGINE_DESCRIPTOR_TYPE_LR10  eng##_engine##_BCAST[NUM_##_engine##_BCAST_ENGINE_LR10];

#define LWSWITCH_LWLIPT_GET_PUBLIC_ID_LR10(_physlinknum) \
    ((_physlinknum)/LWSWITCH_LINKS_PER_LWLIPT)

#define LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(_physlinknum) \
    ((_physlinknum)%LWSWITCH_LINKS_PER_LWLIPT)

#define DISCOVERY_TYPE_UNDEFINED    0
#define DISCOVERY_TYPE_DISCOVERY    1
#define DISCOVERY_TYPE_UNICAST      2
#define DISCOVERY_TYPE_BROADCAST    3

//
// These field #defines describe which physical fabric address bits are
// relevant to the specific remap table address check/remap operation.
//
#define LW_INGRESS_REMAP_ADDR_PHYS_LR10         46:36

#define LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LR10   35:20
#define LW_INGRESS_REMAP_ADR_BASE_PHYS_LR10     35:20
#define LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LR10    35:20

typedef LWSWITCH_LINK_TYPE  LWSWITCH_LINK_TYPE_LR10;

//
// NPORT Portstat information
//

//
// LR10 supports CREQ0(0), DNGRD(1), ATR(2), ATSD(3), PROBE(4), RSP0(5), CREQ1(6), and RSP1(7) VCs. 
// But DNGRD(1), ATR(2), ATSD(3), and PROBE(4) will be never used as PowerPC ATS support is not a POR for LR10 HW.
//
#define LWSWITCH_NUM_VCS_LR10    8

typedef struct
{
    LwU32 count;
    LwU32 low;
    LwU32 medium;
    LwU32 high;
    LwU32 panic;
}
LWSWITCH_LATENCY_BINS_LR10;

typedef struct
{
    LwU32                       count;
    LwU64                       start_time_nsec;
    LwU64                       last_read_time_nsec;
    LWSWITCH_LATENCY_BINS_LR10  aclwm_latency[LWSWITCH_NUM_LINKS_LR10];
}
LWSWITCH_LATENCY_VC_LR10;

typedef struct
{
    LwU32 sample_interval_msec;
    LwU64 last_visited_time_nsec;
    LWSWITCH_LATENCY_VC_LR10 latency[LWSWITCH_NUM_VCS_LR10];
} LWSWITCH_LATENCY_STATS_LR10;

#define LW_NPORT_PORTSTAT_LR10(_block, _reg, _vc, _index)    (LW_NPORT_PORTSTAT ## _block ## _reg ## _0 ## _index +  \
    _vc * (LW_NPORT_PORTSTAT ## _block ## _reg ## _1 ## _index - LW_NPORT_PORTSTAT ## _block ## _reg ## _0 ## _index))

#define LWSWITCH_NPORT_PORTSTAT_RD32_LR10(_d, _engidx, _block, _reg, _vc)               \
    (                                                                                   \
          LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID_LR10(_d, NPORT, _engidx))          \
          ,                                                                             \
          LWSWITCH_PRINT(_d, MMIO,                                                      \
              "%s: MEM_RD NPORT_PORTSTAT[%d]: %s,%s (%06x+%04x)\n",                     \
              __FUNCTION__,                                                             \
              _engidx,                                                                  \
              #_block, #_reg,                                                           \
              LWSWITCH_GET_ENG(_d, NPORT, , _engidx),                                   \
              LW_NPORT_PORTSTAT_LR10(_block, _reg, _vc, _0))                            \
          ,                                                                             \
          lwswitch_reg_read_32(_d,                                                      \
              LWSWITCH_GET_ENG(_d, NPORT, , _engidx) +                                  \
              LW_NPORT_PORTSTAT_LR10(_block, _reg, _vc, _0))                            \
    );                                                                                  \
    ((void)(_d))

#define LWSWITCH_PORTSTAT_BCAST_WR32_LR10(_d, _block, _reg, _idx, _data)                \
    {                                                                                   \
        LWSWITCH_PRINT(_d, MMIO,                                                        \
              "%s: BCAST_WR NPORT_PORTSTAT: %s,%s (%06x+%04x) 0x%08x\n",                \
              __FUNCTION__,                                                             \
              #_block, #_reg,                                                           \
              LWSWITCH_GET_ENG(_d, NPORT, _BCAST, 0),                                   \
              LW_NPORT_PORTSTAT_LR10(_block, _reg, _idx, ), _data);                     \
              LWSWITCH_OFF_WR32(_d,                                                     \
                  LWSWITCH_GET_ENG(_d, NPORT, _BCAST, 0) +                              \
                  LW_NPORT_PORTSTAT_LR10(_block, _reg, _idx, ), _data);                 \
    }

//
// Per-chip device information
//

//
// The chip-specific engine list is used to generate the code to collect
// discovered unit information and coalesce it into the data structures used by
// the common IO library (see io_lwswitch.h).
//
// The PTOP discovery table presents the information on wrappers and sub-units
// in a hierarchical manner.  The top level discovery contains information
// about top level UNICAST units and IP wrappers like NPG, LWLW, and NXBAR.
// Individual units within an IP wrapper are described in discovery sub-tables.
// Each IP wrapper may have MULTICAST descriptors to allow addressing sub-units
// within a wrapper and a cluster of IP wrappers will also have a BCAST
// discovery tables, which have MULTICAST descriptors within them.
// In order to collect all the useful unit information into a single container,
// we need to pick where to find each piece within the parsed discovery table.
// Top level IP wrappers like NPG have a BCAST range to broadcast reads/writes,
// but IP sub-units like NPORT have a MULTICAST range within the BCAST IP 
// wrapper to broadcast to all the sub-units in all the IP wrappers.
// So in the lists below top level IP wrappers (NPG, LWLW, and NXBAR) point
// to the _BCAST IP wrapper, but sub-unit point to the _MULTICAST range inside
// the BCAST unit (_MULTICAST_BCAST).
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
// See https://wiki.lwpu.com/engwiki/index.php/LwSwitch_MMIO_map
#endif
//
// All IP-based (0-based register manuals) engines need to be listed here to
// generate chip-specific handlers as well as in the global common list of all
// engines that have ever existed on *ANY* architecture(s) in order for them
// use common IO wrappers.
//

#define LWSWITCH_LIST_LR10_ENGINES(_op)         \
    _op(XVE, )                                  \
    _op(SAW, )                                  \
    _op(SOE, )                                  \
    _op(SMR, )                                  \
    _op(NPG, _BCAST)                            \
    _op(NPORT, _MULTICAST_BCAST)                \
                                                \
    _op(LWLW, _BCAST)                           \
    _op(MINION, _BCAST)                         \
    _op(LWLIPT, _BCAST)                         \
    _op(LWLIPT_LNK, _MULTICAST_BCAST)           \
    _op(LWLTLC, _MULTICAST_BCAST)               \
    _op(LWLDL, _MULTICAST_BCAST)                \
                                                \
    _op(NXBAR, _BCAST)                          \
    _op(TILE, _MULTICAST_BCAST)                 \
                                                \
    _op(NPG_PERFMON, _BCAST)                    \
    _op(NPORT_PERFMON, _MULTICAST_BCAST)        \
                                                \
    _op(LWLW_PERFMON, _BCAST)                   \
    _op(RX_PERFMON, _MULTICAST_BCAST)           \
    _op(TX_PERFMON, _MULTICAST_BCAST)           \
                                                \
    _op(NXBAR_PERFMON, _BCAST)                  \
    _op(TILE_PERFMON, _MULTICAST_BCAST)         \

typedef struct
{
    struct
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE common[LWSWITCH_ENGINE_ID_SIZE];
    } io;

    LWSWITCH_DECLARE_ENGINE_UC_LR10(PTOP)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(CLKS)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(FUSE)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(JTAG)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(PMGR)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(SAW)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(XP3G)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(XVE)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(ROM)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(EXTDEV)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(PRIVMAIN)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(PRIVLOC)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(PTIMER)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(SOE)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(SMR)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(I2C)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(SE)
    LWSWITCH_DECLARE_ENGINE_UC_LR10(THERM)

    LWSWITCH_DECLARE_ENGINE_LR10(LWLW)
    LWSWITCH_DECLARE_ENGINE_LR10(NXBAR)
    LWSWITCH_DECLARE_ENGINE_LR10(NPG)

    LWSWITCH_DECLARE_ENGINE_LR10(MINION)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLIPT)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLTLC)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLTLC_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLIPT_SYS_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(TX_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(RX_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(TX_PERFMON_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(RX_PERFMON_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(PLL)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLW_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLDL)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLDL_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLIPT_LNK)
    LWSWITCH_DECLARE_ENGINE_LR10(LWLIPT_LNK_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(SYS_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(SYS_PERFMON_MULTICAST)

    LWSWITCH_DECLARE_ENGINE_LR10(NPG_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(NPORT)
    LWSWITCH_DECLARE_ENGINE_LR10(NPORT_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(NPORT_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(NPORT_PERFMON_MULTICAST)

    LWSWITCH_DECLARE_ENGINE_LR10(NXBAR_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(TILE)
    LWSWITCH_DECLARE_ENGINE_LR10(TILE_MULTICAST)
    LWSWITCH_DECLARE_ENGINE_LR10(TILE_PERFMON)
    LWSWITCH_DECLARE_ENGINE_LR10(TILE_PERFMON_MULTICAST)

    // VBIOS configuration Data
    LWSWITCH_BIOS_LWLINK_CONFIG bios_config;

    // GPIO
    const LWSWITCH_GPIO_INFO   *gpio_pin;
    LwU32                       gpio_pin_size;

    // Interrupts
    LwU32                               intr_enable_legacy;
    LwU32                               intr_enable_corr;
    LwU32                               intr_enable_fatal;
    LwU32                               intr_enable_nonfatal;
    LwU32                               intr_minion_dest;

    //
    // Book-keep interrupt masks to restore them after reset.
    // Note: There is no need to book-keep interrupt masks for LWLink units like
    // DL, MINION, TLC etc. because LWLink init routines would setup them.
    //
    struct
    {
        LWSWITCH_INTERRUPT_MASK route;
        LWSWITCH_INTERRUPT_MASK ingress;
        LWSWITCH_INTERRUPT_MASK egress;
        LWSWITCH_INTERRUPT_MASK tstate;
        LWSWITCH_INTERRUPT_MASK sourcetrack;
        LWSWITCH_INTERRUPT_MASK tile;
        LWSWITCH_INTERRUPT_MASK tileout;
    } intr_mask;

    // Latency statistics
    LWSWITCH_LATENCY_STATS_LR10         *latency_stats;

    // External TDIODE info
    LWSWITCH_TDIODE_INFO_TYPE           tdiode;

    // Ganged Link table
    LwU64 *ganged_link_table;
} lr10_device;

#define LWSWITCH_GET_CHIP_DEVICE_LR10(_device)                  \
    (                                                           \
        ((_device)->chip_id == LW_PSMC_BOOT_42_CHIP_ID_LR10) ?  \
            ((lr10_device *) _device->chip_device) :            \
            NULL                                                \
    )

//
// Internal function declarations
//
LwlStatus lwswitch_device_discovery_lr10(lwswitch_device *device, LwU32 discovery_offset);
void lwswitch_filter_discovery_lr10(lwswitch_device *device);
LwlStatus lwswitch_process_discovery_lr10(lwswitch_device *device);
lwswitch_device *lwswitch_get_device_by_pci_info_lr10(lwlink_pci_info *info);
LwlStatus lwswitch_ring_master_cmd_lr10(lwswitch_device *device, LwU32 cmd);
void lwswitch_initialize_interrupt_tree_lr10(lwswitch_device *device);
void lwswitch_lib_enable_interrupts_lr10(lwswitch_device *device);
void lwswitch_lib_disable_interrupts_lr10(lwswitch_device *device);
LwlStatus lwswitch_lib_service_interrupts_lr10(lwswitch_device *device);
LwlStatus lwswitch_lib_check_interrupts_lr10(lwswitch_device *device);
void lwswitch_set_ganged_link_table_lr10(lwswitch_device *device, LwU32 firstIndex, LwU64 *ganged_link_table, LwU32 numEntries);
LwlStatus lwswitch_pmgr_init_config_lr10(lwswitch_device *device);
LwlStatus lwswitch_minion_service_falcon_interrupts_lr10(lwswitch_device *device, LwU32 instance);
LwlStatus lwswitch_ctrl_i2c_indexed_lr10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams);
LwlStatus lwswitch_ctrl_i2c_get_port_info_lr10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams);
void lwswitch_translate_error_lr10(LWSWITCH_ERROR_TYPE         *error_entry,
                                   LWSWITCH_LWLINK_ARCH_ERROR  *arch_error,
                                   LWSWITCH_LWLINK_HW_ERROR    *hw_error);
LwlStatus lwswitch_corelib_add_link_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_remove_link_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_set_dl_link_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_dl_link_mode_lr10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_set_tl_link_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tl_link_mode_lr10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_set_tx_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tx_mode_lr10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_rx_mode_lr10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_detect_lr10(lwlink_link *link, LwU32 flags);
LwlStatus lwswitch_corelib_get_rx_detect_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_write_discovery_token_lr10(lwlink_link *link, LwU64 token);
LwlStatus lwswitch_corelib_read_discovery_token_lr10(lwlink_link *link, LwU64 *token);
void      lwswitch_corelib_training_complete_lr10(lwlink_link *link);
LwBool    lwswitch_link_lane_reversed_lr10(lwswitch_device *device, LwU32 linkId);
void      lwswitch_save_lwlink_seed_data_from_minion_to_inforom_lr10(lwswitch_device *device, LwU32 linkId);
void      lwswitch_store_seed_data_from_inforom_to_corelib_lr10(lwswitch_device *device);
LwBool    lwswitch_is_link_in_reset_lr10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_wait_for_tl_request_ready_lr10(lwlink_link *link);
LwlStatus lwswitch_request_tl_link_state_lr10(lwlink_link *link, LwU32 tlLinkState, LwBool bSync);
void      lwswitch_exelwte_unilateral_link_shutdown_lr10(lwlink_link *link);
LwlStatus lwswitch_get_link_public_id_lr10(lwswitch_device *device, LwU32 linkId, LwU32 *publicId);
LwlStatus lwswitch_get_link_local_idx_lr10(lwswitch_device *device, LwU32 linkId, LwU32 *localLinkIdx);
LwlStatus lwswitch_set_training_error_info_lr10(lwswitch_device *device,
                                                LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams);
LwlStatus lwswitch_ctrl_get_fatal_error_scope_lr10(lwswitch_device *device, LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams);
void      lwswitch_init_scratch_lr10(lwswitch_device *device);
void      lwswitch_init_dlpl_interrupts_lr10(lwlink_link *link);
LwlStatus lwswitch_init_nport_lr10(lwswitch_device *device);
LwlStatus lwswitch_get_soe_ucode_binaries_lr10(lwswitch_device *device, const LwU32 **soe_ucode_data, const LwU32 **soe_ucode_header);
LwlStatus lwswitch_poll_sublink_state_lr10(lwswitch_device *device, lwlink_link *link);
void      lwswitch_setup_link_loopback_mode_lr10(lwswitch_device *device, LwU32 linkNumber);
void lwswitch_reset_persistent_link_hw_state_lr10(lwswitch_device *device, LwU32 linkNumber);
void lwswitch_store_topology_information_lr10(lwswitch_device *device, lwlink_link *link);
void lwswitch_init_lpwr_regs_lr10(lwlink_link *link);
LwlStatus lwswitch_set_training_mode_lr10(lwswitch_device *device);
LwBool lwswitch_i2c_is_device_access_allowed_lr10(lwswitch_device *device, LwU32 port, LwU8 addr, LwBool bIsRead);
LwU32     lwswitch_get_sublink_width_lr10(lwswitch_device *device,LwU32 linkNumber);
LwlStatus lwswitch_parse_bios_image_lr10(lwswitch_device *device);
LwlStatus lwswitch_ctrl_get_throughput_counters_lr10(lwswitch_device *device, LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p);
void lwswitch_corelib_get_uphy_load_lr10(lwlink_link *link, LwBool *bUnlocked);
void      lwswitch_init_buffer_ready_lr10(lwswitch_device *device, lwlink_link *link, LwBool bNportBufferReady);
LwlStatus lwswitch_ctrl_get_lwlink_lp_counters_lr10(lwswitch_device *device, LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params);
LwlStatus lwswitch_service_lwldl_fatal_link_lr10(lwswitch_device *device, LwU32 lwliptInstance, LwU32 link);
LwlStatus lwswitch_ctrl_inband_send_data_lr10(lwswitch_device *device, LWSWITCH_INBAND_SEND_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_read_data_lr10(lwswitch_device *device, LWSWITCH_INBAND_READ_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_flush_data_lr10(lwswitch_device *device, LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_pending_data_stats_lr10(lwswitch_device *device, LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p);
LwlStatus lwswitch_service_minion_link_lr10(lwswitch_device *device, LwU32 lwliptInstance);
void      lwswitch_apply_recal_settings_lr10(lwswitch_device *device, lwlink_link *link);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwBool    lwswitch_is_cci_supported_lr10(lwswitch_device *device);
LwlStatus lwswitch_get_board_id_lr10(lwswitch_device *device, LwU16 *boardId);
LwlStatus lwswitch_is_link_in_repeater_mode_lr10(lwswitch_device *device, LwU32 link_id, LwBool *isRepeaterMode);
void      lwswitch_fetch_active_repeater_mask_lr10(lwswitch_device *device);
LwU64     lwswitch_get_active_repeater_mask_lr10(lwswitch_device *device);
LwlStatus lwswitch_cci_initialization_sequence_lr10(lwswitch_device *device, LwU32 linkNumber);
LwlStatus lwswitch_cci_enable_iobist_lr10(lwswitch_device *device, LwU32 linkNumber, LwBool bEnable);
LwlStatus lwswitch_cci_setup_optical_links_lr10(lwswitch_device *device, LwU64 linkMask);

LwlStatus lwswitch_corelib_set_optical_infinite_mode_lr10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_enable_optical_maintenance_lr10(lwlink_link *link, LwBool bTx);
LwlStatus lwswitch_corelib_set_optical_iobist_lr10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_set_optical_pretrain_lr10(lwlink_link *link, LwBool bTx, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_pretrain_lr10(lwlink_link *link, LwBool bTx, LwBool *bSuccess);
LwlStatus lwswitch_corelib_init_optical_links_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_set_optical_force_eq_lr10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_eom_status_lr10(lwlink_link *link, LwBool *bEomLow);
LwlStatus lwswitch_ctrl_i2c_get_dev_info_lr10(lwswitch_device *device,
                    LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams);

LwlStatus lwswitch_ctrl_set_mc_rid_table_lr10(lwswitch_device *device, LWSWITCH_SET_MC_RID_TABLE_PARAMS *p);
LwlStatus lwswitch_ctrl_get_mc_rid_table_lr10(lwswitch_device *device, LWSWITCH_GET_MC_RID_TABLE_PARAMS *p);

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus lwswitch_launch_ALI_lr10(lwswitch_device *device);
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

#endif //_LR10_H_
