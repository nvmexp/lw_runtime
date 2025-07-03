/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _LS10_H_
#define _LS10_H_

#include "g_lwconfig.h"

#include "export_lwswitch.h"
#include "common_lwswitch.h"

#include "ctrl_dev_internal_lwswitch.h"
#include "ctrl_dev_lwswitch.h"

#include "lwswitch/ls10/dev_master.h"

#define LWSWITCH_NUM_LINKS_LS10                 64
#define LWSWITCH_NUM_LANES_LS10                 2

#define LWSWITCH_LINKS_PER_MINION_LS10          4
#define LWSWITCH_LINKS_PER_LWLIPT_LS10          4
#define LWSWITCH_LINKS_PER_LWLW_LS10            4
#define LWSWITCH_LINKS_PER_NPG_LS10             4

#define LWSWITCH_NPORT_PER_NPG_LS10             LWSWITCH_LINKS_PER_NPG_LS10

#define NUM_PTOP_ENGINE_LS10                    1
#define NUM_FUSE_ENGINE_LS10                    1
#define NUM_GIN_ENGINE_LS10                     1       /* new */
#define NUM_JTAG_ENGINE_LS10                    1

#define NUM_PMGR_ENGINE_LS10                    1
#define NUM_SAW_ENGINE_LS10                     1
#define NUM_ROM_ENGINE_LS10                     1
#define NUM_EXTDEV_ENGINE_LS10                  1
#define NUM_PTIMER_ENGINE_LS10                  1
#define NUM_SOE_ENGINE_LS10                     1
#define NUM_SMR_ENGINE_LS10                     2
#define NUM_SE_ENGINE_LS10                      1
#define NUM_THERM_ENGINE_LS10                   1
#define NUM_XAL_ENGINE_LS10                     1       /* new */
#define NUM_XAL_FUNC_ENGINE_LS10                1       /* new */
#define NUM_XTL_CONFIG_ENGINE_LS10              1       /* new */
#define NUM_XPL_ENGINE_LS10                     1       /* new */
#define NUM_XTL_ENGINE_LS10                     1       /* new */
#define NUM_SYSCTRL_ENGINE_LS10                 1       /* new */
#define NUM_UXL_ENGINE_LS10                     1       /* new */
#define NUM_GPU_PTOP_ENGINE_LS10                1       /* new */
#define NUM_PMC_ENGINE_LS10                     1       /* new */
#define NUM_PBUS_ENGINE_LS10                    1       /* new */
#define NUM_ROM2_ENGINE_LS10                    1       /* new */
#define NUM_GPIO_ENGINE_LS10                    1       /* new */
#define NUM_FSP_ENGINE_LS10                     1       /* new */

#define NUM_CLKS_SYS_ENGINE_LS10                1       /* new */
#define NUM_CLKS_SYSB_ENGINE_LS10               1       /* new */
#define NUM_CLKS_P0_ENGINE_LS10                 4       /* new */
#define NUM_CLKS_P0_BCAST_ENGINE_LS10           1       /* new */
#define NUM_SAW_PM_ENGINE_LS10                  1       /* new */
#define NUM_PCIE_PM_ENGINE_LS10                 1       /* new */
#define NUM_PRT_PRI_HUB_ENGINE_LS10             16      /* new */
#define NUM_PRT_PRI_RS_CTRL_ENGINE_LS10         16      /* new */
#define NUM_PRT_PRI_HUB_BCAST_ENGINE_LS10       1       /* new */
#define NUM_PRT_PRI_RS_CTRL_BCAST_ENGINE_LS10   1       /* new */
#define NUM_SYS_PRI_HUB_ENGINE_LS10             1       /* new */
#define NUM_SYS_PRI_RS_CTRL_ENGINE_LS10         1       /* new */
#define NUM_SYSB_PRI_HUB_ENGINE_LS10            1       /* new */
#define NUM_SYSB_PRI_RS_CTRL_ENGINE_LS10        1       /* new */
#define NUM_PRI_MASTER_RS_ENGINE_LS10           1       /* new */

#define NUM_NPG_ENGINE_LS10                     16
#define NUM_NPG_PERFMON_ENGINE_LS10             NUM_NPG_ENGINE_LS10
#define NUM_NPORT_ENGINE_LS10                   (NUM_NPG_ENGINE_LS10 * LWSWITCH_NPORT_PER_NPG_LS10)
#define NUM_NPORT_MULTICAST_ENGINE_LS10         NUM_NPG_ENGINE_LS10
#define NUM_NPORT_PERFMON_ENGINE_LS10           NUM_NPORT_ENGINE_LS10
#define NUM_NPORT_PERFMON_MULTICAST_ENGINE_LS10 NUM_NPG_ENGINE_LS10

#define NUM_NPG_BCAST_ENGINE_LS10               1
#define NUM_NPG_PERFMON_BCAST_ENGINE_LS10       NUM_NPG_BCAST_ENGINE_LS10
#define NUM_NPORT_BCAST_ENGINE_LS10             LWSWITCH_NPORT_PER_NPG_LS10
#define NUM_NPORT_MULTICAST_BCAST_ENGINE_LS10   NUM_NPG_BCAST_ENGINE_LS10
#define NUM_NPORT_PERFMON_BCAST_ENGINE_LS10     NUM_NPORT_BCAST_ENGINE_LS10
#define NUM_NPORT_PERFMON_MULTICAST_BCAST_ENGINE_LS10 NUM_NPG_BCAST_ENGINE_LS10

#define NUM_LWLW_ENGINE_LS10                            16
#define NUM_LWLIPT_ENGINE_LS10                          NUM_LWLW_ENGINE_LS10
#define NUM_MINION_ENGINE_LS10                          NUM_LWLW_ENGINE_LS10
#define NUM_PLL_ENGINE_LS10                             NUM_LWLW_ENGINE_LS10
#define NUM_CPR_ENGINE_LS10                             NUM_LWLW_ENGINE_LS10        /* new */
#define NUM_LWLW_PERFMON_ENGINE_LS10                    NUM_LWLW_ENGINE_LS10
#define NUM_LWLIPT_SYS_PERFMON_ENGINE_LS10              NUM_LWLW_ENGINE_LS10
#define NUM_LWLDL_MULTICAST_ENGINE_LS10                 NUM_LWLW_ENGINE_LS10
#define NUM_LWLTLC_MULTICAST_ENGINE_LS10                NUM_LWLW_ENGINE_LS10
#define NUM_LWLIPT_LNK_MULTICAST_ENGINE_LS10            NUM_LWLW_ENGINE_LS10
#define NUM_SYS_PERFMON_MULTICAST_ENGINE_LS10           NUM_LWLW_ENGINE_LS10
#define NUM_TX_PERFMON_MULTICAST_ENGINE_LS10            NUM_LWLW_ENGINE_LS10
#define NUM_RX_PERFMON_MULTICAST_ENGINE_LS10            NUM_LWLW_ENGINE_LS10
#define NUM_LWLDL_ENGINE_LS10                           (NUM_LWLW_ENGINE_LS10 * LWSWITCH_LINKS_PER_LWLIPT_LS10)
#define NUM_LWLTLC_ENGINE_LS10                          NUM_LWLDL_ENGINE_LS10
#define NUM_LWLIPT_LNK_ENGINE_LS10                      NUM_LWLDL_ENGINE_LS10
#define NUM_SYS_PERFMON_ENGINE_LS10                     NUM_LWLDL_ENGINE_LS10
#define NUM_TX_PERFMON_ENGINE_LS10                      NUM_LWLDL_ENGINE_LS10
#define NUM_RX_PERFMON_ENGINE_LS10                      NUM_LWLDL_ENGINE_LS10

#define NUM_LWLW_BCAST_ENGINE_LS10                      1
#define NUM_LWLIPT_BCAST_ENGINE_LS10                    NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_MINION_BCAST_ENGINE_LS10                    NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_PLL_BCAST_ENGINE_LS10                       NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_CPR_BCAST_ENGINE_LS10                       NUM_LWLW_BCAST_ENGINE_LS10  /* new */
#define NUM_LWLW_PERFMON_BCAST_ENGINE_LS10              NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_LWLIPT_SYS_PERFMON_BCAST_ENGINE_LS10        NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_LWLDL_MULTICAST_BCAST_ENGINE_LS10           NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_LWLTLC_MULTICAST_BCAST_ENGINE_LS10          NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_LWLIPT_LNK_MULTICAST_BCAST_ENGINE_LS10      NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_SYS_PERFMON_MULTICAST_BCAST_ENGINE_LS10     NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_TX_PERFMON_MULTICAST_BCAST_ENGINE_LS10      NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_RX_PERFMON_MULTICAST_BCAST_ENGINE_LS10      NUM_LWLW_BCAST_ENGINE_LS10
#define NUM_LWLDL_BCAST_ENGINE_LS10                     LWSWITCH_LINKS_PER_LWLIPT_LS10
#define NUM_LWLTLC_BCAST_ENGINE_LS10                    NUM_LWLDL_BCAST_ENGINE_LS10
#define NUM_LWLIPT_LNK_BCAST_ENGINE_LS10                NUM_LWLDL_BCAST_ENGINE_LS10
#define NUM_SYS_PERFMON_BCAST_ENGINE_LS10               NUM_LWLDL_BCAST_ENGINE_LS10
#define NUM_TX_PERFMON_BCAST_ENGINE_LS10                NUM_LWLDL_BCAST_ENGINE_LS10
#define NUM_RX_PERFMON_BCAST_ENGINE_LS10                NUM_LWLDL_BCAST_ENGINE_LS10

#define NUM_NXBAR_ENGINE_LS10                           3
#define NUM_NXBAR_PERFMON_ENGINE_LS10                   NUM_NXBAR_ENGINE_LS10
#define NUM_TILE_MULTICAST_ENGINE_LS10                  NUM_NXBAR_ENGINE_LS10
#define NUM_TILE_PERFMON_MULTICAST_ENGINE_LS10          NUM_NXBAR_ENGINE_LS10
#define NUM_TILE_ENGINE_LS10                            (12 * NUM_NXBAR_ENGINE_LS10)
#define NUM_TILE_PERFMON_ENGINE_LS10                    NUM_TILE_ENGINE_LS10
#define NUM_TILEOUT_MULTICAST_ENGINE_LS10               NUM_NXBAR_ENGINE_LS10               /* new */
#define NUM_TILEOUT_PERFMON_MULTICAST_ENGINE_LS10       NUM_NXBAR_ENGINE_LS10               /* new */
#define NUM_TILEOUT_ENGINE_LS10                         NUM_TILE_ENGINE_LS10                /* new */
#define NUM_TILEOUT_PERFMON_ENGINE_LS10                 NUM_TILE_ENGINE_LS10                /* new */

#define NUM_NXBAR_BCAST_ENGINE_LS10                     1
#define NUM_NXBAR_PERFMON_BCAST_ENGINE_LS10             NUM_NXBAR_BCAST_ENGINE_LS10
#define NUM_TILE_MULTICAST_BCAST_ENGINE_LS10            NUM_NXBAR_BCAST_ENGINE_LS10
#define NUM_TILE_PERFMON_MULTICAST_BCAST_ENGINE_LS10    NUM_NXBAR_BCAST_ENGINE_LS10
#define NUM_TILE_BCAST_ENGINE_LS10                      12
#define NUM_TILE_PERFMON_BCAST_ENGINE_LS10              NUM_TILE_BCAST_ENGINE_LS10
#define NUM_TILEOUT_MULTICAST_BCAST_ENGINE_LS10         NUM_NXBAR_BCAST_ENGINE_LS10         /* new */
#define NUM_TILEOUT_PERFMON_MULTICAST_BCAST_ENGINE_LS10 NUM_NXBAR_BCAST_ENGINE_LS10         /* new */
#define NUM_TILEOUT_BCAST_ENGINE_LS10                   NUM_TILE_BCAST_ENGINE_LS10          /* new */
#define NUM_TILEOUT_PERFMON_BCAST_ENGINE_LS10           NUM_TILE_BCAST_ENGINE_LS10          /* new */

#define LWSWITCH_NUM_LINKS_PER_LWLIPT_LS10              (LWSWITCH_NUM_LINKS_LS10/NUM_LWLIPT_ENGINE_LS10)

#define LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LS10(_physlinknum) \
    ((_physlinknum)%LWSWITCH_NUM_LINKS_PER_LWLIPT_LS10)

#define LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64_LS10(_lwlipt_idx)     \
    (LWBIT64(LWSWITCH_LINKS_PER_LWLIPT_LS10) - 1) << (_lwlipt_idx * LWSWITCH_LINKS_PER_LWLIPT_LS10);

#define DMA_ADDR_WIDTH_LS10     64

//
// Helpful IO wrappers
//

#define LWSWITCH_NPORT_WR32_LS10(_d, _engidx, _dev, _reg, _data)        \
    LWSWITCH_ENG_WR32(_d, NPORT, , _engidx, _dev, _reg, _data)

#define LWSWITCH_NPORT_RD32_LS10(_d, _engidx, _dev, _reg)               \
    LWSWITCH_ENG_RD32(_d, NPORT, , _engidx, _dev, _reg)

#define LWSWITCH_MINION_WR32_LS10(_d, _engidx, _dev, _reg, _data)       \
    LWSWITCH_ENG_WR32(_d, MINION, , _engidx, _dev, _reg, _data)

#define LWSWITCH_MINION_RD32_LS10(_d, _engidx, _dev, _reg)              \
    LWSWITCH_ENG_RD32(_d, MINION, , _engidx, _dev, _reg)

#define LWSWITCH_MINION_WR32_BCAST_LS10(_d, _dev, _reg, _data)          \
    LWSWITCH_ENG_WR32(_d, MINION, _BCAST, 0, _dev, _reg, _data)

#define LWSWITCH_NPG_WR32_LS10(_d, _engidx, _dev, _reg, _data)          \
    LWSWITCH_ENG_WR32(_d, NPG, , _engidx, _dev, _reg, _data)

#define LWSWITCH_NPG_RD32_LS10(_d, _engidx, _dev, _reg)                 \
    LWSWITCH_ENG_RD32(_d, NPG, , _engidx, _dev, _reg)

//
// Per-chip device information
//

#define DISCOVERY_TYPE_UNDEFINED    0
#define DISCOVERY_TYPE_DISCOVERY    1
#define DISCOVERY_TYPE_UNICAST      2
#define DISCOVERY_TYPE_BROADCAST    3

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
} ENGINE_DISCOVERY_TYPE_LS10;

#define LWSWITCH_DECLARE_ENGINE_UC_LS10(_engine)                                \
    ENGINE_DISCOVERY_TYPE_LS10  eng##_engine[NUM_##_engine##_ENGINE_LS10];

#define LWSWITCH_DECLARE_ENGINE_LS10(_engine)                                   \
    ENGINE_DISCOVERY_TYPE_LS10  eng##_engine[NUM_##_engine##_ENGINE_LS10];     \
    ENGINE_DISCOVERY_TYPE_LS10  eng##_engine##_BCAST[NUM_##_engine##_BCAST_ENGINE_LS10];

#define LWSWITCH_LIST_LS10_ENGINE_UC(_op)       \
    _op(PTOP)                                   \
    _op(FUSE)                                   \
    _op(GIN)                                    \
    _op(JTAG)                                   \
    _op(PMGR)                                   \
    _op(SAW)                                    \
    _op(ROM)                                    \
    _op(EXTDEV)                                 \
    _op(PTIMER)                                 \
    _op(SOE)                                    \
    _op(SMR)                                    \
    _op(SE)                                     \
    _op(THERM)                                  \
    _op(XAL)                                    \
    _op(XAL_FUNC)                               \
    _op(XTL_CONFIG)                             \
    _op(XPL)                                    \
    _op(XTL)                                    \
    _op(UXL)                                    \
    _op(GPU_PTOP)                               \
    _op(PMC)                                    \
    _op(PBUS)                                   \
    _op(ROM2)                                   \
    _op(GPIO)                                   \
    _op(FSP)                                    \
    _op(CLKS_SYS)                               \
    _op(CLKS_SYSB)                              \
    _op(CLKS_P0)                                \
    _op(CLKS_P0_BCAST)                          \
    _op(SAW_PM)                                 \
    _op(PCIE_PM)                                \
    _op(SYS_PRI_HUB)                            \
    _op(SYS_PRI_RS_CTRL)                        \
    _op(SYSB_PRI_HUB)                           \
    _op(SYSB_PRI_RS_CTRL)                       \
    _op(PRI_MASTER_RS)                          \

#define LWSWITCH_LIST_PRI_HUB_LS10_ENGINE(_op)  \
    _op(PRT_PRI_HUB)                            \
    _op(PRT_PRI_RS_CTRL)                        \
    _op(PRT_PRI_HUB_BCAST)                      \
    _op(PRT_PRI_RS_CTRL_BCAST)                  \

#define LWSWITCH_LIST_NPG_LS10_ENGINE(_op)      \
    _op(NPG)                                    \
    _op(NPG_PERFMON)                            \
    _op(NPORT)                                  \
    _op(NPORT_MULTICAST)                        \
    _op(NPORT_PERFMON)                          \
    _op(NPORT_PERFMON_MULTICAST)

#define LWSWITCH_LIST_LWLW_LS10_ENGINE(_op)     \
    _op(LWLW)                                   \
    _op(LWLIPT)                                 \
    _op(MINION)                                 \
    _op(CPR)                                    \
    _op(LWLW_PERFMON)                           \
    _op(LWLIPT_SYS_PERFMON)                     \
    _op(LWLDL_MULTICAST)                        \
    _op(LWLTLC_MULTICAST)                       \
    _op(LWLIPT_LNK_MULTICAST)                   \
    _op(SYS_PERFMON_MULTICAST)                  \
    _op(TX_PERFMON_MULTICAST)                   \
    _op(RX_PERFMON_MULTICAST)                   \
    _op(LWLDL)                                  \
    _op(LWLTLC)                                 \
    _op(LWLIPT_LNK)                             \
    _op(SYS_PERFMON)                            \
    _op(TX_PERFMON)                             \
    _op(RX_PERFMON)

#define LWSWITCH_LIST_NXBAR_LS10_ENGINE(_op)    \
    _op(NXBAR)                                  \
    _op(NXBAR_PERFMON)                          \
    _op(TILE_MULTICAST)                         \
    _op(TILE_PERFMON_MULTICAST)                 \
    _op(TILE)                                   \
    _op(TILE_PERFMON)                           \
    _op(TILEOUT_MULTICAST)                      \
    _op(TILEOUT_PERFMON_MULTICAST)              \
    _op(TILEOUT)                                \
    _op(TILEOUT_PERFMON)

#define LWSWITCH_LIST_LS10_ENGINE(_op)          \
    LWSWITCH_LIST_NPG_LS10_ENGINE(_op)          \
    LWSWITCH_LIST_LWLW_LS10_ENGINE(_op)         \
    LWSWITCH_LIST_NXBAR_LS10_ENGINE(_op)

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
// See https://wiki.lwpu.com/engwiki/index.php/LwSwitch_MMIO_map
//
// All IP-based (0-based register manuals) engines need to be listed here to
// generate chip-specific handlers as well as in the global common list of all
// engines that have ever existed on *ANY* architecture(s) in order for them
// use common IO wrappers.
//

#define LWSWITCH_LIST_LS10_ENGINES(_op)         \
    _op(GIN, )                                  \
    _op(XAL, )                                  \
    _op(XPL, )                                  \
    _op(XTL, )                                  \
    _op(SAW, )                                  \
    _op(SOE, )                                  \
    _op(SMR, )                                  \
                                                \
    _op(PRT_PRI_HUB, _BCAST)                    \
    _op(PRT_PRI_RS_CTRL, _BCAST)                \
    _op(SYS_PRI_HUB, )                          \
    _op(SYS_PRI_RS_CTRL, )                      \
    _op(SYSB_PRI_HUB, )                         \
    _op(SYSB_PRI_RS_CTRL, )                     \
    _op(PRI_MASTER_RS, )                        \
    _op(PTIMER, )                               \
    _op(CLKS_SYS, )                             \
    _op(CLKS_SYSB, )                            \
    _op(CLKS_P0, _BCAST)                        \
                                                \
    _op(NPG, _BCAST)                            \
    _op(NPORT, _MULTICAST_BCAST)                \
                                                \
    _op(LWLW, _BCAST)                           \
    _op(MINION, _BCAST)                         \
    _op(LWLIPT, _BCAST)                         \
    _op(CPR, _BCAST)                            \
    _op(LWLIPT_LNK, _MULTICAST_BCAST)           \
    _op(LWLTLC, _MULTICAST_BCAST)               \
    _op(LWLDL, _MULTICAST_BCAST)                \
                                                \
    _op(NXBAR, _BCAST)                          \
    _op(TILE, _MULTICAST_BCAST)                 \
    _op(TILEOUT, _MULTICAST_BCAST)              \
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
    _op(TILEOUT_PERFMON, _MULTICAST_BCAST)      \

//
// These field #defines describe which physical fabric address bits are
// relevant to the specific remap table address check/remap operation.
//

#define LW_INGRESS_REMAP_ADDR_PHYS_LS10         51:39       /* LR10: 46:36 */

#define LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LS10   38:21       /* LR10: 35:20 */
#define LW_INGRESS_REMAP_ADR_BASE_PHYS_LS10     38:21       /* LR10: 35:20 */
#define LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LS10    38:21       /* LR10: 35:20 */

//
// Multicast REMAP table is not indexed through the same _RAM_SEL mechanism as
// other REMAP tables, but we want to be able to use the same set of APIs for
// all the REMAP tables, so define a special RAM_SEL value for MCREMAP that
// does not conflict with the existing definitions.
//
#define LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT_MULTICAST_REMAPRAM (DRF_MASK(LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS) + 1)

//
// NPORT Portstat information
//

//
// LS10 supports CREQ0(0), DNGRD(1), ATR(2), ATSD(3), PROBE(4), RSP0(5), CREQ1(6), and RSP1(7) VCs.
// But DNGRD(1), ATR(2), ATSD(3), and PROBE(4) will be never used as PowerPC ATS support is not a POR for LR10 HW.
//
#define LWSWITCH_NUM_VCS_LS10    8

typedef struct
{
    LwU32 count;
    LwU32 low;
    LwU32 medium;
    LwU32 high;
    LwU32 panic;
}
LWSWITCH_LATENCY_BINS_LS10;

typedef struct
{
    LwU32                       count;
    LwU64                       start_time_nsec;
    LwU64                       last_read_time_nsec;
    LWSWITCH_LATENCY_BINS_LS10  aclwm_latency[LWSWITCH_NUM_LINKS_LS10];
}
LWSWITCH_LATENCY_VC_LS10;

typedef struct
{
    LwU32 sample_interval_msec;
    LwU64 last_visited_time_nsec;
    LWSWITCH_LATENCY_VC_LS10 latency[LWSWITCH_NUM_VCS_LS10];
} LWSWITCH_LATENCY_STATS_LS10;

#define LW_NPORT_PORTSTAT_LS10(_block, _reg, _vc, _hi_lo)    (LW_NPORT_PORTSTAT ## _block ## _reg ## _0 ## _hi_lo +  \
    _vc * (LW_NPORT_PORTSTAT ## _block ## _reg ## _1 ## _hi_lo - LW_NPORT_PORTSTAT ## _block ## _reg ## _0 ## _hi_lo))

#define LWSWITCH_NPORT_PORTSTAT_RD32_LS10(_d, _engidx, _block, _reg, _hi_lo, _vc)   \
    (                                                                               \
          LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID_LS10(_d, NPORT, _engidx))      \
          ,                                                                         \
          LWSWITCH_PRINT(_d, MMIO,                                                  \
              "%s: MEM_RD NPORT_PORTSTAT[%d]: %s,%s,_%s,%s (%06x+%04x)\n",          \
              __FUNCTION__,                                                         \
              _engidx,                                                              \
              #_block, #_reg, #_vc, #_hi_lo,                                        \
              LWSWITCH_GET_ENG(_d, NPORT, , _engidx),                               \
              LW_NPORT_PORTSTAT_LS10(_block, _reg, _vc, _hi_lo))                    \
          ,                                                                         \
          lwswitch_reg_read_32(_d,                                                  \
              LWSWITCH_GET_ENG(_d, NPORT, , _engidx) +                              \
              LW_NPORT_PORTSTAT_LS10(_block, _reg, _vc, _hi_lo))                    \
    );                                                                              \
    ((void)(_d))

#define LWSWITCH_PORTSTAT_BCAST_WR32_LS10(_d, _block, _reg, _idx, _data)            \
    {                                                                               \
         LWSWITCH_PRINT(_d, MMIO,                                                   \
              "%s: BCAST_WR NPORT_PORTSTAT: %s,%s (%06x+%04x) 0x%08x\n",            \
              __FUNCTION__,                                                         \
              #_block, #_reg,                                                       \
              LWSWITCH_GET_ENG(_d, NPORT, _BCAST, 0),                               \
              LW_NPORT_PORTSTAT_LS10(_block, _reg, _idx, ), _data);                 \
          LWSWITCH_OFF_WR32(_d,                                                     \
              LWSWITCH_GET_ENG(_d, NPORT, _BCAST, 0) +                              \
              LW_NPORT_PORTSTAT_LS10(_block, _reg, _idx, ), _data);                 \
    }

typedef struct
{
    struct
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE common[LWSWITCH_ENGINE_ID_SIZE];
    } io;

    LWSWITCH_LIST_LS10_ENGINE_UC(LWSWITCH_DECLARE_ENGINE_UC_LS10)
    LWSWITCH_LIST_PRI_HUB_LS10_ENGINE(LWSWITCH_DECLARE_ENGINE_UC_LS10)
    LWSWITCH_LIST_LS10_ENGINE(LWSWITCH_DECLARE_ENGINE_LS10)

    // Interrupts
    LwU32       intr_minion_dest;

    // VBIOS configuration Data
    LWSWITCH_BIOS_LWLINK_CONFIG bios_config;

    // GPIO
    const LWSWITCH_GPIO_INFO   *gpio_pin;
    LwU32                       gpio_pin_size;

    // Latency statistics
    LWSWITCH_LATENCY_STATS_LS10         *latency_stats;

    // External TDIODE info
    LWSWITCH_TDIODE_INFO_TYPE           tdiode;

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
        LWSWITCH_INTERRUPT_MASK mc_tstate;
        LWSWITCH_INTERRUPT_MASK red_tstate;
    } intr_mask;

    // Ganged Link table
    LwU64 *ganged_link_table;

    //LWSWITCH Minion core
    LwU32 minionEngArch;

    LwBool riscvManifestBoot;

} ls10_device;

//
// Helpful IO wrappers
//

#define LWSWITCH_GET_CHIP_DEVICE_LS10(_device)                  \
    (                                                           \
        ((_device)->chip_id == LW_PMC_BOOT_42_CHIP_ID_LS10) ?   \
            ((ls10_device *) _device->chip_device) :            \
            NULL                                                \
    )

#define LWSWITCH_ENG_VALID_LS10(_d, _eng, _engidx)                      \
    (                                                                   \
        ((_engidx < NUM_##_eng##_ENGINE_LS10) &&                        \
        (LWSWITCH_GET_CHIP_DEVICE_LS10(_d)->eng##_eng[_engidx].valid)) ? \
        LW_TRUE : LW_FALSE                                              \
    )

#define LWSWITCH_ENG_WR32_LS10(_d, _eng, _bcast, _engidx, _dev, _reg, _data) \
    LWSWITCH_ENG_WR32(_d, _eng, _bcast, _engidx, _dev, _reg, _data)

#define LWSWITCH_ENG_RD32_LS10(_d, _eng, _engidx, _dev, _reg)       \
    LWSWITCH_ENG_RD32(_d, _eng, , _engidx, _dev, _reg)

#define LWSWITCH_BCAST_WR32_LS10(_d, _eng, _dev, _reg, _data)       \
    LWSWITCH_ENG_WR32(_d, _eng, _BCAST, 0, _dev, _reg, _data)

#define LWSWITCH_BCAST_RD32_LS10(_d, _eng, _dev, _reg)              \
    LWSWITCH_ENG_RD32(_d, _eng, _BCAST, 0, _dev, _reg)

#define LWSWITCH_SOE_WR32_LS10(_d, _instance, _dev, _reg, _data)    \
    LWSWITCH_ENG_WR32(_d, SOE, , _instance, _dev, _reg, _data)

#define LWSWITCH_SOE_RD32_LS10(_d, _instance, _dev, _reg)           \
    LWSWITCH_ENG_RD32(_d, SOE, , _instance, _dev, _reg)

#define LWSWITCH_NPORT_BCAST_WR32_LS10(_d, _dev, _reg, _data)       \
    LWSWITCH_ENG_WR32(_d, NPORT, _BCAST, 0, _dev, _reg, _data)

#define LWSWITCH_SAW_WR32_LS10(_d, _dev, _reg, _data)               \
    LWSWITCH_ENG_WR32(_d, SAW, , 0, _dev, _reg, _data)

#define LWSWITCH_NPORT_MC_BCAST_WR32_LS10(_d, _dev, _reg, _data)    \
    LWSWITCH_BCAST_WR32_LS10(_d, NPORT, _dev, _reg, _data)

//
// Per link register access routines
// LINK_* MMIO wrappers are used to reference per-link engine instances
//

#define LWSWITCH_IS_LINK_ENG_VALID_LS10(_d, _eng, _linknum)             \
    LWSWITCH_IS_LINK_ENG_VALID(_d, _linknum, _eng)

#define LWSWITCH_LINK_OFFSET_LS10(_d, _physlinknum, _eng, _dev, _reg)   \
    LWSWITCH_LINK_OFFSET(_d, _physlinknum, _eng, _dev, _reg)

#define LWSWITCH_LINK_WR32_LS10(_d, _physlinknum, _eng, _dev, _reg, _data) \
    LWSWITCH_LINK_WR32(_d, _physlinknum, _eng, _dev, _reg, _data)

#define LWSWITCH_LINK_RD32_LS10(_d, _physlinknum, _eng, _dev, _reg)     \
    LWSWITCH_LINK_RD32(_d, _physlinknum, _eng, _dev, _reg)

#define LWSWITCH_LINK_WR32_IDX_LS10(_d, _physlinknum, _eng, _dev, _reg, _idx, _data) \
    LWSWITCH_LINK_WR32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx, _data)

#define LWSWITCH_LINK_RD32_IDX_LS10(_d, _physlinknum, _eng, _dev, _reg, _idx) \
    LWSWITCH_LINK_RD32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx)

#define LWSWITCH_MINION_LINK_WR32_LS10(_d, _physlinknum, _dev, _reg, _data)   \
    LWSWITCH_LINK_WR32(_d, _physlinknum, MINION, _dev, _reg, _data)

#define LWSWITCH_MINION_LINK_RD32_LS10(_d, _physlinknum, _dev, _reg) \
    LWSWITCH_LINK_RD32(_d, _physlinknum, MINION, _dev, _reg)

//
// MINION
//

typedef const struct
{
    LwU32 osCodeOffset;
    LwU32 osCodeSize;
    LwU32 osDataOffset;
    LwU32 osDataSize;
    LwU32 numApps;
    LwU32 appCodeStart;
    LwU32 appDataStart;
    LwU32 codeOffset;
    LwU32 codeSize;
    LwU32 dataOffset;
    LwU32 dataSize;
} FALCON_UCODE_HDR_INFO_LS10, *PFALCON_UCODE_HDR_INFO_LS10;

typedef const struct
{
      //
      // Version 1
      // Version 2
      // Vesrion 3 = for Partition boot
      // Vesrion 4 = for eb riscv boot
      //
      LwU32  version;                         // structure version
      LwU32  bootloaderOffset;
      LwU32  bootloaderSize;
      LwU32  bootloaderParamOffset;
      LwU32  bootloaderParamSize;
      LwU32  riscvElfOffset;
      LwU32  riscvElfSize;
      LwU32  appVersion;                      // Changelist number associated with the image
      //
      // Manifest contains information about Monitor and it is
      // input to BR
      //
      LwU32  manifestOffset;
      LwU32  manifestSize;
      //
      // Monitor Data offset within RISCV image and size
      //
      LwU32  monitorDataOffset;
      LwU32  monitorDataSize;
      //
      // Monitor Code offset withtin RISCV image and size
      //
      LwU32  monitorCodeOffset;
      LwU32  monitorCodeSize;
      LwU32  bIsMonitorEnabled;
      //
      // Swbrom Code offset within RISCV image and size
      //
      LwU32  swbromCodeOffset;
      LwU32  swbromCodeSize;
      //
      // Swbrom Data offset within RISCV image and size
      //
      LwU32  swbromDataOffset;
      LwU32  swbromDataSize;
} RISCV_UCODE_HDR_INFO_LS10, *PRISCV_UCODE_HDR_INFO_LS10;

//
// HAL functions shared by LR10 and used by LS10
//

#define lwswitch_is_link_valid_ls10                 lwswitch_is_link_valid_lr10
#define lwswitch_is_link_in_use_ls10                lwswitch_is_link_in_use_lr10

#define lwswitch_initialize_device_state_ls10       lwswitch_initialize_device_state_lr10
#define lwswitch_deassert_link_reset_ls10           lwswitch_deassert_link_reset_lr10
#define lwswitch_determine_platform_ls10            lwswitch_determine_platform_lr10
#define lwswitch_get_swap_clk_default_ls10          lwswitch_get_swap_clk_default_lr10
#define lwswitch_post_init_device_setup_ls10        lwswitch_post_init_device_setup_lr10
#define lwswitch_ctrl_get_bios_info_ls10            lwswitch_ctrl_get_bios_info_lr10
#define lwswitch_hw_counter_shutdown_ls10           lwswitch_hw_counter_shutdown_lr10
#define lwswitch_hw_counter_read_counter_ls10       lwswitch_hw_counter_read_counter_lr10

#define lwswitch_ecc_writeback_task_ls10            lwswitch_ecc_writeback_task_lr10
#define lwswitch_ctrl_get_routing_id_ls10           lwswitch_ctrl_get_routing_id_lr10
#define lwswitch_ctrl_set_routing_id_valid_ls10     lwswitch_ctrl_set_routing_id_valid_lr10
#define lwswitch_ctrl_set_routing_id_ls10           lwswitch_ctrl_set_routing_id_lr10
#define lwswitch_ctrl_set_routing_lan_ls10          lwswitch_ctrl_set_routing_lan_lr10
#define lwswitch_ctrl_get_routing_lan_ls10          lwswitch_ctrl_get_routing_lan_lr10
#define lwswitch_ctrl_set_routing_lan_valid_ls10    lwswitch_ctrl_set_routing_lan_valid_lr10
#define lwswitch_ctrl_set_ingress_request_table_ls10    lwswitch_ctrl_set_ingress_request_table_lr10
#define lwswitch_ctrl_get_ingress_request_table_ls10    lwswitch_ctrl_get_ingress_request_table_lr10
#define lwswitch_ctrl_set_ingress_request_valid_ls10    lwswitch_ctrl_set_ingress_request_valid_lr10
#define lwswitch_ctrl_get_ingress_response_table_ls10   lwswitch_ctrl_get_ingress_response_table_lr10
#define lwswitch_ctrl_set_ingress_response_table_ls10   lwswitch_ctrl_set_ingress_response_table_lr10

#define lwswitch_ctrl_get_info_ls10                 lwswitch_ctrl_get_info_lr10

#define lwswitch_ctrl_set_switch_port_config_ls10   lwswitch_ctrl_set_switch_port_config_lr10
#define lwswitch_reset_and_drain_links_ls10         lwswitch_reset_and_drain_links_lr10
#define lwswitch_set_fatal_error_ls10               lwswitch_set_fatal_error_lr10
#define lwswitch_ctrl_get_fom_values_ls10           lwswitch_ctrl_get_fom_values_lr10
#define lwswitch_ctrl_get_throughput_counters_ls10  lwswitch_ctrl_get_throughput_counters_lr10

#define lwswitch_save_lwlink_seed_data_from_minion_to_inforom_ls10  lwswitch_save_lwlink_seed_data_from_minion_to_inforom_lr10
#define lwswitch_store_seed_data_from_inforom_to_corelib_ls10       lwswitch_store_seed_data_from_inforom_to_corelib_lr10

#define lwswitch_read_oob_blacklist_state_ls10      lwswitch_read_oob_blacklist_state_lr10
#define lwswitch_write_fabric_state_ls10            lwswitch_write_fabric_state_lr10

#define lwswitch_corelib_add_link_ls10              lwswitch_corelib_add_link_lr10
#define lwswitch_corelib_remove_link_ls10           lwswitch_corelib_remove_link_lr10
#define lwswitch_corelib_set_tl_link_mode_ls10      lwswitch_corelib_set_tl_link_mode_lr10
#define lwswitch_corelib_set_rx_mode_ls10           lwswitch_corelib_set_rx_mode_lr10
#define lwswitch_corelib_set_rx_detect_ls10         lwswitch_corelib_set_rx_detect_lr10
#define lwswitch_corelib_write_discovery_token_ls10 lwswitch_corelib_write_discovery_token_lr10
#define lwswitch_corelib_read_discovery_token_ls10  lwswitch_corelib_read_discovery_token_lr10

#define lwswitch_inforom_lwl_get_minion_data_ls10   lwswitch_inforom_lwl_get_minion_data_lr10
#define lwswitch_inforom_lwl_set_minion_data_ls10   lwswitch_inforom_lwl_set_minion_data_lr10
#define lwswitch_inforom_lwl_get_max_correctable_error_rate_ls10 lwswitch_inforom_lwl_get_max_correctable_error_rate_lr10
#define lwswitch_inforom_lwl_get_errors_ls10        lwswitch_inforom_lwl_get_errors_lr10
#define lwswitch_inforom_ecc_log_error_event_ls10   lwswitch_inforom_ecc_log_error_event_lr10
#define lwswitch_inforom_ecc_get_errors_ls10        lwswitch_inforom_ecc_get_errors_lr10
#define lwswitch_inforom_bbx_get_sxid_ls10          lwswitch_inforom_bbx_get_sxid_lr10

#define lwswitch_init_dlpl_interrupts_ls10          lwswitch_init_dlpl_interrupts_lr10

#define lwswitch_soe_unregister_events_ls10         lwswitch_soe_unregister_events_lr10
#define lwswitch_soe_register_event_callbacks_ls10  lwswitch_soe_register_event_callbacks_lr10

#define lwswitch_setup_link_system_registers_ls10   lwswitch_setup_link_system_registers_lr10

#define lwswitch_minion_get_initoptimize_status_ls10 lwswitch_minion_get_initoptimize_status_lr10

#define lwswitch_poll_sublink_state_ls10             lwswitch_poll_sublink_state_lr10
#define lwswitch_setup_link_loopback_mode_ls10       lwswitch_setup_link_loopback_mode_lr10

#define lwswitch_link_lane_reversed_ls10             lwswitch_link_lane_reversed_lr10
#define lwswitch_store_topology_information_ls10     lwswitch_store_topology_information_lr10
#define lwswitch_request_tl_link_state_ls10          lwswitch_request_tl_link_state_lr10
#define lwswitch_wait_for_tl_request_ready_ls10      lwswitch_wait_for_tl_request_ready_lr10

#define lwswitch_ctrl_i2c_get_port_info_ls10        lwswitch_ctrl_i2c_get_port_info_lr10
#define lwswitch_i2c_set_hw_speed_mode_ls10         lwswitch_i2c_set_hw_speed_mode_lr10
#define lwswitch_ctrl_i2c_indexed_ls10              lwswitch_ctrl_i2c_indexed_lr10
#define lwswitch_i2c_is_device_access_allowed_ls10  lwswitch_i2c_is_device_access_allowed_lr10

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define lwswitch_ctrl_set_port_test_mode_ls10       lwswitch_ctrl_set_port_test_mode_lr10
#define lwswitch_ctrl_jtag_chain_read_ls10          lwswitch_ctrl_jtag_chain_read_lr10
#define lwswitch_ctrl_jtag_chain_write_ls10         lwswitch_ctrl_jtag_chain_write_lr10
#define lwswitch_ctrl_i2c_get_dev_info_ls10         lwswitch_ctrl_i2c_get_dev_info_lr10
#define lwswitch_ctrl_inject_link_error_ls10        lwswitch_ctrl_inject_link_error_lr10
#define lwswitch_ctrl_get_lwlink_caps_ls10          lwswitch_ctrl_get_lwlink_caps_lr10
#define lwswitch_ctrl_get_err_info_ls10             lwswitch_ctrl_get_err_info_lr10
#define lwswitch_ctrl_read_uphy_pad_lane_reg_ls10   lwswitch_ctrl_read_uphy_pad_lane_reg_lr10
#define lwswitch_ctrl_force_thermal_slowdown_ls10   lwswitch_ctrl_force_thermal_slowdown_lr10


LwlStatus lwswitch_ctrl_set_port_test_mode_lr10(lwswitch_device *device, LWSWITCH_SET_PORT_TEST_MODE *p);
LwlStatus lwswitch_ctrl_jtag_chain_read_lr10(lwswitch_device *device, LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain);
LwlStatus lwswitch_ctrl_jtag_chain_write_lr10(lwswitch_device *device, LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain);
LwlStatus lwswitch_ctrl_i2c_get_dev_info_lr10(lwswitch_device *device, LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams);
LwlStatus lwswitch_ctrl_inject_link_error_lr10(lwswitch_device *device, LWSWITCH_INJECT_LINK_ERROR *p);
LwlStatus lwswitch_ctrl_get_lwlink_caps_lr10(lwswitch_device *device, LWSWITCH_GET_LWLINK_CAPS_PARAMS *ret);
LwlStatus lwswitch_ctrl_get_err_info_lr10(lwswitch_device *device, LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS *ret);
LwlStatus lwswitch_ctrl_read_uphy_pad_lane_reg_lr10(lwswitch_device *device, LWSWITCH_CTRL_READ_UPHY_PAD_LANE_REG *p);
LwlStatus lwswitch_ctrl_force_thermal_slowdown_lr10(lwswitch_device *device, LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

LwBool    lwswitch_is_link_valid_lr10(lwswitch_device *device, LwU32 link_id);
LwBool    lwswitch_is_link_in_use_lr10(lwswitch_device *device, LwU32 link_id);

LwlStatus lwswitch_initialize_device_state_lr10(lwswitch_device *device);
LwlStatus lwswitch_deassert_link_reset_lr10(lwswitch_device *device, lwlink_link *link);
void      lwswitch_determine_platform_lr10(lwswitch_device *device);
LwU32     lwswitch_get_swap_clk_default_lr10(lwswitch_device *device);
LwlStatus lwswitch_post_init_device_setup_lr10(lwswitch_device *device);
LwlStatus lwswitch_ctrl_get_bios_info_lr10(lwswitch_device *device, LWSWITCH_GET_BIOS_INFO_PARAMS *p);
void      lwswitch_hw_counter_shutdown_lr10(lwswitch_device *device);
LwU64     lwswitch_hw_counter_read_counter_lr10(lwswitch_device *device);

void      lwswitch_ecc_writeback_task_lr10(lwswitch_device *device);
LwlStatus lwswitch_ctrl_get_routing_id_lr10(lwswitch_device *device, LWSWITCH_GET_ROUTING_ID_PARAMS *params);
LwlStatus lwswitch_ctrl_set_routing_id_valid_lr10(lwswitch_device *device, LWSWITCH_SET_ROUTING_ID_VALID *p);
LwlStatus lwswitch_ctrl_set_routing_id_lr10(lwswitch_device *device, LWSWITCH_SET_ROUTING_ID *p);
LwlStatus lwswitch_ctrl_set_routing_lan_lr10(lwswitch_device *device, LWSWITCH_SET_ROUTING_LAN *p);
LwlStatus lwswitch_ctrl_get_routing_lan_lr10(lwswitch_device *device, LWSWITCH_GET_ROUTING_LAN_PARAMS *params);
LwlStatus lwswitch_ctrl_set_routing_lan_valid_lr10(lwswitch_device *device, LWSWITCH_SET_ROUTING_LAN_VALID *p);
LwlStatus lwswitch_ctrl_set_ingress_request_table_lr10(lwswitch_device *device, LWSWITCH_SET_INGRESS_REQUEST_TABLE *p);
LwlStatus lwswitch_ctrl_get_ingress_request_table_lr10(lwswitch_device *device, LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *params);
LwlStatus lwswitch_ctrl_set_ingress_request_valid_lr10(lwswitch_device *device, LWSWITCH_SET_INGRESS_REQUEST_VALID *p);
LwlStatus lwswitch_ctrl_get_ingress_response_table_lr10(lwswitch_device *device, LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *params);
LwlStatus lwswitch_ctrl_set_ingress_response_table_lr10(lwswitch_device *device, LWSWITCH_SET_INGRESS_RESPONSE_TABLE *p);

LwlStatus lwswitch_ctrl_get_lwlink_status_lr10(lwswitch_device *device, LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret);
LwlStatus lwswitch_ctrl_get_lwlink_status_ls10(lwswitch_device *device, LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret);

LwlStatus lwswitch_ctrl_get_info_lr10(lwswitch_device *device, LWSWITCH_GET_INFO *p);

LwlStatus lwswitch_ctrl_set_switch_port_config_lr10(lwswitch_device *device, LWSWITCH_SET_SWITCH_PORT_CONFIG *p);
LwlStatus lwswitch_reset_and_drain_links_lr10(lwswitch_device *device, LwU64 link_mask);
void      lwswitch_set_fatal_error_lr10(lwswitch_device *device, LwBool device_fatal, LwU32 link_id);
LwlStatus lwswitch_ctrl_get_fom_values_lr10(lwswitch_device *device, LWSWITCH_GET_FOM_VALUES_PARAMS *p);
LwlStatus lwswitch_ctrl_get_throughput_counters_lr10(lwswitch_device *device, LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p);
void      lwswitch_save_lwlink_seed_data_from_minion_to_inforom_lr10(lwswitch_device *device, LwU32 linkId);
void      lwswitch_store_seed_data_from_inforom_to_corelib_lr10(lwswitch_device *device);
LwlStatus lwswitch_read_oob_blacklist_state_lr10(lwswitch_device *device);
LwlStatus lwswitch_write_fabric_state_lr10(lwswitch_device *device);

LwlStatus lwswitch_corelib_add_link_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_remove_link_lr10(lwlink_link *link);
LwlStatus lwswitch_corelib_get_dl_link_mode_lr10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_set_tl_link_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tx_mode_lr10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_rx_mode_lr10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_set_rx_detect_lr10(lwlink_link *link, LwU32 flags);
LwlStatus lwswitch_corelib_write_discovery_token_lr10(lwlink_link *link, LwU64 token);
LwlStatus lwswitch_corelib_read_discovery_token_lr10(lwlink_link *link, LwU64 *token);

LwlStatus lwswitch_inforom_lwl_get_minion_data_lr10(lwswitch_device *device, void *pLwlGeneric, LwU8 linkId, LwU32 *seedData);
LwlStatus lwswitch_inforom_lwl_set_minion_data_lr10(lwswitch_device *device, void *pLwlGeneric, LwU8 linkId, LwU32 *seedData, LwU32 size, LwBool *bDirty);
LwlStatus lwswitch_inforom_lwl_get_max_correctable_error_rate_lr10(lwswitch_device *device, LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params);
LwlStatus lwswitch_inforom_lwl_get_errors_lr10(lwswitch_device *device, LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params);
LwlStatus lwswitch_inforom_ecc_log_error_event_lr10(lwswitch_device *device, INFOROM_ECC_OBJECT *pEccGeneric, INFOROM_LWS_ECC_ERROR_EVENT *err_event);
LwlStatus lwswitch_inforom_ecc_get_errors_lr10(lwswitch_device *device, LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params);
LwlStatus lwswitch_inforom_bbx_get_sxid_lr10(lwswitch_device *device, LWSWITCH_GET_SXIDS_PARAMS *params);

void      lwswitch_init_dlpl_interrupts_lr10(lwlink_link *link);
void      lwswitch_soe_unregister_events_lr10(lwswitch_device *device);
LwlStatus lwswitch_soe_register_event_callbacks_ls10(lwswitch_device *device);

LwlStatus lwswitch_setup_link_system_registers_lr10(lwswitch_device *device);

LwlStatus lwswitch_minion_get_initoptimize_status_lr10(lwswitch_device *device, LwU32 linkId);

LwlStatus lwswitch_poll_sublink_state_lr10(lwswitch_device *device, lwlink_link *link);
void      lwswitch_setup_link_loopback_mode_lr10(lwswitch_device *device, LwU32 linkNumber);

LwBool    lwswitch_link_lane_reversed_lr10(lwswitch_device *device, LwU32 linkId);
void lwswitch_store_topology_information_lr10(lwswitch_device *device, lwlink_link *link);

LwlStatus lwswitch_request_tl_link_state_lr10(lwlink_link *link, LwU32 tlLinkState, LwBool bSync);
LwlStatus lwswitch_wait_for_tl_request_ready_lr10(lwlink_link *link);

LwlStatus lwswitch_parse_bios_image_lr10(lwswitch_device *device);
LwlStatus lwswitch_ctrl_i2c_get_port_info_lr10(lwswitch_device *device, LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams);
void      lwswitch_i2c_set_hw_speed_mode_lr10(lwswitch_device *device, LwU32 port, LwU32 speedMode);
LwlStatus lwswitch_ctrl_i2c_indexed_lr10(lwswitch_device *device, LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams);
LwBool lwswitch_i2c_is_device_access_allowed_lr10(lwswitch_device *device, LwU32 port, LwU8 addr, LwBool bIsRead);

//
// Internal function declarations
//

LwlStatus lwswitch_corelib_set_dl_link_mode_ls10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_set_tx_mode_ls10(lwlink_link *link, LwU64 mode, LwU32 flags);
void lwswitch_init_lpwr_regs_ls10(lwlink_link *link);

LwlStatus lwswitch_minion_service_falcon_interrupts_ls10(lwswitch_device *device, LwU32 instance);

LwlStatus lwswitch_device_discovery_ls10(lwswitch_device *device, LwU32 discovery_offset);
void lwswitch_filter_discovery_ls10(lwswitch_device *device);
LwlStatus lwswitch_process_discovery_ls10(lwswitch_device *device);
void lwswitch_lib_enable_interrupts_ls10(lwswitch_device *device);
void lwswitch_lib_disable_interrupts_ls10(lwswitch_device *device);
LwlStatus lwswitch_lib_service_interrupts_ls10(lwswitch_device *device);
LwlStatus lwswitch_lib_check_interrupts_ls10(lwswitch_device *device);
void      lwswitch_initialize_interrupt_tree_ls10(lwswitch_device *device);
void      lwswitch_corelib_training_complete_ls10(lwlink_link *link);
LwlStatus lwswitch_init_nport_ls10(lwswitch_device *device);
LwlStatus lwswitch_get_soe_ucode_binaries_ls10(lwswitch_device *device, const LwU32 **soe_ucode_data, const LwU32 **soe_ucode_header);
LwlStatus lwswitch_corelib_get_rx_detect_ls10(lwlink_link *link);
void lwswitch_reset_persistent_link_hw_state_ls10(lwswitch_device *device, LwU32 linkNumber);
LwlStatus lwswitch_minion_get_rxdet_status_ls10(lwswitch_device *device, LwU32 linkId);
LwlStatus lwswitch_minion_restore_seed_data_ls10(lwswitch_device *device, LwU32 linkId, LwU32 *seedData);
LwlStatus lwswitch_minion_set_sim_mode_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_minion_set_smf_settings_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_minion_select_uphy_tables_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_set_training_mode_ls10(lwswitch_device *device);
LwlStatus lwswitch_corelib_get_tl_link_mode_ls10(lwlink_link *link, LwU64 *mode);
LwU32     lwswitch_get_sublink_width_ls10(lwswitch_device *device,LwU32 linkNumber);
LwlStatus lwswitch_parse_bios_image_ls10(lwswitch_device *device);
LwBool    lwswitch_is_link_in_reset_ls10(lwswitch_device *device, lwlink_link *link);
void lwswitch_corelib_get_uphy_load_ls10(lwlink_link *link, LwBool *bUnlocked);
LwlStatus lwswitch_ctrl_get_lwlink_lp_counters_ls10(lwswitch_device *device, LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params);
void      lwswitch_init_buffer_ready_ls10(lwswitch_device *device, lwlink_link *link, LwBool bNportBufferReady);
void      lwswitch_apply_recal_settings_ls10(lwswitch_device *device, lwlink_link *link);
LwlStatus lwswitch_corelib_get_dl_link_mode_ls10(lwlink_link *link, LwU64 *mode);
LwlStatus lwswitch_corelib_get_tx_mode_ls10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);
LwlStatus lwswitch_corelib_get_rx_mode_ls10(lwlink_link *link, LwU64 *mode, LwU32 *subMode);

//
// Functions called by LS10 back into LR10 codebase
//
LwlStatus lwswitch_corelib_set_dl_link_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_set_tx_mode_lr10(lwlink_link *link, LwU64 mode, LwU32 flags);
LwlStatus lwswitch_corelib_get_tl_link_mode_lr10(lwlink_link *link, LwU64 *mode);
void      lwswitch_init_buffer_ready_lr10(lwswitch_device *device, lwlink_link *link, LwBool bNportBufferReady);

LwlStatus lwswitch_service_lwldl_fatal_link_ls10(lwswitch_device *device, LwU32 lwliptInstance, LwU32 link);
LwlStatus lwswitch_ctrl_inband_send_data_ls10(lwswitch_device *device, LWSWITCH_INBAND_SEND_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_read_data_ls10(lwswitch_device *device, LWSWITCH_INBAND_READ_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_flush_data_ls10(lwswitch_device *device, LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p);
LwlStatus lwswitch_ctrl_inband_pending_data_stats_ls10(lwswitch_device *device, LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p);
LwlStatus lwswitch_service_minion_link_ls10(lwswitch_device *device, LwU32 lwliptInstance);
void      lwswitch_apply_recal_settings_ls10(lwswitch_device *device, lwlink_link *link);

//
// SU generated functions
//

LwlStatus lwswitch_lws_top_prod_ls10(lwswitch_device *device);
LwlStatus lwswitch_npg_prod_ls10(lwswitch_device *device);
LwlStatus lwswitch_apply_prod_lwlw_ls10(lwswitch_device *device);
LwlStatus lwswitch_apply_prod_nxbar_ls10(lwswitch_device *device);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define lwswitch_fetch_active_repeater_mask_ls10    lwswitch_fetch_active_repeater_mask_lr10
#define lwswitch_get_active_repeater_mask_ls10      lwswitch_get_active_repeater_mask_lr10
void lwswitch_fetch_active_repeater_mask_lr10(lwswitch_device *device);
LwU64 lwswitch_get_active_repeater_mask_lr10(lwswitch_device *device);
LwlStatus lwswitch_is_link_in_repeater_mode_ls10(lwswitch_device *device, LwU32 link_id, LwBool *isRepeaterMode);
LwlStatus lwswitch_get_board_id_lr10(lwswitch_device *device, LwU16 *boardId);
LwlStatus lwswitch_get_board_id_ls10(lwswitch_device *device, LwU16 *boardId);

LwBool    lwswitch_is_cci_supported_ls10(lwswitch_device *device);
LwlStatus lwswitch_corelib_set_optical_infinite_mode_ls10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_enable_optical_maintenance_ls10(lwlink_link *link, LwBool bTx);
LwlStatus lwswitch_corelib_set_optical_iobist_ls10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_set_optical_pretrain_ls10(lwlink_link *link, LwBool bTx, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_pretrain_ls10(lwlink_link *link, LwBool bTx, LwBool *bSuccess);
LwlStatus lwswitch_corelib_init_optical_links_ls10(lwlink_link *link);
LwlStatus lwswitch_corelib_set_optical_force_eq_ls10(lwlink_link *link, LwBool bEnable);
LwlStatus lwswitch_corelib_check_optical_eom_status_ls10(lwlink_link *link, LwBool *bEomLow);

LwlStatus lwswitch_ctrl_set_mc_rid_table_ls10(lwswitch_device *device, LWSWITCH_SET_MC_RID_TABLE_PARAMS *p);
LwlStatus lwswitch_ctrl_get_mc_rid_table_ls10(lwswitch_device *device, LWSWITCH_GET_MC_RID_TABLE_PARAMS *p);

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus lwswitch_launch_ALI_ls10(lwswitch_device *device);
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)

#endif //_LS10_H_

