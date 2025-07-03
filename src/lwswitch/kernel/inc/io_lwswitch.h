/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _IO_LWSWITCH_H_
#define _IO_LWSWITCH_H_

// LWSWITCH_REG_* MMIO wrappers are to be used for absolute symbolic BAR0 offset
// register  references like SMC, CLOCK, BUS, and PRIV_MASTER.
//

#define LWSWITCH_REG_RD32(_d, _dev, _reg)               \
    (                                                   \
        LWSWITCH_PRINT(_d, MMIO,                        \
            "%s: MEM_RD: %s, %s (+%04x)\n",             \
            __FUNCTION__,                               \
            #_dev, #_reg, LW ## _dev ## _reg)           \
    ,                                                   \
        lwswitch_reg_read_32(_d, LW##_dev##_reg)        \
    );                                                  \
    ((void)(_d))

#define LWSWITCH_REG_WR32(_d, _dev, _reg, _data)        \
    LWSWITCH_PRINT(_d, MMIO,                            \
        "%s: MEM_WR: %s, %s (+%04x) 0x%08x\n",          \
        __FUNCTION__,                                   \
        #_dev, #_reg, LW ## _dev ## _reg, _data);       \
    lwswitch_reg_write_32(_d, LW##_dev##_reg, _data);   \
    ((void)(_d))

//
// LWSWITCH_OFF_* MMIO wrappers are used to access a fully formed BAR0 offset.
//

#define LWSWITCH_OFF_RD32(_d, _off)                 \
    lwswitch_reg_read_32(_d, _off);                 \
    ((void)(_d))

#define LWSWITCH_OFF_WR32(_d, _off, _data)          \
    lwswitch_reg_write_32(_d, _off, _data);         \
    ((void)(_d))

#define LWSWITCH_ENGINE_DESCRIPTOR_UC_SIZE      64
#define LWSWITCH_ENGINE_DESCRIPTOR_MC_SIZE      3

#define LWSWITCH_ENGINE_INSTANCE_ILWALID        ((LwU32) (~0))

typedef struct engine_descriptor
{
    const char *eng_name;
    LwU32 eng_id;           // REGISTER_RW_ENGINE_*
    LwU32 eng_count;
    LwU32 uc_addr[LWSWITCH_ENGINE_DESCRIPTOR_UC_SIZE];
    LwU32 bc_addr;
    LwU32 mc_addr[LWSWITCH_ENGINE_DESCRIPTOR_MC_SIZE];
    LwU32 mc_addr_count;
} LWSWITCH_ENGINE_DESCRIPTOR_TYPE;

#define LWSWITCH_DECLARE_IO_DESCRIPTOR(_engine, _bcast)    \
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE     _engine;

#define LWSWITCH_BASE_ADDR_ILWALID          ((LwU32) (~0))

//
// All IP-based (0-based register manuals) engines that ever existed on *ANY*
// architecture(s) must be listed here in order to use the common IO wrappers.
// New engines need to be added here as well as in the chip-specific lists in
// their respective headers that generate chip-specific handlers.
// Absolute BAR0 offset-based units are legacy units in which the unit's offset
// in BAR0 is included in the register definition in the manuals.  For these
// legacy units the discovered base is not used since it is already part of the
// register.  Legacy units (e.g. PSMC, CLOCK, BUS, and PRIV_MASTER) should use
// LWSWITCH_REG_RD/WR IO wrappers.
//

#define LWSWITCH_LIST_ALL_ENGINES(_op)      \
    _op(XVE)                                \
    _op(SAW)                                \
    _op(SOE)                                \
    _op(SMR)                                \
    _op(GIN)                                \
    _op(XAL)                                \
    _op(XAL_FUNC)                           \
    _op(XPL)                                \
    _op(XTL)                                \
    _op(XTL_CONFIG)                         \
    _op(UXL)                                \
    _op(GPU_PTOP)                           \
    _op(PMC)                                \
    _op(PBUS)                               \
    _op(ROM2)                               \
    _op(GPIO)                               \
    _op(FSP)                                \
    _op(SYSCTRL)                            \
    _op(CLKS_SYS)                           \
    _op(CLKS_SYSB)                          \
    _op(CLKS_P0)                            \
    _op(SAW_PM)                             \
    _op(PCIE_PM)                            \
    _op(PRT_PRI_HUB)                        \
    _op(PRT_PRI_RS_CTRL)                    \
    _op(SYS_PRI_HUB)                        \
    _op(SYS_PRI_RS_CTRL)                    \
    _op(SYSB_PRI_HUB)                       \
    _op(SYSB_PRI_RS_CTRL)                   \
    _op(PRI_MASTER_RS)                      \
    _op(PTIMER)                             \
                                            \
    _op(NPG)                                \
    _op(NPORT)                              \
                                            \
    _op(LWLW)                               \
    _op(MINION)                             \
    _op(LWLIPT)                             \
    _op(LWLIPT_LNK)                         \
    _op(LWLTLC)                             \
    _op(LWLDL)                              \
    _op(CPR)                                \
                                            \
    _op(NXBAR)                              \
    _op(TILE)                               \
    _op(TILEOUT)                            \
                                            \
    _op(NPG_PERFMON)                        \
    _op(NPORT_PERFMON)                      \
                                            \
    _op(LWLW_PERFMON)                       \
    _op(RX_PERFMON)                         \
    _op(TX_PERFMON)                         \
                                            \
    _op(NXBAR_PERFMON)                      \
    _op(TILE_PERFMON)                       \
    _op(TILEOUT_PERFMON)                    \

#define ENGINE_ID_LIST(_eng)                \
    LWSWITCH_ENGINE_ID_##_eng,

//
// ENGINE_IDs are the complete list of all engines that are supported on
// *ANY* architecture(s) that may support them.  Any one architecture may or
// may not understand how to operate on any one specific engine.
// Architectures that share a common ENGINE_ID are not guaranteed to have
// compatible manuals.
//
typedef enum lwswitch_engine_id
{
    LWSWITCH_LIST_ALL_ENGINES(ENGINE_ID_LIST)
    LWSWITCH_ENGINE_ID_SIZE,
} LWSWITCH_ENGINE_ID;

//
// LWSWITCH_ENG_* MMIO wrappers are to be used for top level discovered
// devices like SAW, FUSE, PMGR, XVE, etc.
//

#define LWSWITCH_GET_ENG_DESC_TYPE              0
#define LWSWITCH_GET_ENG_DESC_TYPE_UNICAST      LWSWITCH_GET_ENG_DESC_TYPE
#define LWSWITCH_GET_ENG_DESC_TYPE_BCAST        1
#define LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST    2

#define LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx)                 \
    ((_d)->hal.lwswitch_get_eng_base(                               \
        _d,                                                         \
        LWSWITCH_ENGINE_ID_##_eng,                                  \
        LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                         \
        _engidx))

#define LWSWITCH_ENG_COUNT(_d, _eng, _bcast)                        \
    ((_d)->hal.lwswitch_get_eng_count(                              \
        _d,                                                         \
        LWSWITCH_ENGINE_ID_##_eng,                                  \
        LWSWITCH_GET_ENG_DESC_TYPE##_bcast))

#define LWSWITCH_ENG_IS_VALID(_d, _eng, _engidx)                    \
    (                                                               \
        LWSWITCH_GET_ENG(_d, _eng, , _engidx) != LWSWITCH_BASE_ADDR_ILWALID \
    )

#define LWSWITCH_ENG_WR32(_d, _eng, _bcast, _engidx, _dev, _reg, _data) \
    {                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_WR %s[%d]: %s, %s (%06x+%04x) 0x%08x\n",       \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            #_dev, #_reg,                                           \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            LW ## _dev ## _reg, _data);                             \
                                                                    \
        ((_d)->hal.lwswitch_eng_wr(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            LW ## _dev ## _reg, _data));                            \
    }

#define LWSWITCH_ENG_RD32(_d, _eng, _bcast, _engidx, _dev, _reg)    \
    (                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_RD %s[%d]: %s, %s (%06x+%04x)\n",              \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            #_dev, #_reg,                                           \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            LW ## _dev ## _reg)                                     \
    ,                                                               \
        ((_d)->hal.lwswitch_eng_rd(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            LW ## _dev ## _reg))                                    \
    );                                                              \
    ((void)(_d))

#define LWSWITCH_ENG_WR32_IDX(_d, _eng, _bcast, _engidx, _dev, _reg, _idx, _data) \
    {                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_WR %s[%d]: %s, %s(%d) (%06x+%04x) 0x%08x\n",   \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            #_dev, #_reg, _idx,                                     \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            LW ## _dev ## _reg(_idx), _data);                       \
                                                                    \
        ((_d)->hal.lwswitch_eng_wr(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            LW ## _dev ## _reg(_idx), _data));                      \
    }

#define LWSWITCH_ENG_RD32_IDX(_d, _eng, _bcast, _engidx, _dev, _reg, _idx)  \
    (                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_RD %s[%d]: %s, %s(%d) (%06x+%04x)\n",          \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            #_dev, #_reg, _idx,                                     \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            LW ## _dev ## _reg(_idx))                               \
    ,                                                               \
        ((_d)->hal.lwswitch_eng_rd(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            LW ## _dev ## _reg(_idx)))                              \
    );                                                              \
    ((void)(_d))

#define LWSWITCH_ENG_OFF_WR32(_d, _eng, _bcast, _engidx, _offset, _data) \
    {                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_WR %s[%d]: 0x%x (%06x+%04x) 0x%08x\n",         \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            _offset,                                                \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            _offset, _data);                                        \
        ((_d)->hal.lwswitch_eng_wr(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            _offset, _data));                                       \
    }

#define LWSWITCH_ENG_OFF_RD32(_d, _eng, _bcast, _engidx, _offset)   \
    (                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: MEM_RD %s[%d]: 0x%x (%06x+%04x)\n",                \
            __FUNCTION__,                                           \
            #_eng#_bcast, _engidx,                                  \
            _offset,                                                \
            LWSWITCH_GET_ENG(_d, _eng, _bcast, _engidx),            \
            _offset)                                                \
    ,                                                               \
        ((_d)->hal.lwswitch_eng_rd(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE##_bcast,                     \
            _engidx,                                                \
            _offset))                                               \
    )

//
// Per-link information
//

#define LWSWITCH_MAX_LINK_COUNT             64

#define LWSWITCH_MAX_SEED_BUFFER_SIZE         LWSWITCH_MAX_SEED_NUM + 1

#define LWSWITCH_MAX_INBAND_BUFFER_SIZE       256*8
#define LWSWITCH_MAX_INBAND_BITS_SENT_AT_ONCE 32
#define LWSWITCH_MAX_INBAND_BUFFER_ENTRIES    LWSWITCH_MAX_INBAND_BUFFER_SIZE/LWSWITCH_MAX_INBAND_BITS_SENT_AT_ONCE

//
// Inband data structure
//
struct lwswitch_inband_data
{
    // Inband bufer at sender Minion
    LwU32  sendBuffer[LWSWITCH_MAX_INBAND_BUFFER_ENTRIES];

    // Inband buffer at receiver Minion
    LwU32  receiveBuffer[LWSWITCH_MAX_INBAND_BUFFER_ENTRIES];

    // Is the current Minion a sender or receiver of Inband Data?
    LwBool bIsSenderMinion;

    // Bool to say fail or not
    LwBool bTransferFail;

    // # of transmisions done - count
    // LwU32 txCount;
};

typedef struct
{
    LwBool valid;
    LwU32  link_clock_khz;

    LwBool fatal_error_oclwrred;
    LwBool ingress_packet_latched;
    LwBool egress_packet_latched;

    LwBool nea;    // Near end analog
    LwBool ned;    // Near end digital

    LwU32  lane_rxdet_status_mask;

    LwBool bIsRepeaterMode;
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LwBool bActiveRepeaterPresent;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    // Minion Inband Data structure
    struct lwswitch_inband_data inBandData;

} LWSWITCH_LINK_TYPE;

//
// Per link register access routines
// LINK_* MMIO wrappers are used to reference per-link engine instances
//

#define LWSWITCH_LINK_COUNT(_d)                                         \
    (lwswitch_get_num_links(_d))

#define LWSWITCH_GET_LINK_ENG_INST(_d, _linknum, _eng)                  \
    lwswitch_get_link_eng_inst(_d, _linknum, LWSWITCH_ENGINE_ID_##_eng)

#define LWSWITCH_IS_LINK_ENG_VALID(_d, _linknum, _eng)                  \
    (                                                                   \
        (LWSWITCH_GET_ENG(_d, _eng, ,                                   \
            LWSWITCH_GET_LINK_ENG_INST(_d, _linknum, _eng))             \
            != LWSWITCH_BASE_ADDR_ILWALID) &&                           \
        lwswitch_is_link_valid(_d, _linknum)                            \
    )

#define LWSWITCH_LINK_OFFSET(_d, _physlinknum, _eng, _dev, _reg)        \
    (                                                                   \
        LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID(_d, _physlinknum, _eng)) \
        ,                                                               \
        LWSWITCH_PRINT(_d, MMIO,                                        \
            "%s: LINK_OFFSET link[%d] %s: %s,%s (+%04x)\n",             \
            __FUNCTION__,                                               \
            _physlinknum,                                               \
            #_eng, #_dev, #_reg, LW ## _dev ## _reg)                    \
        ,                                                               \
        LWSWITCH_GET_ENG(_d, _eng, ,                                    \
            LWSWITCH_GET_LINK_ENG_INST(_d, _physlinknum, _eng)) +       \
            LW##_dev##_reg                                              \
    )

#define LWSWITCH_LINK_WR32(_d, _physlinknum, _eng, _dev, _reg, _data)   \
    LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID(_d, _physlinknum, _eng)); \
    LWSWITCH_PRINT(_d, MMIO,                                            \
        "%s: LINK_WR link[%d] %s: %s,%s (+%04x) 0x%08x\n",              \
        __FUNCTION__,                                                   \
        _physlinknum,                                                   \
        #_eng, #_dev, #_reg, LW ## _dev ## _reg, _data);                \
    ((_d)->hal.lwswitch_eng_wr(                                         \
            _d,                                                         \
            LWSWITCH_ENGINE_ID_##_eng,                                  \
            LWSWITCH_GET_ENG_DESC_TYPE_UNICAST,                         \
            LWSWITCH_GET_LINK_ENG_INST(_d, _physlinknum, _eng),         \
            LW ## _dev ## _reg, _data));                                \
    ((void)(_d))

#define LWSWITCH_LINK_RD32(_d, _physlinknum, _eng, _dev, _reg)      \
    (                                                               \
        LWSWITCH_ASSERT(LWSWITCH_IS_LINK_ENG_VALID(_d, _physlinknum, _eng)) \
        ,                                                           \
        LWSWITCH_PRINT(_d, MMIO,                                    \
            "%s: LINK_RD link[%d] %s: %s,%s (+%04x)\n",             \
            __FUNCTION__,                                           \
            _physlinknum,                                           \
            #_eng, #_dev, #_reg, LW ## _dev ## _reg)                \
        ,                                                           \
        ((_d)->hal.lwswitch_eng_rd(                                 \
            _d,                                                     \
            LWSWITCH_ENGINE_ID_##_eng,                              \
            LWSWITCH_GET_ENG_DESC_TYPE_UNICAST,                     \
            LWSWITCH_GET_LINK_ENG_INST(_d, _physlinknum, _eng),     \
            LW ## _dev ## _reg))                                    \
    );                                                              \
    ((void)(_d))

#define LWSWITCH_LINK_WR32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx, _data)    \
    LWSWITCH_LINK_WR32(_d, _physlinknum, _eng, _dev, _reg(_idx), _data);           \
    ((void)(_d))

#define LWSWITCH_LINK_RD32_IDX(_d, _physlinknum, _eng, _dev, _reg, _idx)   \
    LWSWITCH_LINK_RD32(_d, _physlinknum, _eng, _dev, _reg(_idx));          \
    ((void)(_d))

#endif //_IO_LWSWITCH_H_
