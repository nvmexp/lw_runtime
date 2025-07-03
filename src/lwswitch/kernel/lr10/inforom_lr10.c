/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/inforom_lr10.h"
#include "inforom/ifrstruct.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"
#include "lwswitch/lr10/dev_lwlsaw_ip_addendum.h"
#include "lwswitch/lr10/dev_pmgr.h"

//
// TODO: Split individual object hals to their own respective files
//
static void _oms_parse(lwswitch_device *device, INFOROM_OMS_STATE *pOmsState);
static void _oms_refresh(lwswitch_device *device, INFOROM_OMS_STATE *pOmsState);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
LwlStatus
lwswitch_inforom_lwl_get_minion_data_lr10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData
)
{
    INFOROM_LWL_OBJECT_V3S *pLwlObject = &((PINFOROM_LWL_OBJECT)pLwlGeneric)->v3s;

    if (linkId >= INFOROM_LWL_OBJECT_V3S_NUM_LINKS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object does not store data for more than %u links (link %u requested)\n",
            INFOROM_LWL_OBJECT_V3S_NUM_LINKS, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_os_memcpy(seedData, pLwlObject->minionData[linkId].data,
                INFOROM_LWL_OBJECT_V3G_MAX_SEED_BUFFER_SIZE * sizeof(seedData[0]));

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_lwl_set_minion_data_lr10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData,
    LwU32                size,
    LwBool              *bDirty
)
{
    LwU64 time_ns;
    LwBool bChanged = LW_FALSE;
    INFOROM_LWL_OBJECT_V3S *pLwlObject = &((PINFOROM_LWL_OBJECT)pLwlGeneric)->v3s;

    if (linkId >= INFOROM_LWL_OBJECT_V3S_NUM_LINKS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object does not store data for more than %u links (link %u requested)\n",
            INFOROM_LWL_OBJECT_V3S_NUM_LINKS, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (size > (INFOROM_LWL_OBJECT_V3G_MAX_SEED_BUFFER_SIZE - 1))
    {
        LWSWITCH_PRINT(device, ERROR,
                    "minion data size is larger than expected: %d\n",
                    size);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (size != pLwlObject->minionData[linkId].data[0])
    {
        bChanged = LW_TRUE;
    }
    else
    {
        if (lwswitch_os_memcmp(&pLwlObject->minionData[linkId].data[1], seedData,
                            size * sizeof(seedData[0])))
        {
            bChanged = LW_TRUE;
        }
    }

    if (bChanged)
    {
        time_ns = lwswitch_os_get_platform_time();
        pLwlObject->minionData[linkId].lastUpdated = (LwU32)(time_ns / LWSWITCH_INTERVAL_1SEC_IN_NS);
        pLwlObject->minionData[linkId].data[0] = size;

        lwswitch_os_memcpy(&pLwlObject->minionData[linkId].data[1], seedData,
                        size * sizeof(seedData[0]));
    }

    *bDirty = bChanged;

    return LWL_SUCCESS;
}

#define LUT_ELEMENT(block, dir, subtype, type, sev)                                 \
    { INFOROM_LWL_ERROR_TYPE ## type,                                               \
      FLD_SET_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _SEVERITY,  sev, 0) |    \
          FLD_SET_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, dir, 0), \
      block ## dir ## subtype ## type,                                              \
      INFOROM_LWL_ERROR_BLOCK_TYPE_ ## block                                        \
    }

static LwlStatus _inforom_lwl_v3_map_error
(
    INFOROM_LWLINK_ERROR_TYPES error,
    LwU8  *pHeader,
    LwU16 *pMetadata,
    LwU8  *pErrorSubtype,
    INFOROM_LWL_ERROR_BLOCK_TYPE *pBlockType
)
{
    static const struct
    { LwU8  header;
      LwU16 metadata;
      LwU8  errorSubtype;
      INFOROM_LWL_ERROR_BLOCK_TYPE blockType;
    } lut[] =
    {
        LUT_ELEMENT(DL,     _RX, _FAULT_DL_PROTOCOL_FATAL,              _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _RX, _FAULT_SUBLINK_CHANGE_FATAL,           _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _RX, _FLIT_CRC_CORR,                        _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _RX, _LANE0_CRC_CORR,                       _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _RX, _LANE1_CRC_CORR,                       _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _RX, _LANE2_CRC_CORR,                       _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _RX, _LANE3_CRC_CORR,                       _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _RX, _LINK_REPLAY_EVENTS_CORR,              _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _TX, _FAULT_RAM_FATAL,                      _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _TX, _FAULT_INTERFACE_FATAL,                _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _TX, _FAULT_SUBLINK_CHANGE_FATAL,           _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _TX, _LINK_REPLAY_EVENTS_CORR,              _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(DL,     _NA, _LTSSM_FAULT_UP_FATAL,                 _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _NA, _LTSSM_FAULT_DOWN_FATAL,               _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _NA, _LINK_RECOVERY_EVENTS_CORR,            _ACLWM, _CORRECTABLE),
        LUT_ELEMENT(TLC,    _RX, _DL_HDR_PARITY_ERR_FATAL,              _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DL_DATA_PARITY_ERR_FATAL,             _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DL_CTRL_PARITY_ERR_FATAL,             _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_AE_FATAL,                     _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_BE_FATAL,                     _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_ADDR_ALIGN_FATAL,             _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _PKTLEN_ERR_FATAL,                     _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSVD_PACKET_STATUS_ERR_FATAL,         _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSVD_CACHE_ATTR_PROBE_REQ_ERR_FATAL,  _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSVD_CACHE_ATTR_PROBE_RSP_ERR_FATAL,  _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DATLEN_GT_RMW_REQ_MAX_ERR_FATAL,      _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DATLEN_LT_ATR_RSP_MIN_ERR_FATAL,      _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_CR_FATAL,                     _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_COLLAPSED_RESPONSE_FATAL,     _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _HDR_OVERFLOW_FATAL,                   _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DATA_OVERFLOW_FATAL,                  _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _STOMP_DETECTED_FATAL,                 _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSVD_CMD_ENC_FATAL,                   _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSVD_DAT_LEN_ENC_FATAL,               _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _ILWALID_PO_FOR_CACHE_ATTR_FATAL,      _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _RSP_STATUS_HW_ERR_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _RX, _RSP_STATUS_UR_ERR_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _RX, _RSP_STATUS_PRIV_ERR_NONFATAL,         _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _RX, _POISON_NONFATAL,                      _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _RX, _AN1_HEARTBEAT_TIMEOUT_NONFATAL,       _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _RX, _ILLEGAL_PRI_WRITE_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _DL_CREDIT_PARITY_ERR_FATAL,           _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _TX, _NCISOC_HDR_ECC_DBE_FATAL,             _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _TX, _NCISOC_PARITY_ERR_FATAL,              _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _TX, _ILLEGAL_PRI_WRITE_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC0_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC1_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC2_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC3_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC4_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC5_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC6_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _AN1_TIMEOUT_VC7_NONFATAL,             _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _POISON_NONFATAL,                      _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _RSP_STATUS_HW_ERR_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _RSP_STATUS_UR_ERR_NONFATAL,           _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _RSP_STATUS_PRIV_ERR_NONFATAL,         _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _SLEEP_WHILE_ACTIVE_LINK_FATAL,        _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _RSTSEQ_PHYCTL_TIMEOUT_FATAL,          _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _RSTSEQ_CLKCTL_TIMEOUT_FATAL,          _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _CLKCTL_ILLEGAL_REQUEST_FATAL,         _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _RSTSEQ_PLL_TIMEOUT_FATAL,             _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _RSTSEQ_PHYARB_TIMEOUT_FATAL,          _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(LWLIPT, _NA, _ILLEGAL_LINK_STATE_REQUEST_NONFATAL,  _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _FAILED_MINION_REQUEST_NONFATAL,       _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _RESERVED_REQUEST_VALUE_NONFATAL,      _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _LINK_STATE_WRITE_WHILE_BUSY_NONFATAL, _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL,  _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(LWLIPT, _NA, _LINK_STATE_REQUEST_TIMEOUT_NONFATAL,  _COUNT, _UNCORRECTABLE_NONFATAL),
        // TODO 3014908 log these in the LWL object until we have ECC object support
        LUT_ELEMENT(TLC,    _RX, _HDR_RAM_ECC_DBE_FATAL,                _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DAT0_RAM_ECC_DBE_FATAL,               _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _RX, _DAT1_RAM_ECC_DBE_FATAL,               _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(TLC,    _TX, _CREQ_DAT_RAM_ECC_DBE_NONFATAL,        _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _RSP_DAT_RAM_ECC_DBE_NONFATAL,         _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _COM_DAT_RAM_ECC_DBE_NONFATAL,         _COUNT, _UNCORRECTABLE_NONFATAL),
        LUT_ELEMENT(TLC,    _TX, _RSP1_DAT_RAM_ECC_DBE_FATAL,           _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _NA, _PHY_A_FATAL,                          _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _RX, _CRC_COUNTER_FATAL,                    _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _TX, _PL_ERROR_FATAL,                       _COUNT, _UNCORRECTABLE_FATAL),
        LUT_ELEMENT(DL,     _RX, _PL_ERROR_FATAL,                       _COUNT, _UNCORRECTABLE_FATAL)
    };

    ct_assert(INFOROM_LWLINK_DL_RX_FAULT_DL_PROTOCOL_FATAL == 0);
    ct_assert(INFOROM_LWLINK_DL_RX_FAULT_SUBLINK_CHANGE_FATAL == 1);
    ct_assert(INFOROM_LWLINK_DL_RX_FLIT_CRC_CORR == 2);
    ct_assert(INFOROM_LWLINK_DL_RX_LANE0_CRC_CORR == 3);
    ct_assert(INFOROM_LWLINK_DL_RX_LANE1_CRC_CORR == 4);
    ct_assert(INFOROM_LWLINK_DL_RX_LANE2_CRC_CORR == 5);
    ct_assert(INFOROM_LWLINK_DL_RX_LANE3_CRC_CORR == 6);
    ct_assert(INFOROM_LWLINK_DL_RX_LINK_REPLAY_EVENTS_CORR == 7);
    ct_assert(INFOROM_LWLINK_DL_TX_FAULT_RAM_FATAL == 8);
    ct_assert(INFOROM_LWLINK_DL_TX_FAULT_INTERFACE_FATAL == 9);
    ct_assert(INFOROM_LWLINK_DL_TX_FAULT_SUBLINK_CHANGE_FATAL == 10);
    ct_assert(INFOROM_LWLINK_DL_TX_LINK_REPLAY_EVENTS_CORR == 11);
    ct_assert(INFOROM_LWLINK_DL_LTSSM_FAULT_UP_FATAL == 12);
    ct_assert(INFOROM_LWLINK_DL_LTSSM_FAULT_DOWN_FATAL == 13);
    ct_assert(INFOROM_LWLINK_DL_LINK_RECOVERY_EVENTS_CORR == 14);
    ct_assert(INFOROM_LWLINK_TLC_RX_DL_HDR_PARITY_ERR_FATAL == 15);
    ct_assert(INFOROM_LWLINK_TLC_RX_DL_DATA_PARITY_ERR_FATAL == 16);
    ct_assert(INFOROM_LWLINK_TLC_RX_DL_CTRL_PARITY_ERR_FATAL == 17);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_AE_FATAL == 18);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_BE_FATAL == 19);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_ADDR_ALIGN_FATAL == 20);
    ct_assert(INFOROM_LWLINK_TLC_RX_PKTLEN_ERR_FATAL == 21);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSVD_PACKET_STATUS_ERR_FATAL == 22);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSVD_CACHE_ATTR_PROBE_REQ_ERR_FATAL == 23);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSVD_CACHE_ATTR_PROBE_RSP_ERR_FATAL == 24);
    ct_assert(INFOROM_LWLINK_TLC_RX_DATLEN_GT_RMW_REQ_MAX_ERR_FATAL == 25);
    ct_assert(INFOROM_LWLINK_TLC_RX_DATLEN_LT_ATR_RSP_MIN_ERR_FATAL == 26);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_CR_FATAL == 27);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_COLLAPSED_RESPONSE_FATAL == 28);
    ct_assert(INFOROM_LWLINK_TLC_RX_HDR_OVERFLOW_FATAL == 29);
    ct_assert(INFOROM_LWLINK_TLC_RX_DATA_OVERFLOW_FATAL == 30);
    ct_assert(INFOROM_LWLINK_TLC_RX_STOMP_DETECTED_FATAL == 31);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSVD_CMD_ENC_FATAL == 32);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSVD_DAT_LEN_ENC_FATAL == 33);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR_FATAL == 34);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSP_STATUS_HW_ERR_NONFATAL == 35);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSP_STATUS_UR_ERR_NONFATAL == 36);
    ct_assert(INFOROM_LWLINK_TLC_RX_RSP_STATUS_PRIV_ERR_NONFATAL == 37);
    ct_assert(INFOROM_LWLINK_TLC_RX_POISON_NONFATAL == 38);
    ct_assert(INFOROM_LWLINK_TLC_RX_AN1_HEARTBEAT_TIMEOUT_NONFATAL == 39);
    ct_assert(INFOROM_LWLINK_TLC_RX_ILLEGAL_PRI_WRITE_NONFATAL == 40);
    ct_assert(INFOROM_LWLINK_TLC_TX_DL_CREDIT_PARITY_ERR_FATAL == 41);
    ct_assert(INFOROM_LWLINK_TLC_TX_NCISOC_HDR_ECC_DBE_FATAL == 42);
    ct_assert(INFOROM_LWLINK_TLC_TX_NCISOC_PARITY_ERR_FATAL == 43);
    ct_assert(INFOROM_LWLINK_TLC_TX_ILLEGAL_PRI_WRITE_NONFATAL == 44);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC0_NONFATAL == 45);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC1_NONFATAL == 46);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC2_NONFATAL == 47);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC3_NONFATAL == 48);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC4_NONFATAL == 49);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC5_NONFATAL == 50);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC6_NONFATAL == 51);
    ct_assert(INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC7_NONFATAL == 52);
    ct_assert(INFOROM_LWLINK_TLC_TX_POISON_NONFATAL == 53);
    ct_assert(INFOROM_LWLINK_TLC_TX_RSP_STATUS_HW_ERR_NONFATAL == 54);
    ct_assert(INFOROM_LWLINK_TLC_TX_RSP_STATUS_UR_ERR_NONFATAL == 55);
    ct_assert(INFOROM_LWLINK_TLC_TX_RSP_STATUS_PRIV_ERR_NONFATAL == 56);
    ct_assert(INFOROM_LWLINK_LWLIPT_SLEEP_WHILE_ACTIVE_LINK_FATAL == 57);
    ct_assert(INFOROM_LWLINK_LWLIPT_RSTSEQ_PHYCTL_TIMEOUT_FATAL == 58);
    ct_assert(INFOROM_LWLINK_LWLIPT_RSTSEQ_CLKCTL_TIMEOUT_FATAL == 59);
    ct_assert(INFOROM_LWLINK_LWLIPT_CLKCTL_ILLEGAL_REQUEST_FATAL == 60);
    ct_assert(INFOROM_LWLINK_LWLIPT_RSTSEQ_PLL_TIMEOUT_FATAL == 61);
    ct_assert(INFOROM_LWLINK_LWLIPT_RSTSEQ_PHYARB_TIMEOUT_FATAL == 62);
    ct_assert(INFOROM_LWLINK_LWLIPT_ILLEGAL_LINK_STATE_REQUEST_NONFATAL == 63);
    ct_assert(INFOROM_LWLINK_LWLIPT_FAILED_MINION_REQUEST_NONFATAL == 64);
    ct_assert(INFOROM_LWLINK_LWLIPT_RESERVED_REQUEST_VALUE_NONFATAL == 65);
    ct_assert(INFOROM_LWLINK_LWLIPT_LINK_STATE_WRITE_WHILE_BUSY_NONFATAL == 66);
    ct_assert(INFOROM_LWLINK_LWLIPT_WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL == 67);
    ct_assert(INFOROM_LWLINK_LWLIPT_LINK_STATE_REQUEST_TIMEOUT_NONFATAL == 68);
    ct_assert(INFOROM_LWLINK_TLC_RX_HDR_RAM_ECC_DBE_FATAL == 69);
    ct_assert(INFOROM_LWLINK_TLC_RX_DAT0_RAM_ECC_DBE_FATAL == 70);
    ct_assert(INFOROM_LWLINK_TLC_RX_DAT1_RAM_ECC_DBE_FATAL == 71);
    ct_assert(INFOROM_LWLINK_TLC_TX_CREQ_DAT_RAM_ECC_DBE_NONFATAL == 72);
    ct_assert(INFOROM_LWLINK_TLC_TX_RSP_DAT_RAM_ECC_DBE_NONFATAL == 73);
    ct_assert(INFOROM_LWLINK_TLC_TX_COM_DAT_RAM_ECC_DBE_NONFATAL == 74);
    ct_assert(INFOROM_LWLINK_TLC_TX_RSP1_DAT_RAM_ECC_DBE_FATAL == 75);
    ct_assert(INFOROM_LWLINK_DL_PHY_A_FATAL == 76);
    ct_assert(INFOROM_LWLINK_DL_RX_CRC_COUNTER_FATAL == 77);
    ct_assert(INFOROM_LWLINK_DL_TX_PL_ERROR_FATAL == 78);
    ct_assert(INFOROM_LWLINK_DL_RX_PL_ERROR_FATAL == 79);

    ct_assert(LW_ARRAY_ELEMENTS(lut) == INFOROM_LWLINK_MAX_ERROR_TYPE);

    if (error >= LW_ARRAY_ELEMENTS(lut))
    {
        return -LWL_BAD_ARGS;
    }

    *pHeader       = lut[error].header;
    *pMetadata     = lut[error].metadata;
    *pErrorSubtype = lut[error].errorSubtype;
    *pBlockType    = lut[error].blockType;
    return LWL_SUCCESS;
}

static LwlStatus
_encode_lwlipt_error_subtype
(
    LwU8 localLinkIdx,
    LwU8 *pSubtype
)
{
    static const LwBool linkIdxValidLut[] =
    {
        LW_TRUE,
        LW_TRUE,
        LW_TRUE,
        LW_FALSE,
        LW_FALSE,
        LW_FALSE,
        LW_TRUE,
        LW_TRUE,
        LW_TRUE,
        LW_TRUE,
        LW_TRUE,
        LW_TRUE
    };

    ct_assert(LWLIPT_NA_SLEEP_WHILE_ACTIVE_LINK_FATAL_COUNT == 0);
    ct_assert(LWLIPT_NA_RSTSEQ_PHYCTL_TIMEOUT_FATAL_COUNT == 1);
    ct_assert(LWLIPT_NA_RSTSEQ_CLKCTL_TIMEOUT_FATAL_COUNT == 2);
    ct_assert(LWLIPT_NA_CLKCTL_ILLEGAL_REQUEST_FATAL_COUNT == 3);
    ct_assert(LWLIPT_NA_RSTSEQ_PLL_TIMEOUT_FATAL_COUNT == 4);
    ct_assert(LWLIPT_NA_RSTSEQ_PHYARB_TIMEOUT_FATAL_COUNT == 5);
    ct_assert(LWLIPT_NA_ILLEGAL_LINK_STATE_REQUEST_NONFATAL_COUNT == 6);
    ct_assert(LWLIPT_NA_FAILED_MINION_REQUEST_NONFATAL_COUNT == 7);
    ct_assert(LWLIPT_NA_RESERVED_REQUEST_VALUE_NONFATAL_COUNT == 8);
    ct_assert(LWLIPT_NA_LINK_STATE_WRITE_WHILE_BUSY_NONFATAL_COUNT == 9);
    ct_assert(LWLIPT_NA_WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL_COUNT == 10);
    ct_assert(LWLIPT_NA_LINK_STATE_REQUEST_TIMEOUT_NONFATAL_COUNT == 11);

    if ((localLinkIdx >= LW_INFOROM_LWL_OBJECT_V3_LWLIPT_ERROR_LINK_ID_COMMON) ||
            (*pSubtype >= LW_ARRAY_ELEMENTS(linkIdxValidLut)))
    {
        return -LWL_BAD_ARGS;
    }

    if (linkIdxValidLut[*pSubtype])
    {
        *pSubtype = FLD_SET_DRF_NUM(_INFOROM_LWL_OBJECT_V3, _LWLIPT_ERROR,
                                    _LINK_ID, localLinkIdx, *pSubtype);
    }
    else
    {
        *pSubtype = FLD_SET_DRF(_INFOROM_LWL_OBJECT_V3, _LWLIPT_ERROR, _LINK_ID,
                                _COMMON, *pSubtype);
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_inforom_lwl_log_error_event_lr10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pLwlErrorEvent,
    LwBool *bDirty
)
{
    LwlStatus status;
    INFOROM_LWL_OBJECT_V3S *pLwlObject = &((PINFOROM_LWL_OBJECT)pLwlGeneric)->v3s;
    INFOROM_LWLINK_ERROR_EVENT *pErrorEvent = (INFOROM_LWLINK_ERROR_EVENT *)pLwlErrorEvent;
    INFOROM_LWL_OBJECT_V3_ERROR_ENTRY *pErrorEntry;
    LwU32 i;
    LwU32 sec;
    LwU8  header = 0;
    LwU16 metadata = 0;
    LwU8  errorSubtype;
    LwU64 aclwmTotalCount;
    INFOROM_LWL_ERROR_BLOCK_TYPE blockType;

    if (pErrorEvent->lwliptInstance > INFOROM_LWL_OBJECT_V3_LWLIPT_INSTANCE_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object cannot log data for more than %u LWLIPTs (LWLIPT = %u requested)\n",
            INFOROM_LWL_OBJECT_V3_LWLIPT_INSTANCE_MAX, pErrorEvent->lwliptInstance);
        return -LWL_BAD_ARGS;
    }

    if (pErrorEvent->localLinkIdx > INFOROM_LWL_OBJECT_V3_BLOCK_ID_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object cannot log data for more than %u internal links (internal link = %u requested)\n",
            INFOROM_LWL_OBJECT_V3_BLOCK_ID_MAX, pErrorEvent->localLinkIdx);
        return -LWL_BAD_ARGS;
    }

    sec = (LwU32) (lwswitch_os_get_platform_time() / LWSWITCH_INTERVAL_1SEC_IN_NS);

    status = _inforom_lwl_v3_map_error(pErrorEvent->error, &header, &metadata,
                                   &errorSubtype, &blockType);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    metadata = FLD_SET_DRF_NUM(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA,
                               _LWLIPT_INSTANCE_ID, pErrorEvent->lwliptInstance, metadata);
    if (blockType == INFOROM_LWL_ERROR_BLOCK_TYPE_DL)
    {
        metadata = FLD_SET_DRF_NUM(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID,
                LW_INFOROM_LWL_OBJECT_V3_ERROR_METADATA_BLOCK_ID_DL(pErrorEvent->localLinkIdx),
                metadata);
    }
    else if (blockType == INFOROM_LWL_ERROR_BLOCK_TYPE_TLC)
    {
        metadata = FLD_SET_DRF_NUM(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID,
                LW_INFOROM_LWL_OBJECT_V3_ERROR_METADATA_BLOCK_ID_TLC(pErrorEvent->localLinkIdx),
                metadata);
    }
    else if (blockType == INFOROM_LWL_ERROR_BLOCK_TYPE_LWLIPT)
    {
        metadata = FLD_SET_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA,
                               _BLOCK_ID, _LWLIPT, metadata);
        status = _encode_lwlipt_error_subtype(pErrorEvent->localLinkIdx,
                                           &errorSubtype);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
    }

    for (i = 0; i < INFOROM_LWL_OBJECT_V3S_NUM_ERROR_ENTRIES; i++)
    {
        pErrorEntry = &pLwlObject->errorLog[i];

        if ((pErrorEntry->header == INFOROM_LWL_ERROR_TYPE_ILWALID) ||
            ((pErrorEntry->metadata == metadata) &&
                (pErrorEntry->errorSubtype == errorSubtype)))
        {
            break;
        }
    }

    if (i >= INFOROM_LWL_OBJECT_V3S_NUM_ERROR_ENTRIES)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: LWL error log is full -- unable to log error\n",
                       __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    if (pErrorEntry->header == INFOROM_LWL_ERROR_TYPE_ILWALID)
    {
        pErrorEntry->header       = header;
        pErrorEntry->metadata     = metadata;
        pErrorEntry->errorSubtype = errorSubtype;
    }

    if (pErrorEntry->header == INFOROM_LWL_ERROR_TYPE_ACLWM)
    {
        aclwmTotalCount = LwU64_ALIGN32_VAL(&pErrorEntry->data.aclwm.totalCount);
        if (aclwmTotalCount != LW_U64_MAX)
        {
            if (pErrorEvent->count > LW_U64_MAX - aclwmTotalCount)
            {
                aclwmTotalCount = LW_U64_MAX;
            }
            else
            {
                aclwmTotalCount += pErrorEvent->count;
            }

            LwU64_ALIGN32_PACK(&pErrorEntry->data.aclwm.totalCount, &aclwmTotalCount);
            if (sec < pErrorEntry->data.aclwm.lastUpdated)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: System clock reporting earlier time than error timestamp\n",
                    __FUNCTION__);
            }
            pErrorEntry->data.aclwm.lastUpdated = sec;
            *bDirty = LW_TRUE;
        }
    }
    else if (pErrorEntry->header == INFOROM_LWL_ERROR_TYPE_COUNT)
    {
        // avgEventDelta_n = (avgEventDelta_n-1 + t_n - t_n-1) / 2
        if (pErrorEntry->data.event.totalCount != LW_U32_MAX)
        {
            pErrorEntry->data.event.totalCount++;
            if (sec < pErrorEntry->data.event.lastError)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: System clock reporting earlier time than error timestamp\n",
                    __FUNCTION__);
            }
            else
            {
                pErrorEntry->data.event.avgEventDelta =
                    (pErrorEntry->data.event.avgEventDelta + sec -
                         pErrorEntry->data.event.lastError) >> 1;
            }
            pErrorEntry->data.event.lastError = sec;
            *bDirty = LW_TRUE;
        }
    }
    else
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    return LWL_SUCCESS;
}

static void
_inforom_lwl_get_new_errors_per_minute
(
    LwU32  value,
    LwU32 *pSum
)
{
    *pSum = (*pSum - (*pSum / 60)) + value;
}

static void
_inforom_lwl_update_correctable_error_rates
(
    INFOROM_LWL_CORRECTABLE_ERROR_RATE_STATE_V3S *pState,
    LwU8 link,
    INFOROM_LWLINK_CORRECTABLE_ERROR_COUNTS *pCounts
)
{
    LwU32 i;
    LwU32 tempFlitCrc, tempRxLinkReplay, tempTxLinkReplay, tempLinkRecovery;
    LwU32 tempLaneCrc[4];

    //
    // If the registers have decreased from last reported, then
    // they must have been reset or have overflowed. Set the last
    // register value to 0.
    //
    if (pCounts->flitCrc < pState->lastRead[link].flitCrc)
    {
        pState->lastRead[link].flitCrc = 0;
    }

    for (i = 0; i < LW_ARRAY_ELEMENTS(pState->lastRead[link].laneCrc); i++)
    {
        if (pCounts->laneCrc[i] < pState->lastRead[link].laneCrc[i])
        {
            pState->lastRead[link].laneCrc[i] = 0;
        }
    }

    // Get number of new errors since the last register read
    tempFlitCrc       = pCounts->flitCrc;
    pCounts->flitCrc -= pState->lastRead[link].flitCrc;

    // Update errors per minute with error delta
    _inforom_lwl_get_new_errors_per_minute(pCounts->flitCrc,
            &pState->errorsPerMinute[link].flitCrc);

    // Save the current register value for the next callback
    pState->lastRead[link].flitCrc = tempFlitCrc;

    for (i = 0; i < LW_ARRAY_ELEMENTS(pState->lastRead[link].laneCrc); i++)
    {
        tempLaneCrc[i] = pCounts->laneCrc[i];
        pCounts->laneCrc[i] -= pState->lastRead[link].laneCrc[i];
        _inforom_lwl_get_new_errors_per_minute(pCounts->laneCrc[i],
                &pState->errorsPerMinute[link].laneCrc[i]);

        pState->lastRead[link].laneCrc[i] = tempLaneCrc[i];
    }

    //
    // We don't track rates for the following errors. We just need to stash
    // the current register value and update pCounts with the delta since
    // the last register read.
    //
    if (pCounts->rxLinkReplay < pState->lastRead[link].rxLinkReplay)
    {
        pState->lastRead[link].rxLinkReplay = 0;
    }
    tempRxLinkReplay = pCounts->rxLinkReplay;
    pCounts->rxLinkReplay -= pState->lastRead[link].rxLinkReplay;
    pState->lastRead[link].rxLinkReplay = tempRxLinkReplay;

    if (pCounts->txLinkReplay < pState->lastRead[link].txLinkReplay)
    {
        pState->lastRead[link].txLinkReplay = 0;
    }
    tempTxLinkReplay = pCounts->txLinkReplay;
    pCounts->txLinkReplay -= pState->lastRead[link].txLinkReplay;
    pState->lastRead[link].txLinkReplay = tempTxLinkReplay;

    if (pCounts->linkRecovery < pState->lastRead[link].linkRecovery)
    {
        pState->lastRead[link].linkRecovery = 0;
    }
    tempLinkRecovery = pCounts->linkRecovery;
    pCounts->linkRecovery -= pState->lastRead[link].linkRecovery;
    pState->lastRead[link].linkRecovery = tempLinkRecovery;
}

static LwBool
_inforom_lwl_should_replace_error_rate_entry
(
    INFOROM_LWL_OBJECT_V3_CORRECTABLE_ERROR_RATE *pErrorRate,
    LwU32  flitCrcRate,
    LwU32 *pLaneCrcRates
)
{
    LwU32 i;
    LwU64 lwrrentLaneCrcRateSum = 0;
    LwU64 maxLaneCrcRateSum     = 0;

    for (i = 0; i < LW_ARRAY_ELEMENTS(pErrorRate->laneCrcErrorsPerMinute); i++)
    {
        lwrrentLaneCrcRateSum += pLaneCrcRates[i];
        maxLaneCrcRateSum     += pErrorRate->laneCrcErrorsPerMinute[i];
    }

    return (flitCrcRate > pErrorRate->flitCrcErrorsPerMinute) ||
                (lwrrentLaneCrcRateSum > maxLaneCrcRateSum);
}

static void
_seconds_to_day_and_month
(
    LwU32 sec,
    LwU32 *pDay,
    LwU32 *pMonth
)
{
    *pDay   = sec / (60 * 60 * 24);
    *pMonth = *pDay / 30;
}

static void
_inforom_lwl_update_error_rate_entry
(
    INFOROM_LWL_OBJECT_V3_CORRECTABLE_ERROR_RATE *pErrorRate,
    LwU32 newSec,
    LwU32 newFlitCrcRate,
    LwU32 *pNewLaneCrcRates
)
{
    pErrorRate->lastUpdated = newSec;
    pErrorRate->flitCrcErrorsPerMinute = newFlitCrcRate;
    lwswitch_os_memcpy(pErrorRate->laneCrcErrorsPerMinute, pNewLaneCrcRates,
                       sizeof(pErrorRate->laneCrcErrorsPerMinute));
}

LwlStatus
lwswitch_inforom_lwl_get_max_correctable_error_rate_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    
    struct inforom *pInforom = device->pInforom;
    INFOROM_LWLINK_STATE *pLwlinkState;
    LwU8 linkID = params->linkId;

    if (linkID >= LWSWITCH_NUM_LINKS_LR10)
    {
        return -LWL_BAD_ARGS;
    }
    
    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_os_memset(params, 0, sizeof(LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS));
    params->linkId = linkID;

    lwswitch_os_memcpy(&params->dailyMaxCorrectableErrorRates, &pLwlinkState->pLwl->v3s.maxCorrectableErrorRates.dailyMaxCorrectableErrorRates[0][linkID],
                       sizeof(params->dailyMaxCorrectableErrorRates));

    lwswitch_os_memcpy(&params->monthlyMaxCorrectableErrorRates, &pLwlinkState->pLwl->v3s.maxCorrectableErrorRates.monthlyMaxCorrectableErrorRates[0][linkID],
                       sizeof(params->monthlyMaxCorrectableErrorRates));

    return LWL_SUCCESS;
}

static LwlStatus _lwswitch_inforom_map_lwl_error_to_userspace_error
(
    INFOROM_LWL_OBJECT_V3_ERROR_ENTRY *pErrorLog,
    LWSWITCH_LWLINK_ERROR_ENTRY *pLwlError
)
{
    static const LwU32 DL_RX_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_DL_RX_FAULT_DL_PROTOCOL_FATAL,
        LWSWITCH_LWLINK_ERR_DL_RX_FAULT_SUBLINK_CHANGE_FATAL,
        LWSWITCH_LWLINK_ERR_DL_RX_FLIT_CRC_CORR, 
        LWSWITCH_LWLINK_ERR_DL_RX_LANE0_CRC_CORR,
        LWSWITCH_LWLINK_ERR_DL_RX_LANE1_CRC_CORR,
        LWSWITCH_LWLINK_ERR_DL_RX_LANE2_CRC_CORR,
        LWSWITCH_LWLINK_ERR_DL_RX_LANE3_CRC_CORR,
        LWSWITCH_LWLINK_ERR_DL_RX_LINK_REPLAY_EVENTS_CORR
    };

    static const LwU32 DL_TX_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_DL_TX_FAULT_RAM_FATAL,
        LWSWITCH_LWLINK_ERR_DL_TX_FAULT_INTERFACE_FATAL,
        LWSWITCH_LWLINK_ERR_DL_TX_FAULT_SUBLINK_CHANGE_FATAL,
        LWSWITCH_LWLINK_ERR_DL_TX_LINK_REPLAY_EVENTS_CORR
    };

    static const LwU32 DL_NA_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_DL_LTSSM_FAULT_UP_FATAL,
        LWSWITCH_LWLINK_ERR_DL_LTSSM_FAULT_DOWN_FATAL,
        LWSWITCH_LWLINK_ERR_DL_LINK_RECOVERY_EVENTS_CORR
    };

    static const LwU32 TLC_RX_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_TLC_RX_DL_HDR_PARITY_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DL_DATA_PARITY_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DL_CTRL_PARITY_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_AE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_BE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_ADDR_ALIGN_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_PKTLEN_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_PACKET_STATUS_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CACHE_ATTR_PROBE_REQ_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CACHE_ATTR_PROBE_RSP_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DATLEN_GT_RMW_REQ_MAX_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DATLEN_LT_ATR_RSP_MIN_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_CR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_COLLAPSED_RESPONSE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_HDR_OVERFLOW_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DATA_OVERFLOW_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_STOMP_DETECTED_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_CMD_ENC_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSVD_DAT_LEN_ENC_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_HW_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_UR_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_RSP_STATUS_PRIV_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_POISON_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_AN1_HEARTBEAT_TIMEOUT_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_ILLEGAL_PRI_WRITE_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_HDR_RAM_ECC_DBE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DAT0_RAM_ECC_DBE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_RX_DAT1_RAM_ECC_DBE_FATAL
    };

    static const LwU32 TLC_TX_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_TLC_TX_DL_CREDIT_PARITY_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_NCISOC_HDR_ECC_DBE_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_NCISOC_PARITY_ERR_FATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_ILLEGAL_PRI_WRITE_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC0_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC1_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC2_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC3_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC4_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC5_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC6_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_AN1_TIMEOUT_VC7_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_POISON_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_HW_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_UR_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_RSP_STATUS_PRIV_ERR_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_CREQ_DAT_RAM_ECC_DBE_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_RSP_DAT_RAM_ECC_DBE_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_COM_DAT_RAM_ECC_DBE_NONFATAL,
        LWSWITCH_LWLINK_ERR_TLC_TX_RSP1_DAT_RAM_ECC_DBE_FATAL
    };

    static const LwU32 LIPT_ERRORS[] = 
    {
        LWSWITCH_LWLINK_ERR_LWLIPT_SLEEP_WHILE_ACTIVE_LINK_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PHYCTL_TIMEOUT_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_CLKCTL_TIMEOUT_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_CLKCTL_ILLEGAL_REQUEST_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PLL_TIMEOUT_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_RSTSEQ_PHYARB_TIMEOUT_FATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_ILLEGAL_LINK_STATE_REQUEST_NONFATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_FAILED_MINION_REQUEST_NONFATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_RESERVED_REQUEST_VALUE_NONFATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_LINK_STATE_WRITE_WHILE_BUSY_NONFATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL,
        LWSWITCH_LWLINK_ERR_LWLIPT_LINK_STATE_REQUEST_TIMEOUT_NONFATAL
    };

    LwU32 subType = 0;
    LwU8 lwliptInstance = 0, localLinkIdx = 0;

    if ((pErrorLog == NULL) || (pLwlError == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    subType = DRF_VAL(_INFOROM_LWL_OBJECT_V3, _LWLIPT_ERROR, _SUBTYPE, pErrorLog->errorSubtype);
    lwliptInstance = DRF_VAL(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _LWLIPT_INSTANCE_ID, pErrorLog->metadata);
    pLwlError->timeStamp = pErrorLog->data.event.lastError;
    
    if (pErrorLog->header == INFOROM_LWL_ERROR_TYPE_COUNT)
    {
        pLwlError->count = (LwU64)pErrorLog->data.event.totalCount;
    }
    else if (pErrorLog->header == INFOROM_LWL_ERROR_TYPE_ACLWM)
    {
        pLwlError->count = pErrorLog->data.aclwm.totalCount.hi;
        pLwlError->count = (pLwlError->count << 32) | pErrorLog->data.aclwm.totalCount.lo;
    }
    else
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL0, pErrorLog->metadata) ||
        FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL1, pErrorLog->metadata) ||
        FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL2, pErrorLog->metadata) ||
        FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL3, pErrorLog->metadata) ||
        FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL4, pErrorLog->metadata) ||
        FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _DL5, pErrorLog->metadata))
    {
        localLinkIdx = DRF_VAL(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, pErrorLog->metadata);
        pLwlError->instance = lwliptInstance * LWSWITCH_LINKS_PER_LWLIPT + localLinkIdx;

        if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, _NA, pErrorLog->metadata) &&
            (subType < (sizeof(DL_NA_ERRORS) / sizeof(DL_NA_ERRORS[0]))))
        {
            pLwlError->error = DL_NA_ERRORS[subType];
            return LWL_SUCCESS;
        }
        else if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, _RX, pErrorLog->metadata) &&
            (subType < (sizeof(DL_RX_ERRORS) / sizeof(DL_RX_ERRORS[0]))))
        {
            pLwlError->error = DL_RX_ERRORS[subType];
            return LWL_SUCCESS;
        }
        else if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, _TX, pErrorLog->metadata) &&
                (subType < (sizeof(DL_TX_ERRORS) / sizeof(DL_TX_ERRORS[0]))))
        {
            pLwlError->error = DL_TX_ERRORS[subType];
            return LWL_SUCCESS;
        }
    }

    else if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC0, pErrorLog->metadata) ||
             FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC1, pErrorLog->metadata) ||
             FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC2, pErrorLog->metadata) ||
             FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC3, pErrorLog->metadata) ||
             FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC4, pErrorLog->metadata) ||
             FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _TLC5, pErrorLog->metadata))
    {
        localLinkIdx = DRF_VAL(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, pErrorLog->metadata)
                                  - LW_INFOROM_LWL_OBJECT_V3_ERROR_METADATA_BLOCK_ID_TLC0;
        pLwlError->instance = lwliptInstance * LWSWITCH_LINKS_PER_LWLIPT + localLinkIdx;
        
        if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, _RX, pErrorLog->metadata) &&
            (subType < (sizeof(TLC_RX_ERRORS) / sizeof(TLC_RX_ERRORS[0]))))
        {
            pLwlError->error = TLC_RX_ERRORS[subType];
            return LWL_SUCCESS;
        }
        else if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _DIRECTION, _TX, pErrorLog->metadata) &&
            (subType < (sizeof(TLC_TX_ERRORS) / sizeof(TLC_TX_ERRORS[0]))))
        {
            pLwlError->error = TLC_TX_ERRORS[subType];
            return LWL_SUCCESS;
        }
    }
    else if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _ERROR_METADATA, _BLOCK_ID, _LWLIPT, pErrorLog->metadata))
    {
        if (subType < (sizeof(LIPT_ERRORS) / sizeof(LIPT_ERRORS[0])))
        {
            if (FLD_TEST_DRF(_INFOROM_LWL_OBJECT_V3, _LWLIPT_ERROR, _LINK_ID, _COMMON, pErrorLog->errorSubtype))
            {
                localLinkIdx = 0; //common lwlipt error
            }
            else
            {
                localLinkIdx = DRF_VAL(_INFOROM_LWL_OBJECT_V3, _LWLIPT_ERROR, _LINK_ID, pErrorLog->errorSubtype);
            }

            pLwlError->instance = lwliptInstance * LWSWITCH_LINKS_PER_LWLIPT + localLinkIdx;
            pLwlError->error = LIPT_ERRORS[subType];
            return LWL_SUCCESS;
        }
    }

    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_get_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_LWLINK_STATE *pLwlinkState;
    LwU32 maxReadSize = sizeof(params->errorLog)/sizeof(LWSWITCH_LWLINK_ERROR_ENTRY);
    LwU32 errorLeftCount = 0, errorReadCount = 0, errIndx = 0;
    LwU32 errorStart = params->errorIndex;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pLwlinkState = pInforom->pLwlinkState;
    if (pLwlinkState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (errorStart >= INFOROM_LWL_OBJECT_V3S_NUM_ERROR_ENTRIES)
    {
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(params->errorLog, 0, sizeof(params->errorLog));

    while (((errorStart + errorLeftCount) < INFOROM_LWL_OBJECT_V3S_NUM_ERROR_ENTRIES) &&
           (pLwlinkState->pLwl->v3s.errorLog[errorStart + errorLeftCount].header != INFOROM_LWL_ERROR_TYPE_ILWALID))
    {
        errorLeftCount++;
    }

    if (errorLeftCount > maxReadSize)
    {
        errorReadCount = maxReadSize;
    }
    else
    {
        errorReadCount = errorLeftCount;
    }

    params->errorIndex = errorStart + errorReadCount;
    params->errorCount = errorReadCount;

    if (errorReadCount > 0)
    {
        for (errIndx = 0; errIndx < errorReadCount; errIndx++)
        {
            if (_lwswitch_inforom_map_lwl_error_to_userspace_error(&pLwlinkState->pLwl->v3s.errorLog[errorStart+errIndx],
                                                                   &params->errorLog[errIndx]) != LWL_SUCCESS)
            {
                return -LWL_ERR_NOT_SUPPORTED;
            }
        }
    }

    return LWL_SUCCESS;
}

LwlStatus lwswitch_inforom_lwl_update_link_correctable_error_info_lr10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pData,
    LwU8 linkId,
    LwU8 lwliptInstance,
    LwU8 localLinkIdx,
    void *pLwlErrorCounts,
    LwBool *bDirty
)
{
    INFOROM_LWL_OBJECT_V3S *pLwlObject = &((PINFOROM_LWL_OBJECT)pLwlGeneric)->v3s;
    INFOROM_LWL_CORRECTABLE_ERROR_RATE_STATE_V3S *pState =
                                        &((INFOROM_LWL_CORRECTABLE_ERROR_RATE_STATE *)pData)->v3s;
    INFOROM_LWLINK_CORRECTABLE_ERROR_COUNTS *pErrorCounts =
                                            (INFOROM_LWLINK_CORRECTABLE_ERROR_COUNTS *)pLwlErrorCounts;

    LwU32 i;
    LwU32 sec;
    LwU32 day, month, lwrrentEntryDay, lwrrentEntryMonth;
    INFOROM_LWL_OBJECT_V3_CORRECTABLE_ERROR_RATE *pErrorRate;
    INFOROM_LWL_OBJECT_V3_CORRECTABLE_ERROR_RATE *pOldestErrorRate = NULL;
    INFOROM_LWL_OBJECT_V3S_MAX_CORRECTABLE_ERROR_RATES *pCorrErrorRates;
    LwBool bUpdated = LW_FALSE;
    INFOROM_LWLINK_ERROR_EVENT errorEvent;
    LwU32 lwrrentFlitCrcRate;
    LwU32 *pLwrrentLaneCrcRates;

    if (bDirty == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    *bDirty = LW_FALSE;

    if (linkId >= INFOROM_LWL_OBJECT_V3S_NUM_LINKS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object does not store data for more than %u links (linkId = %u requested)\n",
             INFOROM_LWL_OBJECT_V3S_NUM_LINKS, linkId);
        return -LWL_BAD_ARGS;
    }

    if (lwliptInstance > INFOROM_LWL_OBJECT_V3_LWLIPT_INSTANCE_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object cannot log data for more than %u LWLIPTs (LWLIPT = %u requested)\n",
            INFOROM_LWL_OBJECT_V3_LWLIPT_INSTANCE_MAX, lwliptInstance);
        return -LWL_BAD_ARGS;
    }

    if (localLinkIdx > INFOROM_LWL_OBJECT_V3_BLOCK_ID_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "object cannot log data for more than %u internal links (internal link = %u requested)\n",
            INFOROM_LWL_OBJECT_V3_BLOCK_ID_MAX, localLinkIdx);
        return -LWL_BAD_ARGS;
    }

    sec = (LwU32) (lwswitch_os_get_platform_time() / LWSWITCH_INTERVAL_1SEC_IN_NS);
    _seconds_to_day_and_month(sec, &day, &month);

    _inforom_lwl_update_correctable_error_rates(pState, linkId, pErrorCounts);
    lwrrentFlitCrcRate   = pState->errorsPerMinute[linkId].flitCrc;
    pLwrrentLaneCrcRates = pState->errorsPerMinute[linkId].laneCrc;
    pCorrErrorRates      = &pLwlObject->maxCorrectableErrorRates;

    for (i = 0; i < LW_ARRAY_ELEMENTS(pCorrErrorRates->dailyMaxCorrectableErrorRates); i++)
    {
        pErrorRate = &pCorrErrorRates->dailyMaxCorrectableErrorRates[i][linkId];
        _seconds_to_day_and_month(pErrorRate->lastUpdated, &lwrrentEntryDay,
                                  &lwrrentEntryMonth);

        if ((pErrorRate->lastUpdated == 0) || (lwrrentEntryDay == day))
        {
            if (_inforom_lwl_should_replace_error_rate_entry(pErrorRate,
                                                             lwrrentFlitCrcRate,
                                                             pLwrrentLaneCrcRates))
            {
                _inforom_lwl_update_error_rate_entry(pErrorRate, sec,
                                                lwrrentFlitCrcRate,
                                                pLwrrentLaneCrcRates);
                bUpdated = LW_TRUE;
            }
            pOldestErrorRate = NULL;
            break;
        }
        else if ((pOldestErrorRate == NULL) ||
                 (pErrorRate->lastUpdated < pOldestErrorRate->lastUpdated))
        {
            pOldestErrorRate = pErrorRate;
        }
    }

    if (pOldestErrorRate != NULL)
    {
        _inforom_lwl_update_error_rate_entry(pOldestErrorRate, sec,
                                             lwrrentFlitCrcRate,
                                             pLwrrentLaneCrcRates);
        bUpdated = LW_TRUE;
    }

    for (i = 0; i < LW_ARRAY_ELEMENTS(pCorrErrorRates->monthlyMaxCorrectableErrorRates); i++)
    {
        pErrorRate = &pCorrErrorRates->monthlyMaxCorrectableErrorRates[i][linkId];
        _seconds_to_day_and_month(pErrorRate->lastUpdated, &lwrrentEntryDay,
                                  &lwrrentEntryMonth);

        if ((pErrorRate->lastUpdated == 0) || (lwrrentEntryMonth == month))
        {
            if (_inforom_lwl_should_replace_error_rate_entry(pErrorRate,
                                                             lwrrentFlitCrcRate,
                                                             pLwrrentLaneCrcRates))
            {
                _inforom_lwl_update_error_rate_entry(pErrorRate, sec,
                                                     lwrrentFlitCrcRate,
                                                     pLwrrentLaneCrcRates);
                bUpdated = LW_TRUE;
            }
            pOldestErrorRate = NULL;
            break;
        }
        else if ((pOldestErrorRate == NULL) ||
                 (pErrorRate->lastUpdated < pOldestErrorRate->lastUpdated))
        {
            pOldestErrorRate = pErrorRate;
        }
    }

    if (pOldestErrorRate != NULL)
    {
        _inforom_lwl_update_error_rate_entry(pOldestErrorRate, sec,
                                             lwrrentFlitCrcRate,
                                             pLwrrentLaneCrcRates);
        bUpdated = LW_TRUE;
    }

    *bDirty = bUpdated;

    // Update aggregate error counts for each correctable error

    errorEvent.lwliptInstance = lwliptInstance;
    errorEvent.localLinkIdx   = localLinkIdx;

    if (pErrorCounts->flitCrc > 0)
    {
        errorEvent.error = INFOROM_LWLINK_DL_RX_FLIT_CRC_CORR;
        errorEvent.count = pErrorCounts->flitCrc;
        lwswitch_inforom_lwl_log_error_event_lr10(device,
                pLwlGeneric, &errorEvent, &bUpdated);
        *bDirty |= bUpdated;
    }

    if (pErrorCounts->rxLinkReplay > 0)
    {
        errorEvent.error = INFOROM_LWLINK_DL_RX_LINK_REPLAY_EVENTS_CORR;
        errorEvent.count = pErrorCounts->rxLinkReplay;
        bUpdated = LW_FALSE;
        lwswitch_inforom_lwl_log_error_event_lr10(device,
                pLwlGeneric, &errorEvent, &bUpdated);
        *bDirty |= bUpdated;
    }

    if (pErrorCounts->txLinkReplay > 0)
    {
        errorEvent.error = INFOROM_LWLINK_DL_TX_LINK_REPLAY_EVENTS_CORR;
        errorEvent.count = pErrorCounts->txLinkReplay;
        bUpdated = LW_FALSE;
        lwswitch_inforom_lwl_log_error_event_lr10(device,
                pLwlGeneric, &errorEvent, &bUpdated);
        *bDirty |= bUpdated;
    }

    if (pErrorCounts->linkRecovery > 0)
    {
        errorEvent.error = INFOROM_LWLINK_DL_LINK_RECOVERY_EVENTS_CORR;
        errorEvent.count = pErrorCounts->linkRecovery;
        bUpdated = LW_FALSE;
        lwswitch_inforom_lwl_log_error_event_lr10(device,
                pLwlGeneric, &errorEvent, &bUpdated);
        *bDirty |= bUpdated;
    }

    for (i = 0; i < 4; i++)
    {
        if (pErrorCounts->laneCrc[i] == 0)
        {
            continue;
        }

        errorEvent.error = INFOROM_LWLINK_DL_RX_LANE0_CRC_CORR + i;
        errorEvent.count = pErrorCounts->laneCrc[i];
        bUpdated = LW_FALSE;
        lwswitch_inforom_lwl_log_error_event_lr10(device,
                pLwlGeneric, &errorEvent, &bUpdated);
        *bDirty |= bUpdated;
    }

    return LWL_SUCCESS;
}
#else
LwlStatus
lwswitch_inforom_lwl_get_minion_data_lr10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_set_minion_data_lr10
(
    lwswitch_device     *device,
    void                *pLwlGeneric,
    LwU8                 linkId,
    LwU32               *seedData,
    LwU32                size,
    LwBool              *bDirty
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_log_error_event_lr10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pLwlErrorEvent,
    LwBool *bDirty
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_get_max_correctable_error_rate_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_MAX_CORRECTABLE_ERROR_RATES_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_inforom_lwl_get_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ERROR_COUNTS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
LwlStatus lwswitch_inforom_lwl_update_link_correctable_error_info_lr10
(
    lwswitch_device *device,
    void *pLwlGeneric,
    void *pData,
    LwU8 linkId,
    LwU8 lwliptInstance,
    LwU8 localLinkIdx,
    void *pLwlErrorCounts,
    LwBool *bDirty
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
static
LwlStatus
_inforom_ecc_find_useable_entry_index
(
    INFOROM_ECC_OBJECT_V6_S0 *pEccObj,
    INFOROM_LWS_ECC_ERROR_EVENT *error_event,
    LwU8 *pEntryIndex
)
{
    LwU8 entry;

    //
    // The size of the "entry" variable needs to be updated if the InfoROM ECC
    // error log ever grows past 256
    //
    ct_assert(INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER_MAX_COUNT <= LW_U8_MAX);

    for (entry = 0; entry < INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER_MAX_COUNT; entry++)
    {
        INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER *pErrorEntry = &(pEccObj->errorEntries[entry]);

        //
        // Check if the entry already exists
        // Ideally the address should be verified only if it is valid, however
        // we scrub an invalid address early on so expect them to match the
        // recorded value in either case
        //
        if ((pErrorEntry->errId == error_event->sxid) &&
            FLD_TEST_DRF_NUM(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER,
                             _ADDR_VALID, error_event->bAddressValid, pErrorEntry->header) &&
            (pErrorEntry->address == error_event->address) &&
            FLD_TEST_DRF_NUM(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _LOCATION,
                             _LINK_ID, error_event->linkId, pErrorEntry->location))
            break;
        //
        // Encountering an empty entry indicates this is the first instance of the error
        // The ECC error log on the InfoROM is never sparse so we can terminate
        // the search early
        //
        else if (FLD_TEST_DRF(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER,
                              _VALID, _FALSE, pErrorEntry->header))
            break;
    }

    if (entry == INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER_MAX_COUNT)
        return -LWL_NOT_FOUND;

    *pEntryIndex = entry;

    return LWL_SUCCESS;
}

static
LwlStatus
_inforom_ecc_calc_timestamp_delta
(
    INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER *pErrorEntry,
    INFOROM_LWS_ECC_ERROR_EVENT *error_event,
    LwU64 existingCount
)
{
    //
    // Subtracting 1 from the existingCount to drop the first error event counter
    // Unfortunately we cannot track the first error events counts so assuming 1
    //

    LwlStatus status = LWL_SUCCESS;
    LwU32 lwrrTime = error_event->timestamp;
    LwU64 tmp = ((LwU64) pErrorEntry->averageEventDelta) * (existingCount - 1);
    LwU64 ovfTmp = tmp + (lwrrTime - pErrorEntry->lastErrorTimestamp);
    LwU64 totCnt, delta;

    if (ovfTmp < tmp)
    {
        status = -LWL_NO_MEM;
        goto _updateEntryTimeFailed;
    }

    totCnt = error_event->errorCount + existingCount - 1;
    delta = ovfTmp / totCnt;

    if (delta > LW_U32_MAX)
    {
        status = -LWL_NO_MEM;
        goto _updateEntryTimeFailed;
    }

    pErrorEntry->averageEventDelta = (LwU32) delta;

_updateEntryTimeFailed:
    return status;
}

static
LwlStatus
_inforom_ecc_record_entry
(
    INFOROM_ECC_OBJECT_V6_S0 *pEccObj,
    INFOROM_LWS_ECC_ERROR_EVENT *error_event,
    LwU8 entry
)
{
    LwBool bNewEntry;
    LwU32 *pErrCnt;

    INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER *pErrorEntry = &(pEccObj->errorEntries[entry]);

    bNewEntry = FLD_TEST_DRF(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER,
                             _VALID, _FALSE, pErrorEntry->header);

    pErrCnt = ((error_event->bUncErr) ? &(pErrorEntry->uncorrectedCount) :
                                      &(pErrorEntry->correctedCount));

    if (bNewEntry)
    {
        pErrorEntry->errId = error_event->sxid;

        pErrorEntry->location = FLD_SET_DRF_NUM(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER,
            _LOCATION, _LINK_ID, error_event->linkId, pErrorEntry->location);

        pErrorEntry->header = FLD_SET_DRF_NUM(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER,
            _HEADER, _ADDR_VALID, error_event->bAddressValid, pErrorEntry->header);

        pErrorEntry->address = error_event->address;

        pErrorEntry->sublocation = 0;

        *pErrCnt = error_event->errorCount;

        pErrorEntry->averageEventDelta = 0;

        pErrorEntry->header = FLD_SET_DRF(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER,
                                          _VALID, _TRUE, pErrorEntry->header);
    }
    else
    {
        LwlStatus status;
        LwU64 tmpCnt;
        LwU64 existingCnt = (LwU64) (pErrorEntry->uncorrectedCount + pErrorEntry->correctedCount);

        status = _inforom_ecc_calc_timestamp_delta(pErrorEntry, error_event, existingCnt);
        if (status != LWL_SUCCESS)
        {
            pErrorEntry->header = FLD_SET_DRF(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER,
                _HEADER, _CORRUPT_TIMEDATA, _TRUE, pErrorEntry->header);
        }

        // Update error counts by summing them up
        tmpCnt = (LwU64) *pErrCnt + error_event->errorCount;

        // Saturate at LwU32 limit
        if (tmpCnt > LW_U32_MAX)
        {
            tmpCnt = LW_U32_MAX;
        }

        *pErrCnt = (LwU32) tmpCnt;
    }

    pErrorEntry->lastErrorTimestamp = error_event->timestamp;

    return LWL_SUCCESS;

}

LwlStatus
lwswitch_inforom_ecc_log_error_event_lr10
(
    lwswitch_device *device,
    INFOROM_ECC_OBJECT *pEccGeneric,
    INFOROM_LWS_ECC_ERROR_EVENT *err_event
)
{
    LwU8 entry;
    LwU64_ALIGN32 *pInforomTotalCount;
    LwU64 tmpCount;
    LwlStatus status;
    INFOROM_ECC_OBJECT_V6_S0 *pEccObj;

    if ((err_event == NULL) || (pEccGeneric == NULL))
        return -LWL_BAD_ARGS;

    pEccObj = &(pEccGeneric->v6s);

    //
    // Find the appropriate entry to log the error event
    // If the function returns "out of memory" error, indicates no free entries
    //
    status = _inforom_ecc_find_useable_entry_index(pEccObj, err_event, &entry);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "InfoROM ECC: Unable to find logging entry rc: %d\n", status);
        goto _ecc_log_error_event_lr10_failed;
    }

    //
    // Record the error data into appropriate members of the error entry struct
    // Also mark the entry as in-use if it is a new entry
    //
    status = _inforom_ecc_record_entry(pEccObj, err_event, entry);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "InfoROM ECC: Unable to record entry:%u rc:%d\n",
                       entry, status);

        goto _ecc_log_error_event_lr10_failed;
    }

    // Log the error count to the InfoROM total values
    if (err_event->bUncErr)
    {
        pInforomTotalCount = &(pEccObj->uncorrectedTotal);
    }
    else
    {
        pInforomTotalCount = &(pEccObj->correctedTotal);
    }

    LwU64_ALIGN32_UNPACK(&tmpCount, pInforomTotalCount);

    tmpCount += err_event->errorCount;
    if (tmpCount < err_event->errorCount)
    {
        tmpCount = LW_U64_MAX;
    }

    LwU64_ALIGN32_PACK(pInforomTotalCount, &tmpCount);

    // Update shared surface counts, non-fatal if we encounter a failure
    status = lwswitch_smbpbi_refresh_ecc_counts(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, WARN, "Failed to update ECC counts on SMBPBI "
                       "shared surface rc:%d\n", status);
    }

    return LWL_SUCCESS;

_ecc_log_error_event_lr10_failed:

    LWSWITCH_PRINT(device, ERROR, "Missed recording sxid=%u, linkId=%u, address=0x%04x, "
                   "timestamp=%u, errorCount=%u\n", err_event->sxid,
                   err_event->linkId, err_event->address, err_event->timestamp,
                   err_event->errorCount);

    return status;
}

void
lwswitch_inforom_ecc_get_total_errors_lr10
(
    lwswitch_device    *device,
    INFOROM_ECC_OBJECT *pEccGeneric,
    LwU64              *pCorrectedTotal,
    LwU64              *pUncorrectedTotal
)
{
    INFOROM_ECC_OBJECT_V6_S0 *pEccObj = &(pEccGeneric->v6s);

    LwU64_ALIGN32_UNPACK(pCorrectedTotal, &pEccObj->correctedTotal);
    LwU64_ALIGN32_UNPACK(pUncorrectedTotal, &pEccObj->uncorrectedTotal);
}

static void _lwswitch_inforom_map_ecc_error_to_userspace_error
(
    INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER *pEccError,
    LWSWITCH_ECC_ERROR_ENTRY *pErrorLog
)
{
    pErrorLog->sxid = pEccError->errId;
    pErrorLog->linkId = DRF_VAL(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _LOCATION, _LINK_ID, pEccError->location);
    pErrorLog->lastErrorTimestamp = pEccError->lastErrorTimestamp;
    pErrorLog->bAddressValid = DRF_VAL(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER, _ADDR_VALID, pEccError->header);
    pErrorLog->address = pEccError->address;
    pErrorLog->correctedCount = pEccError->correctedCount;
    pErrorLog->uncorrectedCount = pEccError->uncorrectedCount;
    return;
}


LwlStatus
lwswitch_inforom_ecc_get_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_ECC_ERROR_COUNTS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;
    PINFOROM_ECC_STATE pEccState;
    INFOROM_ECC_OBJECT  *pEcc;
    LwU32 errIndx;

    /*
     * Compile time check is needed here to make sure that the ECC_ERROR API interface query size is in sync 
     * with its internal counterpart. When the definition of the internal InfoROM error size limit changes, 
     * it will enforce API interface change as well, or use a retry style query with err_index 
     */
    ct_assert(LWSWITCH_ECC_ERRORS_MAX_READ_COUNT == INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER_MAX_COUNT);

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pEccState = pInforom->pEccState;
    if (pEccState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pEcc = pEccState->pEcc;
    if (pEcc == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_os_memset(params->errorLog, 0, sizeof(params->errorLog));

    lwswitch_os_memcpy(&params->correctedTotal, &pEcc->v6s.correctedTotal, sizeof(params->correctedTotal));
    lwswitch_os_memcpy(&params->uncorrectedTotal, &pEcc->v6s.uncorrectedTotal, sizeof(params->uncorrectedTotal));

    for (errIndx = 0; errIndx < LWSWITCH_ECC_ERRORS_MAX_READ_COUNT; errIndx++)
    {
        if (FLD_TEST_DRF(_INFOROM_ECC_OBJECT_V6_S0_ERROR_COUNTER, _HEADER, _VALID, _FALSE,
                         pEcc->v6s.errorEntries[errIndx].header))
        {
            break; // the last entry
        }

        _lwswitch_inforom_map_ecc_error_to_userspace_error(&pEcc->v6s.errorEntries[errIndx],
                                                           &params->errorLog[errIndx]);
    }

    params->errorCount = errIndx;

    return LWL_SUCCESS;
}

static LwU8 _oms_dword_byte_sum(LwU16 dword)
{
    LwU8 i, sum = 0;
    for (i = 0; i < sizeof(dword); i++)
        sum += (LwU8)((dword >> (8*i)) & 0xFF);
    return sum;
}

static void _oms_update_entry_checksum
(
    INFOROM_OMS_OBJECT_V1S_SETTINGS_ENTRY *pEntry
)
{
    LwU8 datasum = 0;

    // Upper byte is the checksum
    datasum += _oms_dword_byte_sum(pEntry->data & ~0xFF00);

    pEntry->data = FLD_SET_REF_NUM(
        INFOROM_OMS_OBJECT_V1S_SETTINGS_ENTRY_DATA_ENTRY_CHECKSUM,
        0x00u - datasum, pEntry->data);
}

static void
_oms_reset_entry_iter
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bStart
)
{
    INFOROM_OMS_OBJECT_V1S *pOms = &pOmsState->pOms->v1s;
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    if (bStart)
    {
        pVerData->pIter = &pOms->settings[0];
    }
    else
    {
        pVerData->pIter = &pOms->settings[
            INFOROM_OMS_OBJECT_V1S_NUM_SETTINGS_ENTRIES - 1];
    }
}

static LwBool
_oms_entry_available
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_OBJECT_V1S_SETTINGS_ENTRY *pEntry = pOmsState->omsData.v1s.pIter;

    if (pEntry == NULL)
        return LW_FALSE;

    return FLD_TEST_REF(INFOROM_OMS_OBJECT_V1_SETTINGS_ENTRY_DATA_ENTRY_AVAILABLE,
                        _YES, pEntry->data);
}

static LwBool
_oms_entry_valid
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_OBJECT_V1S_SETTINGS_ENTRY *pEntry = pOmsState->omsData.v1s.pIter;
    LwU8 sum;

    if (pEntry == NULL)
        return LW_FALSE;

    sum = _oms_dword_byte_sum(pEntry->data);

    return (sum == 0);
}

/*
 *
 * Sets nextIdx to one after lwrrIdx. Returns LW_TRUE if nextIdx
 * is valid. LW_FALSE otherwise.
 *
 */
static LwBool
_oms_entry_iter_next
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_OBJECT_V1S *pOms = &pOmsState->pOms->v1s;
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    if (pVerData->pIter >= pOms->settings +
                            INFOROM_OMS_OBJECT_V1S_NUM_SETTINGS_ENTRIES)
    {
        pVerData->pIter = NULL;
    }
    else
    {
        pVerData->pIter++;
    }

    return (pVerData->pIter != NULL);
}

static void
_oms_refresh
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_OBJECT_V1S *pOms = &pOmsState->pOms->v1s;

    lwswitch_os_memset(pOms->settings, 0xFF, sizeof(pOms->settings));
    pOms->lifetimeRefreshCount++;

    // This is guaranteed to find and set an UpdateEntry now
    _oms_parse(device, pOmsState);
}

static void
_oms_set_lwrrent_entry
(
    INFOROM_OMS_STATE *pOmsState
)
{
    pOmsState->omsData.v1s.prev = *pOmsState->omsData.v1s.pIter;
}

static void
_oms_set_update_entry
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    pVerData->pNext = pVerData->pIter;

    // Next settings always start out the same as the previous
    *pVerData->pNext = pVerData->prev;
}

static LwBool
_oms_entry_iter_prev
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_OBJECT_V1S *pOms = &pOmsState->pOms->v1s;
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    if (pVerData->pIter <= pOms->settings)
    {
        pVerData->pIter = NULL;
    }
    else
    {
        pVerData->pIter--;
    }

    return (pVerData->pIter != NULL);
}

static void
_oms_parse
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
)
{
    LwBool bLwrrentValid = LW_FALSE;
    LwBool bIterValid = LW_TRUE;

    //
    // To find the "latest" entry - the one with the settings that were last
    // flushed to the InfoROM - scan from the end of the array until we find
    // an entry that is not available and is valid.
    //
    _oms_reset_entry_iter(pOmsState, LW_FALSE);
    while (bIterValid)
    {
        if (!_oms_entry_available(pOmsState) &&
            _oms_entry_valid(pOmsState))
        {
            _oms_set_lwrrent_entry(pOmsState);
            bLwrrentValid = LW_TRUE;
            break;
        }

        bIterValid = _oms_entry_iter_prev(pOmsState);
    }

    //
    // To find the "next" entry - one that we will write to if a setting is
    // updated - start scanning from the entry after the latest entry to find
    // an available one. This will skip entries that were previously written
    // to but are invalid.
    //
    if (bLwrrentValid)
    {
        bIterValid = _oms_entry_iter_next(pOmsState);
    }
    else
    {
        _oms_reset_entry_iter(pOmsState, LW_TRUE);
        bIterValid = LW_TRUE;
    }

    while (bIterValid)
    {
        if (_oms_entry_available(pOmsState))
        {
            _oms_set_update_entry(pOmsState);
            break;
        }

        bIterValid = _oms_entry_iter_next(pOmsState);
    }

    if (!bIterValid)
    {
        //
        // No more entries available, we will need to refresh the object.
        // We should have at least one valid recent entry in this case
        // (otherwise every entry is corrupted).
        //
        LWSWITCH_ASSERT(bLwrrentValid);
        _oms_refresh(device, pOmsState);
    }
}

static LwBool
_oms_is_content_dirty
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    if (pVerData->pNext == NULL)
        return LW_FALSE;

    return (pVerData->pNext->data != pVerData->prev.data);
}

LwlStatus
lwswitch_oms_inforom_flush_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status = LWL_SUCCESS;
    struct inforom *pInforom = device->pInforom;
    INFOROM_OMS_STATE *pOmsState;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pOmsState = pInforom->pOmsState;

    if (pOmsState != NULL && _oms_is_content_dirty(pOmsState))
    {
        status = lwswitch_inforom_write_object(device, "OMS",
                                            pOmsState->pFmt, pOmsState->pOms,
                                            pOmsState->pPackedObject);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to flush OMS object to InfoROM, rc: %d\n", status);
        }
        else
        {
            _oms_parse(device, pOmsState);
        }
    }

    return status;
}

void
lwswitch_initialize_oms_state_lr10
(
    lwswitch_device *device,
    INFOROM_OMS_STATE *pOmsState
)
{
    pOmsState->omsData.v1s.pIter = pOmsState->omsData.v1s.pNext = NULL;
    pOmsState->omsData.v1s.prev.data =
            REF_DEF(INFOROM_OMS_OBJECT_V1_SETTINGS_ENTRY_DATA_ENTRY_AVAILABLE, _NO) |
            REF_DEF(INFOROM_OMS_OBJECT_V1_SETTINGS_ENTRY_DATA_FORCE_DEVICE_DISABLE, _NO);
    _oms_update_entry_checksum(&pOmsState->omsData.v1s.prev);

    _oms_parse(device, pOmsState);
}

LwBool
lwswitch_oms_get_device_disable_lr10
(
    INFOROM_OMS_STATE *pOmsState
)
{
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    return FLD_TEST_REF(
                INFOROM_OMS_OBJECT_V1_SETTINGS_ENTRY_DATA_FORCE_DEVICE_DISABLE,
                _YES, pVerData->pNext->data);
}

void
lwswitch_oms_set_device_disable_lr10
(
    INFOROM_OMS_STATE *pOmsState,
    LwBool bForceDeviceDisable
)
{
    INFOROM_OMS_V1S_DATA *pVerData = &pOmsState->omsData.v1s;

    pVerData->pNext->data = FLD_SET_REF_NUM(
                INFOROM_OMS_OBJECT_V1_SETTINGS_ENTRY_DATA_FORCE_DEVICE_DISABLE,
                bForceDeviceDisable, pVerData->pNext->data);

    _oms_update_entry_checksum(pVerData->pNext);
}

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
// BBX related functions
static void
_inforom_bbx_callwlate_moving_sum
(
    LwU32  value,
    LwU32  length,
    LwU32 *pSum,
    LwU32  count
)
{
    LwU32 idx;
    LwU32 len = 0;

    for (idx = 0; idx < count; ++idx)
    {
        len += length;
        pSum[idx] = (pSum[idx] - (pSum[idx] / len)) + value;
    }
}

static void
_inforom_bbx_update_temp_bound
(
    PINFOROM_BBX_DATA                 pData,
    LwU32                             timeLwrr,
    LwU32                             timeInterval,
    INFOROM_BBX_OBJ_V1_00_TEMP_ENTRY *pEntry,
    LwU32                            *pEntryIdx,
    LwU32                             entryCount,
    LwBool                            bMax
)
{
    LwS16  tempLwrr;
    LwBool bUpdate  = LW_FALSE;
    LwU32  idx      = *pEntryIdx;

    tempLwrr = SFXP_24_8_to_SFXP_9_7(pData->temperature.value);

    if ((timeLwrr / timeInterval) == (pEntry[idx].timestamp / timeInterval))
    {
        // Check for new max value
        if (bMax)
        {
            if (((LwS16) pEntry[idx].value) < tempLwrr)
            {
                bUpdate = LW_TRUE;
            }
        }
        // Check for new min value
        else
        {
            if (((LwS16) pEntry[idx].value) > tempLwrr)
            {
                bUpdate = LW_TRUE;
            }
        }
    }
    else
    {
        bUpdate = LW_TRUE;
        idx = BBX_MODULAR_INCREMENT(idx, entryCount);
        *pEntryIdx = idx;
    }

    if (bUpdate)
    {
        pEntry[idx].value     = (LwU16) tempLwrr;
        pEntry[idx].timestamp = timeLwrr;
    }
}

static void
_inforom_bbx_update_temperature_histogram
(
    void                                   *pInforomBbxState,
    LwU32                                  *pTempPrevForHistThld,
    INFOROM_BBX_OBJ_TEMP_HISTOGRAM_FIELDS  *pThldHistogramFields,
    INFOROM_BBX_OBJ_TEMP_HISTOGRAM_FIELDS  *pTimeHistogramFields,
    void                                   *pInforomBbxData
)
{
    LwU32 tempLwrr;
    LwU32 tempPrev;
    LwU32 tempThldHigh;
    LwU32 tempThldLow;
    LwU32 idx;

    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    PINFOROM_BBX_DATA pData = (PINFOROM_BBX_DATA)pInforomBbxData;

    // Colwert LwTemp -> Celsius
    tempLwrr = LW_MAX(pData->temperature.value, 0);
    tempLwrr = LW_TYPES_LW_TEMP_TO_CELSIUS_TRUNCED(tempLwrr);

    //
    // Update histogram for time spent in a parilwlar range
    //
    idx  = LW_MAX(tempLwrr, pTimeHistogramFields->min) - pTimeHistogramFields->min;
    idx  = LW_CEIL(idx, pTimeHistogramFields->step);
    idx  = LW_MIN(idx, pTimeHistogramFields->size - 1);

    pTimeHistogramFields->histogram[idx] += 1;

    //
    // Update histogram for threshold crossover count
    //
    // Temperature thresholds are based on thldHistogramMin and thldHistogramStep
    // Have index 0 for highest temperature.
    //
    // tempThldHigh = Temperature thresholds
    // tempThldLow  = With hysteresis it is 1C below the threshold
    //
    // When temperature increases check for threshold crossover:
    //     Lwrr >= High, and Prev < Low
    // - Increment count
    // - When 90/100C level crossed, immediately flush BBX to Inforom.
    //
    // Prev temp is updated to following value under following condition
    // = Lwrr Temp   - when temperature is decreasing
    // = tempThldLow - when higher threshold is crossed
    //
    tempPrev = *pTempPrevForHistThld;

    if (tempPrev > tempLwrr)
    {
        tempPrev = tempLwrr;
    }
    else if (tempPrev < tempLwrr)
    {
        tempThldHigh = pThldHistogramFields->min;
        idx          = pThldHistogramFields->size;
        do
        {
            --idx;
            tempThldLow = tempThldHigh -
                          pThldHistogramFields->hysterisis;

            if ((tempLwrr >= tempThldHigh) && (tempPrev < tempThldLow))
            {
                pThldHistogramFields->histogram[idx] += 1;
                tempPrev = tempThldLow;

                // Temperature is very high. GPU might shutdown. Flush BBX ASAP.
                if (idx <= 1)
                {
                    pBbxState->bFlushImmediately = LW_TRUE;
                }
            }
            tempThldHigh += pThldHistogramFields->step;
        } while (idx != 0);
    }

    *pTempPrevForHistThld = tempPrev;
}

static void
_inforom_bbx_update_temperature_moving_sum
(
    LwU32                     *pTempSumHour,
    LwU32                     *pTempSumDay,
    LwU32                     *pTempSumMnt,
    PINFOROM_BBX_DATA          pData
)
{
    LwU32 tempLwrr;

    // Colwert LwTemp -> Celsius
    tempLwrr = LW_MAX(pData->temperature.value, 0);
    tempLwrr = LW_TYPES_LW_TEMP_TO_CELSIUS_ROUNDED(tempLwrr);

    // Update moving sum for 1 to 23 hours
    _inforom_bbx_callwlate_moving_sum(tempLwrr, INFOROM_BBX_SEC_IN_HOUR,
        pTempSumHour, INFOROM_BBX_OBJ_V1_00_TEMP_SUM_HOUR_ENTRIES);

    // Update moving sum for 1 to 5 days
    _inforom_bbx_callwlate_moving_sum(tempLwrr, INFOROM_BBX_SEC_IN_DAY,
        pTempSumDay, INFOROM_BBX_OBJ_V1_00_TEMP_SUM_DAY_ENTRIES);

    // Update moving sum for 1 to 3 months
    _inforom_bbx_callwlate_moving_sum(tempLwrr, INFOROM_BBX_SEC_IN_MNT,
        pTempSumMnt, INFOROM_BBX_OBJ_V1_00_TEMP_SUM_MNT_ENTRIES);
}

static void
_inforom_bbx_update_temperature_delta_sum
(
    PINFOROM_BBX_STATE         pBbxState,
    LwTemp                    *pTempPrev,
    LwU32                     *pTempSumDelta,
    PINFOROM_BBX_DATA          pData
)
{
    LwS32 tempLwrr;
    LwS32 tempDelta;

    // Colwert LwTemp -> Celsius with 0.1C granularity
    tempLwrr  =
        LW_TYPES_LW_TEMP_TO_CELSIUS_ROUNDED(10 * pData->temperature.value);

    tempDelta = tempLwrr -
        LW_TYPES_LW_TEMP_TO_CELSIUS_ROUNDED(10 * (*pTempPrev));

    if (tempDelta != 0)
    {
        *pTempSumDelta += LW_ABS(tempDelta);
        *pTempPrev = pData->temperature.value;
    }
}

static void
_inforom_bbx_update_temperature_hourly_max
(
    PINFOROM_BBX_STATE                  pBbxState,
    INFOROM_BBX_OBJ_V1_00_TEMP_ENTRY   *hourlyMaxSample,
    LwU32                               size,
    PINFOROM_BBX_DATA                   pData
)
{
    LwU32 timeRun = *pBbxState->systemState.v1_0.timeRun;
    LwU32 idx = (timeRun / INFOROM_BBX_SEC_IN_HOUR) % size;
    LwS16 tempLwrr = SFXP_24_8_to_SFXP_9_7(pData->temperature.value);

    if (tempLwrr > (LwS16)hourlyMaxSample[idx].value)
    {
        hourlyMaxSample[idx].value = (LwU16)tempLwrr;
        hourlyMaxSample[idx].timestamp = pBbxState->timeLwrr;
    }
}

static void
_inforom_bbx_compress_temp
(
    INFOROM_BBX_OBJ_V1_00_TEMP_ENTRY *compressionBuffer,
    LwU32 size
)
{
    LwU32 writeIdx;
    LwU32 readIdx;
    LwU32 maxIdx;

    for (writeIdx = 0, readIdx = 0; readIdx < size; writeIdx++, readIdx += 2)
    {
        if (compressionBuffer[readIdx].value >
            compressionBuffer[readIdx + 1].value)
        {
            maxIdx = readIdx;
        }
        else
        {
            maxIdx = readIdx + 1;
        }
        compressionBuffer[writeIdx].value = compressionBuffer[maxIdx].value;
        compressionBuffer[writeIdx].timestamp = compressionBuffer[maxIdx].timestamp;
    }
}

static void
_inforom_bbx_update_temperature_compression_buffer
(
    PINFOROM_BBX_STATE                  pBbxState,
    INFOROM_BBX_OBJ_V1_00_TEMP_ENTRY   *pCompressionBuffer,
    LwU32                               compressionBufferSize,
    LwU32                              *pCompressionPeriod,
    LwU32                              *pPeriodIdx,
    PINFOROM_BBX_DATA                   pData
)
{
    LwU32 timeRun = *pBbxState->systemState.v1_0.timeRun;
    LwU32 period;
    LwU32 idx;
    LwU32 lastLimitTimeRun;
    LwS16 tempLwrr;
    LwU32 base;

    // Perform compression if timeRun has exceeded current boundary
    if (timeRun >= (compressionBufferSize * pCompressionPeriod[*pPeriodIdx]))
    {
        *pPeriodIdx += 1;
        _inforom_bbx_compress_temp(pCompressionBuffer,
                                compressionBufferSize);
    }

    //
    // base: After the compression buffer has been filled once, subsequent
    //       values should be populated starting from the middle of the array.
    //       First half of the buffer contains historical data.
    //
    // lastLimitTimeRun: The timeRun value when we entered the current period.
    //                   Used to callwlate the delta from the current time
    //                   in order to figure out the proper bucket in the
    //                   compression buffer.
    //
    period = pCompressionPeriod[*pPeriodIdx];
    base = (*pPeriodIdx == 0) ? 0 : compressionBufferSize / 2;
    lastLimitTimeRun = (*pPeriodIdx == 0) ? 0 :
        compressionBufferSize * pCompressionPeriod[*pPeriodIdx - 1];

    idx = base + ((timeRun - lastLimitTimeRun) / period);

    tempLwrr = SFXP_24_8_to_SFXP_9_7(pData->temperature.value);

    if (tempLwrr > (LwS16)pCompressionBuffer[idx].value)
    {
        pCompressionBuffer[idx].value = (LwU16)tempLwrr;
        pCompressionBuffer[idx].timestamp = pBbxState->timeLwrr;
    }
}

LwlStatus
lwswitch_bbx_setup_prologue_lr10
(
    lwswitch_device    *device,
    void               *pInforomBbxState
)
{
    INFOROM_BBX_OBJ_V2_S0 *pBbxObj;
    LwU32 i;

    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(
                                        pBbxState->pObject->header.subversion))
    {
        LWSWITCH_PRINT(device, ERROR, "BBX version is not for lwswitch: (%d.%d)\n",
                        pBbxState->pObject->header.version,
                        pBbxState->pObject->header.subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pBbxState->pObject->header.version != 2)
    {
        LWSWITCH_PRINT(device, ERROR, "Invalid BBX version (%d.%d)\n",
                        pBbxState->pObject->header.version,
                        pBbxState->pObject->header.subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pBbxObj = &pBbxState->pObject->v2_s0;

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMaxDay, timestamp, pBbxState->tempState.v2_0.tempMaxDayIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_DAY_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMaxWeek, timestamp, pBbxState->tempState.v2_0.tempMaxWeekIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_WEEK_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMaxMnt, timestamp, pBbxState->tempState.v2_0.tempMaxMntIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_MNT_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMaxAll, timestamp, pBbxState->tempState.v2_0.tempMaxAllIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_ALL_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMinDay, timestamp, pBbxState->tempState.v2_0.tempMinDayIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_DAY_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMinWeek, timestamp, pBbxState->tempState.v2_0.tempMinWeekIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_WEEK_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMinMnt, timestamp, pBbxState->tempState.v2_0.tempMinMntIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_MNT_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->deviceTemp.tempMinAll, timestamp, pBbxState->tempState.v2_0.tempMinAllIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_ALL_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->xidFirst, timestamp, pBbxState->xidState.v2_s.xidFirstIdx,
        INFOROM_BBX_OBJ_V1_00_XID_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->xidLast, timestamp, pBbxState->xidState.v2_s.xidLastIdx,
        INFOROM_BBX_OBJ_V1_00_XID_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->xidFirstDetailed, xid.timestamp,
        pBbxState->xidState.v2_s.xidFirstDetailedIdx, INFOROM_BBX_OBJ_V1_00_XID_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->xidLastDetailed, xid.timestamp,
        pBbxState->xidState.v2_s.xidLastDetailedIdx, INFOROM_BBX_OBJ_V1_00_XID_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->system.driver, timestamp,
        pBbxState->systemState.v1_0.systemDriverIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->system.vbios, timestamp,
        pBbxState->systemState.v1_0.systemVbiosIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);

    BBX_LATEST_IDX(pBbxObj->system.os, timestamp,
        pBbxState->systemState.v1_0.systemOsIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);

    // Find initial sampling period of temperature compression buffer
    pBbxState->tempState.v2_0.compressionPeriodIdx = INFOROM_BBX_OBJ_V2_X0_NUM_COMPRESSION_PERIODS;
    for (i = 0; i < INFOROM_BBX_OBJ_V2_X0_NUM_COMPRESSION_PERIODS; i++)
    {
        pBbxState->tempState.v2_0.compressionPeriod[i] = INFOROM_BBX_OBJ_V2_X0_TEMP_COMPRESSION_BUFFER_PERIOD(i);
        if (pBbxObj->timeRun < (LwU32)INFOROM_BBX_OBJ_V2_X0_TEMP_COMPRESSION_BUFFER_LIMIT(i))
        {
            if (pBbxState->tempState.v2_0.compressionPeriodIdx ==
                        INFOROM_BBX_OBJ_V2_X0_NUM_COMPRESSION_PERIODS)
            {
                pBbxState->tempState.v2_0.compressionPeriodIdx = i;
            }
        }
    }

    if (pBbxState->tempState.v2_0.compressionPeriodIdx == INFOROM_BBX_OBJ_V2_X0_NUM_COMPRESSION_PERIODS)
    {
        // GPU would have to have been running for >12 years to be here.
        return LW_ERR_ILWALID_STATE;
    }

    // Assign SW state pointers to BBX object
    pBbxState->tempState.v2_0.gpuTemp = &pBbxObj->deviceTemp;
    pBbxState->tempState.v2_0.tempHourlyMaxSample = pBbxObj->tempHourlyMaxSample;
    pBbxState->tempState.v2_0.tempCompressionBuffer = pBbxObj->tempCompressionBuffer;

    pBbxState->xidState.v2_s.xidCount = &pBbxObj->xidCount;
    pBbxState->xidState.v2_s.xidFirst = pBbxObj->xidFirst;
    pBbxState->xidState.v2_s.xidLast = pBbxObj->xidLast;
    pBbxState->xidState.v2_s.xidFirstDetailed = pBbxObj->xidFirstDetailed;
    pBbxState->xidState.v2_s.xidLastDetailed = pBbxObj->xidLastDetailed;

    pBbxState->systemState.v1_0.timeStart = &pBbxObj->timeStart;
    pBbxState->systemState.v1_0.timeEnd = &pBbxObj->timeEnd;
    pBbxState->systemState.v1_0.timeRun = &pBbxObj->timeRun;
    pBbxState->systemState.v1_0.system = &pBbxObj->system;
    pBbxState->systemState.v2_0.time24Hours = &pBbxObj->time24Hours;
    pBbxState->systemState.v2_0.time100Hours = &pBbxObj->time100Hours;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_bbx_setup_epilogue_lr10
(
    lwswitch_device    *device,
    void               *pInforomBbxState
)
{
    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    INFOROM_BBX_OBJ_V2_S0 *pBbxObj;

    if (!INFOROM_OBJECT_SUBVERSION_SUPPORTS_LWSWITCH(
                                        pBbxState->pObject->header.subversion))
    {
        LWSWITCH_PRINT(device, ERROR, "BBX version is not for lwswitch: (%d.%d)\n",
                        pBbxState->pObject->header.version,
                        pBbxState->pObject->header.subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pBbxState->pObject->header.version != 2)
    {
        LWSWITCH_PRINT(device, ERROR, "Invalid BBX version (%d.%d)\n",
                        pBbxState->pObject->header.version,
                        pBbxState->pObject->header.subversion);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pBbxObj = &pBbxState->pObject->v2_s0;

    if ((pBbxObj->system.driver[pBbxState->systemState.v1_0.systemDriverIdx].lo != pBbxState->systemState.v1_0.systemDriverLo) ||
        (pBbxObj->system.driver[pBbxState->systemState.v1_0.systemDriverIdx].hi != pBbxState->systemState.v1_0.systemDriverHi))
    {
        pBbxState->systemState.v1_0.systemDriverIdx = BBX_MODULAR_INCREMENT(
            pBbxState->systemState.v1_0.systemDriverIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);
        pBbxObj->system.driver[pBbxState->systemState.v1_0.systemDriverIdx].lo = pBbxState->systemState.v1_0.systemDriverLo;
        pBbxObj->system.driver[pBbxState->systemState.v1_0.systemDriverIdx].hi = pBbxState->systemState.v1_0.systemDriverHi;
        pBbxObj->system.driver[pBbxState->systemState.v1_0.systemDriverIdx].timestamp = pBbxState->timeLwrr;
    }

    if ((pBbxObj->system.vbios[pBbxState->systemState.v1_0.systemVbiosIdx].oem   != pBbxState->systemState.v1_0.systemVbiosOem) ||
        (pBbxObj->system.vbios[pBbxState->systemState.v1_0.systemVbiosIdx].vbios != pBbxState->systemState.v1_0.systemVbios))
    {
        pBbxState->systemState.v1_0.systemVbiosIdx = BBX_MODULAR_INCREMENT(
            pBbxState->systemState.v1_0.systemVbiosIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);
        pBbxObj->system.vbios[pBbxState->systemState.v1_0.systemVbiosIdx].oem   = pBbxState->systemState.v1_0.systemVbiosOem;
        pBbxObj->system.vbios[pBbxState->systemState.v1_0.systemVbiosIdx].vbios = pBbxState->systemState.v1_0.systemVbios;
        pBbxObj->system.vbios[pBbxState->systemState.v1_0.systemVbiosIdx].timestamp = pBbxState->timeLwrr;
    }

    if ((pBbxObj->system.os[pBbxState->systemState.v1_0.systemOsIdx].type    != pBbxState->systemState.v1_0.systemOsType) ||
        (pBbxObj->system.os[pBbxState->systemState.v1_0.systemOsIdx].version != pBbxState->systemState.v1_0.systemOs))
    {
        pBbxState->systemState.v1_0.systemOsIdx = BBX_MODULAR_INCREMENT(
            pBbxState->systemState.v1_0.systemOsIdx, INFOROM_BBX_OBJ_V1_00_SYSTEM_ENTRIES);
        pBbxObj->system.os[pBbxState->systemState.v1_0.systemOsIdx].type    = pBbxState->systemState.v1_0.systemOsType;
        pBbxObj->system.os[pBbxState->systemState.v1_0.systemOsIdx].version = pBbxState->systemState.v1_0.systemOs;
        pBbxObj->system.os[pBbxState->systemState.v1_0.systemOsIdx].timestamp = pBbxState->timeLwrr;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_bbx_add_data_time_lr10
(
    lwswitch_device *device,
    void            *pInforomBbxState,
    void            *pInforomBbxData
)
{
    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    PINFOROM_BBX_DATA pData = (PINFOROM_BBX_DATA)pInforomBbxData;
    INFOROM_BBX_SYSTEM_STATE_V2 *pBbxSys = &pBbxState->systemState.v2_0;

    if (*pBbxSys->v1.timeStart == 0)
    {
        *pBbxSys->v1.timeStart = pData->time.sec;
    }

    *pBbxSys->v1.timeEnd = pData->time.sec;
    *pBbxSys->v1.timeRun += 1;

    if ((*pBbxSys->v1.timeRun == INFOROM_BBX_SEC_IN_DAY) &&
        (*pBbxSys->time24Hours == 0))
    {
        *pBbxSys->time24Hours = pData->time.sec;
    }

    if ((*pBbxSys->v1.timeRun == (100 * INFOROM_BBX_SEC_IN_HOUR)) &&
        (*pBbxSys->time100Hours == 0))
    {
        *pBbxSys->time100Hours = pData->time.sec;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_bbx_add_sxid_lr10
(
    lwswitch_device *device,
    void            *pInforomBbxState,
    void            *pInforomBbxData
)
{
    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    PINFOROM_BBX_DATA pData = (PINFOROM_BBX_DATA)pInforomBbxData;
    INFOROM_BBX_XID_STATE_V2S       *pBbxXid = &pBbxState->xidState.v2_s;
    INFOROM_BBX_OBJ_V2_S0_XID_ENTRY *pXid = NULL;
    LwU32                           *pIdx;

    if (pBbxXid->xidFirst[INFOROM_BBX_OBJ_V1_00_XID_ENTRIES - 1].timestamp == 0)
    {
        pIdx = &pBbxState->xidState.v2_s.xidFirstIdx;
        pXid = pBbxXid->xidFirst;
    }
    else
    {
        pIdx = &pBbxState->xidState.v2_s.xidLastIdx;
        pXid = pBbxXid->xidLast;
    }

    *pIdx = BBX_MODULAR_INCREMENT(*pIdx,
                INFOROM_BBX_OBJ_V1_00_XID_ENTRIES);
    pXid = &pXid[*pIdx];

    lwswitch_bbx_collect_lwrrent_time(device, pBbxState);

    *pBbxXid->xidCount += 1;
    pXid->number    = pData->xid.XidNumber;
    pXid->timestamp = pBbxState->timeLwrr;
    pXid->versionOs = pBbxState->systemState.v1_0.systemOs;
    pXid->versionDriverLo = pBbxState->systemState.v1_0.systemDriverLo;
    pXid->versionDriverHi = pBbxState->systemState.v1_0.systemDriverHi;

    //
    // Bug 2873978: Ideally we want to flush the BBX at this moment but hold
    // off on that until we have deferred processing in case we're in IRQ
    // context.  Lwrrently it will be flushed on driver unload.
    //

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_bbx_add_temperature_lr10
(
    lwswitch_device *device,
    void            *pInforomBbxState,
    void            *pInforomBbxData
)
{
    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    PINFOROM_BBX_DATA pData = (PINFOROM_BBX_DATA)pInforomBbxData;
    INFOROM_BBX_TEMP_STATE_V2 *pBbxTemp = &pBbxState->tempState.v2_0;
    INFOROM_BBX_OBJ_TEMP_HISTOGRAM_FIELDS thldHistogramFields = {
                                pBbxTemp->gpuTemp->tempHistogramThld,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_THLD_ENTRIES,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_THLD_MIN,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_THLD_STEP,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_THLD_HYSTERESIS
    };

    INFOROM_BBX_OBJ_TEMP_HISTOGRAM_FIELDS timeHistogramFields = {
                                pBbxTemp->gpuTemp->tempHistogramTime,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_TIME_ENTRIES,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_TIME_MIN,
                                INFOROM_BBX_OBJ_V2_X0_TEMP_HISTOGRAM_TIME_STEP,
                                0
    };

    // Update max entry for daily temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_DAY, pBbxTemp->gpuTemp->tempMaxDay, &pBbxTemp->tempMaxDayIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_DAY_ENTRIES, LW_TRUE);

    // Update min entry for daily temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_DAY, pBbxTemp->gpuTemp->tempMinDay, &pBbxTemp->tempMinDayIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_DAY_ENTRIES, LW_FALSE);

    // Update max entry for weekly temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_WEEK, pBbxTemp->gpuTemp->tempMaxWeek, &pBbxTemp->tempMaxWeekIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_WEEK_ENTRIES, LW_TRUE);

    // Update min entry for weekly temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_WEEK, pBbxTemp->gpuTemp->tempMinWeek, &pBbxTemp->tempMinWeekIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_WEEK_ENTRIES, LW_FALSE);

    // Update max entry for monthly temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_MNT, pBbxTemp->gpuTemp->tempMaxMnt, &pBbxTemp->tempMaxMntIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_MNT_ENTRIES, LW_TRUE);

    // Update min entry for monthly temperature
    _inforom_bbx_update_temp_bound(pData, pBbxState->timeLwrr,
        INFOROM_BBX_SEC_IN_MNT, pBbxTemp->gpuTemp->tempMinMnt, &pBbxTemp->tempMinMntIdx,
        INFOROM_BBX_OBJ_V2_X0_TEMP_MNT_ENTRIES, LW_FALSE);

    _inforom_bbx_update_temperature_histogram(pBbxState, &pBbxTemp->tempPrevForHistThld,
                            &thldHistogramFields,
                            &timeHistogramFields,
                            pData);

    _inforom_bbx_update_temperature_moving_sum(pBbxTemp->gpuTemp->tempSumHour,
                        pBbxTemp->gpuTemp->tempSumDay,
                        pBbxTemp->gpuTemp->tempSumMnt, pData);

    _inforom_bbx_update_temperature_delta_sum(pBbxState, &pBbxTemp->tempPrev,
                                    &pBbxTemp->gpuTemp->tempSumDelta, pData);

    _inforom_bbx_update_temperature_hourly_max(pBbxState, pBbxTemp->tempHourlyMaxSample,
                        INFOROM_BBX_OBJ_V2_X0_TEMP_HOURLY_MAX_ENTRIES, pData);

    _inforom_bbx_update_temperature_compression_buffer(pBbxState,
                            pBbxTemp->tempCompressionBuffer,
                            INFOROM_BBX_OBJ_V2_X0_TEMP_COMPRESS_BUFFER_ENTRIES,
                            pBbxTemp->compressionPeriod,
                            &pBbxTemp->compressionPeriodIdx, pData);

    return LWL_SUCCESS;
}

void
lwswitch_bbx_set_initial_temperature_lr10
(
    lwswitch_device *device,
    void            *pInforomBbxState,
    void            *pInforomBbxData
)
{
    PINFOROM_BBX_STATE pBbxState = (PINFOROM_BBX_STATE)pInforomBbxState;
    PINFOROM_BBX_DATA pData = (PINFOROM_BBX_DATA)pInforomBbxData;
    INFOROM_BBX_TEMP_STATE_V2 *pBbxTemp = &pBbxState->tempState.v2_0;

    pBbxTemp->tempPrev = pData->temperature.value;
    pBbxTemp->tempPrevForHistThld =
            LW_TYPES_LW_TEMP_TO_CELSIUS_TRUNCED(pBbxTemp->tempPrev);
}

LwlStatus
lwswitch_inforom_bbx_get_sxid_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    struct inforom *pInforom = device->pInforom;
    INFOROM_BBX_STATE *pBbxState;
    INFOROM_BBX_XID_STATE_V2S *pXid;
    int sxidIndex = 0;

    if (pInforom == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pBbxState = pInforom->pBbxState;
    if (pBbxState == NULL)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!pBbxState->bValid)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pXid = &pBbxState->xidState.v2_s;

    params->sxidCount = *pXid->xidCount;

    for (sxidIndex = 0; sxidIndex < INFOROM_BBX_OBJ_V1_00_XID_ENTRIES; sxidIndex++)
    {
        params->sxidFirst[sxidIndex].sxid = pXid->xidFirst[sxidIndex].number;
        params->sxidFirst[sxidIndex].timestamp = pXid->xidFirst[sxidIndex].timestamp;
        params->sxidLast[sxidIndex].sxid = pXid->xidLast[sxidIndex].number;
        params->sxidLast[sxidIndex].timestamp = pXid->xidLast[sxidIndex].timestamp;
    }

    return LWL_SUCCESS;
}
#else
LwlStatus
lwswitch_bbx_setup_prologue_lr10
(
    lwswitch_device    *device,
    void               *pInforomBbxState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_setup_epilogue_lr10
(
    lwswitch_device    *device,
    void *pInforomBbxState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_data_time_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_sxid_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_bbx_add_temperature_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
void
lwswitch_bbx_set_initial_temperature_lr10
(
    lwswitch_device *device,
    void *pInforomBbxState,
    void *pInforomBbxData
)
{
    return;
}

LwlStatus
lwswitch_inforom_bbx_get_sxid_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_SXIDS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
