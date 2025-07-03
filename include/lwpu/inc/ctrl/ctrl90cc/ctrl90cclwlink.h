/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl90cc/ctrl90cclwlink.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl90cc/ctrl90ccbase.h"

/* GF100_PROFILER LWLINK control commands and parameters */

/*
 * LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS
 *
 * This command attempts to reserve lwlink counters for use by the calling
 * client. This object should be allocated as a child of a subdevice.
 *
 * If the reservation is held by another client, then this command will fail and
 * will return LW_ERR_STATE_IN_USE. 
 *
 * This command will fail and return LW_ERR_ILWALID_OBJECT_PARENT if this object
 * is not a child of a subdevice. A return status of LW_OK guarantees that the
 * client holds the reservation.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_STATE_IN_USE
 *    LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS (0x90cc0201) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | 0x1" */

/*
 * LW90CC_CTRL_CMD_LWLINK_RELEASE_COUNTERS
 *
 * This command releases an existing reservation of the lwlink counters for
 * the calling client. If the calling client does not lwrrently have the 
 * reservation as acquired by LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS,
 * this command will return LW_ERR_ILWALID_REQUEST.
 *
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_REQUEST
 */
#define LW90CC_CTRL_CMD_LWLINK_RELEASE_COUNTERS (0x90cc0202) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | 0x2" */

#define LW90CC_CTRL_LWLINK_MAX_LINKS            32
#define LW90CC_CTRL_LWLINK_COUNTER_MAX_TYPES    32

/*
 * LW90CC_CTRL_CMD_LWLINK_GET_COUNTERS
 *  This command gets the counts for different counter types. The GF100_PROFILER
 *  instance must hold the LWLINK counters reservation as obtained by
 *  LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS to use this command.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to get the counter data.
 *
 * [in/out] counterData
 *  This parameter specifies both input and output per-link counter data. Each
 *  data element has the following fields:
 *      [in]  counterMask
 *          This parameter specifies the input mask for desired counter types.
 *          Valid fields for this parameter include the
 *          LW90CC_CTRL_LWLINK_COUNTER_* fields defined below.
 *
 *      [out] overflowMask
 *          This parameter identifies which, if any, of the link counters
 *          overflowed. The bit fields of this mask are identical to those of
 *          the counterMask field.
 *
 *      [out] counters
 *          This array contains the error counts for each error type as
 *          requested from the counterMask. The bit index of each counter in
 *          the counterMask is also its array index in this array.
 */

typedef struct LW90CC_CTRL_LWLINK_COUNTER_DATA {
    LwU32 counterMask;
    LwU32 overflowMask;
    LW_DECLARE_ALIGNED(LwU64 counters[LW90CC_CTRL_LWLINK_COUNTER_MAX_TYPES], 8);
} LW90CC_CTRL_LWLINK_COUNTER_DATA;

#define LW90CC_CTRL_LWLINK_GET_COUNTERS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90CC_CTRL_LWLINK_GET_COUNTERS_PARAMS {
    LwU32 linkMask;
    LW_DECLARE_ALIGNED(LW90CC_CTRL_LWLINK_COUNTER_DATA counterData[LW90CC_CTRL_LWLINK_MAX_LINKS], 8);
} LW90CC_CTRL_LWLINK_GET_COUNTERS_PARAMS;

#define LW90CC_CTRL_CMD_LWLINK_GET_COUNTERS                 (0x90cc0203) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_GET_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_COUNTERS_NONE                    0x00000000
#define LW90CC_CTRL_LWLINK_COUNTERS_ALL                     0x0FFF0F0F //only enable the masks been defined

#define LW90CC_CTRL_LWLINK_COUNTER_TL_TX0                       0:0
#define LW90CC_CTRL_LWLINK_COUNTER_TL_TX1                       1:1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_RX0                       2:2
#define LW90CC_CTRL_LWLINK_COUNTER_TL_RX1                       3:3

#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L(i)      \
        (1 << (i + 8)):(1 << (i + 8))
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE__SIZE 4
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L0        8:8
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L1        9:9
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L2        10:10
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L3        11:11

#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT           16:16
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(i)      \
    (1 << (i + 17)):(1 << (i + 17))
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE 8
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0        17:17
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1        18:18
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2        19:19
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3        20:20
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4        21:21
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5        22:22
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6        23:23
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7        24:24
#define LW90CC_CTRL_LWLINK_COUNTER_DL_TX_ERR_REPLAY             25:25
#define LW90CC_CTRL_LWLINK_COUNTER_DL_TX_ERR_RECOVERY           26:26
#define LW90CC_CTRL_LWLINK_COUNTER_DL_RX_ERR_REPLAY             27:27

/*
 * LW90CC_CTRL_CMD_LWLINK_SET_TL_COUNTER_CFG
 *  This command configures the TL counter control registers to specify what is
 *  counted by each of the four sets of TL counters per link. The GF100_PROFILER
 *  instance must hold the LWLINK counters reservation as obtained by
 *  LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS to use this command.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to set the TL counter
 *  control registers.
 *
 * [in] linkCfg
 *  This parameter specifies the per-link configurations for each link specified
 *  in the link mask. Each configuration element has the following fields:
 *      [in] tx0Cfg
 *          Settings for TX Throughput Counter Register 0. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [in] tx1Cfg
 *          Settings for TX Throughput Counter Register 1. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [in] rx0Cfg
 *          Settings for RX Throughput Counter Register 0. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [in] rx1Cfg
 *          Settings for RX Throughput Counter Register 1. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_LWLINK_SET_TL_COUNTER_CFG           (0x90cc0204) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_SET_TL_COUNTER_CFG_PARAMS_MESSAGE_ID" */

typedef struct LW90CC_CTRL_LWLINK_TL_COUNTER_CFG {
    LwU32 tx0Cfg;
    LwU32 tx1Cfg;
    LwU32 rx0Cfg;
    LwU32 rx1Cfg;
} LW90CC_CTRL_LWLINK_TL_COUNTER_CFG;

#define LW90CC_CTRL_LWLINK_SET_TL_COUNTER_CFG_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW90CC_CTRL_LWLINK_SET_TL_COUNTER_CFG_PARAMS {
    LwU32                             linkMask;
    LW90CC_CTRL_LWLINK_TL_COUNTER_CFG linkCfg[LW90CC_CTRL_LWLINK_MAX_LINKS];
} LW90CC_CTRL_LWLINK_SET_TL_COUNTER_CFG_PARAMS;

/*
 * LW90CC_CTRL_CMD_LWLINK_GET_TL_COUNTER_CFG
 *  This command retrieves the current configuration of the TL counter control
 *  registers to identify what is counted by each of the four sets of TL
 *  counters per link.
 *
 *  Note that the parameters for this control command are identical to those for
 *  LW90CC_CTRL_CMD_LWLINK_SET_TL_COUNTER_CFG_PARAMS, except the link
 *  configuration parameters are outputs instead of inputs.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to get the TL counter
 *  control registers.
 *
 * [out] linkCfg
 *  This parameter specifies the per-link configurations for each link specified
 *  in the link mask. Each configuration element has the following fields:
 *      [out] tx0Cfg
 *          Settings for TX Throughput Counter Register 0. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [out] tx1Cfg
 *          Settings for TX Throughput Counter Register 1. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [out] rx0Cfg
 *          Settings for RX Throughput Counter Register 0. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 *      [out] rx1Cfg
 *          Settings for RX Throughput Counter Register 1. Valid fields for this
 *          parameter include the LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_* fields
 *          defined below.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW90CC_CTRL_CMD_LWLINK_GET_TL_COUNTER_CFG (0x90cc0205) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_GET_TL_COUNTER_CFG_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_GET_TL_COUNTER_CFG_PARAMS_MESSAGE_ID (0x5U)

typedef LW90CC_CTRL_LWLINK_SET_TL_COUNTER_CFG_PARAMS LW90CC_CTRL_LWLINK_GET_TL_COUNTER_CFG_PARAMS;

/*
 * Defines what unit of traffic this counter will count if enabled.
 */
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_UNIT                  2:1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_UNIT_CYCLES      0x0
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_UNIT_PACKETS     0x1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_UNIT_FLITS       0x2
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_UNIT_BYTES       0x3

/*
 * Identifies the type(s) of flits to count when Unit = Flits.
 * More than one filter bit may be set, in which case flits of all matching
 * types are counted.
 */
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_FLITFILTER            6:3
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_FLITFILTER_HEAD  0x1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_FLITFILTER_AE    0x2
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_FLITFILTER_BE    0x4
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_FLITFILTER_DATA  0x8

/*
 * Qualifies the counting to particular packet types when the counting unit is
 * Packets, Flits, or Data Bytes. More than one filter bit may be set, in which
 * case the contents of all matching packets are counted.
 */
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER             14:7
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_NOP    0x01
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_READ   0x02
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_WRITE  0x04
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_RATOM  0x08
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_NRATOM 0x10
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_FLUSH  0x20
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_RESPD  0x40
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PKTFILTER_RESP   0x80

/*
 * Qualifies the counting to packets containing particular attributes when the
 * counting unit is Packets, Flits, or Data Bytes. More than one filter bit may
 * be set, in which case the contents of all matching packets are counted. These
 * attributes do not apply to flush requests, so flushes are always considered
 * to match regardless of the values of these bits.
 */
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_ATTRFILTER            16:15
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_ATTRFILTER_NCNP  0x1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_ATTRFILTER_NCP   0x2

/*
 * Indicates how many events are needed to cause a pulse on the appropriate
 * pm_<tx,rx>_cnt<0,1> signal to the perfmon logic. This allows event frequency
 * to be divided down when sysclkSysClk is slow.
 */
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE                19:17
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_1         0x0
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_2         0x1
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_4         0x2
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_8         0x3
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_16        0x4
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_32        0x5
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_64        0x6
#define LW90CC_CTRL_LWLINK_COUNTER_TL_CFG_PMSIZE_128       0x7

/*
 * LW90CC_CTRL_CMD_LWLINK_CLEAR_COUNTERS
 *  This command clears/resets the counters for the specified types. The
 *  GF100_PROFILER instance must hold the LWLINK counters reservation as
 *  obtained by LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS to use this command.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to clear the counters.
 *
 * [in] counterMask
 *  This parameter specifies per-link which counters should be cleared. Note
 *  that not all counters can be cleared. Valid fields for this parameter
 *  include the LW90CC_CTRL_LWLINK_COUNTER_* fields defined for the
 *  LW90CC_CTRL_CMD_LWLINK_GET_COUNTERS control command.
 *
 *  NOTE: Bug# 2098529: On Turing all DL errors and LP counters are cleared
 *        together. They cannot be cleared individually per error type. RM
 *        would possibly move to a new API on Ampere and beyond
 */
#define LW90CC_CTRL_CMD_LWLINK_CLEAR_COUNTERS              (0x90cc0206) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW90CC_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS {
    LwU32 linkMask;
    LwU32 counterMask[LW90CC_CTRL_LWLINK_MAX_LINKS];
} LW90CC_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS;

/*
 * LW90CC_CTRL_CMD_LWLINK_SET_COUNTERS_FROZEN
 *  This command freezes or unfreezes the counters for the specified types. The
 *  GF100_PROFILER instance must hold the LWLINK counters reservation as
 *  obtained by LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS to use this command.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to freeze or unfreeze the
 *  counters.
 *
 * [in] counterMask
 *  This parameter specifies per-link which counters should be frozen or
 *  unfrozen. Note that not all counters can be frozen. Valid fields for this
 *  parameter include the LW90CC_CTRL_LWLINK_COUNTER_* fields defined for the
 *  LW90CC_CTRL_CMD_LWLINK_GET_COUNTERS control command.
 *
 * [in] bFrozen
 *  This parameter specifies whether the counters should be frozen or unfrozen
 *  with this command. A value of LW_TRUE will cause the counters to be frozen,
 *  while a value of LW_FALSE will cause the counters to be unfrozen.
 */
#define LW90CC_CTRL_CMD_LWLINK_SET_COUNTERS_FROZEN (0x90cc0207) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_SET_COUNTERS_FROZEN_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_SET_COUNTERS_FROZEN_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW90CC_CTRL_LWLINK_SET_COUNTERS_FROZEN_PARAMS {
    LwU32  linkMask;
    LwU32  counterMask[LW90CC_CTRL_LWLINK_MAX_LINKS];
    LwBool bFrozen;
} LW90CC_CTRL_LWLINK_SET_COUNTERS_FROZEN_PARAMS;

/*
 * LW90CC_CTRL_CMD_LWLINK_GET_LP_COUNTERS
 *
 */
#define LW90CC_CTRL_CMD_LWLINK_GET_LP_COUNTERS             (0x90cc0208) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_LWHS   0
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_EIGHTH 1
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_OTHER  2
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_NUM_TX_LP_ENTER 3
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_NUM_TX_LP_EXIT  4
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_SLEEP  5
#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_MAX_COUNTERS    6

#define LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS {
    LwU32 linkId;
    LwU32 counterValidMask;
    LwU32 counterValues[LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_MAX_COUNTERS];
} LW90CC_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS;

/*
 * LW90CC_CTRL_CMD_LWLINK_CLEAR_LP_COUNTERS
 *
 *  NOTE: Bug# 2098529: On Turing all DL errors and LP counters are cleared
 *        together. They cannot be cleared individually per error type. RM
 *        would possibly move to a new API on Ampere and beyond
 *
 */
#define LW90CC_CTRL_CMD_LWLINK_CLEAR_LP_COUNTERS (0x90cc0209) /* finn: Evaluated from "(FINN_GF100_PROFILER_LWLINK_INTERFACE_ID << 8) | LW90CC_CTRL_LWLINK_CLEAR_LP_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW90CC_CTRL_LWLINK_CLEAR_LP_COUNTERS_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW90CC_CTRL_LWLINK_CLEAR_LP_COUNTERS_PARAMS {
    LwU32 linkId;
} LW90CC_CTRL_LWLINK_CLEAR_LP_COUNTERS_PARAMS;

/* _ctrl90cclwlink_h_ */
