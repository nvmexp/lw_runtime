/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "common_lwswitch.h"
#include "intr_lwswitch.h"

#include "ls10/ls10.h"
#include "ls10/minion_ls10.h"

#include "lwswitch/ls10/dev_ctrl_ip.h"
#include "lwswitch/ls10/dev_pri_masterstation_ip.h"
#include "lwswitch/ls10/dev_pri_hub_sys_ip.h"
#include "lwswitch/ls10/dev_pri_hub_sysb_ip.h"
#include "lwswitch/ls10/dev_pri_hub_prt_ip.h"

#include "lwswitch/ls10/dev_nport_ip.h"
#include "lwswitch/ls10/dev_route_ip.h"
#include "lwswitch/ls10/dev_ingress_ip.h"
#include "lwswitch/ls10/dev_sourcetrack_ip.h"
#include "lwswitch/ls10/dev_egress_ip.h"
#include "lwswitch/ls10/dev_tstate_ip.h"
#include "lwswitch/ls10/dev_multicasttstate_ip.h"
#include "lwswitch/ls10/dev_reductiontstate_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"
#include "lwswitch/ls10/dev_lwlw_ip.h"
#include "lwswitch/ls10/dev_cpr_ip.h"
#include "lwswitch/ls10/dev_lwldl_ip.h"

#include "lwswitch/ls10/dev_ctrl_ip_addendum.h"

static void
_lwswitch_initialize_route_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _ROUTEBUFERR, _ENABLE)          |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _GLT_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _PDCTRLPARERR, _ENABLE)         |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _LWS_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _CDTPARERR, _ENABLE)            |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _MCRID_ECC_DBE_ERR, _ENABLE)    |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _EXTMCRID_ECC_DBE_ERR, _ENABLE) |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _RAM_ECC_DBE_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _NOPORTDEFINEDERR, _ENABLE)         |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _ILWALIDROUTEPOLICYERR, _ENABLE)    |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _GLT_ECC_LIMIT_ERR, _ENABLE)        |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _LWS_ECC_LIMIT_ERR, _ENABLE)        |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _MCRID_ECC_LIMIT_ERR, _ENABLE)      |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _EXTMCRID_ECC_LIMIT_ERR, _ENABLE)   |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _RAM_ECC_LIMIT_ERR, _ENABLE)        |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _ILWALID_MCRID_ERR, _ENABLE);
    // NOTE: _MC_TRIGGER_ERR is debug-use only 

    contain =
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _ROUTEBUFERR, __PROD)           |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _GLT_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _PDCTRLPARERR, __PROD)          |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _LWS_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _CDTPARERR, __PROD)             |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _MCRID_ECC_DBE_ERR, __PROD)     |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _EXTMCRID_ECC_DBE_ERR, __PROD)  |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _RAM_ECC_DBE_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _ROUTE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.route.fatal = report_fatal;
    chip_device->intr_mask.route.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_ingress_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _CMDDECODEERR, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _EXTAREMAPTAB_ECC_DBE_ERR, _ENABLE)  |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_DBE_ERR, _ENABLE)    |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _ILWALIDVCSET, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _REMAPTAB_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RIDTAB_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RLANTAB_ECC_DBE_ERR, _ENABLE)       |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_PARITY_ERR, _ENABLE)         |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _EXTBREMAPTAB_ECC_DBE_ERR, _ENABLE)  |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _MCREMAPTAB_ECC_DBE_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REQCONTEXTMISMATCHERR, _ENABLE)    |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ACLFAIL, _ENABLE)                  |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_LIMIT_ERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ADDRBOUNDSERR, _ENABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RIDTABCFGERR, _ENABLE)             |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RLANTABCFGERR, _ENABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REMAPTAB_ECC_LIMIT_ERR, _ENABLE)   |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RIDTAB_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RLANTAB_ECC_LIMIT_ERR, _ENABLE)    |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ADDRTYPEERR, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTAREMAPTAB_INDEX_ERR, _ENABLE)   |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTBREMAPTAB_INDEX_ERR, _ENABLE)   |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _MCREMAPTAB_INDEX_ERR, _ENABLE)     |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTAREMAPTAB_REQCONTEXTMISMATCHERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTBREMAPTAB_REQCONTEXTMISMATCHERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _MCREMAPTAB_REQCONTEXTMISMATCHERR, _ENABLE)   |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTAREMAPTAB_ACLFAIL, _ENABLE)     |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTBREMAPTAB_ACLFAIL, _ENABLE)     |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _MCREMAPTAB_ACLFAIL, _ENABLE)       |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTAREMAPTAB_ADDRBOUNDSERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EXTBREMAPTAB_ADDRBOUNDSERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _MCREMAPTAB_ADDRBOUNDSERR, _ENABLE);

    contain =
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _CMDDECODEERR, __PROD)             |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _EXTAREMAPTAB_ECC_DBE_ERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _NCISOC_HDR_ECC_DBE_ERR, __PROD)   |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ILWALIDVCSET, __PROD)             |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _REMAPTAB_ECC_DBE_ERR, __PROD)     |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RIDTAB_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RLANTAB_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _NCISOC_PARITY_ERR, __PROD)        |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _EXTBREMAPTAB_ECC_DBE_ERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _MCREMAPTAB_ECC_DBE_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _INGRESS, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _INGRESS, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _INGRESS, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _INGRESS, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _INGRESS, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.ingress.fatal = report_fatal;
    chip_device->intr_mask.ingress.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_egress_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _EGRESSBUFERR, _ENABLE)                 |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _PKTROUTEERR, _ENABLE)                  |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _SEQIDERR, _ENABLE)                     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOCCREDITOVFL, _ENABLE)             |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _REQTGTIDMISMATCHERR, _ENABLE)          |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _RSPREQIDMISMATCHERR, _ENABLE)          |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _URRSPERR, _ENABLE)                     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _HWRSPERR, _ENABLE)                     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_HDR_PARITY_ERR, _ENABLE)         |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_CREDIT_PARITY_ERR, _ENABLE)     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_FLITTYPE_MISMATCH_ERR, _ENABLE)  |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _CREDIT_TIME_OUT_ERR, _ENABLE)          |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _ILWALIDVCSET_ERR, _ENABLE)             |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_SIDEBAND_PD_PARITY_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, _ENABLE)   |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _PRIVRSPERR, _ENABLE)                  |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RFU, _ENABLE);

    contain =
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _EGRESSBUFERR, __PROD)                 |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _PKTROUTEERR, __PROD)                  |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _SEQIDERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_HDR_ECC_DBE_ERR, __PROD)        |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _RAM_OUT_HDR_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOCCREDITOVFL, __PROD)             |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _REQTGTIDMISMATCHERR, __PROD)          |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _RSPREQIDMISMATCHERR, __PROD)          |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _URRSPERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _HWRSPERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_HDR_PARITY_ERR, __PROD)         |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOC_CREDIT_PARITY_ERR, __PROD)     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_FLITTYPE_MISMATCH_ERR, __PROD)  |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _CREDIT_TIME_OUT_ERR, __PROD)          |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _ILWALIDVCSET_ERR, _ENABLE)            |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_SIDEBAND_PD_PARITY_ERR, _ENABLE);


    enable = report_fatal | report_nonfatal;

    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _EGRESS, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _EGRESS, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _EGRESS, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _EGRESS, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _EGRESS, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.egress.fatal = report_fatal;
    chip_device->intr_mask.egress.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_tstate_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOLBUFERR, _ENABLE)              |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOL_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTOREBUFERR, _ENABLE)           |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_DBE_ERR, _ENABLE)     |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _ATO_ERR, _ENABLE)                    |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CAMRSP_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOL_ECC_LIMIT_ERR, _ENABLE)      |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE);

    contain =
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOLBUFERR, __PROD)             |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOL_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTOREBUFERR, __PROD)          |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_ECC_DBE_ERR, __PROD)    |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _ATO_ERR, __PROD)                   |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CAMRSP_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _TSTATE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _TSTATE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _TSTATE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _TSTATE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _TSTATE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.tstate.fatal = report_fatal;
    chip_device->intr_mask.tstate.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_sourcetrack_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, _ENABLE) |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _DUP_CREQ_TCEN0_TAG_ERR, _ENABLE)     |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _ILWALID_TCEN0_RSP_ERR, _ENABLE)      |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _ILWALID_TCEN1_RSP_ERR, _ENABLE)      |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _SOURCETRACK_TIME_OUT_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE);

    contain =
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, __PROD) |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _DUP_CREQ_TCEN0_TAG_ERR, __PROD)       |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _ILWALID_TCEN0_RSP_ERR, __PROD)        |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _ILWALID_TCEN1_RSP_ERR, __PROD)        |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _SOURCETRACK_TIME_OUT_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _SOURCETRACK, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _SOURCETRACK, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _SOURCETRACK, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _SOURCETRACK, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.sourcetrack.fatal = report_fatal;
    chip_device->intr_mask.sourcetrack.nonfatal = report_nonfatal;

}


static void
_lwswitch_initialize_multicast_tstate_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_MULTICASTTSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOL_ECC_DBE_ERR, _ENABLE)             |
        DRF_DEF(_MULTICASTTSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_BUF_OVERWRITE_ERR, _ENABLE)    |
        DRF_DEF(_MULTICASTTSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_DBE_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_MULTICASTTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOL_ECC_LIMIT_ERR, _ENABLE)       |
        DRF_DEF(_MULTICASTTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE)    |
        DRF_DEF(_MULTICASTTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_MCTO_ERR, _ENABLE);

    contain =
        DRF_DEF(_MULTICASTTSTATE, _ERR_CONTAIN_EN_0, _TAGPOOL_ECC_DBE_ERR, __PROD)           |
        DRF_DEF(_MULTICASTTSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_BUF_OVERWRITE_ERR, __PROD)  |
        DRF_DEF(_MULTICASTTSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_ECC_DBE_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _MULTICASTTSTATE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.mc_tstate.fatal = report_fatal;
    chip_device->intr_mask.mc_tstate.nonfatal = report_nonfatal;
}


static void
_lwswitch_initialize_reduction_tstate_interrupts
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_REDUCTIONTSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOL_ECC_DBE_ERR, _ENABLE)            |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_BUF_OVERWRITE_ERR, _ENABLE)   |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_DBE_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_REDUCTIONTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOL_ECC_LIMIT_ERR, _ENABLE)      |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE)   |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_RTO_ERR, _ENABLE);

    contain =
        DRF_DEF(_REDUCTIONTSTATE, _ERR_CONTAIN_EN_0, _TAGPOOL_ECC_DBE_ERR, __PROD)          |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_BUF_OVERWRITE_ERR, __PROD) |
        DRF_DEF(_REDUCTIONTSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_ECC_DBE_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _REDUCTIONTSTATE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.red_tstate.fatal = report_fatal;
    chip_device->intr_mask.red_tstate.nonfatal = report_nonfatal;
}

void
_lwswitch_initialize_nport_interrupts_ls10
(
    lwswitch_device *device
)
{
    LwU32 val;

    val =
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _CORRECTABLEENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _FATALENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _NONFATALENABLE, 1);
    LWSWITCH_NPORT_BCAST_WR32_LS10(device, _NPORT, _ERR_CONTROL_COMMON_NPORT, val);

    _lwswitch_initialize_route_interrupts(device);
    _lwswitch_initialize_ingress_interrupts(device);
    _lwswitch_initialize_egress_interrupts(device);
    _lwswitch_initialize_tstate_interrupts(device);
    _lwswitch_initialize_sourcetrack_interrupts(device);
    _lwswitch_initialize_multicast_tstate_interrupts(device);
    _lwswitch_initialize_reduction_tstate_interrupts(device);
}

/*
 * @brief Service MINION Falcon interrupts on the requested interrupt tree
 *        Falcon Interrupts are a little unique in how they are handled:#include <assert.h>
 *        IRQSTAT is used to read in interrupt status from FALCON
 *        IRQMASK is used to read in mask of interrupts
 *        IRQDEST is used to read in enabled interrupts that are routed to the HOST
 *
 *        IRQSTAT & IRQMASK gives the pending interrupting on this minion 
 *
 * @param[in] device   MINION on this device
 * @param[in] instance MINION instance
 *
 */
LwlStatus
lwswitch_minion_service_falcon_interrupts_ls10
(
    lwswitch_device *device,
    LwU32           instance
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled, intr, link;

    link = instance * LWSWITCH_LINKS_PER_MINION_LS10;
    report.raw_pending = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_IRQSTAT);
    report.raw_enable = chip_device->intr_minion_dest;
    report.mask = LWSWITCH_MINION_RD32_LS10(device, instance, _CMINION, _FALCON_IRQMASK);

    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending; 

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _WDTMR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_WATCHDOG, "MINION Watchdog timer ran out", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _HALT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_HALT, "MINION HALT", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _EXTERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_EXTERR, "MINION EXTERR", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN0, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION Falcon SWGEN0 interrupt on MINION %d.\n",
                      __FUNCTION__, instance);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN1, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_PRINT(device, INFO,
                       "%s: Received MINION Falcon SWGEN1 interrupt on MINION %d.\n",
                      __FUNCTION__, instance);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (device->link[link].fatal_error_oclwrred)
    {
        intr = LWSWITCH_MINION_RD32_LS10(device, instance, _MINION, _MINION_INTR_STALL_EN);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FATAL, _DISABLE, intr);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FALCON_STALL, _DISABLE, intr);
        LWSWITCH_MINION_WR32_LS10(device, instance, _MINION, _MINION_INTR_STALL_EN, intr);
    }

    // Write to IRQSCLR to clear status of interrupt
    LWSWITCH_MINION_WR32_LS10(device, instance, _CMINION, _FALCON_IRQSCLR, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Send priv ring command and wait for completion
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 * @param[in] cmd           encoded priv ring command
 */
static LwlStatus
_lwswitch_ring_master_cmd_ls10
(
    lwswitch_device *device,
    LwU32 cmd
)
{
    LwU32 value;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    LWSWITCH_ENG_WR32(device, PRI_MASTER_RS, , 0, _PPRIV_MASTER, _RING_COMMAND, cmd);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        value = LWSWITCH_ENG_RD32(device, PRI_MASTER_RS, , 0, _PPRIV_MASTER, _RING_COMMAND);
        if (FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for RING_COMMAND == NO_CMD (cmd=0x%x).\n",
            __FUNCTION__, cmd);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_priv_ring_ls10
(
    lwswitch_device *device
)
{
    LwU32 pending, i;
    LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;
    LwlStatus status = LWL_SUCCESS;

    pending = LWSWITCH_ENG_RD32(device, PRI_MASTER_RS, , 0, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    //
    // SYS
    //

    if (FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 1, pending))
    {
        pri_error.addr = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_ENG_RD32(device, SYS_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_CODE);

        LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_ERROR, "PRI WRITE SYS error", LWSWITCH_PPRIV_WRITE_SYS, 0, pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "SYS PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending = FLD_SET_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 0, pending);
    }

    //
    // SYSB
    //

    if (FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYSB, 1, pending))
    {
        pri_error.addr = LWSWITCH_ENG_RD32(device, SYSB_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_ENG_RD32(device, SYSB_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_ENG_RD32(device, SYSB_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_ENG_RD32(device, SYSB_PRI_HUB, , 0, _PPRIV_SYS, _PRIV_ERROR_CODE);

        LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_ERROR, "PRI WRITE SYSB error", LWSWITCH_PPRIV_WRITE_SYS, 1, pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "SYSB PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending = FLD_SET_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYSB, 0, pending);
    }

    //
    // per-PRT
    //

    for (i = 0; i < NUM_PRT_PRI_HUB_ENGINE_LS10; i++)
    {
        if (DRF_VAL(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, pending) & LWBIT(i))
        {
            pri_error.addr = LWSWITCH_ENG_RD32(device, PRT_PRI_HUB, , i, _PPRIV_PRT, _PRIV_ERROR_ADR);
            pri_error.data = LWSWITCH_ENG_RD32(device, PRT_PRI_HUB, , i, _PPRIV_PRT, _PRIV_ERROR_WRDAT);
            pri_error.info = LWSWITCH_ENG_RD32(device, PRT_PRI_HUB, , i, _PPRIV_PRT, _PRIV_ERROR_INFO);
            pri_error.code = LWSWITCH_ENG_RD32(device, PRT_PRI_HUB, , i, _PPRIV_PRT, _PRIV_ERROR_CODE);

            LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_ERROR, "PRI WRITE PRT error", LWSWITCH_PPRIV_WRITE_PRT, i, pri_error);

            LWSWITCH_PRINT(device, ERROR,
                "PRT%d PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
                i, pri_error.addr, pri_error.data, pri_error.info, pri_error.code);

            pending &= ~DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                _GBL_WRITE_ERROR_FBP, LWBIT(i));
        }
    }

    if (pending != 0)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, Unexpected PRI error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 2, 0, LW_FALSE, &pending);

        LWSWITCH_PRINT(device, ERROR,
            "Unexpected PRI error 0x%08x\n", pending);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    // TODO reset the priv ring like GPU driver?

    // acknowledge the interrupt to the ringmaster
    status = _lwswitch_ring_master_cmd_ls10(device,
        DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ACK_INTERRUPT));
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Timeout ACK'ing PRI error\n");
        //
        // Don't return error code -- there is nothing kernel SW can do about it if ACK failed.
        // Likely it is PLM protected and SOE needs to handle it.
        //
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Enable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_enable_interrupts_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_ENG_WR32(device, GIN, , 0, _CTRL, _CPU_INTR_LEAF_EN_SET(LW_CTRL_CPU_INTR_UNITS_IDX),
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PMGR_HOST, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER_ALARM, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XTL_CPU, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XAL_EP, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PRIV_RING, 1));

    LWSWITCH_ENG_WR32(device, GIN, , 0, _CTRL, _CPU_INTR_TOP_EN_SET(0), 0xFFFFFFFF);
}

/*
 * @Brief : Disable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_disable_interrupts_ls10
(
    lwswitch_device *device
)
{
    LWSWITCH_ENG_WR32(device, GIN, , 0, _CTRL, _CPU_INTR_LEAF_EN_CLEAR(LW_CTRL_CPU_INTR_UNITS_IDX),
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PMGR_HOST, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER_ALARM, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XTL_CPU, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XAL_EP, 1) |
        DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PRIV_RING, 1));

    LWSWITCH_ENG_WR32(device, GIN, , 0, _CTRL, _CPU_INTR_TOP_EN_CLEAR(0), 0xFFFFFFFF);
}

//
// Check if there are interrupts pending.
//
// On silicon/emulation we only use MSIs which are not shared, so this
// function does not need to be called.
//
// FSF/RTMsim does not model interrupts correctly.  The interrupt is shared
// with USB so we must check the HW status.  In addition we must disable
// interrupts to run the interrupt thread. On silicon this is done
// automatically in XVE.
//
// This is called in the ISR context by the Linux driver.  The WAR does
// access more of device outside the Linux mutex than it should. Sim only
// supports 1 device lwrrently so these fields are safe while interrupts
// are enabled.
//
// TODO: Bug 1881361 to remove the FSF WAR
//
LwlStatus
lwswitch_lib_check_interrupts_ls10
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU32 val;

    val = LWSWITCH_ENG_RD32(device, GIN, , 0, _CTRL, _CPU_INTR_TOP(0));
    if (DRF_NUM(_CTRL, _CPU_INTR_TOP, _VALUE, val) != 0)
    {
        retval = -LWL_MORE_PROCESSING_REQUIRED;
    }

    return retval;
}

//
// Service interrupt and re-enable interrupts. Interrupts should disabled when
// this is called.
//
LwlStatus
lwswitch_lib_service_interrupts_ls10
(
    lwswitch_device *device
)
{
    LwlStatus   status = LWL_SUCCESS;
    LwlStatus   return_status = LWL_SUCCESS;
    LwU32 val[8];

    // Check UNITS
    val[0] = LWSWITCH_ENG_RD32(device, GIN, , 0, _CTRL, _CPU_INTR_UNITS);
    if (val[0] != 0)
    {
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PMGR_HOST, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PMGR_HOST interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PTIMER interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PTIMER_ALARM, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PTIMER_ALARM interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _SEC0_INTR0_0, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _SEC0_INTR0_0 interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _SOE_SHIM_ILLEGAL_OP, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _SOE_SHIM_ILLEGAL_OP interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _SOE_SHIM_FLUSH, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _SOE_SHIM_FLUSH interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XTL_CPU, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _XTL_CPU interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _XAL_EP, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _XAL_EP interrupt pending\n",
                __FUNCTION__);
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _PRIV_RING, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _PRIV_RING interrupt pending\n",
                __FUNCTION__);
            status = _lwswitch_service_priv_ring_ls10(device);
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR, "%s: Problem handling PRI errors\n",
                    __FUNCTION__);
                return_status = status;
            }
        }
        if (FLD_TEST_DRF_NUM(_CTRL, _CPU_INTR_UNITS, _FSP, 1, val[0]))
        {
            LWSWITCH_PRINT(device, ERROR, "%s: _FSP interrupt pending\n",
                __FUNCTION__);
        }
    }

    return return_status;
}

/*
 * Initialize interrupt tree HW for all units.
 *
 * Init and servicing both depend on bits matching across STATUS/MASK
 * and IErr STATUS/LOG/REPORT/CONTAIN registers.
 */
void
lwswitch_initialize_interrupt_tree_ls10
(
    lwswitch_device *device
)
{
    LwU64 link_mask = lwswitch_get_enabled_link_mask(device);
    LwU32 i, val;

    // NPG/NPORT
    _lwswitch_initialize_nport_interrupts_ls10(device);

    FOR_EACH_INDEX_IN_MASK(64, i, link_mask)
    {
        val = LWSWITCH_LINK_RD32(device, i,
                  LWLW, _LWLW, _LINK_INTR_0_MASK(i));
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_0_MASK, _FATAL,       _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_0_MASK, _NONFATAL,    _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_0_MASK, _CORRECTABLE, _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_0_MASK, _INTR0,       _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_0_MASK, _INTR1,       _ENABLE, val);
        LWSWITCH_LINK_WR32(device, i, LWLW, _LWLW, _LINK_INTR_0_MASK(i), val);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    FOR_EACH_INDEX_IN_MASK(64, i, link_mask)
    {
        val = LWSWITCH_LINK_RD32(device, i,
                  LWLW, _LWLW, _LINK_INTR_1_MASK(i));
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_1_MASK, _FATAL,       _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_1_MASK, _NONFATAL,    _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_1_MASK, _CORRECTABLE, _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_1_MASK, _INTR0,       _ENABLE, val);
        val = FLD_SET_DRF(_LWLW, _LINK_INTR_1_MASK, _INTR1,       _ENABLE, val);
        LWSWITCH_LINK_WR32(device, i, LWLW, _LWLW, _LINK_INTR_1_MASK(i), val);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    val = LWSWITCH_ENG_RD32(device, CPR, _BCAST, 0, _CPR_SYS, _ERR_LOG_EN_0);
    val = FLD_SET_DRF(_CPR_SYS, _ERR_LOG_EN_0, _ENGINE_RESET_ERR, __PROD, val);
    LWSWITCH_ENG_WR32(device, CPR, _BCAST, 0, _CPR_SYS, _ERR_LOG_EN_0, val);
}

//
// Service Lwswitch LWLDL Fatal interrupts
//
LwlStatus
lwswitch_service_lwldl_fatal_link_ls10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LS10(device, link, LWLDL, _LWLDL_TOP, _INTR);
    report.raw_enable = LWSWITCH_LINK_RD32_LS10(device, link, LWLDL, _LWLDL_TOP, _INTR_STALL_EN);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LS10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_RAM, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_RAM, "TX Fault Ram", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_RAM_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_INTERFACE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_INTERFACE, "TX Fault Interface", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_INTERFACE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_SUBLINK_CHANGE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_SUBLINK_CHANGE, "TX Fault Sublink Change", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_SUBLINK_CHANGE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_FAULT_SUBLINK_CHANGE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_SUBLINK_CHANGE, "RX Fault Sublink Change", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_RX_FAULT_SUBLINK_CHANGE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_FAULT_DL_PROTOCOL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_DL_PROTOCOL, "RX Fault DL Protocol", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_RX_FAULT_DL_PROTOCOL_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_FAULT_DOWN, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_FAULT_DOWN, "LTSSM Fault Down", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_LTSSM_FAULT_DOWN_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_FAULT_UP, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_FAULT_UP, "LTSSM Fault Up", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_LTSSM_FAULT_UP_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_PROTOCOL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_PROTOCOL, "LTSSM Protocol Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);

        // TODO 2827793 this should be logged to the InfoROM as fatal
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _PHY_A, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_PHY_A, "PHY_A Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_PHY_A_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_PL_ERROR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_PL_ERROR, "TX_PL Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_PL_ERROR_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
   }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_PL_ERROR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_PL_ERROR, "RX_PL Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_RX_PL_ERROR_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LS10(device, link, LWLDL, _LWLDL_TOP, _INTR_STALL_EN,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LINK_WR32_LS10(device, link, LWLDL, _LWLDL_TOP, _INTR, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLDL fatal interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_service_minion_link_ls10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, unhandled, minionIntr, linkIntr, reg, enabledLinks, bit;
    LwU32 localLinkIdx, link;

    //
    // _MINION_MINION_INTR shows all interrupts lwrrently at the host on this minion
    // Note: _MINIO_MINION_INTR is not used to clear link specific interrupts
    //
    minionIntr = LWSWITCH_MINION_RD32_LS10(device, instance, _MINION, _MINION_INTR);

    // get all possible interrupting links associated with this minion
    report.raw_pending = DRF_VAL(_MINION, _MINION_INTR, _LINK, minionIntr);

    // read in the enaled minion interrupts on this minion
    reg = LWSWITCH_MINION_RD32_LS10(device, instance, _MINION, _MINION_INTR_STALL_EN);

    // get the links with enabled interrupts on this minion
    enabledLinks = DRF_VAL(_MINION, _MINION_INTR_STALL_EN, _LINK, reg);

    report.raw_enable = enabledLinks;
    report.mask = report.raw_enable;

    // pending bit field contains interrupting links after being filtered
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    FOR_EACH_INDEX_IN_MASK(32, localLinkIdx, pending)
    {
        link = (instance * LWSWITCH_LINKS_PER_LWLIPT_LS10) + localLinkIdx;
        bit = LWBIT(localLinkIdx);

        // read in the interrupt register for the given link
        linkIntr = LWSWITCH_MINION_LINK_RD32_LS10(device, link, _MINION, _LWLINK_LINK_INTR(localLinkIdx));

        // _STATE must be set for _CODE to be valid
        if (!DRF_VAL(_MINION, _LWLINK_LINK_INTR, _STATE, linkIntr))
        {
            continue;
        }

        report.data[0] = linkIntr;

        switch(DRF_VAL(_MINION, _LWLINK_LINK_INTR, _CODE, linkIntr))
        {
            case LW_MINION_LWLINK_LINK_INTR_CODE_NA:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link NA interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_SWREQ:
                LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION Link SW Generate interrupt on MINION %d : link %d.\n",
                      __FUNCTION__, instance, link);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_DLREQ:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link DLREQ interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_PMDISABLED:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link PMDISABLED interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_DLCMDFAULT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link DLCMDFAULT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_TLREQ:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link TLREQ interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NOINIT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link NOINIT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NOTIFY:
                LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION NOTIFY interrupt on MINION %d : link %d.\n",
                      __FUNCTION__, instance, link);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_LOCAL_CONFIG_ERR:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link Local-Config-Error interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NEGOTIATION_CONFIG_ERR:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link Negotiation Config Err Interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_BADINIT: 
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link BADINIT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_PMFAIL:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link PMFAIL interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_INBAND_BUFFER_AVAILABLE:
            {
                LwlStatus status;

                device->link[link].inBandData.bTransferFail = LW_FALSE;
                LWSWITCH_PRINT(device, INFO,
                      "Received INBAND_BUFFER_AVAILABLE interrupt on MINION %d,\n", instance);

                status = lwswitch_minion_receive_inband_data_ls10(device, localLinkIdx);
                if (status != LWL_SUCCESS)
                    return status;
                break;
            }

            case LW_MINION_LWLINK_LINK_INTR_CODE_INBAND_BUFFER_COMPLETE:
            {
                LWSWITCH_PRINT(device, INFO,
                      "Received INBAND_BUFFER_COMPLETE interrupt on MINION %d,\n", instance);

                // Inform the client that the Inband Buffer Data transfer was successful
                LWSWITCH_PRINT(device, INFO, "In Band Buffer Transfer Complete\n");

                // Mark Inband transfer as success 
                device->link[link].inBandData.bTransferFail = LW_FALSE;

                // call the wrapper, which checks how many chunks of 256B are pending
                break;
            }

            case LW_MINION_LWLINK_LINK_INTR_CODE_INBAND_BUFFER_FAIL:
            {
                LwlStatus status;

                LWSWITCH_PRINT(device, INFO,
                      "Received INBAND_BUFFER_FAIL interrupt on MINION %d,\n", instance);

                // Mark Inband transfer as unsuccessful 
                device->link[link].inBandData.bTransferFail = LW_TRUE;

                if (device->link[link].inBandData.bIsSenderMinion)
                {
                    status = lwswitch_minion_send_inband_data_ls10(device, localLinkIdx);
                    if (status != LWL_SUCCESS)
                        return status;
                }
                else
                {
                    status = lwswitch_minion_receive_inband_data_ls10(device, localLinkIdx);
                    if (status != LWL_SUCCESS)
                        return status;
                }
                break;
            }

            default:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Interrupt code unknown", LW_FALSE);
        }
        lwswitch_clear_flags(&unhandled, bit);

        // Disable interrupt bit for the given link - fatal error olwrred before
        if (device->link[link].fatal_error_oclwrred)
        {
            enabledLinks &= ~bit;
            reg = DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, enabledLinks);
            LWSWITCH_MINION_LINK_WR32_LS10(device, link, _MINION, _MINION_INTR_STALL_EN, reg);
        }

        //
        // _MINION_INTR_LINK is a read-only register field for the host
        // Host must write 1 to _LWLINK_LINK_INTR_STATE to clear the interrupt on the link
        //
        reg = DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, 1);
        LWSWITCH_MINION_WR32_LS10(device, instance, _MINION, _LWLINK_LINK_INTR(localLinkIdx), reg);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}
