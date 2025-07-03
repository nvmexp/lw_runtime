/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SUGEN_LS10_H_
#define _SUGEN_LS10_H_

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"

#include "lwswitch/ls10/dev_lws_top.h"
#include "lwswitch/ls10/dev_pri_masterstation_ip.h"
#include "lwswitch/ls10/dev_pri_ringstation_sys_ip.h"
#include "lwswitch/ls10/dev_pri_ringstation_sysb_ip.h"
#include "lwswitch/ls10/dev_pri_ringstation_prt_ip.h"
#include "lwswitch/ls10/dev_pri_hub_sys_ip.h"
#include "lwswitch/ls10/dev_pri_hub_sysb_ip.h"
#include "lwswitch/ls10/dev_pri_hub_prt_ip.h"
#include "lwswitch/ls10/dev_lwlsaw_ip.h"
#include "lwswitch/ls10/dev_ctrl_ip.h"
#include "lwswitch/ls10/dev_timer_ip.h"
#include "lwswitch/ls10/dev_trim.h"
#include "lwswitch/ls10/dev_lw_xal_ep.h"
#include "lwswitch/ls10/dev_lw_xpl.h"
#include "lwswitch/ls10/dev_xtl_ep_pri.h"
#include "lwswitch/ls10/dev_soe_ip.h"
#include "lwswitch/ls10/dev_se_pri.h"
#include "lwswitch/ls10/dev_fuse.h"
#include "lwswitch/ls10/dev_perf.h"
#include "lwswitch/ls10/dev_pmgr.h"
#include "lwswitch/ls10/dev_therm.h"

// LWLW
#include "lwswitch/ls10/dev_lwlw_ip.h"
#include "lwswitch/ls10/dev_cpr_ip.h"
#include "lwswitch/ls10/dev_lwlipt_ip.h"
#include "lwswitch/ls10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/ls10/dev_lwltlc_ip.h"
#include "lwswitch/ls10/dev_lwldl_ip.h"
#include "lwswitch/ls10/dev_minion_ip.h"

// NPG/NPORT
#include "lwswitch/ls10/dev_npg_ip.h"
#include "lwswitch/ls10/dev_npgperf_ip.h"
#include "lwswitch/ls10/dev_nport_ip.h"
#include "lwswitch/ls10/dev_route_ip.h"
#include "lwswitch/ls10/dev_tstate_ip.h"
#include "lwswitch/ls10/dev_egress_ip.h"
#include "lwswitch/ls10/dev_ingress_ip.h"
#include "lwswitch/ls10/dev_sourcetrack_ip.h"
#include "lwswitch/ls10/dev_multicasttstate_ip.h"
#include "lwswitch/ls10/dev_reductiontstate_ip.h"

// NXBAR
#include "lwswitch/ls10/dev_nxbar_tile_ip.h"
#include "lwswitch/ls10/dev_nxbar_tileout_ip.h"

#endif //_SUGEN_LS10_H_
