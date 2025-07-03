/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HALDEFS_SOE_LWSWITCH_H_
#define _HALDEFS_SOE_LWSWITCH_H_

#include "g_lwconfig.h"
#include "lwstatus.h"
#include "flcnifcmn.h"
#include "flcn/haldefs_flcnable_lwswitch.h"

struct SOE;

typedef struct {
    // needs to be the first thing in this struct so that a soe_hal* can be
    // re-interpreted as a flcnable_hal* and vise-versa.
    flcnable_hal base;

    //add any hal functions specific to SOE here
    LW_STATUS                   (*processMessages)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LW_STATUS                   (*waitForInitAck)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);

    LwU32                       (*service)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    void                        (*serviceHalt)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    void                        (*ememTransfer)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe,
                                        LwU32                           dmemAddr,
                                        LwU8                           *pBuf,
                                        LwU32                           sizeBytes,
                                        LwU8                            port,
                                        LwBool                          bCopyFrom);
    LwU32                       (*getEmemSize)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LwU32                       (*getEmemStartOffset)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LW_STATUS                   (*ememPortToRegAddr)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe,
                                        LwU32                           port,
                                        LwU32                          *pEmemCAddr,
                                        LwU32                          *pEmemDAddr);
    void                        (*serviceExterr)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LW_STATUS                   (*getExtErrRegAddrs)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe,
                                        LwU32                          *pExtErrAddr,
                                        LwU32                          *pExtErrStat);
    LwU32                       (*ememPortSizeGet)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LwBool                      (*isCpuHalted)(
                                        struct lwswitch_device         *device,
                                        struct SOE                     *pSoe);
    LwlStatus                    (*testDma)(
                                        struct lwswitch_device         *device);
    LwlStatus                   (*setPexEOM)(
                                        struct lwswitch_device         *device,
                                        LwU8                            mode, 
                                        LwU8                            nblks,
                                        LwU8                            nerrs,
                                        LwU8                            berEyeSel);
    LwlStatus                   (*getPexEomStatus)(
                                        struct lwswitch_device         *device,
                                        LwU8                            mode,
                                        LwU8                            nblks,
                                        LwU8                            nerrs,
                                        LwU8                            berEyeSel,
                                        LwU32                           laneMask,
                                        LwU16                          *pEomStatus);
    LwlStatus                   (*getUphyDlnCfgSpace)(
                                        struct lwswitch_device         *device,
                                        LwU32                           regAddress,
                                        LwU32                           laneSelectMask,
                                        LwU16                          *pRegValue);
    LwlStatus                   (*forceThermalSlowdown)(
                                        struct lwswitch_device         *device,
                                        LwBool                          slowdown,
                                        LwU32                           periodUs);
    LwlStatus                   (*setPcieLinkSpeed)(
                                        struct lwswitch_device         *device,
                                        LwU32                           linkSpeed);
} soe_hal;

// HAL functions
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LR10)
void soeSetupHal_LR10(struct SOE *pSoe);
#endif
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
void soeSetupHal_LS10(struct SOE *pSoe);
#endif

#endif //_HALDEFS_SOE_LWSWITCH_H_
