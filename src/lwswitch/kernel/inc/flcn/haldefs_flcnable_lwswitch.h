/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _HALDEFS_FLCNABLE_LWSWITCH_H_
#define _HALDEFS_FLCNABLE_LWSWITCH_H_

#include "lwstatus.h"
#include "flcnifcmn.h"

struct lwswitch_device;
struct LWSWITCH_TIMEOUT;
struct FLCNABLE;
struct FALCON_EXTERNAL_CONFIG;
struct FLCN_QMGR_SEQ_INFO;
union RM_FLCN_MSG;
union RM_FLCN_CMD;
struct ENGINE_DESCRIPTOR_TYPE;

typedef struct {
    LwU8                        (*readCoreRev)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable);
    void                        (*getExternalConfig)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        struct FALCON_EXTERNAL_CONFIG  *pConfig);
    void                        (*ememCopyFrom)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        LwU32                           src,
                                        LwU8                           *pDst,
                                        LwU32                           sizeBytes,
                                        LwU8                            port);
    void                        (*ememCopyTo)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        LwU32                           dst,
                                        LwU8                           *pSrc,
                                        LwU32                           sizeBytes,
                                        LwU8                            port);
    LW_STATUS                   (*handleInitEvent)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        union RM_FLCN_MSG              *pGenMsg);
    struct FLCN_QMGR_SEQ_INFO*  (*queueSeqInfoGet)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        LwU32                           seqIndex);
    void                        (*queueSeqInfoClear)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        struct FLCN_QMGR_SEQ_INFO      *pSeqInfo);
    void                        (*queueSeqInfoFree)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        struct FLCN_QMGR_SEQ_INFO      *pSeqInfo);
    LwBool                      (*queueCmdValidate)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        union RM_FLCN_CMD              *pCmd,
                                        union RM_FLCN_MSG              *pMsg,
                                        void                           *pPayload,
                                        LwU32                           queueIdLogical);
    LW_STATUS                   (*queueCmdPostExtension)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        union RM_FLCN_CMD              *pCmd,
                                        union RM_FLCN_MSG              *pMsg,
                                        void                           *pPayload,
                                        struct LWSWITCH_TIMEOUT        *pTimeout,
                                        struct FLCN_QMGR_SEQ_INFO      *pSeqInfo);
    void                        (*postDiscoveryInit)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable);



    LW_STATUS                   (*construct)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable);
    void                        (*destruct)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable);
    void                        (*fetchEngines)(
                                        struct lwswitch_device         *device,
                                        struct FLCNABLE                *pFlcnable,
                                        struct ENGINE_DESCRIPTOR_TYPE  *pEngDeslwc,
                                        struct ENGINE_DESCRIPTOR_TYPE  *pEngDescBc);

} flcnable_hal;

#endif //_HALDEFS_FLCNABLE_LWSWITCH_H_
