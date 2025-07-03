/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _FLCNABLE_LWSWITCH_H_
#define _FLCNABLE_LWSWITCH_H_

/*!
 * @file   flcnable_lwswitch.h
 * @brief  Provides definitions for all FLCNABLE data structures and interfaces.
 */

#include "flcn/haldefs_flcnable_lwswitch.h"

#include "flcnifcmn.h"

#include "lwlink_errors.h"

struct lwswitch_device;
struct FLCN;
struct FALCON_EXTERNAL_CONFIG;
struct FLCN_QMGR_SEQ_INFO;
union RM_FLCN_MSG;
union RM_FLCN_CMD;
struct ENGINE_DESCRIPTOR_TYPE;

/*!
 * Defines the structure used to contain all generic information related to
 * the FLCNABLE.
 */
typedef struct FLCNABLE
{
    // pointer to our function table - should always be the first thing in any object
    flcnable_hal *pHal;
    // we don't have a parent class, so we go straight to our members

    /* Pointer to FLCN object for the object represented by this FLCNABLE */
    struct FLCN *pFlcn;

} FLCNABLE, *PFLCNABLE;

LwlStatus flcnableInit(struct lwswitch_device *device, PFLCNABLE pFlcnable, LwU32 pci_device_id);
void flcnableDestroy(struct lwswitch_device *device, PFLCNABLE pFlcnable);

/*!
 * Utility to get the FLCN object for the engine
 */
#define ENG_GET_FLCN(pObj) (((PFLCNABLE)pObj)->pFlcn)

/*!
 * Safe (from NULL parent) version of utility to get the FLCN object for the engine
 */
#define ENG_GET_FLCN_IFF(pObj) ((NULL!=(pObj))?ENG_GET_FLCN(pObj):NULL)

// hal functions
LwU8                        flcnableReadCoreRev                         (struct lwswitch_device *device, PFLCNABLE);
void                        flcnableGetExternalConfig                   (struct lwswitch_device *device, PFLCNABLE, struct FALCON_EXTERNAL_CONFIG *);
void                        flcnableEmemCopyFrom                        (struct lwswitch_device *device, PFLCNABLE, LwU32, LwU8 *, LwU32, LwU8);
void                        flcnableEmemCopyTo                          (struct lwswitch_device *device, PFLCNABLE, LwU32, LwU8 *, LwU32, LwU8);
LW_STATUS                   flcnableHandleInitEvent                     (struct lwswitch_device *device, PFLCNABLE, union RM_FLCN_MSG *);
struct FLCN_QMGR_SEQ_INFO * flcnableQueueSeqInfoGet                     (struct lwswitch_device *device, PFLCNABLE, LwU32);
void                        flcnableQueueSeqInfoClear                   (struct lwswitch_device *device, PFLCNABLE, struct FLCN_QMGR_SEQ_INFO *);
void                        flcnableQueueSeqInfoFree                    (struct lwswitch_device *device, PFLCNABLE, struct FLCN_QMGR_SEQ_INFO *);
LwBool                      flcnableQueueCmdValidate                    (struct lwswitch_device *device, PFLCNABLE, union RM_FLCN_CMD *, union RM_FLCN_MSG *, void *, LwU32);
LW_STATUS                   flcnableQueueCmdPostExtension               (struct lwswitch_device *device, PFLCNABLE, union RM_FLCN_CMD *, union RM_FLCN_MSG *, void *, struct LWSWITCH_TIMEOUT *, struct FLCN_QMGR_SEQ_INFO *);
void                        flcnablePostDiscoveryInit                   (struct lwswitch_device *device, PFLCNABLE);

LW_STATUS                   flcnableConstruct_HAL                       (struct lwswitch_device *device, PFLCNABLE);
void                        flcnableDestruct_HAL                        (struct lwswitch_device *device, PFLCNABLE);

void                        flcnableFetchEngines_HAL                    (struct lwswitch_device *device, PFLCNABLE, struct ENGINE_DESCRIPTOR_TYPE *, struct ENGINE_DESCRIPTOR_TYPE *);


#endif // _FLCNABLE_LWSWITCH_H_
