/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOE_LWSWITCH_H_
#define _SOE_LWSWITCH_H_

#include "lwlink_errors.h"
#include "lwtypes.h"
#include "lwstatus.h"

typedef struct SOE SOE, *PSOE;
struct FLCNABLE;
struct lwswitch_device;

SOE *soeAllocNew(void);
LwlStatus soeInit(struct lwswitch_device *device, PSOE pSoe, LwU32 pci_device_id);
void soeDestroy(struct lwswitch_device *device, PSOE pSoe);

//HAL functions
LW_STATUS   soeProcessMessages          (struct lwswitch_device *device, PSOE pSoe);
LW_STATUS   soeWaitForInitAck           (struct lwswitch_device *device, PSOE pSoe);


LwU32       soeService_HAL              (struct lwswitch_device *device, PSOE pSoe);
void        soeServiceHalt_HAL          (struct lwswitch_device *device, PSOE pSoe);
void        soeEmemTransfer_HAL         (struct lwswitch_device *device, PSOE pSoe, LwU32 dmemAddr, LwU8 *pBuf, LwU32 sizeBytes, LwU8 port, LwBool bCopyFrom);
LwU32       soeGetEmemSize_HAL          (struct lwswitch_device *device, PSOE pSoe);
LwU32       soeGetEmemStartOffset_HAL   (struct lwswitch_device *device, PSOE pSoe);
LW_STATUS   soeEmemPortToRegAddr_HAL    (struct lwswitch_device *device, PSOE pSoe, LwU32 port, LwU32 *pEmemCAddr, LwU32 *pEmemDAddr);
void        soeServiceExterr_HAL        (struct lwswitch_device *device, PSOE pSoe);
LW_STATUS   soeGetExtErrRegAddrs_HAL    (struct lwswitch_device *device, PSOE pSoe, LwU32 *pExtErrAddr, LwU32 *pExtErrStat);
LwU32       soeEmemPortSizeGet_HAL      (struct lwswitch_device *device, PSOE pSoe);
LwBool      soeIsCpuHalted_HAL          (struct lwswitch_device *device, PSOE pSoe);
LwlStatus   soeTestDma_HAL              (struct lwswitch_device *device, PSOE pSoe);
LwlStatus   soeSetPexEOM_HAL            (struct lwswitch_device *device, LwU8 mode, LwU8 nblks, LwU8 nerrs, LwU8 berEyeSel);
LwlStatus   soeGetPexEomStatus_HAL      (struct lwswitch_device *device, LwU8 mode, LwU8 nblks, LwU8 nerrs, LwU8 berEyeSel, LwU32 laneMask, LwU16  *pEomStatus);
LwlStatus   soeGetUphyDlnCfgSpace_HAL   (struct lwswitch_device *device, LwU32 regAddress, LwU32 laneSelectMask, LwU16 *pRegValue);
LwlStatus   soeForceThermalSlowdown_HAL (struct lwswitch_device *device, LwBool slowdown, LwU32  periodUs);
LwlStatus   soeSetPcieLinkSpeed_HAL     (struct lwswitch_device *device, LwU32 linkSpeed);

#endif //_SOE_LWSWITCH_H_
