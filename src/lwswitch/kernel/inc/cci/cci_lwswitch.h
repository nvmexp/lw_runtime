/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021  by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _CCI_LWSWITCH_H_
#define _CCI_LWSWITCH_H_

#include "lwlink_errors.h"
#include "lwtypes.h"
#include "lwfixedtypes.h"
#include "lwstatus.h"
#include "ctrl_dev_lwswitch.h"
#include "ctrl_dev_internal_lwswitch.h"
#include "export_lwswitch.h"

typedef struct CCI CCI, *PCCI;
struct lwswitch_device;

// Polling Callback ids
#define LWSWITCH_CCI_CALLBACK_SMBPBI        0
#define LWSWITCH_CCI_CALLBACK_LED_UPDATE    1
#define LWSWITCH_CCI_CALLBACK_NUM_MAX       2

//
// Determines the range of frequencies that functions can
// run at.
// This rate must be divisible by client provided frequencies.
//
#define LWSWITCH_CCI_POLLING_RATE_HZ  5

//
// Mapping between XCVR module, lwlink-id, xvcr lane-mask
//
typedef struct lwswitch_cci_module_link_lane_map
{
    LwU8 moduleId;
    LwU8 linkId;
    LwU8 laneMask;
} LWSWITCH_CCI_MODULE_LINK_LANE_MAP;


CCI*      cciAllocNew(void);
LwBool    cciSupported(lwswitch_device *device);
LwlStatus cciInit(lwswitch_device *device, PCCI pCci, LwU32 pci_device_id);
LwlStatus cciLoad(lwswitch_device *device);
void      cciDestroy(lwswitch_device *device, PCCI pCci);
LwlStatus cciResetAllPartitions(struct lwswitch_device *device);
LwlStatus cciResetLinks(struct lwswitch_device *device, LwU64 linkMask);
LwlStatus cciGetLinkPartners(struct lwswitch_device *device, LwU8 linkId, LwU64* pLinkMask);
LwlStatus cciRead(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 addr, LwU32 length, LwU8 *pVal);
LwlStatus cciReadBlk(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 addr, LwU32 length, LwU8 *pValArray);
LwlStatus cciWrite(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 addr, LwU32 length, LwU8 *pVal);
LwlStatus cciWriteBlk(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 addr, LwU32 length, LwU8 *pValArray);
LwlStatus cciSetBankAndPage(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 bank, LwU8 page);
LwlStatus cciSendCDBCommandAndGetResponse(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 command, LwU32 payLength, LwU8 *payload, LwU32 *resLength, LwU8 *response, LwBool padding);
LwlStatus cciSendCDBCommand(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU32 command, LwU32 length, LwU8 *pValArray, LwBool padding);
LwlStatus cciGetBankAndPage(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 *pBank, LwU8 *pPage);
LwlStatus cciGetCDBResponse(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 *response, LwU32 *resLength);
LwlStatus cciGetCDBStatus(lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 *pStatus);
LwlStatus cciWaitForCDBComplete(lwswitch_device *device, LwU32 client, LwU32 osfp);
LwlStatus cciRegisterCallback(lwswitch_device *device, LwU32 callbackId, void (*functionPtr)(lwswitch_device *device), LwU32 rateHz);

LwU32       cciService               (lwswitch_device *device, PCCI pCci);
LwBool      cciIsLinkManaged         (lwswitch_device *device, LwU32 linkNumber);
LwlStatus   cciGetTemperature        (lwswitch_device *device, LwU32 client, LwU32 linkId, LwTemp *pTemperature);
LwlStatus   cciGetXcvrTemperature    (lwswitch_device *device, LwU32 client, LwU32 osfp, LwTemp *pTemperature);
LwlStatus   cciGetCapabilities       (lwswitch_device *device, LwU32 client, LwU32 linkId, LWSWITCH_CCI_CAPABILITIES *pCapabilities);
LwlStatus   cciGetFWRevisions        (lwswitch_device *device, LwU32 client, LwU32 linkId, LWSWITCH_CCI_GET_FW_REVISIONS *pRevisions);
LwlStatus   cciGetXcvrFWRevisions    (lwswitch_device *device, LwU32 client, LwU32 osfp, LWSWITCH_CCI_GET_FW_REVISIONS *pRevisions);
void        cciDetectXcvrsPresent    (lwswitch_device *device);
LwlStatus   cciGetXcvrMask           (lwswitch_device *device, LwU32 *pMaskAll, LwU32 *pMaskPresent);
LwlStatus   cciSetXcvrPresent        (lwswitch_device *device, LwU32 osfp, LwBool present);
LwlStatus   cciGetXcvrStaticIdInfo   (lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 *pSerial, LwU8 *pPart, LwU8 *pHwRev, LwU8 **ppFru);
LwlStatus   cciGetXcvrLedState       (lwswitch_device *device, LwU32 client, LwU32 osfp, LwU8 *pLedState);
LwlStatus   cciSetXcvrLedState       (lwswitch_device *device, LwU32 client, LwU32 osfp, LwBool bSetLocate);
LwlStatus   cciRomCache              (lwswitch_device *device, LwU32 client);
LwlStatus   cciSetupLanes            (lwswitch_device *device, LwU32 client, LwBool bTx, LwBool bEnable);
LwlStatus   cciOpticalPretrain       (lwswitch_device *device);
LwlStatus   cciPollRxCdrLock         (lwswitch_device *device, LwU32 client);
LwlStatus   cciConfigureLwlinkMode   (lwswitch_device *device, LwU32 client, LwU32 linkId, LwBool bTx, LwBool freeze_maintenance, LwBool restart_training, LwBool lwlink_mode);
LwlStatus   cciPollForPreTraining    (lwswitch_device *device, LwU32 client, LwU32 linkId, LwBool bTx);
LwlStatus   cciApplyControlSetValues (lwswitch_device *device, LwU32 client, LwU32 moduleMask);
LwlStatus   cciGetGradingValues      (lwswitch_device *device, LwU32 client, LwU32 linkId, LwU8 *laneMask, LWSWITCH_CCI_GRADING_VALUES *pGrading);
LwlStatus   cciGetModuleState        (lwswitch_device *device, LwU32 client, LwU32 linkId, LWSWITCH_CCI_MODULE_STATE *pInfo);
LwlStatus   cciGetModuleFlags        (lwswitch_device *device, LwU32 client, LwU32 linkId, LWSWITCH_CCI_MODULE_FLAGS *pModuleFlags);
LwlStatus   cciGetVoltage            (lwswitch_device *device, LwU32 client, LwU32 linkId, LWSWITCH_CCI_VOLTAGE *pVoltage);
LwlStatus   cciGetCageMapping        (lwswitch_device *device, LwU8 cageIndex, LwU64 *pLinkMask, LwU64 *pEncodedValue);
LwlStatus   cciCmisRead              (lwswitch_device *device, LwU8 cageIndex, LwU8 bank, LwU8 page, LwU8 address, LwU8 count, LwU8 *pData);
LwlStatus   cciCmisWrite             (lwswitch_device *device, LwU8 cageIndex, LwU8 bank, LwU8 page, LwU8 address, LwU8 count, LwU8 *pData);
void        cciPrintGradingValues    (lwswitch_device *device, LwU32 client, LwU32 linkId);
LwlStatus   cciCmisCageBezelMarking  (lwswitch_device *device, LwU8 cageIndex, char *pBezelMarking);

// CCI Control calls
LwlStatus   lwswitch_ctrl_get_cci_capabilities     (lwswitch_device *device, LWSWITCH_CCI_GET_CAPABILITIES_PARAMS *pParams);
LwlStatus   lwswitch_ctrl_get_cci_temperature      (lwswitch_device *device, LWSWITCH_CCI_GET_TEMPERATURE  *pParams);
LwlStatus   lwswitch_ctrl_get_cci_fw_revisions     (lwswitch_device *device, LWSWITCH_CCI_GET_FW_REVISION_PARAMS *pParams);
LwlStatus   lwswitch_ctrl_get_grading_values       (lwswitch_device *device, LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS *pParams);
LwlStatus   lwswitch_ctrl_get_module_state         (lwswitch_device *device, LWSWITCH_CCI_GET_MODULE_STATE *pParams);
LwlStatus   lwswitch_ctrl_get_module_flags         (lwswitch_device *device, LWSWITCH_CCI_GET_MODULE_FLAGS *pParams);
LwlStatus   lwswitch_ctrl_get_voltage              (lwswitch_device *device, LWSWITCH_CCI_GET_VOLTAGE *pParams);

#endif //_CCI_LWSWITCH_H_
