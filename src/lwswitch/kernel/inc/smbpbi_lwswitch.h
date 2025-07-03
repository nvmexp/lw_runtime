/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021  by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SMBPBI_LWSWITCH_H_
#define _SMBPBI_LWSWITCH_H_

#include "soe/soeifsmbpbi.h"
#include "smbpbi_shared_lwswitch.h"
#include "oob/smbpbi_priv.h"

typedef struct
{
    LwBool isValid;
    LwU64  attemptedTrainingMask0;
    LwU64  trainingErrorMask0;
} LWSWITCH_LINK_TRAINING_ERROR_INFO;

typedef struct
{
    LwBool isValid;
    LwU64  mask0;
} LWSWITCH_LINK_RUNTIME_ERROR_INFO;

struct smbpbi
{
    SOE_SMBPBI_SHARED_SURFACE       *sharedSurface;
    LwU64                           dmaHandle;
};

LwlStatus lwswitch_smbpbi_init(lwswitch_device *);
LwlStatus lwswitch_smbpbi_post_init(lwswitch_device *);
LwlStatus lwswitch_smbpbi_set_link_error_info(lwswitch_device *,
                                              LWSWITCH_LINK_TRAINING_ERROR_INFO *pLinkTrainingErrorInfo,
                                              LWSWITCH_LINK_RUNTIME_ERROR_INFO  *pLinkRuntimeError);
void lwswitch_smbpbi_unload(lwswitch_device *);
void lwswitch_smbpbi_destroy(lwswitch_device *);
LwlStatus lwswitch_smbpbi_refresh_ecc_counts(lwswitch_device *);
void lwswitch_smbpbi_log_message(lwswitch_device *device, LwU32 num, LwU32 msglen, LwU8 *osErrorString);

#endif //_SMBPBI_LWSWITCH_H_
