/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_INDEX_H__
#define __DRF_INDEX_H__

#include "drf.h"
#include "drf_device.h"

int drf_index_get_devices(drf_state_t *state, drf_device_t ***devices,
        uint32_t *n_devices);

#endif /* __DRF_INDEX_H__ */
