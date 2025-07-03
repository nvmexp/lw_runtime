/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_DEVICE_H__
#define __DRF_DEVICE_H__

#include <drf_types.h>
#include "drf.h"
#include "drf_register.h"

typedef struct drf_device {
    const char *fname;
    uint32_t initial_base, initial_extent;
    uint32_t base, extent;
    drf_register_t **registers;
    uint32_t n_registers;
    char name[1];
} drf_device_t;

#define DRF_DEVICE_MATCH_INITIAL(device, address) \
    (((address) >= (device)->initial_base) && ((address) < (device)->initial_extent))

#define DRF_DEVICE_MATCH(device, address) \
    (((address) >= (device)->base) && ((address) < (device)->extent))

#endif /* __DRF_DEVICE_H__ */
