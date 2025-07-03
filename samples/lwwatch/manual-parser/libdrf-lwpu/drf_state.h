/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_STATE_H__
#define __DRF_STATE_H__

#include "drf_mem_buffer.h"
#include "drf_device.h"

typedef struct {
    drf_mem_buffer_t *mem_buffers;
    char **manuals, **addendums;
    drf_device_t **devices, *lwrrent_device;
    uint32_t n_devices;
    int debug;
} __drf_state_t;

#endif /* __DRF_STATE_H__ */
