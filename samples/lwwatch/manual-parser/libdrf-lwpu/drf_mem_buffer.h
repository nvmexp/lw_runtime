/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_MEM_BUFFER_H__
#define __DRF_MEM_BUFFER_H__

#include <drf_types.h>

typedef struct {
    const char *data;
    uint32_t data_size;
} drf_mem_buffer_t;

#endif /* __DRF_MEM_BUFFER_H__ */
