/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_FIELD_H__
#define __DRF_FIELD_H__

#include <drf_types.h>
#include "drf.h"
#include "drf_define.h"

typedef struct {
    uint32_t msb, lsb;
    drf_define_t **defines;
    uint32_t n_defines;
    char name[1];
} drf_field_t;

#endif /* __DRF_FIELD_H__ */
