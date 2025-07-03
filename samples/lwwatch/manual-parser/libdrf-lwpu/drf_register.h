/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_REGISTER_H__
#define __DRF_REGISTER_H__

#include <drf_types.h>
#include "drf.h"
#include "drf_field.h"

typedef struct {
    uint32_t address;
    drf_field_t **fields;
    uint32_t n_fields;
    char name[1];
} drf_register_t;

#endif /* __DRF_REGISTER_H__ */
