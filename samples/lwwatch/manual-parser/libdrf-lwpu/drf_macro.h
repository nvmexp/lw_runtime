/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef __DRF_MACRO_H__
#define __DRF_MACRO_H__

#include <drf_types.h>
#include "uthash.h"

typedef enum {
    MACRO_TYPE_ZERO_LENGTH = 0,
    MACRO_TYPE_CONSTANT,
    MACRO_TYPE_RANGE,
    MACRO_TYPE_UNKNOWN_OTHER
} drf_macro_type;

#define DRF_MACRO_FLAGS_ALIAS (1 << 0)

typedef struct drf_macro {
    UT_hash_handle hh;
    struct drf_macro *next, *overlay;
    const char *fname;
    uint8_t flags, n_args;
    drf_macro_type macro_type;
    uint32_t a, b;
    char name[1];
} drf_macro_t;

#define DRF_MACRO_MAX_N_ARGS 8
#define DRF_MACRO_MAX_N_ARGS_DIMENSION_SIZE 1024

#endif /* __DRF_MACRO_H__ */
