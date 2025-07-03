/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _cl00db_h_
#define _cl00db_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LW40_DEBUG_BUFFER                               (0x000000db)

/* LwRmAlloc() parameters */
typedef struct {
    LwU32 size; /* Desired message size / actual size returned */
    LwU32 tag; /* Protobuf tag for message location in dump message */
} LW00DB_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl00db_h_ */
