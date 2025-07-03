/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _CLC301_H_
#define _CLC301_H_

#define LWC301_NPU_RESOURCE 0xc301

/* LwRmAlloc parameters */
typedef struct {
    LwU64 size LW_ALIGN_BYTES(8);   /* Returns resource size on successful allocation*/
} LWC301_ALLOCATION_PARAMETERS;

#endif // _CLC301_H_
