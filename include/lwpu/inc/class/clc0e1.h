/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _clc0e1_h_
#define _clc0e1_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GP100_SUBDEVICE_FB                                       (0x0000c0e1)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* 0x0000 - 0x0fff */
} LWC0E1MapTypedef, GP100SubdeviceFBMap;
#define  LWC0E1_TYPEDEF                                            GP100SubdeviceFBMap

/* GP100 */
#define FB_GP100_ECC_PARTITION_COUNT         (0x00000010)
#define FB_GP100_ECC_SLICE_COUNT             (0x00000002)
#define FB_GP100_ECC_SUBPARTITION_COUNT      (0x00000002)
/* GP10B */
#define FB_GP10B_ECC_PARTITION_COUNT         (0x00000002)
#define FB_GP10B_ECC_SLICE_COUNT             (0x00000002)
#define FB_GP10B_ECC_SUBPARTITION_COUNT      (0x00000002)
/* GP102 */
#define FB_GP102_ECC_PARTITION_COUNT         (0x0000000c)
#define FB_GP102_ECC_SLICE_COUNT             (0x00000002)
#define FB_GP102_ECC_SUBPARTITION_COUNT      (0x00000002)
/* GP104 */
#define FB_GP104_ECC_PARTITION_COUNT         (0x00000008)
#define FB_GP104_ECC_SLICE_COUNT             (0x00000002)
#define FB_GP104_ECC_SUBPARTITION_COUNT      (0x00000002)
/* GP106, GP107, and GP108 do not have FB ECC enabled*/
#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc0e1_h_ */
