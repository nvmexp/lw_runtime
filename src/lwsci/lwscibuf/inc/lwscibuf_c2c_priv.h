/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

/* TODO: If LwSciC2c abstracts the header file in future for PCIe vs NPM then
 * we need to include that here.
 */

#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
#include "lwscibuf_obj_mgmt.h"

#define LWSCIBUF_C2C_SOURCE_HANDLE_MAGIC (0xFACEB11EU)
#define LWSCIBUF_C2C_TARGET_HANDLE_MAGIC (0x1834B14EU)

typedef union {
    LwSciC2cPcieBufSourceHandle pcieSourceHandle;
} LwSciC2cInterfaceSourceHandle;

typedef struct LwSciC2cBufSourceHandleRec {
    uint64_t magic;
    LwSciC2cHandle channelHandle;
    LwSciBufObj bufObj;
    LwSciC2cInterfaceSourceHandle interfaceSourceHandle;
} LwSciC2cBufSourceHandlePriv;

typedef struct LwSciC2cBufTargetHandleRec {
    uint64_t magic;
    LwSciC2cHandle channelHandle;
    LwSciBufObj bufObj;
    LwSciC2cInterfaceTargetHandle interfaceTargetHandle;
} LwSciC2cBufTargetHandlePriv;

#endif
