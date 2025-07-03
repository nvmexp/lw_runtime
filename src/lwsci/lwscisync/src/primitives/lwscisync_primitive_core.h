/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciSync generic primitive private structures</b>
 *
 * @b Description: This file declares core structures of primitive unit
 */

#ifndef INCLUDED_LWSCISYNC_PRIMITIVE_CORE_H
#define INCLUDED_LWSCISYNC_PRIMITIVE_CORE_H

#include "lwscisync_core.h"
#include "lwscisync_internal.h"
#include "lwscisync_primitive_type.h"

#include "lwscisync_c2c_priv.h"

typedef struct
{
    LwSciError(* Init)(
        LwSciSyncAttrList reconciledList,
        LwSciSyncCorePrimitive primitive);
    void(* Deinit)(
        LwSciSyncCorePrimitive primitive);
    LwSciError(* Export)(
        LwSciSyncCorePrimitive primitive,
        LwSciSyncAccessPerm permissions,
        LwSciIpcEndpoint ipcEndpoint,
        void** data,
        size_t* len);
    LwSciError(* Import)(
        LwSciIpcEndpoint ipcEndpoint,
        LwSciSyncAttrList reconciledList,
        const void* data,
        size_t len,
        LwSciSyncCorePrimitive primitive);
    LwSciError(* Signal)(
        LwSciSyncCorePrimitive primitive);
    LwSciError(* WaitOn)(
        LwSciSyncCorePrimitive primitive,
        LwSciSyncCpuWaitContext waitContext,
        uint64_t id,
        uint64_t value,
        int64_t timeout_us);
    uint64_t(* GetNewFence)(
        LwSciSyncCorePrimitive primitive);
    void (* GetSpecificData)(
        LwSciSyncCorePrimitive primitive,
        void** data);
    LwSciError(* CheckIdValue)(
        LwSciSyncCorePrimitive primitive,
        uint64_t id,
        uint64_t value);
#if (LW_IS_SAFETY == 0) && (LW_L4T == 0)
    LwSciError(* GetC2cSyncHandle)(
        LwSciSyncCorePrimitive primitive,
        LwSciC2cPcieSyncHandle* syncHandle);
    LwSciError(* GetC2cRmHandle)(
        LwSciSyncCorePrimitive primitive,
        LwSciC2cPcieSyncRmHandle* syncRmHandle);
#endif
    LwSciError(* ImportThreshold)(
        LwSciSyncCorePrimitive primitive,
        uint64_t* threshold);
} LwSciSyncPrimitiveOps;

extern const LwSciSyncPrimitiveOps LwSciSyncBackEndSyncpoint;
extern const LwSciSyncPrimitiveOps LwSciSyncBackEndSysmemSema;
extern const LwSciSyncPrimitiveOps LwSciSyncBackEndSysmemSemaPayload64b;

extern const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreSupportedPrimitives[MAX_PRIMITIVE_TYPE];

extern const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreSupportedC2cCpuPrimitives[MAX_PRIMITIVE_TYPE];

extern const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreDeterministicPrimitives[MAX_PRIMITIVE_TYPE];

/**
 * \brief Represents LwSciSync core primitive
 */
struct LwSciSyncCorePrimitiveRec {
    /** underlying primitive type */
    LwSciSyncInternalAttrValPrimitiveType type;
    /** primitive's identifier */
    uint64_t id;
    /** last generated fence value */
    uint64_t lastFence;
    /** Operations of primitive selected */
    const LwSciSyncPrimitiveOps* ops;
    /** Data specific to actual primitive */
    void* specificData;
    /** true if owns a primitive */
    bool ownsPrimitive;
#ifdef LWSCISYNC_EMU_SUPPORT
    /** true if has external primitive info */
    bool hasExternalPrimitiveInfo;
#endif
};

#endif
