/*
 * Copyright (c) 2019-2022, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscisync_primitive_core.h"

/**
 * \brief Supported primitive types
 */
const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreSupportedPrimitives[MAX_PRIMITIVE_TYPE] = {
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
};

/**
 * \brief Supported C2C CPU primitive types
 */
const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreSupportedC2cCpuPrimitives[MAX_PRIMITIVE_TYPE] = {
    LwSciSyncInternalAttrValPrimitiveType_Syncpoint,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
};

/**
 * \brief Supported deterministic primitive types
 */
const LwSciSyncInternalAttrValPrimitiveType
LwSciSyncCoreDeterministicPrimitives[MAX_PRIMITIVE_TYPE] = {
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphore,
    LwSciSyncInternalAttrValPrimitiveType_SysmemSemaphorePayload64b,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
    LwSciSyncInternalAttrValPrimitiveType_LowerBound,
};
