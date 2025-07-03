/*
 * Copyright (c) 2019-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwscibuf_utils_x86.h"
#include "lwscilog.h"

LwSciError LwRmShimErrorToLwSciError(
    LwRmShimError shimErr)
{
    LwSciError sciErr = LwSciError_Unknown;
    LwSciError rmShimToSciMap [] = {
        [LWRMSHIM_OK] = LwSciError_Success,
        [LWRMSHIM_ERR_GENERIC] = LwSciError_LwSciBufUnknown,
        [LWRMSHIM_ERR_ILLEGAL_ACTION] = LwSciError_IlwalidOperation,
        [LWRMSHIM_ERR_INSUFFICIENT_RESOURCES] = LwSciError_InsufficientResource,
        [LWRMSHIM_ERR_INSUFFICIENT_PERMISSIONS] = LwSciError_NotPermitted,
        [LWRMSHIM_ERR_ILWALID_ACCESS_TYPE] = LwSciError_AccessDenied,
        [LWRMSHIM_ERR_ILWALID_ADDRESS] = LwSciError_BadAddress,
        [LWRMSHIM_ERR_ILWALID_ARGUMENT] = LwSciError_BadParameter,
        [LWRMSHIM_ERR_ILWALID_DATA] = LwSciError_BadParameter,
        [LWRMSHIM_ERR_ILWALID_DEVICE] = LwSciError_NoSuchDevice,
        [LWRMSHIM_ERR_ILWALID_FUNCTION] = LwSciError_IlwalidOperation,
        [LWRMSHIM_ERR_ILWALID_INDEX] = LwSciError_BadParameter,
        [LWRMSHIM_ERR_ILWALID_OPERATION] = LwSciError_IlwalidOperation,
        [LWRMSHIM_ERR_ILWALID_POINTER] = LwSciError_BadAddress,
        [LWRMSHIM_ERR_ILWALID_STATE] = LwSciError_IlwalidState,
        [LWRMSHIM_ERR_NO_MEMORY] = LwSciError_InsufficientMemory,
        [LWRMSHIM_ERR_NOT_COMPATIBLE] = LwSciError_NotSupported,
        [LWRMSHIM_ERR_NOT_SUPPORTED] = LwSciError_NotSupported,
        [LWRMSHIM_ERR_OPERATION_FAILED] = LwSciError_IlwalidOperation,
        [LWRMSHIM_ERR_ILWALID] = LwSciError_Unknown,
    };

    LWSCI_FNENTRY("");

    if (shimErr > LWRMSHIM_ERR_MAX) {
        goto ret;
    }

    sciErr = rmShimToSciMap[shimErr];

ret:
    LWSCI_FNEXIT("");
    return sciErr;
}


