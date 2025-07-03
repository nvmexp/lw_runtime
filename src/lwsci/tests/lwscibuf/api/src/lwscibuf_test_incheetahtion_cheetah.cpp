/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscibuf_test_integration.h"

using namespace std;
LwU64 GetMemorySize(LwSciBufRmHandle rmhandle)
{
    LwError lwErr = LwError_Success;
    LwRmMemHandleParams params;
    LwU64 size = 0;

    lwErr = LwRmMemQueryHandleParams(rmhandle.memHandle, rmhandle.memHandle, &params,
                sizeof(params));
    if (lwErr != LwSuccess) {
        printf("LwRmMemQueryHandleParams failed\n");
        goto ret;
    }

    size = params.Size;

ret:
    return size;
}

LwU32 GetLwRmAccessFlags(
    LwSciBufAttrValAccessPerm perm)
{
    switch (perm) {
    case LwSciBufAccessPerm_Readonly:
        return (LWOS_MEM_READ);
    case LwSciBufAccessPerm_ReadWrite:
        return (LWOS_MEM_READ_WRITE);
    default:
        return (LWOS_MEM_NONE);
    }
}

bool CheckBufferAccessFlags(
    LwSciBufObj bufObj,
    LwSciBufRmHandle rmHandle)
{
    bool status = false;

    LwSciBufAttrList reconciledAttrList;
    LwRmMemHandleParams params;
    LwSciBufAttrValAccessPerm accessPerms;
    LwSciBufAttrKeyValuePair keyValuePair = {
        LwSciBufGeneralAttrKey_ActualPerm,
    };

    if (LwSciBufObjGetAttrList(bufObj, &reconciledAttrList) !=
              LwSciError_Success) {
        goto ret;
    }

    if (LwSciBufAttrListGetAttrs(reconciledAttrList, &keyValuePair, 1) !=
           LwSciError_Success) {
        goto ret;
    }

    accessPerms = *(const LwSciBufAttrValAccessPerm *)keyValuePair.value;

    LwRmMemQueryHandleParams(rmHandle.memHandle, 0, &params, sizeof(params));

    if (params.AccessFlags != GetLwRmAccessFlags(accessPerms))  {
        goto ret;
    }
    status = true;

ret:
    return status;
}

bool CompareRmHandlesAccessPermissions(
    LwSciBufRmHandle rmHandle1,
    LwSciBufRmHandle rmHandle2)
{
    bool status = false;

    LwRmMemHandleParams params1;
    LwRmMemHandleParams params2;

    LwRmMemQueryHandleParams(rmHandle1.memHandle, 0, &params1, sizeof(params1));
    LwRmMemQueryHandleParams(rmHandle2.memHandle, 0, &params2, sizeof(params2));

    if (params1.AccessFlags != params2.AccessFlags)  {
        goto ret;
    }
    status = true;

ret:
    return status;
}

// TODO Is there a better way to check is RM handle is freed?
bool isRMHandleFree(LwSciBufRmHandle rmHandle)
{
    bool ret = true;
    LwRmMemHandle dummyHandle;
    if (LwRmMemHandleDuplicate(rmHandle.memHandle, LWOS_MEM_READ_WRITE, &dummyHandle) == LwError_Success) {
        ret = false;
    }
    return ret;
}
