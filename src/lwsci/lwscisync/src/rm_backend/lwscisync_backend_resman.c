/*
 * Copyright (c) 2019-2021, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwscisync_backend.h"

LwSciError LwSciSyncCoreRmAlloc(
    LwSciSyncCoreRmBackEnd* backEnd)
{
    (void)backEnd;

    return LwSciError_Success;
}

void LwSciSyncCoreRmFree(
    LwSciSyncCoreRmBackEnd backEnd)
{
    (void)backEnd;

    return;
}

LwSciError LwSciSyncCoreRmWaitCtxBackEndAlloc(
    LwSciSyncCoreRmBackEnd rmBackEnd,
    LwSciSyncCoreRmWaitContextBackEnd* waitContextBackEnd)
{
    (void)rmBackEnd;
    (void)waitContextBackEnd;

    return LwSciError_Success;
}

void LwSciSyncCoreRmWaitCtxBackEndFree(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd)
{
    (void)waitContextBackEnd;

    return;
}

LwSciError LwSciSyncCoreRmWaitCtxBackEndValidate(
    LwSciSyncCoreRmWaitContextBackEnd waitContextBackEnd)
{
    (void)waitContextBackEnd;

    return LwSciError_Success;
}
