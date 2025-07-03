/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef EGL_TESTCLIENT_STREAM2_H
#define EGL_TESTCLIENT_STREAM2_H

#if defined(LW_EGL_DESKTOP_COMPATIBLE_HEADERS)
// For mobile base types and lwos/lwutil functionality on desktop builds
#include "mobile_common.h"
#endif

#include "defs.h"
#include "eglapiinterface.h"

#ifdef __cplusplus
extern "C" {
#endif

// Consumer

bool TestExpStream2ConsumerReserve(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiClientStream* handle);

bool TestExpStream2ConsumerGetElw(
    LwEglApiClientStream handle,
    LwEglApiStream2Elw* elw);

bool TestExpStream2ConsumerConnect(
    LwEglApiClientStream stream,
    LwEglApiStream2ConsumerCaps *consumerCaps);

bool TestExpStream2ConsumerGetCaps(LwEglApiClientStream stream,
                                   LwEglApiStream2ConsumerCaps *caps);

bool TestExpStream2ConsumerBufferObtain(
    LwEglApiClientStream handle,
    LwEglApiBufferHandle* image);

bool TestExpStream2ConsumerBufferNotify(
    LwEglApiClientStream handle,
    LwEglApiBufferHandle image,
    LwError err,
    LwEglApiClientBuffer buffer);

bool TestExpStream2ConsumerFrameAcquire(
    LwEglApiClientStream           handle,
    LwS64                          usec,
    LwEglApiStream2Frame*          frame);

bool TestExpStream2ConsumerFrameRelease(
    LwEglApiClientStream           handle,
    LwEglApiStream2Frame*          frame);

bool TestExpStream2ConsumerMetaDataGet(
    LwEglApiClientStream           handle,
    LwEglApiClientBuffer           buffer,
    LwU32                          index,
    LwU32                          offset,
    LwU32                          size,
    void*                          data);

bool TestExpStream2ConsumerDisconnect(
    LwEglApiClientStream handle);

// Producer

bool TestExpStream2ProducerReserve(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiClientStream* handle);

bool TestExpStream2ProducerGetElw(
    LwEglApiClientStream handle,
    LwEglApiStream2Elw* elw);

bool TestExpStream2ProducerConnectStream(
    LwEglStreamHandle stream,
    LwEglApiStream2ProducerCaps *producerCaps);

bool TestExpStream2ProducerGetCaps(LwEglStreamHandle stream,
                                   LwEglApiStream2ProducerCaps *caps);

bool TestExpStream2ProducerRegisterBuffer(
    LwEglStreamHandle handle,
    LwEglApiBufferHandle image,
    LwEglApiClientBuffer buffer);

bool TestExpStream2ProducerUnregisterBuffer(
    LwEglStreamHandle handle,
    LwEglApiClientBuffer buffer);

bool TestExpStream2ProducerPresentFrame(
    LwEglStreamHandle handle,
    LwEglApiStream2Frame *frame);

bool TestExpStream2ProducerReturnFrame(
    LwEglApiClientStream           handle,
    LwS64                          usec,
    LwEglApiClientBuffer           buffer,
    LwEglApiStream2Frame*          frame);

bool TestExpStream2ProducerMetaDataSet(
    LwEglApiClientStream           handle,
    LwU32                          index,
    LwU32                          offset,
    LwU32                          size,
    const void*                    data);

bool TestExpStream2ProducerDisconnect(
    LwEglStreamHandle handle);

// Stream

bool TestExpStream2StatePoll(
    LwEglStreamHandle stream,
    LwU64* state);

bool TestExpStream2StateWait(
    LwEglStreamHandle stream,
    LwS64 timeoutUsec,
    LwU64 stateToMatch,
    LwU64* state);

bool TestExpStream2GetConstAttr(
    LwEglApiClientStream handle,
    LwEglApiStream2ConstantAttr* attr);

#ifdef __cplusplus
}
#endif

#endif
