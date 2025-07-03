/*
 * Copyright (c) 2016-2018, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>
#include "testclient.h"
#include "testclientStream2.h"

extern LwEglApiAccessFuncs g_eglApiAccessFuncs;

//Consumer

bool TestExpStream2ConsumerReserve(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiClientStream* handle)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream2.consumer.reserve(display, stream, handle);

    return err == LwSuccess;
}

bool TestExpStream2ConsumerGetElw(
    LwEglApiClientStream handle,
    LwEglApiStream2Elw* elw)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.common.elwGet(handle, elw);

    LOG_INFO("%s: Printing Consumer Elw: \n", __FUNCTION__);
    LOG_INFO("\tconsumerHV: %d\n", elw->consumerHV);
    LOG_INFO("\tconsumerRM: %d\n", elw->consumerRM);
    LOG_INFO("\tconsumerGPU: %d\n", elw->consumerGPU);

    return err == LwSuccess;
}

bool TestExpStream2ConsumerConnect(
    LwEglApiClientStream handle,
    LwEglApiStream2ConsumerCaps *consumerCaps)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.consumer.connect(handle, consumerCaps);
    return err == LwSuccess;
}

static void PrintBufSupport(const char *param, LwEglBufSupport val)
{
    switch (val) {
    case LwEglBufSupport_Direct: LOG_INFO("%s = LwEglBufSupport_Direct\n", param); break;
    case LwEglBufSupport_Copy: LOG_INFO("%s = LwEglBufSupport_Copy\n", param); break;
    case LwEglBufSupport_Slow: LOG_INFO("%s = LwEglBufSupport_Slow\n", param); break;
    case LwEglBufSupport_Unsupported:
    default: LOG_INFO("%s = LwEglBufSupport_Unsupported\n", param); break;
    }
}

static void PrintConsumerCaps(LwEglApiStream2ConsumerCaps *caps)
{
    LOG_INFO("Consumer Caps:\n");

    LOG_INFO("\tImage:\n");
    LOG_INFO("\t\tscanout = %d\n", (int)caps->image.scanout);
    LOG_INFO("\t\tcompression = %d\n", (int)caps->image.compression);
    LOG_INFO("\t\torigin = %d\n", (int)caps->image.origin);

    LOG_INFO("\tsync:\n");
    LOG_INFO("\t\tuseLwRm = %d\n", (int)caps->sync.useLwRm);

    LOG_INFO("\tmetadata:\n");
    for (int i = 0; i < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; ++i) {
        LOG_INFO("\t\tsupportedTypes[%d] = %d\n", i, (int)caps->metadata.supportedTypes[i]);
    }
}

bool TestExpStream2ConsumerGetCaps(
    LwEglApiClientStream handle,
    LwEglApiStream2ConsumerCaps *caps)
{
    LOG_FUNC("handle %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.common.consumerCapsGet(handle, caps);

    if (err == LwSuccess) {
        PrintConsumerCaps(caps);
        return true;
    } else {
        return false;
    }
}

bool TestExpStream2ConsumerBufferObtain(
    LwEglApiClientStream handle,
    LwEglApiBufferHandle* image)
{
    LOG_FUNC("handle %p, image %p.\n", handle, *image);

    LwEglApiBufferHandle imageHandle;
    LwError err = g_eglApiAccessFuncs.stream2.consumer.bufferObtain(handle, &imageHandle);
    *image = imageHandle;

    return err == LwSuccess;
}

bool TestExpStream2ConsumerBufferNotify(
    LwEglApiClientStream handle,
    LwEglApiBufferHandle image,
    LwError err,
    LwEglApiClientBuffer buffer)
{
    LOG_FUNC("handle %p, image %p, buffer %p.\n", handle, image, buffer);

    LwError error = g_eglApiAccessFuncs.stream2.consumer.bufferNotify(handle, image, err, buffer);

    return error == LwSuccess;
}

bool TestExpStream2ConsumerFrameAcquire(
    LwEglApiClientStream           handle,
    LwS64                          usec,
    LwEglApiStream2Frame*          frame)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.consumer.frameAcquire(handle, usec, frame);

    return err == LwSuccess;
}

bool TestExpStream2ConsumerFrameRelease(
    LwEglApiClientStream           handle,
    LwEglApiStream2Frame*          frame)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.consumer.frameRelease(handle, frame);

    return err == LwSuccess;
}

bool TestExpStream2ConsumerMetaDataGet(
    LwEglApiClientStream           handle,
    LwEglApiClientBuffer           buffer,
    LwU32                          index,
    LwU32                          offset,
    LwU32                          size,
    void*                          data)
{
    LOG_FUNC("stream %p, data %p.\n", handle, data);

    LwError err = g_eglApiAccessFuncs.stream2.consumer.metadataGet(handle, buffer, index, offset, size, data);

    return err == LwSuccess;
}

bool TestExpStream2ConsumerDisconnect(
    LwEglApiClientStream handle)
{
    LwError err = g_eglApiAccessFuncs.stream2.consumer.disconnect(handle, (LwError) 0);
    return err == LwSuccess;
}

// Producer

bool TestExpStream2ProducerReserve(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiClientStream* handle)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream2.producer.reserve(display, stream, handle);

    return err == LwSuccess;
}

bool TestExpStream2ProducerConnectStream(
    LwEglStreamHandle handle,
    LwEglApiStream2ProducerCaps *producerCaps)
{
    LOG_FUNC("handle %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.producer.connect(handle, producerCaps);
    return err == LwSuccess;
}

static void PrintProducerCaps(LwEglApiStream2ProducerCaps *caps)
{
    LOG_INFO("Producer Caps:\n");

    LOG_INFO("\tImage:\n");
    LOG_INFO("\t\tcompression = %d\n", (int)caps->image.compression);
    LOG_INFO("\t\torigin = %d\n", (int)caps->image.origin);

    LOG_INFO("\tsync:\n");
    LOG_INFO("\t\tuseLwRm = %d\n", (int)caps->sync.useLwRm);

    LOG_INFO("\tmetadata:\n");
    for (int i = 0; i < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; ++i) {
        LOG_INFO("\t\t%d: type = %d, size = %d\n",
                i, (int)caps->metadata.type[i], (int)caps->metadata.size[i]);
    }
}

bool TestExpStream2ProducerGetCaps(
    LwEglStreamHandle handle,
    LwEglApiStream2ProducerCaps *caps)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.common.producerCapsGet(handle, caps);

    if (err == LwSuccess) {
        PrintProducerCaps(caps);
        return true;
    } else {
        return false;
    }
}

bool TestExpStream2ProducerRegisterBuffer(
    LwEglStreamHandle handle,
    LwEglApiBufferHandle image,
    LwEglApiClientBuffer buffer)
{
    LOG_FUNC("stream %p, image %p, buffer %p.\n", handle, image, buffer);

    LwError err = g_eglApiAccessFuncs.stream2.producer.bufferRegister(handle, image, buffer);
    return err == LwSuccess;
}

bool TestExpStream2ProducerUnregisterBuffer(
    LwEglStreamHandle handle,
    LwEglApiClientBuffer buffer)
{
    LOG_FUNC("stream %p, buffer %p.\n", handle, buffer);

    LwError err = g_eglApiAccessFuncs.stream2.producer.bufferUnregister(handle, buffer);
    return err == LwSuccess;
}

bool TestExpStream2ProducerPresentFrame(
    LwEglStreamHandle handle,
    LwEglApiStream2Frame *frame)
{
    LOG_FUNC("stream %p, newFrame %p.\n", handle, frame);

    LwError err = g_eglApiAccessFuncs.stream2.producer.framePresent(handle, frame);
    return err == LwSuccess;
}

bool TestExpStream2ProducerReturnFrame(
    LwEglApiClientStream           handle,
    LwS64                          usec,
    LwEglApiClientBuffer           buffer,
    LwEglApiStream2Frame*          frame)
{
    LOG_FUNC("handle %p, usec %lld, buffer %p.\n", handle, usec, buffer);

    LwError err = g_eglApiAccessFuncs.stream2.producer.frameReturn(handle, usec, buffer, frame);
    return err == LwSuccess;
}

bool TestExpStream2ProducerMetaDataSet(
    LwEglApiClientStream           handle,
    LwU32                          index,
    LwU32                          offset,
    LwU32                          size,
    const void*                    data)
{
    LOG_FUNC("stream %p, data %p.\n", handle, data);

    LwError err = g_eglApiAccessFuncs.stream2.producer.metadataSet(handle, index, offset, size, data);

    return err == LwSuccess;
}

bool TestExpStream2ProducerDisconnect(
    LwEglStreamHandle handle)
{
    LOG_FUNC("stream %p.\n", handle);

    LwError err = g_eglApiAccessFuncs.stream2.producer.disconnect(handle, (LwError) 0);
    return err == LwSuccess;
}

// Stream

bool TestExpStream2StatePoll(
    LwEglStreamHandle handle,
    LwU64* state)
{
    LOG_FUNC("stream %p.\n", handle);

    LwU64 value = 0;
    LwError err = g_eglApiAccessFuncs.stream2.common.statePoll(handle, &value);
    *state = value;
    return err == LwSuccess;
}

bool TestExpStream2StateWait(
    LwEglStreamHandle handle,
    LwS64 timeoutUsec,
    LwU64 stateToMatch,
    LwU64* state)
{
    LOG_FUNC("stream %p.\n", handle);

    LwU64 value = 0;
    LwError err = g_eglApiAccessFuncs.stream2.common.stateWait(handle, timeoutUsec, stateToMatch, &value);
    *state = value;
    return err == LwSuccess;
}

bool TestExpStream2GetConstAttr(
    LwEglApiClientStream handle,
    LwEglApiStream2ConstantAttr* attr)
{
    LwError err = g_eglApiAccessFuncs.stream2.common.constantAttrGet(handle, attr);

    LOG_INFO("Printing Constant Attr: \n");
    LOG_INFO("\tfifoSize: %d\n", attr->fifoSize);
    LOG_INFO("\tfifoSync: %d\n", attr->fifoSync);
    LOG_INFO("\treset: %d\n", attr->reset);
    LOG_INFO("\treuse: %d\n", attr->reuse);

    return err == LwSuccess;
}

