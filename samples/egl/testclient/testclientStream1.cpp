/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
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
#include "testclientStream1.h"

extern LwEglApiAccessFuncs g_eglApiAccessFuncs;

bool TestExpStream1ConsumerConnect(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiConsumerInfo *consumerInfo)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.consumer.connect(display,
                      stream, consumerInfo);
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

static void PrintCosumerCaps(LwEglApiConsumerCaps *caps)
{
    LOG_INFO("Consumer Caps:\n");
    LOG_INFO("\tmaxAcquiredFrames = %d\n", caps->maxAcquiredFrames);
    LOG_INFO("\tbufferCache = %d\n", caps->bufferCache);
    LOG_INFO("\tsupportCompressedImage = %d\n", caps->supportCompressedImage);
    LOG_INFO("\tsupportAutoUpdate = %d\n", caps->supportAutoUpdate);
    LOG_INFO("\tsupportClientUpdate = %d\n", caps->supportClientUpdate);
    LOG_INFO("\tautoDisconnects = %d\n", caps->autoDisconnects);
    LOG_INFO("\tautoAcquireDefault = %d\n", caps->autoAcquireDefault);
    LOG_INFO("\tsupportScanout = %d\n", caps->supportScanout);

    PrintBufSupport("\tsupportSysmem", caps->supportSysmem);
    PrintBufSupport("\tsupportVidmem", caps->supportVidmem);
    PrintBufSupport("\tsupportPitch", caps->supportPitch);
    PrintBufSupport("\tsupportCommonBlock", caps->supportCommonBlock);
    PrintBufSupport("\tsupportTegraBlock16Bx2", caps->supportTegraBlock16Bx2);
    PrintBufSupport("\tsupportDgpuBlock16Bx2", caps->supportDgpuBlock16Bx2);

    LOG_INFO("\tsyncTypeSupport = 0x%x\n", caps->syncTypeSupport);
    LOG_INFO("\tsyncType = %d\n", caps->syncType);

    for (int i = 0; i < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; ++i) {
        LOG_INFO("\tsupportedMetadataTypes[%d] = %d\n", i, caps->supportedMetadataTypes[i]);
    }
}

bool TestExpStream1ConsumerGetCaps(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiConsumerCaps *caps)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.consumer.getCaps(display,
                      stream, caps);

    if (err == LwSuccess) {
        PrintCosumerCaps(caps);
        return true;
    } else {
        return false;
    }
}

bool TestExpStream1ConsumerUpdateFrameAttr(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwBool acquire,
    LwBool release,
    const LwEglApiStreamUpdateAttrs *attrs)
{
    LOG_FUNC("display %p, stream %p, acquire %d, release %d.\n",
        display, stream, acquire, release);

    LwError err = g_eglApiAccessFuncs.stream.consumer.frameUpdate(
                      display, stream, acquire, release, attrs);
    return err == LwSuccess;
}

bool TestExpStream1ConsumerQueryMetadata(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwU32 name,
    LwU8  index,
    LwU32 offset,
    LwU32 size,
    void* data)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.consumer.metadataQuery(
                      display, stream, name, index, offset, size, data);
    return err == LwSuccess;
}

bool TestExpStream1ConsumerDisconnect(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.consumer.disconnect(display, stream);
    return err == LwSuccess;
}

bool TestExpStream1ProducerConnect(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiProducerInfo *producerInfo)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.connect(display,
                      stream, producerInfo);
    return err == LwSuccess;
}

static void PrintProducerCaps(LwEglApiProducerCaps *caps)
{
    LOG_INFO("Producer Caps:\n");
    LOG_INFO("\tmaxBufferCount = %d\n", caps->maxBufferCount);
    LOG_INFO("\tbufferCache = %d\n", caps->bufferCache);
    LOG_INFO("\tsupportCompressedImage = %d\n", caps->supportCompressedImage);
    LOG_INFO("\tsyncTypeSupport = 0x%x\n", caps->syncTypeSupport);
    LOG_INFO("\tsyncType = %d\n", caps->syncType);

    for (int i = 0; i < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; ++i) {
        LOG_INFO("\tmetadataType[%d] = %d\n", i, caps->metadataType[i]);
        LOG_INFO("\tmetadataSize[%d] = 0x%x\n", i, caps->metadataSize[i]);
    }
}

bool TestExpStream1ProducerGetCaps(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiProducerCaps *caps)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.getCaps(display, stream, caps);

    if (err == LwSuccess) {
        PrintProducerCaps(caps);
        return true;
    } else {
        return false;
    }
}

bool TestExpStream1ProducerRegisterBuffer(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiStreamFrame *newBuffer)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.bufferRegister(display,
                      stream, newBuffer);
    return err == LwSuccess;
}

bool TestExpStream1ProducerUnregisterBuffer(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiStreamFrame *oldBuffer)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.bufferUnregister(display,
                      stream, oldBuffer);
    return err == LwSuccess;
}

bool TestExpStream1ProducerSetMetadata(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwU8 index,
    LwU32 offset,
    LwU32 size,
    const void* data)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.metadataSet(display,
                      stream, index, offset, size, data);
    return err == LwSuccess;
}

bool TestExpStream1ProducerPresentFrame(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream,
    LwEglApiStreamFrame *newFrame)
{
    LOG_FUNC("display %p, stream %p, newFrame %p.\n",
        display, stream, newFrame);

    LwError err = g_eglApiAccessFuncs.stream.producer.framePresent(display,
                      stream, newFrame);
    return err == LwSuccess;
}

bool TestExpStream1ProducerDisconnect(
    LwEglDisplayHandle display,
    LwEglStreamHandle stream)
{
    LOG_FUNC("display %p, stream %p.\n", display, stream);

    LwError err = g_eglApiAccessFuncs.stream.producer.disconnect(display, stream);
    return err == LwSuccess;
}
