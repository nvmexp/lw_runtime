/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#ifndef EGL_TESTCLIENT_STREAM1_H
#define EGL_TESTCLIENT_STREAM1_H

#if defined(LW_EGL_DESKTOP_COMPATIBLE_HEADERS)
// For mobile base types and lwos/lwutil functionality on desktop builds
#include "mobile_common.h"
#endif

#include "defs.h"
#include "eglapiinterface.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test EGL export functions called by client API.
 */
bool TestExpStream1ConsumerConnect(LwEglDisplayHandle display,
                                  LwEglStreamHandle stream,
                                  LwEglApiConsumerInfo *consumerInfo);

bool TestExpStream1ConsumerGetCaps(LwEglDisplayHandle display,
                                  LwEglStreamHandle stream,
                                  LwEglApiConsumerCaps *caps);

bool TestExpStream1ConsumerUpdateFrameAttr(LwEglDisplayHandle display,
                                          LwEglStreamHandle stream,
                                          LwBool acquire,
                                          LwBool release,
                                          const LwEglApiStreamUpdateAttrs *attrs);

bool TestExpStream1ConsumerQueryMetadata(LwEglDisplayHandle display,
                                        LwEglStreamHandle stream,
                                        LwU32 name,
                                        LwU8  index,
                                        LwU32 offset,
                                        LwU32 size,
                                        void* data);

bool TestExpStream1ConsumerDisconnect(LwEglDisplayHandle display,
                                     LwEglStreamHandle stream);

bool TestExpStream1ProducerConnect(LwEglDisplayHandle display,
                                  LwEglStreamHandle stream,
                                  LwEglApiProducerInfo *producerInfo);

bool TestExpStream1ProducerGetCaps(LwEglDisplayHandle display,
                                  LwEglStreamHandle stream,
                                  LwEglApiProducerCaps *caps);

bool TestExpStream1ProducerRegisterBuffer(LwEglDisplayHandle display,
                                         LwEglStreamHandle stream,
                                         LwEglApiStreamFrame *newBuffer);

bool TestExpStream1ProducerUnregisterBuffer(LwEglDisplayHandle display,
                                           LwEglStreamHandle stream,
                                           LwEglApiStreamFrame *oldBuffer);

bool TestExpStream1ProducerSetMetadata(LwEglDisplayHandle display,
                                      LwEglStreamHandle stream,
                                      LwU8 index,
                                      LwU32 offset,
                                      LwU32 size,
                                      const void* data);

bool TestExpStream1ProducerPresentFrame(LwEglDisplayHandle display,
                                       LwEglStreamHandle stream,
                                       LwEglApiStreamFrame *newFrame);

bool TestExpStream1ProducerDisconnect(LwEglDisplayHandle display,
                                     LwEglStreamHandle stream);

#ifdef __cplusplus
}
#endif

#endif
