/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_c.h"
#include "lwn_utils.h"
#include "readpixels.h"

static MemoryPoolAllocator *allocatorCpu = NULL;
static LWNbuffer *colorBufferData = NULL;
static int colorBufferSize = 0;
static unsigned char *colorBufferDataPtr = NULL;

void CreateFramebufferBuffer()
{
    LWNdevice *device = g_lwnDevice;

    // Create buffer for readback of color data
    colorBufferSize = lwrrentWindowWidth * lwrrentWindowHeight * 4;
    allocatorCpu = new MemoryPoolAllocator(device, NULL, colorBufferSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    LWNbufferBuilder bb;
    lwnBufferBuilderSetDevice(&bb, device);
    lwnBufferBuilderSetDefaults(&bb);

    colorBufferData = allocatorCpu->allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, colorBufferSize);
    colorBufferDataPtr = (unsigned char *) lwnBufferMap(colorBufferData);
}

void DestroyFramebufferBuffer()
{
    if (colorBufferData) {
        allocatorCpu->freeBuffer(colorBufferData);
        colorBufferData = NULL;
    }
    colorBufferSize = 0;
    colorBufferDataPtr = NULL;
    if (allocatorCpu) {
        delete allocatorCpu;
        allocatorCpu = NULL;
    }
}

void GetFramebufferData(unsigned char *out)
{
    lwnQueueFinish(g_lwnQueue);

    g_lwnWindowFramebuffer.readPixels(colorBufferData);
    memcpy(out, colorBufferDataPtr, colorBufferSize);
}


