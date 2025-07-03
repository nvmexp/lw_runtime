/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "surface.h"
#include "egltest.h"

// WAR the issue that lwos.h cannot be included
#ifndef LWOS_MEM_READ
#define LWOS_MEM_READ     0x1
#define LWOS_MEM_WRITE    0x2
#define LWOS_MEM_READ_WRITE (LWOS_MEM_READ | LWOS_MEM_WRITE)
#endif

static LwRmDeviceHandle localDeviceHandle = 0;


static bool
GetSurfacePlanesAndFormats(
    LwU32 type,
    LwU32 width,
    LwU32 height,
    LwU32 *widths,
    LwU32 *heights,
    int *numSurfaces,
    LwColorFormat *formats
)
{
    switch (type) {
    case EGLTEST_DATA_FORMAT_TYPE_YUV422:
        width = (width + 1) & ~1;
        *numSurfaces = 3;
        widths[0] = width;
        widths[1] = widths[2] = width >> 1;
        heights[0] = heights[1] = heights[2] = height;
        formats[0] = formats[1] = formats[2] = LwColorFormat_A8;
        break;
    default:
        LOG_ERR("Invalid format.\n");
        return false;
    }
    return true;
}

static bool
InitRmSurface(
    LwRmDeviceHandle hRmDev,
    LwRmSurface *surf,
    LwU32 width,
    LwU32 height,
    LwColorFormat format
)
{
    LwU32 pad = 2047;

    surf->Width = width;
    surf->Height = height;
    surf->ColorFormat = format;
    surf->Layout = LwRmSurfaceLayout_Pitch;

    surf->Pitch = (((LW_COLOR_GET_BPP(format) * width) + pad) & ~pad) >> 3;
    LwRmSurfaceComputeSize(surf);

    return true;
}

static LwRmMemHandle
AllocMemory(
    LwRmDeviceHandle hRmDev,
    LwU32 alignment,
    LwU32 size,
    bool vm,
    LwU32 vmId,
    unsigned char **mapping
)
{
    LwRmMemHandle handle;
    // Allocate WC memory for all cases including even those not accessed on CPU
    LwU32 Alignment = alignment & 0xffff;   // 16 lsb is used for alignment
    LWRM_DEFINE_MEM_HANDLE_ATTR(attr);

    if (vm) {
        LwRmHeap heaps[] = {LwRmHeap_IVC};
        LWRM_MEM_HANDLE_SET_HEAP_ATTR_EXT(attr, heaps, LW_ARRAY_SIZE(heaps), &vmId);
    }

    LWRM_MEM_HANDLE_SET_ATTR(attr, Alignment, LwOsMemAttribute_WriteBack, size, LWRM_MEM_TAG_TVMR_MISC);

    if (LwSuccess != LwRmMemHandleAllocAttr(hRmDev, &attr, &handle)) {
        LOG_ERR("LwRmMemHandleAllocAttr failed.\n");
        return 0;
    }

    if (mapping) {
        if (LwSuccess != LwRmMemMap(handle, 0, size, LWOS_MEM_READ_WRITE, (void **)mapping)) {
            LOG_ERR("LwRmMemMap failed.\n");
            LwRmMemHandleFree(handle);
            return 0;
        }
    }
    return handle;
}

static LwRmDeviceHandle
GetDeviceHandle(void)
{
    if (!localDeviceHandle) {
        LwRmOpenNew(&localDeviceHandle);
    }
    return localDeviceHandle;
}

EglTestSurface *
LwEglTestSurfaceCreate(
    LwU32 type,
    LwU32 width,
    LwU32 height,
    bool vm,
    LwU32 vmId
)
{
    LwRmDeviceHandle hRmDev = GetDeviceHandle();
    EglTestSurface *testSurf;
    LwU32 widths[6], heights[6];
    LwColorFormat formats[6];
    LwRmMemHandle hMem;
    unsigned char *mapping = NULL;
    int i, numSurfaces;
    LwU32 size, alignment;

    testSurf = (EglTestSurface*)calloc(1, sizeof(EglTestSurface));
    if (!(testSurf)) {
        LOG_ERR("Cannot alloc EglTestSurface.\n");
        return NULL;
    }

    if (!GetSurfacePlanesAndFormats(type, width, height,
                                    widths, heights, &numSurfaces,
                                    formats)) {
       return NULL;
    }
    testSurf->numSurfaces = numSurfaces;
    testSurf->rmDev = hRmDev;

    size = 0;

    for (i = 0; i < numSurfaces; i++) {
        if (!InitRmSurface(hRmDev,
                           &testSurf->rmSurface[i],
                           widths[i], heights[i],
                           formats[i])) {
            return NULL;
        }
        size += testSurf->rmSurface[i].Size;
    }

    alignment = LwRmSurfaceComputeAlignment(hRmDev,
                    (LwRmSurface*)&testSurf->rmSurface);

    if (!(hMem = AllocMemory(hRmDev, alignment, size, vm, vmId, &mapping))) {
        return NULL;
    }

    size = 0;
    for (i = 0; i < numSurfaces; i++) {
        testSurf->rmSurface[i].hMem = hMem;
        testSurf->rmSurface[i].Offset = size;
        if (mapping) {
            testSurf->mapping = (void*)(mapping);
        }
        size += testSurf->rmSurface[i].Size;
    }

    LOG_INFO("Created a EglTestSurface %p.\n", testSurf);

    return testSurf;
}

void
LwEglTestSurfaceDestroy(EglTestSurface *testSurf)
{
   LwU32 i, size = 0;

    if (!testSurf) {
        return;
    }

    for (i = 0; i < testSurf->numSurfaces; i++) {
        size += testSurf->rmSurface[i].Size;
    }

    if (testSurf->mapping) {
        LwRmMemUnmap(testSurf->rmSurface[0].hMem, (void*)testSurf->mapping, size);
    }

    LwRmMemHandleFree(testSurf->rmSurface[0].hMem);

    free(testSurf);
    testSurf = NULL;
}
