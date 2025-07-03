/*
 * Copyright (c) 2017 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"

#include "lwn_utils.h"
#include "lwddk_vic.h"
#include "lwnExt/lwnExt_Internal.h"
#include <sys/time.h>
#include <unistd.h>

using namespace lwn;

class LWLWicInteropTest
{
public:
    LWLWicInteropTest() {}
    LWNTEST_CppMethods();
};

lwString LWLWicInteropTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Basic VIC interop test.  Clears a texture to green with GPU and passes "
        "it along with a fence to VIC.  VIC copies the texture to another "
        "texture, and passes the second texture along with a fence back to GPU.  "
        "The GPU then copies the texture to display.  If the test passes, a green "
        "screen should be displayed.";
    return sb.str();
}

int LWLWicInteropTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 203);
}

void LWLWicInteropTest::doGraphics() const
{
    LwError err;
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    PFNLWNTEXTUREGETLWRMSURFACELWXPROC lwnTextureGetLwRmSurfaceLWX =
        (PFNLWNTEXTUREGETLWRMSURFACELWXPROC)device->GetProcAddress("lwnTextureGetLwRmSurfaceLWX");
    PFNLWNSYNCINITIALIZEFROMLWRMSYNCLWXPROC lwnSyncInitializeFromLwRmSyncLWX =
        (PFNLWNSYNCINITIALIZEFROMLWRMSYNCLWXPROC)device->GetProcAddress("lwnSyncInitializeFromLwRmSyncLWX");
    PFNLWNSYNCGETLWRMSYNCLWXPROC lwnSyncGetLwRmSyncLWX =
        (PFNLWNSYNCGETLWRMSYNCLWXPROC)device->GetProcAddress("lwnSyncGetLwRmSyncLWX");

    MemoryPoolAllocator texAllocator(device, NULL, 128 * 1024 * 1024,
                                        (LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                         LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT));
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetFormat(Format::RGBA8);
    tb.SetSize2D(1000, 1000);
    tb.SetTarget(TextureTarget::TARGET_2D);
    Texture *tex1 = texAllocator.allocTexture(&tb);
    Texture *tex2 = texAllocator.allocTexture(&tb);

    LwRmDeviceHandle hDevice = NULL;
    LwDdkVicSession *hSession = NULL;
    LwRmOpenNew(&hDevice);
    LwDdkVicCreateSession(hDevice, NULL, &hSession);

    LwRmSurface surf1, surf2;
    lwnTextureGetLwRmSurfaceLWX(reinterpret_cast<const LWNtexture *>(tex1), NULL, &surf1);
    lwnTextureGetLwRmSurfaceLWX(reinterpret_cast<const LWNtexture *>(tex2), NULL, &surf2);
    LwDdkVicConfigParameters configParams = {};
    LwDdkVicConfigureSourceSurface(hSession, &configParams, 0, &surf1, 1, NULL, NULL);
    LwDdkVicConfigureTargetSurface(hSession, &configParams, &surf2, 1, NULL);
    LwDdkVicConfigure(hSession, &configParams);

    Sync sync;
    sync.Initialize(device);

    const float red[] = { 1.0, 0.0, 0.0, 1.0 };
    const float green[] = { 0.0, 1.0, 0.0, 1.0 };

    // Clear output surface, tex1 and tex2 to red to start with.
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, red, ClearColorMask::RGBA);
    queueCB.SetRenderTargets(1, &tex1, NULL, NULL, NULL);
    queueCB.SetScissor(0, 0, tex1->GetWidth(), tex1->GetHeight());
    queueCB.ClearColor(0, red, ClearColorMask::RGBA);
    queueCB.SetRenderTargets(1, &tex2, NULL, NULL, NULL);
    queueCB.ClearColor(0, red, ClearColorMask::RGBA);
    queueCB.submit();
    queue->Finish();

    // Clear tex1 to green with GPU, don't flush that clear yet.
    queueCB.SetRenderTargets(1, &tex1, NULL, NULL, NULL);
    queueCB.ClearColor(0, green, ClearColorMask::RGBA);
    queueCB.submit();

    // Sync from GPU to VIC.
    queueCB.FenceSync(&sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queueCB.submit();
    LwRmSyncWithStorage<1> fenceIn;
    lwnSyncGetLwRmSyncLWX(reinterpret_cast<const LWNsync *>(&sync), &fenceIn);
    sync.Finalize();

    // Blit from tex1 to tex2 with VIC (should remain blocked until GPU clear to green completes).
    LwRmSync *fenceOut;
    LwDdkVicExecParameters execParms = {};
    execParms.InputSurfaces[0][0] = &surf1;
    execParms.OutputSurface = &surf2;
    err = LwDdkVicExelwteSync(hSession, &execParms, &fenceIn, &fenceOut);

    // Sync from VIC to GPU.
    lwnSyncInitializeFromLwRmSyncLWX(reinterpret_cast<LWNsync *>(&sync),
                                     reinterpret_cast<LWNdevice *>(device),
                                     fenceOut);
    LwRmSyncClose(fenceOut);
    queueCB.WaitSync(&sync);

    // Blit from tex2 to window framebuffer with GPU.
    Texture *windowTex = g_lwnWindowFramebuffer.getAcquiredTexture();
    CopyRegion srcRegion = { 0, 0, 0, tex2->GetWidth(), tex2->GetHeight(), 1 };
    CopyRegion dstRegion = { 0, 0, 0, windowTex->GetWidth(), windowTex->GetHeight(), 1 };
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.CopyTextureToTexture(tex2, NULL, &srcRegion, windowTex, NULL, &dstRegion, CopyFlags::NONE);
    queueCB.submit();

    // Wait a bit to make sure VIC hw has become blocked on the input fence, and
    // only flush the GPU commands after that.
    usleep(1000);
    queue->Finish();

    sync.Finalize();
}

OGTEST_CppTest(LWLWicInteropTest, lwn_vic_interop, );
