/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;

#define BLOCK_DIV(N, S) (((N) + (S) - 1) / (S))
#define ROUND_UP(N, S) (BLOCK_DIV(N, S) * (S))
#define ROUND_DN(N, S) ((N) - ((N) % (S)))


class LWNCopyMirror
{
    static const int texSize = 16;
    static const int texBorder = 4;

    typedef enum MirrorAxis {
        MirrorAxis_X = 1,
        MirrorAxis_Y = 2,
        MirrorAxis_Z = 4,
        MirrorAxis_Permutations = 8
    } MirrorAxis;

    typedef enum MirrorFunc {
        MirrorFunc_CopyBufferToTexture,
        MirrorFunc_CopyBufferToTexture_Engine2D,
        MirrorFunc_CopyTextureToTexture,
        MirrorFunc_CopyTextureToBuffer,
        MirrorFunc_Count
    } MirrorFunc;

public:
    LWNTEST_CppMethods();

    void fillMemory(uint8_t *mem) const;
    bool checkResult(int mirrorFlags, uint8_t *mem) const;
    void setupCopyRegion(CopyRegion *region) const;
    void setupStrides(ptrdiff_t *bufOffset, ptrdiff_t *rowStride, ptrdiff_t *imgStride) const;
};


lwString LWNCopyMirror::getDescription() const
{
    lwStringBuf sb;
    sb << "Tests all legal combinations of CopyFlags::MIRROR_X, CopyFlags::MIRROR_Y and\n"
          "CopyFlags::MIRROR_Z for each Copy command where they are legal. (Invalid cases\n"
          "are marked in blue.) The valid cases tested are all permutations of:\n"
          "* CopyBufferToTexture: MIRROR_Y and MIRROR_Z\n"
          "* CopyBufferToTexture with ENGINE_2D: MIRROR_X, MIRROR_Y and MIRROR_Z\n"
          "* CopyTextureToTexture: MIRROR_X, MIRROR_Y and MIRROR_Z\n"
          "* CopyTextureToBuffer: MIRROR_Y and MIRROR_Z\n";
    return sb.str();
}

int LWNCopyMirror::isSupported() const
{
    return lwogCheckLWNAPIVersion(53, 4);
}

void LWNCopyMirror::fillMemory(uint8_t *mem) const
{
    assert(texSize <= 256);
    for (int z=0; z<texSize; z++) {
        for (int y=0; y<texSize; y++) {
            for (int x=0; x<texSize; x++) {
                mem[0] = x;
                mem[1] = y;
                mem[2] = z;
                mem[3] = 255;
                mem += 4;
            }
        }
    }
}

bool LWNCopyMirror::checkResult(int mirrorFlags, uint8_t *mem) const
{
    uint8_t *m = mem;
    bool flipX = !!(mirrorFlags & MirrorAxis_X);
    bool flipY = !!(mirrorFlags & MirrorAxis_Y);
    bool flipZ = !!(mirrorFlags & MirrorAxis_Z);
    int xStart = flipX ? (texSize - 1) : 0;
    int yStart = flipY ? (texSize - 1) : 0;
    int zStart = flipZ ? (texSize - 1) : 0;
    int xEnd = flipX ? -1 : texSize;
    int yEnd = flipY ? -1 : texSize;
    int zEnd = flipZ ? -1 : texSize;
    int xInc = flipX ? -1 : 1;
    int yInc = flipY ? -1 : 1;
    int zInc = flipZ ? -1 : 1;

    int incorrect = 0;
    for (int z = zStart; z != zEnd; z += zInc) {
        for (int y = yStart; y != yEnd; y += yInc) {
            for (int x = xStart; x != xEnd; x += xInc) {
                if ((z < texBorder) || (z > (texSize-texBorder-1)) ||
                    (y < texBorder) || (y > (texSize-texBorder-1)) ||
                    (x < texBorder) || (x > (texSize-texBorder-1)))
                {
                    if ((m[0] != 0) || (m[1] != 0) || (m[2] != 0) || (m[3] != 0)) {
                        incorrect++;
#if DEBUG_MODE
#define DEBUG_INCORRECT_LIMIT 10
                        if (incorrect < DEBUG_INCORRECT_LIMIT) {
                            DEBUG_PRINT(("offset %5u,  expected: 00 00 00 00  got: %02x %02x %02x %02x\n",
                                         m - mem,
                                         m[0], m[1], m[2], m[3]));
                        }
#else
                        goto done;
#endif
                    }
                }
                else
                {
                    if ((m[0] != x) || (m[1] != y) || (m[2] != z) || (m[3] != 255)) {
                        incorrect++;
#if DEBUG_MODE
                        if (incorrect < DEBUG_INCORRECT_LIMIT) {
                            DEBUG_PRINT(("offset %5u,  expected: %02x %02x %02x %02x  got: %02x %02x %02x %02x\n",
                                         m - mem,
                                         x, y, z, 0xff,
                                         m[0], m[1], m[2], m[3]));
                        }
#else
                        goto done;
#endif
                    }
                }
                m += 4;
            }
        }
    }
    DEBUG_PRINT(("incorrect: %u\n", incorrect));
#if !DEBUG_MODE
done:
#endif
    return (incorrect == 0);
}


void LWNCopyMirror::setupCopyRegion(CopyRegion *region) const
{
    int regionSize = texSize - (2 * texBorder);
    region->xoffset = texBorder;
    region->yoffset = texBorder;
    region->zoffset = texBorder;
    region->width   = regionSize;
    region->height  = regionSize;
    region->depth   = regionSize;
}

void LWNCopyMirror::setupStrides(ptrdiff_t *bufOffset, ptrdiff_t *rowStride, ptrdiff_t *imgStride) const
{
    int bpp = 4;
    int rowPitch = texSize * bpp;
    int imgPitch = texSize * rowPitch;

    *bufOffset = (texBorder * imgPitch) + (texBorder * rowPitch) + (texBorder * bpp);
    *rowStride = rowPitch;
    *imgStride = imgPitch;
}


void LWNCopyMirror::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = 8;
    const int cellHeight = MirrorFunc_Count;
    int cellNum = 0;
    cellTestInit(cellWidth, cellHeight);

    float blackColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    // memory pools: allocate enough space for the largest formats
    TextureBuilder tb;
    tb.SetDevice(device)
      .SetDefaults()
      .SetTarget(TextureTarget::TARGET_2D_ARRAY)
      .SetFormat(Format::RGBA8)
      .SetSize3D(texSize, texSize, texSize);

    // allocate memory
    size_t texStorageSize = tb.GetPaddedStorageSize();
    MemoryPoolAllocator texAllocator(device, NULL, 2 * texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    Texture *srcTex = texAllocator.allocTexture(&tb);
    Texture *dstTex = texAllocator.allocTexture(&tb);

    size_t maxMemSize = texStorageSize;
    MemoryPoolAllocator bufAllocator(device, NULL, 2 * maxMemSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer * srcBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxMemSize);
    BufferAddress srcBufAddr = srcBuf->GetAddress();
    Buffer * dstBuf = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, maxMemSize);
    BufferAddress dstBufAddr = dstBuf->GetAddress();
    uint8_t * srcMem = (uint8_t *) srcBuf->Map();
    uint8_t * dstMem = (uint8_t *) dstBuf->Map();

    fillMemory(srcMem);

    for (int mirrorFunc = 0; mirrorFunc < MirrorFunc_Count; mirrorFunc++) {
        for (int mirrorAxes = 0; mirrorAxes < MirrorAxis_Permutations; mirrorAxes++) {
            DEBUG_PRINT(("func %i, flags: %x\n", mirrorFunc, mirrorAxes));
            // skip invalid permutations
            bool validCase = true;
            switch (mirrorFunc) {
            case MirrorFunc_CopyBufferToTexture:
            case MirrorFunc_CopyTextureToBuffer:
                if (mirrorAxes & MirrorAxis_X) {
                    validCase = false;
                }
                break;
            default:
                break;
            }
            if (!validCase) {
                // Mark invalid cases as blue
                SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
                queueCB.ClearColor(0, 0.0, 0.0, 1.0);
                queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
                queueCB.submit();
                cellNum++;
                continue;
            }

            CopyRegion srcRegion;
            CopyRegion dstRegion;
            CopyRegion fullRegion = { 0, 0, 0, texSize, texSize, texSize };
            ptrdiff_t bufOffset, rowStride, imgStride;
            int baseFlags = CopyFlags::NONE;

            ct_assert((MirrorAxis_X << 2) == CopyFlags::MIRROR_X);
            ct_assert((MirrorAxis_Y << 2) == CopyFlags::MIRROR_Y);
            ct_assert((MirrorAxis_Z << 2) == CopyFlags::MIRROR_Z);
            int mirrorFlags = (mirrorAxes << 2);

            switch (mirrorFunc) {
            case MirrorFunc_CopyBufferToTexture_Engine2D:
                mirrorFlags |= CopyFlags::ENGINE_2D;
            case MirrorFunc_CopyBufferToTexture:
                queueCB.ClearTexture(srcTex, NULL, &fullRegion, blackColor, LWN_CLEAR_COLOR_MASK_RGBA);
                setupStrides(&bufOffset, &rowStride, &imgStride);
                setupCopyRegion(&dstRegion);
                queueCB.SetCopyRowStride(rowStride);
                queueCB.SetCopyImageStride(imgStride);
                queueCB.CopyBufferToTexture(srcBufAddr + bufOffset, srcTex, NULL, &dstRegion, baseFlags | mirrorFlags);
                break;
            default:
                queueCB.SetCopyRowStride(0);
                queueCB.SetCopyImageStride(0);
                queueCB.CopyBufferToTexture(srcBufAddr, srcTex, NULL, &fullRegion, CopyFlags::NONE);
                break;
            }

            Texture *readTex = dstTex;
            if (mirrorFunc == MirrorFunc_CopyTextureToTexture) {
                queueCB.ClearTexture(dstTex, NULL, &fullRegion, blackColor, LWN_CLEAR_COLOR_MASK_RGBA);
                setupCopyRegion(&srcRegion);
                setupCopyRegion(&dstRegion);
                queueCB.CopyTextureToTexture(srcTex, NULL, &srcRegion, dstTex, NULL, &dstRegion, mirrorFlags);
            } else {
                readTex = srcTex;
            }

            if (mirrorFunc == MirrorFunc_CopyTextureToBuffer) {
                setupCopyRegion(&srcRegion);
                setupStrides(&bufOffset, &rowStride, &imgStride);
                queueCB.ClearBuffer(dstBufAddr, texStorageSize, 0);
                queueCB.SetCopyRowStride(rowStride);
                queueCB.SetCopyImageStride(imgStride);
                queueCB.CopyTextureToBuffer(readTex, NULL, &srcRegion, dstBufAddr + bufOffset, mirrorFlags);
            } else {
                queueCB.SetCopyRowStride(0);
                queueCB.SetCopyImageStride(0);
                queueCB.CopyTextureToBuffer(readTex, NULL, &fullRegion, dstBufAddr, CopyFlags::NONE);
            }

            queueCB.submit();
            queue->Finish();

            // Check resulting data for correctness
            bool passed = checkResult(mirrorAxes, dstMem);
            SetCellViewportScissorPadded(queueCB, cellNum % cellWidth, cellNum / cellWidth, 1);
            queueCB.ClearColor(0, passed ? 0.0 : 1.0, passed ? 1.0 : 0.0, 0.0);
            queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
            cellNum++;
        }
    }

    queueCB.submit();
    queue->Finish();

    // free memory
    bufAllocator.freeBuffer(srcBuf);
    bufAllocator.freeBuffer(dstBuf);
    texAllocator.freeTexture(srcTex);
    texAllocator.freeTexture(dstTex);
}


OGTEST_CppTest(LWNCopyMirror, lwn_copy_image_mirror, );

