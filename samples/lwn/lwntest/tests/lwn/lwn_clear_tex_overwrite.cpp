/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <array>
#include <utility>
#include <vector>

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

class LWNClearTexOverwrite
{
public:
    LWNTEST_CppMethods();

private:
    static const uint8_t pattern = 0xff;
    static const size_t mempoolSize = 256 << 10;

    bool verifyResult(uint8_t const* start, size_t offset, size_t texSize) const;
};

namespace
{
    const std::array<std::pair<Format, uint32_t>, 9> formatList = { { {Format::R8,      1},
                                                                      {Format::R16F,    2},
                                                                      {Format::R32F,    4},
                                                                      {Format::RG8,     2},
                                                                      {Format::RG16,    4},
                                                                      {Format::RG32F,   8},
                                                                      {Format::RGBA8,   4},
                                                                      {Format::RGBA16,  8},
                                                                      {Format::RGBA32F, 16} } };

    const std::array<int[2], 4> sizeList = { { {8, 8}, {32, 16}, {40, 20}, {113, 71} } };
}

lwString LWNClearTexOverwrite::getDescription() const
{
    lwStringBuf sb;
    sb << "Simple test to verify that ClearTexture does not write to memory "
          "that does not belong to the texture (See Bug 2737287).\n"
          "The test creates a memory pool and initializes the memory with a "
          "defined pattern. Then it creates a texture inside the pool using an "
          "offset that respects the alignment constraints. The texture is cleared "
          "and the test verifies that the memory regions not covered by the "
          "texture still contain the initial pattern.\n";

    return sb.str();
}

int LWNClearTexOverwrite::isSupported() const
{
    return lwogCheckLWNAPIVersion(55, 6);
}

bool LWNClearTexOverwrite::verifyResult(uint8_t const* start, size_t offset, size_t texSize) const
{
    // Verfy that the memory region that does not belong tpo the
    // texture (start -> offset and offset + size -> start + mempoolSize)
    // still contains the pattern that was written before clearing the tex.
    for (size_t i = 0; i < offset; ++i) {
        if (start[i] != pattern) {
            return false;
        }
    }

    for (size_t i = (offset + texSize); i < mempoolSize; ++i) {
        if (start[i] != pattern) {
            return false;
        }
    }

    return true;
}

void LWNClearTexOverwrite::doGraphics() const
{
    DeviceState* deviceState = DeviceState::GetActive();
    Device* device = deviceState->getDevice();
    QueueCommandBuffer& queueCB = deviceState->getQueueCB();
    Queue* queue = deviceState->getQueue();

    auto asset = lwnUtil::AlignedStorageAlloc(mempoolSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT);
    memset(asset, pattern, mempoolSize);

    MemoryPoolBuilder mb;
    mb.SetDefaults().SetDevice(device)
      .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
      .SetStorage(asset, mempoolSize);

    MemoryPool memPool;
    memPool.Initialize(&mb);

    uint8_t* memPoolPtr = (uint8_t*)memPool.Map();

    std::vector<bool> results;

    int linearAlignment = 0;
    device->GetInteger(DeviceInfo::LINEAR_RENDER_TARGET_STRIDE_ALIGNMENT, &linearAlignment);

    for (auto f : formatList) {
        for (auto s : sizeList) {

            const ptrdiff_t stride = ((s[0] * f.second) + (linearAlignment - 1)) & ~(linearAlignment - 1);

            memset(memPoolPtr, pattern, mempoolSize);
            memPool.FlushMappedRange(0, mempoolSize);

            TextureBuilder tb;
            tb.SetDefaults().SetDevice(device)
                .SetTarget(TextureTarget::TARGET_2D)
                .SetSize2D(s[0], s[1])
                .SetFormat(f.first)
                .SetFlags(TextureFlags::LINEAR_RENDER_TARGET)
                .SetStride(stride);

            auto texAlignment = tb.GetStorageAlignment();
            auto texSize = tb.GetPaddedStorageSize();

            tb.SetStorage(&memPool, texAlignment);
            Texture tex;
            tex.Initialize(&tb);

            CopyRegion region = { 0, 0, 0, s[0], s[1], 1 };
            const float color[] = { 0.0f, 0.0f, 0.0f, 0.0f };

            queueCB.ClearTexture(&tex, nullptr, &region, color, ClearColorMask::RGBA);

            queueCB.submit();
            queue->Finish();

            results.push_back(verifyResult(memPoolPtr, texAlignment, texSize));

            tex.Finalize();
        }
    }

    memPool.Finalize();
    lwnUtil::AlignedStorageFree(asset);

    // Draw results
    const int cellWidth = lwrrentWindowWidth / 16;
    const int cellHeight = lwrrentWindowHeight / 12;

    int cellX = 0, cellY = 0;

    g_lwnWindowFramebuffer.bind();

    queueCB.SetViewportScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
    queueCB.ClearColor(0, 0.2f, 0.2f, 0.2f, 1.0f);

    for (auto r : results) {

        queueCB.SetViewportScissor(cellX + 1, cellY + 1, cellWidth - 1, cellHeight - 1);

        if (r) {
            queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
        }
        else {
            queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
        }

        cellX += cellWidth;
        if (cellX >= lwrrentWindowWidth) {
            cellX = 0;
            cellY += cellHeight;
        }
    }

    queueCB.submit();
    queue->Finish();
}


OGTEST_CppTest(LWNClearTexOverwrite, lwn_clear_tex_overwrite, );