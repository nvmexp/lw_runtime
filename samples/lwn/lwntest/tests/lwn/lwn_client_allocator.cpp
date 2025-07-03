/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#include <time.h>

//////////////////////////////////////////////////////////////////////////

using namespace lwn;

class LWNClientAllocator
{
    static const int cellSize = 20;
    static const int cellMargin = 1;
    static const int cellsX = 20;
    static const int cellsY = 10;

public:
    LWNTEST_CppMethods();

    LWNClientAllocator(bool expanding_pool) 
        : _expanding_pool(expanding_pool) {
    }

private:
    void drawResult(CellIterator2D& cell, bool result) const;

    bool    _expanding_pool;
};

int LWNClientAllocator::isSupported() const
{
    return lwogCheckLWNAPIVersion(11, 0);
}

lwString LWNClientAllocator::getDescription() const
{
    return "test client side memory pool allocator"
           "MemoryPoolAllocator is used to simulate a real application's memory allocation pattern."
           "It is a simple list-based memory allocator that allows non-overlapping buffer and texture"
           "allocation from memory pools, wrapping this in a C++ class.";
}

// show result as red/green square
void LWNClientAllocator::drawResult(CellIterator2D& cell, bool result) const{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;

    g_lwnWindowFramebuffer.bind();

    LWNfloat color[] = { 0.0, 0.0, 0.0, 1.0 };
    if (!result) {
        // shade of red
        color[0] = 1.0;
    } else {
        // green
        color[1] = 1.0;
    }
    // scissor
    queueCB.SetScissor(cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
                      cellSize - 2*cellMargin, cellSize - 2*cellMargin);

    // clear
    queueCB.ClearColor(0, color, LWN_CLEAR_COLOR_MASK_RGBA);

    // advance in grid
    cell++;
}

void LWNClientAllocator::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();

    CellIterator2D cell(cellsX, cellsY);

    // clear
    LWNfloat clearColor[] = {0.3, 0.3, 0.3, 1};

    queueCB.ClearColor(0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);

    const int memSize = _expanding_pool ? 0 : 0x10000; // limit to 64k to allow for a decent test runtime
    struct _poolTypes 
    {
        LWNmemoryPoolFlags  poolFlags;
        bool                mappable;
        bool                texture;
    }poolTypes[] = 
    { 
        { LWN_MEMORY_POOL_TYPE_CPU_COHERENT,        true,   false}, // coherent pool allows no textures
        { LWN_MEMORY_POOL_TYPE_CPU_NON_COHERENT,    true,   true},
        { LWN_MEMORY_POOL_TYPE_GPU_ONLY,            false,  true}
    };

    bool result;
    for (size_t poolType = 0; poolType < __GL_ARRAYSIZE(poolTypes); poolType++) {

        MemoryPoolAllocator allocator(device, NULL, memSize, poolTypes[poolType].poolFlags);

        BufferBuilder builder;
        builder.SetDevice(device).SetDefaults();

        if (memSize) {
            Buffer *buffer = allocator.allocBuffer(&builder, BUFFER_ALIGN_COPY_WRITE_BIT, memSize);
            result = buffer ? true : false;
            drawResult(cell, result);

            allocator.discard();

            // over-allocate by one byte, should create a new pool
            buffer = allocator.allocBuffer(&builder, BUFFER_ALIGN_COPY_WRITE_BIT, memSize + 1);
            result = buffer ? true : false;
            drawResult(cell, result);
        }

        allocator.discard();

        // allocate size 0
        Buffer *buffer = allocator.allocBuffer(&builder, BUFFER_ALIGN_COPY_WRITE_BIT, 0);
        result = buffer ? false : true;
        drawResult(cell, result);

        // allocate some buffers, check allocation count
        Buffer *bo[100];
        for (unsigned int i = 0; i< __GL_ARRAYSIZE(bo); i++) {
            bo[i] = allocator.allocBuffer(&builder, BUFFER_ALIGN_COPY_WRITE_BIT, sizeof(float));
        }

        result = (allocator.numAllocs() == __GL_ARRAYSIZE(bo));
        drawResult(cell, result);

        for (unsigned int i = 0; i< __GL_ARRAYSIZE(bo); i++) {
            allocator.freeBuffer(bo[i]);
        }

        result = (allocator.numAllocs() == 0);
        drawResult(cell, result);

        // throw away everything w/o having to keep track
        // of handles
        allocator.discard();

        if (poolTypes[poolType].texture) {
            TextureBuilder textureBuilder;
            const int texWidth = 4, texHeight = 4;

            textureBuilder.SetDevice(device).SetDefaults().SetTarget(TextureTarget::TARGET_2D).
                SetFormat(Format::RGBA8).SetSize2D(texWidth, texHeight);

            // allocate some textures, check allocation count
            Texture *tex[100];
            for (unsigned int i = 0; i< __GL_ARRAYSIZE(tex); i++) {
                tex[i] = allocator.allocTexture(&textureBuilder);
            }

            result = (allocator.numAllocs() == __GL_ARRAYSIZE(tex));
            drawResult(cell, result);

            for (unsigned int i = 0; i< __GL_ARRAYSIZE(tex); i++) {
                allocator.freeTexture(tex[i]);
            }

            result = (allocator.numAllocs() == 0);
            drawResult(cell, result);

            // throw away everything w/o having to keep track
            // of handles
            allocator.discard();

            queueCB.submit();
        }
    }

}

OGTEST_CppTest(LWNClientAllocator, lwn_client_allocator, (false));
OGTEST_CppTest(LWNClientAllocator, lwn_client_allocator_expand, (true));
