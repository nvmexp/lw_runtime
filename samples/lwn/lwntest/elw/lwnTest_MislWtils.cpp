/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_Mislwtils.h"

#include "lwnTest/lwnTest_Objects.h"

using namespace lwnUtil;

namespace lwnTest {

namespace {
    bool s_failed = false;
    lwString s_testOutputBuffer;
}

bool failed()
{
    return s_failed;
}

void fail(const char *format, ...)
{
    char tmp[256];
    va_list ap;
    va_start(ap, format);
    vsnprintf(tmp, sizeof(tmp), format, ap);
    va_end(ap);
    s_testOutputBuffer.append("   - ");
    s_testOutputBuffer.append(tmp);
    s_testOutputBuffer.append("\n");
    s_failed = true;
}
void printFailures(FILE *stream)
{
    assert(s_failed);
    fprintf(stream, "%s", s_testOutputBuffer.c_str());
}

void resetFailedStatus()
{
    s_failed = false;
    s_testOutputBuffer = "";
}

// Utility method to allocate and fill a buffer object belonging to device 
// <device>, copying <sizeofdata> bytes from <data> using allocator <allocator>.   
// <access> specifies the access bits associated with the buffer.  If <useCopy> 
// is true, the data should be streamed in via a copy on <queue> (from a 
// second staging buffer); otherwise, the buffer should be copied to using 
// a CPU mapping.
// Non-coherent allocations will be flushed.
LWNbuffer *AllocAndFillBuffer(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmdBuf,
                              MemoryPoolAllocator& allocator,
                              const void *data, int sizeofdata, 
                              BufferAlignBits alignBits, bool useCopy)
{
    LWNbufferBuilder bb;
    LWNbuffer *buffer;
    LWNmemoryPool *pool;
    void *ptr;

    lwnBufferBuilderSetDefaults(&bb);
    lwnBufferBuilderSetDevice(&bb, device);

    if (useCopy) {
        buffer = allocator.allocBuffer(&bb, BufferAlignBits(alignBits | BUFFER_ALIGN_COPY_WRITE_BIT), sizeofdata);

        assert(data);

        LWNbuffer *tempbo = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, sizeofdata);
        ptr = lwnBufferMap(tempbo);
        memcpy(ptr, data, sizeofdata);

        // allocator's pool may be non-coherent
        pool = allocator.pool(tempbo);
        if (pool) {
            lwnMemoryPoolFlushMappedRange(pool, allocator.offset(tempbo), allocator.size(tempbo));
        }

        lwnCommandBufferCopyBufferToBuffer(cmdBuf, lwnBufferGetAddress(tempbo), lwnBufferGetAddress(buffer),
                                           sizeofdata, LWN_COPY_FLAGS_NONE);

        // Flush command buffer contents before calling Finish; reopen the
        // command buffer for further recording.
        LWNcommandHandle handle = lwnCommandBufferEndRecording(cmdBuf);
        lwnQueueSubmitCommands(queue, 1, &handle);
        lwnCommandBufferBeginRecording(cmdBuf);
        lwnQueueFinish(queue);

        allocator.freeBuffer(tempbo);
    } else {
        buffer = allocator.allocBuffer(&bb, alignBits, sizeofdata);
        if (data) {
            ptr = lwnBufferMap(buffer);
            memcpy(ptr, data, sizeofdata);

            // allocator's pool may be non-coherent
            pool = allocator.pool(buffer);
            if (pool) {
                lwnMemoryPoolFlushMappedRange(pool, allocator.offset(buffer), allocator.size(buffer));
            }
        }
    }

    return buffer;
}

// Utility method to allocate and fill a <format> 2D texture object
// belonging to device <device>, copying <width> * <height> elements of
// size <sizeofelement> bytes from <data> on <queue>.
// Using two allocators to account for http://lwbugs/1655482 where
// one non-coherent pool allocator does not properly work. <tex_allocator> 
// can be non-coherent or GPU only but the <coherent_allocator> always 
// needs to be coherent!!! The coherent allocator must be able to 
// allocate a PBO of size <width> x <height> x <sizeoftexel>. The buffer 
// will be freed before the function returns.
LWNtexture *AllocAndFillTexture2D(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmdBuf,
                                  MemoryPoolAllocator& tex_allocator,
                                  MemoryPoolAllocator& coherent_allocator,
                                  const void *data, int sizeoftexel,
                                  int width, int height, LWNformat format)
{
    int size = width * height * sizeoftexel;
    LWNbuffer *pbo = AllocAndFillBuffer(device, queue, cmdBuf, coherent_allocator, data, size,
                                        BUFFER_ALIGN_COPY_READ_BIT, false);

    LWNtextureBuilder tb;
    LWNcopyRegion copyRegion = { 0, 0, 0, width, height, 1 };
    lwnTextureBuilderSetDefaults(&tb);
    lwnTextureBuilderSetDevice(&tb, device);
    lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D);
    lwnTextureBuilderSetFormat(&tb, format);
    lwnTextureBuilderSetSize2D(&tb, width, height);
    if (FormatIsDepthStencil(format)) {
        lwnTextureBuilderSetFlags(&tb, LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
    }

    LWNtexture *tex = tex_allocator.allocTexture(&tb);
    lwnCommandBufferCopyBufferToTexture(cmdBuf, lwnBufferGetAddress(pbo), tex, NULL, &copyRegion,
                                        LWN_COPY_FLAGS_NONE);

    // Flush command buffer contents before calling Finish; reopen the
    // command buffer for further recording.
    LWNcommandHandle handle = lwnCommandBufferEndRecording(cmdBuf);
    lwnQueueSubmitCommands(queue, 1, &handle);
    lwnCommandBufferBeginRecording(cmdBuf);
    lwnQueueFinish(queue);

    coherent_allocator.freeBuffer(pbo);
    return tex;
}

// Utility method to set both viewport and stencil to the region defined by
// <cell_x>, <cell_y> with <padding> pixels of padding. Commands are issued
// on <cmd>.
void SetCellViewportScissorPadded(LWNcommandBuffer *cmd, int cell_x, int cell_y,
                                  int padding)
{
    int view_x, view_y, view_width, view_height;
    cellGetRectPadded(cell_x, cell_y, padding, &view_x, &view_y, &view_width, &view_height);
    lwnCommandBufferSetViewport(cmd, view_x, view_y, view_width, view_height);
    lwnCommandBufferSetScissor(cmd, view_x, view_y, view_width, view_height);
}

void ReadTextureData(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmd,
                     LWNtexture *tex, int width, int height, int depth, int bpp,
                     void *texData)
{
    int size = width * height * depth * bpp;

    LWNbufferBuilder bb;
    lwnBufferBuilderSetDefaults(&bb);
    lwnBufferBuilderSetDevice(&bb, device);

    MemoryPoolAllocator readbackAllocator(device, NULL, size, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    LWNbuffer *readbackBuffer = readbackAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, size);

    ptrdiff_t rowStride = lwnCommandBufferGetCopyRowStride(cmd);
    ptrdiff_t imgStride = lwnCommandBufferGetCopyImageStride(cmd);
    lwnCommandBufferSetCopyRowStride(cmd, 0);
    lwnCommandBufferSetCopyImageStride(cmd, 0);

    LWNcopyRegion copyRegion = { 0, 0, 0, width, height, depth };
    lwnCommandBufferCopyTextureToBuffer(cmd, tex, NULL, &copyRegion,
                                        lwnBufferGetAddress(readbackBuffer), LWN_COPY_FLAGS_NONE);

    lwnCommandBufferSetCopyRowStride(cmd, rowStride);
    lwnCommandBufferSetCopyImageStride(cmd, imgStride);

    LWNcommandHandle handle = lwnCommandBufferEndRecording(cmd);
    lwnQueueSubmitCommands(queue, 1, &handle);
    lwnCommandBufferBeginRecording(cmd);
    lwnQueueFinish(queue);

    void *ptr = lwnBufferMap(readbackBuffer);
    memcpy(texData, ptr, size);
    readbackAllocator.freeBuffer(readbackBuffer);
}

void ReadTextureDataRGBA8(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmd,
                          LWNtexture *tex, int width, int height, void *texData)
{
    ReadTextureData(device, queue, cmd, tex, width, height, 1, 4, texData);
}

} // namespace lwnTest
