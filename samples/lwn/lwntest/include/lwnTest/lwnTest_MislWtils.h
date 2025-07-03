/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_Mislwtils_h__
#define __lwnTest_Mislwtils_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#include "lwn/lwn_Cpp.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"

namespace lwnTest {


// Utilities for tests to report failure. The messages will be printed
// after the "test_name: FAIL! (...)" line in stdout and result file.
// Note that the test is still responsible for producing a failing
// result image for gilding.
bool failed();
void fail(const char *format, ...);
void printFailures(FILE *stream);
void resetFailedStatus();

// Utility method to allocate and fill a buffer object belonging to device
// <device>, copying <sizeofdata> bytes from <data>.  <access> specifies the
// access bits associated with the buffer.  If <useCopy> is true, the data
// should be streamed in via a copy on <queue> (from a second staging buffer);
// otherwise, the buffer should be copied to using a CPU mapping.
LWNbuffer *AllocAndFillBuffer(LWNdevice *device, LWNqueue *queue,
                              LWNcommandBuffer *cmdBuf,
                              lwnUtil::MemoryPoolAllocator& allocator,
                              const void *data, int sizeofdata,
                              lwnUtil::BufferAlignBits alignBits, bool useCopy);

// Utility method to allocate and fill a <format> 2D texture object
// belonging to device <device>, copying <width> * <height> elements of
// size <sizeofelement> bytes from <data> on <queue>.
LWNtexture *AllocAndFillTexture2D(LWNdevice *device, LWNqueue *queue,
                                  LWNcommandBuffer *cmdBuf,
                                  lwnUtil::MemoryPoolAllocator& tex_allocator,
                                  lwnUtil::MemoryPoolAllocator& coherent_allocator,
                                  const void *data, int sizeoftexel,
                                  int width, int height, LWNformat format);

// Utility method to set both viewport and scissor to the region defined by
// <cell_x>, <cell_y> with <padding> pixels of padding. Commands are issued
// on <cmd>.
void SetCellViewportScissorPadded(LWNcommandBuffer *cmd, int cell_x, int cell_y,
                                  int padding);

// Utility method to read a non-compressed texture back from <tex> to system 
// memory <texData>.
void ReadTextureData(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmd,
                     LWNtexture *tex, int width, int height, int depth, int bpp,
                     void *texData);

// Utility method to read an RGBA8 texture back from <tex> to system memory
// <texData>.
void ReadTextureDataRGBA8(LWNdevice *device, LWNqueue *queue, LWNcommandBuffer *cmd,
                          LWNtexture *tex, int width, int height, void *texData);

inline bool FormatIsDepthStencil(LWNformat format)
{
    return (format >= LWN_FORMAT_STENCIL8 && format <= LWN_FORMAT_DEPTH32F_STENCIL8);
}

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
//
// Utility functions providing a native C++ interface on top of the core
// utility functions, using reinterpret_cast to colwert between C and C++
// object types.
//
lwn::Buffer *AllocAndFillBuffer(lwn::Device *device, lwn::Queue *queue,
                                lwn::CommandBuffer *cmdBuf,
                                lwnUtil::MemoryPoolAllocator& allocator,
                                const void *data, int sizeofdata,
                                lwnUtil::BufferAlignBits alignBits, bool useCopy);

// Utility method to allocate and fill a <format> 2D texture object
// belonging to device <device>, copying <width> * <height> elements of
// size <sizeofelement> bytes from <data> on <queue>.
lwn::Texture *AllocAndFillTexture2D(lwn::Device *device, lwn::Queue *queue,
                                    lwn::CommandBuffer *cmdBuf,
                                    lwnUtil::MemoryPoolAllocator& tex_allocator,
                                    lwnUtil::MemoryPoolAllocator& coherent_allocator,
                                    const void *data, int sizeoftexel,
                                    int width, int height, lwn::Format format);

// Utility method to set both viewport and scissor to the region defined by
// <cell_x>, <cell_y> with <padding> pixels of padding. Commands are issued
// on <cmd>.
void SetCellViewportScissorPadded(lwn::CommandBuffer *cmd, int cell_x, int cell_y,
                                  int padding);

// Utility method to read a non-compressed texture back from <tex> to system 
// memory <texData>.
void ReadTextureData(lwn::Device *device, lwn::Queue *queue, lwn::CommandBuffer *cmd,
                     lwn::Texture *tex, int width, int height, int depth, int bpp,
                     void *texData);

// Utility method to read an RGBA8 texture back from <tex> to system memory
// <texData>.
void ReadTextureDataRGBA8(lwn::Device *device, lwn::Queue *queue, lwn::CommandBuffer *cmd,
                          lwn::Texture *tex, int width, int height, void *texData);

inline bool FormatIsDepthStencil(lwn::Format format)
{
    return (format >= lwn::Format::STENCIL8 && format <= lwn::Format::DEPTH32F_STENCIL8);
}

#endif

} // namespace lwnTest

#endif // #ifndef __lwnTest_Mislwtils_h__
