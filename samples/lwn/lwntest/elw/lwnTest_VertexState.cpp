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
#include "lwnTest/lwnTest_VertexState.h"

#include "lwnTest/lwnTest_Mislwtils.h"

namespace lwnTest {

// Utility method to allocate and optionally fill a vertex buffer holding
// <lwertices> worth of data for the stream, provided by <data> (if non-NULL).
LWNbuffer *VertexStream::AllocateVertexBuffer(LWNdevice *device, int lwertices,
                                              lwnUtil::MemoryPoolAllocator& allocator,
                                              const void *data /*= NULL*/)
{
    return lwnTest::AllocAndFillBuffer(device, NULL, NULL, allocator, data, lwertices * m_stride,
                                       lwnUtil::BUFFER_ALIGN_VERTEX_BIT, false);
}

} // namespace lwnTest
