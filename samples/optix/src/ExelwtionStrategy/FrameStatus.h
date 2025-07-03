// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

namespace cort {

struct FrameStatus
{
    // It might be useful to add other types or error codes as the need arises.  The reason
    // timedout is separate from failed, is to prevent a timeout even overwriting a failure
    // mode (using |= operator would also require a read of the failure bit which I want to
    // avoid).
    enum
    {
        FRAME_STATUS_NO_ERROR = 0,
        FRAME_STATUS_BAD_STATE
    };
    unsigned int failed;
    bool         timedout;
    int          i1, i2, i3, i4;
    float        f1, f2, f3, f4;
    void *       ptr1, *ptr2, *ptr3, *ptr4;
};
}
