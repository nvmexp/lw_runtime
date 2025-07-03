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

#include <string>


namespace optix {

// The details for these policies are found in PolicyDetails.cpp.
enum class MBufferPolicy
{
    readonly,
    readonly_discard_hostmem,
    readwrite,
    writeonly,

    readonly_raw,
    readonly_discard_hostmem_raw,
    readwrite_raw,
    writeonly_raw,

    readonly_lwdaInterop,
    readwrite_lwdaInterop,
    writeonly_lwdaInterop,

    readonly_lwdaInterop_copyOnDirty,
    readwrite_lwdaInterop_copyOnDirty,
    writeonly_lwdaInterop_copyOnDirty,

    readonly_demandload,
    texture_readonly_demandload,
    tileArray_readOnly_demandLoad,
    readonly_sparse_backing,

    gpuLocal,

    readonly_gfxInterop,
    readwrite_gfxInterop,
    writeonly_gfxInterop,

    texture_linear,
    texture_linear_discard_hostmem,
    texture_array,
    texture_array_discard_hostmem,
    texture_gfxInterop,

    internal_readonly,
    internal_readwrite,
    internal_writeonly,

    internal_hostonly,
    internal_readonly_deviceonly,

    internal_readonly_manualSync,
    internal_readwrite_manualSync,

    internal_texheapBacking,
    internal_preferTexheap,

    unused
};

std::string toString( MBufferPolicy policy );

// Given a "basic" policy, select a more specific one based on the given flags.
MBufferPolicy translatePolicy( MBufferPolicy policy, bool rawAccessRequired, bool lwdaInterop, bool copyOnDirty );

}  // namespace
