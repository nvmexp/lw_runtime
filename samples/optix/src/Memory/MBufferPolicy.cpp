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

#include <Memory/MBufferPolicy.h>

using namespace optix;

#define CASE( enum_ )                                                                                                  \
    case MBufferPolicy::enum_:                                                                                         \
        return #enum_

std::string optix::toString( MBufferPolicy policy )
{
    switch( policy )
    {
        CASE( readonly );
        CASE( readonly_discard_hostmem );
        CASE( readwrite );
        CASE( writeonly );

        CASE( readonly_raw );
        CASE( readonly_discard_hostmem_raw );
        CASE( readwrite_raw );
        CASE( writeonly_raw );

        CASE( readonly_lwdaInterop );
        CASE( readwrite_lwdaInterop );
        CASE( writeonly_lwdaInterop );

        CASE( readonly_lwdaInterop_copyOnDirty );
        CASE( readwrite_lwdaInterop_copyOnDirty );
        CASE( writeonly_lwdaInterop_copyOnDirty );

        CASE( readonly_demandload );
        CASE( texture_readonly_demandload );
        CASE( tileArray_readOnly_demandLoad );
        CASE( readonly_sparse_backing );

        CASE( gpuLocal );

        CASE( readonly_gfxInterop );
        CASE( readwrite_gfxInterop );
        CASE( writeonly_gfxInterop );

        CASE( texture_linear );
        CASE( texture_linear_discard_hostmem );
        CASE( texture_array );
        CASE( texture_array_discard_hostmem );
        CASE( texture_gfxInterop );

        CASE( internal_readonly );
        CASE( internal_readwrite );
        CASE( internal_writeonly );

        CASE( internal_hostonly );
        CASE( internal_readonly_deviceonly );

        CASE( internal_readonly_manualSync );
        CASE( internal_readwrite_manualSync );

        CASE( internal_texheapBacking );
        CASE( internal_preferTexheap );

        CASE( unused );

        // Default case intentionally omitted
    }
    return "Invalid policy";
}

#undef CASE

MBufferPolicy optix::translatePolicy( MBufferPolicy policy, bool rawAccessRequired, bool lwdaInterop, bool copyOnDirty )
{
    // Note that it's possible to have copyOnDirty==true even for
    // non-LWCA-interop buffers, since the user can create a buffer with the
    // copy-on-dirty flag, but never call get/set pointer on it. In that case,
    // we just behave like a normal buffer.

    if( lwdaInterop && copyOnDirty )
    {
        switch( policy )
        {
            case MBufferPolicy::readonly:
                return MBufferPolicy::readonly_lwdaInterop_copyOnDirty;
            case MBufferPolicy::readwrite:
                return MBufferPolicy::readwrite_lwdaInterop_copyOnDirty;
            case MBufferPolicy::writeonly:
                return MBufferPolicy::writeonly_lwdaInterop_copyOnDirty;
            default:;
        }
    }
    else if( lwdaInterop && !copyOnDirty )
    {
        switch( policy )
        {
            case MBufferPolicy::readonly:
                return MBufferPolicy::readonly_lwdaInterop;
            case MBufferPolicy::readwrite:
                return MBufferPolicy::readwrite_lwdaInterop;
            case MBufferPolicy::writeonly:
                return MBufferPolicy::writeonly_lwdaInterop;
            default:;
        }
    }
    else if( rawAccessRequired )
    {
        switch( policy )
        {
            case MBufferPolicy::readonly:
                return MBufferPolicy::readonly_raw;
            case MBufferPolicy::readonly_discard_hostmem:
                return MBufferPolicy::readonly_discard_hostmem_raw;
            case MBufferPolicy::readwrite:
                return MBufferPolicy::readwrite_raw;
            case MBufferPolicy::writeonly:
                return MBufferPolicy::writeonly_raw;
            default:;
        }
    }

    // Leave other policies untouched.
    return policy;
}
