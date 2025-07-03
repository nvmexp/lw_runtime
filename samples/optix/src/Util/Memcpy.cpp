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

/*
   == WARNING - BEWARE GCC OPTIMIZATIONS ==

   This translation unit needs SSE4.1 and, at the time of writing this, not every system where OptiX runs on
   supports them. Having -msse4.1 in the compiler's command line means two things:

   - you can use SSE4.1 intrinsics and headers
   - GCC CAN OPTIMIZE AWAY WITH SSE4.1 ANYTHING IT LIKES

   OptiX can only afford the latter with a controlled runtime CpuSupportsSSE41() check. 

   The code below does this runtime check but IT IS NOT BULLET-PROOF AGAINST GCC'S OPTIMIZATION!

   An example:

    if( !CpuSupportsSSE41() )
    {
      return memcpy( dst, src, bytes ); // Standard fallback code
    }
    else
    {
      ... SSE4.1 code...
    }

  If gcc will ever decide to also optimize memcpy() with SSE4.1 instructions, no fallback code will be provided
  and OptiX will crash on older systems where there's no SSE4.1 support.

  Thus if you want to render this code 100% secure-from-crashing, it would need some refactoring to just compile 
  SSE4.1 code and have the fallback code into another translation unit.
*/

#include <Util/Memcpy.h>

#ifdef _WIN32
// In order to work around a VS9 bug, math.h needs to be included before intrin.h.
// Since this is the only place where this is done, we will pre-include math.h here.
// https://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=381422&wa=wsignin1.0
#include <intrin.h>  // need this for __cpuid intrinsic
#include <math.h>
#endif

#include <sse_support.h>

#ifdef SSE_41_AVAILABLE
#include <emmintrin.h>  // SSE2
#include <smmintrin.h>  // SSE4.1
#include <xmmintrin.h>  // SSE
#endif


#ifdef SSE_41_AVAILABLE

namespace optix {
static bool g_sse41supportDetermined = false;
static bool g_supportsSSE41;
}

#endif

#if defined( SSE_41_AVAILABLE )

void CpuID( int CPUInfo[4], int InfoType )
{
#ifdef _WIN32
    __cpuid( CPUInfo, InfoType );
#else
    // I found a better way to call cpuid that seems more reliable.  cpuid reads eax,
    // and writes to eax, ebx, ecx, and edx.  This is represented as the "=a", "=b",
    // etc. arguments to the assembly.
    __asm__ __volatile__( "cpuid"
                          : "=a"( CPUInfo[0] ), "=b"( CPUInfo[1] ), "=c"( CPUInfo[2] ), "=d"( CPUInfo[3] )
                          : "a"( InfoType ) );
#endif
}

// Determines if the CPU supports SSE4.1 instructions
bool optix::CpuSupportsSSE41()
{
    if( !g_sse41supportDetermined )
    {
        g_supportsSSE41 = false;
        int info[4];
        CpuID( info, 0 );  // cpuid argument of 0 gives us the max argument to cpuid
        int numIds = info[0];
        if( numIds > 0 )
        {
            CpuID( info, 1 );            // argument of 1 will give us sse info
            if( info[2] & ( 1 << 19 ) )  // sse4.1 support is in bit 19
            {
                g_supportsSSE41 = true;
            }
        }
        g_sse41supportDetermined = true;
    }

    return g_supportsSSE41;
}

void* optix::memcpy_sse( void* dst, const void* src, size_t bytes )
{
    // Don't use SSE copy if data is not 16-byte aligned or we copy less than 64 bytes.
    if( ( ( (size_t)dst | (size_t)src ) & 15 ) || bytes < 64 || !CpuSupportsSSE41() )
    {
        return memcpy( dst, src, bytes );
    }
    else
    {
        // Copy multiples of 64 bytes.
        const size_t size = bytes & ~( 64 - 1 );

        __m128i* sse_dst = (__m128i*)( (char*)dst );
        __m128i* sse_src = (__m128i*)( (char*)src );
        __m128i* src_end = (__m128i*)( (char*)src + size );

        do
        {
            _mm_prefetch( ( (char*)sse_src ) + 32 * sizeof( __m128i ), _MM_HINT_NTA );
            _mm_stream_si128( sse_dst + 0, _mm_stream_load_si128( sse_src + 0 ) );
            _mm_stream_si128( sse_dst + 1, _mm_stream_load_si128( sse_src + 1 ) );
            _mm_stream_si128( sse_dst + 2, _mm_stream_load_si128( sse_src + 2 ) );
            _mm_stream_si128( sse_dst + 3, _mm_stream_load_si128( sse_src + 3 ) );

            sse_src += 4;
            sse_dst += 4;
        } while( sse_src != src_end );

        // Copy remaining bytes if needed.
        int unalignedBytesToCopy = bytes & 63;
        if( unalignedBytesToCopy )
        {
            memcpy( (char*)dst + size, (const char*)src + size, unalignedBytesToCopy );
        }

        return dst;
    }
}
#else  // #ifdef SSE_41_AVAILABLE

void* optix::memcpy_sse( void* dst, const void* src, size_t bytes )
{
    return memcpy( dst, src, bytes );
}

bool CpuSupportsSSE41()
{
    return false;
}

#endif
