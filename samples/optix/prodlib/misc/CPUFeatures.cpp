// Copyright LWPU Corporation 2019
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <prodlib/misc/CPUFeatures.h>

#if defined( CPU_FEATURES_X86_MSC )
#include <intrin.h>
#endif

namespace prodlib {

CPUFeatures CPUFeatures::m_cpuFeatures;

#if defined( CPU_FEATURES_X86_MSC )
static void CpuID( int CPUInfo[4], int function, int subfunction )
{
#ifdef _WIN32
    __cpuidex( CPUInfo, function, subfunction );
#else
    // I found a better way to call cpuid that seems more reliable.  cpuid reads eax,
    // and writes to eax, ebx, ecx, and edx.  This is represented as the "=a", "=b",
    // etc. arguments to the assembly.
    __asm__ __volatile__( "cpuid"
                          : "=a"( CPUInfo[0] ), "=b"( CPUInfo[1] ), "=c"( CPUInfo[2] ), "=d"( CPUInfo[3] )
                          : "a"( function ), "c"( subfunction ) );
#endif
}
#endif

CPUFeatures::CPUFeatures()
{
#if defined( CPU_FEATURES_X86_MSC )
    // https://en.wikipedia.org/wiki/CPUID has a very good summary of all CPU feature bits
    int info[4];
    CpuID( info, 0, 0 );  // cpuid argument of 0 gives us the max argument to cpuid
    int numFunctions = info[0];
    // SSE feature bits are in function 1
    if( numFunctions >= 1 )
    {
        CpuID( info, 1, 0 );  // argument of 1 will give us sse info
        f1_2 = info[2];
        f1_3 = info[3];
    }
    // AVX feature bits are in function 7
    if( numFunctions >= 7 )
    {
        CpuID( info, 7, 0 );  // argument of 1 will give us sse info
        f7_1 = info[1];
    }
#endif

#if defined( CPU_FEATURES_X86_GCC )
    m_sse2 = __builtin_cpu_supports ("sse2");
    m_sse41 = __builtin_cpu_supports ("sse4.1");
    m_avx2 = __builtin_cpu_supports ("avx2");
    m_avx512f = __builtin_cpu_supports ("avx512f");
#endif
}
}
