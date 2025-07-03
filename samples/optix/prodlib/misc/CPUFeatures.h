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

#pragma once

#include <stdint.h>

namespace prodlib {

#if defined( _MSC_VER) && (defined( _M_IX86 ) || defined( _M_X64 ) )
#define CPU_FEATURES_X86_MSC
#define CPU_FEATURES_X86
#endif

#if defined( __GNUC__) && defined( __x86_64 )
#define CPU_FEATURES_X86_GCC
#define CPU_FEATURES_X86
#endif

struct CPUFeatures
{
    CPUFeatures();

#if defined( CPU_FEATURES_X86_MSC )
    static bool SSE2() { return ( m_cpuFeatures.f1_3 & ( 1 << 26 ) ) != 0; }
    static bool SSE41() { return ( m_cpuFeatures.f1_2 & ( 1 << 19 ) ) != 0; }
    static bool AVX2() { return ( m_cpuFeatures.f7_1 & ( 1 << 5 ) ) != 0; }
    static bool AVX512F() { return ( m_cpuFeatures.f7_1 & ( 1 << 16 ) ) != 0; }

    uint32_t f1_2;  // SSE status bits ECX
    uint32_t f1_3;  // SSE status bits EDX
    uint32_t f7_1;  // AVX status bits EBX
#elif defined( CPU_FEATURES_X86_GCC )
    static bool SSE2() { return m_cpuFeatures.m_sse2; }
    static bool SSE41() { return m_cpuFeatures.m_sse41; }
    static bool AVX2() { return m_cpuFeatures.m_avx2; }
    static bool AVX512F() { return  m_cpuFeatures.m_avx512f; }

    bool m_sse2;  
    bool m_sse41; 
    bool m_avx2; 
    bool m_avx512f;
#endif


  private:
    static CPUFeatures m_cpuFeatures;
};
}
