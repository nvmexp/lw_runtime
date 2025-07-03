// Copyright LWPU Corporation 2018-2019
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "Encryption.h"

#include <corelib/math/MathUtil.h>
#include <prodlib/misc/CPUFeatures.h>

#if defined( CPU_FEATURES_X86 ) && (_MSC_VER >= 1911 || __GNUC__ > 4 || ( __GNUC__ == 4 && __GNUC_MINOR >= 9 ) )    
#define X86_VECTORIZATION_AVAILABLE
#endif

#if defined( X86_VECTORIZATION_AVAILABLE )
#include <immintrin.h>
#endif

prodlib::CPUFeatures cpuFeatures;

// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
//
// Keep in sync with the copy in optix_ptx_encryption.h

static const uint32_t MAGIC = 0x9e3779b9;

static void encrypt( uint32_t* v, const uint32_t k[4] )
{
    uint32_t v0 = v[0];
    uint32_t v1 = v[1];
    uint32_t s0 = 0;

    for( uint32_t n = 0; n < 16; n++ )
    {
        s0 += MAGIC;
        v0 += ( ( v1 << 4 ) + k[0] ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + k[1] );
        v1 += ( ( v0 << 4 ) + k[2] ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + k[3] );
    }
    v[0] = v0;
    v[1] = v1;
}

static void decrypt_scalar( uint32_t* v, const uint32_t k[4] )
{
    uint32_t v0 = v[0];
    uint32_t v1 = v[1];
    uint32_t s0 = 0xe3779b90;  //MAGIC*16

    for( uint32_t n = 0; n < 16; n++ )
    {
        v1 -= ( ( v0 << 4 ) + k[2] ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + k[3] );
        v0 -= ( ( v1 << 4 ) + k[0] ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + k[1] );
        s0 -= MAGIC;
    }
    v[0] = v0;
    v[1] = v1;
}

#if defined( X86_VECTORIZATION_AVAILABLE )

// deocde 4 pairs at once
static void decrypt_sse( uint32_t* v, const uint32_t k[4] )
{
    __m128i s0 = _mm_set_epi32( 0xe3779b90, 0xe3779b90, 0xe3779b90, 0xe3779b90 );  //MAGIC*16 //SSE2
    __m128i kv[4];
    for( int i = 0; i < 4; ++i )
    {
        kv[i] = _mm_set_epi32( k[i], k[i], k[i], k[i] );
    }

    const __m128i MAGICv = _mm_set_epi32( MAGIC, MAGIC, MAGIC, MAGIC );  // SSE2

    __m128i v0p = _mm_loadu_si128( reinterpret_cast<__m128i*>( v ) );      // SSE2
    __m128i v1p = _mm_loadu_si128( reinterpret_cast<__m128i*>( v ) + 1 );  // SSE2

    // shuffle input
    // v0p: v0[0], v1[0], v0[1], v1[1]
    // vp1: v0[2], v1[2], v0[3], v1[3]
    // ->
    // v0p: v0[0], v0[1], v0[2], v0[3]
    // vp1: v1[0], v1[1], v1[2], v1[3]
    __m128i lo = _mm_unpacklo_epi32( v0p, v1p );  // SSE2
    __m128i hi = _mm_unpackhi_epi32( v0p, v1p );  // SSE2
    __m128i v0 = _mm_unpacklo_epi32( lo, hi );    // SSE2
    __m128i v1 = _mm_unpackhi_epi32( lo, hi );    // SSE2

    for( uint32_t n = 0; n < 16; n++ )
    {
        __m128i v0_l4    = _mm_slli_epi32( v0, 4 );                                                           // SSE2
        __m128i v0_l4_k2 = _mm_add_epi32( v0_l4, kv[2] );                                                     // SSE2
        __m128i v0_s0    = _mm_add_epi32( v0, s0 );                                                           // SSE2
        __m128i v0_r5    = _mm_srli_epi32( v0, 5 );                                                           // SSE2
        __m128i v0_r5_k3 = _mm_add_epi32( v0_r5, kv[3] );                                                     // SSE2
        v1               = _mm_sub_epi32( v1, _mm_xor_si128( _mm_xor_si128( v0_l4_k2, v0_s0 ), v0_r5_k3 ) );  // SSE2

        __m128i v1_l4    = _mm_slli_epi32( v1, 4 );                                                           // SSE2
        __m128i v1_l4_k0 = _mm_add_epi32( v1_l4, kv[0] );                                                     // SSE2
        __m128i v1_s0    = _mm_add_epi32( v1, s0 );                                                           // SSE2
        __m128i v1_r5    = _mm_srli_epi32( v1, 5 );                                                           // SSE2
        __m128i v1_r5_k1 = _mm_add_epi32( v1_r5, kv[1] );                                                     // SSE2
        v0               = _mm_sub_epi32( v0, _mm_xor_si128( _mm_xor_si128( v1_l4_k0, v1_s0 ), v1_r5_k1 ) );  // SSE2

        s0 = _mm_sub_epi32( s0, MAGICv );  // SSE2
    }

    // interleave v0 and v1 to store the result
    v0p = _mm_unpacklo_epi32( v0, v1 );  // SSE2
    v1p = _mm_unpackhi_epi32( v0, v1 );  // SSE2

    _mm_storeu_si128( reinterpret_cast<__m128i*>( v ), v0p );      // SSE2
    _mm_storeu_si128( reinterpret_cast<__m128i*>( v ) + 1, v1p );  // SSE2
}

// On GCC AVX needs a different ABI which is not enabled by default. Thus enable AVX only on msvc compilers.

#if defined( __GNUC__ )
__attribute__( ( target( "avx2" ) ) )
#endif
static void decrypt_avx2( uint32_t* v, const uint32_t k[4] )
{
    __m256i s0 = _mm256_set1_epi32( 0xe3779b90 );  //MAGIC*16
    __m256i kv[4];
    for( int i = 0; i < 4; ++i )
    {
        kv[i] = _mm256_set1_epi32( k[i] );
    }

    const __m256i MAGICv = _mm256_set1_epi32( MAGIC );

    __m256i v0p = _mm256_loadu_si256( reinterpret_cast<__m256i*>( v ) );
    __m256i v1p = _mm256_loadu_si256( reinterpret_cast<__m256i*>( v ) + 1 );

    // shuffle input
    // v0p: v0[0], v1[0], v0[1], v1[1]
    // vp1: v0[2], v1[2], v0[3], v1[3]
    // ->
    // v0p: v0[0], v0[1], v0[2], v0[3]
    // vp1: v1[0], v1[1], v1[2], v1[3]
    __m256i lo = _mm256_unpacklo_epi32( v0p, v1p );  // AVX2
    __m256i hi = _mm256_unpackhi_epi32( v0p, v1p );  // AVX2
    __m256i v0 = _mm256_unpacklo_epi32( lo, hi );    // AVX2
    __m256i v1 = _mm256_unpackhi_epi32( lo, hi );    // AVX2

    for( uint32_t n = 0; n < 16; n++ )
    {
        __m256i v0_l4    = _mm256_slli_epi32( v0, 4 );                                                   // AVX2
        __m256i v0_l4_k2 = _mm256_add_epi32( v0_l4, kv[2] );                                             // AVX2
        __m256i v0_s0    = _mm256_add_epi32( v0, s0 );                                                   // AVX2
        __m256i v0_r5    = _mm256_srli_epi32( v0, 5 );                                                   // AVX2
        __m256i v0_r5_k3 = _mm256_add_epi32( v0_r5, kv[3] );                                             // AVX2
        v1 = _mm256_sub_epi32( v1, _mm256_xor_si256( _mm256_xor_si256( v0_l4_k2, v0_s0 ), v0_r5_k3 ) );  // AVX2

        __m256i v1_l4    = _mm256_slli_epi32( v1, 4 );                                                   // AVX2
        __m256i v1_l4_k0 = _mm256_add_epi32( v1_l4, kv[0] );                                             // AVX2
        __m256i v1_s0    = _mm256_add_epi32( v1, s0 );                                                   // AVX2
        __m256i v1_r5    = _mm256_srli_epi32( v1, 5 );                                                   // AVX2
        __m256i v1_r5_k1 = _mm256_add_epi32( v1_r5, kv[1] );                                             // AVX2
        v0 = _mm256_sub_epi32( v0, _mm256_xor_si256( _mm256_xor_si256( v1_l4_k0, v1_s0 ), v1_r5_k1 ) );  // AVX2

        s0 = _mm256_sub_epi32( s0, MAGICv );  // AVX2
    }

    // interleave v0 and v1
    v0p = _mm256_unpacklo_epi32( v0, v1 );  // AVX2
    v1p = _mm256_unpackhi_epi32( v0, v1 );  // AVX2

    _mm256_storeu_si256( reinterpret_cast<__m256i*>( v ), v0p );
    _mm256_storeu_si256( reinterpret_cast<__m256i*>( v ) + 1, v1p );
}

// deocde 16 pairs at once
#if defined(  __GNUC__  )
__attribute__( ( target( "avx512f" ) ) )
#endif
static void decrypt_avx512( uint32_t* v, const uint32_t k[4] )
{
    __m512i s0 = _mm512_set1_epi32( 0xe3779b90 );  //MAGIC*16
    __m512i kv[4];
    for( int i = 0; i < 4; ++i )
    {
        kv[i] = _mm512_set1_epi32( k[i] );
    }

    const __m512i MAGICv = _mm512_set1_epi32( MAGIC );

    __m512i v0p = _mm512_loadu_si512( reinterpret_cast<__m512i*>( v ) );
    __m512i v1p = _mm512_loadu_si512( reinterpret_cast<__m512i*>( v ) + 1 );

    // shuffle input
    // v0p: v0[0], v1[0], v0[1], v1[1]
    // vp1: v0[2], v1[2], v0[3], v1[3]
    // ->
    // v0p: v0[0], v0[1], v0[2], v0[3]
    // vp1: v1[0], v1[1], v1[2], v1[3]
    __m512i lo = _mm512_unpacklo_epi32( v0p, v1p );  // AVX512F
    __m512i hi = _mm512_unpackhi_epi32( v0p, v1p );  // AVX512F
    __m512i v0 = _mm512_unpacklo_epi32( lo, hi );    // AVX512F
    __m512i v1 = _mm512_unpackhi_epi32( lo, hi );    // AVX512F

    for( uint32_t n = 0; n < 16; n++ )
    {
        __m512i v0_l4    = _mm512_slli_epi32( v0, 4 );                                                   // AVX512F
        __m512i v0_l4_k2 = _mm512_add_epi32( v0_l4, kv[2] );                                             // AVX512F
        __m512i v0_s0    = _mm512_add_epi32( v0, s0 );                                                   // AVX512F
        __m512i v0_r5    = _mm512_srli_epi32( v0, 5 );                                                   // AVX512F
        __m512i v0_r5_k3 = _mm512_add_epi32( v0_r5, kv[3] );                                             // AVX512F
        v1 = _mm512_sub_epi32( v1, _mm512_xor_si512( _mm512_xor_si512( v0_l4_k2, v0_s0 ), v0_r5_k3 ) );  // AVX512F

        __m512i v1_l4    = _mm512_slli_epi32( v1, 4 );                                                   // AVX512F
        __m512i v1_l4_k0 = _mm512_add_epi32( v1_l4, kv[0] );                                             // AVX512F
        __m512i v1_s0    = _mm512_add_epi32( v1, s0 );                                                   // AVX512F
        __m512i v1_r5    = _mm512_srli_epi32( v1, 5 );                                                   // AVX512F
        __m512i v1_r5_k1 = _mm512_add_epi32( v1_r5, kv[1] );                                             // AVX512F
        v0 = _mm512_sub_epi32( v0, _mm512_xor_si512( _mm512_xor_si512( v1_l4_k0, v1_s0 ), v1_r5_k1 ) );  // AVX512F

        s0 = _mm512_sub_epi32( s0, MAGICv );  // AVX512F
    }

    // interleave v0 and v1
    v0p = _mm512_unpacklo_epi32( v0, v1 );  // AVX512F
    v1p = _mm512_unpackhi_epi32( v0, v1 );  // AVX512F

    _mm512_storeu_si512( reinterpret_cast<__m512i*>( v ), v0p );
    _mm512_storeu_si512( reinterpret_cast<__m512i*>( v ) + 1, v1p );
}
#endif

static const unsigned char KEY[7] = {164, 195, 147, 255, 203, 161, 184};

void prodlib::tea_encrypt( unsigned char* data, size_t sz, const uint32_t key[4] )
{
    // Encrypt 8 byte blocks with TEA

    const size_t n = corelib::idivFloor( sz, static_cast<size_t>( 8 ) );
    uint32_t*    v = reinterpret_cast<uint32_t*>( data );
    for( size_t i = 0; i < n; ++i )
    {
        encrypt( &v[2 * i], key );
    }

    // Slightly obfuscate leftover bytes (at most 7) with simple xor.

    for( size_t i = 8 * n, k = 0; i < sz; ++i, ++k )
    {
        data[i] = data[i] ^ KEY[k];
    }
}

namespace {
using fn_tea_decrypt = void ( * )( unsigned char* data, size_t sz, const uint32_t key[4] );
using fn_decrypt     = void ( * )( uint32_t* v, const uint32_t key[4] );
}

template <int VECTOR_SIZE, fn_decrypt decrypt>
static void tea_decrypt_vectorized( unsigned char* data, size_t sz, const uint32_t key[4] )
{
    // Decrypt 8 byte blocks with TEA
    const size_t n = corelib::idivFloor( sz, static_cast<size_t>( 8 ) );
    uint32_t*    v = reinterpret_cast<uint32_t*>( data );

    for( size_t i = 0; i < n / VECTOR_SIZE; ++i )
    {
        decrypt( v, key );
        v += 2 * VECTOR_SIZE;
    }

    // decode remainer elements which do not form a full vector
    for( size_t i = 0; i < n % VECTOR_SIZE; ++i )
    {
        decrypt_scalar( v, key );
        v += 2;
    }

    // Leftover blocks (at most 7) are obfuscated with simple xor
    for( size_t i = 8 * n, k = 0; i < sz; ++i, ++k )
    {
        data[i] = data[i] ^ KEY[k];
    }
}

static fn_tea_decrypt determine_tea_decrypt_fn()
{
#if defined( X86_VECTORIZATION_AVAILABLE )
    if( cpuFeatures.AVX512F() )
    {
        return tea_decrypt_vectorized<16, decrypt_avx512>;
    }
    if( cpuFeatures.AVX2() )
    {
        return tea_decrypt_vectorized<8, decrypt_avx2>;
    }
    if( cpuFeatures.SSE2() )
    {
        return tea_decrypt_vectorized<4, decrypt_sse>;
    }
#endif
    return tea_decrypt_vectorized<1, decrypt_scalar>;
}

void prodlib::tea_decrypt( unsigned char* data, size_t sz, const uint32_t key[4] )
{
    static fn_tea_decrypt fn = determine_tea_decrypt_fn();
    fn( data, sz, key );
}
