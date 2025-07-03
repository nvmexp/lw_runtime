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

#include <internal/optix_defines.h>

#if defined(_WIN64) || defined(__LP64__)
#define KERNEL_64BIT
#endif

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5

// Set this to the maximum threads on any architecture, i.e.
// don't use __LWDA_ARCH__ to select between Fermi/Tesla etc,
// since we might be compiling with a lower -arch than what
// we are running on.
// Max threads/CTA (not necessarily the same as max threads/SM) is:
//   512    on Tesla
//  1024    on Fermi
//  1024    on Kepler
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARPS_PER_BLOCK ( MAX_THREADS_PER_BLOCK / WARP_SIZE )


unsigned int __device__ __forceinline__ get_thread_idx_x()
{
    unsigned int val;
    asm( "  mov.b32 %0,%tid.x;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_thread_idx_y()
{
    unsigned int val;
    asm( "  mov.b32 %0,%tid.y;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_thread_idx_z()
{
    unsigned int val;
    asm( "  mov.b32 %0,%tid.z;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_block_idx_x()
{
    unsigned int val;
    asm( "  mov.b32 %0,%ctaid.x;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_block_idx_y()
{
    unsigned int val;
    asm( "  mov.b32 %0,%ctaid.y;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_block_dim_x()
{
    unsigned int val;
    asm( "  mov.b32 %0,%ntid.x;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_block_dim_y()
{
    unsigned int val;
    asm( "  mov.b32 %0,%ntid.y;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_block_dim_z()
{
    unsigned int val;
    asm( "  mov.b32 %0,%ntid.z;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_grid_dim_x()
{
    unsigned int val;
    asm( "  mov.b32 %0,%nctaid.x;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_grid_dim_y()
{
    unsigned int val;
    asm( "  mov.b32 %0,%nctaid.y;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_grid_dim_z()
{
    unsigned int val;
    asm( "  mov.b32 %0,%nctaid.z;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_laneid()
{
    unsigned int val;
    asm( "  mov.u32 %0,%laneid;" : "=r"( val ) : : );
    return val;
}

unsigned int __device__ __forceinline__ get_warpid()
{
    return get_thread_idx_y();
}

unsigned int __device__ __forceinline__ get_num_warps()
{
    return get_block_dim_y();
}

// A predefined, read-only special register that returns the processor (SM) identifier on
// which a particular thread is exelwting.  Notes SM identifier numbering is not
// guaranteed to be contiguous.
unsigned int __device__ __forceinline__ get_sm_id()
{
    unsigned int val;
    asm( "  mov.u32  %0, %smid;" : "=r"( val ) : : );
    return val;
}

//
// Smem instructions
//

template <typename T>
void __device__ __forceinline__ store_shared( void* ptr, T data )
{
}

template <>
void __device__ __forceinline__ store_shared<char>( void* ptr, char s8 )
{
    asm volatile( "st.shared.s8 [%0],%1;" : : OPTIX_ASM_PTR( ptr ), "r"( (int)s8 ) : );
}

template <>
void __device__ __forceinline__ store_shared<unsigned char>( void* ptr, unsigned char u8 )
{
    asm volatile( "st.shared.u8 [%0],%1;" : : OPTIX_ASM_PTR( ptr ), "r"( (unsigned int)u8 ) : );
}

template <>
void __device__ __forceinline__ store_shared<int>( void* ptr, int s32 )
{
    asm volatile( "st.shared.s32 [%0],%1;" : : OPTIX_ASM_PTR( ptr ), "r"( s32 ) : );
}

template <>
void __device__ __forceinline__ store_shared<unsigned int>( void* ptr, unsigned int u32 )
{
    asm volatile( "st.shared.u32 [%0],%1;" : : OPTIX_ASM_PTR( ptr ), "r"( u32 ) : );
}

//
// Video instructions
//

/*
   WARNING:
     Floats need to be interpreted as integers to be used with video instructions.
     This means that comparing positive/positive or positive/negative numbers in
     a min/max instruction yields the correct result, but comparing two negative
     numbers yields a wrong value.
     This is not a problem for tests in traversal since we ignore negatve t values.
*/

float __device__ __forceinline__ fvminminf( float a, float b, float c )
{
    int val;
    asm volatile( "vmin.s32.s32.s32.min %0, %1, %2, %3;"
                  : "=r"( val )
                  : "r"( __float_as_int( a ) ), "r"( __float_as_int( b ) ), "r"( __float_as_int( c ) ) );
    return __int_as_float( val );
}

float __device__ __forceinline__ fvminmaxf( float a, float b, float c )
{
    int val;
    asm volatile( "vmin.s32.s32.s32.max %0, %1, %2, %3;"
                  : "=r"( val )
                  : "r"( __float_as_int( a ) ), "r"( __float_as_int( b ) ), "r"( __float_as_int( c ) ) );
    return __int_as_float( val );
}

float __device__ __forceinline__ fvmaxmaxf( float a, float b, float c )
{
    int val;
    asm volatile( "vmax.s32.s32.s32.max %0, %1, %2, %3;"
                  : "=r"( val )
                  : "r"( __float_as_int( a ) ), "r"( __float_as_int( b ) ), "r"( __float_as_int( c ) ) );
    return __int_as_float( val );
}

float __device__ __forceinline__ fvmaxminf( float a, float b, float c )
{
    int val;
    asm volatile( "vmax.s32.s32.s32.min %0, %1, %2, %3;"
                  : "=r"( val )
                  : "r"( __float_as_int( a ) ), "r"( __float_as_int( b ) ), "r"( __float_as_int( c ) ) );
    return __int_as_float( val );
}

float __device__ __forceinline__ fvminf( float3 f )
{
    return fvminminf( f.x, f.y, f.z );
}

float __device__ __forceinline__ fvmaxf( float3 f )
{
    return fvmaxmaxf( f.x, f.y, f.z );
}


template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Size1, typename Size2, typename T1, typename T2>
__device__ void warp_multi_fill_n( RandomAccessIterator1 result1, RandomAccessIterator2 result2, Size1 n, Size2 lane_idx, T1 value1, T2 value2 )
{
    for( Size2 i = lane_idx; i < n; i += WARP_SIZE )
    {
        result1[i] = value1;
        result2[i] = value2;
    }
}

template <typename RandomAccessIterator, typename Size1, typename Size2, typename T>
__device__ void warpfill_n( RandomAccessIterator result, Size1 n, Size2 lane_idx, T value )
{
    for( Size2 i = lane_idx; i < n; i += WARP_SIZE )
    {
        result[i] = value;
    }
}

template <class T, bool inclusive>
__device__ T scanwarp( T* ptr )
{
    int       idx      = get_thread_idx_x();
    const int maxlevel = LOG_WARP_SIZE;

    T oval;
    if( !inclusive )
        oval = ptr[idx];

    if( 0 <= maxlevel )
    {
        if( ( idx & 31 ) >= 1 )
            ptr[idx] += ptr[idx - 1];
    }
    if( 1 <= maxlevel )
    {
        if( ( idx & 31 ) >= 2 )
            ptr[idx] += ptr[idx - 2];
    }
    if( 2 <= maxlevel )
    {
        if( ( idx & 31 ) >= 4 )
            ptr[idx] += ptr[idx - 4];
    }
    if( 3 <= maxlevel )
    {
        if( ( idx & 31 ) >= 8 )
            ptr[idx] += ptr[idx - 8];
    }
    if( 4 <= maxlevel )
    {
        if( ( idx & 31 ) >= 16 )
            ptr[idx] += ptr[idx - 16];
    }

    if( inclusive )
    {
        return ptr[idx];
    }
    else
    {
        return ptr[idx] - oval;  // colwert inclusive -> exclusive
    }
}

// shared's size MUST be at least MAX_THREADS_PER_BLOCK.
template <typename T>
__forceinline__ __device__ unsigned int warpcount( const unsigned int lane_idx, const unsigned int warp_idx, T x, T value, volatile unsigned int* shared )
{
    // point to this warp's data
    volatile unsigned int* warp_data = shared + WARP_SIZE * warp_idx;

    unsigned int y;
    warp_data[lane_idx] = y = ( value == x );
    warp_data[lane_idx] = y = y + warp_data[lane_idx | 1];
    warp_data[lane_idx] = y = y + warp_data[lane_idx | 2];
    warp_data[lane_idx] = y = y + warp_data[lane_idx | 4];
    warp_data[lane_idx] = y = y + warp_data[lane_idx | 8];
    warp_data[lane_idx] = y = y + warp_data[lane_idx | 16];
    return warp_data[0];  // Get result from lane 0
}

// shared's size MUST be at least MAX_THREADS_PER_BLOCK.
template <typename T>
__forceinline__ __device__ T warpmin( T x, volatile T* shared )
{
    // point to this warp's data
    volatile T* warp_data = shared + WARP_SIZE * get_warpid();

    const unsigned int lane_idx = get_laneid();

    warp_data[lane_idx] = x;
    warp_data[lane_idx] = x = min( x, warp_data[lane_idx | 1] );
    warp_data[lane_idx] = x = min( x, warp_data[lane_idx | 2] );
    warp_data[lane_idx] = x = min( x, warp_data[lane_idx | 4] );
    warp_data[lane_idx] = x = min( x, warp_data[lane_idx | 8] );
    warp_data[lane_idx] = x = min( x, warp_data[lane_idx | 16] );
    return warp_data[0];  // Get result from lane 0
}

template <>
__forceinline__ __device__ unsigned int warpmin( unsigned int x, volatile unsigned int* )
{
    unsigned int y;

    // Below fluff is just to keep the compiler happy (782216)
    unsigned int& _x = x;
    _x               = _x;
    unsigned int& _y = y;
    _y               = _y;
    // </fluff>

    asm( "  shfl.bfly.b32 %0, %1, 0x1,  0x1f;" : "=r"( y ) : "r"( x ) );
    asm( "  min.u32 %0, %1, %2;" : "=r"( x ) : "r"( x ), "r"( y ) );
    asm( "  shfl.bfly.b32 %0, %1, 0x2,  0x1f;" : "=r"( y ) : "r"( x ) );
    asm( "  min.u32 %0, %1, %2;" : "=r"( x ) : "r"( x ), "r"( y ) );
    asm( "  shfl.bfly.b32 %0, %1, 0x4,  0x1f;" : "=r"( y ) : "r"( x ) );
    asm( "  min.u32 %0, %1, %2;" : "=r"( x ) : "r"( x ), "r"( y ) );
    asm( "  shfl.bfly.b32 %0, %1, 0x8,  0x1f;" : "=r"( y ) : "r"( x ) );
    asm( "  min.u32 %0, %1, %2;" : "=r"( x ) : "r"( x ), "r"( y ) );
    asm( "  shfl.bfly.b32 %0, %1, 0x10, 0x1f;" : "=r"( y ) : "r"( x ) );
    asm( "  min.u32 %0, %1, %2;" : "=r"( x ) : "r"( x ), "r"( y ) );
    return x;
}


// shared's size MUST be at least MAX_THREADS_PER_BLOCK.
template <typename T>
__forceinline__ __device__ T warpmax( const unsigned int lane_idx, const unsigned int warp_idx, T x, volatile T* shared )
{
    // point to this warp's data
    volatile T* warp_data = shared + WARP_SIZE * warp_idx;

    warp_data[lane_idx] = x;
    warp_data[lane_idx] = x = max( x, warp_data[lane_idx | 1] );
    warp_data[lane_idx] = x = max( x, warp_data[lane_idx | 2] );
    warp_data[lane_idx] = x = max( x, warp_data[lane_idx | 4] );
    warp_data[lane_idx] = x = max( x, warp_data[lane_idx | 8] );
    warp_data[lane_idx] = x = max( x, warp_data[lane_idx | 16] );
    return warp_data[0];  // Get result from lane 0
}

// This function asynchronously uses an entire block of threads to copy two inputs to two outputs.
template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename RandomAccessIterator3, typename RandomAccessIterator4>
__forceinline__ __device__ void block_multi_copy( RandomAccessIterator1 first1,
                                                  RandomAccessIterator1 last1,
                                                  RandomAccessIterator2 first2,
                                                  RandomAccessIterator3 result1,
                                                  RandomAccessIterator4 result2 )
{
    unsigned int block_size = get_block_dim_x() * get_block_dim_y() * get_block_dim_z();
    unsigned int n          = last1 - first1;

    // advance iterators to my position
    // XXX generalize to 3D blocks
    unsigned int i = get_thread_idx_y() * get_block_dim_x() + get_thread_idx_x();

    // use bracket operator rather than pointer arithmetic for speed on Tesla
    while( i < n )
    {
        result1[i] = first1[i];
        result2[i] = first2[i];
        i += block_size;
    }
}

// This function asynchronously uses an entire block of threads to copy one input to one shared memory output.
template <typename RandomAccessIterator>
__forceinline__ __device__ void block_shared_copy( RandomAccessIterator first, RandomAccessIterator last, RandomAccessIterator result )
{
    unsigned int block_size = get_block_dim_x() * get_block_dim_y() * get_block_dim_z();
    unsigned int n          = last - first;

    // advance iterators to my position
    // XXX generalize to 3D blocks
    unsigned int i = get_thread_idx_y() * get_block_dim_x() + get_thread_idx_x();

    while( i < n )
    {
        store_shared( result + i, first[i] );
        i += block_size;
    }
}

// Find leading one.
__device__ __inline__ int flo( int v )
{
    unsigned int r;
    asm( "  bfind.u32 %0, %1;" : "=r"( r ) : "r"( v ) );
    return r;
}

// Returns true for exactly one of the active lanes.
__device__ __inline__ bool single_lane()
{
    return get_laneid() == flo( __ballot( true ) );
}


//
// SASS_MARKER
//
// Inserts marker instructions into the generated SASS to make code sections
// easy to find. The markers should always compile down to a single SASS
// instruction, keeping clutter to a minimum.  Here's a basic usage example:
//
//      SASS_MARKER( 0xcafe01 );
//
// This will insert a single shared memory store of 0 to marker address
// 0xcafe01. (The resulting kernel will crash, but who cares).  The marker
// constant can be anything that's easy to grep for in the final SASS; ideally
// it should be 24 bits or less so it can be compiled into an immediate (i.e.
// 0xcafe01 is good, 0xdecaf01 is bad since it will need an extra MOV).
//
// The basic SASS_MARKER leaves some room for the compiler to move the
// generated instruction around, which can be confusing. For more control,
// there are 3 additional flavors, which produce and/or consume variables:
//
//      SASS_MARKER_CONSUME( marker, variable )
//      SASS_MARKER_PRODUCE( marker, variable )
//      SASS_MARKER_PRODUCE_CONSUME( marker, variable1, variable2 )
//
// Produced/consumed variables must be 32-bit types. Here's an example that
// uses SASS_MARKER_CONSUME to make sure a marker is placed after a code block
// of interest:
//              ...
//              <code that computes variable x>
//              SASS_MARKER_CONSUME( 0xcafe02, x );
//              ...
//
// For the most precise placement, use SASS_MARKER_PROLWDE_CONSUME.
//
#define SASS_MARKER( mark )                                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        asm( "st.volatile.shared.b32 [%0], %1;" : : OPTIX_ASM_PTR( (int*)( mark ) ), "r"( 0 ) : );                     \
    } while( 0 )
#define SASS_MARKER_CONSUME( mark, cons )                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        asm( "st.volatile.shared.b32 [%0], %1;" : : OPTIX_ASM_PTR( (int*)( mark ) ), "r"( (int)( cons ) ) : );         \
    } while( 0 )
#define SASS_MARKER_PRODUCE( mark, prod )                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        asm( "ld.volatile.shared.b32 %0, [%1];" : "=r"( (int)( prod ) ) : OPTIX_ASM_PTR( (int*)( mark ) ) : );         \
    } while( 0 )
#define SASS_MARKER_PRODUCE_CONSUME( mark, prod, cons )                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        asm( "xor.b32 %0, %1, %2;" : "=r"( (int)( prod ) ) : "r"( (int)( cons ) ), "r"( (int)( mark ) ) : );           \
    } while( 0 )
