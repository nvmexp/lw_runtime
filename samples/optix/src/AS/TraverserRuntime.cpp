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

#include "../ExelwtionStrategy/CORTTypes.h"
#include "../ExelwtionStrategy/CommonRuntime.h"
#include <corelib/compiler/LWVMAddressSpaces.h>

#define __shared__ __attribute__( ( address_space( corelib::ADDRESS_SPACE_SHARED ) ) )
#define LO32( v ) ( (unsigned)( (unsigned long long)v ) )

using namespace cort;
using namespace lwca;

const int LOCAL_STACK_SIZE = 64;

struct BvhNode
{
    float3   bbmin0;
    float3   bbmax0;
    unsigned n0begin;
    unsigned n0end;

    float3   bbmin1;
    float3   bbmax1;
    unsigned n1begin;
    unsigned n1end;
};

struct Entity
{
    uint child;
    uint primIdx;
};

typedef int           int32_t;
typedef unsigned int  uint32_t;
typedef unsigned char uint8_t;
typedef float         vfloat4 __attribute__( ( vector_size( 16 ) ) );
typedef float         vfloat2 __attribute__( ( vector_size( 8 ) ) );
typedef unsigned      vuint2 __attribute__( ( vector_size( 8 ) ) );

struct Bvh8StackEntry
{
    int32_t  firstChildIdx;
    uint32_t hits;
};


extern "C" {
// optixi_ runtime functions
void optixi_setLwrrentAcceleration( CanonicalState* state );
cort::OptixRay optixi_getLwrrentRay( CanonicalState* state );
float optixi_getLwrrentTmax( CanonicalState* state );
unsigned optixi_getPrimitiveIndexOffset( CanonicalState* state, unsigned int GIOffset );
void optixi_intersectPrimitive( CanonicalState* state, unsigned int GIOffset, unsigned int primitiveIndex );
void optixi_intersectNode( CanonicalState* state, unsigned int child );

// Stub functions that need to substituted for buffer/variable reference funcs
vfloat4 getBufferElement_bvh( CanonicalState* state, uint elementSize, uint64 offset, uint64 coordinate );
uint2 getBufferElement_primitive_entities( CanonicalState* state, uint elementSize, uint64 offset, uint64 coordinate );
uint getBufferElement_group_entities( CanonicalState* state, uint elementSize, uint64 offset, uint64 coordinate );
uint getBufferElement_prim_counts( CanonicalState* state, uint elementSize, uint64 offset, uint64 coordinate );

int getVariableValue_motion_steps( CanonicalState* state, uint64 offset );
int getVariableValue_motion_stride( CanonicalState* state, uint64 offset );
vfloat2 getVariableValue_motion_time_range( CanonicalState* state, uint64 offset );

// Stub function and their implementations that are inlined by the OptiX compiler

float raySpanMin( float3 t0, float3 t1, float t );  // stub
float raySpanMin_default( float3 t0, float3 t1, float t );
float raySpanMin_sm20( float3 t0, float3 t1, float t );
float raySpanMin_sm30( float3 t0, float3 t1, float t );
float raySpanMin_sm50( float3 t0, float3 t1, float t );

float raySpanMax( float3 t0, float3 t1, float t );  // stub
float raySpanMax_default( float3 t0, float3 t1, float t );
float raySpanMax_sm20( float3 t0, float3 t1, float t );
float raySpanMax_sm30( float3 t0, float3 t1, float t );
float raySpanMax_sm50( float3 t0, float3 t1, float t );

void getLeafRange( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd );  // stub
void getLeafRange_contiguous( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd );
void getLeafRange_ordered( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd );
void getLeafRange_general( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd );

uint getGiOffset( CanonicalState* state, uint child );  // stub
uint getGiOffset_default( CanonicalState* state, uint child );
uint getGiOffset_baked( CanonicalState* state, uint child );

void intersectLeaf( CanonicalState* state, uint idx, bool dbg = false );  // stub
void intersectLeaf_geometry( CanonicalState* state, uint idx, bool dbg = false );
void intersectLeaf_group( CanonicalState* state, uint idx, bool dbg = false );

int SMEM_STACK_BYTES = 16 * 4;  // Will be set by createTraverserFunc()
};


#define BVH2_SMEM_STACK_SIZE ( SMEM_STACK_BYTES / sizeof( uint ) )
#define BVH8_SMEM_STACK_SIZE ( SMEM_STACK_BYTES / sizeof( Bvh8StackEntry ) )

extern __shared__ uint2 dynamicSmem[];  // we want to make sure this is 8-byte aligned for BVH8


////////////////////////////////////////////////////////////////////////////////
//
// Utility / math functions
//
////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline void swap( T& a, T& b )
{
    T tmp = a;
    a     = b;
    b     = tmp;
}

template <typename T>
inline T min( const T& a, const T& b )
{
    return a < b ? a : b;
}

template <typename T>
inline T max( const T& a, const T& b )
{
    return a > b ? a : b;
}

inline float fminf( float a, float b )
{
    return lwca::minf( a, b );
}

inline float fmaxf( float a, float b )
{
    return lwca::maxf( a, b );
}

inline float fminf( float3 v )
{
    return fminf( fminf( v.x, v.y ), v.z );
}

inline float fmaxf( float3 v )
{
    return fmaxf( fmaxf( v.x, v.y ), v.z );
}

inline float3 fminf( float3 a, float3 b )
{
    return float3( fminf( a.x, b.x ), fminf( a.y, b.y ), fminf( a.z, b.z ) );
}

inline float3 fmaxf( float3 a, float3 b )
{
    return float3( fmaxf( a.x, b.x ), fmaxf( a.y, b.y ), fmaxf( a.z, b.z ) );
}

inline float clamp( const float f, const float a, const float b )
{
    return fmaxf( a, fminf( f, b ) );
}

inline float3 clamp( const float3& v, const float a, const float b )
{
    return float3( clamp( v.x, a, b ), clamp( v.y, a, b ), clamp( v.z, a, b ) );
}

inline float3 rcp( const float3& a )
{
    return float3( rcp( a.x ), rcp( a.y ), rcp( a.z ) );
}

inline float3 fmad( float3 a, float3 b, float3 c )
{
    return float3( fmad( a.x, b.x, c.x ), fmad( a.y, b.y, c.y ), fmad( a.z, b.z, c.z ) );
}

inline int findLeadingOne( unsigned int v )
{
    return ffs( v );
}

static inline float3 swizzle( float3 v, int axis )
{
    if( axis == 0 )
        return float3( v.y, v.z, v.x );
    if( axis == 1 )
        return float3( v.z, v.x, v.y );
    return v;
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

// The same as fminf(fminf(a, b), c)
inline float fvminminf( float a, float b, float c )
{
    return int_as_float( minmin( float_as_int( a ), float_as_int( b ), float_as_int( c ) ) );
}

// The same as fmaxf(fminf(a, b), c)
inline float fvminmaxf( float a, float b, float c )
{
    return int_as_float( minmax( float_as_int( a ), float_as_int( b ), float_as_int( c ) ) );
}

// The same as fminf(fmaxf(a, b), c);
inline float fvmaxminf( float a, float b, float c )
{
    return int_as_float( maxmin( float_as_int( a ), float_as_int( b ), float_as_int( c ) ) );
}

// The same as fmaxf(fmaxf(a, b), c)
inline float fvmaxmaxf( float a, float b, float c )
{
    return int_as_float( maxmax( float_as_int( a ), float_as_int( b ), float_as_int( c ) ) );
}

inline float fvminf( float3 f )
{
    return fvminminf( f.x, f.y, f.z );
}

inline float fvmaxf( float3 f )
{
    return fvmaxmaxf( f.x, f.y, f.z );
}

float raySpanMin_default( float3 t0, float3 t1, float t )
{
    return fmaxf( fmaxf( fminf( t0, t1 ) ), t );  // 6 ops
}

float raySpanMin_sm20( float3 t0, float3 t1, float t )
{
    return fvminmaxf( t0.z, t1.z, fvminmaxf( t0.y, t1.y, fvminmaxf( t0.x, t1.x, t ) ) );  // 3 ops
}

float raySpanMin_sm30( float3 t0, float3 t1, float t )
{
    return fvmaxmaxf( fminf( t0.x, t1.x ), fminf( t0.y, t1.y ), fvminmaxf( t0.z, t1.z, t ) );  // 4 ops
}

float raySpanMin_sm50( float3 t0, float3 t1, float t )
{
    return fvminmaxf( t0.z, t1.z, fvminmaxf( t0.y, t1.y, fvminmaxf( t0.x, t1.x, t ) ) );
}

float raySpanMax_default( float3 t0, float3 t1, float t )
{
    return fminf( fminf( fmaxf( t0, t1 ) ), t );  // 6 ops
}

float raySpanMax_sm20( float3 t0, float3 t1, float t )
{
    return fvmaxminf( t0.z, t1.z, fvmaxminf( t0.y, t1.y, fvmaxminf( t0.x, t1.x, t ) ) );  // 3 ops
}

float raySpanMax_sm30( float3 t0, float3 t1, float t )
{
    return fvminminf( fmaxf( t0.x, t1.x ), fmaxf( t0.y, t1.y ), fvmaxminf( t0.z, t1.z, t ) );  // 4 ops
}

float raySpanMax_sm50( float3 t0, float3 t1, float t )
{
    return fvmaxminf( t0.z, t1.z, fvmaxminf( t0.y, t1.y, fvmaxminf( t0.x, t1.x, t ) ) );
}

// Replicate sign bits of 4 packed signed bytes.
inline unsigned int vsignExtend4( unsigned int bytes )
{
    return permute( bytes, bytes, 8 | ( 9 << 4 ) | ( 10 << 8 ) | ( 11 << 12 ) );
}

inline unsigned int duplicateByte( unsigned char value )
{
    return permute( value, value, 0 );
}


////////////////////////////////////////////////////////////////////////////////
//
// Core BVH traversal functions
//
////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------
void printNode( int lwr, const BvhNode& node )
{
    // TODO: It would be better to have this be a single call so that the message
    // doesn't get split. We need to use variadic templates for that.
    printf( "%5d:[%7.3f,%7.3f,%7.3f][%7.3f,%7.3f,%7.3f] %5d %5d | ", lwr, node.bbmin0.x, node.bbmin0.y, node.bbmin0.z,
            node.bbmax0.x, node.bbmax0.y, node.bbmax0.z, node.n0begin, node.n0end );
    printf( "[%7.3f,%7.3f,%7.3f][%7.3f,%7.3f,%7.3f] %5d %5d\n", node.bbmin1.x, node.bbmin1.y, node.bbmin1.z,
            node.bbmax1.x, node.bbmax1.y, node.bbmax1.z, node.n1begin, node.n1end );
}

//------------------------------------------------------------------------------
// Load BvhNode with v4 loads.
inline BvhNode fetchNode( CanonicalState* state, uint idx )
{
    vfloat4 n0a = getBufferElement_bvh( state, 16, 0, idx + 0 );
    vfloat4 n0b = getBufferElement_bvh( state, 16, 0, idx + 1 );
    vfloat4 n1a = getBufferElement_bvh( state, 16, 0, idx + 2 );
    vfloat4 n1b = getBufferElement_bvh( state, 16, 0, idx + 3 );

    BvhNode node;
    node.bbmin0  = float3( n0a[0], n0a[1], n0a[2] );
    node.bbmax0  = float3( n0a[3], n0b[0], n0b[1] );
    node.n0begin = lwca::float_as_int( n0b[2] );
    node.n0end   = lwca::float_as_int( n0b[3] );
    node.bbmin1  = float3( n1a[0], n1a[1], n1a[2] );
    node.bbmax1  = float3( n1a[3], n1b[0], n1b[1] );
    node.n1begin = lwca::float_as_int( n1b[2] );
    node.n1end   = lwca::float_as_int( n1b[3] );
    return node;
}

//------------------------------------------------------------------------------
// Fetch the current ray and return the initial node idx in the BVH
// iD  - the ilwerse (reciprocal) of the ray direction
// oiD - ray origin scaled by the ilwerse ray direction
inline unsigned init( CanonicalState* state, cort::OptixRay* ray, float3* iD, float3* oiD, bool dbg = false )
{
    optixi_setLwrrentAcceleration( state );
    *ray = optixi_getLwrrentRay( state );
    *iD  = clamp( rcp( ray->direction ), -1.e16f, 1.e16f );  // clamp to avoid NaNs and Infs in traversal
    *oiD = ray->origin * ( *iD );

    uint root = 0;
    if( dbg )
    {
        printf( "Top nodes---------------\n" );
        printf( "root:%d\n", root );
        for( int i = 0; i < 10; ++i )
        {
            BvhNode node = fetchNode( state, 4 * i );
            printNode( 4 * i, node );
        }
    }
    return fetchNode( state, root ).n0end;
}

//------------------------------------------------------------------------------
// Intersect the each AABB of the given BVH node and compute tmin and hit for each
__attribute__( ( always_inline ) ) static void intersectNodeAabbs( BvhNode&  node,
                                                                   float3    iD,
                                                                   float3    oiD,
                                                                   float     ray_tmin,
                                                                   float     ray_tmax,
                                                                   float*    out_tmin0,
                                                                   float*    out_tmin1,
                                                                   unsigned* out_hit0,
                                                                   unsigned* out_hit1 )
{
    float3 t0    = fmad( node.bbmin0, iD, -oiD );
    float3 t1    = fmad( node.bbmax0, iD, -oiD );
    float  tmin0 = raySpanMin( t0, t1, ray_tmin );
    float  tmax0 = raySpanMax( t0, t1, ray_tmax );
    *out_tmin0   = tmin0;
    *out_hit0    = ( tmin0 <= tmax0 );

    t0          = fmad( node.bbmin1, iD, -oiD );
    t1          = fmad( node.bbmax1, iD, -oiD );
    float tmin1 = raySpanMin( t0, t1, ray_tmin );
    float tmax1 = raySpanMax( t0, t1, ray_tmax );
    *out_tmin1  = tmin1;
    *out_hit1   = ( tmin1 <= tmax1 );
}

//------------------------------------------------------------------------------
// Intersect the AABBs of the node at index lwr and return the node and hit info
__attribute__( ( always_inline ) ) static BvhNode intersectNodeAabbs( CanonicalState* state,
                                                                      unsigned        lwr,
                                                                      float3          iD,
                                                                      float3          oiD,
                                                                      float           ray_tmin,
                                                                      float           ray_tmax,
                                                                      float*          out_tmin0,
                                                                      float*          out_tmin1,
                                                                      unsigned*       out_hit0,
                                                                      unsigned*       out_hit1 )
{
    BvhNode node = fetchNode( state, lwr );
    intersectNodeAabbs( node, iD, oiD, ray_tmin, ray_tmax, out_tmin0, out_tmin1, out_hit0, out_hit1 );
    return node;
}

// Lerp using FMA method from "GPU Pro Tip: Lerp Faster in C++".

// round up
inline float3 lerp_rp( float3 a, float3 b, float t )
{
    return float3( fmad_rp( t, b.x, fmad_rp( -t, a.x, a.x ) ), fmad_rp( t, b.y, fmad_rp( -t, a.y, a.y ) ),
                   fmad_rp( t, b.z, fmad_rp( -t, a.z, a.z ) ) );
}

// round down
inline float3 lerp_rm( float3 a, float3 b, float t )
{
    return float3( fmad_rm( t, b.x, fmad_rm( -t, a.x, a.x ) ), fmad_rm( t, b.y, fmad_rm( -t, a.y, a.y ) ),
                   fmad_rm( t, b.z, fmad_rm( -t, a.z, a.z ) ) );
}

__attribute__( ( always_inline ) ) static BvhNode lerpNode( const BvhNode& node0, const BvhNode& node1, float t )
{
    BvhNode node;
    node.bbmin0  = lerp_rm( node0.bbmin0, node1.bbmin0, t );
    node.bbmax0  = lerp_rp( node0.bbmax0, node1.bbmax0, t );
    node.n0begin = node0.n0begin;
    node.n0end   = node0.n0end;
    node.bbmin1  = lerp_rm( node0.bbmin1, node1.bbmin1, t );
    node.bbmax1  = lerp_rp( node0.bbmax1, node1.bbmax1, t );
    node.n1begin = node0.n1begin;
    node.n1end   = node0.n1end;

    return node;
}

//------------------------------------------------------------------------------
// Intersect the lerped motion AABBs of the node at index lwr and return the node and hit info
__attribute__( ( always_inline ) ) static BvhNode intersectNodeAabbsWithMotion( CanonicalState* state,
                                                                                unsigned        lwr,
                                                                                float3          iD,
                                                                                float3          oiD,
                                                                                float           ray_tmin,
                                                                                float           ray_tmax,
                                                                                float ray_time,  // normalized within single motion step
                                                                                unsigned  time0_idx,
                                                                                unsigned  time1_idx,
                                                                                unsigned  motionStride,
                                                                                float*    out_tmin0,
                                                                                float*    out_tmin1,
                                                                                unsigned* out_hit0,
                                                                                unsigned* out_hit1 )
{
    BvhNode node0 = fetchNode( state, lwr + time0_idx * motionStride );
    BvhNode node1 = fetchNode( state, lwr + time1_idx * motionStride );

    BvhNode node = lerpNode( node0, node1, ray_time );

    intersectNodeAabbs( node, iD, oiD, ray_tmin, ray_tmax, out_tmin0, out_tmin1, out_hit0, out_hit1 );
    return node;
}

//------------------------------------------------------------------------------
// Return true if t0 < t1.
// Uses a small epsilon to filter out rounding noise. By operating on the integer
// representation of the values the epsilon is automatically scaled.
inline bool outOfOrder( float t0, float t1 )
{
    const int IEPS = 32;
    return ( float_as_int( t1 ) + IEPS < float_as_int( t0 ) );
}

//------------------------------------------------------------------------------
inline void storeShared( int stackPtr, uint entry )
{
    __shared__ uint* smemStackPtr = (__shared__ uint*)( stackPtr * sizeof( uint ) );
    *smemStackPtr                 = entry;
}

//------------------------------------------------------------------------------
inline void loadShared( int stackPtr, uint* entry )
{
    __shared__ uint* smemStackPtr = (__shared__ uint*)( stackPtr * sizeof( uint ) );
    *entry                        = *smemStackPtr;
}

//------------------------------------------------------------------------------
inline void stackInit( int& stackPtr, int& stackBase )
{
    if( BVH2_SMEM_STACK_SIZE )
    {
        stackBase = LO32( dynamicSmem ) >> 2;  // Bake in smemStack base.
        stackBase += BVH2_SMEM_STACK_SIZE * ( tid().x + tid().y * ntid().x + tid().z * 32 );
        stackPtr = stackBase;
    }
    else
    {
        stackBase = stackPtr = 0;
    }
}

//------------------------------------------------------------------------------
inline void stackPush( uint entry, int& stackPtr, int stackBase, uint* localStack )
{
    if( BVH2_SMEM_STACK_SIZE )
    {
        int stackEnd = stackBase + BVH2_SMEM_STACK_SIZE;
        if( stackPtr >= stackEnd )
            localStack[stackPtr - stackEnd] = entry;  // spill to local stack
        else
            storeShared( stackPtr, entry );

        stackPtr++;
    }
    else
    {
        localStack[stackPtr++] = entry;
    }
}

//------------------------------------------------------------------------------
inline uint stackPop( int& stackPtr, int stackBase, uint* localStack )
{
    if( BVH2_SMEM_STACK_SIZE )
    {
        stackPtr--;

        uint entry;
        int  stackEnd = stackBase + BVH2_SMEM_STACK_SIZE;
        if( stackPtr >= stackEnd )
            entry = localStack[stackPtr - stackEnd];  // pop from local stack
        else
            loadShared( stackPtr, &entry );

        return entry;
    }
    else
    {
        return localStack[--stackPtr];
    }
}

//------------------------------------------------------------------------------
// Given a node and hit info determine the next node to traverse.
inline unsigned chooseNextNode( BvhNode& node, float tmin0, float tmin1, unsigned hit0, unsigned hit1, unsigned* localStack, int* sp, int sb )
{
    unsigned interior0 = ( node.n0begin == ~0 );
    unsigned interior1 = ( node.n1begin == ~0 );

    unsigned trav0 = hit0 & interior0;
    unsigned trav1 = hit1 & interior1;
    unsigned lwr   = trav0 ? node.n0end : node.n1end;
    if( !( trav0 ^ trav1 ) )  // Not only left or right
    {
        if( trav0 )  // Hit both nodes => choose closest and push the other.
        {
            lwr           = node.n0end;
            unsigned push = node.n1end;
            if( outOfOrder( tmin0, tmin1 ) )
                swap( lwr, push );
            stackPush( push, *sp, sb, localStack );
        }
        else  // Neither node hit => pop the stack
        {
            lwr = ( *sp == sb ) ? 0 : stackPop( *sp, sb, localStack );
        }
    }
    return lwr;
}

//------------------------------------------------------------------------------
// Computes the range of leaf indices to intersect. The range is empty if the
// node has no hit children.
//
// This variant assumes that if both children are leaves, then their leaf ranges
// are ordered and contiguous, i.e. node.n0end = node.n1begin.
void getLeafRange_contiguous( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd )
{
    unsigned interior1 = ( node.n1begin == ~0 );
    *leafBegin         = hit0 ? node.n0begin : ~0u;
    *leafEnd           = node.n0end;
    if( hit1 )
    {
        *leafBegin = min( *leafBegin, node.n1begin );
        *leafEnd   = interior1 ? node.n0end : node.n1end;
    }
    *gapBegin = *leafEnd;
    *gapEnd   = *leafEnd;
}

//------------------------------------------------------------------------------
// Computes the range of leaf indices to intersect. The range is empty if the
// node has no hit children.
//
// This variant assumes that if both children are leaves, then their leaf ranges
// are ordered, i.e. node.n0end < node.n1begin.
void getLeafRange_ordered( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd )
{
    unsigned interior0 = ( node.n0begin == ~0 );
    unsigned interior1 = ( node.n1begin == ~0 );
    unsigned leaf0     = ( hit0 & ( 1 - interior0 ) );
    unsigned leaf1     = ( hit1 & ( 1 - interior1 ) );
    *leafBegin         = leaf0 ? node.n0begin : ( leaf1 ? node.n1begin : ~0u );
    *leafEnd           = leaf1 ? node.n1end : node.n0end;
    *gapBegin          = node.n0end;
    *gapEnd            = node.n1begin;
}

//------------------------------------------------------------------------------
// Computes the range of leaf indices to intersect. The range is empty if the
// node has no hit children.
//
// This variant makes no assumptions on leaf ranges
void getLeafRange_general( BvhNode& node, unsigned hit0, unsigned hit1, unsigned* leafBegin, unsigned* leafEnd, unsigned* gapBegin, unsigned* gapEnd )
{
    *leafBegin = minui( hit0 ? node.n0begin : ~0, hit1 ? node.n1begin : ~0 );
    *leafEnd   = maxui( hit0 ? node.n0end : 0, hit1 ? node.n1end : 0 );
    *gapBegin  = minui( node.n0end, node.n1end );
    *gapEnd    = maxui( node.n0begin, node.n1begin );
}


//------------------------------------------------------------------------------
uint getGiOffset_default( CanonicalState* state, uint child )
{
    AbstractGroupHandle g        = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int        bufferId = AbstractGroup_getChildren( state, g );
    const unsigned int  eltSize  = sizeof( unsigned int );
    char                stackTemp[eltSize];
    const unsigned int* addr =
        reinterpret_cast<const unsigned int*>( Buffer_getElementAddress1dFromId( state, bufferId, eltSize, stackTemp, child ) );
    return *addr;
}

//------------------------------------------------------------------------------
uint getGiOffset_baked( CanonicalState* state, uint child )
{
    return child;
}

//------------------------------------------------------------------------------
void intersectLeaf_geometry( CanonicalState* state, uint idx, bool dbg )
{
    uint2  val      = getBufferElement_primitive_entities( state, 8, 0, idx );
    Entity entity   = {val.x, val.y};
    uint   giOffset = getGiOffset( state, entity.child );

    if( dbg )
        printf( "Intersect geometry idx:%d entity:%d primIdx:%d giOffset:%08x\n", idx, entity.child, entity.primIdx, giOffset );

    optixi_intersectPrimitive( state, giOffset, entity.primIdx );
}

//------------------------------------------------------------------------------
void intersectLeaf_group( CanonicalState* state, uint idx, bool dbg )
{
    uint   val      = getBufferElement_group_entities( state, 4, 0, idx );
    Entity entity   = {val, 0};
    uint   giOffset = getGiOffset( state, entity.child );

    if( dbg )
        printf( "Intersect group idx:%d entity:%d giOffset:%08x\n", idx, entity.child, giOffset );

    optixi_intersectNode( state, giOffset );
}


////////////////////////////////////////////////////////////////////////////////
//
// Traverser base functions (loop structure)
//
////////////////////////////////////////////////////////////////////////////////

// pmevents provide a quick way to see how many times each loop is exelwting and
// impose very little or no performance cost (they compile to a variant of NOP).
// Using lwperf to display events, look for prof_trigger events with the same
// number.
//#define PMEVENT( x ) asm __volatile__("pmevent " #x ";")
#define PMEVENT( x )

//------------------------------------------------------------------------------
// A single while loop that includes intersection.
extern "C" void traverse_bvh_w( CanonicalState* state )
{
    cort::OptixRay ray;
    float3         iD, oiD;
    unsigned       hit0, hit1, leafBegin, leafEnd, gapBegin, gapEnd;

    unsigned lwr = init( state, &ray, &iD, &oiD );
    unsigned localStack[LOCAL_STACK_SIZE];
    int      sp = 0, sb = 0;
    stackInit( sp, sb );
    PMEVENT( 0 );
    while( lwr )
    {
        PMEVENT( 1 );
        float   tmin0, tmin1, ray_tmax = optixi_getLwrrentTmax( state );
        BvhNode node = intersectNodeAabbs( state, lwr, iD, oiD, ray.tmin, ray_tmax, &tmin0, &tmin1, &hit0, &hit1 );

        lwr = chooseNextNode( node, tmin0, tmin1, hit0, hit1, localStack, &sp, sb );

        getLeafRange( node, hit0, hit1, &leafBegin, &leafEnd, &gapBegin, &gapEnd );

        unsigned i = leafBegin;
        while( i < leafEnd )
        {
            PMEVENT( 2 );
            intersectLeaf( state, i );
            ++i;
            if( i == gapBegin )
                i = gapEnd;
        }
        PMEVENT( 3 );
    }
    PMEVENT( 4 );
}


#define NOP() asm __volatile__( "{.reg .b32 tmp; mov.b32 tmp,%laneid;}" );

//------------------------------------------------------------------------------
// Synchronize the warp before falling out of the traversal loop and computing
// intersections.
extern "C" void traverse_bvh_ww( CanonicalState* state )
{
    cort::OptixRay ray;
    float3         iD, oiD;
    unsigned       hit0, hit1, leafBegin, leafEnd, gapBegin, gapEnd;

    unsigned lwr = init( state, &ray, &iD, &oiD );
    unsigned localStack[LOCAL_STACK_SIZE];
    int      sp = 0, sb = 0;
    stackInit( sp, sb );
    PMEVENT( 0 );
    while( lwr )
    {
        NOP();  // HACK: This creates a preheader for the inner loop, which seems to be necessary to get the compiler to insert the warp sync tokens. Use thread_groups when LWCA 8 comes out
        while( lwr )
        {
            PMEVENT( 1 );
            float   tmin0, tmin1, ray_tmax = optixi_getLwrrentTmax( state );
            BvhNode node = intersectNodeAabbs( state, lwr, iD, oiD, ray.tmin, ray_tmax, &tmin0, &tmin1, &hit0, &hit1 );

            lwr = chooseNextNode( node, tmin0, tmin1, hit0, hit1, localStack, &sp, sb );

            getLeafRange( node, hit0, hit1, &leafBegin, &leafEnd, &gapBegin, &gapEnd );
            if( leafBegin < leafEnd )
                break;
        };

        unsigned i = leafBegin;
        while( i < leafEnd )
        {
            PMEVENT( 2 );
            intersectLeaf( state, i );
            ++i;
            if( i == gapBegin )
                i = gapEnd;
        }
        PMEVENT( 3 );
    }
    PMEVENT( 4 );
}

// Motion version
extern "C" void traverse_bvh_ww_with_motion( CanonicalState* state )
{
    cort::OptixRay ray;
    float3         iD, oiD;
    unsigned       hit0, hit1, leafBegin, leafEnd, gapBegin, gapEnd;

    AbstractGroupHandle      g     = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    AccelerationHandle       accel = AbstractGroup_getAcceleration( state, g );
    MotionAccelerationHandle maccel       = Acceleration_colwertToMotionAccelerationHandle( state, accel );
    int                      motionSteps  = MotionAcceleration_getMotionSteps( state, maccel );
    unsigned                 motionStride = MotionAcceleration_getMotionStride( state, maccel );
    float                    timeBegin    = MotionAcceleration_getTimeBegin( state, maccel );
    float                    timeEnd      = MotionAcceleration_getTimeEnd( state, maccel );
    float                    rayTime      = Runtime_getLwrrentTime( state );

    // clamp
    rayTime = min( max( rayTime, timeBegin ), timeEnd );

    // bin into motion segment.  Must have motionSteps > 1.
    const float step_size    = ( timeEnd - timeBegin ) / ( motionSteps - 1 );
    const int   time0_idx    = min( motionSteps - 1, int( ( rayTime - timeBegin ) / step_size ) );
    const int   time1_idx    = min( motionSteps - 1, time0_idx + 1 );
    const float segment_time = ( rayTime - timeBegin ) / step_size - time0_idx;

    unsigned lwr = init( state, &ray, &iD, &oiD );
    unsigned localStack[LOCAL_STACK_SIZE];
    int      sp = 0, sb = 0;
    stackInit( sp, sb );
    PMEVENT( 0 );
    while( lwr )
    {
        NOP();  // HACK: This creates a preheader for the inner loop, which seems to be necessary to get the compiler to insert the warp sync tokens. Use thread_groups when LWCA 8 comes out
        while( lwr )
        {
            PMEVENT( 1 );
            float   tmin0, tmin1, ray_tmax = optixi_getLwrrentTmax( state );
            BvhNode node = intersectNodeAabbsWithMotion( state, lwr, iD, oiD, ray.tmin, ray_tmax, segment_time, time0_idx,
                                                         time1_idx, motionStride, &tmin0, &tmin1, &hit0, &hit1 );

            lwr = chooseNextNode( node, tmin0, tmin1, hit0, hit1, localStack, &sp, sb );

            getLeafRange( node, hit0, hit1, &leafBegin, &leafEnd, &gapBegin, &gapEnd );
            if( leafBegin < leafEnd )
                break;
        };

        unsigned i = leafBegin;
        while( i < leafEnd )
        {
            PMEVENT( 2 );
            intersectLeaf( state, i );
            ++i;
            if( i == gapBegin )
                i = gapEnd;
        }
        PMEVENT( 3 );
    }
    PMEVENT( 4 );
}


static const int USE_WATERTIGHT = 0;

//------------------------------------------------------------------------
// Type definitions for 8-wide BVHs.
// For further details, please refer to:
// https://p4viewer.lwpu.com/get///research/research/hylitie/docs/Efficient-RT%202016-07-09.pptx

//------------------------------------------------------------------------
// Meta struct contains child node pointers, child node type (inner/leaf)
// and number of primitives in leaf child nodes.
// Values in bits 5-7:
//      Empty child               -> 0b00000000
//      1 primitive or inner node -> 0b00100000
//      2 primitives              -> 0b01100000
//      3 primitives              -> 0b11100000
// Values in bits 0-4:
//      inner node                -> child slot index + 24
//      leaf node                 -> child slot index
//
// BVH8NodeHeader contains type and indexing information of child nodes.
// Child slots store child node index offsets from base index of corresponding
// child node type (inner/leaf). Inner and leaf/primitive remap child nodes are
// stored continuously and compactly in separate arrays, in same order as child slots.
// Child slots are sorted in Z-order for ordered traversal:
// inner, leaf and empty children can end up in any slot.
//      BVH8Node* childi = nodes[firstChildIdx + getOffset(meta[i])];

// Child node bounding boxes are quantized to 8-bit in coordinate system
// given by BVH8NodeHeader. Uncompressed boxes can be obtained by
// box.lo.x = header.pos[0] + lox[0] * header.scale[0] etc.

// float         pos[3];
// unsigned char scale[3];
// unsigned char innerMask;      // Bitmask of filled inner node children.
// int           firstChildIdx;  // Index of first child node in subtree.
// int           firstRemapIdx;  // Index of first primitive remap.
// unsigned char meta[8];        // Index offsets and child types for each child slot.
//
// Quantized child bounding boxes for each child slot.
// unsigned char lox[8];
// unsigned char loy[8];
// unsigned char loz[8];
// unsigned char hix[8];
// unsigned char hiy[8];
// unsigned char hiz[8];

struct BVH8Node  // 80 bytes
{
    vfloat4 header0;
    vfloat4 header1;

    vfloat4 boxes0;
    vfloat4 boxes1;
    vfloat4 boxes2;

    inline float3   pos() const;
    inline uint32_t packedScale() const;
    inline float3   scale() const;
    inline int      firstChildIdx() const;
    inline int      firstRemapIdx() const;
    inline uint32_t meta( int j ) const;

    inline uint3 vmin( int j ) const;
    inline uint3 vmax( int j ) const;

    inline Aabb getAabb() const;

  private:
    inline static unsigned char slice( unsigned int val, int pos );
};

float3 BVH8Node::pos() const
{
    return float3( header0[0], header0[1], header0[2] );
}

uint32_t BVH8Node::packedScale() const
{
    return float_as_int( header0[3] );
}

float3 BVH8Node::scale() const
{
    uint32_t pScale = packedScale();
    // Shift 8-bit scale to floating-point exponent bits.
    float3 scale = float3( int_as_float( vshl_clamp_b0( pScale, 23 ) ), int_as_float( vshl_clamp_b1( pScale, 23 ) ),
                           int_as_float( vshl_clamp_b2( pScale, 23 ) ) );

    return scale;
}

int BVH8Node::firstChildIdx() const
{
    return float_as_int( header1[0] );
}

int BVH8Node::firstRemapIdx() const
{
    return float_as_int( header1[1] );
}

uint32_t BVH8Node::meta( int j ) const
{
    return float_as_int( header1[2 + j] );
}

uint3 BVH8Node::vmin( int j ) const
{
    if( j == 0 )
        return uint3( float_as_int( boxes0[0] ), float_as_int( boxes0[2] ), float_as_int( boxes1[0] ) );
    else
        return uint3( float_as_int( boxes0[1] ), float_as_int( boxes0[3] ), float_as_int( boxes1[1] ) );
}

uint3 BVH8Node::vmax( int j ) const
{
    if( j == 0 )
        return uint3( float_as_int( boxes1[2] ), float_as_int( boxes2[0] ), float_as_int( boxes2[2] ) );
    else
        return uint3( float_as_int( boxes1[3] ), float_as_int( boxes2[1] ), float_as_int( boxes2[3] ) );
}

unsigned char BVH8Node::slice( unsigned int v, int pos )
{
    return ( v >> pos * 8 ) & 0xFF;
}

Aabb BVH8Node::getAabb() const
{
    float3 pos   = this->pos();
    float3 scale = this->scale();

    Aabb aabb;
    aabb.ilwalidate();

    for( int group = 0; group < 2; ++group )
    {
        uint32_t meta = this->meta( group );
        uint3    vmin = this->vmin( group );
        uint3    vmax = this->vmax( group );

        for( int i = 0; i < 4; i++ )
        {
            // Detect and skip empty nodes, which have valid aabbs in the BVH8 compression scheme.
            if( slice( meta, i ) != 0 )
            {
                float3 lo( slice( vmin.x, i ), slice( vmin.y, i ), slice( vmin.z, i ) );
                float3 hi( slice( vmax.x, i ), slice( vmax.y, i ), slice( vmax.z, i ) );

                // header.pos[0] + lox[0] * header.scale[0]
                float3 min = pos + lo * scale;
                float3 max = pos + hi * scale;

                aabb.include( Aabb( min, max ) );
            }
        }
    }

    return aabb;
}

struct RayBvh8
{
    float3       orig;
    float3       dir;
    float        tmin;
    float        tmax;
    unsigned int mask;

    // Additional data for ray/triangle test and ray/AABB test.
    float3 translate;
    float3 shear;
    float3 ilwDir;
    int    majorAxis;
};

// According to mask, select set bits from a and unset bits from b.
//static inline unsigned int  bitSelectInt       (unsigned int a, unsigned int b, unsigned int mask)     { return (a & mask) | (b & ~mask); }
//static inline float         bitSelect       (float a, float b, float mask)                          { return int_as_float(bitSelectInt(float_as_int(a), float_as_int(b), float_as_int(mask))); }

static inline void setupRay( float3 orig, float3 dir, int& axis, float3& translate, float3& shear )
{
    // Choose main axis.
    float ax = fabs( dir.x );
    float ay = fabs( dir.y );
    float az = fabs( dir.z );
    axis     = ( ax >= ay && ax >= az ) ? 0 : ( ay >= az ) ? 1 : 2;

    // Swizzle origin and direction.
    float3 so = swizzle( orig, axis );
    float3 sd = swizzle( dir, axis );
    float  rz = 1.0f / sd.z;

    // Output ray.
    translate.x = so.x;
    translate.y = so.y;
    translate.z = so.z;
    shear.x     = rz * sd.x;
    shear.y     = rz * sd.y;
    shear.z     = rz;
}

static inline void setupRay( RayBvh8& ray )
{
// Clamp ray direction to avoid division by zero.
// Note: In the watertight mode, we disable the clamping to get as accurate results as possible.
#if 0
  // clang generate trap() here, it seems clang thinks that (fabs( ray.dir.x ) < eps) branch is unreachable
  if( !USE_WATERTIGHT )
  {
    float eps = exp2f( -80.0f );
    ray.dir.x = ( fabs( ray.dir.x ) >= eps ) ? ray.dir.x : bitSelect( ray.dir.x, eps, int_as_float( 0x80000000 ) );
    ray.dir.y = ( fabs( ray.dir.y ) >= eps ) ? ray.dir.y : bitSelect( ray.dir.y, eps, int_as_float( 0x80000000 ) );
    ray.dir.z = ( fabs( ray.dir.z ) >= eps ) ? ray.dir.z : bitSelect( ray.dir.z, eps, int_as_float( 0x80000000 ) );
  }
  setupRay( ray.orig, ray.dir, ray.majorAxis, ray.translate, ray.shear );
  ray.ilwDir.x = 1.0f / ray.dir.x;
  ray.ilwDir.y = 1.0f / ray.dir.y;
  ray.ilwDir.z = 1.0f / ray.dir.z;

#else
    setupRay( ray.orig, ray.dir, ray.majorAxis, ray.translate, ray.shear );
    if( !USE_WATERTIGHT )
    {
        ray.ilwDir = clamp( rcp( ray.dir ), -1.e16f, 1.e16f );  // clamp to avoid NaNs and Infs in traversal
    }
#endif
}

//------------------------------------------------------------------------------
inline void storeShared( int stackPtr, Bvh8StackEntry entry )
{
    __shared__ vuint2* smemStackPtr = (__shared__ vuint2*)( stackPtr * sizeof( Bvh8StackEntry ) );
    *smemStackPtr                   = *(vuint2*)&entry;
}

//------------------------------------------------------------------------------
inline void loadShared( int stackPtr, Bvh8StackEntry* entry )
{
    __shared__ vuint2* smemStackPtr                   = (__shared__ vuint2*)( stackPtr * sizeof( Bvh8StackEntry ) );
    vuint2                                        val = *smemStackPtr;
    *entry                                            = {(int32_t)val[0], val[1]};
}


//------------------------------------------------------------------------------
inline void bvh8StackInit( int& stackPtr, int& stackBase )
{
    if( BVH8_SMEM_STACK_SIZE )
    {
        stackBase = LO32( dynamicSmem ) >> 3;
        stackBase += BVH8_SMEM_STACK_SIZE * ( tid().x + tid().y * ntid().x + tid().z * 32 );
        stackPtr = stackBase;
    }
    else
    {
        stackBase = stackPtr = 0;
    }
}

//------------------------------------------------------------------------------
inline void bvh8StackPush( Bvh8StackEntry entry, int& stackPtr, int stackBase, Bvh8StackEntry* localStack )
{
    if( BVH8_SMEM_STACK_SIZE )
    {
        int stackEnd = stackBase + BVH8_SMEM_STACK_SIZE;
        if( stackPtr >= stackEnd )
            localStack[stackPtr - stackEnd] = entry;  // spill to local stack
        else
            storeShared( stackPtr, entry );

        stackPtr++;
    }
    else
    {
        localStack[stackPtr++] = entry;
    }
}

//------------------------------------------------------------------------------
inline Bvh8StackEntry bvh8StackPop( int& stackPtr, int stackBase, Bvh8StackEntry* localStack )
{
    if( BVH8_SMEM_STACK_SIZE )
    {
        stackPtr--;

        Bvh8StackEntry entry;
        int            stackEnd = stackBase + BVH8_SMEM_STACK_SIZE;
        if( stackPtr >= stackEnd )
            entry = localStack[stackPtr - stackEnd];  // pop from local stack
        else
            loadShared( stackPtr, &entry );

        return entry;
    }
    else
    {
        return localStack[--stackPtr];
    }
}

struct OctantData
{
    uint32_t duplicatedOctant;
    uint32_t prmtKeysX;
    uint32_t prmtKeysY;
};

#define MAKE_PRMT_KEYS( i0, i1, i2, i3 ) ( ( i3 ) << 12 ) | ( ( i2 ) << 8 ) | ( ( i1 ) << 4 ) | ( ( i0 ) << 0 )

inline OctantData initOctantData( float3 dir )
{
    OctantData data;

    // Use integer sign bit to prevent -0.0 to be treated as 0.0 = positive.
    data.duplicatedOctant =
        duplicateByte( ( uint8_t )( select( 0, 4, float_as_int( dir.z ) ) | select( 0, 2, float_as_int( dir.y ) )
                                    | select( 0, 1, float_as_int( dir.x ) ) ) );
    data.prmtKeysX = select( MAKE_PRMT_KEYS( 0, 1, 2, 3 ), MAKE_PRMT_KEYS( 4, 5, 6, 7 ), float_as_int( dir.x ) );
    data.prmtKeysY = select( MAKE_PRMT_KEYS( 0, 1, 2, 3 ), MAKE_PRMT_KEYS( 4, 5, 6, 7 ), float_as_int( dir.y ) );

    return data;
}

//------------------------------------------------------------------------
inline float3 getTRangeMultiplier( const RayBvh8& ray )
{
    if( !USE_WATERTIGHT )
        return ray.ilwDir;  // Approximate => premultiply tlo and thi with ilwDir; more efficient.
    else
        return float3( 1.0f, 1.0f, 1.0f );  // Watertight => do not premultiply; more accurate.
}

inline bool intersectTRanges( float3 tlo, float3 thi, const RayBvh8& ray )
{
    // Here, tlo and thi correspond to the per-slab t-ranges computed in intersectNode():
    // xrange = [tlo.x, thi.x]
    // yrange = [tlo.y, thi.y]
    // zrange = [min(tlo.z,thi.z), max(tlo.z,thi.z)]

    if( !USE_WATERTIGHT )
    {
        float tmin = fvmaxmaxf( fvminmaxf( tlo.z, thi.z, ray.tmin ), tlo.x, tlo.y );
        float tmax = fvminminf( fvmaxminf( tlo.z, thi.z, ray.tmax ), thi.x, thi.y );
        return ( tmin <= tmax );
    }
    else
    {
        // References:
        // Robust BVH Ray Traversal, Thiago Ize, JCGT 2013
        // Berger-Perrin, 2004, SSE ray/box intersection test (flipcode)

        // Multiply tlo and thi with ilwDir, since we did not pre-multiply them in intersectNode().

        tlo = tlo * ray.ilwDir;
        thi = thi * ray.ilwDir;

        // Turn NaNs into +/- inf.  A NaN can occur when (bbox.lo - ray.orig) == 0 and ray.ilwDir == inf, for some axis.
        // see Berger-Perrin.

        float inf  = int_as_float( 0x7f800000 );
        float tmin = fvmaxmaxf( fvminmaxf( fmaxf( tlo.z, -inf ), fmaxf( thi.z, -inf ), ray.tmin ), fmaxf( tlo.x, -inf ),
                                fmaxf( tlo.y, -inf ) );
        float tmax = fvminminf( fvmaxminf( fminf( tlo.z, +inf ), fminf( thi.z, +inf ), ray.tmax ), fminf( thi.x, +inf ),
                                fminf( thi.y, +inf ) );

        // 1+3*ulp, suggested by cwaechter

        tmax *= 1.0000003576278687f;
        return ( tmin <= tmax );
    }
}

static inline void computeHits( uint32_t&      hits,
                                int            i,
                                const RayBvh8& ray,
                                float3         m,
                                float3         a,
                                uint32_t       ofs,
                                uint32_t       trimask,
                                uint32_t       nearx,
                                uint32_t       farx,
                                uint32_t       neary,
                                uint32_t       fary,
                                uint32_t       nearz,
                                uint32_t       farz )
{
    // At the moment this compiles to separate bfe, i2f when i%4 == 0 (extracting low byte).
    // Optimal sass would be i2f.u8 r, r.b0. Approx 1-3% perf lost.

    float3 tlo, thi;
    tlo.x = a.x + m.x * (float)bfe( nearx, ( i % 4 ) * 8, 8 );
    tlo.y = a.y + m.y * (float)bfe( neary, ( i % 4 ) * 8, 8 );
    tlo.z = a.z + m.z * (float)bfe( nearz, ( i % 4 ) * 8, 8 );
    thi.x = a.x + m.x * (float)bfe( farx, ( i % 4 ) * 8, 8 );
    thi.y = a.y + m.y * (float)bfe( fary, ( i % 4 ) * 8, 8 );
    thi.z = a.z + m.z * (float)bfe( farz, ( i % 4 ) * 8, 8 );

    if( intersectTRanges( tlo, thi, ray ) )
    {
        // If hit, insert bit(s) to correct position in hitmask
        if( ( i % 4 ) == 0 )
            hits = vshl_wrap_add_b0_b0( trimask, ofs, hits );
        if( ( i % 4 ) == 1 )
            hits = vshl_wrap_add_b1_b1( trimask, ofs, hits );
        if( ( i % 4 ) == 2 )
            hits = vshl_wrap_add_b2_b2( trimask, ofs, hits );
        if( ( i % 4 ) == 3 )
            hits = vshl_wrap_add_b3_b3( trimask, ofs, hits );
    }
}

static inline void computeHits( uint32_t& hits, int j, const BVH8Node& node, const RayBvh8& ray, const OctantData& octantData, float3 m, float3 a )
{
    uint3 vmin = node.vmin( j );
    uint3 vmax = node.vmax( j );

    // Select tnear, tfar by ray sign for x and y. These replace 2 vmnmx per box.
    // Note: tnear, tfar selection for z is done in TraceSpecialization::intersectTRanges() because it's free there.
    uint32_t nearx = permute( vmin.x, vmax.x, octantData.prmtKeysX );
    uint32_t farx  = permute( vmax.x, vmin.x, octantData.prmtKeysX );
    uint32_t neary = permute( vmin.y, vmax.y, octantData.prmtKeysY );
    uint32_t fary  = permute( vmax.y, vmin.y, octantData.prmtKeysY );
    uint32_t nearz = vmin.z;
    uint32_t farz  = vmax.z;

    // Leaves insert numtris bits to dynamic hitmask; inner nodes insert 1 and empty nodes 0.
    uint32_t ofs     = node.meta( j );
    uint32_t trimask = ( ofs >> 5 ) & 0x07070707;  // empty children insert 0 even if hit (shouldn't normally happen).
    uint32_t innermask = ( ofs & ( ofs << 1 ) ) & 0x10101010;  // all inner nodes have bit 5,4 (16,8). Leaves may have one set bot not both.
    innermask = vsignExtend4( innermask << 3 );  // sbyte sign marks inner nodes, extend sign to all bits.

    ofs ^= ( octantData.duplicatedOctant & innermask );  // compute traversal priority for inner node children only.
    ofs &= 0x1F1F1F1F;  // Low 5 bits contain the offset, mask others out. Not needed with vshl.wrap.

    computeHits( hits, 0, ray, m, a, ofs, trimask, nearx, farx, neary, fary, nearz, farz );
    computeHits( hits, 1, ray, m, a, ofs, trimask, nearx, farx, neary, fary, nearz, farz );
    computeHits( hits, 2, ray, m, a, ofs, trimask, nearx, farx, neary, fary, nearz, farz );
    computeHits( hits, 3, ray, m, a, ofs, trimask, nearx, farx, neary, fary, nearz, farz );
}


static inline void intersectNode( const BVH8Node& node, const RayBvh8& ray, const OctantData& octantData, Bvh8StackEntry& innerHits, Bvh8StackEntry& leafHits )
{
    float3 pos              = node.pos();
    float3 scale            = node.scale();
    innerHits.firstChildIdx = node.firstChildIdx();
    leafHits.firstChildIdx  = node.firstRemapIdx();
    uint32_t hits           = 0;

    float3 ilwDir = getTRangeMultiplier( ray );
    float3 m      = scale * ilwDir;
    float3 a      = ( pos - ray.orig ) * ilwDir;

    computeHits( hits, 0, node, ray, octantData, m, a );
    computeHits( hits, 1, node, ray, octantData, m, a );

    // Extract inner, leaf node hits
    leafHits.hits = hits & 0x00FFFFFF;
    innerHits.hits = permute( hits, node.packedScale(), MAKE_PRMT_KEYS( 7, 1, 2, 3 ) );  // Move valid inner node mask to low byte.
}

extern "C" void traverse_bvh8_ww( CanonicalState* state )
{
    // Set acceleration so that lookups on  the node will work
    optixi_setLwrrentAcceleration( state );

    cort::OptixRay optixRay = optixi_getLwrrentRay( state );
    RayBvh8        ray;

    ray.orig      = optixRay.origin;
    ray.dir       = optixRay.direction;
    ray.tmin      = optixRay.tmin;
    ray.tmax      = optixRay.tmax;
    ray.mask      = 0;
    ray.translate = float3( 0.0f, 0.0f, 0.0f );
    ray.shear     = float3( 0.0f, 0.0f, 0.0f );
    ray.ilwDir    = float3( 0.0f, 0.0f, 0.0f );
    ray.majorAxis = 0;

    setupRay( ray );
    OctantData octantData = initOctantData( ray.dir );

    Bvh8StackEntry localStack[LOCAL_STACK_SIZE];
    Bvh8StackEntry stackTop;
    int            stackPtr  = 0;
    int            stackBase = 0;

    bvh8StackInit( stackBase, stackPtr );
    stackTop.hits          = 1 << 24;
    stackTop.firstChildIdx = 0;  // root

    // Traversal loop.
    while( true )
    {
        uint32_t primitiveHits;
        uint32_t itemStart;

        if( stackTop.hits > 0x00FFFFFF )  // if inner node hits
        {
            int idx       = findLeadingOne( stackTop.hits );
            stackTop.hits = bfi( 0, stackTop.hits, idx, 1 );
            idx -= 24;
            idx ^= ( octantData.duplicatedOctant & 0x7 );  // inner node idx in range [0,7]
            uint32_t validChildren = stackTop.hits;        // Valid mask in low byte.
            idx = popc( validChildren & ~( -1 << idx ) );  // Compute number if sibling nodes in memory before this node.

            // Issue global loads as early as possible
            BVH8Node node;
            node.header0 = getBufferElement_bvh( state, 16, 0, 5 * ( stackTop.firstChildIdx + idx ) + 0 );
            node.header1 = getBufferElement_bvh( state, 16, 0, 5 * ( stackTop.firstChildIdx + idx ) + 1 );
            node.boxes0  = getBufferElement_bvh( state, 16, 0, 5 * ( stackTop.firstChildIdx + idx ) + 2 );
            node.boxes1  = getBufferElement_bvh( state, 16, 0, 5 * ( stackTop.firstChildIdx + idx ) + 3 );
            node.boxes2  = getBufferElement_bvh( state, 16, 0, 5 * ( stackTop.firstChildIdx + idx ) + 4 );


            // If stacktop still contains hits to test, push it to stack.
            if( stackTop.hits > 0x00FFFFFF )
                bvh8StackPush( stackTop, stackPtr, stackBase, localStack );

            Bvh8StackEntry leafHits;

            ray.tmax = optixi_getLwrrentTmax( state );

            intersectNode( node, ray, octantData, stackTop, leafHits );
            itemStart     = leafHits.firstChildIdx;
            primitiveHits = leafHits.hits;
        }
        else
        {
            primitiveHits = stackTop.hits;
            itemStart     = stackTop.firstChildIdx;
            stackTop.hits = 0;
        }

        int ntrav = popc( ballot( true ) );
        while( primitiveHits )
        {
            if( popc( ballot( true ) ) * 5 < ntrav )  // reduces simd-eff in inner node test. postpone leaf in regs?
            {
                // never leave tris to be tested to stackTop, always push to stack if any remain.
                Bvh8StackEntry tricluster;
                tricluster.firstChildIdx = itemStart;
                tricluster.hits          = primitiveHits;

                bvh8StackPush( tricluster, stackPtr, stackBase, localStack );
                break;
            }

            // Select a Triangle
            int idx = findLeadingOne( primitiveHits );
            // Clear it from list
            primitiveHits = bfi( 0, primitiveHits, idx, 1 );

            intersectLeaf( state, itemStart + idx );
        }


        // pop
        if( stackTop.hits <= 0x00FFFFFF )
        {
            if( stackPtr == stackBase )
            {
                break;
            }
            stackTop = bvh8StackPop( stackPtr, stackBase, localStack );
        }
    }  // traversal
}


//------------------------------------------------------------------------------
void printAcceleration( CanonicalState* state )
{
    AbstractGroupHandle g     = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    AccelerationHandle  accel = AbstractGroup_getAcceleration( state, g );
    printf( "acceleration:%08x (%d)", accel, accel );
}

//------------------------------------------------------------------------------
void printRay( const cort::OptixRay& ray )
{
    printf( "ray-o:[%7.3f,%7.3f,%7.3f] d:[%7.3f,%7.3f,%7.3f] t:[%g,%g]\n", ray.origin.x, ray.origin.y, ray.origin.z,
            ray.direction.x, ray.direction.y, ray.direction.z, ray.tmin, ray.tmax );
}

//------------------------------------------------------------------------------
// A single while loop that includes intersection.
extern "C" void traverse_bvh_dbg( CanonicalState* state )
{
    cort::OptixRay ray;
    float3         iD, oiD;
    unsigned       hit0, hit1, leafBegin, leafEnd, gapBegin, gapEnd;

    bool     print = cort::Raygen_isPrintEnabledAtLwrrentLaunchIndex( state );
    unsigned lwr   = init( state, &ray, &iD, &oiD, print );
    unsigned localStack[LOCAL_STACK_SIZE];
    int      sp = 0, sb = 0;
    stackInit( sp, sb );
    PMEVENT( 0 );

    unsigned startLwr = lwr;
    if( print )
    {
        printAcceleration( state );
        printf( " start lwr:%d\n", startLwr );
        printRay( ray );
    }
    while( lwr )
    {
        PMEVENT( 1 );
        float   tmin0, tmin1, ray_tmax = optixi_getLwrrentTmax( state );
        BvhNode node = intersectNodeAabbs( state, lwr, iD, oiD, ray.tmin, ray_tmax, &tmin0, &tmin1, &hit0, &hit1 );
        if( print )
        {
            printNode( lwr, node );
            printf( "         child0 - %s (t=%f)   child1 - %s (t=%f)\n", hit0 ? "hit" : "miss", tmin0, hit1 ? "hit" : "miss", tmin1 );
        }

        lwr = chooseNextNode( node, tmin0, tmin1, hit0, hit1, localStack, &sp, sb );

        getLeafRange( node, hit0, hit1, &leafBegin, &leafEnd, &gapBegin, &gapEnd );

        unsigned i = leafBegin;
        while( i < leafEnd )
        {
            PMEVENT( 2 );
            intersectLeaf( state, i, print );
            ++i;
            if( i == gapBegin )
                i = gapEnd;
        }
        PMEVENT( 3 );
    }
    PMEVENT( 4 );
    if( print )
    {
        printAcceleration( state );
        printf( " end\n" );
    }
}

//------------------------------------------------------------------------------
extern "C" void traverse_noaccel_group( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    AbstractGroupHandle g        = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int        bufferId = AbstractGroup_getChildren( state, g );
    uint                i        = Buffer_getSizeFromId( state, bufferId ).x;  // implicit truncation
    while( i-- )
    {
        uint giOffset = getGiOffset_default( state, i );
        optixi_intersectNode( state, giOffset );
    }
}

//------------------------------------------------------------------------------
extern "C" void traverse_noaccel_geometry( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    AbstractGroupHandle g        = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int        bufferId = AbstractGroup_getChildren( state, g );
    uint                i        = Buffer_getSizeFromId( state, bufferId ).x;  // implicit truncation
    while( i-- )
    {
        uint         giOffset      = getGiOffset_default( state, i );
        unsigned int def           = 99;
        char*        addr          = Runtime_lookupVariableAddress( state, 12, (char*)&def );
        uint         primCount     = getBufferElement_prim_counts( state, 4, 0, i );
        uint         primIdxOffset = optixi_getPrimitiveIndexOffset( state, giOffset );
        for( int primIdx = primIdxOffset; primIdx < primCount + primIdxOffset; ++primIdx )
            optixi_intersectPrimitive( state, giOffset, primIdx );
    }
}

//------------------------------------------------------------------------------
extern "C" void bounds_selector( CanonicalState* state )
{
    // Note: will not get used for gathering motion boxes. These are
    // computed on the host.
    GeneralBB*     genbb    = (GeneralBB*)AABB_getPtr( state );
    SelectorHandle sel      = GraphNode_colwertToSelectorHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int   bufferId = Selector_getChildren( state, sel );
    uint           i        = Buffer_getSizeFromId( state, bufferId ).x;  // implicit truncation to 32-bit
    Aabb           aabb;
    aabb.ilwalidate();
    while( i-- )
    {
        const unsigned int  eltSize = sizeof( unsigned int );
        char                stackTemp[eltSize];
        const unsigned int* addr =
            reinterpret_cast<const unsigned int*>( Buffer_getElementAddress1dFromId( state, bufferId, eltSize, stackTemp, i ) );
        uint      childOffset = *addr;
        GeneralBB childbb;
        Runtime_computeGraphNodeGeneralBB( state, childOffset, &childbb );
        aabb.include( childbb );
    }
    genbb->set( aabb );
}

//------------------------------------------------------------------------------
extern "C" void bounds_noaccel_group( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    GeneralBB*          genbb    = (GeneralBB*)AABB_getPtr( state );
    AbstractGroupHandle g        = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int        bufferId = AbstractGroup_getChildren( state, g );
    uint                i        = Buffer_getSizeFromId( state, bufferId ).x;  // implicit truncation to 32-bit

    // Clear the motion index so that we can relwrsively compute a
    // union.
    int motionIndex = AABB_setMotionIndex( state, 0 );

    // Collect AABB union
    Aabb aabb;
    aabb.ilwalidate();
    while( i-- )
    {
        uint      childOffset = getGiOffset_default( state, i );
        GeneralBB childbb;
        Runtime_computeGraphNodeGeneralBB( state, childOffset, &childbb );
        aabb.include( childbb );
    }

    // Save or return the box
    if( motionIndex == ~0 )
    {
        Aabb* aabbs = (Aabb*)AABB_getPtr( state );
        aabbs[0]    = aabb;
    }
    else
    {
        GeneralBB* genbb = (GeneralBB*)AABB_getPtr( state );
        genbb->set( aabb );
    }

    // Reset the motion index
    AABB_setMotionIndex( state, motionIndex );
}

//------------------------------------------------------------------------------
extern "C" void bounds_noaccel_geometry( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    // Note: will not get used for gathering motion boxes. These are
    // computed on the host.
    GeneralBB*          genbb    = (GeneralBB*)AABB_getPtr( state );
    AbstractGroupHandle g        = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    unsigned int        bufferId = AbstractGroup_getChildren( state, g );
    uint                i        = Buffer_getSizeFromId( state, bufferId ).x;  // implicit truncation to 32-bit
    Aabb                aabb;
    aabb.ilwalidate();
    while( i-- )
    {
        uint giOffset = getGiOffset_default( state, i );
        Aabb childbb;
        int  motionStep = AABB_getMotionIndex( state );
        Runtime_computeGeometryInstanceAABB( state, giOffset, i, motionStep, (float*)&childbb );
        aabb.include( childbb );
    }
    genbb->set( aabb );
}

//------------------------------------------------------------------------------
extern "C" void bounds_transform( CanonicalState* state )
{
    // Note: will not get used for gathering motion boxes. These are
    // computed on the host.
    GeneralBB*      genbb = (GeneralBB*)AABB_getPtr( state );
    TransformHandle self  = GraphNode_colwertToTransformHandle( state, Traversal_getLwrrentNode( state ) );

    // compute child bb
    GeneralBB childbb;
    childbb.ilwalidate();
    childbb.anchor = childbb.v0 = childbb.v1 = childbb.v2 = float3( 0, 0, 0 );
    Runtime_computeGraphNodeGeneralBB( state, Transform_getChild( state, self ), &childbb );

    // Transform bb
    const Matrix4x4 m = Transform_getMatrix( state, self );
    genbb->anchor     = m * float4( childbb.anchor, 1.f );
    genbb->v0         = m * float4( childbb.v0, 0.f );
    genbb->v1         = m * float4( childbb.v1, 0.f );
    genbb->v2         = m * float4( childbb.v2, 0.f );
    genbb->valid      = childbb.valid;
}

//------------------------------------------------------------------------------
extern "C" void bounds_bvh_nomotion( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    // Pull the bounding box from the root of the bvh
    uint    root     = 0;
    BvhNode rootNode = fetchNode( state, root );
    Aabb    aabb( rootNode.bbmin0, rootNode.bbmax0 );

    if( AABB_getMotionIndex( state ) == ~0 )
    {
        // Save motion aabb for single step
        Aabb* aabbs = (Aabb*)AABB_getPtr( state );
        aabbs[0]    = aabb;
    }
    else
    {
        // Save it as a general bounding box
        GeneralBB* genbb = (GeneralBB*)AABB_getPtr( state );
        genbb->set( aabb );
    }
}

//------------------------------------------------------------------------------
extern "C" void bounds_bvh_motion( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    // Pull the bounding box from the root(s) of the bvh and compute a
    // union for the generalbb. Optionally save the boxes for each
    // motion step.
    AbstractGroupHandle      g     = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    AccelerationHandle       accel = AbstractGroup_getAcceleration( state, g );
    MotionAccelerationHandle maccel       = Acceleration_colwertToMotionAccelerationHandle( state, accel );
    unsigned int             motionSteps  = MotionAcceleration_getMotionSteps( state, maccel );
    unsigned int             motionStride = MotionAcceleration_getMotionStride( state, maccel );

    if( AABB_getMotionIndex( state ) == ~0 )
    {
        // Gather all motion steps
        Aabb* aabbs = (Aabb*)AABB_getPtr( state );
        for( unsigned int i = 0; i < motionSteps; ++i )
        {
            BvhNode rootNode = fetchNode( state, i * motionStride );
            Aabb    aabb( rootNode.bbmin0, rootNode.bbmax0 );
            aabbs[i] = aabb;
        }
    }
    else
    {
        // Compute the union across all motion steps
        GeneralBB* genbb = (GeneralBB*)AABB_getPtr( state );
        Aabb       unionAabb;
        unionAabb.ilwalidate();
        for( unsigned int i = 0; i < motionSteps; ++i )
        {
            BvhNode rootNode = fetchNode( state, i * motionStride );
            Aabb    aabb( rootNode.bbmin0, rootNode.bbmax0 );
            unionAabb.include( aabb );
        }
    }
}

//------------------------------------------------------------------------------
extern "C" void bounds_bvh8( CanonicalState* state )
{
    // Set acceleration so that lookups on the node will work
    optixi_setLwrrentAcceleration( state );

    // Pull the bounding box from the root of the bvh
    uint root = 0;
    // Issue global loads as early as possible
    BVH8Node node;
    node.header0 = getBufferElement_bvh( state, 16, 0, 0 );
    node.header1 = getBufferElement_bvh( state, 16, 0, 1 );
    node.boxes0  = getBufferElement_bvh( state, 16, 0, 2 );
    node.boxes1  = getBufferElement_bvh( state, 16, 0, 3 );
    node.boxes2  = getBufferElement_bvh( state, 16, 0, 4 );
    Aabb aabb    = node.getAabb();

    if( AABB_getMotionIndex( state ) == ~0 )
    {
        // Save motion aabb for single step
        Aabb* aabbs = (Aabb*)AABB_getPtr( state );
        aabbs[0]    = aabb;
    }
    else
    {
        // Save it as a general bounding box
        GeneralBB* genbb = (GeneralBB*)AABB_getPtr( state );
        genbb->set( aabb );
    }
}

//------------------------------------------------------------------------------
extern "C" float* optixi_getAABBArgToComputeAABB( CanonicalState* state );
extern "C" void bounds_rtctransform( CanonicalState* state )
{
    // Use optixi_getAABBArgToComputeAABB instead of AABB_getPtr so
    // that rtxcompile can understand it.
    GeneralBB* genbb = (GeneralBB*)optixi_getAABBArgToComputeAABB( state );

    // SGP TODO: This will not get ilwoked for 2-level traversal.
    printf( ":where to get bounds?\n" );  // And we do not always need it...
#if 0
    TransformHandle self = GraphNode_colwertToTransformHandle( state, Traversal_getLwrrentNode( state ) );

    // compute child bb
    GeneralBB childbb;
    childbb.ilwalidate();
    childbb.anchor = childbb.v0 = childbb.v1 = childbb.v2 = float3(0,0,0);
    Runtime_computeGraphNodeGeneralBB( state, Transform_getChild( state, self ), &childbb );

    // Transform bb
    const Matrix4x4 m  = Transform_getMatrix( state, self );
    genbb->anchor = m * float4( childbb.anchor, 1.f );
    genbb->v0 = m * float4( childbb.v0, 0.f );
    genbb->v1 = m * float4( childbb.v1, 0.f );
    genbb->v2 = m * float4( childbb.v2, 0.f );
    genbb->valid = childbb.valid;
#endif
}

//------------------------------------------------------------------------------
extern "C" void bounds_rtcbvh_nomotion( CanonicalState* state )
{
    // Use optixi_getAABBArgToComputeAABB instead of AABB_getPtr so
    // that rtxcompile can understand it.
    GeneralBB* genbb = (GeneralBB*)optixi_getAABBArgToComputeAABB( state );

    // SGP TODO: This will not get ilwoked for 2-level traversal.
    printf( ":where to get bounds?\n" );
#if 0
    uint root = 0;
    BvhNode rootNode = fetchNode( state, root );
    Aabb aabb( rootNode.bbmin0, rootNode.bbmax0 );
    genbb->set( aabb );
    if( aabbOutputPtr )
        *aabbOutputPtr = aabb;
#endif
}

//------------------------------------------------------------------------------
extern "C" void bounds_rtcbvh_motion( CanonicalState* state )
{
    // Use optixi_getAABBArgToComputeAABB instead of AABB_getPtr so
    // that rtxcompile can understand it.
    GeneralBB* genbb = (GeneralBB*)optixi_getAABBArgToComputeAABB( state );

    // SGP TODO: This will not get ilwoked for 2-level traversal.
    printf( ":where to get bounds?\n" );
#if 0
    uint root = 0;
    BvhNode rootNode = fetchNode(state, root);
    Aabb aabb(rootNode.bbmin0, rootNode.bbmax0);
    genbb->set(aabb);
    if(aabbOutputPtr)
        *aabbOutputPtr = aabb;
#endif
}
