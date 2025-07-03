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

#include "CORTTypes.h"

#include "../Objects/SemanticType.h"

// This file has a lot of custom formatting, leave this off by default, and turn it on
// when needed.

// clang-format off


/*
 * Simple runtime available for use in clang-based runtime code.
 * Keep this using simple types if possible.
 */

#if defined(__GNUC__)
#  define __align__(n) \
        __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#  define __align__(n) \
        __declspec(align(n))
#else /* __GNUC__ || _MSC_VER */
#error "Unknown compiler.  You need to figure out how to set alignment"
#endif

/*
 * Macro for helping to define functions that can be overridden in alternative exelwtion
 * strategies.
 */
#ifdef OPTIX_BUILD_RUNTIME
#  define CORT_OVERRIDABLE __attribute__ ((weak))
#endif

/*
 * Constants
 */
namespace cort {
static const uint warpBits = 5;
static const uint warpSize = ( 1 << warpBits );
}

// clang-format on
#ifdef OPTIX_BUILD_RUNTIME

/*
 * LWCA functions - implemented in CommonRuntime_ll.ll
 */
namespace lwca {
cort::uint3 tid();
cort::uint3 ntid();
cort::uint3 ctaid();
cort::uint3 nctaid();

cort::uint   warpid();
cort::uint   nwarpid();
cort::uint   smid();
cort::uint   nsmid();
cort::uint64 gridid();

cort::uint laneid();
cort::uint lanemask_eq();
cort::uint lanemask_le();
cort::uint lanemask_lt();
cort::uint lanemask_ge();
cort::uint lanemask_gt();

cort::uint ffs( cort::uint );     // Returns first most significant non-sign bit [0,31].
cort::uint popc( cort::uint );    // Count the number of one bits.
cort::uint ballot( cort::uint );  // Copies the predicate from each thread into the corresponding bit position.
cort::uint float_as_int( float );
float      int_as_float( cort::uint );

cort::uint   clock();
cort::uint64 clock64();

void dbgbreak();

int vprintf( const char*, char* );

// math intrinsics
float        uint2float_rz( unsigned int );
float        int2float_rz( int );
int          float2int_rz( float );
int          float2int_rm( float );
float        minf( float, float );
float        maxf( float, float );
int          mini( int, int );
int          maxi( int, int );
unsigned int minui( unsigned int, unsigned int );
unsigned int maxui( unsigned int, unsigned int );
int minmin( int a, int b, int c );  // min(min(a,b),c)
int minmax( int a, int b, int c );  // max(min(a,b),c)
int maxmin( int a, int b, int c );  // min(max(a,b),c)
int maxmax( int a, int b, int c );  // max(max(a,b),c)
float floorf( float );
float ceilf( float );
float mul_rz( float, float );
float add_rz( float, float );
float rcp( float );
float fmad( float a, float b, float c );     // a*b + c
float fmad_rp( float a, float b, float c );  // a*b + c, round up
float fmad_rm( float a, float b, float c );  // a*b + c, round down
float saturate( float );                     // clamp to [0,1]
float fabs( float x );
int shfl( int value, int lane ); // shfl without sync deprecated ( ptx6.0 )
int shfl_xor( int value, int laneMask ); // shfl_xor without sync deprecated ( ptx6.0 )
int shfl_sync( int value, int lane, int threadMask ); // shfl_sync ( as of ptx6.0 )
int shfl_xor_sync( int value, int laneMask, int threadMask ); // shfl_xor_sync ( as of ptx6.0 )
unsigned int clz( unsigned int value );  // Count leading zeros.
float exp2( float );
float log2( float );
int select( int a, int b, int c );  // c ? a : b
unsigned int permute( unsigned int indices, unsigned int valuesLo, unsigned int valuesHi );

// Bitfield extract/insert.
unsigned int bfe( unsigned int val, int pos, int len );
unsigned int bfi( unsigned int src, unsigned int dst, int pos, int len );

// Extract byte and shift it left.
unsigned int vshl_clamp_b0( unsigned int val, unsigned int shift );
unsigned int vshl_clamp_b1( unsigned int val, unsigned int shift );
unsigned int vshl_clamp_b2( unsigned int val, unsigned int shift );
unsigned int vshl_clamp_b3( unsigned int val, unsigned int shift );

// Extract byte value, shift it left with another extracted byte and add the result to u32.
unsigned int vshl_wrap_add_b0_b0( unsigned int val, unsigned int shift, unsigned int addend );
unsigned int vshl_wrap_add_b1_b1( unsigned int val, unsigned int shift, unsigned int addend );
unsigned int vshl_wrap_add_b2_b2( unsigned int val, unsigned int shift, unsigned int addend );
unsigned int vshl_wrap_add_b3_b3( unsigned int val, unsigned int shift, unsigned int addend );
}

#endif
// clang-format off

namespace cort {


#if defined(OPTIX_BUILD_RUNTIME) && (__cplusplus >= 201103L)
// TODO SGP: can these be retired in favor of the now correct printf lowering?

// Internal cort debugging printf facility only available with c++11 due to
// templated function parameters


// printf expects floats to be promoted to doubles.  This bit of template specialization
// will take care of that.  It also expects char/short to be promoted to int.
template<typename T> struct PrintPacked { T t; };
template<>           struct PrintPacked<float> { double t; };
template<>           struct PrintPacked<char>                    { int t; };
template<>           struct PrintPacked<unsigned char>  { unsigned int t; };
template<>           struct PrintPacked<short>                    { int t; };
template<>           struct PrintPacked<unsigned short>  { unsigned int t; };

struct PrintVoid {};
template<typename Arg1=PrintVoid, typename Arg2=PrintVoid, typename Arg3=PrintVoid,
         typename Arg4=PrintVoid, typename Arg5=PrintVoid, typename Arg6=PrintVoid,
         typename Arg7=PrintVoid, typename Arg8=PrintVoid, typename Arg9=PrintVoid,
         typename Arg10=PrintVoid, typename Arg11=PrintVoid, typename Arg12=PrintVoid>
int printf(const char* fmt,
           Arg1 a1=Arg1(), Arg2 a2=Arg2(), Arg3 a3=Arg3(),
           Arg4 a4=Arg4(), Arg5 a5=Arg5(), Arg6 a6=Arg6(),
           Arg7 a7=Arg7(), Arg8 a8=Arg8(), Arg9 a9=Arg9(),
           Arg10 a10=Arg10(), Arg11 a11=Arg11(), Arg12 a12=Arg12())
{
  struct {
    PrintPacked<Arg1> a1; PrintPacked<Arg2> a2; PrintPacked<Arg3> a3;
    PrintPacked<Arg4> a4; PrintPacked<Arg5> a5; PrintPacked<Arg6> a6;
    PrintPacked<Arg7> a7; PrintPacked<Arg8> a8; PrintPacked<Arg9> a9;
    PrintPacked<Arg10> a10; PrintPacked<Arg11> a11; PrintPacked<Arg12> a12;
  } packed = { a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 };
  return lwca::vprintf( fmt, reinterpret_cast<char*>(&packed) );
}
#endif

// clang-format on
#ifdef OPTIX_BUILD_RUNTIME

// TODO SGP: These should go into the lwca namespace.
int atomicAdd( int* address, int val );
unsigned int atomicAdd( unsigned int* address, unsigned int val );
unsigned long long int atomicAdd( unsigned long long int* address, unsigned long long int val );
float atomicAdd( float* address, float val );

int atomicSub( int* address, int val );
unsigned int atomicSub( unsigned int* address, unsigned int val );

int atomicExch( int* address, int val );
unsigned int atomicExch( unsigned int* address, unsigned int val );
unsigned long long int atomicExch( unsigned long long int* address, unsigned long long int val );
float atomicExch( float* address, float val );

int atomicMin( int* address, int val );
unsigned int atomicMin( unsigned int* address, unsigned int val );
unsigned long long int atomicMin( unsigned long long int* address, unsigned long long int val );

int atomicMax( int* address, int val );
unsigned int atomicMax( unsigned int* address, unsigned int val );
unsigned long long int atomicMax( unsigned long long int* address, unsigned long long int val );

unsigned int atomicInc( unsigned int* address, unsigned int val );

unsigned int atomicDec( unsigned int* address, unsigned int val );

int atomicCAS( int* address, int compare, int val );
unsigned int atomicCAS( unsigned int* address, unsigned int compare, unsigned int val );
unsigned long long int atomicCAS( unsigned long long int* address, unsigned long long int compare, unsigned long long int val );

int atomicAnd( int* address, int val );
unsigned int atomicAnd( unsigned int* address, unsigned int val );
unsigned long long int atomicAnd( unsigned long long int* address, unsigned long long int val );

int atomicOr( int* address, int val );
unsigned int atomicOr( unsigned int* address, unsigned int val );
unsigned long long int atomicOr( unsigned long long int* address, unsigned long long int val );

int atomicXor( int* address, int val );
unsigned int atomicXor( unsigned int* address, unsigned int val );
unsigned long long int atomicXor( unsigned long long int* address, unsigned long long int val );
#endif
// clang-format off


struct FrameStatus;

enum class PagingMode
{
    UNKNOWN = 0,
    WHOLE_MIPLEVEL,
    SOFTWARE_SPARSE,
    LWDA_SPARSE_HYBRID,
    LWDA_SPARSE_HARDWARE
};

// Global state
//
// In order to add a member to this struct, remember to:
//   - Add it to this struct here in CommonRuntime.h.
//   - Ensure the type matches the one in C14nRuntime.ll. You may need to check
//     with the LLVM output if any additional padding was added.
//   - Add the corresponding getter function here in CommonRuntime.h and in CommonRuntime.cpp.
//   - Override the implementation in the Megakernel and RTX runtimes.
//
struct __align__( 16 ) Global
{     // Always available
  char*             objectRecords;
  Buffer*           bufferTable;
  TextureSampler*   textureTable;
  ProgramHeader*    programTable;
  TraversableHeader* traversableTable;
  unsigned short*   dynamicVariableTable;

  // demand load
  uint*             pageUsageBits;
  uint*             pageResidenceBits;
  uint64*           pageTable;
  uint64*           tileArrays;
  PagingMode        demandLoadMode;

  uint64*           profileData;
  FrameStatus*      statusReturn;
  unsigned int      objectRecordsSize;
  unsigned int      numBuffers;
  unsigned int      numTextures;
  unsigned int      numPrograms;
  unsigned int      numTraversables;
  uint3             launchDim;
  uint3             printIndex;
  bool              printEnabled;
  unsigned short    dimensionality;
  unsigned short    entry;
  unsigned int      rayTypeCount;

  // Progressive API and VCA Support
  unsigned int      subframeIndex;

  // For timeout detection
  long long         clocksBeforeTimeout;

  // Parameters for compute_aabb
  AabbRequest       aabbRequest;

  // MultiGPU
  // The following make the device aware of other GPUs active, and enumerates them.
  // A device know how many GPUs there are, and which index it corresponds to itself.
  //
  unsigned short    activeDeviceIndex;        // MultiGPU: Index of the current active device
  unsigned short    activeDeviceCount;        // MultiGPU: Number of devices active.

  // MultiGPU Static LoadBalancer
  //
  // The static load balancer uses global parameters to resolve at runtime the assignment of workloads
  // to multiple devices. Knowing the parameters, the number of devices and the device ID, each device
  // can decide without extra communication which part of the image it has to compute.
  //
  // Load Balancing uses a linear enumeration of tiles as basic partition unit (which can be defined as
  // stripes or rectangular areas in the image). Important is that it is linear (single index), conselwtive
  // (increment linearly), compact (no gaps), and unique (different tiles->different indexes).
  //
  // The actual tile assignment may happen at runtime in the GPU depending on the LoadBalancer Strategy used.
  // As long as we define the size of the stride, and how it is divided among GPUs and follow the constraints
  // for linear enumeration, the actual tile definition is left to the load balancer strategy.
  //
  // With Persistent Threads (using fullwarp), a LB tile is an actual tile, which is a warp.
  //
  // A stride is a collection of tiles of equal size to be computed by multiple GPUs.
  // Inside the stride, each device can compute its own start point and how many tiles to compute, based on
  // start and size.
  //
  // Example for 3 GPUs of different size:
  //
  //          +--------------------------+
  //          +   GPU1   | GPU2 |  GPU3  +
  //          +--------------------------+
  //          |   sz=3   | sz=1 |  sz=2  |
  //          |          |      |<-------- start GPU3=4
  //          |          |<--------------- start GPU2=3
  //          |<-------------------------- start GPU1=0
  //          |<------- stride=6 ------->|
  //
  unsigned short    loadBalancerStart;        // MultiGPU LB: Offset of the current device segment in the stride
  unsigned short    loadBalancerSize;         // MultiGPU LB: Size of the current device segment in the stride
  unsigned short    loadBalancerStride;       // MultiGPU LB: Stride size

  char              padding[14];               // Make padding explicit for comparison with C14nRuntime.ll
};

struct Raygen {
  uint3             launchIndex;
};

struct Exception {
  unsigned int      code;
  uint64            detail64[12];
};

struct TerminateRay {
  // This flag is set as soon as rtTerminateRay is called. It is used to
  // determine whether we need to unwind the stack after a function returns
  // that may have called rtTerminateRay.
  bool              isRequested;
};

// Lwrrently active lookup scopes
struct Scopes {
  LexicalScopeHandle scope1;
  LexicalScopeHandle scope2;
};
struct ActiveProgram {
  ProgramHandle     program;
};


/*
 * Call frame
 * Always struct ActiveProgram plus one of the other four
 * SGP: we should move these into the same call frames used by callable programs
 */

struct AabbParameters {
  // For calls to compute_bounds
  float*            ptr;
  unsigned int      primitive;
  unsigned int      motion_index;
};
struct TraversalParameters {
  // For calls to graphnodes (Group, GeometryGroup, Selector, Transform)
  GraphNodeHandle   lwrrentNode;
};
struct IntersectParameters {
  // For calls to intersectPrimitive
  unsigned int      primitive;
};


/*
 * Trace frame
 */
struct __align__(16) TraceFrame {
  inline TraceFrame() {} // we need this constructor since we have the private copy constructor

  // Traversal
  float               lwrrentTmax;
  float               committedTmax;
  float               rayTime;
  unsigned short      rayType;
  unsigned char       lwrrentTransformDepth;
  unsigned char       committedTransformDepth;
  // lwrrentTransforms and committedTransforms at end of struct

  // Intersect
  GeometryInstanceHandle lwrrentInstance;
  GeometryInstanceHandle committedInstance;
  MaterialHandle lwrrentMaterial;
  MaterialHandle committedMaterial;
  // lwrrentAttributes and committedAttributes at the end of the struct

  // The attribute switch marks whether we lwrrently read/write lwrrentAttributes or
  // committedAttributes. This allows us to get rid of explicit copies.
  bool attributeSwitch;

  // Ray payload. We only need to store a pointer, copies are inserted outside
  // Runtime_trace during canonicalization.
  char* payload;

  // These all need to be at the end of the stuct in case we need to resize them.  You
  // should never access these directly, but use the accessor functions.  Also, don't
  // add new items at the beginning or end of this list without adding the appropriate
  // accessors.
  char __align__(16) lwrrentAttributes[MAXATTRIBUTE_TOTALSIZE];
  char __align__(16) committedAttributes[MAXATTRIBUTE_TOTALSIZE];
  TransformHandle    lwrrentTransforms[MAXTRANSFORMDEPTH];
  TransformHandle    committedTransforms[MAXTRANSFORMDEPTH];

  private:
  // Don't allow this class to be copied
  TraceFrame(const TraceFrame&) = delete;
};


/*
 * Ray frame:
 */

struct RayFrame {
  float3            o;
  float3            d;
  float             tmin;
};


/*
 * Canonical state object
 */
struct CanonicalState {
  Global          global;
  Raygen          raygen;
  Exception       exception;
  TerminateRay    terminateRay;
  Scopes          scopes;

  // Call frame
  ActiveProgram   active;
  AabbParameters  aabb;
  TraversalParameters  traversal;
  IntersectParameters  intersect;

  // Trace frame
  TraceFrame*     trace;

  // Ray frame
  RayFrame*       ray;
};

/************************************************************************************************************
 * Runtime state access and manipulation
 ************************************************************************************************************/

// Canonical program methods.  Note: this is always implemented by a particular ES.  It
// has the semantics of an indirect function call to the specified canonical program
// ID. optixi_callIndirect_standard is reserved for standard optix programs functions
// that (lwrrently) take no arguments and have no return values.  All other signatures
// are dynamically created from the stub version.
typedef int ArgPlaceholder;
typedef int RetPlaceholder;
extern "C" void optixi_callIndirect_standard( CanonicalState* state, unsigned short cpid, optix::SemanticType callType );
extern "C" RetPlaceholder optixi_callIndirect_stub( CanonicalState* state, unsigned short cpid, optix::SemanticType callType, ArgPlaceholder );

// Runtime methods
void                    Runtime_trace( CanonicalState* state, unsigned topOffset, float o1, float o2, float o3, float d1, float d2, float d3, unsigned int rayType, float tmin, float tmax, float time, unsigned int hasTime, char* payload );
void                    Runtime_trace_global_payload( CanonicalState* state, unsigned int topOffset, float ox, float oy, float oz, float dx, float dy, float dz, unsigned   int rayType, float tmin, float tmax, float time, unsigned int hasTime );
void                    Runtime_traverse( CanonicalState* state, GraphNodeHandle node );
void                    Runtime_shade( CanonicalState* state );
void                    Runtime_intersectNode( CanonicalState* state, GraphNodeHandle node );
void                    Runtime_intersectPrimitive( CanonicalState* state, GeometryInstanceHandle gi, unsigned int primitiveIndex );
void                    Runtime_computeGeometryInstanceAABB( CanonicalState* state, GeometryInstanceHandle gi, unsigned int primitive, unsigned int motionIndex, float* aabb );
void                    Runtime_computeGroupChildAABB( CanonicalState* state, AbstractGroupHandle grp, unsigned int childIndex, Aabb* aabb );
void                    Runtime_computeGraphNodeGeneralBB( CanonicalState* state, GraphNodeHandle graphNode, GeneralBB* genbb );
void                    Runtime_gatherMotionAABBs( CanonicalState* state, GraphNodeHandle graphNode, Aabb* aabbs );
void                    Runtime_setLwrrentAcceleration( CanonicalState* state );
float4                  Runtime_applyLwrrentTransforms( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w );
float4                  Runtime_applyLwrrentTransforms_atMostOne( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w );
float                   Runtime_getLwrrentTime( CanonicalState* state );
Matrix4x4               Runtime_getTransform( CanonicalState* state, unsigned int transform_kind );
Matrix4x4               Runtime_getTransform_atMostOne( CanonicalState* state, unsigned int transform_kind );
void                    Runtime_visitTransformNode( CanonicalState* state );
char*                   Runtime_lookupVariableAddress( CanonicalState* state, unsigned short token, char* defaultValue );
char*                   Runtime_lookupVariableAddress_ProgramScope( CanonicalState* state, unsigned short token );
char*                   Runtime_lookupVariableAddress_Scope1( CanonicalState* state, unsigned short token );
char*                   Runtime_lookupVariableAddress_Scope2( CanonicalState* state, unsigned short token );
char*                   Runtime_lookupVariableAddress_GlobalScope( CanonicalState* state, unsigned short token );
unsigned int            Runtime_lookupIdVariableValue( CanonicalState* state, unsigned short token );
void                    Runtime_ilwokeProgram( CanonicalState* state, ProgramHandle program, optix::SemanticType callType, LexicalScopeHandle scope1, LexicalScopeHandle scope2 );
RetPlaceholder          Runtime_ilwokeBoundCallableProgram_proto( CanonicalState* state, ProgramId pid, ArgPlaceholder args);
RetPlaceholder          Runtime_ilwokeBindlessCallableProgram_proto( CanonicalState* state, ProgramId pid, ArgPlaceholder args );
OptixRay                Runtime_getLwrrentRay( CanonicalState* state );
OptixRay                Runtime_getLwrrentObjectSpaceRay( CanonicalState* state );

// Global methods
void                    Global_set( CanonicalState* state,
                                    FrameStatus*    staturReturn,
                                    char*           objectRecords,
                                    Buffer*         bufferTable,
                                    TextureSampler* textureTable,
                                    ProgramHeader*  programTable,
                                    TraversableHeader* traversableTable,
                                    unsigned short* dynamicVariableTable,
                                    unsigned int    numBuffers,
                                    unsigned int    numTextures,
                                    unsigned int    numPrograms,
                                    unsigned int    numTraversables,
                                    const uint3&    launchDim,
                                    const uint3&    printIndex,
                                    bool            printEnabled,
                                    unsigned short  dimensionality,
                                    unsigned short  entry,
                                    unsigned int    subframeIndex,
                                    const AabbRequest& aabbRequest );
FrameStatus*            Global_getStatusReturn(CanonicalState* state);
uint64*                 Global_getProfileData(CanonicalState* state);
uint3                   Global_getLaunchDim(CanonicalState* state);
uint3                   Global_getPrintIndex(CanonicalState* state);
bool                    Global_getPrintEnabled(CanonicalState* state);
unsigned short          Global_getDimensionality(CanonicalState* state);
unsigned short          Global_getEntry(CanonicalState* state);
unsigned int            Global_getRayTypeCount(CanonicalState* state);
long long               Global_getClocksBeforeTimeout(CanonicalState* state);

unsigned int            Global_getSubframeIndex(CanonicalState* state);

AabbRequest             Global_getAabbRequest( CanonicalState* state );

unsigned short          Global_getDeviceIndex(CanonicalState* state);
unsigned short          Global_getDeviceCount(CanonicalState* state);
unsigned short          Global_getLoadBalancerStart(CanonicalState* state);
unsigned short          Global_getLoadBalancerSize(CanonicalState* state);
unsigned short          Global_getLoadBalancerStride(CanonicalState* state);

char*                   Global_getObjectRecord( CanonicalState* state, ObjectRecordOffset offset );
template<class T, class H>
inline T*               Global_getObjectRecord( CanonicalState* state, H handle ) {
  return reinterpret_cast<T*>(Global_getObjectRecord( state, handle ));
}
Buffer                  Global_getBufferHeader( CanonicalState* state, unsigned int bufferId );
TextureSampler          Global_getTextureSamplerHeader( CanonicalState* state, unsigned int textureId );
ProgramHeader           Global_getProgramHeader( CanonicalState* state, unsigned int programId );
TraversableHeader       Global_getTraversableHeader( CanonicalState* state, unsigned int traversableId );
unsigned short*         Global_getDynamicVariableTable(CanonicalState* state, DynamicVariableTableOffset offset);
unsigned int            Global_getNumBuffers( CanonicalState* state );
unsigned int            Global_getNumTextures( CanonicalState* state );
unsigned int            Global_getNumPrograms( CanonicalState* state );
unsigned int            Global_getNumTraversables( CanonicalState* state );
char*                   Global_getBufferAddress( CanonicalState* state, unsigned short token );
char*                   Global_getTextureSamplerAddress( CanonicalState* state, unsigned short token );
char*                   Global_getProgramAddress( CanonicalState* state, unsigned short token );
char*                   Global_getTextureSamplerAddressFromId( CanonicalState* state, unsigned int textureId );
char*                   Global_getProgramAddressFromId( CanonicalState* state, unsigned int programId );

// Raygen methods
void                    Raygen_set( CanonicalState* state, uint3 launchIndex );
uint3                   Raygen_getLaunchIndex(CanonicalState* state);
bool                    Raygen_isPrintEnabledAtLwrrentLaunchIndex( CanonicalState* state );

// Exception methods
int                     Exception_throw( CanonicalState* state );
unsigned int            Exception_getCode(CanonicalState* state);
void                    Exception_setCode( CanonicalState* state, unsigned int code );
unsigned int            Exception_getDetail(CanonicalState* state, unsigned int which);
uint64                  Exception_getDetail64(CanonicalState* state, unsigned int which);
void                    Exception_setDetail( CanonicalState* state, unsigned int detail, unsigned int which );
void                    Exception_setDetail64( CanonicalState* state, uint64 detail64, unsigned int which );
bool                    Exception_checkStackOverflow( CanonicalState* state, unsigned int size );
unsigned int            Exception_checkIdIlwalid( unsigned int id, unsigned int tableSize );
unsigned int            Exception_checkBufferIdIlwalid( CanonicalState* state, unsigned int bufferId );
unsigned int            Exception_checkTextureIdIlwalid( CanonicalState* state, unsigned int textureId );
unsigned int            Exception_checkProgramIdIlwalid( CanonicalState* state, unsigned int programId );
void                    Exception_checkIlwalidProgramId( CanonicalState* state, unsigned int programId );
void                    Exception_checkIlwalidRay( CanonicalState* state, float ox, float oy, float oz, float dx, float dy, float dz, unsigned int rayType, float tmin, float tmax );

// TerminateRay methods.
void                    TerminateRay_initialize( CanonicalState* state );
void                    TerminateRay_saveState( CanonicalState* state, int traceCallSiteID );
void                    TerminateRay_unwindFinished( CanonicalState* state, int traceCallSiteID );
void                    TerminateRay_terminate( CanonicalState* state );
unsigned int            TerminateRay_unwind( CanonicalState* state );
void                    TerminateRay_setIsRequested( CanonicalState* state, bool isRequested );
bool                    TerminateRay_isRequested( CanonicalState* state );

// State.Scopes methods (scope management)
void                    Scopes_set( CanonicalState* state, LexicalScopeHandle scope1, LexicalScopeHandle scope2 );
void                    Scopes_set( CanonicalState* state, Scopes scopes );
Scopes                  Scopes_get( CanonicalState* state );
LexicalScopeHandle      Scopes_getScope1(CanonicalState* state);
void                    Scopes_setScope2( CanonicalState* state, LexicalScopeHandle scope );
LexicalScopeHandle      Scopes_getScope2(CanonicalState* state);

// ActiveProgram methods (active/current program)
void                    ActiveProgram_set( CanonicalState* state, ProgramHandle program );
ProgramHandle           ActiveProgram_get( CanonicalState* state );

// AABB methods (used only during AABB computation)
void                    AABB_set( CanonicalState* state, float* ptr, unsigned int primitive, unsigned int motion_index );
float*                  AABB_getPtr(CanonicalState* state);
unsigned int            AABB_getPrimitive(CanonicalState* state);
unsigned int            AABB_getMotionIndex(CanonicalState* state);
unsigned int            AABB_setMotionIndex(CanonicalState* state, unsigned int motionIndex );

// Traversal methods
void                    Traversal_set( CanonicalState* state, GraphNodeHandle node );
GraphNodeHandle         Traversal_getLwrrentNode(CanonicalState* state);

// Intersect methods (used only during primitive intersection)
void                    Intersect_set(CanonicalState* state, unsigned int primitiveIndex);
unsigned int            Intersect_getPrimitive(CanonicalState* state);

// RayFrame methods
RayFrame*               RayFrame_push( CanonicalState* state, RayFrame* oldray, float3 o, float3 d, float tmin );
void                    RayFrame_pop( CanonicalState* state, RayFrame* ray );
float3                  RayFrame_getRayO( CanonicalState* state );
float3                  RayFrame_getRayD( CanonicalState* state );
float                   RayFrame_getRayTmin( CanonicalState* state );

// TraceFrame methods
TraceFrame*             TraceFrame_push( CanonicalState* state, TraceFrame* newFrame, unsigned int rayType, float tmax, float rayTime, unsigned int hasTime, char* payload );
void                    TraceFrame_pop( CanonicalState* state, TraceFrame* frame );
bool                    TraceFrame_hasPotentialIntersection( CanonicalState* state, float tmax );
bool                    TraceFrame_reportIntersection( CanonicalState* state, unsigned int matlIndex );
void                    TraceFrame_ignoreIntersection( CanonicalState* state );
bool                    TraceFrame_intersectionWasIgnored( CanonicalState* state );
void                    TraceFrame_pushTransform( CanonicalState* state, TransformHandle node );
void                    TraceFrame_pushTransform_atMostOne( CanonicalState* state, TransformHandle node );
void                    TraceFrame_popTransform( CanonicalState* state );
void                    TraceFrame_popTransform_atMostOne( CanonicalState* state );
void                    TraceFrame_commitHit( CanonicalState* state );
void                    TraceFrame_commitHit_atMostOne( CanonicalState* state );
void                    TraceFrame_restoreHit( CanonicalState* state );
void                    TraceFrame_restoreHit_atMostOne( CanonicalState* state );
void                    TraceFrame_commitAttributes( CanonicalState* state );
void                    TraceFrame_restoreAttributes( CanonicalState* state );

unsigned short          TraceFrame_getRayType(CanonicalState* state);

void                    TraceFrame_setCommittedTmax( CanonicalState* state, float tmax );
float                   TraceFrame_getCommittedTmax( CanonicalState* state );
void                    TraceFrame_setCommittedInstance( CanonicalState* state, GeometryInstanceHandle gi );
GeometryInstanceHandle  TraceFrame_getCommittedInstance( CanonicalState* state );
void                    TraceFrame_setCommittedMaterial( CanonicalState* state, MaterialHandle material );
MaterialHandle          TraceFrame_getCommittedMaterial( CanonicalState* state );
void                    TraceFrame_setCommittedTransformDepth( CanonicalState* state, unsigned char depth );
unsigned char           TraceFrame_getCommittedTransformDepth( CanonicalState* state );
void                    TraceFrame_setCommittedTransformByDepth( CanonicalState* state, unsigned char depth, TransformHandle th );
TransformHandle         TraceFrame_getCommittedTransformByDepth( CanonicalState* state, unsigned char depth );

void                    TraceFrame_setLwrrentRayTime( CanonicalState* state, float rayTime );
float                   TraceFrame_getLwrrentRayTime( CanonicalState* state );

void                    TraceFrame_setLwrrentTmax( CanonicalState* state, float tmax );
float                   TraceFrame_getLwrrentTmax( CanonicalState* state );
void                    TraceFrame_setLwrrentInstance( CanonicalState* state, GeometryInstanceHandle gi );
GeometryInstanceHandle  TraceFrame_getLwrrentInstance( CanonicalState* state );
void                    TraceFrame_setLwrrentMaterial( CanonicalState* state, MaterialHandle material );
MaterialHandle          TraceFrame_getLwrrentMaterial( CanonicalState* state );
void                    TraceFrame_setLwrrentTransformDepth( CanonicalState* state, unsigned char depth );
unsigned char           TraceFrame_getLwrrentTransformDepth( CanonicalState* state );
void                    TraceFrame_setLwrrentTransformByDepth( CanonicalState* state, unsigned char depth, TransformHandle th );
TransformHandle         TraceFrame_getLwrrentTransformByDepth( CanonicalState* state, unsigned char depth );
char*                   TraceFrame_getLwrrentAttributeAddress( CanonicalState* state, unsigned short token, unsigned int offset );

// RTX only
bool                    TraceFrame_isTriangleHit(CanonicalState* state);
bool                    TraceFrame_isTriangleHitBackFace(CanonicalState* state);
bool                    TraceFrame_isTriangleHitFrontFace(CanonicalState* state);

void                    TraceFrame_setAttributeSwitch( CanonicalState* state, bool value );
bool                    TraceFrame_getAttributeSwitch( CanonicalState* state );

// Helpers for computing addresses in the traceframe
char*                   TraceFrame_getPayloadAddress(CanonicalState* state);
char*                   TraceFrame_getLwrrentAttributeFrame(CanonicalState* state);
char*                   TraceFrame_getCommittedAttributeFrame(CanonicalState* state);
TransformHandle*        TraceFrame_getLwrrentTransforms(CanonicalState* state);
TransformHandle*        TraceFrame_getCommittedTransforms(CanonicalState* state);

// Debug methods (cpu only at present)
void                    Debug_dump( CanonicalState* state, int id );

// Profiling methods
void                    Profile_setup( CanonicalState* state, uint64* profileData );
void                    Profile_count( CanonicalState* state, int index, uint64 added );
void                    Profile_event( CanonicalState* state, int index );
void                    Profile_timerStart( CanonicalState* state, int index );
void                    Profile_timerStop( CanonicalState* state, int index );

/************************************************************************************************************
 * Data access
 ************************************************************************************************************/

// Buffer methods
char*                   Buffer_getElementAddress1d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x);
char*                   Buffer_getElementAddress2d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y);
char*                   Buffer_getElementAddress3d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z);
char*                   Buffer_getElementAddress1dFromId( CanonicalState* state, unsigned int id, unsigned int eltSize, char* stackTmp, uint64 x);
char*                   Buffer_getElementAddress2dFromId( CanonicalState* state, unsigned int id, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y);
char*                   Buffer_getElementAddress3dFromId( CanonicalState* state, unsigned int id, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z);
char*                   Buffer_decodeBufferHeader1d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x);
char*                   Buffer_decodeBufferHeader2d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y);
char*                   Buffer_decodeBufferHeader3d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z);
size3                   Buffer_getSize( CanonicalState* state, unsigned short token );
size3                   Buffer_getSizeFromId( CanonicalState* state, unsigned int id );

// Acceleration methods
LexicalScopeHandle      Acceleration_colwertToLexicalScopeHandle( CanonicalState* state, AccelerationHandle accel );
MotionAccelerationHandle Acceleration_colwertToMotionAccelerationHandle( CanonicalState* state, AccelerationHandle accel );

// MotionAcceleration methods
float                   MotionAcceleration_getTimeBegin( CanonicalState* state, MotionAccelerationHandle maccel );
float                   MotionAcceleration_getTimeEnd( CanonicalState* state, MotionAccelerationHandle maccel );
unsigned int            MotionAcceleration_getMotionSteps( CanonicalState* state, MotionAccelerationHandle maccel );
unsigned int            MotionAcceleration_getMotionStride( CanonicalState* state, MotionAccelerationHandle maccel );

// GlobalScope  methods
GlobalScopeHandle       GlobalScope_get( CanonicalState* state );
ProgramHandle           GlobalScope_getRaygen( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short entry );
ProgramHandle           GlobalScope_getMiss( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short rayType );
ProgramHandle           GlobalScope_getException( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short index );
LexicalScopeHandle      GlobalScope_colwertToLexicalScopeHandle( CanonicalState* state, GlobalScopeHandle globalScope );

// LexicalScope methods
DynamicVariableTableOffset  LexicalScope_getDynamicVariableTable(CanonicalState* state, LexicalScopeHandle object);
char*                       LexicalScope_lookupVariable( CanonicalState* state, LexicalScopeHandle object, unsigned short token );
char*                       LexicalScope_lookupVariableUnchecked( CanonicalState* state, LexicalScopeHandle object, unsigned short token );
AbstractGroupHandle         LexicalScope_colwertToAbstractGroupHandle( CanonicalState* state, LexicalScopeHandle object );
GraphNodeHandle             LexicalScope_colwertToGraphNodeHandle( CanonicalState* state, LexicalScopeHandle object );
GeometryInstanceHandle      LexicalScope_colwertToGeometryInstanceHandle( CanonicalState* state, LexicalScopeHandle object );

// GraphNode methods
ProgramHandle           GraphNode_getTraverse( CanonicalState* state, GraphNodeHandle node );
ProgramHandle           GraphNode_getBBProgram( CanonicalState* state, GraphNodeHandle node );
TraversableId           GraphNode_getTraversableId( CanonicalState* state, GraphNodeHandle node );
AbstractGroupHandle     GraphNode_colwertToAbstractGroupHandle( CanonicalState* state, GraphNodeHandle node );
TransformHandle         GraphNode_colwertToTransformHandle( CanonicalState* state, GraphNodeHandle node );
SelectorHandle          GraphNode_colwertToSelectorHandle( CanonicalState* state, GraphNodeHandle node );
LexicalScopeHandle      GraphNode_colwertToLexicalScopeHandle( CanonicalState* state, GraphNodeHandle node );


// AbstractGroup methods
AccelerationHandle      AbstractGroup_getAcceleration( CanonicalState* state, AbstractGroupHandle g );
BufferId                AbstractGroup_getChildren( CanonicalState* state, AbstractGroupHandle g );

// GeometryInstance methods
GeometryHandle          GeometryInstance_getGeometry( CanonicalState* state, GeometryInstanceHandle gi );
unsigned int            GeometryInstance_getNumMaterials( CanonicalState* state, GeometryInstanceHandle gi );
MaterialHandle          GeometryInstance_getMaterial( CanonicalState* state, GeometryInstanceHandle gi, unsigned int matlIndex );
char*                   GeometryInstance_getMaterialAddress( CanonicalState* state, GeometryInstanceHandle gi, unsigned int matlIndex );
LexicalScopeHandle      GeometryInstance_colwertToLexicalScopeHandle( CanonicalState* state, GeometryInstanceHandle gi );

// Geometry methods
unsigned int            Geometry_getPrimitiveIndexOffset( CanonicalState* state, GeometryHandle geometry );
ProgramHandle           Geometry_getIntersectProgram( CanonicalState* state, GeometryHandle geometry );
ProgramHandle           Geometry_getAABBProgram( CanonicalState* state, GeometryHandle geometry );
LexicalScopeHandle      Geometry_colwertToLexicalScopeHandle( CanonicalState* state, GeometryHandle geometry );

// Material methods
ProgramHandle           Material_getAHProgram( CanonicalState* state, MaterialHandle material, unsigned char rayType );
ProgramHandle           Material_getCHProgram( CanonicalState* state, MaterialHandle material, unsigned char rayType );
LexicalScopeHandle      Material_colwertToLexicalScopeHandle( CanonicalState* state, MaterialHandle material );

// Program methods
ProgramId               Program_getProgramID( CanonicalState* state, ProgramHandle program );
unsigned short          Program_getCanonicalProgramID( CanonicalState* state, ProgramHandle program );
ProgramHandle           Program_getProgramHandle( CanonicalState* state, ProgramId pid );
LexicalScopeHandle      Program_colwertToLexicalScopeHandle( CanonicalState* state, ProgramHandle program );

// Selector methods
SelectorHandle          Selector_getLwrrent( CanonicalState* state );
BufferId                Selector_getChildren( CanonicalState* state, SelectorHandle selector );
char*                   Selector_getChildAddress( CanonicalState* state, unsigned int child );
GraphNodeHandle         Selector_getChildNode( CanonicalState* state, unsigned int child );
uint64                  Selector_getNumChildren( CanonicalState* state );

// Transform methods
GraphNodeHandle         Transform_getChild( CanonicalState* state, TransformHandle transform);
BufferId                Transform_getMotionDataBufferId( CanonicalState* state, TransformHandle transform);
Matrix4x4               Transform_getMatrix( CanonicalState* state, TransformHandle transform);
Matrix4x4               Transform_getMatrixAtLwrrentTime( CanonicalState* state, TransformHandle transform);
Matrix4x4               Transform_getIlwMatrix( CanonicalState* state, TransformHandle transform);
Matrix4x4               Transform_getIlwMatrixAtLwrrentTime( CanonicalState* state, TransformHandle transform);

/*
 * Texture methods used by canonicalization.  Texture flow looks like:
 * Texture_getElement_token_: performs variable lookup to determine the ID, calls Texture_getElement_id_
 * Texture_getElement_id_: Looks up texture header, calls either SW texture or HW texture
 * Texture_getElement_sw_: Software interpolation
 * Texture_getElement_hw_: Bindless(SM_30+) or Indirect (SM_20) hardware interpolation
 *
 * Specialization can short-circuit one or both steps.
 */

//
// Variable token-based lookup
//

// Texture size functions
unsigned int Texture_getElement_token_txq_width( CanonicalState* state, unsigned short token, bool hwonly, bool swonly );
unsigned int Texture_getElement_token_txq_height( CanonicalState* state, unsigned short token, bool hwonly, bool swonly );
unsigned int Texture_getElement_token_txq_depth( CanonicalState* state, unsigned short token, bool hwonly, bool swonly );
uint3        Texture_getElement_token_size( CanonicalState* state, unsigned short token, bool hwonly, bool swonly );

// Tex functions
float4 Texture_getElement_token_tex_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x );
float4 Texture_getElement_token_tex_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_token_tex_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z );
float4 Texture_getElement_token_tex_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x );
float4 Texture_getElement_token_tex_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y );
float4 Texture_getElement_token_tex_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z );
float4 Texture_getElement_token_tex_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z );

// TLD/"fetch" functions (linear memory only)
float4 Texture_getElement_token_texfetch_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x );
float4 Texture_getElement_token_texfetch_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_token_texfetch_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, int x, int y, int z ); // Not exposed in lwca
float4 Texture_getElement_token_texfetch_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, int x ); // Not exposed in lwca
float4 Texture_getElement_token_texfetch_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_token_texfetch_2dms( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int s, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_token_texfetch_a2dms( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int s, unsigned int a, int x, int y ); // Not exposed in lwca

// Mip level
float4 Texture_getElement_token_texlevel_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float level );
float4 Texture_getElement_token_texlevel_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float level );
float4 Texture_getElement_token_texlevel_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z, float level );
float4 Texture_getElement_token_texlevel_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float level );
float4 Texture_getElement_token_texlevel_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float level );
float4 Texture_getElement_token_texlevel_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z, float level);
float4 Texture_getElement_token_texlevel_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z, float level);

// Mip grad
float4 Texture_getElement_token_texgrad_1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float dPdx, float dPdy );
float4 Texture_getElement_token_texgrad_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_token_texgrad_3d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdx_z, float dPdy_x, float dPdy_y, float dPdy_z );
float4 Texture_getElement_token_texgrad_a1d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float dPdx, float dPdy );
float4 Texture_getElement_token_texgrad_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_token_texgrad_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_token_texgrad_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );

// TLD4
float4 Texture_getElement_token_tld4r_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_token_tld4g_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_token_tld4b_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_token_tld4a_2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_token_tld4r_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_token_tld4g_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_token_tld4b_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_token_tld4a_a2d( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_token_tld4r_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4g_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4b_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4a_lwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4r_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4g_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4b_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_token_tld4a_alwbe( CanonicalState* state, unsigned short token, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX


//
// OptiX texture ID-based lookup
//

// Texture size functions
unsigned int Texture_getElement_id_txq_width( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly );
unsigned int Texture_getElement_id_txq_height( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly );
unsigned int Texture_getElement_id_txq_depth( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly );
uint3        Texture_getElement_id_size( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly );

// Tex functions
float4 Texture_getElement_id_tex_1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x );
float4 Texture_getElement_id_tex_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_id_tex_3d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z );
float4 Texture_getElement_id_tex_a1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x );
float4 Texture_getElement_id_tex_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y );
float4 Texture_getElement_id_tex_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z );
float4 Texture_getElement_id_tex_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z );

// TLD/"fetch" functions (linear memory only)
float4 Texture_getElement_id_texfetch_1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, int x );
float4 Texture_getElement_id_texfetch_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_id_texfetch_3d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, int x, int y, int z ); // Not exposed in lwca
float4 Texture_getElement_id_texfetch_a1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, int x ); // Not exposed in lwca
float4 Texture_getElement_id_texfetch_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_id_texfetch_2dms( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int s, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_id_texfetch_a2dms( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int s, unsigned int a, int x, int y ); // Not exposed in lwca

// Mip level
float4 Texture_getElement_id_texlevel_1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float level );
float4 Texture_getElement_id_texlevel_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float level );
float4 Texture_getElement_id_texlevel_3d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z, float level );
float4 Texture_getElement_id_texlevel_a1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float level );
float4 Texture_getElement_id_texlevel_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float level );
float4 Texture_getElement_id_texlevel_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z, float level);
float4 Texture_getElement_id_texlevel_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z, float level);

// Mip grad
float4 Texture_getElement_id_texgrad_1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float dPdx, float dPdy );
float4 Texture_getElement_id_texgrad_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_id_texgrad_3d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdx_z, float dPdy_x, float dPdy_y, float dPdy_z );
float4 Texture_getElement_id_texgrad_a1d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float dPdx, float dPdy );
float4 Texture_getElement_id_texgrad_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_id_texgrad_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_id_texgrad_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );

// TLD4
float4 Texture_getElement_id_tld4r_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_id_tld4g_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_id_tld4b_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_id_tld4a_2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y );
float4 Texture_getElement_id_tld4r_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_id_tld4g_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_id_tld4b_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_id_tld4a_a2d( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_id_tld4r_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4g_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4b_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4a_lwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4r_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4g_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4b_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_id_tld4a_alwbe( CanonicalState* state, unsigned int texid, bool hwonly, bool swonly, unsigned int a, float x, float y, float z ); // Not exposed in PTX

//
// header-based software interpolation
//

// Texture size functions
unsigned int Texture_getElement_sw_txq_width( cort::TextureSampler texture );
unsigned int Texture_getElement_sw_txq_height( cort::TextureSampler texture );
unsigned int Texture_getElement_sw_txq_depth( cort::TextureSampler texture );
uint3        Texture_getElement_sw_size( cort::TextureSampler texture );

// Tex functions
float4 Texture_getElement_sw_tex_1d( cort::TextureSampler texture, float x );
float4 Texture_getElement_sw_tex_2d( cort::TextureSampler texture, float x, float y );
float4 Texture_getElement_sw_tex_3d( cort::TextureSampler texture, float x, float y, float z );
float4 Texture_getElement_sw_tex_a1d( cort::TextureSampler texture, unsigned int a, float x );
float4 Texture_getElement_sw_tex_a2d( cort::TextureSampler texture, unsigned int a, float x, float y );
float4 Texture_getElement_sw_tex_lwbe( cort::TextureSampler texture, float x, float y, float z );
float4 Texture_getElement_sw_tex_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z );

// TLD/"fetch" functions (linear memory only)
float4 Texture_getElement_sw_texfetch_1d( cort::TextureSampler texture, int x );
float4 Texture_getElement_sw_texfetch_2d( cort::TextureSampler texture, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_sw_texfetch_3d( cort::TextureSampler texture, int x, int y, int z ); // Not exposed in lwca
float4 Texture_getElement_sw_texfetch_a1d( cort::TextureSampler texture, unsigned int a, int x ); // Not exposed in lwca
float4 Texture_getElement_sw_texfetch_a2d( cort::TextureSampler texture, unsigned int a, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_sw_texfetch_2dms( cort::TextureSampler texture, unsigned int s, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_sw_texfetch_a2dms( cort::TextureSampler texture, unsigned int s, unsigned int a, int x, int y ); // Not exposed in lwca

// Mip level
float4 Texture_getElement_sw_texlevel_1d( cort::TextureSampler texture, float x, float level );
float4 Texture_getElement_sw_texlevel_2d( cort::TextureSampler texture, float x, float y, float level );
float4 Texture_getElement_sw_texlevel_3d( cort::TextureSampler texture, float x, float y, float z, float level );
float4 Texture_getElement_sw_texlevel_a1d( cort::TextureSampler texture, unsigned int a, float x, float level );
float4 Texture_getElement_sw_texlevel_a2d( cort::TextureSampler texture, unsigned int a, float x, float y, float level );
float4 Texture_getElement_sw_texlevel_lwbe( cort::TextureSampler texture, float x, float y, float z, float level);
float4 Texture_getElement_sw_texlevel_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z, float level);

// Mip grad
float4 Texture_getElement_sw_texgrad_1d( cort::TextureSampler texture, float x, float dPdx, float dPdy );
float4 Texture_getElement_sw_texgrad_2d( cort::TextureSampler texture, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_sw_texgrad_3d( cort::TextureSampler texture, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdx_z, float dPdy_x, float dPdy_y, float dPdy_z );
float4 Texture_getElement_sw_texgrad_a1d( cort::TextureSampler texture, unsigned int a, float x, float dPdx, float dPdy );
float4 Texture_getElement_sw_texgrad_a2d( cort::TextureSampler texture, unsigned int a, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_sw_texgrad_lwbe( cort::TextureSampler texture, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_sw_texgrad_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );

// TLD4
float4 Texture_getElement_sw_tld4r_2d( cort::TextureSampler texture, float x, float y );
float4 Texture_getElement_sw_tld4g_2d( cort::TextureSampler texture, float x, float y );
float4 Texture_getElement_sw_tld4b_2d( cort::TextureSampler texture, float x, float y );
float4 Texture_getElement_sw_tld4a_2d( cort::TextureSampler texture, float x, float y );
float4 Texture_getElement_sw_tld4r_a2d( cort::TextureSampler texture, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4g_a2d( cort::TextureSampler texture, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4b_a2d( cort::TextureSampler texture, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4a_a2d( cort::TextureSampler texture, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4r_lwbe( cort::TextureSampler texture, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4g_lwbe( cort::TextureSampler texture, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4b_lwbe( cort::TextureSampler texture, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4a_lwbe( cort::TextureSampler texture, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4r_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4g_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4b_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_sw_tld4a_alwbe( cort::TextureSampler texture, unsigned int a, float x, float y, float z ); // Not exposed in PTX

//
// Hardware-based interpolation based on indirect texture ID (Fermi)
// or bindless textures (Kepler or better).
//

// Texture size functions
unsigned int Texture_getElement_hw_txq_width( uint64 texref );
unsigned int Texture_getElement_hw_txq_height( uint64 texref );
unsigned int Texture_getElement_hw_txq_depth( uint64 texref );
uint3        Texture_getElement_hw_size( uint64 texref );

// Tex functions
float4 Texture_getElement_hw_tex_1d( uint64 texref, float x );
float4 Texture_getElement_hw_tex_2d( uint64 texref, float x, float y );
float4 Texture_getElement_hw_tex_3d( uint64 texref, float x, float y, float z );
float4 Texture_getElement_hw_tex_a1d( uint64 texref, unsigned int a, float x );
float4 Texture_getElement_hw_tex_a2d( uint64 texref, unsigned int a, float x, float y );
float4 Texture_getElement_hw_tex_lwbe( uint64 texref, float x, float y, float z );
float4 Texture_getElement_hw_tex_alwbe( uint64 texref, unsigned int a, float x, float y, float z );

// TLD/"fetch" functions (linear memory only)
float4 Texture_getElement_hw_texfetch_1d( uint64 texref, int x );
float4 Texture_getElement_hw_texfetch_2d( uint64 texref, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_hw_texfetch_3d( uint64 texref, int x, int y, int z ); // Not exposed in lwca
float4 Texture_getElement_hw_texfetch_a1d( uint64 texref, unsigned int a, int x ); // Not exposed in lwca
float4 Texture_getElement_hw_texfetch_a2d( uint64 texref, unsigned int a, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_hw_texfetch_2dms( uint64 texref, unsigned int s, int x, int y ); // Not exposed in lwca
float4 Texture_getElement_hw_texfetch_a2dms( uint64 texref, unsigned int s, unsigned int a, int x, int y ); // Not exposed in lwca

// Mip level
float4 Texture_getElement_hw_texlevel_1d( uint64 texref, float x, float level );
float4 Texture_getElement_hw_texlevel_2d( uint64 texref, float x, float y, float level );
float4 Texture_getElement_hw_texlevel_3d( uint64 texref, float x, float y, float z, float level );
float4 Texture_getElement_hw_texlevel_a1d( uint64 texref, unsigned int a, float x, float level );
float4 Texture_getElement_hw_texlevel_a2d( uint64 texref, unsigned int a, float x, float y, float level );
float4 Texture_getElement_hw_texlevel_lwbe( uint64 texref, float x, float y, float z, float level);
float4 Texture_getElement_hw_texlevel_alwbe( uint64 texref, unsigned int a, float x, float y, float z, float level);

// Mip grad
float4 Texture_getElement_hw_texgrad_1d( uint64 texref, float x, float dPdx, float dPdy );
float4 Texture_getElement_hw_texgrad_2d( uint64 texref, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_hw_texgrad_2d_isResident( uint64 texref, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, int *isResident );
float4 Texture_getElement_hw_texgrad_3d( uint64 texref, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdx_z, float dPdy_x, float dPdy_y, float dPdy_z );
float4 Texture_getElement_hw_texgrad_a1d( uint64 texref, unsigned int a, float x, float dPdx, float dPdy );
float4 Texture_getElement_hw_texgrad_a2d( uint64 texref, unsigned int a, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_hw_texgrad_lwbe( uint64 texref, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );
float4 Texture_getElement_hw_texgrad_alwbe( uint64 texref, unsigned int a, float x, float y, float z, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y );

// Mip grad footprint
uint4 Texture_getElement_hw_texgrad_footprint_2d( unsigned int granularity, uint64 texref, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, int *coversSingleMipLevel );
uint4 Texture_getElement_hw_texgrad_footprint_coarse_2d( unsigned int granularity, uint64 texref, float x, float y, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, int *coversSingleMipLevel );

// TLD4
float4 Texture_getElement_hw_tld4r_2d( uint64 texref, float x, float y );
float4 Texture_getElement_hw_tld4g_2d( uint64 texref, float x, float y );
float4 Texture_getElement_hw_tld4b_2d( uint64 texref, float x, float y );
float4 Texture_getElement_hw_tld4a_2d( uint64 texref, float x, float y );
float4 Texture_getElement_hw_tld4r_a2d( uint64 texref, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4g_a2d( uint64 texref, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4b_a2d( uint64 texref, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4a_a2d( uint64 texref, unsigned int a, float x, float y ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4r_lwbe( uint64 texref, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4g_lwbe( uint64 texref, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4b_lwbe( uint64 texref, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4a_lwbe( uint64 texref, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4r_alwbe( uint64 texref, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4g_alwbe( uint64 texref, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4b_alwbe( uint64 texref, unsigned int a, float x, float y, float z ); // Not exposed in PTX
float4 Texture_getElement_hw_tld4a_alwbe( uint64 texref, unsigned int a, float x, float y, float z ); // Not exposed in PTX

//
// Hardware-based interpolation based on indirect texture ID (both Fermi and Kepler)
// Used for texture heap
//

// TLD/"fetch" functions (linear memory only)
float4 Texture_getElement_hwtexref_texfetch_1d( unsigned int texref, int x );

// Internal functionality shared between different strategies
unsigned short findVariableOffset( unsigned short token, const unsigned short* table );

}
