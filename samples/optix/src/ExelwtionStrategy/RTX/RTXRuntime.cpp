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

#include <ExelwtionStrategy/RTX/RTXRuntime.h>

#include <ExelwtionStrategy/CORTTypes.h>

#include <Memory/DemandLoad/TileLocator.h>

#define TILE_INDEXING_DEVICE 1
#include <Memory/DemandLoad/TileIndexing.h>

#include <corelib/compiler/LWVMAddressSpaces.h>
#include <o6/optix.h>
#include <rtcore/interface/types.h>

using namespace cort;

#define __constant__ __attribute__( ( address_space( corelib::ADDRESS_SPACE_CONST ) ) )
#define __forceinline__ __attribute__( ( always_inline ) )

#ifndef FLT_EPSILON
#define FLT_EPSILON 1e-06
#endif

//#define ENABLE_PRINTING

#ifdef ENABLE_PRINTING
inline bool isPrintEnabled()
{
    uint3    threadIdx = lwca::tid();
    uint3    blockDim  = lwca::ntid();
    uint3    blockIdx  = lwca::ctaid();
    unsigned x         = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned y         = threadIdx.y + blockDim.y * blockIdx.y;
    return ( x == 640 ) && ( y == 360 );
}
#else
inline bool isPrintEnabled()
{
    return false;
}
#endif

extern "C" void RTX_throwException( int exceptionCode, int exceptionData[23] );

extern "C" bool RTX_isInfOrNan( float x );

extern "C" bool RTX_isNan( float x );

extern "C" int RTX_getExceptionCode();

///////////////////////////////////////////////////////////
//
// Access to the SBTRecordData
//
extern "C" SBTRecordData* RTX_getSBTRecordData();

CORT_OVERRIDABLE
SBTRecordData* getGeometryInstanceSBTRecordPointer()
{
    return RTX_getSBTRecordData();
}

LexicalScopeHandle cort::getMaterialHandle()
{
    SBTRecordData* data = getGeometryInstanceSBTRecordPointer();
    return data->GeometryInstanceData.materialOffset;
}

// NOTE: If the function below is not marked as noinline, later replacement by an argument of the
// bounding box program has no effect on uses in this file, such as e.g. the one in
// Megakernel_getObjectBase_GeometryInstance(), since they get inlined during offline compilation
// already.
__attribute__( ( noinline ) ) LexicalScopeHandle cort::getGeometryInstanceHandle()
{
    SBTRecordData* data = getGeometryInstanceSBTRecordPointer();
    return data->GeometryInstanceData.giOffset;
}

int cort::getSBTSkip( unsigned int matlIndex )
{
    return ( matlIndex == 0 ) ? 0 : getGeometryInstanceSBTRecordPointer()->GeometryInstanceData.skip + matlIndex - 1;
}

GeometryHandle cort::getGeometryHandle()
{
    GeometryInstanceHandle giOffset = getGeometryInstanceHandle();
    return GeometryInstance_getGeometry( nullptr, giOffset );
}

ProgramHandle cort::ActiveProgram_get( CanonicalState* state, optix::SemanticType callType )
{
    switch( callType )
    {
        case optix::ST_RAYGEN:
        case optix::ST_MISS:
        case optix::ST_EXCEPTION:
        case optix::ST_INTERNAL_AABB_ITERATOR:
        case optix::ST_INTERNAL_AABB_EXCEPTION:
            return RayGenMissExceptionProgram_get( state );

        case optix::ST_CLOSEST_HIT:
            return ClosestHitProgram_get( state );

        case optix::ST_ANY_HIT:
            return AnyHitProgram_get( state );

        case optix::ST_INTERSECTION:
            return IntersectionProgram_get( state );

        case optix::ST_ATTRIBUTE:
            return AttributeProgram_get( state );

        case optix::ST_BINDLESS_CALLABLE_PROGRAM:
        case optix::ST_BOUNDING_BOX:
            return BindlessCallableProgram_get( state );

        case optix::ST_BOUND_CALLABLE_PROGRAM:
            return BoundCallableProgram_get( state );

        case optix::ST_NODE_VISIT:  // TODO
        default:
            return {0};  // TODO: Exception? Failure in debug mode?
    }
}

ProgramHandle cort::RayGenMissExceptionProgram_get( CanonicalState* state )
{
    SBTRecordData* data = RTX_getSBTRecordData();
    return data->ProgramData.programOffset;
}

ProgramHandle cort::IntersectionProgram_get( CanonicalState* state )
{
    GeometryHandle geometry = getGeometryHandle();
    return Geometry_getIntersectProgram( state, geometry );
}

ProgramHandle cort::AttributeProgram_get( CanonicalState* state )
{
    GeometryHandle geometry = getGeometryHandle();
    return Geometry_getAttributeProgram( state, geometry );
}

ProgramHandle cort::ClosestHitProgram_get( CanonicalState* state )
{
    MaterialHandle material = getMaterialHandle();
    return Material_getCHProgram( state, material );
}

ProgramHandle cort::AnyHitProgram_get( CanonicalState* state )
{
    MaterialHandle material = getMaterialHandle();
    return Material_getAHProgram( state, material );
}

ProgramHandle cort::BindlessCallableProgram_get( CanonicalState* state )
{
    SBTRecordData* data = RTX_getSBTRecordData();
    return data->ProgramData.programOffset;
}

ProgramHandle cort::BoundCallableProgram_get( CanonicalState* state )
{
    SBTRecordData* data = RTX_getSBTRecordData();
    return data->ProgramData.programOffset;
}

LexicalScopeHandle cort::Program_colwertToLexicalScopeHandle( CanonicalState* state, ProgramHandle program )
{
    return program;
}


///////////////////////////////////////////////////////////
//
// Global constants that live in constant memory.
//

char* cort_castConstToGeneric( __constant__ char* );

extern __constant__ char const_Global[sizeof( Global )];

static __constant__ Global& get_const_Global()
{
    return *reinterpret_cast<__constant__ Global*>( const_Global );
}

extern __constant__ __align__( 16 ) char const_ObjectRecord[0];

extern __constant__ __align__( 16 ) char const_BufferHeaderTable[0];
static __constant__ Buffer* get_const_BufferHeaderTable()
{
    return reinterpret_cast<__constant__ Buffer*>( const_BufferHeaderTable );
}

extern __constant__ __align__( 16 ) char const_ProgramHeaderTable[0];
static __constant__ ProgramHeader* get_const_ProgramHeaderTable()
{
    return reinterpret_cast<__constant__ ProgramHeader*>( const_ProgramHeaderTable );
}

extern __constant__ __align__( 16 ) char const_TextureHeaderTable[0];
static __constant__ TextureSampler* get_const_TextureHeaderTable()
{
    return reinterpret_cast<__constant__ TextureSampler*>( const_TextureHeaderTable );
}

extern __constant__ unsigned int  const_MinMaxLaunchIndex[6];
static __constant__ unsigned int* get_const_MinMaxLaunchIndex()
{
    return reinterpret_cast<__constant__ unsigned int*>( const_MinMaxLaunchIndex );
}

///////////////////////////////////////////////////////////
//
// Access to object record.
//

extern "C" char* Megakernel_getObjectBase_GlobalScope( CanonicalState* state, unsigned int offset )
{
    // If the location of the GlobalScope moved away from the canonical object record, then
    // you would need to add a different base pointer.
    return Global_getObjectRecord( state, offset );
}

extern "C" char* Megakernel_getObjectBase_GeometryInstance( CanonicalState* state, unsigned int offset )
{
    GeometryInstanceHandle gi = getGeometryInstanceHandle();
    return Global_getObjectRecord( state, gi + offset );
}

extern "C" char* Megakernel_getObjectBase_Geometry( CanonicalState* state, unsigned int offset )
{
    GeometryInstanceHandle gi = getGeometryInstanceHandle();
    return Global_getObjectRecord( state, GeometryInstance_getGeometry( state, gi ) + offset );
}

extern "C" char* Megakernel_getObjectBase_Material( CanonicalState* state, unsigned int offset )
{
    MaterialHandle material = getMaterialHandle();
    return Global_getObjectRecord( state, material + offset );
}

extern "C" char* Megakernel_getObjectBase_Program( CanonicalState* state, unsigned int offset, optix::SemanticType callType )
{
    return Global_getObjectRecord( state, ActiveProgram_get( state, callType ) + offset );
}

TraversableId cort::GraphNode_getTraversableId( CanonicalState* state, GraphNodeHandle node )
{
    GraphNodeRecord* gnr = Global_getObjectRecord<GraphNodeRecord>( state, node );
    return gnr->traversableId;
}

ProgramHandle cort::GraphNode_getBBProgram( CanonicalState* state, GraphNodeHandle node )
{
    GraphNodeRecord* gnr = Global_getObjectRecord<GraphNodeRecord>( state, node );
    return gnr->bounds;
}

extern "C" uint64 RTX_getRtcTraversableHandle( CanonicalState* state, unsigned int topOffset )
{
    GraphNodeHandle gn            = {topOffset};
    TraversableId   traversableId = GraphNode_getTraversableId( state, gn );
    uint64          traversable   = Global_getTraversableHeader( state, traversableId ).traversable;
    return traversable;
}

extern "C" bool RTX_indexIsOutsideOfLaunch( CanonicalState* state )
{
    const uint3 idx     = Raygen_getLaunchIndex( state );
    const uint3 dim     = Global_getLaunchDim( state );
    const bool  outside = idx.x >= dim.x || idx.z >= dim.z;
    return outside;
}


///////////////////////////////////////////////////////////
//
// Access to const object record.
//

extern "C" __constant__ char* Megakernel_getObjectBase_GlobalScope_FromConst( CanonicalState* state, unsigned int offset )
{
    return const_ObjectRecord + offset;
}

extern "C" __constant__ char* Megakernel_getObjectBase_GeometryInstance_FromConst( CanonicalState* state, unsigned int offset )
{
    GeometryInstanceHandle gi = getGeometryInstanceHandle();
    return const_ObjectRecord + ( gi + offset );
}

extern "C" __constant__ char* Megakernel_getObjectBase_Geometry_FromConst( CanonicalState* state, unsigned int offset )
{
    GeometryInstanceHandle gi = getGeometryInstanceHandle();
    return const_ObjectRecord + ( GeometryInstance_getGeometry( state, gi ) + offset );
}

extern "C" __constant__ char* Megakernel_getObjectBase_Material_FromConst( CanonicalState* state, unsigned int offset )
{
    MaterialHandle material = getMaterialHandle();
    return const_ObjectRecord + ( material + offset );
}

extern "C" __constant__ char* Megakernel_getObjectBase_Program_FromConst( CanonicalState* state, unsigned int offset, optix::SemanticType callType )
{
    return const_ObjectRecord + ( ActiveProgram_get( state, callType ) + offset );
}

extern "C" char* Megakernel_getObjectRecordFromConst( CanonicalState* state, unsigned int offset )
{
    return cort_castConstToGeneric( const_ObjectRecord ) + offset;
}

///////////////////////////////////////////////////////////
//
// Geometry instance.
//

GeometryHandle cort::GeometryInstance_getGeometry( CanonicalState* state, GeometryInstanceHandle gi )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return gir->geometry;
}

unsigned int cort::GeometryInstance_getNumMaterials( CanonicalState* state, GeometryInstanceHandle gi )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return gir->numMaterials;
}

LexicalScopeHandle cort::GeometryInstance_colwertToLexicalScopeHandle( CanonicalState* state, GeometryInstanceHandle gi )
{
    return gi;
}


///////////////////////////////////////////////////////////
//
// Geometry.
//

unsigned int cort::Geometry_getPrimitiveIndexOffset( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->indexOffset;
}

ProgramHandle cort::Geometry_getIntersectProgram( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->intersectOrAttribute;
}

ProgramHandle cort::Geometry_getAABBProgram( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->aabb;
}

LexicalScopeHandle cort::Geometry_colwertToLexicalScopeHandle( CanonicalState* state, GeometryHandle geometry )
{
    return geometry;
}

LexicalScopeHandle cort::Geometry_getAttributeKind( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->attributeKind;
}

ProgramHandle cort::Geometry_getAttributeProgram( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->intersectOrAttribute;
}

///////////////////////////////////////////////////////////
//
// Geometry triangles.
//

CORT_OVERRIDABLE
int cort::GeometryTriangles_getVertexBufferId( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->vertexBufferID;
}

CORT_OVERRIDABLE
long long cort::GeometryTriangles_getVertexBufferOffset( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->vertexBufferOffset;
}

CORT_OVERRIDABLE
unsigned long long cort::GeometryTriangles_getVertexBufferStride( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->vertexBufferStride;
}

CORT_OVERRIDABLE
int cort::GeometryTriangles_getIndexBufferId( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->indexBufferID;
}

CORT_OVERRIDABLE
long long cort::GeometryTriangles_getIndexBufferOffset( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->indexBufferOffset;
}

CORT_OVERRIDABLE
unsigned long long cort::GeometryTriangles_getIndexBufferStride( CanonicalState* state, GeometryTrianglesHandle geometryTriangles )
{
    GeometryTrianglesRecord* gtr = Global_getObjectRecord<GeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->indexBufferStride;
}

///////////////////////////////////////////////////////////
//
// Motion geometry triangles.
//

CORT_OVERRIDABLE
unsigned long long cort::MotionGeometryTriangles_getVertexBufferMotionStride( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->vertexBufferMotionStride;
}

CORT_OVERRIDABLE
int cort::MotionGeometryTriangles_getMotionNumIntervals( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->motionNumIntervals;
}

CORT_OVERRIDABLE
float cort::MotionGeometryTriangles_getTimeBegin( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->timeBegin;
}

CORT_OVERRIDABLE
float cort::MotionGeometryTriangles_getTimeEnd( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->timeEnd;
}

CORT_OVERRIDABLE
int cort::MotionGeometryTriangles_getMotionBorderModeBegin( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->motionBorderModeBegin;
}

CORT_OVERRIDABLE
int cort::MotionGeometryTriangles_getMotionBorderModeEnd( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles )
{
    MotionGeometryTrianglesRecord* gtr = Global_getObjectRecord<MotionGeometryTrianglesRecord>( state, geometryTriangles );
    return gtr->motionBorderModeEnd;
}


///////////////////////////////////////////////////////////
//
// Material.
//
extern "C" int RTX_getSBTRecordOffset();

ProgramHandle cort::Material_getAHProgram( CanonicalState* state, MaterialHandle material )
{
    const int       rayType = RTX_getSBTRecordOffset();
    MaterialRecord* mr      = Global_getObjectRecord<MaterialRecord>( state, material );
    return mr->programs[rayType].anyHit;
}

ProgramHandle cort::Material_getCHProgram( CanonicalState* state, MaterialHandle material )
{
    const int       rayType = RTX_getSBTRecordOffset();
    MaterialRecord* mr      = Global_getObjectRecord<MaterialRecord>( state, material );
    return mr->programs[rayType].closestHit;
}

LexicalScopeHandle cort::Material_colwertToLexicalScopeHandle( CanonicalState* state, MaterialHandle material )
{
    return material;
}


///////////////////////////////////////////////////////////
//
// Program
//
ProgramId cort::Program_getProgramID( CanonicalState* state, ProgramHandle program )
{
    ProgramRecord* pr = Global_getObjectRecord<ProgramRecord>( state, program );
    return pr->programID;
}


///////////////////////////////////////////////////////////
//
// Access to global state.
//

// BufferHeader
extern "C" Buffer Megakernel_getBufferHeaderFromConst( CanonicalState* state, unsigned int bufferid )
{
    __constant__ Buffer* bc = get_const_BufferHeaderTable() + bufferid;
    Buffer b = {{{bc->di.size.x, bc->di.size.y, bc->di.size.z}, {bc->di.pageSize.x, bc->di.pageSize.y, bc->di.pageSize.z}},
                {bc->dd.data, bc->dd.texUnit}};
    return b;
}

char* cort::Global_getObjectRecord( CanonicalState* state, unsigned int offset )
{
    return get_const_Global().objectRecords + offset;
}

FrameStatus* cort::Global_getStatusReturn( CanonicalState* state )
{
    return get_const_Global().statusReturn;
}

// NOTE: If the functions below are not marked as noinline, later replacement
// by "FromConst" functions has no effect on uses in this file, such as the one
// in Buffer_getElementAddress1dFromId_linear since they get inlined during
// offline compilation already.
__attribute__( ( noinline ) ) Buffer cort::Global_getBufferHeader( CanonicalState* state, unsigned int bufferid )
{
    return get_const_Global().bufferTable[bufferid];
}

__attribute__( ( noinline ) ) ProgramHeader cort::Global_getProgramHeader( CanonicalState* state, unsigned int programid )
{
    return get_const_Global().programTable[programid];
}

unsigned int cort::CallableProgram_getSBTBaseIndex( CanonicalState* state, unsigned int programid )
{
    ProgramHeader ph     = cort::Global_getProgramHeader( state, programid );
    unsigned int  sbtIdx = ph.di.programOffset;
    return sbtIdx;
}

__attribute__( ( noinline ) ) TextureSampler cort::Global_getTextureSamplerHeader( CanonicalState* state, unsigned int textureid )
{
    return get_const_Global().textureTable[textureid];
}

__attribute__( ( noinline ) ) TraversableHeader cort::Global_getTraversableHeader( CanonicalState* state, unsigned int traversableid )
{
    return get_const_Global().traversableTable[traversableid];
}


unsigned int cort::Global_getNumBuffers( CanonicalState* state )
{
    return get_const_Global().numBuffers;
}

unsigned int cort::Global_getNumTextures( CanonicalState* state )
{
    return get_const_Global().numTextures;
}

unsigned int cort::Global_getNumPrograms( CanonicalState* state )
{
    return get_const_Global().numPrograms;
}

unsigned int cort::Global_getNumTraversables( CanonicalState* state )
{
    return get_const_Global().numTraversables;
}

uint3 cort::Global_getLaunchDim( CanonicalState* state )
{
    return uint3( get_const_Global().launchDim.x, get_const_Global().launchDim.y, get_const_Global().launchDim.z );
}

unsigned int cort::Global_getSubframeIndex( CanonicalState* state )
{
    return get_const_Global().subframeIndex;
}

extern "C" uint2 RTX_getLaunchIndex();

static uint2 getTiledLaunchIndex( CanonicalState* state )
{
    const uint2          idx         = RTX_getLaunchIndex();
    const unsigned short deviceCount = Global_getDeviceCount( state );
    if( deviceCount == 1 )
        return idx;

    const unsigned short deviceIndex = Global_getDeviceIndex( state );
    // SHIFTX and SHIFTY are powers of 2 that determine the tile size. 6 and 3 for 64x8 (one warp).
    static const uint SHIFT_X           = 6;
    static const uint SHIFT_Y           = 3;
    static const uint TILE_SIZE_X       = 1 << SHIFT_X;
    static const uint MASK_X            = TILE_SIZE_X - 1;
    const uint        xTileIdx          = idx.x >> SHIFT_X;
    const uint        yTileIdx          = idx.y >> SHIFT_Y;
    const uint        stretchedXTileIdx = ( deviceIndex + yTileIdx ) % deviceCount + deviceCount * xTileIdx;
    const uint        xPixelOffset      = idx.x & MASK_X;
    const uint        newXIdx           = ( stretchedXTileIdx << SHIFT_X ) | xPixelOffset;
    return uint2( newXIdx, idx.y );
}

uint3 cort::Raygen_getLaunchIndex1dOr2d( CanonicalState* state )
{
    const uint2 idx = getTiledLaunchIndex( state );
    return uint3( idx.x, idx.y, 0 );
}

uint3 cort::Raygen_getLaunchIndex3d( CanonicalState* state )
{
    const uint2 idx = getTiledLaunchIndex( state );
    const uint  x0  = Global_getLaunchDim( state ).x;
    return uint3( idx.x % x0, idx.y, idx.x / x0 );
}

// Check if the launch index of the thread is outside of the specified values
// (only compiled in if launch.limitActiveIndices knob is set).
extern "C" bool RTX_indicesOutsideOfLimitedRange( CanonicalState* state )
{
    const uint3                idx     = Raygen_getLaunchIndex( state );
    __constant__ unsigned int* indices = get_const_MinMaxLaunchIndex();
    const bool outside = idx.x < indices[0] || idx.x > indices[1] || idx.y < indices[2] || idx.y > indices[3]
                         || idx.z < indices[4] || idx.z > indices[5];
    return outside;
}

AabbRequest cort::Global_getAabbRequest( CanonicalState* state )
{
    // LLVM has a hard time with copy ctors between multiple address
    // spaces, so split the struct out.
    __attribute__( ( address_space( 4 ) ) ) AabbRequest& tmp = get_const_Global().aabbRequest;
    return AabbRequest( tmp.isGroup, tmp.recordOffset, tmp.buildMotionSteps, tmp.geometryMotionSteps, tmp.computeUnion,
                        tmp.aabbOutputPointer, tmp.motionAabbRequests );
}

uint3 cort::Global_getPrintIndex( CanonicalState* state )
{
    return uint3( get_const_Global().printIndex.x, get_const_Global().printIndex.y, get_const_Global().printIndex.z );
}

bool cort::Global_getPrintEnabled( CanonicalState* state )
{
    return get_const_Global().printEnabled;
}

unsigned short cort::Global_getDimensionality( CanonicalState* state )
{
    return get_const_Global().dimensionality;
}

unsigned short cort::Global_getEntry( CanonicalState* state )
{
    return get_const_Global().entry;
}

unsigned int cort::Global_getRayTypeCount( CanonicalState* state )
{
    return get_const_Global().rayTypeCount;
}

unsigned short cort::Global_getDeviceIndex( CanonicalState* state )
{
    return get_const_Global().activeDeviceIndex;
}

CORT_OVERRIDABLE unsigned short cort::Global_getDeviceCount( CanonicalState* state )
{
    return get_const_Global().activeDeviceCount;
}

CORT_OVERRIDABLE PagingMode cort::Global_getDemandLoadMode( CanonicalState* state )
{
    return get_const_Global().demandLoadMode;
}

// Taken verbatum from CommonRuntime.cpp
bool cort::Raygen_isPrintEnabledAtLwrrentLaunchIndex( CanonicalState* state )
{
    if( !Global_getPrintEnabled( state ) )
        return false;

    const uint3& pli = Global_getPrintIndex( state );

    const int  PRINT_INDEX_ALL_ENABLED = -1;  // Default value in PrintManager.cpp
    const bool allX                    = pli.x == PRINT_INDEX_ALL_ENABLED;
    const bool allY                    = pli.y == PRINT_INDEX_ALL_ENABLED;
    const bool allZ                    = pli.z == PRINT_INDEX_ALL_ENABLED;

    if( allX && allY && allZ )
        return true;

    const uint3& rli = Raygen_getLaunchIndex( state );

    if( !allX && pli.x != rli.x )
        return false;

    if( !allY && pli.y != rli.y )
        return false;

    if( !allZ && pli.z != rli.z )
        return false;

    return true;
}

// ProgramHeader
extern "C" ProgramHeader Megakernel_getProgramHeaderFromConst( CanonicalState* state, unsigned int programid )
{
    __constant__ ProgramHeader* pc = get_const_ProgramHeaderTable() + programid;
    ProgramHeader               p  = {{pc->di.programOffset}, {pc->dd.canonicalProgramID}};
    return p;
}

// TextureHeader
extern "C" TextureSampler Megakernel_getTextureHeaderFromConst( CanonicalState* state, unsigned int textureid )
{
    __constant__ TextureSampler* tc = get_const_TextureHeaderTable() + textureid;

    TextureSampler t = {tc->width,
                        tc->height,
                        tc->depth,
                        tc->mipLevels,
                        tc->mipTailFirstLevel,
                        tc->ilwAnisotropy,
                        tc->format,
                        tc->wrapMode0,
                        tc->wrapMode1,
                        tc->wrapMode2,
                        tc->normCoord,
                        tc->filterMode,
                        tc->normRet,
                        tc->isDemandLoad,
                        tc->tileWidth,
                        tc->tileHeight,
                        tc->tileGutterWidth,
                        tc->isInitialized,
                        tc->isSquarePowerOfTwo,
                        tc->mipmapFilterMode,
                        tc->padding,
                        tc->padding2,
                        {tc->dd.texref, tc->dd.swptr, tc->dd.minMipLevel, tc->dd.padding[0], tc->dd.padding[1],
                         tc->dd.padding[2], tc->dd.padding[3], tc->dd.padding[4], tc->dd.padding[5], tc->dd.padding[6]}};
    return t;
}

///////////////////////////////////////////////////////////
//
// DemandLoadBuffer
//
extern "C" uint RTX_atomicOr( uint* address, uint val );

inline void atomicSetBit( uint bitIndex, uint* bitVector )
{
    const uint wordIndex = bitIndex >> 5;
    const uint bitOffset = bitIndex % 32;
    const uint mask      = 1U << bitOffset;
    RTX_atomicOr( bitVector + wordIndex, mask );
}

inline bool checkBitSet( uint bitIndex, const uint* bitVector )
{
    const uint wordIndex = bitIndex >> 5;
    const uint bitOffset = bitIndex % 32;
    return ( bitVector[wordIndex] & ( 1U << bitOffset ) ) != 0;
}

inline uint64 optixPagingMapOrRequest( uint* usageBits, uint* residenceBits, uint64* pageTable, uint page, bool* valid )
{
    bool requested = checkBitSet( page, usageBits );
    if( !requested )
    {
        atomicSetBit( page, usageBits );
    }

    bool mapped = checkBitSet( page, residenceBits );
    *valid      = mapped;

    return mapped ? pageTable[page] : 0;
}

inline char* requestPage( uint requestedPage )
{
    uint*        usageBits     = get_const_Global().pageUsageBits;
    uint*        residenceBits = get_const_Global().pageResidenceBits;
    uint64*      pageTable     = get_const_Global().pageTable;
    bool         valid         = false;
    const uint64 result        = optixPagingMapOrRequest( usageBits, residenceBits, pageTable, requestedPage, &valid );
    return valid ? reinterpret_cast<char*>( result ) : nullptr;
}

extern "C" char* RTX_requestBufferElement1( cort::CanonicalState* state, unsigned int bufferId, unsigned int elementSize, cort::uint64 x )
{
    Buffer     buffer        = Global_getBufferHeader( state, bufferId );
    const uint pageSize      = buffer.di.pageSize.x;
    const uint pageIndex     = static_cast<uint>( x ) / pageSize;
    const uint firstPage     = reinterpret_cast<unsigned long long>( buffer.dd.data );
    const uint requestedPage = firstPage + pageIndex;
    char*      result        = requestPage( requestedPage );
    if( !result )
        return nullptr;

    return result + ( x % pageSize ) * elementSize;
}

extern "C" char* RTX_requestBufferElement2( cort::CanonicalState* state,
                                            unsigned int          bufferId,
                                            unsigned int          elementSize,
                                            cort::uint64          x,
                                            cort::uint64          y )
{
    Buffer     buffer     = Global_getBufferHeader( state, bufferId );
    const uint pageWidth  = buffer.di.pageSize.x;
    const uint pageHeight = buffer.di.pageSize.y;
    const uint pageX      = static_cast<uint>( x ) / pageWidth;
    const uint pageY      = static_cast<uint>( y ) / pageHeight;

    const uint firstPage     = reinterpret_cast<unsigned long long>( buffer.dd.data );
    const uint widthInPages  = buffer.di.size.x / pageWidth;
    const uint requestedPage = firstPage + pageY * widthInPages + pageX;
    char*      result        = requestPage( requestedPage );
    if( !result )
        return nullptr;

    const uint xOffset = x % pageWidth;
    const uint yOffset = y % pageHeight;
    return result + ( yOffset * pageWidth + xOffset ) * elementSize;
}

extern "C" char* RTX_requestBufferElement3( cort::CanonicalState* state,
                                            unsigned int          bufferId,
                                            unsigned int          elementSize,
                                            cort::uint64          x,
                                            cort::uint64          y,
                                            cort::uint64          z )
{
    Buffer     buffer     = Global_getBufferHeader( state, bufferId );
    const uint pageWidth  = buffer.di.pageSize.x;
    const uint pageHeight = buffer.di.pageSize.y;
    const uint pageDepth  = buffer.di.pageSize.z;
    const uint pageX      = static_cast<uint>( x ) / pageWidth;
    const uint pageY      = static_cast<uint>( y ) / pageHeight;
    const uint pageZ      = static_cast<uint>( z ) / pageDepth;

    const uint firstPage     = reinterpret_cast<unsigned long long>( buffer.dd.data );
    const uint widthInPages  = buffer.di.size.x / pageWidth;
    const uint heightInPages = buffer.di.size.y / pageHeight;
    const uint requestedPage = firstPage + pageZ * widthInPages * heightInPages + pageY * widthInPages + pageX;
    char*      result        = requestPage( firstPage + requestedPage );
    if( !result )
        return nullptr;

    const uint xOffset          = x % pageWidth;
    const uint yOffset          = y % pageHeight;
    const uint zOffset          = z % pageDepth;
    const uint offsetWithinPage = ( zOffset * pageWidth * pageHeight + yOffset * pageWidth + xOffset ) * elementSize;
    return reinterpret_cast<char*>( result + offsetWithinPage );
}

///////////////////////////////////////////////////////////
//
// Demand textures
//

inline int clampi( int f, int a, int b )
{
    return lwca::maxi( a, lwca::mini( f, b ) );
}

inline void requestMiplevel( const TextureSampler& sampler, uint miplevel, bool* isResident )
{
    if( !sampler.isDemandLoad )
    {
        *isResident = true;
        return;
    }
    const uint firstPage     = getDemandTextureFirstPage( sampler.dd );
    const uint requestedPage = firstPage + miplevel;

    uint*   usageBits     = get_const_Global().pageUsageBits;
    uint*   residenceBits = get_const_Global().pageResidenceBits;
    uint64* pageTable     = get_const_Global().pageTable;

    optixPagingMapOrRequest( usageBits, residenceBits, pageTable, requestedPage, isResident );
}

inline void requestLod( const TextureSampler& sampler, float level, bool* isResident )
{
    if( !sampler.isDemandLoad )
    {
        *isResident = true;
        return;
    }

    // The software callwlation of the MIP level is not exactly the same as the hardware,
    // so conservatively load extra MIP levels
    const float MIP_REQUEST_OFFSET = 0.2f;

    const int  coarsestMiplevel = sampler.mipLevels - 1;
    const uint lowerLevel       = clampi( static_cast<uint>( level - MIP_REQUEST_OFFSET ), 0, coarsestMiplevel );
    const uint upperLevel       = clampi( static_cast<uint>( level + MIP_REQUEST_OFFSET + 1 ), 0, coarsestMiplevel );

    requestMiplevel( sampler, lowerLevel, isResident );
    if( ( lowerLevel + 1 ) < upperLevel )
    {
        bool isResident2 = true;
        requestMiplevel( sampler, lowerLevel + 1, &isResident2 );
        *isResident = *isResident && isResident2;
    }
    if( upperLevel != lowerLevel )
    {
        bool isResident3 = true;
        requestMiplevel( sampler, upperLevel, &isResident3 );
        *isResident = *isResident && isResident3;
    }
}

inline void requestGrad( const TextureSampler& sampler, float filterWidth, bool* isResident )
{
    // Callwlate the desired miplevel, which is where the filter radius (i.e. half the width)
    // times the miplevel dimension is one texel.  For example, if the filter radius is .01 the
    // ideal miplevel would be 100x100 texels.  If the texture dimensions are 2^k x 2^k, then
    // radius * 2^k == 1 exactly when k == log2(1/radius) == -log2(radius).  (Using the width
    // instead of the radius shifts the result by one.)  For example, if the filter radius is
    // .01 we will interpolate between miplevels 3 and 4 of a 1024x1024 texture, which are
    // 128x128 and 64x64.
    float miplevel = sampler.mipLevels + lwca::log2( filterWidth ) - 1;

    requestLod( sampler, miplevel, isResident );
}

inline void requestGrad1( const TextureSampler& sampler, float dPdx, float dPdy, bool* isResident )
{
    if( !sampler.isDemandLoad )
    {
        *isResident = true;
        return;
    }
    float filterWidth = lwca::maxf( lwca::fabs( dPdx ), lwca::fabs( dPdy ) );
    requestGrad( sampler, filterWidth, isResident );
}

inline void requestGrad2( const TextureSampler& sampler, float dPdx_x, float dPdx_y, float dPdy_x, float dPdy_y, bool* isResident )
{
    if( !sampler.isDemandLoad )
    {
        *isResident = true;
        return;
    }

    float xWidth = sqrt( dPdx_x * dPdx_x + dPdx_y * dPdx_y );
    float yWidth = sqrt( dPdy_x * dPdy_x + dPdy_y * dPdy_y );
    float filterWidth;

    if( xWidth < yWidth )
        filterWidth = lwca::maxf( xWidth, yWidth * ILW_ANISOTROPY );
    else
        filterWidth = lwca::maxf( yWidth, xWidth * ILW_ANISOTROPY );

    requestGrad( sampler, filterWidth, isResident );
}

inline void requestGrad3( const TextureSampler& sampler, float dPdx_x, float dPdx_y, float dPdx_z, float dPdy_x, float dPdy_y, float dPdy_z, bool* isResident )
{
    if( !sampler.isDemandLoad )
    {
        *isResident = true;
        return;
    }
    float xWidth = sqrt( dPdx_x * dPdx_x + dPdx_y * dPdx_y + dPdx_z * dPdx_z );
    float yWidth = sqrt( dPdy_x * dPdy_x + dPdy_y * dPdy_y + dPdy_z * dPdy_z );
    float filterWidth;

    if( xWidth < yWidth )
        filterWidth = lwca::maxf( xWidth, yWidth * ILW_ANISOTROPY );
    else
        filterWidth = lwca::maxf( yWidth, xWidth * ILW_ANISOTROPY );

    requestGrad( sampler, filterWidth, isResident );
}

extern "C" cort::float4 RTX_textureLoadOrRequest1( cort::CanonicalState* state, uint textureId, float x, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestMiplevel( sampler, 0, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_tex_1d( texref, x );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureLoadOrRequest2( cort::CanonicalState* state, uint textureId, float x, float y, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestMiplevel( sampler, 0, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_tex_2d( texref, x, y );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureLoadOrRequest3( cort::CanonicalState* state, uint textureId, float x, float y, float z, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestMiplevel( sampler, 0, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_tex_3d( texref, x, y, z );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureLodLoadOrRequest1( cort::CanonicalState* state, uint textureId, float x, float level, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestLod( sampler, level, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        // The actual level is the nominal level minus the minimum loaded miplevel.
        unsigned int minMipLevel = sampler.dd.minMipLevel;
        return Texture_getElement_hw_texlevel_1d( texref, x, level - minMipLevel );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

cort::float4 wholeMipLevelLodLoadOrRequest2( cort::CanonicalState* state, uint textureId, float x, float y, float level, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestLod( sampler, level, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        // The actual level is the nominal level minus the minimum loaded miplevel.
        unsigned int minMipLevel = sampler.dd.minMipLevel;
        return Texture_getElement_hw_texlevel_2d( texref, x, y, level - minMipLevel );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureLodLoadOrRequest3( cort::CanonicalState* state, uint textureId, float x, float y, float z, float level, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestLod( sampler, level, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        // The actual level is the nominal level minus the minimum loaded miplevel.
        unsigned int minMipLevel = sampler.dd.minMipLevel;
        return Texture_getElement_hw_texlevel_3d( texref, x, y, z, level - minMipLevel );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureGradLoadOrRequest1( cort::CanonicalState* state, uint textureId, float x, float dPdx, float dPdy, uint64 isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestGrad1( sampler, dPdx, dPdy, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_texgrad_1d( texref, x, dPdx, dPdy );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

cort::float4 wholeMipLevelGradLoadOrRequest2( cort::CanonicalState* state,
                                              uint                  textureId,
                                              float                 x,
                                              float                 y,
                                              float                 dPdx_x,
                                              float                 dPdx_y,
                                              float                 dPdy_x,
                                              float                 dPdy_y,
                                              uint64                isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestGrad2( sampler, dPdx_x, dPdx_y, dPdy_x, dPdy_y, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_texgrad_2d( texref, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}

extern "C" cort::float4 RTX_textureGradLoadOrRequest3( cort::CanonicalState* state,
                                                       uint                  textureId,
                                                       float                 x,
                                                       float                 y,
                                                       float                 z,
                                                       float                 dPdx_x,
                                                       float                 dPdx_y,
                                                       float                 dPdx_z,
                                                       float                 dPdy_x,
                                                       float                 dPdy_y,
                                                       float                 dPdy_z,
                                                       uint64                isResidentPtr )
{
    bool*          isResident = reinterpret_cast<bool*>( isResidentPtr );
    TextureSampler sampler    = Global_getTextureSamplerHeader( state, textureId );
    requestGrad3( sampler, dPdx_x, dPdx_y, dPdx_z, dPdy_x, dPdy_y, dPdy_z, isResident );
    if( *isResident )
    {
        uint64 texref = sampler.dd.texref;
        return Texture_getElement_hw_texgrad_3d( texref, x, y, z, dPdx_x, dPdx_y, dPdx_z, dPdy_x, dPdy_y, dPdy_z );
    }
    return cort::float4( 0.f, 0.f, 0.f, 0.f );
}


// Defined as alternate to float2 in CORTTypes.h, to prevent compilation error under Windows.
struct flt2
{
    float x, y;

    flt2( float x, float y )
        : x( x )
        , y( y )
    {
    }
    flt2( const flt2& copy )
        : x( copy.x )
        , y( copy.y )
    {
    }
    flt2  operator-() { return flt2( -x, -y ); }
    flt2& operator=( const flt2& copy )
    {
        x = copy.x;
        y = copy.y;
        return *this;
    }
    flt2 operator+( const flt2& b ) const { return flt2( x + b.x, y + b.y ); }
    flt2 operator-( const flt2& b ) const { return flt2( x - b.x, y - b.y ); }
    flt2 operator*( const flt2& b ) const { return flt2( x * b.x, y * b.y ); }
    flt2& operator*=( float b )
    {
        x *= b;
        y *= b;
        return *this;
    }
    flt2& operator*=( const flt2& b )
    {
        x *= b.x;
        y *= b.y;
        return *this;
    }
    float length() { return sqrt( x * x + y * y ); }
    ~flt2(){};
    static flt2 min( const flt2& a, const flt2& b ) { return flt2( lwca::minf( a.x, b.x ), lwca::minf( a.y, b.y ) ); }
    static flt2 max( const flt2& a, const flt2& b ) { return flt2( lwca::maxf( a.x, b.x ), lwca::maxf( a.y, b.y ) ); }
};

inline float getMipLevelFromTextureGradients( flt2 ddx, flt2 ddy, int textureWidth, int textureHeight, float ilwAnisotropy )
{
    flt2 scalingFactor = {(float)textureWidth, (float)textureHeight};
    ddx *= scalingFactor;
    ddy *= scalingFactor;

    float A    = ddy.x * ddy.x + ddy.y * ddy.y;
    float B    = -2.0f * ( ddx.x * ddy.x + ddx.y * ddy.y );
    float C    = ddx.x * ddx.x + ddx.y * ddx.y;
    float root = sqrt( lwca::maxf( A * A - 2.0f * A * C + C * C + B * B, 0.0f ) );

    float minorRadius2 = ( A + C - root ) * 0.5f;
    float majorRadius2 = ( A + C + root ) * 0.5f;
    float filterWidth2 = lwca::maxf( minorRadius2, majorRadius2 * ilwAnisotropy * ilwAnisotropy );

    return lwca::log2( filterWidth2 ) * 0.5f;
}

inline float clampf( float f, float a, float b )
{
    return lwca::maxf( a, lwca::minf( f, b ) );
}

inline uint2 getTexelSizesFromGranularity( unsigned int granularity )
{
    uint2 result;
    // clang-format off
    switch( granularity )
    {
        case 0: result.x = 0; result.y = 0;
                // We shouldn't get here; a granularity of 0 means it accepted the
                // one we suggested; we should pass the suggested value to this
                // function in that case.
                break;
        case 1: result.x = 2; result.y = 2;
                break;
        case 2: result.x = 4; result.y = 2;
                break;
        case 3: result.x = 4; result.y = 4;
                break;
        case 4: result.x = 8; result.y = 4;
                break;
        case 5: result.x = 8; result.y = 8;
                break;
        case 6: result.x = 16; result.y = 8;
                break;
        case 7: result.x = 16; result.y = 16;
                break;
        case 8: result.x = 0; result.y = 0; // Not valid
                break;
        case 9: result.x = 0; result.y = 0; // Not valid
                break;
        case 10: result.x = 0; result.y = 0; // Not valid
                 break;
        case 11: result.x = 64; result.y = 64;
                 break;
        case 12: result.x = 128; result.y = 64;
                 break;
        case 13: result.x = 128; result.y = 128;
                 break;
        case 14: result.x = 256; result.y = 128;
                 break;
        case 15: result.x = 256; result.y = 256;
                 break;
        default: result.x = 0; result.y = 0; // Not valid
    }
    // clang-format on
    return result;
}

typedef struct lwdaTexture2DFootprint
{
    unsigned int  tileX;
    unsigned int  tileY;
    unsigned int  dx;
    unsigned int  dy;
    unsigned long mask;
    unsigned int  level;
    unsigned int  granularity;
} lwdaTexture2DFootprint;

inline lwdaTexture2DFootprint unpackFootprintResult( uint4 result, unsigned int requestedGranularity )
{
    // Result packing documentation can be found here: https://p4viewer.lwpu.com/get///hw/doc/gpu/turing/turing/design/IAS/SM/ISA/opcodes/opFOOTPRINT.htm

    // Result packing:
    // result.x        = mask[31:0]
    // result.y        = mask[63:32]
    // result.z[11:0]  = tileY
    // result.z[16:18] = dx
    // result.z[19:21] = dy
    // result.z[24:27] = granularity
    // result.w[11:0]  = tileX
    // result.w[12:15] = level

    lwdaTexture2DFootprint footprint;
    footprint.tileX       = ( result.w & 0xFFF );
    footprint.tileY       = ( result.z & 0xFFF );
    footprint.dx          = ( result.z & 0x070000 ) >> 16;
    footprint.dy          = ( result.z & 0x380000 ) >> 19;
    footprint.mask        = ( (unsigned long)result.y << 32 ) | result.x;
    footprint.level       = ( result.w & 0xF000 ) >> 12;

    // If the returned granularity is 0, the instruction accepted our requested
    // granularity, and we should use that instead.
    footprint.granularity = ( result.z & 0x0F000000 ) >> 24;
    if( footprint.granularity == 0 )
        footprint.granularity = requestedGranularity;

    return footprint;
}

inline void getTexelCoordsFromFootprint( lwdaTexture2DFootprint footprint, uint2 texelSizes, unsigned int* xCoords, unsigned int* yCoords, unsigned int* outNumTexels )
{
    // Based on example in documentation at https://p4viewer.lwpu.com/get///hw/doc/gpu/turing/turing/design/IAS/SM/ISA/opcodes/opFOOTPRINT.htm

    // Form initial masks
    unsigned long xMask = ( 0xff >> ( footprint.dx & 7 ) ) * 0x0101010101010101ull;  // Replicates byte 8 times to fill out a 64-bit word
    unsigned long yMask = 0xffffffffffffffffull >> ( footprint.dy << 3 );

    // If the anchor tile's phase isn't 0, initial masks have to be ilwerted
    if( footprint.tileX & 1 )
        xMask = ~xMask;
    if( footprint.tileY & 1 )
        yMask = ~yMask;

    // Callwlate x/y steps given anchor tile's phase
    int xStep = ( footprint.tileX & 1 ) ? 1 : -1;
    int yStep = ( footprint.tileY & 1 ) ? 1 : -1;

    // Get the bitmasks and positions of each tile
    unsigned long maskOfTile0 = footprint.mask & xMask & yMask;
    unsigned long maskOfTile1 = footprint.mask & ~xMask & yMask;
    unsigned long maskOfTile2 = footprint.mask & ~xMask & ~yMask;
    unsigned long maskOfTile3 = footprint.mask & xMask & ~yMask;

    unsigned int tile0X = footprint.tileX & ~1;
    unsigned int tile0Y = footprint.tileY & ~1;

    unsigned int tile1X = tile0X + xStep;
    unsigned int tile1Y = tile0Y;

    unsigned int tile2X = tile0X + xStep;
    unsigned int tile2Y = tile0Y + yStep;

    unsigned int tile3X = tile0X;
    unsigned int tile3Y = tile0Y + yStep;

    unsigned int tileGroupsLog2 = 3;

    unsigned int numTexels = 0;

    unsigned int texel_x0 = ( ( tile0X << tileGroupsLog2 ) - footprint.dx ) * texelSizes.x;
    unsigned int texel_y0 = ( ( tile0Y << tileGroupsLog2 ) - footprint.dy ) * texelSizes.y;

    for( unsigned int i = 0; i < 64; ++i )
    {
        if( ( ( maskOfTile0 >> i ) & 1 ) == 1 )
        {
            unsigned int xCoord = texel_x0 + ( i % 8 ) * texelSizes.x + ( footprint.dx * texelSizes.x );
            unsigned int yCoord = texel_y0 + ( i / 8 ) * texelSizes.y + ( footprint.dy * texelSizes.y );
            xCoords[numTexels]  = xCoord;
            yCoords[numTexels]  = yCoord;
            numTexels++;
        }
    }

    unsigned int texel_x1 = ( ( tile1X << tileGroupsLog2 ) - footprint.dx ) * texelSizes.x;
    unsigned int texel_y1 = ( ( tile1Y << tileGroupsLog2 ) - footprint.dy ) * texelSizes.y;

    for( unsigned int i = 0; i < 64; ++i )
    {
        if( ( ( maskOfTile1 >> i ) & 1 ) == 1 )
        {
            unsigned int xCoord = texel_x1 + ( i % 8 ) * texelSizes.x + ( footprint.dx * texelSizes.x );
            unsigned int yCoord = texel_y1 + ( i / 8 ) * texelSizes.y + ( footprint.dy * texelSizes.y );
            xCoords[numTexels]  = xCoord;
            yCoords[numTexels]  = yCoord;
            numTexels++;
        }
    }

    unsigned int texel_x2 = ( ( tile2X << tileGroupsLog2 ) - footprint.dx ) * texelSizes.x;
    unsigned int texel_y2 = ( ( tile2Y << tileGroupsLog2 ) - footprint.dy ) * texelSizes.y;

    for( unsigned int i = 0; i < 64; ++i )
    {
        if( ( ( maskOfTile2 >> i ) & 1 ) == 1 )
        {
            unsigned int xCoord = texel_x2 + ( i % 8 ) * texelSizes.x + ( footprint.dx * texelSizes.x );
            unsigned int yCoord = texel_y2 + ( i / 8 ) * texelSizes.y + ( footprint.dy * texelSizes.y );
            xCoords[numTexels]  = xCoord;
            yCoords[numTexels]  = yCoord;
            numTexels++;
        }
    }

    unsigned int texel_x3 = ( ( tile3X << tileGroupsLog2 ) - footprint.dx ) * texelSizes.x;
    unsigned int texel_y3 = ( ( tile3Y << tileGroupsLog2 ) - footprint.dy ) * texelSizes.y;
    for( unsigned int i = 0; i < 64; ++i )
    {
        if( ( ( maskOfTile3 >> i ) & 1 ) == 1 )
        {
            unsigned int xCoord = texel_x3 + ( i % 8 ) * texelSizes.x + ( footprint.dx * texelSizes.x );
            unsigned int yCoord = texel_y3 + ( i / 8 ) * texelSizes.y + ( footprint.dy * texelSizes.y );
            xCoords[numTexels]  = xCoord;
            yCoords[numTexels]  = yCoord;
            numTexels++;
        }
    }

    *outNumTexels = numTexels;
}

inline unsigned int getGranularityForTileSize( unsigned int tileWidth, unsigned int tileHeight )
{
    if( tileWidth == 64 && tileHeight == 64 )
        return 11;
    else if( tileWidth == 128 && tileHeight == 64 )
        return 12;
    else if( tileWidth == 128 && tileHeight == 128 )
        return 13;
    else
        return 1;
}

inline cort::float4 textureGradLoadOrRequest2_lwdaSparse_HW( const TextureSampler& sampler, float x, float y, flt2 ddx, flt2 ddy, uint64 isResidentPtr )
{
    int    isResident32 = 0;
    float4 result =
        Texture_getElement_hw_texgrad_2d_isResident( sampler.dd.texref, x, y, ddx.x, ddx.y, ddy.x, ddy.y, &isResident32 );

    bool* isResident = reinterpret_cast<bool*>( isResidentPtr );
    if( !isResident32 )
    {
        *isResident = true;
        return result;
    }

    *isResident = false;

    // The footprint instruction assumes CLAMP_TO_EDGE wrapping, so we must
    // wrap the coordinates ourselves before calling it.
    x = optix::TileIndexing::wrapNormCoord( x, static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ) );
    y = optix::TileIndexing::wrapNormCoord( y, static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ) );

    int       coversSingleMipLevel = 0;
    const int granularityToRequest = getGranularityForTileSize( sampler.tileWidth, sampler.tileHeight );
    uint4 fineFootprintData = Texture_getElement_hw_texgrad_footprint_2d( granularityToRequest, sampler.dd.texref, x, y,
                                                                          ddx.x, ddx.y, ddy.x, ddy.y, &coversSingleMipLevel );
    lwdaTexture2DFootprint fineFootprint  = unpackFootprintResult( fineFootprintData, granularityToRequest );
    uint2                  fineTexelSizes = getTexelSizesFromGranularity( fineFootprint.granularity );

    const int MAX_TEXELS_COVERED = 256;  // Footprint instruction can return at most 4 tiles each containing 8 x 8 texels.
    unsigned int fineTexelXCoords[MAX_TEXELS_COVERED];
    unsigned int fineTexelYCoords[MAX_TEXELS_COVERED];
    unsigned int fineNumTexelsCovered = 0;
    getTexelCoordsFromFootprint( fineFootprint, fineTexelSizes, fineTexelXCoords, fineTexelYCoords, &fineNumTexelsCovered );

    optix::TileIndexing tileIndexing( sampler.width, sampler.height, sampler.tileWidth, sampler.tileHeight );

    unsigned int tilesToRequest[2 * optix::TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest = 0;

    tileIndexing.callwlateTileRequestsFromTexels(
        fineFootprint.level, fineTexelXCoords, fineTexelYCoords, fineNumTexelsCovered, fineTexelSizes.x,
        fineTexelSizes.y, sampler.mipTailFirstLevel, static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
        static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ), tilesToRequest, numTilesToRequest );

    if( !coversSingleMipLevel )
    {
        int   coversSingleMipLevel = 0;  // Unused; we already know we're using multiple mip levels
        uint4 coarseFootprintData =
            Texture_getElement_hw_texgrad_footprint_coarse_2d( granularityToRequest, sampler.dd.texref, x, y, ddx.x,
                                                               ddx.y, ddy.x, ddy.y, &coversSingleMipLevel );
        lwdaTexture2DFootprint coarseFootprint  = unpackFootprintResult( coarseFootprintData, granularityToRequest );
        uint2                  coarseTexelSizes = getTexelSizesFromGranularity( coarseFootprint.granularity );

        unsigned int coarseTexelXCoords[MAX_TEXELS_COVERED];
        unsigned int coarseTexelYCoords[MAX_TEXELS_COVERED];
        unsigned int coarseNumTexelsCovered = 0;
        getTexelCoordsFromFootprint( coarseFootprint, coarseTexelSizes, coarseTexelXCoords, coarseTexelYCoords, &coarseNumTexelsCovered );


        unsigned int  numTilesToRequestInCoarseLevel = 0;
        unsigned int* endOfPreviousRequests          = tilesToRequest + numTilesToRequest;

        tileIndexing.callwlateTileRequestsFromTexels( coarseFootprint.level, coarseTexelXCoords, coarseTexelYCoords, coarseNumTexelsCovered,
                                                      coarseTexelSizes.x, coarseTexelSizes.y, sampler.mipTailFirstLevel,
                                                      static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                                      static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ),
                                                      endOfPreviousRequests, numTilesToRequestInCoarseLevel );
        numTilesToRequest += numTilesToRequestInCoarseLevel;
    }

    uint*   usageBits     = get_const_Global().pageUsageBits;
    uint*   residenceBits = get_const_Global().pageResidenceBits;
    uint64* pageTable     = get_const_Global().pageTable;

    // Request tiles
    uint startPage    = getDemandTextureFirstPage( sampler.dd );
    bool swIsResident = true;  // Unused; we determined the tile isn't resident via the initial texture fetch.
    for( unsigned int i = 0; i < numTilesToRequest; ++i )
    {
        unsigned int pageId = startPage + tilesToRequest[i];
        optixPagingMapOrRequest( usageBits, residenceBits, pageTable, pageId, &swIsResident );
    }

    return float4( 1.0f, 0.0f, 1.0f, 0.0f );
}

inline cort::float4 textureGradLoadOrRequest2_lwdaSparse_hybrid( const TextureSampler& sampler, float x, float y, flt2 ddx, flt2 ddy, uint64 isResidentPtr )
{
    float mipLevel = getMipLevelFromTextureGradients( ddx, ddy, sampler.width, sampler.height, sampler.ilwAnisotropy );
    // Note that if the texture is not mipmapped the mipLevel will be clamped to 0
    mipLevel = clampf( mipLevel, 0.0f, sampler.mipLevels - 1.0f );

    // Snap to the nearest mip level if we're doing point filtering
    if( sampler.mipmapFilterMode == lwca::lwdaFilterModePoint )
        mipLevel = lwca::floorf( mipLevel + 0.5f );

    // The desired miplevel is a float.  Request tiles from the coarser adjacent miplevel.  Note
    // that anisotropic filtering might require several tiles per miplevel, and we might request
    // tiles from two miplevels (hence the factor of two on the array size).
    unsigned int coarseLevel = static_cast<unsigned int>( lwca::ceilf( mipLevel ) );
    unsigned int tilesToRequest[2 * optix::TileIndexing::MAX_TILES_CALLWLATED];
    unsigned int numTilesToRequest = 0;

    optix::TileIndexing tileIndexing( sampler.width, sampler.height, sampler.tileWidth, sampler.tileHeight );
    tileIndexing.callwlateTileRequests( coarseLevel, x, y, sampler.mipTailFirstLevel,
                                        static_cast<unsigned int>( 1.f / sampler.ilwAnisotropy ),
                                        static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                        static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ), tilesToRequest,
                                        numTilesToRequest );

    // Request tiles from the finer adjacent miplevel, unless the desired miplevel is integral.
    // Note that the miplevel will be 0.0 if the texture is non-mipmapped, so we don't need
    // a special case for that.
    unsigned int fineLevel = static_cast<unsigned int>( lwca::floorf( mipLevel ) );
    if( fineLevel != coarseLevel )
    {
        unsigned int  numTilesToRequestInFineLevel = 0;
        unsigned int* endOfPreviousRequests        = tilesToRequest + numTilesToRequest;
        tileIndexing.callwlateTileRequests( fineLevel, x, y, sampler.mipTailFirstLevel, 1.f / sampler.ilwAnisotropy,
                                            static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                            static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ),
                                            endOfPreviousRequests, numTilesToRequestInFineLevel );
        numTilesToRequest += numTilesToRequestInFineLevel;
    }

    uint*   usageBits     = get_const_Global().pageUsageBits;
    uint*   residenceBits = get_const_Global().pageResidenceBits;
    uint64* pageTable     = get_const_Global().pageTable;

    bool* isResident = reinterpret_cast<bool*>( isResidentPtr );

    // Request tiles
    uint startPage = getDemandTextureFirstPage( sampler.dd );
    *isResident    = true;
    for( unsigned int i = 0; i < numTilesToRequest; ++i )
    {
        bool         tileIsResident;
        unsigned int pageId = startPage + tilesToRequest[i];
        optixPagingMapOrRequest( usageBits, residenceBits, pageTable, pageId, &tileIsResident );
        *isResident = *isResident && tileIsResident;
    }

    if( *isResident )
    {
        // If running a debug or develop build, make sure OptiX's notion of
        // residency agrees with LWCA's. If not, return a debug value to make
        // it apparent.
#if defined( DEBUG ) || defined( DEVELOP )
        int isResident32 = 0;
        float4 result = Texture_getElement_hw_texgrad_2d_isResident( sampler.dd.texref, x, y, ddx.x, ddx.y, ddy.x, ddy.y, &isResident32 );
        if( !isResident32 )
            return float4( 1.0f, 1.0f, 0.0f, 0.0f ); // Yellow
#else
        float4 result = Texture_getElement_hw_texgrad_2d( sampler.dd.texref, x, y, ddx.x, ddx.y, ddy.x, ddy.y );
#endif
        return result;
    }
    else
    {
        return float4( 1.0f, 0.0f, 1.0f, 0.0f );
    }
}

inline cort::float4 fetchFromSoftwareTile4f( const TextureSampler& sampler,
                                             float                 x,
                                             float                 y,
                                             flt2                  ddx,
                                             flt2                  ddy,
                                             unsigned int          tileIndex,
                                             const TileLocator&    tileLocator,
                                             unsigned int          levelWidth,
                                             unsigned int          levelHeight )
{
    // If we're trying to fetch from the mip tail, do a standard texture call.
    if( optix::TileIndexing::isMipTailIndex( tileIndex ) )
        return Texture_getElement_hw_texgrad_2d( sampler.dd.texref, x, y, ddx.x, ddx.y, ddy.x, ddy.y );

    optix::TileIndexing tileIndexing( sampler.width, sampler.height, sampler.tileWidth, sampler.tileHeight );
    float               s, t;
    tileIndexing.textureToSoftwareTileCoord( sampler.tileGutterWidth, levelWidth, levelHeight, x, y,
                                             static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                             static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ), s, t );

    // Scale the derivatives, because they touch a larger portion of the tile than the texture
    flt2 scale( 0.0f, 0.0f );
    tileIndexing.getScaleFactorForTileDerivatives( levelWidth, levelHeight, scale.x, scale.y );
    ddx *= scale;
    ddy *= scale;

    // Get the sampler array for the texture
    unsigned long long tileArray  = get_const_Global().tileArrays[tileLocator.unpacked.tileArray];
    unsigned int       layerIndex = tileLocator.unpacked.tileIndex;

    return Texture_getElement_hw_texgrad_a2d( tileArray, layerIndex, s, t, ddx.x, ddx.y, ddy.x, ddy.y );
}

inline cort::float4 textureGradLoadOrRequest2_SWSparse( const TextureSampler& sampler, float x, float y, flt2 ddx, flt2 ddy, uint64 isResidentPtr )
{
    bool* isResident = reinterpret_cast<bool*>( isResidentPtr );

    float mipLevel = getMipLevelFromTextureGradients( ddx, ddy, sampler.width, sampler.height, sampler.ilwAnisotropy );
    // Note that if the texture is not mipmapped the mipLevel will be clamped to 0
    mipLevel = clampf( mipLevel, 0.0f, sampler.mipLevels - 1.0f );

    // Snap to the nearest mip level if we're doing point filtering
    if( sampler.mipmapFilterMode == lwca::lwdaFilterModePoint )
        mipLevel = lwca::floorf( mipLevel + 0.5f );

    optix::TileIndexing tileIndexing( sampler.width, sampler.height, sampler.tileWidth, sampler.tileHeight );

    unsigned int coarseLevel = static_cast<unsigned int>( lwca::ceilf( mipLevel ) );
    unsigned int coarseLevelWidth;
    unsigned int coarseLevelHeight;
    tileIndexing.callwlateLevelDims( coarseLevel, coarseLevelWidth, coarseLevelHeight );

    // We need to floor these so they don't round up when their result is between -1 and 0.
    int          coarsePixelX = lwca::floorf( x * coarseLevelWidth );
    int          coarsePixelY = lwca::floorf( y * coarseLevelHeight );
    unsigned int coarseIndex =
        tileIndexing.callwlateTileIndex( coarseLevel, coarsePixelX, coarsePixelY,
                                         static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                         static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ),
                                         coarseLevelWidth, coarseLevelHeight, sampler.mipTailFirstLevel );

    uint*   usageBits     = get_const_Global().pageUsageBits;
    uint*   residenceBits = get_const_Global().pageResidenceBits;
    uint64* pageTable     = get_const_Global().pageTable;

    const uint startPage = getDemandTextureFirstPage( sampler.dd );

    *isResident = true;
    unsigned long long coarsePageTableEntry =
        optixPagingMapOrRequest( usageBits, residenceBits, pageTable, startPage + coarseIndex, isResident );
    TileLocator coarseTileLocator;
    coarseTileLocator.packed = static_cast<unsigned int>( coarsePageTableEntry );

    unsigned int fineLevel = static_cast<unsigned int>( lwca::floorf( mipLevel ) );

    unsigned int fineIndex       = 0;
    unsigned int fineLevelWidth  = 0;
    unsigned int fineLevelHeight = 0;
    TileLocator fineTileLocator;
    if( fineLevel != coarseLevel )
    {
        tileIndexing.callwlateLevelDims( fineLevel, fineLevelWidth, fineLevelHeight );

        // We need to floor these so they don't round up when their result is between -1 and 0.
        int finePixelX = lwca::floorf( x * fineLevelWidth );
        int finePixelY = lwca::floorf( y * fineLevelHeight );
        fineIndex = tileIndexing.callwlateTileIndex( fineLevel, finePixelX, finePixelY,
                                                     static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode0 ),
                                                     static_cast<lwca::lwdaTextureAddressMode>( sampler.wrapMode1 ),
                                                     fineLevelWidth, fineLevelHeight, sampler.mipTailFirstLevel );

        bool               fineIsResident;
        unsigned long long finePageTableEntry =
            optixPagingMapOrRequest( usageBits, residenceBits, pageTable, startPage + fineIndex, &fineIsResident );
        fineTileLocator.packed = static_cast<unsigned int>( finePageTableEntry );
        *isResident            = *isResident && fineIsResident;
    }

    if( !*isResident )
    {
        return float4( 1.0f, 0.0f, 1.0f, 0.0f );
    }

    cort::float4 coarseColor = fetchFromSoftwareTile4f( sampler, x, y, ddx, ddy, coarseIndex, coarseTileLocator,
                                                        coarseLevelWidth, coarseLevelHeight );

    if( fineLevel == coarseLevel )
    {
        return coarseColor;
    }

    cort::float4 fineColor =
        fetchFromSoftwareTile4f( sampler, x, y, ddx, ddy, fineIndex, fineTileLocator, fineLevelWidth, fineLevelHeight );

    // Combine the two levels and return
    float weight = coarseLevel - mipLevel;
    return coarseColor * ( 1.f - weight ) + fineColor * weight;
}

extern "C" cort::float4 RTX_textureGradLoadOrRequest2( cort::CanonicalState* state,
                                                       uint                  textureId,
                                                       float                 x,
                                                       float                 y,
                                                       float                 dPdx_x,
                                                       float                 dPdx_y,
                                                       float                 dPdy_x,
                                                       float                 dPdy_y,
                                                       uint64                isResidentPtr )
{
    flt2                  ddx( dPdx_x, dPdx_y );
    flt2                  ddy( dPdy_x, dPdy_y );
    const TextureSampler& sampler = Global_getTextureSamplerHeader( state, textureId );

    if( !sampler.isDemandLoad )
    {
        bool* isResident = reinterpret_cast<bool*>( isResidentPtr );
        *isResident      = true;
        return Texture_getElement_hw_texgrad_2d( sampler.dd.texref, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y );
    }

    switch( Global_getDemandLoadMode( state ) )
    {
        case PagingMode::WHOLE_MIPLEVEL:
            return wholeMipLevelGradLoadOrRequest2( state, textureId, x, y, dPdx_x, dPdx_y, dPdy_x, dPdy_y, isResidentPtr );
        case PagingMode::LWDA_SPARSE_HYBRID:
            return textureGradLoadOrRequest2_lwdaSparse_hybrid( sampler, x, y, ddx, ddy, isResidentPtr );
        case PagingMode::LWDA_SPARSE_HARDWARE:
            return textureGradLoadOrRequest2_lwdaSparse_HW( sampler, x, y, ddx, ddy, isResidentPtr );
        case PagingMode::SOFTWARE_SPARSE:
            return textureGradLoadOrRequest2_SWSparse( sampler, x, y, ddx, ddy, isResidentPtr );
        case PagingMode::UNKNOWN:
            // We should never get here. Return a debug color.
            bool* isResident = reinterpret_cast<bool*>( isResidentPtr );
            *isResident = true;
            return float4( 1.0f, 0.0f, 1.0f, 0.0f );
    }
}

// Do isotropic sample of a demand loaded tiled texture based on a mip level
extern "C" cort::float4 RTX_textureLodLoadOrRequest2( cort::CanonicalState* state, unsigned int textureId, float s, float t, float mipLevel, uint64 isResidentPtr )
{
    if( Global_getDemandLoadMode( state ) == PagingMode::WHOLE_MIPLEVEL )
    {
        return wholeMipLevelLodLoadOrRequest2( state, textureId, s, t, mipLevel, isResidentPtr );
    }
    else
    {
        const TextureSampler& sampler = Global_getTextureSamplerHeader( state, textureId );

        const float expMipLevel  = lwca::exp2( mipLevel );
        const float sampleWidth  = expMipLevel / sampler.width;
        const float sampleHeight = expMipLevel / sampler.height;
        return RTX_textureGradLoadOrRequest2( state, textureId, s, t, sampleWidth, 0.0f, 0.0f, sampleHeight, isResidentPtr );
    }
}

///////////////////////////////////////////////////////////
//
// Buffer lookup specializations.
//

///////////////////////////////////////////////////////////
//
// Exception methods
//

void cort::Runtime_throwException( CanonicalState* state, unsigned int code )
{
    int data[23];
    RTX_throwException( code, data );
}

extern "C" void RTX_throwBufferIndexOutOfBoundsException( int          code,
                                                          cort::uint64 description,
                                                          cort::uint64 index_x,
                                                          cort::uint64 index_y,
                                                          cort::uint64 index_z,
                                                          int          dimensionality,
                                                          int          elementSize,
                                                          int          bufferId )
{
    cort::size3 bufferSize = Buffer_getSizeFromId( nullptr, bufferId );

    bool invalid = false;
    invalid |= index_x >= bufferSize.x;
    if( dimensionality > 1 )
    {
        invalid |= index_y >= bufferSize.y;
        if( dimensionality > 2 )
            invalid |= index_z >= bufferSize.z;
    }

    if( invalid )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64 = reinterpret_cast<cort::uint64*>( data );
        data64[0]                 = description;
        data64[1]                 = bufferSize.x;
        data64[2]                 = bufferSize.y;
        data64[3]                 = bufferSize.z;
        data64[4]                 = index_x;
        data64[5]                 = index_y;
        data64[6]                 = index_z;
        data[14]                  = dimensionality;
        data[15]                  = elementSize;
        data[16]                  = bufferId;
        RTX_throwException( code, data );
    }
}

extern "C" void RTX_throwExceptionCodeOutOfBoundsException( int code, cort::uint64 description, int exceptionCode, int exceptionCodeMin, int exceptionCodeMax )
{
    if( exceptionCode < exceptionCodeMin || exceptionCode > exceptionCodeMax )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64 = reinterpret_cast<cort::uint64*>( data );
        data64[0]                 = description;
        data[14]                  = exceptionCode;
        RTX_throwException( code, data );
    }
}

extern "C" void RTX_throwIlwalidIdException( int code, cort::uint64 description, int id, int idCheck )
{
    if( idCheck != 0 )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64 = reinterpret_cast<cort::uint64*>( data );
        data64[0]                 = description;
        data[14]                  = id;
        data[15]                  = idCheck;
        RTX_throwException( code, data );
    }
}

extern "C" void RTX_throwIlwalidRayException( int          code,
                                              cort::uint64 description,
                                              float        ox,
                                              float        oy,
                                              float        oz,
                                              float        dx,
                                              float        dy,
                                              float        dz,
                                              int          rayType,
                                              float        tmin,
                                              float        tmax )
{
    bool invalid = false;

    // Check if origin/direction contains NaN, -inf, or inf
    invalid |= RTX_isInfOrNan( ox );
    invalid |= RTX_isInfOrNan( oy );
    invalid |= RTX_isInfOrNan( oz );
    invalid |= RTX_isInfOrNan( dx );
    invalid |= RTX_isInfOrNan( dy );
    invalid |= RTX_isInfOrNan( dz );

    // Check if rayType is negative
    invalid |= rayType < 0;

    // Check if tmin or tmax is NaN
    invalid |= RTX_isNan( tmin );
    invalid |= RTX_isNan( tmax );

    if( invalid )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64    = reinterpret_cast<cort::uint64*>( data );
        float*             dataFloat = reinterpret_cast<float*>( data );
        data64[0]                    = description;
        dataFloat[14]                = ox;
        dataFloat[15]                = oy;
        dataFloat[16]                = oz;
        dataFloat[17]                = dx;
        dataFloat[18]                = dy;
        dataFloat[19]                = dz;
        data[20]                     = rayType;
        dataFloat[21]                = tmin;
        dataFloat[22]                = tmax;
        RTX_throwException( code, data );
    }
}

extern "C" void RTX_throwMaterialIndexOutOfBoundsException( int code, cort::uint64 description, cort::uint64 numMaterials, cort::uint64 index )
{
    if( index >= numMaterials )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64 = reinterpret_cast<cort::uint64*>( data );
        data64[0]                 = description;
        data64[1]                 = numMaterials;
        data64[2]                 = index;
        RTX_throwException( code, data );
    }
}

extern "C" void RTX_throwPayloadAccessOutOfBoundsException( int          code,
                                                            cort::uint64 description,
                                                            cort::uint64 valueOffset,
                                                            cort::uint64 valueSize,
                                                            cort::uint64 payloadSize,
                                                            cort::uint64 valueEnd )
{
    if( valueEnd > payloadSize )
    {
        __align__( 8 ) int data[23];
        cort::uint64*      data64 = reinterpret_cast<cort::uint64*>( data );
        data64[0]                 = description;
        data64[1]                 = valueOffset;
        data64[2]                 = valueSize;
        data64[3]                 = payloadSize;
        RTX_throwException( code, data );
    }
}

unsigned int cort::Exception_getCode( CanonicalState* state )
{
    unsigned int exceptionCode = RTX_getExceptionCode();

    if( exceptionCode == RTC_EXCEPTION_CODE_STACK_OVERFLOW )
        return RT_EXCEPTION_STACK_OVERFLOW;
    if( exceptionCode == RTC_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED )
        return RT_EXCEPTION_TRACE_DEPTH_EXCEEDED;
    return exceptionCode;
}

// Check whether the given id is valid or not.
//
// \return
//    - 0: Valid.
//    - 1: Invalid (id is 0).
//    - 2: Invalid (id is >= tableSize).
//    - 3: Invalid (id is -1).
unsigned int cort::Exception_checkIdIlwalid( unsigned int id, unsigned int tableSize )
{
    if( id == 0 )
        return 1;

    // Check this case first before the >= comparison below.
    if( id == (unsigned int)-1 )
        return 3;

    if( id >= tableSize )
        return 2;

    return 0;
}

// Check whether the given buffer id is valid or not.
//
// \return
//    -  0: Valid.
//    - -1: Invalid (id is 0).
//    - -2: Invalid (id is >= number of buffers).
//    - -3: Invalid (id is -1).
unsigned int cort::Exception_checkBufferIdIlwalid( CanonicalState* state, unsigned int bufferId )
{
    return Exception_checkIdIlwalid( bufferId, Global_getNumBuffers( state ) );
}

// Check whether the given texture id is valid or not.
//
// \return
//    -  0: Valid.
//    - -1: Invalid (id is 0).
//    - -2: Invalid (id is >= number of textures).
//    - -3: Invalid (id is -1).
unsigned int cort::Exception_checkTextureIdIlwalid( CanonicalState* state, unsigned int textureId )
{
    return Exception_checkIdIlwalid( textureId, Global_getNumTextures( state ) );
}

// Check whether the given program id is valid or not.
//
// \return
//    -  0: Valid.
//    - -1: Invalid (id is 0).
//    - -2: Invalid (id is >= number of programs).
//    - -3: Invalid (id is -1).
unsigned int cort::Exception_checkProgramIdIlwalid( CanonicalState* state, unsigned int programId )
{
    return Exception_checkIdIlwalid( programId, Global_getNumPrograms( state ) );
}

///////////////////////////////////////////////////////////
//
// Buffer lookup specializations.
//

char* Megakernel::Buffer_getElementAddress1dFromId_linear( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, uint64 x )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader1d_linear( state, buffer, eltSize, x );
}

char* Megakernel::Buffer_getElementAddress2dFromId_linear( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, uint64 x, uint64 y )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader2d_linear( state, buffer, eltSize, x, y );
}

char* Megakernel::Buffer_getElementAddress3dFromId_linear( CanonicalState* state,
                                                           unsigned int    bufferid,
                                                           unsigned int    eltSize,
                                                           uint64          x,
                                                           uint64          y,
                                                           uint64          z )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader3d_linear( state, buffer, eltSize, x, y, z );
}

char* Megakernel::Buffer_decodeBufferHeader1d_linear( CanonicalState* state, Buffer buffer, unsigned int eltSize, uint64 x )
{
    return buffer.dd.data + x * eltSize;
}

char* Megakernel::Buffer_decodeBufferHeader2d_linear( CanonicalState* state, Buffer buffer, unsigned int eltSize, uint64 x, uint64 y )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x );
}

char* Megakernel::Buffer_decodeBufferHeader3d_linear( CanonicalState* state, Buffer buffer, unsigned int eltSize, uint64 x, uint64 y, uint64 z )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x + z * buffer.di.size.x * buffer.di.size.y );
}

///////////////////////////////////////////////////////////
//
// Buffer lookup.
//

char* cort::Buffer_getElementAddress1d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress1dFromId( state, bufferid, eltSize, stackTmp, x );
}

char* cort::Buffer_getElementAddress2d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress2dFromId( state, bufferid, eltSize, stackTmp, x, y );
}

char* cort::Buffer_getElementAddress3d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress3dFromId( state, bufferid, eltSize, stackTmp, x, y, z );
}

char* cort::Buffer_getElementAddress1dFromId( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, char* stackTmp, uint64 x )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader1d_generic( state, buffer, eltSize, stackTmp, x );
}

char* cort::Buffer_getElementAddress2dFromId( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader2d_generic( state, buffer, eltSize, stackTmp, x, y );
}

char* cort::Buffer_getElementAddress3dFromId( CanonicalState* state,
                                              unsigned int    bufferid,
                                              unsigned int    eltSize,
                                              char*           stackTmp,
                                              uint64          x,
                                              uint64          y,
                                              uint64          z )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader3d_generic( state, buffer, eltSize, stackTmp, x, y, z );
}

char* cort::Buffer_decodeBufferHeader1d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x )
{
    // Texheap path is required only for 1d buffers with an element size of 16
    if( eltSize == 16 && buffer.dd.texUnit != -3 )
    {
        int offset               = (int)(long long)buffer.dd.data;
        *(cort::float4*)stackTmp = Texture_getElement_hwtexref_texfetch_1d( buffer.dd.texUnit, x + offset );
        return stackTmp;
    }
    else
    {
        return buffer.dd.data + x * eltSize;
    }
}

char* cort::Buffer_decodeBufferHeader2d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x );
}

char* cort::Buffer_decodeBufferHeader3d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x + z * buffer.di.size.x * buffer.di.size.y );
}

size3 cort::Buffer_getSize( CanonicalState* state, unsigned short token )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getSizeFromId( state, bufferid );
}

size3 cort::Buffer_getSizeFromId( CanonicalState* state, unsigned int bufferid )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return buffer.di.size;
}

unsigned int cort::Intersect_getPrimitive( CanonicalState* state )
{
    return RTX_getPrimitiveIdx();
}

extern "C" unsigned int RTX_getAttributeKind()
{
    GeometryHandle geom = getGeometryHandle();
    return Geometry_getAttributeKind( nullptr, geom );
}

#if 1  // TODO(bigler) where should I get these values from?  They are lwrrently defined in spec-lwvm-rt.txt.
/*! Hit kind flags as reported from intersection */
enum RThitkindflags
{
    RT_HIT_KIND_BUILTIN_PRIMITIVE_TYPE = 0x80,
    RT_HIT_KIND_TRIANGLE_FRONT_FACE    = 0xFE,
    RT_HIT_KIND_TRIANGLE_BACK_FACE     = 0xFF,
};
#endif

extern "C" unsigned char RTX_getHitKind();
bool cort::TraceFrame_isTriangleHit( CanonicalState* state )
{
    unsigned char hitKind = RTX_getHitKind();
    return hitKind & RT_HIT_KIND_BUILTIN_PRIMITIVE_TYPE;
}

bool cort::TraceFrame_isTriangleHitBackFace( CanonicalState* state )
{
    return RTX_getHitKind() == RT_HIT_KIND_TRIANGLE_BACK_FACE;
}

bool cort::TraceFrame_isTriangleHitFrontFace( CanonicalState* state )
{
    return RTX_getHitKind() == RT_HIT_KIND_TRIANGLE_FRONT_FACE;
}

///////////////////////////////////////////////////////////
//
// Trace setters and getters.
// FIXME: we are going to change the name of these functions, we should have no references to TraceFrame in the RTXRuntime.

void cort::TraceFrame_setCommittedTransformDepth( CanonicalState* state, unsigned char depth )
{
}

unsigned char cort::TraceFrame_getCommittedTransformDepth( CanonicalState* state )
{
    return 0;
}

void cort::TraceFrame_setLwrrentTransformDepth( CanonicalState* state, unsigned char depth )
{
}

unsigned char cort::TraceFrame_getLwrrentTransformDepth( CanonicalState* state )
{
    return 0;
}

extern "C" float RTX_getLwrrentTmax();

float cort::TraceFrame_getLwrrentTmax( CanonicalState* state )
{
    return RTX_getLwrrentTmax();
}

extern "C" void RTX_terminateRay();

void cort::TerminateRay_terminate( CanonicalState* state )
{
    RTX_terminateRay();
}

extern "C" void RTX_ignoreIntersection();

void cort::TraceFrame_ignoreIntersection( CanonicalState* state )
{
    RTX_ignoreIntersection();
}

extern "C" float RTX_getLwrrentTime();

float cort::Runtime_getLwrrentTime( CanonicalState* state )
{
    return RTX_getLwrrentTime();
}

extern "C" char* RTX_getPayloadPointer();

char* cort::TraceFrame_getPayloadAddress( CanonicalState* )
{
    // Note that RTXPlan/RTXCompile already take care to avoid promoting
    // the payload into registers if "cp->canPayloadPointerEscape()"
    return RTX_getPayloadPointer();
}

///////////////////////////////////////////////////////////
//
// LexicalScope methods.
//
AbstractGroupHandle cort::LexicalScope_colwertToAbstractGroupHandle( CanonicalState* state, LexicalScopeHandle ls )
{
    return ls;
}

GeometryInstanceHandle cort::LexicalScope_colwertToGeometryInstanceHandle( CanonicalState* state, LexicalScopeHandle ls )
{
    return ls;
}


GraphNodeHandle cort::LexicalScope_colwertToGraphNodeHandle( CanonicalState* state, LexicalScopeHandle object )
{
    return object;
}


///////////////////////////////////////////////////////////
//
// GraphNode methods.
//
CORT_OVERRIDABLE
TransformHandle cort::GraphNode_colwertToTransformHandle( CanonicalState* state, GraphNodeHandle node )
{
    return node;
}


///////////////////////////////////////////////////////////
//
// AbstractGroup methods.
//

AccelerationHandle cort::AbstractGroup_getAcceleration( CanonicalState* state, AbstractGroupHandle g )
{
    AbstractGroupRecord* agr = Global_getObjectRecord<AbstractGroupRecord>( state, g );
    return agr->accel;
}

BufferId cort::AbstractGroup_getChildren( CanonicalState* state, AbstractGroupHandle g )
{
    AbstractGroupRecord* agr = Global_getObjectRecord<AbstractGroupRecord>( state, g );
    return agr->children;
}


///////////////////////////////////////////////////////////
//
// Variable lookup.
//

DynamicVariableTableOffset cort::LexicalScope_getDynamicVariableTable( CanonicalState* state, LexicalScopeHandle object )
{
    LexicalScopeRecord* lsr = Global_getObjectRecord<LexicalScopeRecord>( state, object );
    return lsr->dynamicVariableTable;
}

LexicalScopeHandle getGraphNode()
{
    return 0;  // TODO
}

LexicalScopeHandle getScope1( optix::SemanticType callType )
{
    // clang-format off
  switch( callType )
  {
    case optix::ST_RAYGEN:                  return {0};
    case optix::ST_MISS:                    return {0};
    case optix::ST_EXCEPTION:               return {0};
    case optix::ST_INTERNAL_AABB_EXCEPTION: return {0};
    case optix::ST_CLOSEST_HIT:             return getGeometryInstanceHandle();
    case optix::ST_ANY_HIT:                 return getGeometryInstanceHandle();
    case optix::ST_INTERSECTION:            return getGeometryInstanceHandle();
    case optix::ST_ATTRIBUTE:               return getGeometryInstanceHandle();
    case optix::ST_BOUNDING_BOX:            return getGeometryInstanceHandle();
    case optix::ST_NODE_VISIT:              return getGraphNode();  // TODO: Implement GraphNode lookup.
    default:                                return {0};  // TODO: Exception? Failure in debug mode?
  }
    // clang-format on
}

LexicalScopeHandle getScope2( optix::SemanticType callType )
{
    // clang-format off
  switch( callType )
  {

    case optix::ST_RAYGEN:                  return {0};
    case optix::ST_MISS:                    return {0};
    case optix::ST_EXCEPTION:               return {0};
    case optix::ST_INTERNAL_AABB_EXCEPTION: return {0};
    case optix::ST_CLOSEST_HIT:             return getMaterialHandle();
    case optix::ST_ANY_HIT:                 return getMaterialHandle();
    case optix::ST_INTERSECTION:            return getGeometryHandle();
    case optix::ST_ATTRIBUTE:               return getGeometryHandle();
    case optix::ST_BOUNDING_BOX:            return getGeometryHandle();
    case optix::ST_NODE_VISIT:              return {0};
    default:                                return {0}; // TODO: Exception? Failure in debug mode?
  }
    // clang-format on
}

// Generic lookup.
// Calls to this should be generated with a constant semantic type so the code can be optimized.
char* cort::Runtime_lookupVariableAddress( CanonicalState*     state,
                                           unsigned short      token,
                                           char*               defaultValue,
                                           optix::SemanticType callType,
                                           optix::SemanticType inheritedType )
{
    // Look on scope 0 (Program).
    LexicalScopeHandle program = Program_colwertToLexicalScopeHandle( state, ActiveProgram_get( state, callType ) );
    if( char* data = LexicalScope_lookupVariable( state, program, token ) )
        return data;

    // Look on scope 1 (GraphNode or GI) if set.
    LexicalScopeHandle scope1 = getScope1( inheritedType );
    if( scope1 )
    {
        if( char* data = LexicalScope_lookupVariable( state, scope1, token ) )
            return data;
    }

    // Look on scope 2 (Geometry or Material) if set.
    LexicalScopeHandle scope2 = getScope2( inheritedType );
    if( scope2 )
    {
        if( char* data = LexicalScope_lookupVariable( state, scope2, token ) )
            return data;
    }

    // Look on scope 3 (GlobalScope).
    LexicalScopeHandle globalScope = GlobalScope_colwertToLexicalScopeHandle( state, GlobalScope_get( state ) );
    if( char* data = LexicalScope_lookupVariable( state, globalScope, token ) )
        return data;

    return defaultValue;
}

// Look up the requested variable in scope 1, based on the semantic type we know
// if the look up should be performed in the GeometryInstance or GraphNode.
// NOTE: Only call this if it is statically known that the variable is in scope 1.
char* cort::Runtime_lookupVariableAddress_Scope1( CanonicalState* state, unsigned short token, optix::SemanticType callType )
{
    LexicalScopeHandle scope1 = getScope1( callType );
    return LexicalScope_lookupVariableUnchecked( state, scope1, token );
}

// Look up the requested variable in scope 2, based on the semantic type we know
// if the look up should be performed in the Geometry or Material.
// NOTE: Only call this if it is statically known that the variable is in scope 2.
char* cort::Runtime_lookupVariableAddress_Scope2( CanonicalState* state, unsigned short token, optix::SemanticType callType )
{
    LexicalScopeHandle scope2 = getScope2( callType );
    return LexicalScope_lookupVariableUnchecked( state, scope2, token );
}

// DynamicVariableTable
CORT_OVERRIDABLE
unsigned short* cort::Global_getDynamicVariableTable( CanonicalState* state, DynamicVariableTableOffset offset )
{
    return get_const_Global().dynamicVariableTable + offset;
}

// The following functionality regarding dynamicVariableTable lookup is duplicated from common runtime.
namespace {

// Functionality operating on the dynamicVariableTable, which is actually just a collection of
// pairs of shorts, representing the variable token id and its offset into the table.

unsigned short getUnmarkedTokenId( const unsigned short* table, unsigned int index )
{
    return cort::getUnmarkedVariableTokenId( {table + 2 * index} );
}

bool isEntryALeaf( const unsigned short* table, unsigned int index )
{
    return cort::isVariableTableLeaf( ( table + 2 * index ) );
}

unsigned short getOffsetValue( const unsigned short* table, unsigned int index )
{
    return *( table + 2 * index + 1 );
}

}  // namespace

// Return the offset of a variable entry for a given token id inside a dynamicVariableTable or -1.
//
// The given table is actually a complete binary tree implicitly represented in an array. This
// allows for binary searching for a given token by starting at the root, ie index 0. As long as
// the token was not found, the search goes from index i to either child node 2*i+1 or 2*i+2
// whenever the current token is bigger or smaller then the one we are looking for, respectively.
// The leaf nodes are marked specifically, hence the value of each token is retrieved via a
// helper function getUnmarkedTokenId().
CORT_OVERRIDABLE
unsigned short cort::findVariableOffset( unsigned short token, const unsigned short* table )
{
    unsigned int index = 0;
    // loop until found - this is substitute for a relwrsive call
    while( true )
    {
        unsigned short lwrrentToken = getUnmarkedTokenId( table, index );
        if( lwrrentToken == token )
            return getOffsetValue( table, index );
        if( isEntryALeaf( table, index ) )
            return -1;

        if( lwrrentToken > token )
            index = 2 * index + 1;
        else
            index = 2 * index + 2;
    }
    //RT_ASSERT(!"We cannot get here!");
    return -1;
}

char* cort::LexicalScope_lookupVariable( CanonicalState* state, LexicalScopeHandle object, unsigned short token )
{
    // NOTE that the offset is in bytes, while the table is in unsigned short
    DynamicVariableTableOffset tableOffset = LexicalScope_getDynamicVariableTable( state, object );
    // if table is empty this returns -1
    unsigned short offset = ILWALIDOFFSET;
    if( tableOffset != (unsigned int)-1 )
    {
        unsigned short* table = Global_getDynamicVariableTable( state, tableOffset / sizeof( unsigned short ) );
        offset                = findVariableOffset( token, table );
    }
    if( offset == ILWALIDOFFSET )
        return 0;
    else
        return Global_getObjectRecord( state, object + offset );
}

char* cort::LexicalScope_lookupVariableUnchecked( CanonicalState* state, LexicalScopeHandle object, unsigned short token )
{
    // NOTE that the offset is in bytes, while the table is in unsigned short
    DynamicVariableTableOffset tableOffset = LexicalScope_getDynamicVariableTable( state, object );
    // if table is empty this returns -1
    unsigned short offset = ILWALIDOFFSET;
    if( tableOffset != (unsigned int)-1 )
    {
        unsigned short* table = Global_getDynamicVariableTable( state, tableOffset / sizeof( unsigned short ) );
        offset                = findVariableOffset( token, table );
    }
    return Global_getObjectRecord( state, object + offset );
}

GlobalScopeHandle cort::GlobalScope_get( CanonicalState* state )
{
    return 0;
}

LexicalScopeHandle cort::GlobalScope_colwertToLexicalScopeHandle( CanonicalState* state, GlobalScopeHandle globalScope )
{
    return globalScope;
}

///////////////////////////////////////////////////////////
//
// Transforms.
//

extern "C" int          RTX_getTransformListSize();
extern "C" uint64       RTX_getTransformListHandle( int );
extern "C" int          RTX_getTransformTypeFromHandle( uint64 );
extern "C" const uint64 RTX_getMatrixMotionTransformFromHandle( uint64 );
extern "C" const uint64 RTX_getSRTMotionTransformFromHandle( uint64 );
extern "C" const uint64 RTX_getStaticTransformFromHandle( uint64 );
extern "C" const uint64 RTX_getInstanceTransformFromHandle( uint64 );
extern "C" const uint64 RTX_getInstanceIlwerseTransformFromHandle( uint64 );

// We should consider how to unify this between API code and here.  The definition is in
// include/internal/optix_defines.h which doesn't have a lot of other code in it.
enum RTtransformkind
{
    RT_WORLD_TO_OBJECT = 0xf00, /*!< World to Object transformation */
    RT_OBJECT_TO_WORLD          /*!< Object to World transformation */
};

/*! Transform flags */
enum RTtransformflags
{
    RT_INTERNAL_ILWERSE_TRANSPOSE = 0x1000 /*!< Ilwerse transpose flag */
};

// TODO the current implementation of applyLwrrentTransforms supports only one level of transforms.
// Implement full support when we know how to support multiple levels.

CORT_OVERRIDABLE
GraphNodeHandle cort::Transform_getChild( CanonicalState* state, TransformHandle transform )
{
    TransformRecord* tr = Global_getObjectRecord<TransformRecord>( state, transform );
    return tr->child;
}

extern "C" uint4 RTX_vectorizedLoadTextureCache( char* address );

namespace {

template <class T>
__forceinline__ T loadReadOnlyAlign16( const T* ptr )
{
    T v;
    for( int ofs                     = 0; ofs < sizeof( T ); ofs += 16 )
        *(uint4*)( (char*)&v + ofs ) = RTX_vectorizedLoadTextureCache( (char*)ptr + ofs );
    return v;
}

// use vector type to force vector load
typedef int int4_align __attribute__( ( ext_vector_type( 4 ) ) );
template <class T>
__forceinline__ T loadAlign16( const T* ptr )
{
    T v;
    for( int ofs                          = 0; ofs < sizeof( T ); ofs += 16 )
        *(int4_align*)( (char*)&v + ofs ) = *( (int4_align*)( (char*)ptr + ofs ) );
    return v;
}

// extract row from identity matrix by index
__forceinline__ float4 loadItentityRow( int row )
{
    if( row == 0 )
        return float4{1, 0, 0, 0};
    else if( row == 1 )
        return float4{0, 1, 0, 0};
    return float4{0, 0, 1, 0};
}

// extract row from matrix by index
__forceinline__ float4 selectRrow( const float4 m0, const float4 m1, const float4 m2, const int row )
{
    if( row == 0 )
        return m0;
    else if( row == 1 )
        return m1;

    return m2;
}

// right multiply a row with a matrix
__forceinline__ float4 rowMatrixMul( const float4 m0, const float4 m1, const float4 m2, const float4 in )
{
    float4 out;

    out.x = in.x * m0.x + in.y * m1.x + in.z * m2.x;
    out.y = in.x * m0.y + in.y * m1.y + in.z * m2.y;
    out.z = in.x * m0.z + in.y * m1.z + in.z * m2.z;
    out.w = in.x * m0.w + in.y * m1.w + in.z * m2.w + in.w;

    return out;
}

// left multiply a column with a matrix
__forceinline__ float4 matrixColMul( const float4 m0, const float4 m1, const float4 m2, const float4 in )
{
    float4 out;
    out.x = in.x * m0.x + in.y * m0.y + in.z * m0.z + in.w * m0.w;
    out.y = in.x * m1.x + in.y * m1.y + in.z * m1.z + in.w * m1.w;
    out.z = in.x * m2.x + in.y * m2.y + in.z * m2.z + in.w * m2.w;
    out.w = in.w;

    return out;
}

__forceinline__ void getMatrixFromSrt( float4& r0, float4& r1, float4& r2, const SrtTransform& srt )
{
    const float4 q = {srt.qx, srt.qy, srt.qz, srt.qw};

    // normalize
    const float  ilw_sql = 1.f / ( srt.qx * srt.qx + srt.qy * srt.qy + srt.qz * srt.qz + srt.qw * srt.qw );
    const float4 nq      = q * ilw_sql;

    const float sqw = q.w * nq.w;
    const float sqx = q.x * nq.x;
    const float sqy = q.y * nq.y;
    const float sqz = q.z * nq.z;

    const float xy = q.x * nq.y;
    const float zw = q.z * nq.w;
    const float xz = q.x * nq.z;
    const float yw = q.y * nq.w;
    const float yz = q.y * nq.z;
    const float xw = q.x * nq.w;

    r0.x = ( sqx - sqy - sqz + sqw );
    r0.y = 2.0f * ( xy - zw );
    r0.z = 2.0f * ( xz + yw );

    r1.x = 2.0f * ( xy + zw );
    r1.y = ( -sqx + sqy - sqz + sqw );
    r1.z = 2.0f * ( yz - xw );

    r2.x = 2.0f * ( xz - yw );
    r2.y = 2.0f * ( yz + xw );
    r2.z = ( -sqx - sqy + sqz + sqw );

    r0.w = r0.x * srt.pvx + r0.y * srt.pvy + r0.z * srt.pvz + srt.tx;
    r1.w = r1.x * srt.pvx + r1.y * srt.pvy + r1.z * srt.pvz + srt.ty;
    r2.w = r2.x * srt.pvx + r2.y * srt.pvy + r2.z * srt.pvz + srt.tz;

    r0.z = r0.x * srt.b + r0.y * srt.c + r0.z * srt.sz;
    r1.z = r1.x * srt.b + r1.y * srt.c + r1.z * srt.sz;
    r2.z = r2.x * srt.b + r2.y * srt.c + r2.z * srt.sz;

    r0.y = r0.x * srt.a + r0.y * srt.sy;
    r1.y = r1.x * srt.a + r1.y * srt.sy;
    r2.y = r2.x * srt.a + r2.y * srt.sy;

    r0.x = r0.x * srt.sx;
    r1.x = r1.x * srt.sx;
    r2.x = r2.x * srt.sx;
}

__forceinline__ void loadInterpolatedMatrixKey( float4& trf0, float4& trf1, float4& trf2, const float4* transform, const float t1 )
{
    // Set the new ray
    trf0 = loadReadOnlyAlign16( &transform[0] );
    trf1 = loadReadOnlyAlign16( &transform[1] );
    trf2 = loadReadOnlyAlign16( &transform[2] );

    if( t1 > 0.f )
    {
        // without the motion conditional both keys are loaded conlwrently leading to spills
        const float t0 = 1.f - t1;
        trf0           = trf0 * t0 + loadReadOnlyAlign16( &transform[3] ) * t1;
        trf1           = trf1 * t0 + loadReadOnlyAlign16( &transform[4] ) * t1;
        trf2           = trf2 * t0 + loadReadOnlyAlign16( &transform[5] ) * t1;
    }
}

__forceinline__ void loadInterpolatedSrtKey( float4& srt0, float4& srt1, float4& srt2, float4& srt3, const float4* srtPtr, const float t )
{
    srt0 = loadReadOnlyAlign16( &srtPtr[0] );
    srt1 = loadReadOnlyAlign16( &srtPtr[1] );
    srt2 = loadReadOnlyAlign16( &srtPtr[2] );
    srt3 = loadReadOnlyAlign16( &srtPtr[3] );

    if( t > 0.f )
    {
        // without the motion conditional both keys are loaded conlwrently leading to spills
        const float t0 = 1.f - t;

        srt0 = srt0 * t0 + loadReadOnlyAlign16( &srtPtr[4] ) * t;
        srt1 = srt1 * t0 + loadReadOnlyAlign16( &srtPtr[5] ) * t;
        srt2 = srt2 * t0 + loadReadOnlyAlign16( &srtPtr[6] ) * t;
        srt3 = srt3 * t0 + loadReadOnlyAlign16( &srtPtr[7] ) * t;
    }
}

template <typename T>
__forceinline__ void resolveMotionKey( float& localt, int& key, const T* data, const float globalt )
{
    const float timeBegin    = loadReadOnlyAlign16( data ).timeBegin;
    const float timeEnd      = loadReadOnlyAlign16( data ).timeEnd;
    const float numIntervals = (float)( loadReadOnlyAlign16( data ).numKeys - 1 );

    // no need to check the motion flags. none of the handles on a valid transform list can be in vanish mode.

    const float time =
        lwca::maxf( 0.f, lwca::minf( numIntervals, ( globalt - timeBegin ) * numIntervals / ( timeEnd - timeBegin ) ) );
    const float fltKey = lwca::floorf( time );

    localt = time - fltKey;
    key    = lwca::float2int_rz( fltKey );
}

__forceinline__ void resolveMotionMatrix( float4& trf0, float4& trf1, float4& trf2, const RtcMatrixMotionTransform* transformData, const float time )
{
    // compute key and intra key time
    float keyTime;
    int   key;
    resolveMotionKey( keyTime, key, transformData, time );

    // get pointer to left key
    const float4* transform = (const float4*)( &transformData->transform[key][0] );

    // load and interpolate matrix keys
    loadInterpolatedMatrixKey( trf0, trf1, trf2, transform, keyTime );
}

__forceinline__ void resolveMotionMatrix( float4& trf0, float4& trf1, float4& trf2, const RtcSRTMotionTransform* transformData, const float time )
{
    // compute key and intra key time
    float keyTime;
    int   key;
    resolveMotionKey( keyTime, key, transformData, time );

    // get pointer to left key
    const float4* dataPtr = reinterpret_cast<const float4*>( &transformData->quaternion[key][0] );

    // load and interpolated srt keys
    float4 data[4];
    loadInterpolatedSrtKey( data[0], data[1], data[2], data[3], dataPtr, keyTime );

    SrtTransform srt = {data[0].x, data[0].y, data[0].z, data[0].w, data[1].x, data[1].y, data[1].z, data[1].w,
                        data[2].x, data[2].y, data[2].z, data[2].w, data[3].x, data[3].y, data[3].z, data[3].w};

    // colwert srt to matrix form
    getMatrixFromSrt( trf0, trf1, trf2, srt );
}

// ilwert 4x4 affine matrix in place
__forceinline__ void ilwerseAffineMatrix( float4& trf0, float4& trf1, float4& trf2 )
{
    const float det3 = trf0.x * ( trf1.y * trf2.z - trf1.z * trf2.y ) - trf0.y * ( trf1.x * trf2.z - trf1.z * trf2.x )
                       + trf0.z * ( trf1.x * trf2.y - trf1.y * trf2.x );

    const float ilw_det3 = 1.0f / det3;

    float ilw3[3][3];
    ilw3[0][0] = ilw_det3 * ( trf1.y * trf2.z - trf2.y * trf1.z );
    ilw3[0][1] = ilw_det3 * ( trf0.z * trf2.y - trf2.z * trf0.y );
    ilw3[0][2] = ilw_det3 * ( trf0.y * trf1.z - trf1.y * trf0.z );

    ilw3[1][0] = ilw_det3 * ( trf1.z * trf2.x - trf2.z * trf1.x );
    ilw3[1][1] = ilw_det3 * ( trf0.x * trf2.z - trf2.x * trf0.z );
    ilw3[1][2] = ilw_det3 * ( trf0.z * trf1.x - trf1.z * trf0.x );

    ilw3[2][0] = ilw_det3 * ( trf1.x * trf2.y - trf2.x * trf1.y );
    ilw3[2][1] = ilw_det3 * ( trf0.y * trf2.x - trf2.y * trf0.x );
    ilw3[2][2] = ilw_det3 * ( trf0.x * trf1.y - trf1.x * trf0.y );

    const float b[3] = {trf0.w, trf1.w, trf2.w};

    trf0.x = ilw3[0][0];
    trf0.y = ilw3[0][1];
    trf0.z = ilw3[0][2];
    trf0.w = -ilw3[0][0] * b[0] - ilw3[0][1] * b[1] - ilw3[0][2] * b[2];

    trf1.x = ilw3[1][0];
    trf1.y = ilw3[1][1];
    trf1.z = ilw3[1][2];
    trf1.w = -ilw3[1][0] * b[0] - ilw3[1][1] * b[1] - ilw3[1][2] * b[2];

    trf2.x = ilw3[2][0];
    trf2.y = ilw3[2][1];
    trf2.z = ilw3[2][2];
    trf2.w = -ilw3[2][0] * b[0] - ilw3[2][1] * b[1] - ilw3[2][2] * b[2];
}

__forceinline__ void loadMatrixFromHandle( float4&      trf0,
                                           float4&      trf1,
                                           float4&      trf2,
                                           const uint64 handle,
                                           const bool   enableMotion,
                                           const float  time,
                                           const bool   objectToWorld )
{
    const RtcTransformType type = (RtcTransformType)RTX_getTransformTypeFromHandle( handle );
    if( enableMotion && ( type == RTC_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRANSFORM_TYPE_SRT_MOTION_TRANSFORM ) )
    {
        // TODO: we could share the motion key computation
        if( type == RTC_TRANSFORM_TYPE_MATRIX_MOTION_TRANSFORM )
        {
            const RtcMatrixMotionTransform* transformData =
                (const RtcMatrixMotionTransform*)RTX_getMatrixMotionTransformFromHandle( handle );
            resolveMotionMatrix( trf0, trf1, trf2, transformData, time );
        }
        else
        {
            const RtcSRTMotionTransform* transformData =
                (const RtcSRTMotionTransform*)RTX_getSRTMotionTransformFromHandle( handle );
            resolveMotionMatrix( trf0, trf1, trf2, transformData, time );
        }
        if( !objectToWorld )
        {
            // the ilwerse motion transform is not available so it needs to be ilwerted at runtime
            ilwerseAffineMatrix( trf0, trf1, trf2 );
        }
    }
    else if( type == RTC_TRANSFORM_TYPE_INSTANCE || type == RTC_TRANSFORM_TYPE_STATIC_TRANSFORM )
    {
        const float4* transform;

        if( type == RTC_TRANSFORM_TYPE_INSTANCE )
        {
            transform = ( objectToWorld ) ? (const float4*)RTX_getInstanceTransformFromHandle( handle ) :
                                            (const float4*)RTX_getInstanceIlwerseTransformFromHandle( handle );
        }
        else
        {
            const RtcTravStaticTransform* traversable = (const RtcTravStaticTransform*)RTX_getStaticTransformFromHandle( handle );

            transform = ( objectToWorld ) ? (const float4*)traversable->transform : (const float4*)traversable->ilwTransform;
        }

        trf0 = loadReadOnlyAlign16( &transform[0] );
        trf1 = loadReadOnlyAlign16( &transform[1] );
        trf2 = loadReadOnlyAlign16( &transform[2] );
    }
    else
    {
        trf0 = {1, 0, 0, 0};
        trf1 = {0, 1, 0, 0};
        trf2 = {0, 0, 1, 0};
    }
}

CORT_OVERRIDABLE extern "C" int RTX_hasMotionTransforms = 1;

extern "C" Matrix4x4 getWorldToObjectTransformMatrix()
{
    // generic world to object transform computation
    float4 r0, r1, r2;

    const float time = RTX_getLwrrentTime();
    const int   size = RTX_getTransformListSize();

    if( size == 0 )
    {
        return Matrix4x4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }

#pragma unroll 1
    for( unsigned int i = 0; i < size; ++i )
    {
        RtcTraversableHandle handle = (RtcTraversableHandle)RTX_getTransformListHandle( i );

        // TODO: detect when motion is off
        float4 trf0, trf1, trf2;
        loadMatrixFromHandle( trf0, trf1, trf2, handle, RTX_hasMotionTransforms, time, false );

        if( i == 0 )
        {
            r0 = trf0;
            r1 = trf1;
            r2 = trf2;
        }
        else
        {
            // right multiply rows with pre-multiplied matrix
            float4 m0 = r0, m1 = r1, m2 = r2;
            r0 = rowMatrixMul( m0, m1, m2, trf0 );
            r1 = rowMatrixMul( m0, m1, m2, trf1 );
            r2 = rowMatrixMul( m0, m1, m2, trf2 );
        }
    }

    return Matrix4x4{r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, 0, 0, 0, 1};
}

extern "C" Matrix4x4 getObjectToWorldTransformMatrix()
{
    // generic world to object transform computation
    float4 r0, r1, r2;

    const float time = RTX_getLwrrentTime();
    const int   size = RTX_getTransformListSize();

    if( size == 0 )
    {
        return Matrix4x4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    }

#pragma unroll 1
    for( int i = size - 1; i >= 0; --i )
    {
        RtcTraversableHandle handle = (RtcTraversableHandle)RTX_getTransformListHandle( i );

        // TODO: detect when motion is off
        float4 trf0, trf1, trf2;
        loadMatrixFromHandle( trf0, trf1, trf2, handle, RTX_hasMotionTransforms, time, true );

        if( i == size - 1 )
        {
            r0 = trf0;
            r1 = trf1;
            r2 = trf2;
        }
        else
        {
            // right multiply rows with pre-multiplied matrix
            float4 m0 = r0, m1 = r1, m2 = r2;
            r0 = rowMatrixMul( m0, m1, m2, trf0 );
            r1 = rowMatrixMul( m0, m1, m2, trf1 );
            r2 = rowMatrixMul( m0, m1, m2, trf2 );
        }
    }

    return Matrix4x4{r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, 0, 0, 0, 1};
}
}

// -----------------------------------------------------------------------------
float4 cort::Runtime_applyLwrrentTransforms( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w )
{
    const bool         ilw_transpose = transform_kind & RT_INTERNAL_ILWERSE_TRANSPOSE;
    const unsigned int kind          = transform_kind & ~RT_INTERNAL_ILWERSE_TRANSPOSE;

    Matrix4x4 matrix;
    if( kind == RT_WORLD_TO_OBJECT )
    {
        if( !ilw_transpose )
            matrix = getWorldToObjectTransformMatrix();
        else
            matrix = getObjectToWorldTransformMatrix();
    }
    else if( kind == RT_OBJECT_TO_WORLD )
    {
        if( !ilw_transpose )
            matrix = getObjectToWorldTransformMatrix();
        else
            matrix = getWorldToObjectTransformMatrix();
    }

    float v0, v1, v2, v3;
    if( ilw_transpose )
    {
        v0 = matrix.matrix[0][0] * x;
        v0 += matrix.matrix[1][0] * y;
        v0 += matrix.matrix[2][0] * z;
        v0 += matrix.matrix[3][0] * w;
        v1 = matrix.matrix[0][1] * x;
        v1 += matrix.matrix[1][1] * y;
        v1 += matrix.matrix[2][1] * z;
        v1 += matrix.matrix[3][1] * w;
        v2 = matrix.matrix[0][2] * x;
        v2 += matrix.matrix[1][2] * y;
        v2 += matrix.matrix[2][2] * z;
        v2 += matrix.matrix[3][2] * w;
        v3 = matrix.matrix[0][3] * x;
        v3 += matrix.matrix[1][3] * y;
        v3 += matrix.matrix[2][3] * z;
        v3 += matrix.matrix[3][3] * w;
    }
    else
    {
        v0 = matrix.matrix[0][0] * x;
        v0 += matrix.matrix[0][1] * y;
        v0 += matrix.matrix[0][2] * z;
        v0 += matrix.matrix[0][3] * w;
        v1 = matrix.matrix[1][0] * x;
        v1 += matrix.matrix[1][1] * y;
        v1 += matrix.matrix[1][2] * z;
        v1 += matrix.matrix[1][3] * w;
        v2 = matrix.matrix[2][0] * x;
        v2 += matrix.matrix[2][1] * y;
        v2 += matrix.matrix[2][2] * z;
        v2 += matrix.matrix[2][3] * w;
        v3 = matrix.matrix[3][0] * x;
        v3 += matrix.matrix[3][1] * y;
        v3 += matrix.matrix[3][2] * z;
        v3 += matrix.matrix[3][3] * w;
    }

    return float4( v0, v1, v2, v3 );
}

// -----------------------------------------------------------------------------
float4 cort::Runtime_applyLwrrentTransforms_atMostOne( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w )
{
    const bool         ilw_transpose = transform_kind & RT_INTERNAL_ILWERSE_TRANSPOSE;
    const unsigned int kind          = transform_kind & ~RT_INTERNAL_ILWERSE_TRANSPOSE;

    Matrix4x4 matrix;
    if( kind == RT_WORLD_TO_OBJECT )
    {
        if( !ilw_transpose )
            matrix = getWorldToObjectTransformMatrix();
        else
            matrix = getObjectToWorldTransformMatrix();
    }
    else if( kind == RT_OBJECT_TO_WORLD )
    {
        if( !ilw_transpose )
            matrix = getObjectToWorldTransformMatrix();
        else
            matrix = getWorldToObjectTransformMatrix();
    }

    float v0, v1, v2, v3;
    if( ilw_transpose )
    {
        v0 = matrix.matrix[0][0] * x;
        v0 += matrix.matrix[1][0] * y;
        v0 += matrix.matrix[2][0] * z;
        v0 += matrix.matrix[3][0] * w;
        v1 = matrix.matrix[0][1] * x;
        v1 += matrix.matrix[1][1] * y;
        v1 += matrix.matrix[2][1] * z;
        v1 += matrix.matrix[3][1] * w;
        v2 = matrix.matrix[0][2] * x;
        v2 += matrix.matrix[1][2] * y;
        v2 += matrix.matrix[2][2] * z;
        v2 += matrix.matrix[3][2] * w;
        v3 = matrix.matrix[0][3] * x;
        v3 += matrix.matrix[1][3] * y;
        v3 += matrix.matrix[2][3] * z;
        v3 += matrix.matrix[3][3] * w;
    }
    else
    {
        v0 = matrix.matrix[0][0] * x;
        v0 += matrix.matrix[0][1] * y;
        v0 += matrix.matrix[0][2] * z;
        v0 += matrix.matrix[0][3] * w;
        v1 = matrix.matrix[1][0] * x;
        v1 += matrix.matrix[1][1] * y;
        v1 += matrix.matrix[1][2] * z;
        v1 += matrix.matrix[1][3] * w;
        v2 = matrix.matrix[2][0] * x;
        v2 += matrix.matrix[2][1] * y;
        v2 += matrix.matrix[2][2] * z;
        v2 += matrix.matrix[2][3] * w;
        v3 = matrix.matrix[3][0] * x;
        v3 += matrix.matrix[3][1] * y;
        v3 += matrix.matrix[3][2] * z;
        v3 += matrix.matrix[3][3] * w;
    }

    return float4( v0, v1, v2, v3 );
}


// -----------------------------------------------------------------------------
Matrix4x4 cort::Runtime_getTransform( CanonicalState* state, unsigned int transform_kind )
{
    if( transform_kind == RT_WORLD_TO_OBJECT )
        return getWorldToObjectTransformMatrix();
    else
        return getObjectToWorldTransformMatrix();
}

// -----------------------------------------------------------------------------
Matrix4x4 cort::Runtime_getTransform_atMostOne( CanonicalState* state, unsigned int transform_kind )
{
    if( transform_kind == RT_WORLD_TO_OBJECT )
        return getWorldToObjectTransformMatrix();
    else
        return getObjectToWorldTransformMatrix();
}

///////////////////////////////////////////////////////////
//
// Bounding box programs.
//

extern "C" void RTX_computeGeometryInstanceAABB_BoundingBoxProgramStub( CanonicalState*        state,
                                                                        ProgramId              pid,
                                                                        GeometryInstanceHandle gi,
                                                                        unsigned int           primitive,
                                                                        unsigned int           motionIndex,
                                                                        float*                 aabb );

extern "C" CORT_OVERRIDABLE void RTX_computeGeometryInstanceAABB( CanonicalState*        state,
                                                                  GeometryInstanceHandle gi,
                                                                  unsigned int           primitiveIndex,
                                                                  unsigned int           motionIndex,
                                                                  float*                 aabb )
{
    GeometryHandle geometry        = GeometryInstance_getGeometry( state, gi );
    ProgramHandle  program         = Geometry_getAABBProgram( state, geometry );
    ProgramId      pid             = Program_getProgramID( state, program );
    unsigned int   offsetPrimitive = primitiveIndex + Geometry_getPrimitiveIndexOffset( state, geometry );
    RTX_computeGeometryInstanceAABB_BoundingBoxProgramStub( state, pid, gi, offsetPrimitive, motionIndex, aabb );
}

void RTX_computeGraphNodeGeneralBB( CanonicalState* state, GraphNodeHandle graphNode, GeneralBB* genbb )
{
    ProgramHandle program = GraphNode_getBBProgram( state, graphNode );
    ProgramId     pid     = Program_getProgramID( state, program );
    RTX_computeGeometryInstanceAABB_BoundingBoxProgramStub( state, pid, graphNode, 0, 0, (float*)genbb );
}

extern "C" CORT_OVERRIDABLE void RTX_computeGroupChildAABB( CanonicalState* state, AbstractGroupHandle grp, unsigned int child, float* aabbf )
{
    // Find the child
    BufferId            children = AbstractGroup_getChildren( state, grp );
    const unsigned int  eltSize  = sizeof( unsigned int );
    char                stackTemp[eltSize];
    const unsigned int* addr =
        reinterpret_cast<const unsigned int*>( Buffer_getElementAddress1dFromId( state, children, eltSize, stackTemp, child ) );
    GraphNodeHandle childNode = LexicalScope_colwertToGraphNodeHandle( state, {*addr} );

    // Set up bounding box as invalid (probably not necessary, but paranoid).
    GeneralBB genbb;
    genbb.ilwalidate();

    // Visit the child
    RTX_computeGraphNodeGeneralBB( state, childNode, &genbb );

    // Write out the bounding box
    Aabb* aabb = (Aabb*)aabbf;
    aabb->set( genbb );
}

extern "C" CORT_OVERRIDABLE void RTX_gatherMotionAABBs( CanonicalState* state, unsigned int groupOffset, float* aabb )
{
    printf( "!!Not yet implemented!!.\n" );
}
