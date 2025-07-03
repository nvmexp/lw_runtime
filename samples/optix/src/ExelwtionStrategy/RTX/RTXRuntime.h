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

#include "../CommonRuntime.h"

extern "C" unsigned int   RTX_getInstanceFlags();
extern "C" unsigned short RTX_getRayFlags();
extern "C" unsigned char  RTX_getRayMask();
extern "C" cort::float2   RTX_getTriangleBarycentrics();
extern "C" unsigned int   RTX_getPrimitiveIdx();
extern "C" unsigned int   RTX_getAttributeKind();
extern "C" unsigned int   RTX_getInstanceIndex();

extern "C" void RTX_throwBufferIndexOutOfBoundsException( int          code,
                                                          cort::uint64 description,
                                                          cort::uint64 index_x,
                                                          cort::uint64 index_y,
                                                          cort::uint64 index_z,
                                                          int          dimensionality,
                                                          int          elementSize,
                                                          int          bufferId );
extern "C" void RTX_throwExceptionCodeOutOfBoundsException( int code, cort::uint64 description, int exceptionCode, int exceptionCodeMin, int exceptionCodeMax );
extern "C" void RTX_throwIlwalidIdException( int code, cort::uint64 description, int id, int idCheck );
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
                                              float        tmax );
extern "C" void RTX_throwMaterialIndexOutOfBoundsException( int code, cort::uint64 description, cort::uint64 numMaterials, cort::uint64 index );
extern "C" void RTX_throwPayloadAccessOutOfBoundsException( int          code,
                                                            cort::uint64 description,
                                                            cort::uint64 valueOffset,
                                                            cort::uint64 valueSize,
                                                            cort::uint64 payloadSize,
                                                            cort::uint64 valueEnd );

// DemandBuffer routines
extern "C" char* RTX_requestBufferElement1( cort::CanonicalState* state, unsigned int bufferId, unsigned int elementSize, cort::uint64 x );
extern "C" char* RTX_requestBufferElement2( cort::CanonicalState* state,
                                            unsigned int          bufferId,
                                            unsigned int          elementSize,
                                            cort::uint64          x,
                                            cort::uint64          y );
extern "C" char* RTX_requestBufferElement3( cort::CanonicalState* state,
                                            unsigned int          bufferId,
                                            unsigned int          elementSize,
                                            cort::uint64          x,
                                            cort::uint64          y,
                                            cort::uint64          z );

namespace cort {
// SGP: These functions should be cleaned up and made consistent with other naming colwentions. Or removed.
LexicalScopeHandle getMaterialHandle();
LexicalScopeHandle getGeometryInstanceHandle();
GeometryHandle     getGeometryHandle();
int getSBTSkip( unsigned int matlIndex );
ProgramHandle ActiveProgram_get( CanonicalState* state, optix::SemanticType callType );
ProgramHandle RayGenMissExceptionProgram_get( CanonicalState* state );
ProgramHandle IntersectionProgram_get( CanonicalState* state );
ProgramHandle AttributeProgram_get( CanonicalState* state );
ProgramHandle ClosestHitProgram_get( CanonicalState* state );
ProgramHandle AnyHitProgram_get( CanonicalState* state );
ProgramHandle BindlessCallableProgram_get( CanonicalState* state );
ProgramHandle BoundCallableProgram_get( CanonicalState* state );
ProgramHandle Material_getAHProgram( CanonicalState* state, MaterialHandle material );
ProgramHandle Material_getCHProgram( CanonicalState* state, MaterialHandle material );
unsigned int CallableProgram_getSBTBaseIndex( CanonicalState* state, unsigned int programid );
char* Runtime_lookupVariableAddress( CanonicalState*     state,
                                     unsigned short      token,
                                     char*               defaultValue,
                                     optix::SemanticType callType,
                                     optix::SemanticType inheritedType );
char* Runtime_lookupVariableAddress_Scope1( CanonicalState* state, unsigned short token, optix::SemanticType callType );
char* Runtime_lookupVariableAddress_Scope2( CanonicalState* state, unsigned short token, optix::SemanticType callType );
char* TraceFrame_getPayloadAddress( CanonicalState* state );
void Runtime_throwException( CanonicalState* state, unsigned int code );
uint3 Raygen_getLaunchIndex1dOr2d( CanonicalState* state );
uint3 Raygen_getLaunchIndex3d( CanonicalState* state );

unsigned int Geometry_getAttributeKind( CanonicalState* state, GeometryHandle geometry );
ProgramHandle Geometry_getAttributeProgram( CanonicalState* state, GeometryHandle geometry );
// GeometryTriangles methods
int GeometryTriangles_getVertexBufferId( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );
long long GeometryTriangles_getVertexBufferOffset( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );
unsigned long long GeometryTriangles_getVertexBufferStride( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );
int GeometryTriangles_getIndexBufferId( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );
long long GeometryTriangles_getIndexBufferOffset( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );
unsigned long long GeometryTriangles_getIndexBufferStride( CanonicalState* state, GeometryTrianglesHandle geometryTriangles );

// MotionGeometryTriangles methods
unsigned long long MotionGeometryTriangles_getVertexBufferMotionStride( CanonicalState*               state,
                                                                        MotionGeometryTrianglesHandle geometryTriangles );
int MotionGeometryTriangles_getMotionNumIntervals( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles );
float MotionGeometryTriangles_getTimeBegin( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles );
float MotionGeometryTriangles_getTimeEnd( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles );
int MotionGeometryTriangles_getMotionBorderModeBegin( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles );
int MotionGeometryTriangles_getMotionBorderModeEnd( CanonicalState* state, MotionGeometryTrianglesHandle geometryTriangles );

PagingMode Global_getDemandLoadMode( CanonicalState* state );
}  // namespace cort

namespace Megakernel {

// SGP: Megakernel namespace should not be used here. Refactor.

char* Buffer_getElementAddress1dFromId_linear( cort::CanonicalState* state, unsigned int bufferid, unsigned int eltSize, cort::uint64 x );
char* Buffer_getElementAddress2dFromId_linear( cort::CanonicalState* state,
                                               unsigned int          bufferid,
                                               unsigned int          eltSize,
                                               cort::uint64          x,
                                               cort::uint64          y );
char* Buffer_getElementAddress3dFromId_linear( cort::CanonicalState* state,
                                               unsigned int          bufferid,
                                               unsigned int          eltSize,
                                               cort::uint64          x,
                                               cort::uint64          y,
                                               cort::uint64          z );
char* Buffer_decodeBufferHeader1d_linear( cort::CanonicalState* state, cort::Buffer buffer, unsigned int eltSize, cort::uint64 x );
char* Buffer_decodeBufferHeader2d_linear( cort::CanonicalState* state, cort::Buffer buffer, unsigned int eltSize, cort::uint64 x, cort::uint64 y );
char* Buffer_decodeBufferHeader3d_linear( cort::CanonicalState* state,
                                          cort::Buffer          buffer,
                                          unsigned int          eltSize,
                                          cort::uint64          x,
                                          cort::uint64          y,
                                          cort::uint64          z );
}  // namespace Megakernel
