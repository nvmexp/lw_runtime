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

#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/RTX/RTXRuntime.h>

using namespace cort;

// Dummy function used to assist type matching during link. If the source module
// has non-dotted types these will not be matched to the corresponding type in
// the destination except if referenced by some linked global. So we place a
// function in the common runtime that takes all the types that need to be matched
// as parameters. A corresponding function declaration needs to be placed in the
// source module.
//
// The signature needs to match the one found in C14n::addTypeLinkageFunction().
//
// TODO: It might be better to use global variables, e.g. link_CanonicalState.
// if different types are needed by different link steps.
extern "C" void optixi_linkTypes( CanonicalState* state, cort::float4* )
{
}

// decl for llvm... to make type cort::OptixRay known to llvm
extern "C" cort::OptixRay optixi_getLwrrentRay( CanonicalState* state );

extern "C" float optixi_getLwrrentTime( CanonicalState* state )
{
    return Runtime_getLwrrentTime( state );
}

extern "C" float optixi_getLwrrentTmax( CanonicalState* state )
{
    return TraceFrame_getLwrrentTmax( state );
}

extern "C" void optixi_getAabbRequest( CanonicalState* state, cort::AabbRequest* req )
{
    *req = Global_getAabbRequest( state );
}

extern "C" unsigned int optixi_getExceptionCode( CanonicalState* state )
{
    return Exception_getCode( state );
}

extern "C" unsigned int optixi_getExceptionDetail( CanonicalState* state, unsigned int which )
{
    return Exception_getDetail( state, which );
}

extern "C" uint64 optixi_getExceptionDetail64( CanonicalState* state, unsigned int which )
{
    return Exception_getDetail64( state, which );
}

extern "C" cort::FrameStatus* optixi_getFrameStatus( CanonicalState* state )
{
    return Global_getStatusReturn( state );
}

extern "C" uint3 optixi_getLaunchDim( CanonicalState* state )
{
    return Global_getLaunchDim( state );
}

extern "C" uint3 optixi_getLaunchIndex( CanonicalState* state )
{
    return Raygen_getLaunchIndex( state );
}

extern "C" unsigned int optixi_getSubframeIndex( CanonicalState* state )
{
    return Global_getSubframeIndex( state );
}

extern "C" unsigned short optixi_getEntryPointIndex( CanonicalState* state )
{
    return Global_getEntry( state );
}

extern "C" unsigned int optixi_getPrimitiveArgToComputeAABB( CanonicalState* state )
{
    return AABB_getPrimitive( state );
}

extern "C" unsigned int optixi_getPrimitiveIndex( CanonicalState* state )
{
    return RTX_getPrimitiveIdx();
}

extern "C" unsigned int optixi_getPrimitiveArgToIntersect( CanonicalState* state )
{
    return RTX_getPrimitiveIdx();
}

extern "C" unsigned int optixi_getPrimitiveIndexOffset( CanonicalState* state, unsigned int GIOffset )
{
    GeometryInstanceHandle gi = LexicalScope_colwertToGeometryInstanceHandle( state, {GIOffset} );
    GeometryHandle         g  = GeometryInstance_getGeometry( state, gi );
    return Geometry_getPrimitiveIndexOffset( state, g );
}

extern "C" bool optixi_isTriangleHit( CanonicalState* state )
{
    return TraceFrame_isTriangleHit( state );
}

extern "C" bool optixi_isTriangleHitBackFace( CanonicalState* state )
{
    return TraceFrame_isTriangleHitBackFace( state );
}

extern "C" bool optixi_isTriangleHitFrontFace( CanonicalState* state )
{
    return TraceFrame_isTriangleHitFrontFace( state );
}

// RTX only function, there is no MK pendant
extern "C" unsigned int optixi_getInstanceFlags( CanonicalState* state )
{
    return RTX_getInstanceFlags();
}

extern "C" unsigned int optixi_getRayFlags( CanonicalState* state )
{
    return RTX_getRayFlags();
}

extern "C" unsigned int optixi_getRayMask( CanonicalState* state )
{
    return RTX_getRayMask();
}

extern "C" unsigned int optixi_getLowestGroupChildIndex( CanonicalState* state )
{
    return RTX_getInstanceIndex();
}

// TODO (OP-1113): NO POINTERS!
extern "C" float* optixi_getAABBArgToComputeAABB( CanonicalState* state )
{
    return AABB_getPtr( state );
}

extern "C" void optixi_handleTransformNode( CanonicalState* state )
{
    Runtime_visitTransformNode( state );
}

extern "C" void optixi_ignoreIntersection( CanonicalState* state )
{
    TraceFrame_ignoreIntersection( state );
}

extern "C" void optixi_intersectChild( CanonicalState* state, unsigned int child )
{
    GraphNodeHandle ch = Selector_getChildNode( state, child );
    Runtime_intersectNode( state, ch );
}

extern "C" void optixi_intersectPrimitive( CanonicalState* state, unsigned int GIOffset, unsigned int primitiveIndex )
{
    GeometryInstanceHandle GI = LexicalScope_colwertToGeometryInstanceHandle( state, {GIOffset} );
    Runtime_intersectPrimitive( state, GI, primitiveIndex );
}

extern "C" void optixi_intersectNode( CanonicalState* state, unsigned int child )
{
    GraphNodeHandle ch = LexicalScope_colwertToGraphNodeHandle( state, {child} );
    Runtime_intersectNode( state, ch );
}

extern "C" int optixi_isPrintingEnabled( CanonicalState* state )
{
    return Raygen_isPrintEnabledAtLwrrentLaunchIndex( state );
}

extern "C" void optixi_setLwrrentAcceleration( CanonicalState* state )
{
    Runtime_setLwrrentAcceleration( state );
}

extern "C" cort::float4 optixi_transformTuple( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w )
{
    return Runtime_applyLwrrentTransforms( state, transform_kind, x, y, z, w );
}

extern "C" Matrix4x4 optixi_getTransform( CanonicalState* state, unsigned int transform_kind )
{
    return Runtime_getTransform( state, transform_kind );
}

extern "C" void optixi_profileEvent( CanonicalState* state, unsigned int idx )
{
    Profile_event( state, idx );
}

extern "C" void optixi_throw( CanonicalState* state, unsigned int code )
{
    Runtime_throwException( state, code );
}

extern "C" void optixi_terminateRay( CanonicalState* state )
{
    TerminateRay_terminate( state );
}

extern "C" bool optixi_terminateRayRequested( CanonicalState* state )
{
    return TerminateRay_isRequested( state );
}

extern "C" unsigned int optixi_terminateRayUnwind( CanonicalState* state )
{
    // Ignore the return value. It is only used by MegakernelES, and there we add
    // all uses explicitly in TerminateRayInstrumenter. Beforehand, we expect no
    // call to this function to have a use (there's an assertion for this).
    TerminateRay_unwind( state );
    return 0;
}

extern "C" cort::float2 optixi_getTriangleBarycentrics( CanonicalState* state )
{
    return RTX_getTriangleBarycentrics();
}

// Triangle data access

extern "C" int optixi_getGeometryTrianglesVertexBufferID( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getVertexBufferId( state, (GeometryTrianglesHandle)lwrrGeometry );
}

extern "C" long long optixi_getGeometryTrianglesVertexBufferOffset( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getVertexBufferOffset( state, (GeometryTrianglesHandle)lwrrGeometry );
}

extern "C" unsigned long long optixi_getGeometryTrianglesVertexBufferStride( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getVertexBufferStride( state, (GeometryTrianglesHandle)lwrrGeometry );
}

extern "C" int optixi_getGeometryTrianglesIndexBufferID( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getIndexBufferId( state, (GeometryTrianglesHandle)lwrrGeometry );
}

extern "C" long long optixi_getGeometryTrianglesIndexBufferOffset( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getIndexBufferOffset( state, (GeometryTrianglesHandle)lwrrGeometry );
}

extern "C" unsigned long long optixi_getGeometryTrianglesIndexBufferStride( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return GeometryTriangles_getIndexBufferStride( state, (GeometryTrianglesHandle)lwrrGeometry );
}

// Motion triangle data access.

extern "C" unsigned long long optixi_getMotionGeometryTrianglesVertexBufferMotionStride( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getVertexBufferMotionStride( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" int optixi_getMotionGeometryTrianglesMotionNumIntervals( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getMotionNumIntervals( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" float optixi_getMotionGeometryTrianglesTimeBegin( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getTimeBegin( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" float optixi_getMotionGeometryTrianglesTimeEnd( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getTimeEnd( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" int optixi_getMotionGeometryTrianglesMotionBorderModeBegin( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getMotionBorderModeBegin( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" int optixi_getMotionGeometryTrianglesMotionBorderModeEnd( CanonicalState* state )
{
    GeometryHandle lwrrGeometry = getGeometryHandle();
    return MotionGeometryTriangles_getMotionBorderModeEnd( state, (MotionGeometryTrianglesHandle)lwrrGeometry );
}

extern "C" unsigned char RTX_getHitKind();
extern "C" unsigned char optixi_getDeferredAttributeKind( CanonicalState* state )
{
    return RTX_getHitKind();
}
