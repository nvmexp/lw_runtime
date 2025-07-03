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

#include <ExelwtionStrategy/CommonRuntime.h>
#include <ExelwtionStrategy/FrameStatus.h>

// These can be used on the CPU for debugging
extern "C" void exit( int code );

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

// TODO (OP-1113): NO POINTERS!! Make the aabb argument not a pointer.  This means we should
// probably return something that looks like a float6?
extern "C" void optixi_computeGeometryInstanceAABB( CanonicalState* state,
                                                    unsigned int    GIOffset,
                                                    unsigned int    primitive,
                                                    unsigned int    motionStep,
                                                    float*          aabb )
{
    GeometryInstanceHandle gi = LexicalScope_colwertToGeometryInstanceHandle( state, {GIOffset} );
    Runtime_computeGeometryInstanceAABB( state, gi, primitive, motionStep, aabb );
}

extern "C" void optixi_computeGroupChildAABB( CanonicalState* state, unsigned int groupOffset, unsigned int child, float* aabb )
{
    AbstractGroupHandle grp = LexicalScope_colwertToAbstractGroupHandle( state, {groupOffset} );
    Runtime_computeGroupChildAABB( state, grp, child, (Aabb*)aabb );
}

extern "C" void optixi_gatherMotionAABBs( CanonicalState* state, unsigned int groupOffset, float* aabb )
{
    AbstractGroupHandle grp = LexicalScope_colwertToAbstractGroupHandle( state, {groupOffset} );
    Runtime_gatherMotionAABBs( state, grp, (Aabb*)aabb );
}

extern "C" cort::OptixRay optixi_getLwrrentRay( CanonicalState* state )
{
    return Runtime_getLwrrentRay( state );
}

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
    return Intersect_getPrimitive( state );
}

extern "C" unsigned int optixi_getPrimitiveArgToIntersect( CanonicalState* state )
{
    return Intersect_getPrimitive( state );
}

extern "C" unsigned int optixi_getPrimitiveIndexOffset( CanonicalState* state, unsigned int GIOffset )
{
    GeometryInstanceHandle gi = LexicalScope_colwertToGeometryInstanceHandle( state, {GIOffset} );
    GeometryHandle         g  = GeometryInstance_getGeometry( state, gi );
    return Geometry_getPrimitiveIndexOffset( state, g );
}

extern "C" unsigned int optixi_getMotionIndexArgToComputeAABB( CanonicalState* state )
{
    return AABB_getMotionIndex( state );
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

extern "C" bool optixi_isPotentialIntersection( CanonicalState* state, float t )
{
    return TraceFrame_hasPotentialIntersection( state, t );
}

extern "C" int optixi_isPrintingEnabled( CanonicalState* state )
{
    return Raygen_isPrintEnabledAtLwrrentLaunchIndex( state );
}

extern "C" bool optixi_reportIntersection( CanonicalState* state, unsigned int matlIndex, unsigned char hitkind )
{
    return TraceFrame_reportIntersection( state, matlIndex );
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
    // We set the code here, and make sure that all calls to "optixi_throw" are
    // properly guarded by try-catch blocks (see RuntimeException.cpp).
    Exception_setCode( state, code );
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
