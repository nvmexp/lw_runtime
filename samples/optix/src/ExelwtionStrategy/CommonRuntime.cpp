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

/************************************************************************************************************
 * Runtime state access and manipulation
 ************************************************************************************************************/

/*
 * Runtime methods
 */

CORT_OVERRIDABLE
void cort::Runtime_trace( CanonicalState* state,
                          unsigned int    topOffset,
                          float           ox,
                          float           oy,
                          float           oz,
                          float           dx,
                          float           dy,
                          float           dz,
                          unsigned int    rayType,
                          float           tmin,
                          float           tmax,
                          float           time,
                          unsigned int    hasTime,
                          char*           payload )
{
    Profile_count( state, PROFILE_TRACE_COUNTER, 1 );

    // Set up trace frame and save old one
    TraceFrame  newTrace;
    TraceFrame* oldFrame = TraceFrame_push( state, &newTrace, rayType, tmax, time, hasTime, payload );

    // Set up ray frame and save old one.
    RayFrame  newRay;
    RayFrame* oldRay = RayFrame_push( state, &newRay, float3( ox, oy, oz ), float3( dx, dy, dz ), tmin );

    // If exceptions are enabled, make sure the ray is okay. Otherwise this is a no-op.
    Exception_checkIlwalidRay( state, ox, oy, oz, dx, dy, dz, rayType, tmin, tmax );

    // Store some information for handling calls of rtTerminateRay and save the
    // old state. The actual trace call site id is inserted during
    // canonicalization so that each saveState call has a unique identifier.
    TerminateRay_saveState( state, -1 /* traceCallSiteID */ );

    // Ilwoke traversal
    GraphNodeHandle node = LexicalScope_colwertToGraphNodeHandle( state, {topOffset} );
    Runtime_traverse( state, node );

    // Mark this place in the code as the "return" place of traversal.
    TerminateRay_unwindFinished( state, -1 /* traceCallSiteID */ );

    // Ilwoke shading (closest_hit or miss)
    Runtime_shade( state );

    // Restore old frames
    RayFrame_pop( state, oldRay );
    TraceFrame_pop( state, oldFrame );
}

CORT_OVERRIDABLE
void cort::Runtime_trace_global_payload( CanonicalState* state,
                                         unsigned int    topOffset,
                                         float           ox,
                                         float           oy,
                                         float           oz,
                                         float           dx,
                                         float           dy,
                                         float           dz,
                                         unsigned int    rayType,
                                         float           tmin,
                                         float           tmax,
                                         float           time,
                                         unsigned int    hasTime )
{
    char* payload = (char*)TraceFrame_getPayloadAddress( state );
    Runtime_trace( state, topOffset, ox, oy, oz, dx, dy, dz, rayType, tmin, tmax, time, hasTime, payload );
}

CORT_OVERRIDABLE
void cort::Runtime_traverse( CanonicalState* state, GraphNodeHandle node )
{
    // Ilwoke traversal
    Traversal_set( state, node );
    ProgramHandle traverse = GraphNode_getTraverse( state, node );
    Runtime_ilwokeProgram( state, traverse, optix::ST_NODE_VISIT, GraphNode_colwertToLexicalScopeHandle( state, node ), {0} );
}

CORT_OVERRIDABLE
void cort::Runtime_shade( CanonicalState* state )
{
    // Get the necessary data
    // Why not forward the rayType from Runtime_trace?
    // AM: This is not a good idea.
    // In the current configuration getting the raytype is just a load.
    // Notice that between the definition of the rayType in Runtime_trace and here there is a continuation call (Runtime_traverse).
    // If we were to pass the rayType as a parameter it would have to be saved on the stack and reloaded back (1 store, 1 load).

    unsigned int rayType = TraceFrame_getRayType( state );

    // Determine closest hit or miss
    GeometryInstanceHandle instance;
    ProgramHandle          shadeProgram;
    MaterialHandle         material = TraceFrame_getCommittedMaterial( state );
    if( material )
    {
        // Hit
        instance     = TraceFrame_getCommittedInstance( state );
        shadeProgram = Material_getCHProgram( state, material, rayType );

        // Load the current hit from the committed hit if there is a
        // closest-hit program
        if( shadeProgram )
            TraceFrame_restoreHit( state );
    }
    else
    {
        // Miss
        instance                      = {0};
        GlobalScopeHandle globalScope = GlobalScope_get( state );
        shadeProgram                  = GlobalScope_getMiss( state, globalScope, rayType );
    }

    // Ilwoke the program
    Runtime_ilwokeProgram( state, shadeProgram, optix::ST_CLOSEST_HIT,
                           GeometryInstance_colwertToLexicalScopeHandle( state, instance ),
                           Material_colwertToLexicalScopeHandle( state, material ) );
}

CORT_OVERRIDABLE
void cort::Runtime_intersectNode( CanonicalState* state, GraphNodeHandle node )
{
    // Save old values
    GraphNodeHandle oldNode = Traversal_getLwrrentNode( state );

    // Set up state
    Traversal_set( state, node );

    // Ilwoke traversal
    Runtime_traverse( state, node );

    if( TerminateRay_isRequested( state ) )
    {
        // There was a call to rtTerminateRay and we have not yet reached the start
        // place of the traversal, so we have to keep unwinding.
        // NOTE: In the compile stage, the code is modified to unwind one more level
        //       of the call stack, i.e., this return will not only return from this
        //       function but return from the entire sub state, unless this part of
        //       the trace call stack is inlined. If modified, this will result in
        //       an additional return edge in the call graph, which in turn will
        //       ensure that live values are properly stored and reloaded.
        TerminateRay_unwind( state );
        return;
    }

    // Restore old state
    Traversal_set( state, oldNode );
}

CORT_OVERRIDABLE
void cort::Runtime_intersectPrimitive( CanonicalState* state, GeometryInstanceHandle gi, unsigned int primitiveIndex )
{
    GeometryHandle geometry = GeometryInstance_getGeometry( state, gi );
    TraceFrame_setLwrrentInstance( state, gi );
    Intersect_set( state, primitiveIndex );
    ProgramHandle intersect = Geometry_getIntersectProgram( state, geometry );
    Runtime_ilwokeProgram( state, intersect, optix::ST_INTERSECTION, GeometryInstance_colwertToLexicalScopeHandle( state, gi ),
                           Geometry_colwertToLexicalScopeHandle( state, geometry ) );

    if( TerminateRay_isRequested( state ) )
    {
        // There was a call to rtTerminateRay and we have not yet reached the start
        // place of the traversal, so we have to keep unwinding.
        // NOTE: In the compile stage, the code is modified to unwind one more level
        //       of the call stack, i.e., this return will not only return from this
        //       function but return from the entire sub state, unless this part of
        //       the trace call stack is inlined. If modified, this will result in
        //       an additional return edge in the call graph, which in turn will
        //       ensure that live values are properly stored and reloaded.
        TerminateRay_unwind( state );
        return;
    }
}

CORT_OVERRIDABLE
void cort::Runtime_computeGeometryInstanceAABB( CanonicalState*        state,
                                                GeometryInstanceHandle gi,
                                                unsigned int           primitiveIndex,
                                                unsigned int           motionStep,
                                                float*                 aabb )
{
    // Forge a trace frame just to contain the GeometryInstance info
    TraceFrame  newTrace;
    TraceFrame* oldFrame = TraceFrame_push( state, &newTrace, 0, 0, 0, 1, 0 );
    TraceFrame_setLwrrentInstance( state, gi );

    // Set up the AABB callframe
    GeometryHandle geometry        = GeometryInstance_getGeometry( state, gi );
    unsigned int   offsetPrimitive = primitiveIndex + Geometry_getPrimitiveIndexOffset( state, geometry );
    AABB_set( state, aabb, offsetPrimitive, motionStep );

    // Ilwoke the program
    ProgramHandle program = Geometry_getAABBProgram( state, geometry );
    Runtime_ilwokeProgram( state, program, optix::ST_BOUNDING_BOX, GeometryInstance_colwertToLexicalScopeHandle( state, gi ),
                           Geometry_colwertToLexicalScopeHandle( state, geometry ) );

    // Restore the old frame
    TraceFrame_pop( state, oldFrame );
}

CORT_OVERRIDABLE
void cort::Runtime_computeGroupChildAABB( CanonicalState* state, AbstractGroupHandle grp, unsigned int child, Aabb* aabb )
{
    // Find the child
    BufferId            children = AbstractGroup_getChildren( state, grp );
    const unsigned int  eltSize  = sizeof( unsigned int );
    char                stackTemp[eltSize];
    const unsigned int* addr =
        reinterpret_cast<const unsigned int*>( Buffer_getElementAddress1dFromId( state, children, eltSize, stackTemp, child ) );
    GraphNodeHandle childNode = LexicalScope_colwertToGraphNodeHandle( state, {*addr} );

    // Set traversal state for the current node pointer
    Traversal_set( state, childNode );

    // Set up bounding box as invalid (probably not necessary, but paranoid).
    GeneralBB genbb;
    genbb.ilwalidate();

    // Visit the child
    Runtime_computeGraphNodeGeneralBB( state, childNode, &genbb );

    // Colwert and write out the bounding box
    aabb->set( genbb );
}

CORT_OVERRIDABLE
void cort::Runtime_computeGraphNodeGeneralBB( CanonicalState* state, GraphNodeHandle graphNode, GeneralBB* genbb )
{
    // Save the old node
    GraphNodeHandle oldNode = Traversal_getLwrrentNode( state );

    // Set up the AABB callframe. With a motion index of zero, we will
    // retrieve the union across all motion steps.
    AABB_set( state, (float*)genbb, 0, 0 );

    // Ilwoke the bounding box  program on the graph node
    Traversal_set( state, graphNode );
    ProgramHandle program = GraphNode_getBBProgram( state, graphNode );
    Runtime_ilwokeProgram( state, program, optix::ST_BOUNDING_BOX,
                           GraphNode_colwertToLexicalScopeHandle( state, graphNode ), {0} );

    // Restore the old node
    Traversal_set( state, oldNode );
}


CORT_OVERRIDABLE
void cort::Runtime_gatherMotionAABBs( CanonicalState* state, AbstractGroupHandle grp, Aabb* aabbs )
{
    // Set up the AABB callframe. With a motion index of ~0, we will
    // retrieve individual aabbs for all motion steps.
    AABB_set( state, (float*)aabbs, 0, ~0 );

    // Ilwoke the bounding box  program on the graph node
    Traversal_set( state, {grp} );
    ProgramHandle program = GraphNode_getBBProgram( state, {grp} );
    Runtime_ilwokeProgram( state, program, optix::ST_BOUNDING_BOX, GraphNode_colwertToLexicalScopeHandle( state, {grp} ), {0} );
}

CORT_OVERRIDABLE
void cort::Runtime_setLwrrentAcceleration( CanonicalState* state )
{
    AbstractGroupHandle g     = GraphNode_colwertToAbstractGroupHandle( state, Traversal_getLwrrentNode( state ) );
    AccelerationHandle  accel = AbstractGroup_getAcceleration( state, g );
    Scopes_setScope2( state, Acceleration_colwertToLexicalScopeHandle( state, accel ) );
}

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

CORT_OVERRIDABLE
float4 cort::Runtime_applyLwrrentTransforms( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w )
{
    // Get Transform information
    unsigned char transformDepth = TraceFrame_getLwrrentTransformDepth( state );

    const bool         ilw_transpose = transform_kind & RT_INTERNAL_ILWERSE_TRANSPOSE;
    const unsigned int kind          = transform_kind & ~RT_INTERNAL_ILWERSE_TRANSPOSE;

    signed char begin, end, inc;
    if( kind == RT_WORLD_TO_OBJECT )
    {
        begin = 0;
        end   = transformDepth;
        inc   = 1;
    }
    else
    {
        begin = transformDepth - 1;
        end   = -1;
        inc   = -1;
    }

    float v0 = x;
    float v1 = y;
    float v2 = z;
    float v3 = w;
    for( char depth = begin; depth != end; depth += inc )
    {
        TransformHandle transform = TraceFrame_getLwrrentTransformByDepth( state, depth );
        Matrix4x4       matrix;
        if( kind == RT_WORLD_TO_OBJECT )
            matrix = ilw_transpose ? Transform_getMatrixAtLwrrentTime( state, transform ) :
                                     Transform_getIlwMatrixAtLwrrentTime( state, transform );
        else
            matrix = ilw_transpose ? Transform_getIlwMatrixAtLwrrentTime( state, transform ) :
                                     Transform_getMatrixAtLwrrentTime( state, transform );

        float ta, tb, tc, td;
        if( ilw_transpose )
        {
            ta = matrix.matrix[0][0] * v0;
            ta += matrix.matrix[1][0] * v1;
            ta += matrix.matrix[2][0] * v2;
            ta += matrix.matrix[3][0] * v3;
            tb = matrix.matrix[0][1] * v0;
            tb += matrix.matrix[1][1] * v1;
            tb += matrix.matrix[2][1] * v2;
            tb += matrix.matrix[3][1] * v3;
            tc = matrix.matrix[0][2] * v0;
            tc += matrix.matrix[1][2] * v1;
            tc += matrix.matrix[2][2] * v2;
            tc += matrix.matrix[3][2] * v3;
            td = matrix.matrix[0][3] * v0;
            td += matrix.matrix[1][3] * v1;
            td += matrix.matrix[2][3] * v2;
            td += matrix.matrix[3][3] * v3;
        }
        else
        {
            ta = matrix.matrix[0][0] * v0;
            ta += matrix.matrix[0][1] * v1;
            ta += matrix.matrix[0][2] * v2;
            ta += matrix.matrix[0][3] * v3;
            tb = matrix.matrix[1][0] * v0;
            tb += matrix.matrix[1][1] * v1;
            tb += matrix.matrix[1][2] * v2;
            tb += matrix.matrix[1][3] * v3;
            tc = matrix.matrix[2][0] * v0;
            tc += matrix.matrix[2][1] * v1;
            tc += matrix.matrix[2][2] * v2;
            tc += matrix.matrix[2][3] * v3;
            td = matrix.matrix[3][0] * v0;
            td += matrix.matrix[3][1] * v1;
            td += matrix.matrix[3][2] * v2;
            td += matrix.matrix[3][3] * v3;
        }

        v0 = ta;
        v1 = tb;
        v2 = tc;
        v3 = td;
    }

    return float4( v0, v1, v2, v3 );
}

CORT_OVERRIDABLE
float4 cort::Runtime_applyLwrrentTransforms_atMostOne( CanonicalState* state, unsigned int transform_kind, float x, float y, float z, float w )
{
    // Get Transform information
    bool hasTransform = TraceFrame_getLwrrentTransformDepth( state ) > 0;

    const bool         ilw_transpose = transform_kind & RT_INTERNAL_ILWERSE_TRANSPOSE;
    const unsigned int kind          = transform_kind & ~RT_INTERNAL_ILWERSE_TRANSPOSE;

    float v0 = x;
    float v1 = y;
    float v2 = z;
    float v3 = w;
    if( hasTransform )
    {
        TransformHandle transform = TraceFrame_getLwrrentTransformByDepth( state, 0 );
        Matrix4x4       matrix;
        if( kind == RT_WORLD_TO_OBJECT )
            matrix = ilw_transpose ? Transform_getMatrixAtLwrrentTime( state, transform ) :
                                     Transform_getIlwMatrixAtLwrrentTime( state, transform );
        else
            matrix = ilw_transpose ? Transform_getIlwMatrixAtLwrrentTime( state, transform ) :
                                     Transform_getMatrixAtLwrrentTime( state, transform );

        float ta, tb, tc, td;
        if( ilw_transpose )
        {
            ta = matrix.matrix[0][0] * v0;
            ta += matrix.matrix[1][0] * v1;
            ta += matrix.matrix[2][0] * v2;
            ta += matrix.matrix[3][0] * v3;
            tb = matrix.matrix[0][1] * v0;
            tb += matrix.matrix[1][1] * v1;
            tb += matrix.matrix[2][1] * v2;
            tb += matrix.matrix[3][1] * v3;
            tc = matrix.matrix[0][2] * v0;
            tc += matrix.matrix[1][2] * v1;
            tc += matrix.matrix[2][2] * v2;
            tc += matrix.matrix[3][2] * v3;
            td = matrix.matrix[0][3] * v0;
            td += matrix.matrix[1][3] * v1;
            td += matrix.matrix[2][3] * v2;
            td += matrix.matrix[3][3] * v3;
        }
        else
        {
            ta = matrix.matrix[0][0] * v0;
            ta += matrix.matrix[0][1] * v1;
            ta += matrix.matrix[0][2] * v2;
            ta += matrix.matrix[0][3] * v3;
            tb = matrix.matrix[1][0] * v0;
            tb += matrix.matrix[1][1] * v1;
            tb += matrix.matrix[1][2] * v2;
            tb += matrix.matrix[1][3] * v3;
            tc = matrix.matrix[2][0] * v0;
            tc += matrix.matrix[2][1] * v1;
            tc += matrix.matrix[2][2] * v2;
            tc += matrix.matrix[2][3] * v3;
            td = matrix.matrix[3][0] * v0;
            td += matrix.matrix[3][1] * v1;
            td += matrix.matrix[3][2] * v2;
            td += matrix.matrix[3][3] * v3;
        }

        v0 = ta;
        v1 = tb;
        v2 = tc;
        v3 = td;
    }

    return float4( v0, v1, v2, v3 );
}

CORT_OVERRIDABLE
float cort::Runtime_getLwrrentTime( CanonicalState* state )
{
    return cort::TraceFrame_getLwrrentRayTime( state );
}

CORT_OVERRIDABLE
Matrix4x4 cort::Runtime_getTransform( CanonicalState* state, unsigned int transform_kind )
{
    unsigned char transformDepth = TraceFrame_getLwrrentTransformDepth( state );

    signed char begin, end, inc;
    if( transform_kind == RT_WORLD_TO_OBJECT )
    {
        begin = 0;
        end   = transformDepth;
        inc   = 1;
    }
    else
    {
        begin = transformDepth - 1;
        end   = -1;
        inc   = -1;
    }

    Matrix4x4 result;
    // Initialize to the identity matrix.
    result.matrix[0][0] = 1.0f;
    result.matrix[0][1] = 0.0f;
    result.matrix[0][2] = 0.0f;
    result.matrix[0][3] = 0.0f;

    result.matrix[1][0] = 0.0f;
    result.matrix[1][1] = 1.0f;
    result.matrix[1][2] = 0.0f;
    result.matrix[1][3] = 0.0f;

    result.matrix[2][0] = 0.0f;
    result.matrix[2][1] = 0.0f;
    result.matrix[2][2] = 1.0f;
    result.matrix[2][3] = 0.0f;

    result.matrix[3][0] = 0.0f;
    result.matrix[3][1] = 0.0f;
    result.matrix[3][2] = 0.0f;
    result.matrix[3][3] = 1.0f;

    for( char depth = begin; depth != end; depth += inc )
    {
        TransformHandle transform = TraceFrame_getLwrrentTransformByDepth( state, depth );
        Matrix4x4 matrix = ( transform_kind == RT_WORLD_TO_OBJECT ) ? Transform_getIlwMatrixAtLwrrentTime( state, transform ) :
                                                                      Transform_getMatrixAtLwrrentTime( state, transform );

        Matrix4x4 lwrrent_result;

        // lwrrent_result = matrix * result;
        // TODO: Unroll these loops.
        for( int row = 0; row < 4; ++row )
            for( int column = 0; column < 4; ++column )
            {
                float output_element = 0.0f;
                for( int index = 0; index < 4; ++index )
                    output_element = lwca::fmad( matrix.matrix[row][index], result.matrix[index][column], output_element );
                lwrrent_result.matrix[row][column] = output_element;
            }
        result = lwrrent_result;
    }

    return result;
}

CORT_OVERRIDABLE
Matrix4x4 cort::Runtime_getTransform_atMostOne( CanonicalState* state, unsigned int transform_kind )
{
    bool hasTransform = TraceFrame_getLwrrentTransformDepth( state ) > 0;
    if( hasTransform )
    {
        TransformHandle transform = TraceFrame_getLwrrentTransformByDepth( state, 0 );
        Matrix4x4 matrix = ( transform_kind == RT_WORLD_TO_OBJECT ) ? Transform_getIlwMatrixAtLwrrentTime( state, transform ) :
                                                                      Transform_getMatrixAtLwrrentTime( state, transform );
        return matrix;
    }

    Matrix4x4 identity;
    identity.matrix[0][0] = 1.0f;
    identity.matrix[0][1] = 0.0f;
    identity.matrix[0][2] = 0.0f;
    identity.matrix[0][3] = 0.0f;

    identity.matrix[1][0] = 0.0f;
    identity.matrix[1][1] = 1.0f;
    identity.matrix[1][2] = 0.0f;
    identity.matrix[1][3] = 0.0f;

    identity.matrix[2][0] = 0.0f;
    identity.matrix[2][1] = 0.0f;
    identity.matrix[2][2] = 1.0f;
    identity.matrix[2][3] = 0.0f;

    identity.matrix[3][0] = 0.0f;
    identity.matrix[3][1] = 0.0f;
    identity.matrix[3][2] = 0.0f;
    identity.matrix[3][3] = 1.0f;

    return identity;
}

// Compile-time constant, defaults to most general case
CORT_OVERRIDABLE extern "C" int Motion_transforms_enabled = 1;

CORT_OVERRIDABLE
void cort::Runtime_visitTransformNode( CanonicalState* state )
{
    TransformHandle self = GraphNode_colwertToTransformHandle( state, Traversal_getLwrrentNode( state ) );

    // Motion blur: intersect current ray time against valid transform time range.
    // If the times do not intersect, then traversal stops here.

    BufferId motionDataBufferId = Transform_getMotionDataBufferId( state, self );
    if( Motion_transforms_enabled && motionDataBufferId != 0 )
    {
        const unsigned int eltSize = sizeof( float );
        char               stackTmp[eltSize];
        const float*       motionData =
            reinterpret_cast<const float*>( Buffer_getElementAddress1dFromId( state, motionDataBufferId, eltSize, stackTmp, 0 ) );

        const int mode0 = lwca::float_as_int( motionData[MDOFFSET_BEGIN_BORDER_MODE] );
        const int mode1 = lwca::float_as_int( motionData[MDOFFSET_END_BORDER_MODE] );

        const cort::InternalMotionBorderMode e_mode0 = (cort::InternalMotionBorderMode)mode0;
        const cort::InternalMotionBorderMode e_mode1 = (cort::InternalMotionBorderMode)mode1;

        if( e_mode0 == cort::MOTIONBORDERMODE_VANISH || e_mode1 == cort::MOTIONBORDERMODE_VANISH )
        {
            const float t0       = motionData[MDOFFSET_TIME_BEGIN];
            const float t1       = motionData[MDOFFSET_TIME_END];
            const float ray_time = Runtime_getLwrrentTime( state );

            if( ( e_mode0 == cort::MOTIONBORDERMODE_VANISH && ray_time < t0 )
                || ( e_mode1 == cort::MOTIONBORDERMODE_VANISH && ray_time > t1 ) )
                return;
        }
    }

    // End motion blur

    // Get the old ray
    float3 old_o    = RayFrame_getRayO( state );
    float3 old_d    = RayFrame_getRayD( state );
    float  old_tmin = RayFrame_getRayTmin( state );

    // Transform into the new frame
    const Matrix4x4 ilwM      = Transform_getIlwMatrixAtLwrrentTime( state, self );
    float4          to        = ilwM * float4( old_o, 1.0f );
    float3          origin    = float3( to.x / to.w, to.y / to.w, to.z / to.w );
    float3          direction = float3( ilwM * float4( old_d, 0.0f ) );
    // TODO: this renormalizes direction on each transform (and some
    // optix samples assume normalized rays ) Consider adding a property
    // to opt out of normalized rays
    float scale     = direction.length();
    float ilw_scale = 1.f / scale;
    direction *= ilw_scale;

    // Update state with new ray frame
    RayFrame  newRay;
    RayFrame* oldRay      = RayFrame_push( state, &newRay, origin, direction, old_tmin * scale );
    float     xformedTmax = scale * TraceFrame_getLwrrentTmax( state );
    TraceFrame_setLwrrentTmax( state, xformedTmax );

    // Update transform stack
    TraceFrame_pushTransform( state, self );

    // Visit child
    Runtime_intersectNode( state, Transform_getChild( state, self ) );

    // Pop transform stack
    TraceFrame_popTransform( state );

    // Restore old ray
    RayFrame_pop( state, oldRay );

    // Rescale committed tmax if we hit something inside Runtime_intersectNode
    float tmax = TraceFrame_getLwrrentTmax( state );
    if( tmax != xformedTmax )
    {
        TraceFrame_setCommittedTmax( state, ilw_scale * TraceFrame_getCommittedTmax( state ) );
    }

    // Rescale current tmax
    TraceFrame_setLwrrentTmax( state, ilw_scale * tmax );
}

/*
 * Fully generic "reference implementation" of variable get. This will
 * usually get specialized by the ES.
 */
CORT_OVERRIDABLE
__attribute__( ( const ) ) char* cort::Runtime_lookupVariableAddress( CanonicalState* state, unsigned short token, char* defaultValue )
{
    // Look on scope 0 (Program).
    LexicalScopeHandle program = Program_colwertToLexicalScopeHandle( state, ActiveProgram_get( state ) );
    if( char* data = LexicalScope_lookupVariable( state, program, token ) )
        return data;

    // Look on scope 1 (GraphNode or GI) if set.
    LexicalScopeHandle scope1 = Scopes_getScope1( state );
    if( scope1 )
    {
        if( char* data = LexicalScope_lookupVariable( state, scope1, token ) )
            return data;
    }

    // Look on scope 2 (Geometry or Material) if set.
    LexicalScopeHandle scope2 = Scopes_getScope2( state );
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

// Look up the requested variable in the program scope.
// NOTE: Only call this if it is statically known that the variable is in the program scope.
CORT_OVERRIDABLE
__attribute__( ( const ) ) char* cort::Runtime_lookupVariableAddress_ProgramScope( CanonicalState* state, unsigned short token )
{
    LexicalScopeHandle program = Program_colwertToLexicalScopeHandle( state, ActiveProgram_get( state ) );
    return LexicalScope_lookupVariableUnchecked( state, program, token );
}

// Look up the requested variable in the first scope (GraphNode or GeometryInstance).
// NOTE: Only call this if it is statically known that the variable is in scope1.
CORT_OVERRIDABLE
__attribute__( ( const ) ) char* cort::Runtime_lookupVariableAddress_Scope1( CanonicalState* state, unsigned short token )
{
    LexicalScopeHandle scope1 = Scopes_getScope1( state );
    return LexicalScope_lookupVariableUnchecked( state, scope1, token );
}

// Look up the requested variable in the second scope (Geometry or Material).
// NOTE: Only call this if it is statically known that the variable is in scope2.
CORT_OVERRIDABLE
__attribute__( ( const ) ) char* cort::Runtime_lookupVariableAddress_Scope2( CanonicalState* state, unsigned short token )
{
    LexicalScopeHandle scope2 = Scopes_getScope2( state );
    return LexicalScope_lookupVariableUnchecked( state, scope2, token );
}

// Look up the requested variable in the global scope.
// NOTE: Only call this if it is statically known that the variable is in the global scope.
CORT_OVERRIDABLE
__attribute__( ( const ) ) char* cort::Runtime_lookupVariableAddress_GlobalScope( CanonicalState* state, unsigned short token )
{
    LexicalScopeHandle globalScope = GlobalScope_colwertToLexicalScopeHandle( state, GlobalScope_get( state ) );
    return LexicalScope_lookupVariableUnchecked( state, globalScope, token );
}


CORT_OVERRIDABLE
unsigned int cort::Runtime_lookupIdVariableValue( CanonicalState* state, unsigned short token )
{
    char* ptr = Runtime_lookupVariableAddress( state, token, 0 );
    // If the lookup failed, we return ID 0.
    return ptr ? *reinterpret_cast<unsigned int*>( ptr ) : 0;
}

/*
 * Program invocation. Note that there are three slightly different
 * flavors - for ilwoking normal optix programs, for bound callable
 * programs, and for bindless callable programs.
 */

// Invocation of normal optix programs - with no arguments
CORT_OVERRIDABLE
void cort::Runtime_ilwokeProgram( CanonicalState* state, ProgramHandle program, optix::SemanticType callType, LexicalScopeHandle scope1, LexicalScopeHandle scope2 )
{
    // If exceptions are enabled, make sure the program id is okay. Otherwise this is a no-op.
    // TODO: We may remove this check to only check callable program ids.
    Exception_checkIlwalidProgramId( state, Program_getProgramID( state, program ) );

    unsigned short cpid = Program_getCanonicalProgramID( state, program );

    // Save old scopes and set new scopes
    Scopes oldScopes = Scopes_get( state );
    Scopes_set( state, scope1, scope2 );

    // Save old active program and set new one
    ProgramHandle oldActiveProgram = ActiveProgram_get( state );
    ActiveProgram_set( state, program );

    // Ilwoke function
    optixi_callIndirect_standard( state, cpid, callType );

    // Restore old program
    ActiveProgram_set( state, oldActiveProgram );

    // Restore old scopes
    Scopes_set( state, oldScopes );
}

// for bound callable programs. This function is cloned for each function signature.
CORT_OVERRIDABLE
RetPlaceholder cort::Runtime_ilwokeBoundCallableProgram_proto( CanonicalState* state, ProgramId pid, ArgPlaceholder args )
{
    // If exceptions are enabled, make sure the program id is okay. Otherwise this is a no-op.
    Exception_checkIlwalidProgramId( state, pid );

    ProgramHandle  program = Program_getProgramHandle( state, pid );
    unsigned short cpid    = Program_getCanonicalProgramID( state, program );

    // Save old scopes (since they are caller saved).  Note: scopes are not changed for bound callable programs
    Scopes oldScopes = Scopes_get( state );
    Scopes_set( state, oldScopes );  // Note: the ABI for optixi_callIndirect is to have a
                                     // set of the new scopes before the call followed by a
                                     // set of the old scopes after the call.  Even though we
                                     // aren't changing the scopes, we need to follow this
                                     // procedure.

    // Save old active program
    ProgramHandle oldActiveProgram = ActiveProgram_get( state );
    ActiveProgram_set( state, program );

    // Ilwoke function
    RetPlaceholder retval = optixi_callIndirect_stub( state, cpid, optix::ST_BOUND_CALLABLE_PROGRAM, args );

    // Restore old program
    ActiveProgram_set( state, oldActiveProgram );

    // Restore old scopes
    Scopes_set( state, oldScopes );

    // A bound callable program can call rtTerminateRay().
    if( TerminateRay_isRequested( state ) )
    {
        // There was a call to rtTerminateRay and we have not yet reached the start
        // place of the traversal, so we have to keep unwinding.
        // NOTE: In the compile stage, the code is modified to unwind one more level
        //       of the call stack, i.e., this return will not only return from this
        //       function but return from the entire sub state, unless this part of
        //       the trace call stack is inlined. If modified, this will result in
        //       an additional return edge in the call graph, which in turn will
        //       ensure that live values are properly stored and reloaded.
        TerminateRay_unwind( state );
        return retval;
    }

    return retval;
}

// For bindless callable programs. This function is cloned for each function signature.
CORT_OVERRIDABLE
RetPlaceholder cort::Runtime_ilwokeBindlessCallableProgram_proto( CanonicalState* state, ProgramId pid, ArgPlaceholder args )
{
    // If exceptions are enabled, make sure the program id is okay. Otherwise this is a no-op.
    Exception_checkIlwalidProgramId( state, pid );

    ProgramHandle  program = Program_getProgramHandle( state, pid );
    unsigned short cpid    = Program_getCanonicalProgramID( state, program );

    // Save old scopes and set them to null
    Scopes oldScopes = Scopes_get( state );
    Scopes_set( state, {0}, {0} );

    // Save old active program
    ProgramHandle oldActiveProgram = ActiveProgram_get( state );
    ActiveProgram_set( state, program );

    // Ilwoke function
    RetPlaceholder retval = optixi_callIndirect_stub( state, cpid, optix::ST_BINDLESS_CALLABLE_PROGRAM, args );

    // Restore old program
    ActiveProgram_set( state, oldActiveProgram );

    // Restore old scopes
    Scopes_set( state, oldScopes );

    // Note: a bindless callable program cannot participate in a terminate ray unwind because it cannot call rtTerminateRay()

    return retval;
}

CORT_OVERRIDABLE
OptixRay cort::Runtime_getLwrrentRay( CanonicalState* state )
{
    OptixRay ray;
    ray.origin    = RayFrame_getRayO( state );
    ray.direction = RayFrame_getRayD( state );
    ray.ray_type  = TraceFrame_getRayType( state );
    ray.tmin      = RayFrame_getRayTmin( state );
    ray.tmax      = TraceFrame_getLwrrentTmax( state );

    return ray;
}

CORT_OVERRIDABLE
OptixRay cort::Runtime_getLwrrentObjectSpaceRay( CanonicalState* state )
{
    bool hasTransform = TraceFrame_getLwrrentTransformDepth( state ) > 0;
    if( !hasTransform )
        // OptiX has an unfortunate behavior with respect to ray
        // normalization. The transform object normalizes the ray and some
        // of the samples depend on it. However, the incoming ray is not
        // normalized nor is it required to be. Therefore we only
        // normalize if there is an active transform.
        return Runtime_getLwrrentRay( state );

    float3 o = RayFrame_getRayO( state );
    float3 d = RayFrame_getRayD( state );

    OptixRay ray;
    ray.origin    = Runtime_applyLwrrentTransforms( state, RT_WORLD_TO_OBJECT, o.x, o.y, o.z, 1.f );
    ray.direction = Runtime_applyLwrrentTransforms( state, RT_WORLD_TO_OBJECT, d.x, d.y, d.z, 0.f );
    ray.tmin      = RayFrame_getRayTmin( state );
    ray.tmax      = TraceFrame_getLwrrentTmax( state );

    // Rescale tmin / tmax
    float scale     = ray.direction.length();
    float ilw_scale = 1.f / scale;
    ray.direction *= ilw_scale;
    ray.tmin *= scale;
    ray.tmax *= scale;

    return ray;
}

/*
 * Global methods
 */

CORT_OVERRIDABLE
void cort::Global_set( CanonicalState*    state,
                       FrameStatus*       statusReturn,
                       char*              objectRecords,
                       Buffer*            bufferTable,
                       TextureSampler*    textureTable,
                       ProgramHeader*     programTable,
                       TraversableHeader* traversableTable,
                       unsigned short*    dynamicVariableTable,
                       unsigned int       numBuffers,
                       unsigned int       numTextures,
                       unsigned int       numPrograms,
                       unsigned int       numTraversables,
                       const uint3&       launchDim,
                       const uint3&       printIndex,
                       bool               printEnabled,
                       unsigned short     dimensionality,
                       unsigned short     entry,
                       unsigned int       subframeIndex,
                       const AabbRequest& aabbRequest )
{
    state->global.statusReturn         = statusReturn;
    state->global.objectRecords        = objectRecords;
    state->global.bufferTable          = bufferTable;
    state->global.textureTable         = textureTable;
    state->global.programTable         = programTable;
    state->global.dynamicVariableTable = dynamicVariableTable;
    state->global.numBuffers           = numBuffers;
    state->global.numTextures          = numTextures;
    state->global.numPrograms          = numPrograms;
    state->global.launchDim            = launchDim;
    state->global.printIndex           = printIndex;
    state->global.printEnabled         = printEnabled;
    state->global.dimensionality       = dimensionality;
    state->global.entry                = entry;
    state->global.subframeIndex        = subframeIndex;
    state->global.aabbRequest          = aabbRequest;
}

CORT_OVERRIDABLE
FrameStatus* cort::Global_getStatusReturn( CanonicalState* state )
{
    return state->global.statusReturn;
}

CORT_OVERRIDABLE
uint64* cort::Global_getProfileData( CanonicalState* state )
{
    return state->global.profileData;
}

CORT_OVERRIDABLE
uint3 cort::Global_getLaunchDim( CanonicalState* state )
{
    return state->global.launchDim;
}

CORT_OVERRIDABLE
unsigned int cort::Global_getSubframeIndex( CanonicalState* state )
{
    return state->global.subframeIndex;
}

CORT_OVERRIDABLE
AabbRequest cort::Global_getAabbRequest( CanonicalState* state )
{
    return state->global.aabbRequest;
}

CORT_OVERRIDABLE
uint3 cort::Global_getPrintIndex( CanonicalState* state )
{
    return state->global.printIndex;
}

CORT_OVERRIDABLE
bool cort::Global_getPrintEnabled( CanonicalState* state )
{
    return state->global.printEnabled;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getDimensionality( CanonicalState* state )
{
    return state->global.dimensionality;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getEntry( CanonicalState* state )
{
    return state->global.entry;
}

CORT_OVERRIDABLE
long long cort::Global_getClocksBeforeTimeout( CanonicalState* state )
{
    return state->global.clocksBeforeTimeout;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getDeviceIndex( CanonicalState* state )
{
    return state->global.activeDeviceIndex;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getDeviceCount( CanonicalState* state )
{
    return state->global.activeDeviceCount;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getLoadBalancerStart( CanonicalState* state )
{
    return state->global.loadBalancerStart;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getLoadBalancerSize( CanonicalState* state )
{
    return state->global.loadBalancerSize;
}

CORT_OVERRIDABLE
unsigned short cort::Global_getLoadBalancerStride( CanonicalState* state )
{
    return state->global.loadBalancerStride;
}

CORT_OVERRIDABLE
char* cort::Global_getObjectRecord( CanonicalState* state, ObjectRecordOffset offset )
{
    return state->global.objectRecords + offset;
}

CORT_OVERRIDABLE
unsigned short* cort::Global_getDynamicVariableTable( CanonicalState* state, DynamicVariableTableOffset offset )
{
    return state->global.dynamicVariableTable + offset;
}

CORT_OVERRIDABLE
Buffer cort::Global_getBufferHeader( CanonicalState* state, unsigned int bufferid )
{
    return state->global.bufferTable[bufferid];
}

CORT_OVERRIDABLE
TextureSampler cort::Global_getTextureSamplerHeader( CanonicalState* state, unsigned int textureid )
{
    return state->global.textureTable[textureid];
}

CORT_OVERRIDABLE
ProgramHeader cort::Global_getProgramHeader( CanonicalState* state, unsigned int programid )
{
    return state->global.programTable[programid];
}

CORT_OVERRIDABLE
unsigned int cort::Global_getNumBuffers( CanonicalState* state )
{
    return state->global.numBuffers;
}

CORT_OVERRIDABLE
unsigned int cort::Global_getNumTextures( CanonicalState* state )
{
    return state->global.numTextures;
}

CORT_OVERRIDABLE
unsigned int cort::Global_getNumPrograms( CanonicalState* state )
{
    return state->global.numPrograms;
}

CORT_OVERRIDABLE
unsigned int cort::Global_getRayTypeCount( CanonicalState* state )
{
    return state->global.rayTypeCount;
}

// NOTE: Only used to fill exception detail, so no need to override or use
// replacement magic to force constant loads.
char* cort::Global_getBufferAddress( CanonicalState* state, unsigned short token )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return reinterpret_cast<char*>( state->global.bufferTable + bufferid );
}

CORT_OVERRIDABLE
char* cort::Global_getTextureSamplerAddress( CanonicalState* state, unsigned short token )
{
    unsigned int textureid = Runtime_lookupIdVariableValue( state, token );
    return Global_getTextureSamplerAddressFromId( state, textureid );
}

CORT_OVERRIDABLE
char* cort::Global_getProgramAddress( CanonicalState* state, unsigned short token )
{
    unsigned int programid = Runtime_lookupIdVariableValue( state, token );
    return Global_getProgramAddressFromId( state, programid );
}

CORT_OVERRIDABLE
char* cort::Global_getTextureSamplerAddressFromId( CanonicalState* state, unsigned int textureid )
{
    return reinterpret_cast<char*>( state->global.textureTable + textureid );
}

CORT_OVERRIDABLE
char* cort::Global_getProgramAddressFromId( CanonicalState* state, unsigned int programid )
{
    return reinterpret_cast<char*>( state->global.programTable + programid );
}

/*
 * Raygen methods
 */
CORT_OVERRIDABLE
void cort::Raygen_set( CanonicalState* state, uint3 launchIndex )
{
    state->raygen.launchIndex = launchIndex;
}

CORT_OVERRIDABLE
uint3 cort::Raygen_getLaunchIndex( CanonicalState* state )
{
    return state->raygen.launchIndex;
}

CORT_OVERRIDABLE
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

/*
 * Exception methods
 * These will usually be used like this:
 * if ( Exception_checkXXX( state, frameSize ) )
 * {
 *   Exception_setCode( state, RT_EXCEPTION_XXXXX );
 *   Exception_setDetail64( state, detail0, 0 );
 *   Exception_setDetail64( state, detail1, 1 );
 *   ...
 *   Exception_setDetail64( state, detailN, N );
 *   return Exception_throw( state );
 * }
 *
 * For usage examples see RuntimeExceptionInstrumenter.cpp.
 *
 * NOTE: This function is overridden in MegakernelRuntime.
 */
CORT_OVERRIDABLE
int cort::Exception_throw( CanonicalState* state )
{
    // Determine the current entry.
    const GlobalScopeHandle globalScope = GlobalScope_get( state );
    const unsigned short    entry       = Global_getEntry( state );

    // Determine the ProgramHandle of the raygen function (the top level frame).
    ProgramHandle raygenProgram = GlobalScope_getRaygen( state, globalScope, entry );

    // Determine the current ProgramHandle.
    ProgramHandle lwrProgram = ActiveProgram_get( state );

    // If we are not in the raygen program, unwind one level.
    if( lwrProgram != raygenProgram )
        return -1;  // Dummy value, unused.

    // Otherwise, the stack is cleared and we can now execute the exception program.

    // Determine the ProgramHandle of the exception program.
    const ProgramHandle exc = GlobalScope_getException( state, globalScope, entry );

    // Call the exception program directly from here.
    LexicalScopeHandle n = {0};
    Runtime_ilwokeProgram( state, exc, optix::ST_EXCEPTION, n, n );

    return -1;  // Dummy value, unused.
}

CORT_OVERRIDABLE
unsigned int cort::Exception_getCode( CanonicalState* state )
{
    return state->exception.code;
}

CORT_OVERRIDABLE
void cort::Exception_setCode( CanonicalState* state, unsigned int code )
{
    state->exception.code = code;
}

CORT_OVERRIDABLE
unsigned int cort::Exception_getDetail( CanonicalState* state, unsigned int which )
{
    // TODO: Assert that 'which' is <= 10.
    // We map indices 0,1,2 to 9,10,11 to prevent overlap for BUFFER_INDEX_OUT_OF_BOUNDS.
    unsigned int idx = which < 3 ? which + 9 : which;
    return static_cast<unsigned int>( state->exception.detail64[idx] );
}

CORT_OVERRIDABLE
uint64 cort::Exception_getDetail64( CanonicalState* state, unsigned int which )
{
    // TODO: Assert that 'which' is <= 10.
    return state->exception.detail64[which];
}
CORT_OVERRIDABLE
void cort::Exception_setDetail( CanonicalState* state, unsigned int detail, unsigned int which )
{
    // TODO: Assert that 'which' is <= 10.
    // We map indices 0,1,2 to 9,10,11 to prevent overlap for BUFFER_INDEX_OUT_OF_BOUNDS.
    unsigned int idx               = which < 3 ? which + 9 : which;
    state->exception.detail64[idx] = static_cast<uint64>( detail );
}

CORT_OVERRIDABLE
void cort::Exception_setDetail64( CanonicalState* state, uint64 detail64, unsigned int which )
{
    // TODO: Assert that 'which' is <= 10.
    state->exception.detail64[which] = detail64;
}

CORT_OVERRIDABLE
bool cort::Exception_checkStackOverflow( CanonicalState* state, unsigned int size )
{
    // By default there is no stack that we have to maintain manually.
    // MegakernelRuntime overrides this.
    return false;
}

// Check if the given id is valid or not.
// If valid, return 0.
// Otherwise, return the id error:
// 1 - "index into the id table is 0"
// 2 - "index into the id table is >= table_size"
// 3 - "referenced id is invalid (-1)"
CORT_OVERRIDABLE
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

CORT_OVERRIDABLE
unsigned int cort::Exception_checkBufferIdIlwalid( CanonicalState* state, unsigned int bufferId )
{
    return Exception_checkIdIlwalid( bufferId, Global_getNumBuffers( state ) );
}

CORT_OVERRIDABLE
unsigned int cort::Exception_checkTextureIdIlwalid( CanonicalState* state, unsigned int textureId )
{
    return Exception_checkIdIlwalid( textureId, Global_getNumTextures( state ) );
}

CORT_OVERRIDABLE
unsigned int cort::Exception_checkProgramIdIlwalid( CanonicalState* state, unsigned int programId )
{
    return Exception_checkIdIlwalid( programId, Global_getNumPrograms( state ) );
}

// This is a dummy function that serves to mark the (few) places where we need
// to check if a program id is invalid. This remains a no-op if the exception is
// not enabled. Otherwise, it is replaced by Exception_checkProgramIdIlwalid.
CORT_OVERRIDABLE
void cort::Exception_checkIlwalidProgramId( CanonicalState* state, unsigned int programId )
{
}

// This is a dummy function that serves to collect all the values to
// check if the ray is invalid. This is necessary because we split
// the values of the ray between the TraceFrame and the RayFrame.
// This remains a no-op if the exception is not enabled.
CORT_OVERRIDABLE
void cort::Exception_checkIlwalidRay( CanonicalState* state, float ox, float oy, float oz, float dx, float dy, float dz, unsigned int rayType, float tmin, float tmax )
{
}

/*
 * TerminateRay methods
 */

CORT_OVERRIDABLE
void cort::TerminateRay_initialize( CanonicalState* state )
{
    TerminateRay_setIsRequested( state, false );
}

CORT_OVERRIDABLE
void cort::TerminateRay_saveState( CanonicalState* state, int traceCallSiteID )
{
    // This is a dummy function to mark the place where a trace starts.
}

CORT_OVERRIDABLE
void cort::TerminateRay_unwindFinished( CanonicalState* state, int traceCallSiteID )
{
    // Termination is finished, so we set "isRequested" to false (not necessary
    // if no termination was requested, but that branch would be more expensive
    // than just setting the same value again).
    TerminateRay_setIsRequested( state, false );
}

CORT_OVERRIDABLE
void cort::TerminateRay_terminate( CanonicalState* state )
{
    TerminateRay_setIsRequested( state, true );
    TraceFrame_commitHit( state );  // Return value not required since we unwind.
}

CORT_OVERRIDABLE
unsigned int cort::TerminateRay_unwind( CanonicalState* state )
{
    // This is a dummy function to mark the places from where we need to unwind
    // further. The return value is only used in the optimized MegakernelES
    // variant (which overrides this function).
    return 0;
}

CORT_OVERRIDABLE
void cort::TerminateRay_setIsRequested( CanonicalState* state, bool isRequested )
{
    state->terminateRay.isRequested = isRequested;
}

CORT_OVERRIDABLE
bool cort::TerminateRay_isRequested( CanonicalState* state )
{
    return state->terminateRay.isRequested;
}

/*
 * Scopes methods
 */
CORT_OVERRIDABLE
void cort::Scopes_set( CanonicalState* state, LexicalScopeHandle scope1, LexicalScopeHandle scope2 )
{
    state->scopes.scope1 = scope1;
    state->scopes.scope2 = scope2;
}

CORT_OVERRIDABLE
void cort::Scopes_set( CanonicalState* state, Scopes scopes )
{
    // MegakernelCompile::removeRedundantPushPop works on the version with 3 parameters.
    // Defer this call to make sure it doesn't get missed.
    Scopes_set( state, scopes.scope1, scopes.scope2 );
}

CORT_OVERRIDABLE
Scopes cort::Scopes_get( CanonicalState* state )
{
    return state->scopes;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::Scopes_getScope1( CanonicalState* state )
{
    return state->scopes.scope1;
}

CORT_OVERRIDABLE
void cort::Scopes_setScope2( CanonicalState* state, LexicalScopeHandle scope2 )
{
    state->scopes.scope2 = scope2;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::Scopes_getScope2( CanonicalState* state )
{
    return state->scopes.scope2;
}


/*
 * ActiveProgram methods
 */

CORT_OVERRIDABLE
void cort::ActiveProgram_set( CanonicalState* state, ProgramHandle program )
{
    state->active.program = program;
}

CORT_OVERRIDABLE
ProgramHandle cort::ActiveProgram_get( CanonicalState* state )
{
    return state->active.program;
}

/*
 * AABB methods
 */

CORT_OVERRIDABLE
void cort::AABB_set( CanonicalState* state, float* ptr, unsigned int primitive, unsigned int motion_index )
{
    state->aabb.ptr          = ptr;
    state->aabb.primitive    = primitive;
    state->aabb.motion_index = motion_index;
}

CORT_OVERRIDABLE
float* cort::AABB_getPtr( CanonicalState* state )
{
    return state->aabb.ptr;
}

CORT_OVERRIDABLE
unsigned int cort::AABB_getPrimitive( CanonicalState* state )
{
    return state->aabb.primitive;
}

CORT_OVERRIDABLE
unsigned int cort::AABB_getMotionIndex( CanonicalState* state )
{
    return state->aabb.motion_index;
}

CORT_OVERRIDABLE
unsigned int cort::AABB_setMotionIndex( CanonicalState* state, unsigned int newIndex )
{
    unsigned int oldIndex    = state->aabb.motion_index;
    state->aabb.motion_index = newIndex;
    return oldIndex;
}

/*
 * Traversal methods
 */
CORT_OVERRIDABLE
void cort::Traversal_set( CanonicalState* state, GraphNodeHandle node )
{
    state->traversal.lwrrentNode = node;
}

CORT_OVERRIDABLE
GraphNodeHandle cort::Traversal_getLwrrentNode( CanonicalState* state )
{
    return state->traversal.lwrrentNode;
}

/*
 * Intersect methods
 */
CORT_OVERRIDABLE
void cort::Intersect_set( CanonicalState* state, unsigned int primitiveIndex )
{
    state->intersect.primitive = primitiveIndex;
}

CORT_OVERRIDABLE
unsigned int cort::Intersect_getPrimitive( CanonicalState* state )
{
    return state->intersect.primitive;
}


/*
 * RayFrame methods
 */
CORT_OVERRIDABLE
RayFrame* cort::RayFrame_push( CanonicalState* state, RayFrame* ray, float3 o, float3 d, float tmin )
{
    ray->o    = o;
    ray->d    = d;
    ray->tmin = tmin;

    RayFrame* oldRay = state->ray;
    state->ray       = ray;
    return oldRay;
}

CORT_OVERRIDABLE
void cort::RayFrame_pop( CanonicalState* state, RayFrame* ray )
{
    state->ray = ray;
}

CORT_OVERRIDABLE
float3 cort::RayFrame_getRayO( CanonicalState* state )
{
    return state->ray->o;
}

CORT_OVERRIDABLE
float3 cort::RayFrame_getRayD( CanonicalState* state )
{
    return state->ray->d;
}

CORT_OVERRIDABLE
float cort::RayFrame_getRayTmin( CanonicalState* state )
{
    return state->ray->tmin;
}

/*
 * TraceFrame methods
 */

CORT_OVERRIDABLE extern "C" unsigned int TraceFrame_attributeSize = MAXATTRIBUTE_TOTALSIZE;

CORT_OVERRIDABLE
TraceFrame* cort::TraceFrame_push( CanonicalState* state, TraceFrame* frame, unsigned int rayType, float tmax, float rayTime, unsigned int hasTime, char* payload )
{
    // The constant below means the time has not been set by the user, and we should inherit it.
    // see C14n::getInheritedTimeConstant
    if( !hasTime )
        rayTime = TraceFrame_getLwrrentRayTime( state );

    // Push frame
    TraceFrame* oldFrame = state->trace;
    state->trace         = frame;

    // Store payload pointer in frame.
    frame->payload = payload;

    // Set up state
    frame->rayType = rayType;
    TraceFrame_setLwrrentRayTime( state, rayTime );
    TraceFrame_setLwrrentTmax( state, tmax );
    TraceFrame_setLwrrentTransformDepth( state, 0 );
    TraceFrame_setCommittedTmax( state, tmax );
    TraceFrame_setCommittedTransformDepth( state, 0 );
    TraceFrame_setCommittedMaterial( state, {0} );
    TraceFrame_setAttributeSwitch( state, true );

    return oldFrame;
}

CORT_OVERRIDABLE
void cort::TraceFrame_pop( CanonicalState* state, TraceFrame* frame )
{
    state->trace = frame;
}

CORT_OVERRIDABLE
bool cort::TraceFrame_hasPotentialIntersection( CanonicalState* state, float t )
{
    float tmin = RayFrame_getRayTmin( state );
    float tmax = TraceFrame_getLwrrentTmax( state );
    if( t > tmin && t < tmax )
    {
        TraceFrame_setCommittedTmax( state, tmax );
        TraceFrame_setLwrrentTmax( state, t );
        return true;
    }
    else
    {
        return false;
    }
}

CORT_OVERRIDABLE
bool cort::TraceFrame_reportIntersection( CanonicalState* state, unsigned int matlIndex )
{
    // Set the material
    GeometryInstanceHandle instance = TraceFrame_getLwrrentInstance( state );
    MaterialHandle         material = GeometryInstance_getMaterial( state, instance, matlIndex );
    TraceFrame_setLwrrentMaterial( state, material );

    // Call the anyhit
    unsigned int  rayType   = TraceFrame_getRayType( state );
    ProgramHandle ahprogram = Material_getAHProgram( state, material, rayType );
    Runtime_ilwokeProgram( state, ahprogram, optix::ST_ANY_HIT, GeometryInstance_colwertToLexicalScopeHandle( state, instance ),
                           Material_colwertToLexicalScopeHandle( state, material ) );

    if( TerminateRay_isRequested( state ) )
    {
        // There was a call to rtTerminateRay and we have not yet reached the start
        // place of the traversal, so we have to keep unwinding.
        // NOTE: In the compile stage, the code is modified to unwind one more level
        //       of the call stack, i.e., this return will not only return from this
        //       function but return from the entire sub state, unless this part of
        //       the trace call stack is inlined. If modified, this will result in
        //       an additional return edge in the call graph, which in turn will
        //       ensure that live values are properly stored and reloaded.
        TerminateRay_unwind( state );
        return true;
    }

    // Copy data from the current hit to the committed hit for
    // subsequent intersections, unless the intersection was ignored.
    if( TraceFrame_intersectionWasIgnored( state ) )
        return false;

    TraceFrame_commitHit( state );
    return true;
}

CORT_OVERRIDABLE
void cort::TraceFrame_ignoreIntersection( CanonicalState* state )
{
    // Flag the ignored intersection by clearing the material.
    // Canonicalization will also guarantee a return directly after this
    // function call.
    TraceFrame_setLwrrentMaterial( state, {0} );
    TraceFrame_setLwrrentTmax( state, TraceFrame_getCommittedTmax( state ) );
}

CORT_OVERRIDABLE
bool cort::TraceFrame_intersectionWasIgnored( CanonicalState* state )
{
    return TraceFrame_getLwrrentMaterial( state ) == 0;
}

CORT_OVERRIDABLE
void cort::TraceFrame_pushTransform( CanonicalState* state, TransformHandle node )
{
    const unsigned char depth = TraceFrame_getLwrrentTransformDepth( state );
    TraceFrame_setLwrrentTransformDepth( state, depth + 1 );
    TraceFrame_getLwrrentTransforms( state )[depth] = node;
}

CORT_OVERRIDABLE
void cort::TraceFrame_pushTransform_atMostOne( CanonicalState* state, TransformHandle node )
{
    TraceFrame_setLwrrentTransformDepth( state, 1 );
    TraceFrame_getLwrrentTransforms( state )[0] = node;
}

CORT_OVERRIDABLE
void cort::TraceFrame_popTransform( CanonicalState* state )
{
    const unsigned char depth = TraceFrame_getLwrrentTransformDepth( state );
    TraceFrame_setLwrrentTransformDepth( state, depth - 1 );
}


CORT_OVERRIDABLE
void cort::TraceFrame_popTransform_atMostOne( CanonicalState* state )
{
    TraceFrame_setLwrrentTransformDepth( state, 0 );
}

void cort::TraceFrame_commitHit( CanonicalState* state )
{
    TraceFrame_setCommittedTmax( state, TraceFrame_getLwrrentTmax( state ) );
    TraceFrame_setCommittedInstance( state, TraceFrame_getLwrrentInstance( state ) );
    TraceFrame_setCommittedMaterial( state, TraceFrame_getLwrrentMaterial( state ) );
    int d = TraceFrame_getLwrrentTransformDepth( state );
    TraceFrame_setCommittedTransformDepth( state, d );
    for( int i = 0; i < d; ++i )
        TraceFrame_setCommittedTransformByDepth( state, i, TraceFrame_getLwrrentTransformByDepth( state, i ) );
    TraceFrame_commitAttributes( state );
}

void cort::TraceFrame_commitHit_atMostOne( CanonicalState* state )
{
    TraceFrame_setCommittedTmax( state, TraceFrame_getLwrrentTmax( state ) );
    TraceFrame_setCommittedInstance( state, TraceFrame_getLwrrentInstance( state ) );
    TraceFrame_setCommittedMaterial( state, TraceFrame_getLwrrentMaterial( state ) );
    int d = TraceFrame_getLwrrentTransformDepth( state );
    TraceFrame_setCommittedTransformDepth( state, d );
    if( d > 0 )
        TraceFrame_setCommittedTransformByDepth( state, 0, TraceFrame_getLwrrentTransformByDepth( state, 0 ) );
    TraceFrame_commitAttributes( state );
}

void cort::TraceFrame_restoreHit( CanonicalState* state )
{
    TraceFrame_setLwrrentTmax( state, TraceFrame_getCommittedTmax( state ) );
    TraceFrame_setLwrrentInstance( state, TraceFrame_getCommittedInstance( state ) );
    TraceFrame_setLwrrentMaterial( state, TraceFrame_getCommittedMaterial( state ) );
    int d = TraceFrame_getCommittedTransformDepth( state );
    TraceFrame_setLwrrentTransformDepth( state, d );
    for( int i = 0; i < d; ++i )
        TraceFrame_setLwrrentTransformByDepth( state, i, TraceFrame_getCommittedTransformByDepth( state, i ) );
    TraceFrame_restoreAttributes( state );
}

void cort::TraceFrame_restoreHit_atMostOne( CanonicalState* state )
{
    TraceFrame_setLwrrentTmax( state, TraceFrame_getCommittedTmax( state ) );
    TraceFrame_setLwrrentInstance( state, TraceFrame_getCommittedInstance( state ) );
    TraceFrame_setLwrrentMaterial( state, TraceFrame_getCommittedMaterial( state ) );
    int d = TraceFrame_getCommittedTransformDepth( state );
    TraceFrame_setLwrrentTransformDepth( state, d );
    if( d > 0 )
        TraceFrame_setLwrrentTransformByDepth( state, 0, TraceFrame_getCommittedTransformByDepth( state, 0 ) );
    TraceFrame_restoreAttributes( state );
}

#define PING_PONG_ATTRIBUTES 1

void cort::TraceFrame_commitAttributes( CanonicalState* state )
{
#if PING_PONG_ATTRIBUTES
    TraceFrame_setAttributeSwitch( state, !TraceFrame_getAttributeSwitch( state ) );
#else
    cort_memcpy( TraceFrame_getCommittedAttributeFrame( state ), TraceFrame_getLwrrentAttributeFrame( state ),
                 TraceFrame_attributeSize, 16 );
#endif
}

void cort::TraceFrame_restoreAttributes( CanonicalState* state )
{
#if PING_PONG_ATTRIBUTES
    TraceFrame_setAttributeSwitch( state, !TraceFrame_getAttributeSwitch( state ) );
#else
    cort_memcpy( TraceFrame_getLwrrentAttributeFrame( state ), TraceFrame_getCommittedAttributeFrame( state ),
                 TraceFrame_attributeSize, 16 );
#endif
}

CORT_OVERRIDABLE
unsigned short cort::TraceFrame_getRayType( CanonicalState* state )
{
    return state->trace->rayType;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setCommittedTmax( CanonicalState* state, float tmax )
{
    state->trace->committedTmax = tmax;
}

CORT_OVERRIDABLE
float cort::TraceFrame_getCommittedTmax( CanonicalState* state )
{
    return state->trace->committedTmax;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setCommittedInstance( CanonicalState* state, GeometryInstanceHandle gi )
{
    state->trace->committedInstance = gi;
}

CORT_OVERRIDABLE
GeometryInstanceHandle cort::TraceFrame_getCommittedInstance( CanonicalState* state )
{
    return state->trace->committedInstance;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setCommittedMaterial( CanonicalState* state, MaterialHandle m )
{
    state->trace->committedMaterial = m;
}

CORT_OVERRIDABLE
MaterialHandle cort::TraceFrame_getCommittedMaterial( CanonicalState* state )
{
    return state->trace->committedMaterial;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setCommittedTransformDepth( CanonicalState* state, unsigned char depth )
{
    state->trace->committedTransformDepth = depth;
}

CORT_OVERRIDABLE
unsigned char cort::TraceFrame_getCommittedTransformDepth( CanonicalState* state )
{
    return state->trace->committedTransformDepth;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setCommittedTransformByDepth( CanonicalState* state, unsigned char depth, TransformHandle th )
{
    TraceFrame_getCommittedTransforms( state )[depth] = th;
}

CORT_OVERRIDABLE
TransformHandle cort::TraceFrame_getCommittedTransformByDepth( CanonicalState* state, unsigned char depth )
{
    return TraceFrame_getCommittedTransforms( state )[depth];
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentRayTime( CanonicalState* state, float rayTime )
{
    state->trace->rayTime = rayTime;
}

CORT_OVERRIDABLE
float cort::TraceFrame_getLwrrentRayTime( CanonicalState* state )
{
    if( state->trace )
        return state->trace->rayTime;
    return 0.0f;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentTmax( CanonicalState* state, float tmax )
{
    state->trace->lwrrentTmax = tmax;
}

CORT_OVERRIDABLE
float cort::TraceFrame_getLwrrentTmax( CanonicalState* state )
{
    return state->trace->lwrrentTmax;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentInstance( CanonicalState* state, GeometryInstanceHandle gi )
{
    state->trace->lwrrentInstance = gi;
}

CORT_OVERRIDABLE
GeometryInstanceHandle cort::TraceFrame_getLwrrentInstance( CanonicalState* state )
{
    return state->trace->lwrrentInstance;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentMaterial( CanonicalState* state, MaterialHandle m )
{
    state->trace->lwrrentMaterial = m;
}

CORT_OVERRIDABLE
MaterialHandle cort::TraceFrame_getLwrrentMaterial( CanonicalState* state )
{
    return state->trace->lwrrentMaterial;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentTransformDepth( CanonicalState* state, unsigned char depth )
{
    state->trace->lwrrentTransformDepth = depth;
}

CORT_OVERRIDABLE
unsigned char cort::TraceFrame_getLwrrentTransformDepth( CanonicalState* state )
{
    return state->trace->lwrrentTransformDepth;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setLwrrentTransformByDepth( CanonicalState* state, unsigned char depth, TransformHandle th )
{
    TraceFrame_getLwrrentTransforms( state )[depth] = th;
}

CORT_OVERRIDABLE
TransformHandle cort::TraceFrame_getLwrrentTransformByDepth( CanonicalState* state, unsigned char depth )
{
    return TraceFrame_getLwrrentTransforms( state )[depth];
}

CORT_OVERRIDABLE
char* cort::TraceFrame_getLwrrentAttributeAddress( CanonicalState* state, unsigned short token, unsigned int offset )
{
    char* attribute = TraceFrame_getLwrrentAttributeFrame( state ) + offset;
    return attribute;
}

CORT_OVERRIDABLE
void cort::TraceFrame_setAttributeSwitch( CanonicalState* state, bool value )
{
    state->trace->attributeSwitch = value;
}

CORT_OVERRIDABLE
bool cort::TraceFrame_getAttributeSwitch( CanonicalState* state )
{
    return state->trace->attributeSwitch;
}

CORT_OVERRIDABLE
char* cort::TraceFrame_getPayloadAddress( CanonicalState* state )
{
    return state->trace->payload;
}

CORT_OVERRIDABLE
char* cort::TraceFrame_getLwrrentAttributeFrame( CanonicalState* state )
{
#if PING_PONG_ATTRIBUTES
    if( TraceFrame_getAttributeSwitch( state ) )
        return state->trace->lwrrentAttributes;
    else
        return state->trace->committedAttributes;
#else
    return state->trace->lwrrentAttributes;
#endif
}

CORT_OVERRIDABLE
char* cort::TraceFrame_getCommittedAttributeFrame( CanonicalState* state )
{
#if PING_PONG_ATTRIBUTES
    if( TraceFrame_getAttributeSwitch( state ) )
        return state->trace->committedAttributes;
    else
        return state->trace->lwrrentAttributes;
#else
    return state->trace->committedAttributes;
#endif
}

CORT_OVERRIDABLE
TransformHandle* cort::TraceFrame_getLwrrentTransforms( CanonicalState* state )
{
    return state->trace->lwrrentTransforms;
}

CORT_OVERRIDABLE
TransformHandle* cort::TraceFrame_getCommittedTransforms( CanonicalState* state )
{
    return state->trace->committedTransforms;
}

/*
 * Profiling methods
 */
// These variables can be modified/specialized by compile so that they
// can become compile-time constants.
CORT_OVERRIDABLE extern "C" int Profile_enabled       = 0;
CORT_OVERRIDABLE extern "C" int Profile_counterOffset = 1 * PROFILE_NUM_RESERVED_COUNTERS + 0;
CORT_OVERRIDABLE extern "C" int Profile_eventOffset = 2 * PROFILE_NUM_RESERVED_EVENTS + ( 1 * PROFILE_NUM_RESERVED_COUNTERS );
CORT_OVERRIDABLE extern "C" int Profile_timerOffset =
    3 * PROFILE_NUM_RESERVED_TIMERS + ( 1 * PROFILE_NUM_RESERVED_COUNTERS + 2 * PROFILE_NUM_RESERVED_EVENTS );
// This is what to do, but clang is choking on it
//CORT_OVERRIDABLE extern "C" int Profile_eventOffset   = 2 * PROFILE_NUM_RESERVED_EVENTS    + Profile_counterOffset;
//CORT_OVERRIDABLE extern "C" int Profile_timerOffset   = 3 * PROFILE_NUM_RESERVED_TIMERS    + Profile_eventOffset;

// TODO: determine if this needs to be a separate function or if
// its functionality can be combined with setState_Global.
CORT_OVERRIDABLE
void cort::Profile_setup( CanonicalState* state, uint64* globalValues )
{
    if( !Profile_enabled )
        return;

    state->global.profileData = globalValues;
}

CORT_OVERRIDABLE
void cort::Profile_count( CanonicalState* state, int index, uint64 value )
{
    if( !Profile_enabled )
        return;

    int     slot       = index + Profile_counterOffset;
    uint64* globalVals = Global_getProfileData( state ) + slot;
    atomicAdd( globalVals, value );
}

CORT_OVERRIDABLE
void cort::Profile_event( CanonicalState* state, int index )
{
    if( !Profile_enabled )
        return;

    const cort::uint active = lwca::ballot( 1 );
    if( lwca::ffs( active ) == lwca::laneid() )
    {  // Nominate one active lane to issue the atomics
        int     slot       = index * 2 + Profile_eventOffset;
        uint64* globalVals = Global_getProfileData( state ) + slot;

        // Add utilization and count
        const uint64 utilization = lwca::popc( active );
        atomicAdd( &globalVals[0], utilization );
        atomicAdd( &globalVals[1], 1 );
    }
}

CORT_OVERRIDABLE
void cort::Profile_timerStart( CanonicalState* state, int index )
{
    if( !Profile_enabled )
        return;

    const cort::uint active = lwca::ballot( 1 );
    if( lwca::ffs( active ) == lwca::laneid() )
    {
        // Nominate one active lane to issue the atomics
        int     slot       = index * 3 + Profile_timerOffset;
        uint64* globalVals = Global_getProfileData( state ) + slot;

        // Count time
        const uint64 tick = -lwca::clock64();
        atomicAdd( &globalVals[0], tick );

        // And utilization
        const uint64 utilization = lwca::popc( active );
        atomicAdd( &globalVals[1], utilization );
        atomicAdd( &globalVals[2], 1 );
    }
}

CORT_OVERRIDABLE
void cort::Profile_timerStop( CanonicalState* state, int index )
{
    if( !Profile_enabled )
        return;

    const cort::uint active = lwca::ballot( 1 );
    if( lwca::ffs( active ) == lwca::laneid() )
    {
        // Nominate one active lane to issue the atomics
        int     slot       = index * 3 + Profile_timerOffset;
        uint64* globalVals = Global_getProfileData( state ) + slot;

        // Stop time
        const uint64 tick = lwca::clock64();
        atomicAdd( &globalVals[0], tick );
    }
}

/************************************************************************************************************
 * Data access
 ************************************************************************************************************/

/*
 * Buffer methods
 */
CORT_OVERRIDABLE
char* cort::Buffer_getElementAddress1d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress1dFromId( state, bufferid, eltSize, stackTmp, x );
}

CORT_OVERRIDABLE
char* cort::Buffer_getElementAddress2d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress2dFromId( state, bufferid, eltSize, stackTmp, x, y );
}

CORT_OVERRIDABLE
char* cort::Buffer_getElementAddress3d( CanonicalState* state, unsigned short token, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getElementAddress3dFromId( state, bufferid, eltSize, stackTmp, x, y, z );
}

CORT_OVERRIDABLE
char* cort::Buffer_getElementAddress1dFromId( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, char* stackTmp, uint64 x )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader1d_generic( state, buffer, eltSize, stackTmp, x );
}

CORT_OVERRIDABLE
char* cort::Buffer_getElementAddress2dFromId( CanonicalState* state, unsigned int bufferid, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return Buffer_decodeBufferHeader2d_generic( state, buffer, eltSize, stackTmp, x, y );
}

CORT_OVERRIDABLE
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

CORT_OVERRIDABLE
char* cort::Buffer_decodeBufferHeader1d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x )
{
    // Texheap path is required only for 1d buffers with an element size of 16
    if( eltSize == 16 && buffer.dd.texUnit != Buffer::UseDataAsPointer )
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

CORT_OVERRIDABLE
char* cort::Buffer_decodeBufferHeader2d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x );
}

CORT_OVERRIDABLE
char* cort::Buffer_decodeBufferHeader3d_generic( CanonicalState* state, Buffer buffer, unsigned int eltSize, char* stackTmp, uint64 x, uint64 y, uint64 z )
{
    return buffer.dd.data + eltSize * ( x + y * buffer.di.size.x + z * buffer.di.size.x * buffer.di.size.y );
}


CORT_OVERRIDABLE
size3 cort::Buffer_getSize( CanonicalState* state, unsigned short token )
{
    unsigned int bufferid = Runtime_lookupIdVariableValue( state, token );
    return Buffer_getSizeFromId( state, bufferid );
}

CORT_OVERRIDABLE
size3 cort::Buffer_getSizeFromId( CanonicalState* state, unsigned int bufferid )
{
    Buffer buffer = Global_getBufferHeader( state, bufferid );
    return buffer.di.size;
}

/*
 * Acceleration methods
 */
CORT_OVERRIDABLE
LexicalScopeHandle cort::Acceleration_colwertToLexicalScopeHandle( CanonicalState* state, AccelerationHandle accel )
{
    return accel;
}

CORT_OVERRIDABLE
MotionAccelerationHandle cort::Acceleration_colwertToMotionAccelerationHandle( CanonicalState* state, AccelerationHandle accel )
{
    return accel;
}


/*
 * MotionAcceleration methods
 */
CORT_OVERRIDABLE
float cort::MotionAcceleration_getTimeBegin( CanonicalState* state, MotionAccelerationHandle maccel )
{
    MotionAccelerationRecord* mar = Global_getObjectRecord<MotionAccelerationRecord>( state, maccel );
    return mar->timeBegin;
}

CORT_OVERRIDABLE
float cort::MotionAcceleration_getTimeEnd( CanonicalState* state, MotionAccelerationHandle maccel )
{
    MotionAccelerationRecord* mar = Global_getObjectRecord<MotionAccelerationRecord>( state, maccel );
    return mar->timeEnd;
}

CORT_OVERRIDABLE
unsigned int cort::MotionAcceleration_getMotionSteps( CanonicalState* state, MotionAccelerationHandle maccel )
{
    MotionAccelerationRecord* mar = Global_getObjectRecord<MotionAccelerationRecord>( state, maccel );
    return mar->motionSteps;
}

CORT_OVERRIDABLE
unsigned int cort::MotionAcceleration_getMotionStride( CanonicalState* state, MotionAccelerationHandle maccel )
{
    MotionAccelerationRecord* mar = Global_getObjectRecord<MotionAccelerationRecord>( state, maccel );
    return mar->motionStride;
}


/*
 * GlobalScope methods
 */

CORT_OVERRIDABLE
GlobalScopeHandle cort::GlobalScope_get( CanonicalState* state )
{
    return 0;
}

CORT_OVERRIDABLE
ProgramHandle cort::GlobalScope_getRaygen( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short entry )
{
    GlobalScopeRecord* gsr = Global_getObjectRecord<GlobalScopeRecord>( state, globalScope );
    return gsr->programs[entry].raygen;
}

CORT_OVERRIDABLE
ProgramHandle cort::GlobalScope_getMiss( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short rayType )
{
    GlobalScopeRecord* gsr = Global_getObjectRecord<GlobalScopeRecord>( state, globalScope );
    return gsr->programs[rayType].miss;
}

CORT_OVERRIDABLE
ProgramHandle cort::GlobalScope_getException( CanonicalState* state, GlobalScopeHandle globalScope, unsigned short index )
{
    GlobalScopeRecord* gsr = Global_getObjectRecord<GlobalScopeRecord>( state, globalScope );
    return gsr->programs[index].exception;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::GlobalScope_colwertToLexicalScopeHandle( CanonicalState* state, GlobalScopeHandle globalScope )
{
    return globalScope;
}

/*
 * LexicalScope methods
 */

CORT_OVERRIDABLE
DynamicVariableTableOffset cort::LexicalScope_getDynamicVariableTable( CanonicalState* state, LexicalScopeHandle object )
{
    LexicalScopeRecord* lsr = Global_getObjectRecord<LexicalScopeRecord>( state, object );
    return lsr->dynamicVariableTable;
}

// The following functionality regarding dynamicVariableTable lookup is duplicated in RTX runtime.
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

CORT_OVERRIDABLE
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

CORT_OVERRIDABLE
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


CORT_OVERRIDABLE
AbstractGroupHandle cort::LexicalScope_colwertToAbstractGroupHandle( CanonicalState* state, LexicalScopeHandle object )
{
    return object;
}

CORT_OVERRIDABLE
GraphNodeHandle cort::LexicalScope_colwertToGraphNodeHandle( CanonicalState* state, LexicalScopeHandle object )
{
    return object;
}

CORT_OVERRIDABLE
GeometryInstanceHandle cort::LexicalScope_colwertToGeometryInstanceHandle( CanonicalState* state, LexicalScopeHandle object )
{
    return object;
}

/*
 * GraphNode methods
 */

CORT_OVERRIDABLE
ProgramHandle cort::GraphNode_getTraverse( CanonicalState* state, GraphNodeHandle node )
{
    GraphNodeRecord* gnr = Global_getObjectRecord<GraphNodeRecord>( state, node );
    return gnr->traverse;
}

CORT_OVERRIDABLE
ProgramHandle cort::GraphNode_getBBProgram( CanonicalState* state, GraphNodeHandle node )
{
    GraphNodeRecord* gnr = Global_getObjectRecord<GraphNodeRecord>( state, node );
    return gnr->bounds;
}

CORT_OVERRIDABLE
TraversableId cort::GraphNode_getTraversableId( CanonicalState* state, GraphNodeHandle node )
{
    GraphNodeRecord* gnr = Global_getObjectRecord<GraphNodeRecord>( state, node );
    return gnr->traversableId;
}

CORT_OVERRIDABLE
AbstractGroupHandle cort::GraphNode_colwertToAbstractGroupHandle( CanonicalState* state, GraphNodeHandle node )
{
    return node;
}

CORT_OVERRIDABLE
TransformHandle cort::GraphNode_colwertToTransformHandle( CanonicalState* state, GraphNodeHandle node )
{
    return node;
}

CORT_OVERRIDABLE
SelectorHandle cort::GraphNode_colwertToSelectorHandle( CanonicalState* state, GraphNodeHandle node )
{
    return node;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::GraphNode_colwertToLexicalScopeHandle( CanonicalState* state, GraphNodeHandle node )
{
    return node;
}

/*
 * AbstractGroup methods
 */
CORT_OVERRIDABLE
AccelerationHandle cort::AbstractGroup_getAcceleration( CanonicalState* state, AbstractGroupHandle g )
{
    AbstractGroupRecord* agr = Global_getObjectRecord<AbstractGroupRecord>( state, g );
    return agr->accel;
}

CORT_OVERRIDABLE
BufferId cort::AbstractGroup_getChildren( CanonicalState* state, AbstractGroupHandle g )
{
    AbstractGroupRecord* agr = Global_getObjectRecord<AbstractGroupRecord>( state, g );
    return agr->children;
}


/*
 * GeometryInstance methods
 */
CORT_OVERRIDABLE
GeometryHandle cort::GeometryInstance_getGeometry( CanonicalState* state, GeometryInstanceHandle gi )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return gir->geometry;
}

CORT_OVERRIDABLE
unsigned int cort::GeometryInstance_getNumMaterials( CanonicalState* state, GeometryInstanceHandle gi )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return gir->numMaterials;
}

CORT_OVERRIDABLE
MaterialHandle cort::GeometryInstance_getMaterial( CanonicalState* state, GeometryInstanceHandle gi, unsigned int matlIndex )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return gir->materials[matlIndex];
}

CORT_OVERRIDABLE
char* cort::GeometryInstance_getMaterialAddress( CanonicalState* state, GeometryInstanceHandle gi, unsigned int matlIndex )
{
    GeometryInstanceRecord* gir = Global_getObjectRecord<GeometryInstanceRecord>( state, gi );
    return reinterpret_cast<char*>( gir->materials + matlIndex );
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::GeometryInstance_colwertToLexicalScopeHandle( CanonicalState* state, GeometryInstanceHandle gi )
{
    return gi;
}

/*
 * Geometry methods
 */
CORT_OVERRIDABLE
unsigned int cort::Geometry_getPrimitiveIndexOffset( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->indexOffset;
}

CORT_OVERRIDABLE
ProgramHandle cort::Geometry_getIntersectProgram( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->intersectOrAttribute;
}

CORT_OVERRIDABLE
ProgramHandle cort::Geometry_getAABBProgram( CanonicalState* state, GeometryHandle geometry )
{
    GeometryRecord* gr = Global_getObjectRecord<GeometryRecord>( state, geometry );
    return gr->aabb;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::Geometry_colwertToLexicalScopeHandle( CanonicalState* state, GeometryHandle geometry )
{
    return geometry;
}


/*
 * Material methods
 */

CORT_OVERRIDABLE
ProgramHandle cort::Material_getAHProgram( CanonicalState* state, MaterialHandle material, unsigned char rayType )
{
    MaterialRecord* mr = Global_getObjectRecord<MaterialRecord>( state, material );
    return mr->programs[rayType].anyHit;
}

CORT_OVERRIDABLE
ProgramHandle cort::Material_getCHProgram( CanonicalState* state, MaterialHandle material, unsigned char rayType )
{
    MaterialRecord* mr = Global_getObjectRecord<MaterialRecord>( state, material );
    return mr->programs[rayType].closestHit;
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::Material_colwertToLexicalScopeHandle( CanonicalState* state, MaterialHandle material )
{
    return material;
}

/*
 * Program methods
 */

CORT_OVERRIDABLE
ProgramId cort::Program_getProgramID( CanonicalState* state, ProgramHandle program )
{
    ProgramRecord* pr = Global_getObjectRecord<ProgramRecord>( state, program );
    return pr->programID;
}

CORT_OVERRIDABLE
unsigned short cort::Program_getCanonicalProgramID( CanonicalState* state, ProgramHandle program )
{
    ProgramId     pid = Program_getProgramID( state, program );
    ProgramHeader ph  = Global_getProgramHeader( state, pid );
    return ph.dd.canonicalProgramID;
}

CORT_OVERRIDABLE
ProgramHandle cort::Program_getProgramHandle( CanonicalState* state, ProgramId pid )
{
    ProgramHeader ph = Global_getProgramHeader( state, pid );
    return {ph.di.programOffset};
}

CORT_OVERRIDABLE
LexicalScopeHandle cort::Program_colwertToLexicalScopeHandle( CanonicalState* state, ProgramHandle program )
{
    return program;
}

/*
 * Selector methods
 */

CORT_OVERRIDABLE
SelectorHandle cort::Selector_getLwrrent( CanonicalState* state )
{
    GraphNodeHandle lwrNode = Traversal_getLwrrentNode( state );
    return GraphNode_colwertToSelectorHandle( state, lwrNode );
}

CORT_OVERRIDABLE
BufferId cort::Selector_getChildren( CanonicalState* state, SelectorHandle selector )
{
    SelectorRecord* sr = Global_getObjectRecord<SelectorRecord>( state, selector );
    return sr->children;
}

CORT_OVERRIDABLE
char* cort::Selector_getChildAddress( CanonicalState* state, unsigned int child )
{
    // Get the current selector.
    SelectorHandle lwrSelector = Selector_getLwrrent( state );

    // Get the buffer id that corresponds to the children of the current selector.
    const BufferId bufferId = Selector_getChildren( state, lwrSelector );

    // Get the address of that child element.
    const unsigned int eltSize = sizeof( ObjectRecordOffset );
    char               stackTmp[eltSize];
    return Buffer_getElementAddress1dFromId( state, bufferId, eltSize, stackTmp, child );
}

CORT_OVERRIDABLE
GraphNodeHandle cort::Selector_getChildNode( CanonicalState* state, unsigned int child )
{
    // Get the address.
    char*               ptr     = Selector_getChildAddress( state, child );
    ObjectRecordOffset* address = reinterpret_cast<ObjectRecordOffset*>( ptr );

    // Load from the address.
    return LexicalScope_colwertToGraphNodeHandle( state, {*address} );
}

CORT_OVERRIDABLE
uint64 cort::Selector_getNumChildren( CanonicalState* state )
{
    // Get the current selector.
    SelectorHandle lwrSelector = Selector_getLwrrent( state );

    // Get the buffer id that corresponds to the children of the current selector.
    const BufferId bufferId = Selector_getChildren( state, lwrSelector );

    // Get the size of the buffer.
    const size3 bufferSize = Buffer_getSizeFromId( state, bufferId );

    return bufferSize.x;
}

/*
 * Transform methods
 */
CORT_OVERRIDABLE
GraphNodeHandle cort::Transform_getChild( CanonicalState* state, TransformHandle transform )
{
    TransformRecord* tr = Global_getObjectRecord<TransformRecord>( state, transform );
    return tr->child;
}

float lerp( float a, float b, float t )
{
    return a + t * ( b - a );
}

Matrix4x4 getMatrix4x4FromKey( int keyType, const float* key )
{
    Matrix4x4 m;

    if( keyType == MOTIONKEYTYPE_MATRIX_FLOAT12 )
    {
        m.matrix[0][0] = key[0];
        m.matrix[0][1] = key[1];
        m.matrix[0][2] = key[2];
        m.matrix[0][3] = key[3];

        m.matrix[1][0] = key[4];
        m.matrix[1][1] = key[5];
        m.matrix[1][2] = key[6];
        m.matrix[1][3] = key[7];

        m.matrix[2][0] = key[8];
        m.matrix[2][1] = key[9];
        m.matrix[2][2] = key[10];
        m.matrix[2][3] = key[11];

        m.matrix[3][0] = 0.0f;
        m.matrix[3][1] = 0.0f;
        m.matrix[3][2] = 0.0f;
        m.matrix[3][3] = 1.0f;
    }
    else
    {
        Matrix4x4 scale;
        Matrix4x4 rotate;

        m.matrix[0][0] = 0.0f;
        m.matrix[0][1] = 0.0f;
        m.matrix[0][2] = 0.0f;
        m.matrix[0][3] = 0.0f;

        m.matrix[1][0] = 0.0f;
        m.matrix[1][1] = 0.0f;
        m.matrix[1][2] = 0.0f;
        m.matrix[1][3] = 0.0f;

        m.matrix[2][0] = 0.0f;
        m.matrix[2][1] = 0.0f;
        m.matrix[2][2] = 0.0f;
        m.matrix[2][3] = 0.0f;

        m.matrix[3][0] = 0.0f;
        m.matrix[3][1] = 0.0f;
        m.matrix[3][2] = 0.0f;
        m.matrix[3][3] = 1.0f;

        scale.matrix[0][0] = key[0];
        scale.matrix[0][1] = key[1];
        scale.matrix[0][2] = key[2];
        scale.matrix[0][3] = -key[3];

        scale.matrix[1][0] = 0.f;
        scale.matrix[1][1] = key[4];
        scale.matrix[1][2] = key[5];
        scale.matrix[1][3] = -key[6];

        scale.matrix[2][0] = 0.f;
        scale.matrix[2][1] = 0.f;
        scale.matrix[2][2] = key[7];
        scale.matrix[2][3] = -key[8];

        // rotation
        float4 q = float4( key[9], key[10], key[11], key[12] );
        // assumption: q is normalized!
        // since q is user input, we normalize it in Transform.cpp
        rotate.matrix[0][0] = 1.0f - 2.0f * q.y * q.y - 2.0f * q.z * q.z;
        rotate.matrix[0][1] = 2.0f * q.x * q.y - 2.0f * q.z * q.w;
        rotate.matrix[0][2] = 2.0f * q.x * q.z + 2.0f * q.y * q.w;
        rotate.matrix[0][3] = 0.0f;

        rotate.matrix[1][0] = 2.0f * q.x * q.y + 2.0f * q.z * q.w;
        rotate.matrix[1][1] = 1.0f - 2.0f * q.x * q.x - 2.0f * q.z * q.z;
        rotate.matrix[1][2] = 2.0f * q.y * q.z - 2.0f * q.x * q.w;
        rotate.matrix[1][3] = 0.0f;

        rotate.matrix[2][0] = 2.0f * q.x * q.z - 2.0f * q.y * q.w;
        rotate.matrix[2][1] = 2.0f * q.y * q.z + 2.0f * q.x * q.w;
        rotate.matrix[2][2] = 1.0f - 2.0f * q.x * q.x - 2.0f * q.y * q.y;
        rotate.matrix[2][3] = 0.0f;

        // compute R*S
        for( int i = 0; i < 3; ++i )
            for( int j = 0; j < 4; ++j )
                for( int k = 0; k < 3; ++k )
                    m.matrix[i][j] += rotate.matrix[i][k] * scale.matrix[k][j];

        // translation
        m.matrix[0][3] += key[13] + key[3];
        m.matrix[1][3] += key[14] + key[6];
        m.matrix[2][3] += key[15] + key[8];
    }
    return m;
}

void interpolateKey( int keyType, const float* key0, const float* key1, float t, float* result )
{
    if( keyType == MOTIONKEYTYPE_MATRIX_FLOAT12 )
    {
        result[0]  = lerp( key0[0], key1[0], t );
        result[1]  = lerp( key0[1], key1[1], t );
        result[2]  = lerp( key0[2], key1[2], t );
        result[3]  = lerp( key0[3], key1[3], t );
        result[4]  = lerp( key0[4], key1[4], t );
        result[5]  = lerp( key0[5], key1[5], t );
        result[6]  = lerp( key0[6], key1[6], t );
        result[7]  = lerp( key0[7], key1[7], t );
        result[8]  = lerp( key0[8], key1[8], t );
        result[9]  = lerp( key0[9], key1[9], t );
        result[10] = lerp( key0[10], key1[10], t );
        result[11] = lerp( key0[11], key1[11], t );
    }
    else
    {
        result[0] = lerp( key0[0], key1[0], t );
        result[1] = lerp( key0[1], key1[1], t );
        result[2] = lerp( key0[2], key1[2], t );
        result[3] = lerp( key0[3], key1[3], t );
        result[4] = lerp( key0[4], key1[4], t );
        result[5] = lerp( key0[5], key1[5], t );
        result[6] = lerp( key0[6], key1[6], t );
        result[7] = lerp( key0[7], key1[7], t );
        result[8] = lerp( key0[8], key1[8], t );

        // quaternion, use nlerp
        float4 quat;
        quat.x           = lerp( key0[9], key1[9], t );
        quat.y           = lerp( key0[10], key1[10], t );
        quat.z           = lerp( key0[11], key1[11], t );
        quat.w           = lerp( key0[12], key1[12], t );
        float ilw_length = 1.f / sqrt( quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w );
        result[9]        = quat.x * ilw_length;
        result[10]       = quat.y * ilw_length;
        result[11]       = quat.z * ilw_length;
        result[12]       = quat.w * ilw_length;

        // translate
        result[13] = lerp( key0[13], key1[13], t );
        result[14] = lerp( key0[14], key1[14], t );
        result[15] = lerp( key0[15], key1[15], t );
    }
}


Matrix4x4 affine_ilwerse( const Matrix4x4& m )
{
    const float det3 = m.matrix[0][0] * ( m.matrix[1][1] * m.matrix[2][2] - m.matrix[1][2] * m.matrix[2][1] )
                       - m.matrix[0][1] * ( m.matrix[1][0] * m.matrix[2][2] - m.matrix[1][2] * m.matrix[2][0] )
                       + m.matrix[0][2] * ( m.matrix[1][0] * m.matrix[2][1] - m.matrix[1][1] * m.matrix[2][0] );

    const float ilw_det3 = 1.0f / det3;

    float ilw3[3][3];
    ilw3[0][0] = ilw_det3 * ( m.matrix[1][1] * m.matrix[2][2] - m.matrix[2][1] * m.matrix[1][2] );
    ilw3[0][1] = ilw_det3 * ( m.matrix[0][2] * m.matrix[2][1] - m.matrix[2][2] * m.matrix[0][1] );
    ilw3[0][2] = ilw_det3 * ( m.matrix[0][1] * m.matrix[1][2] - m.matrix[1][1] * m.matrix[0][2] );

    ilw3[1][0] = ilw_det3 * ( m.matrix[1][2] * m.matrix[2][0] - m.matrix[2][2] * m.matrix[1][0] );
    ilw3[1][1] = ilw_det3 * ( m.matrix[0][0] * m.matrix[2][2] - m.matrix[2][0] * m.matrix[0][2] );
    ilw3[1][2] = ilw_det3 * ( m.matrix[0][2] * m.matrix[1][0] - m.matrix[1][2] * m.matrix[0][0] );

    ilw3[2][0] = ilw_det3 * ( m.matrix[1][0] * m.matrix[2][1] - m.matrix[2][0] * m.matrix[1][1] );
    ilw3[2][1] = ilw_det3 * ( m.matrix[0][1] * m.matrix[2][0] - m.matrix[2][1] * m.matrix[0][0] );
    ilw3[2][2] = ilw_det3 * ( m.matrix[0][0] * m.matrix[1][1] - m.matrix[1][0] * m.matrix[0][1] );

    const float b[3] = {m.matrix[0][3], m.matrix[1][3], m.matrix[2][3]};

    Matrix4x4 m_ilw;
    m_ilw.matrix[0][0] = ilw3[0][0];
    m_ilw.matrix[0][1] = ilw3[0][1];
    m_ilw.matrix[0][2] = ilw3[0][2];
    m_ilw.matrix[0][3] = -ilw3[0][0] * b[0] - ilw3[0][1] * b[1] - ilw3[0][2] * b[2];

    m_ilw.matrix[1][0] = ilw3[1][0];
    m_ilw.matrix[1][1] = ilw3[1][1];
    m_ilw.matrix[1][2] = ilw3[1][2];
    m_ilw.matrix[1][3] = -ilw3[1][0] * b[0] - ilw3[1][1] * b[1] - ilw3[1][2] * b[2];

    m_ilw.matrix[2][0] = ilw3[2][0];
    m_ilw.matrix[2][1] = ilw3[2][1];
    m_ilw.matrix[2][2] = ilw3[2][2];
    m_ilw.matrix[2][3] = -ilw3[2][0] * b[0] - ilw3[2][1] * b[1] - ilw3[2][2] * b[2];

    m_ilw.matrix[3][0] = 0.0f;
    m_ilw.matrix[3][1] = 0.0f;
    m_ilw.matrix[3][2] = 0.0f;
    m_ilw.matrix[3][3] = 1.0f;

    return m_ilw;
}

CORT_OVERRIDABLE
BufferId cort::Transform_getMotionDataBufferId( CanonicalState* state, TransformHandle transform )
{
    TransformRecord* tr = Global_getObjectRecord<TransformRecord>( state, transform );
    return tr->motionData;
}

CORT_OVERRIDABLE
Matrix4x4 cort::Transform_getMatrix( CanonicalState* state, TransformHandle transform )
{
    TransformRecord* tr = Global_getObjectRecord<TransformRecord>( state, transform );
    return tr->matrix;
}

CORT_OVERRIDABLE
Matrix4x4 cort::Transform_getMatrixAtLwrrentTime( CanonicalState* state, TransformHandle transform )
{
    BufferId motionDataBufferId = Transform_getMotionDataBufferId( state, transform );

    if( !Motion_transforms_enabled || motionDataBufferId == 0 )
        return Transform_getMatrix( state, transform );

    const float time = Runtime_getLwrrentTime( state );

    const unsigned int eltSize = sizeof( float );
    char               stackTmp[eltSize];

    const float* motionData =
        reinterpret_cast<const float*>( Buffer_getElementAddress1dFromId( state, motionDataBufferId, eltSize, stackTmp, 0 ) );

    //const int    keyType         = lwca::float_as_int( motionData[ MDOFFSET_KEY_TYPE          ] );
    const int    keyType = lwca::float_as_int( *motionData );
    const int    mode0   = lwca::float_as_int( motionData[MDOFFSET_BEGIN_BORDER_MODE] );
    const int    mode1   = lwca::float_as_int( motionData[MDOFFSET_END_BORDER_MODE] );
    const float  t0      = motionData[MDOFFSET_TIME_BEGIN];
    const float  t1      = motionData[MDOFFSET_TIME_END];
    const int    numKeys = lwca::float_as_int( motionData[MDOFFSET_NUM_KEYS] );
    const float* keys    = motionData + MDOFFSET_KEYS;

    const int keySize = keyType == MOTIONKEYTYPE_MATRIX_FLOAT12 ? 12 : 16;

    if( time <= t0 )
        return getMatrix4x4FromKey( keyType, keys );

    if( time >= t1 )
        return getMatrix4x4FromKey( keyType, keys + ( numKeys - 1 ) * keySize );

    const float stepSize = numKeys == 1 ? 1.0f : ( t1 - t0 ) / ( numKeys - 1.0f );

    const int   t0_idx = lwca::mini( numKeys - 1, static_cast<int>( ( time - t0 ) / stepSize ) );
    const int   t1_idx = lwca::mini( numKeys - 1, t0_idx + 1 );
    const float pt     = ( time - t0 ) / stepSize - t0_idx;

    const float* key0 = keys + t0_idx * keySize;
    const float* key1 = keys + t1_idx * keySize;
    float        interpKey[16];
    interpolateKey( keyType, key0, key1, pt, interpKey );

    return getMatrix4x4FromKey( keyType, interpKey );
}

CORT_OVERRIDABLE
Matrix4x4 cort::Transform_getIlwMatrix( CanonicalState* state, TransformHandle transform )
{
    TransformRecord* tr = Global_getObjectRecord<TransformRecord>( state, transform );
    return tr->ilwerse_matrix;
}

CORT_OVERRIDABLE
Matrix4x4 cort::Transform_getIlwMatrixAtLwrrentTime( CanonicalState* state, TransformHandle transform )
{
    BufferId motionDataBufferId = Transform_getMotionDataBufferId( state, transform );
    if( !Motion_transforms_enabled || motionDataBufferId == 0 )
        return Transform_getIlwMatrix( state, transform );

    Matrix4x4 m = Transform_getMatrixAtLwrrentTime( state, transform );
    return affine_ilwerse( m );
}
