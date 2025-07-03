// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Objects/GeometryTriangles.h>

#include <limits>
#include <sstream>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <Context/SharedProgramManager.h>
#include <Context/UpdateManager.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTrianglesPrograms_ptx_bin.h>
#include <Objects/GlobalScope.h>
#include <Objects/Program.h>
#include <Objects/ProgramRoot.h>
#include <Objects/Variable.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/LinkedPtrHelpers.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>


using namespace optix;
using namespace prodlib;


//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

GeometryTriangles::GeometryTriangles( Context* context )
    : Geometry( context, true )
    , m_multiBufferIds{context, RT_BUFFER_INPUT}
{
    m_multiBufferIds.markAsBindlessForInternalUse();
    setMotionSteps( 1 );
    setMotionRange( 0, 1 );
    setMotionBorderMode( RT_MOTIONBORDERMODE_CLAMP, RT_MOTIONBORDERMODE_CLAMP );

    setAttributeProgram( getDefaultAttributeProgram() );

    // The multi material feature is optional and we want triangles to work without it.
    setMaterialCount( 1 );

    reallocateRecord();
}

GeometryTriangles::~GeometryTriangles()
{
    deleteVariables();
    setAttributeProgram( nullptr );
}

void GeometryTriangles::getPrograms( Program*& aabbProg, Program*& intersectProg )
{
    // pick specialized program
    // specialization is based on: motion yes/no, indexed yes/no, index/vertex buffer offset, index/vertex stride, index/position format
    // see template parameters and macros in GeometryTrianglesPrograms.lw
    std::function<void( std::string& )> extendName = [&]( std::string& progName ) {
        if( getMotionSteps() > 1 )
        {
            progName += "motion_";
        }
        if( m_isIndexedTriangles )
        {
            progName += "indexed_";
            progName += m_triIndicesFormat == RT_FORMAT_UNSIGNED_INT3 ? "1" : "0";
            progName += "_";
        }

        progName +=
            ( m_positionFormat == RT_FORMAT_FLOAT3 ) ?
                "0" :
                ( ( m_positionFormat == RT_FORMAT_FLOAT2 ) ? "1" : ( ( m_positionFormat == RT_FORMAT_HALF3 ) ? "2" : "3" ) );
    };

    std::string aabbProgName      = "aabb_";
    std::string intersectProgName = "intersect_";
    extendName( aabbProgName );
    extendName( intersectProgName );
    aabbProg = m_context->getSharedProgramManager()->getProgram( data::getGeometryTrianglesProgramsSources(), aabbProgName );
    intersectProg = m_context->getSharedProgramManager()->getProgram( data::getGeometryTrianglesProgramsSources(), intersectProgName );
}

void GeometryTriangles::setSpecializedPrograms()
{
    Program *aabbProg, *intersectProg;
    getPrograms( aabbProg, intersectProg );
    if( m_boundingBoxProgram.get() != aabbProg )
        setBoundingBoxProgram( aabbProg );
    if( m_intersectionProgram.get() != intersectProg )
        setIntersectionProgram( intersectProg );
}

//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void GeometryTriangles::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    if( !m_context->useRtxDataModel() )
        throw ValidationError( RT_EXCEPTION_INFO, "GeometryTriangles only works in RTX mode" );

    //////////////////////////////////////////////////////////////////////////
    // lazily set some variables
    // and the AABB/IS fallback programs
    // this should NOT be set here in validate, but needs to be set lazily (to avoid creating useless intermediate programs)
    // unfortunately, there is no way of doing this deferred
    //if( !m_isIndexedTriangles )
    //    m_varVertexBufferByteOffset->set( m_vertexBufferByteOffsetInIS );
    //const_cast<GeometryTriangles*>(this)->setSpecializedPrograms();
    //m_context->getValidationManager()->unsubscribeForValidation( this );
    //////////////////////////////////////////////////////////////////////////

    // see Geometry::validate
    if( !m_initialized )
        throw ValidationError( RT_EXCEPTION_INFO, "No triangles set for GeometryTriangles" );

    if( !( m_positionFormat == RT_FORMAT_FLOAT3 || m_positionFormat == RT_FORMAT_HALF3
           || m_positionFormat == RT_FORMAT_FLOAT2 || m_positionFormat == RT_FORMAT_HALF2 ) )
        throw ValidationError( RT_EXCEPTION_INFO,
                               "GeometryTriangles has invalid position format, must be RT_FORMAT_FLOAT3, "
                               "RT_FORMAT_HALF3, "
                               "RT_FORMAT_FLOAT2, or RT_FORMAT_HALF2" );

    if( hasMultiBufferMotion() && static_cast<int>( m_vertexBuffers.size() ) != getMotionSteps() )
    {
        throw ValidationError( RT_EXCEPTION_INFO,
                               "GeometryTriangles: Number of motion vertex buffers does not match number of motion "
                               "steps." );
    }
    if( static_cast<int>( m_vertexBuffers.size() ) != ( hasMultiBufferMotion() ? getMotionSteps() : 1 )
        || algorithm::any_of( m_vertexBuffers, []( const TrianglesPtr& b ) { return !b.get(); } ) )
    {
        throw ValidationError( RT_EXCEPTION_INFO, "GeometryTriangles has no vertex buffer" );
    }

    size_t positionSize = ( m_positionFormat == RT_FORMAT_FLOAT3 ) ?
                              sizeof( float ) * 3 :
                              ( ( m_positionFormat == RT_FORMAT_FLOAT2 ) ?
                                    sizeof( float ) * 2 :
                                    ( ( m_positionFormat == RT_FORMAT_HALF3 ) ? sizeof( float ) / 2 * 3 : sizeof( float ) ) );
    if( m_vertexByteStride < positionSize )
        throw ValidationError( RT_EXCEPTION_INFO, "GeometryTriangles: vertex byte stride < position byte size" );

    if( m_numVertices > 0 )
    {
        const std::size_t minSize = m_vertexByteStride * ( m_numVertices - 1 ) + positionSize + m_vertexBufferByteOffset;
        for( const auto& b : m_vertexBuffers )
        {
            if( b->getTotalSizeInBytes() < minSize )
            {
                throw ValidationError( RT_EXCEPTION_INFO,
                                       "GeometryTriangles: vertex buffer is too small to contain the specified number "
                                       "of vertices." );
            }
        }
    }

    if( m_isIndexedTriangles )
    {
        if( !m_indexBuffer.get() )
            throw ValidationError( RT_EXCEPTION_INFO, "GeometryTriangles has no index buffer" );

        if( !( m_triIndicesFormat == RT_FORMAT_UNSIGNED_INT3 || m_triIndicesFormat == RT_FORMAT_UNSIGNED_SHORT3 ) )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryTriangles has invalid index format, must be RT_FORMAT_UNSIGNED_INT3, or "
                                   "RT_FORMAT_UNSIGNED_SHORT3" );

        size_t triIndicesSize =
            3 * ( m_triIndicesFormat == RT_FORMAT_UNSIGNED_INT3 ? sizeof( unsigned int ) : sizeof( unsigned short ) );
        if( m_triIndicesByteStride < triIndicesSize )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryTriangles: triangle indices byte stride < triangle indices byte size" );

        if( getPrimitiveCount() > 0 && ( m_indexBuffer->getTotalSizeInBytes()
                                         < m_triIndicesByteStride * ( getPrimitiveCount() - 1 ) + triIndicesSize + m_indexBufferByteOffset ) )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryTriangles: index buffer is too small to contain the specified number of "
                                   "indices." );
    }
    else
    {
        if( m_numVertices != (unsigned int)getPrimitiveCount() * 3 )
            throw ValidationError( RT_EXCEPTION_INFO,
                                   "GeometryTriangles: number of vertices must be 3 * number of triangles" );
    }

    for( auto parentLink : m_linkedPointers )
    {
        if( GeometryInstance* parent = getLinkToGeometryFrom<GeometryInstance>( parentLink ) )
        {
            if( parent->getMaterialCount() != (int)m_numMaterials )
            {
                std::stringstream ss;
                ss << "GeometryTriangles: number of materials (" << m_numMaterials << ") "
                   << "must match the number of materials set on the GeometryInstance node ("
                   << parent->getMaterialCount() << ")";
                throw ValidationError( RT_EXCEPTION_INFO, ss.str() );
            }
        }
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to GeometryTriangles" );
    }

    Geometry::validate();
}

void optix::GeometryTriangles::setTriangleIndices( Buffer* indexBuffer, RTsize indexBufferByteOffset, RTsize triIndicesByteStride, RTformat triIndicesFormat )
{
    if( indexBuffer != m_indexBuffer.get() )
    {
        m_indexBuffer.set( this, indexBuffer );
        if( indexBuffer )
            m_indexBuffer->markAsBindlessForInternalUse();
    }
    m_isIndexedTriangles = indexBuffer != nullptr;
    if( m_isIndexedTriangles )
    {
        m_indexBufferByteOffset = indexBufferByteOffset;
        m_triIndicesByteStride  = triIndicesByteStride;
        m_triIndicesFormat      = triIndicesFormat;

        // see comment in setMotiolwertices
        m_indexBufferByteOffsetInIS = (long long)m_indexBufferByteOffset - m_primitiveIndexOffset * (long long)m_triIndicesByteStride;
    }

    subscribeForValidation();
    FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
    writeRecord();
}

void optix::GeometryTriangles::setVertices( unsigned int numVertices,
                                            Buffer*      vertexBuffer,
                                            RTsize       vertexBufferByteOffset,
                                            RTsize       vertexByteStride,
                                            RTformat     positionFormat )
{
    setMotiolwertices( numVertices, vertexBuffer, vertexBufferByteOffset, vertexByteStride, 0, positionFormat );
}

void optix::GeometryTriangles::setMotiolwertices( unsigned int numVertices,
                                                  Buffer*      vertexBuffer,
                                                  RTsize       vertexBufferByteOffset,
                                                  RTsize       vertexByteStride,
                                                  RTsize       vertexMotionStepByteStride,
                                                  RTformat     positionFormat )
{
    // TODO kill m_multiBufferIds?
    m_vertexBuffers.resize( 1 );
    auto& internal = m_vertexBuffers.front();
    if( vertexBuffer != internal.get() )
    {
        internal.set( this, vertexBuffer );
        internal->markAsBindlessForInternalUse();
    }
    m_numVertices                = numVertices;
    m_vertexBufferByteOffset     = vertexBufferByteOffset;
    m_vertexByteStride           = vertexByteStride;
    m_vertexMotionStepByteStride = vertexMotionStepByteStride;
    m_positionFormat             = positionFormat;

    {
        // Warning, this may even be negative!
        // The reason is that primitiveIndexOffset and the buffer byte offset are decoupled!
        // IS and AABB will use the primIdx, which includes the primIndexOffset.
        // We need to factor out the primIndexOffset since we already have the buffer byte offset.
        // The buffer will be accessed at [(m_primitiveIndexOffset + primIdx) * stride + byteOffset]
        // The offset used in the program must be adjusted by the byteOffset as well as the indexOffset set by the user.
        // This must be allowed to be zero.
        m_vertexBufferByteOffsetInIS =
            (long long)m_vertexBufferByteOffset - m_primitiveIndexOffset * (long long)m_vertexByteStride * 3;

        // we need to set m_vertexBufferByteOffsetInIS instead of m_vertexBufferByteOffset if no index buffer is used
        // defer that decision to validation!
    }

    FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();

    subscribeForValidation();
    writeRecord();
}

void GeometryTriangles::setMotiolwerticesMultiBuffer( unsigned int numVertices,
                                                      Buffer**     vertexBuffers,
                                                      unsigned int vertexBufferCount,
                                                      RTsize       vertexBufferByteOffset,
                                                      RTsize       vertexByteStride,
                                                      RTformat     positionFormat )
{
    if( vertexBufferCount == 1 )
    {
        setVertices( numVertices, vertexBuffers[0], vertexBufferByteOffset, vertexByteStride, positionFormat );
        return;
    }
    m_numVertices                = numVertices;
    m_vertexBufferByteOffset     = vertexBufferByteOffset;
    m_vertexByteStride           = vertexByteStride;
    m_vertexMotionStepByteStride = MULTI_BUFFER_MOTION_STRIDE;
    m_positionFormat             = positionFormat;

    m_vertexBuffers.resize( vertexBufferCount );
    m_multiBufferIds.setFormat( RT_FORMAT_INT );
    m_multiBufferIds.setSize1D( vertexBufferCount );
    auto ids = static_cast<int*>( m_multiBufferIds.map( MAP_WRITE_DISCARD ) );
    for( unsigned int i = 0; i < vertexBufferCount; ++i )
    {
        m_vertexBuffers[i].set( this, vertexBuffers[i] );
        vertexBuffers[i]->markAsBindlessForInternalUse();
        ids[i] = vertexBuffers[i]->getId();
    }
    m_multiBufferIds.unmap();

    {
        // Warning, this may even be negative!
        // The reason is that primitiveIndexOffset and the buffer byte offset are decoupled!
        // IS and AABB will use the primIdx, which includes the primIndexOffset.
        // We need to factor out the primIndexOffset since we already have the buffer byte offset.
        // The buffer will be accessed at [(m_primitiveIndexOffset + primIdx) * stride + byteOffset]
        // The offset used in the program must be adjusted by the byteOffset as well as the indexOffset set by the user.
        // This must be allowed to be zero.
        m_vertexBufferByteOffsetInIS =
            (long long)m_vertexBufferByteOffset - m_primitiveIndexOffset * (long long)m_vertexByteStride * 3;

        // we need to set m_vertexBufferByteOffsetInIS instead of m_vertexBufferByteOffset if no index buffer is used
        // defer that decision to validation!
    }

    FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();

    subscribeForValidation();
    writeRecord();
}

void GeometryTriangles::FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS()
{
    if( algorithm::any_of( m_vertexBuffers, []( const TrianglesPtr& p ) { return p.get(); } ) )
    {
        setSpecializedPrograms();
    }
}

Program* GeometryTriangles::getDefaultAttributeProgram()
{
    return m_context->getSharedProgramManager()->getProgram( data::getGeometryTrianglesProgramsSources(),
                                                             "defaultTriangleAttributeProgram" );
}

void GeometryTriangles::detachDefaultAttributeProgramFromParents( Context* context )
{
    if( Program* prog = context->getSharedProgramManager()->getCachedProgram( data::getGeometryTrianglesProgramsSources(),
                                                                              "defaultTriangleAttributeProgram" ) )
    {
        prog->detachFromParents();
    }
}

void GeometryTriangles::setPrimitiveIndexOffset( int primitiveIndexOffset )
{
    Geometry::setPrimitiveIndexOffset( primitiveIndexOffset );

    // see comment in setMotionTriangles
    m_indexBufferByteOffsetInIS = (long long)m_indexBufferByteOffset - m_primitiveIndexOffset * (long long)m_triIndicesByteStride;
    m_vertexBufferByteOffsetInIS =
        (long long)m_vertexBufferByteOffset - m_primitiveIndexOffset * (long long)m_vertexByteStride * 3;
    // we may need to set m_varIndexBufferByteOffset, but we defer that decision to validation!

    subscribeForValidation();
    FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
    writeRecord();
}

bool GeometryTriangles::isIndexedTriangles() const
{
    return m_isIndexedTriangles;
}

void GeometryTriangles::setAttributeProgram( Program* program )
{
    // Validate the semantic type before setting the program to avoid
    // exceptions during graph property propagation.
    if( program )
        program->validateSemanticType( ST_ATTRIBUTE );

    // Geometry program properties
    //
    // Semantic type:                    originates here
    // Attachment:                       propagates from parent
    // Direct caller:                    propagates from intersection program
    // Unresolved attribute references:  propagates from child
    // Unresolved references:            propagates from child

    ProgramRoot root( getScopeID(), ST_ATTRIBUTE, 0 );

    if( Program* oldProgram = m_attributeProgram.get() )
    {
        // Remove properties from old program before updating the pointer
        oldProgram->attachOrDetachProperty_UnresolvedReference( this, root, false );
        oldProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, false );
        if( m_intersectionProgram )
            m_intersectionProgram->attachOrDetachProperty_DirectCaller( oldProgram, false );
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_ATTRIBUTE, false );
    }

    m_attributeProgram.set( this, program );

    if( Program* newProgram = m_attributeProgram.get() )
    {
        // Add new properties
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_ATTRIBUTE, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
        if( m_intersectionProgram )
            m_intersectionProgram->attachOrDetachProperty_DirectCaller( newProgram, true );
        newProgram->attachOrDetachProperty_UnresolvedAttributeReference( this, true );
        newProgram->attachOrDetachProperty_UnresolvedReference( this, root, true );
    }


    subscribeForValidation();
    writeRecord();
}

Program* GeometryTriangles::getAttributeProgram() const
{
    return m_attributeProgram.get();
}

void GeometryTriangles::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( link == &m_indexBuffer )
    {
        m_indexBuffer.set( this, nullptr );
        subscribeForValidation();
    }
    else if( link == &m_materialIndexBuffer )
    {
        m_materialIndexBuffer.set( this, nullptr );
        subscribeForValidation();
    }
    else if( link == &m_attributeProgram )
    {
        Program* defaultAttributeProgram = getDefaultAttributeProgram();
        if( m_attributeProgram == defaultAttributeProgram )
            setAttributeProgram( m_context->getSharedProgramManager()->getNullProgram() );
        else
            setAttributeProgram( defaultAttributeProgram );
    }
    else
    {
        auto pos = algorithm::find_if( m_vertexBuffers, [link]( const TrianglesPtr& p ) { return link == &p; } );
        if( pos != m_vertexBuffers.end() )
        {
            pos->reset();
            subscribeForValidation();
        }
        else
        {
            Geometry::detachLinkedChild( link );
        }
    }
}

//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void GeometryTriangles::sendPropertyDidChange_Attachment( bool added ) const
{
    Geometry::sendPropertyDidChange_Attachment( added );
    if( m_attributeProgram )
        m_attributeProgram->receivePropertyDidChange_Attachment( added );
}

void GeometryTriangles::intersectionProgramDidChange( Program* program, bool added )
{
    if( !m_attributeProgram )
        return;
    if( !program )
        return;

    program->attachOrDetachProperty_DirectCaller( m_attributeProgram.get(), added );
}


void GeometryTriangles::bufferFormatDidChange()
{
    subscribeForValidation();
}

const GeometryTriangles::VertexBuffers& GeometryTriangles::getVertexBuffers() const
{
    return m_vertexBuffers;
}

unsigned int GeometryTriangles::getNumVertices() const
{
    return m_numVertices;
}

unsigned long long GeometryTriangles::getVertexByteStride() const
{
    return m_vertexByteStride;
}

unsigned long long GeometryTriangles::getVertexMotionByteStride() const
{
    return m_vertexMotionStepByteStride;
}

unsigned long long GeometryTriangles::getVertexBufferByteOffset() const
{
    return m_vertexBufferByteOffset;
}

RTformat GeometryTriangles::getPositionFormat() const
{
    return m_positionFormat;
}

bool GeometryTriangles::hasMultiBufferMotion() const
{
    return MULTI_BUFFER_MOTION_STRIDE == m_vertexMotionStepByteStride;
}

Buffer* GeometryTriangles::getIndexBuffer() const
{
    return m_indexBuffer.get();
}

unsigned int GeometryTriangles::getTriIndicesByteStride() const
{
    return m_triIndicesByteStride;
}

unsigned long long GeometryTriangles::getIndexBufferByteOffset() const
{
    return m_indexBufferByteOffset;
}

RTformat GeometryTriangles::getTriIndicesFormat() const
{
    return m_triIndicesFormat;
}

Buffer* GeometryTriangles::getMaterialIndexBuffer() const
{
    return m_materialIndexBuffer.get();
}

unsigned long long GeometryTriangles::getMaterialIndexBufferByteOffset() const
{
    return m_materialIndexBufferByteOffset;
}

unsigned int GeometryTriangles::getMaterialIndexByteStride() const
{
    return m_materialIndexByteStride;
}

RTformat GeometryTriangles::getMaterialIndexFormat() const
{
    return m_materialIndexFormat;
}

RTgeometryflags optix::GeometryTriangles::getFlags() const
{
    if( m_geometryFlags.size() == 1 )
        return m_geometryFlags[0];
    return RT_GEOMETRY_FLAG_NONE;
}

const std::vector<RTgeometryflags>& GeometryTriangles::getGeometryFlags() const
{
    return m_geometryFlags;
}

RTgeometrybuildflags GeometryTriangles::buildFlags() const
{
    return m_buildFlags;
}

void GeometryTriangles::setMotionSteps( int n )
{
    const bool wasUsingMotion = getMotionSteps() > 1;

    Geometry::setMotionSteps( n );

    const bool isUsingMotion = getMotionSteps() > 1;

    if( wasUsingMotion != isUsingMotion )
    {
        reallocateRecord();
        FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
        writeRecord();
    }
}

void GeometryTriangles::setMotionRange( float timeBegin, float timeEnd )
{
    Geometry::setMotionRange( timeBegin, timeEnd );

    if( getMotionSteps() > 1 )
    {
        FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
        writeRecord();
    }
}

void GeometryTriangles::setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    Geometry::setMotionBorderMode( beginMode, endMode );

    if( getMotionSteps() > 1 )
    {
        FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
        writeRecord();
    }
}

void GeometryTriangles::setPreTransformMatrix( const float* matrix, bool transpose )
{
    if( !matrix )
    {
        m_transform.reset();
        return;
    }
    if( !transpose )
    {
        m_transform = std::unique_ptr<Matrix<4, 3>>( new Matrix<4, 3>( matrix ) );
    }
    else
    {
        m_transform  = std::unique_ptr<Matrix<4, 3>>( new Matrix<4, 3>() );
        *m_transform = reinterpret_cast<const Matrix<3, 4>*>( matrix )->transpose();
    }
    FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
}

void GeometryTriangles::getPreTransformMatrix( float* matrix, bool transpose ) const
{
    if( m_transform )
    {
        if( !transpose )
        {
            Matrix<4, 3>* dest = reinterpret_cast<Matrix<4, 3>*>( matrix );
            *dest = *m_transform;
        }
        else
        {
            Matrix<3, 4>* dest = reinterpret_cast<Matrix<3, 4>*>( matrix );
            *dest = m_transform->transpose();
        }
    }
    else
    {
        if( !transpose )
        {
            for( int i    = 0; i < 12; ++i )
                matrix[i] = i % 5 == 0 ? 1.0f : 0.0f;
        }
        else
        {
            for( int i    = 0; i < 12; ++i )
                matrix[i] = i % 4 == 0 ? 1.0f : 0.0f;
        }
    }
}

void GeometryTriangles::setBuildFlags( RTgeometrybuildflags buildFlags )
{
    m_buildFlags = buildFlags;
}

void GeometryTriangles::setMaterialCount( unsigned int numMaterials )
{
    if( numMaterials == 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material count must be >= 1", numMaterials );
    m_numMaterials = numMaterials;
    m_geometryFlags.resize( m_numMaterials );
}

unsigned int GeometryTriangles::getMaterialCount() const
{
    return m_numMaterials;
}

void GeometryTriangles::setMaterialIndices( Buffer*  materialIndexBuffer,
                                            RTsize   materialIndexBufferByteOffset,
                                            RTsize   materialIndexByteStride,
                                            RTformat materialIndexFormat )
{
    if( materialIndexBuffer != m_materialIndexBuffer.get() )
    {
        m_materialIndexBuffer.set( this, materialIndexBuffer );
        m_materialIndexBuffer->markAsBindlessForInternalUse();
    }
    m_materialIndexBufferByteOffset = materialIndexBufferByteOffset;
    m_materialIndexByteStride       = materialIndexByteStride;
    m_materialIndexFormat           = materialIndexFormat;
}

void GeometryTriangles::setFlagsPerMaterial( unsigned int material_index, RTgeometryflags flags )
{
    if( material_index >= m_numMaterials )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material index out of bounds", material_index );

    m_geometryFlags[material_index] = flags;
}

RTgeometryflags GeometryTriangles::getFlagsPerMaterial( unsigned int material_index ) const
{
    if( material_index >= m_numMaterials )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Material index out of bounds", material_index );

    return m_geometryFlags[material_index];
}

//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t GeometryTriangles::getRecordBaseSize() const
{
    return getMotionSteps() > 1 ? sizeof( cort::MotionGeometryTrianglesRecord ) : sizeof( cort::GeometryTrianglesRecord );
}

int GeometryTriangles::getVertexBufferId() const
{
    if( m_vertexBuffers.empty() || !m_vertexBuffers.front() )
        return 0;

    return hasMultiBufferMotion() ? m_multiBufferIds.getId() : m_vertexBuffers.front()->getId();
}

void GeometryTriangles::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    Geometry::writeRecord();
    cort::GeometryTrianglesRecord* g = getObjectRecord<cort::GeometryTrianglesRecord>();
    RT_ASSERT( g != nullptr );

    g->intersectOrAttribute = getSafeOffset( m_attributeProgram.get() );
    if( m_attributeProgram )
        g->attributeKind = m_attributeProgram->get32bitAttributeKind();

    g->vertexBufferID     = getVertexBufferId();
    g->vertexBufferOffset = m_isIndexedTriangles ? m_vertexBufferByteOffset : m_vertexBufferByteOffsetInIS;
    g->vertexBufferStride = m_vertexByteStride;
    g->indexBufferID      = m_indexBuffer ? m_indexBuffer->getId() : 0;
    g->indexBufferOffset  = m_indexBufferByteOffsetInIS;
    g->indexBufferStride  = m_triIndicesByteStride;

    if( getMotionSteps() > 1 )
    {
        cort::MotionGeometryTrianglesRecord* mg = (cort::MotionGeometryTrianglesRecord*)g;

        mg->vertexBufferMotionStride = m_vertexMotionStepByteStride;
        mg->motionNumIntervals       = m_motionSteps - 1;
        mg->timeBegin                = m_timeBegin;
        mg->timeEnd                  = m_timeEnd;
        mg->motionBorderModeBegin    = m_beginBorderMode;
        mg->motionBorderModeEnd      = m_endBorderMode;
    }
}
