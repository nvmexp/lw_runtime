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

#pragma once

#include <Objects/Buffer.h>
#include <Objects/Geometry.h>

namespace optix {

class GeometryTriangles : public Geometry
{
  public:
    using TrianglesPtr  = LinkedPtr<GeometryTriangles, Buffer>;
    using VertexBuffers = std::vector<TrianglesPtr>;


    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    GeometryTriangles( Context* context );
    ~GeometryTriangles() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------

    void validate() const override;

    void setTriangleIndices( Buffer* indexBuffer, RTsize indexBufferByteOffset, RTsize triIndicesByteStride, RTformat triIndicesFormat );

    void setAttributeProgram( Program* program );
    Program* getAttributeProgram() const;

    void setVertices( unsigned int numVertices, Buffer* vertexBuffer, RTsize vertexBufferByteOffset, RTsize vertexByteStride, RTformat positionFormat );

    void setMotiolwertices( unsigned int numVertices,
                            Buffer*      vertexBuffer,
                            RTsize       vertexBufferByteOffset,
                            RTsize       vertexByteStride,
                            RTsize       vertexMotionStepByteStride,
                            RTformat     positionFormat );

    void setMotiolwerticesMultiBuffer( unsigned int numVertices,
                                       Buffer**     vertexBuffers,
                                       unsigned int vertexBufferCount,
                                       RTsize       vertexBufferByteOffset,
                                       RTsize       vertexByteStride,
                                       RTformat     positionFormat );

    // 3 row x 4 col, row-major affine transformation matrix, if input/output is a column-major matrix, transpose must be true
    void setPreTransformMatrix( const float* matrix, bool transpose );
    void getPreTransformMatrix( float* matrix, bool transpose ) const;

    void setBuildFlags( RTgeometrybuildflags buildFlags );

    void setMaterialCount( unsigned int numMaterials );
    unsigned int getMaterialCount() const;
    void setMaterialIndices( Buffer*  materialIndexBuffer,
                             RTsize   materialIndexBufferByteOffset,
                             RTsize   materialIndexByteStride,
                             RTformat materialIndexFormat );
    void setFlagsPerMaterial( unsigned int materialIndex, RTgeometryflags flags );
    RTgeometryflags getFlagsPerMaterial( unsigned int materialIndex ) const;

    //------------------------------------------------------------------------
    // WARNING, the following functions shadow the Geometry:: version of it.
    // None of these functions is declared virtual!

    // we need to override it here to properly handle the used offsets in the motion blur IS, AABB program
    void setPrimitiveIndexOffset( int primitiveIndexOffset );

    // Motion blur data
    // override required to update program variables
    void setMotionSteps( int numMotionSteps );
    void setMotionRange( float timeBegin, float timeEnd );
    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );


    //------------------------------------------------------------------------

    //------------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------------

    const VertexBuffers& getVertexBuffers() const;
    unsigned int         getNumVertices() const;
    unsigned long long   getVertexBufferByteOffset() const;
    unsigned long long   getVertexByteStride() const;
    unsigned long long   getVertexMotionByteStride() const;
    RTformat             getPositionFormat() const;
    bool                 hasMultiBufferMotion() const;

    Buffer*            getIndexBuffer() const;
    unsigned long long getIndexBufferByteOffset() const;
    unsigned int       getTriIndicesByteStride() const;
    RTformat           getTriIndicesFormat() const;

    Buffer*            getMaterialIndexBuffer() const;
    unsigned long long getMaterialIndexBufferByteOffset() const;
    unsigned int       getMaterialIndexByteStride() const;
    RTformat           getMaterialIndexFormat() const;

    // 'global' flags if there is only one material slot
    RTgeometryflags getFlags() const override;
    // per material slot flags
    const std::vector<RTgeometryflags>& getGeometryFlags() const;

    //------------------------------------------------------------------------
    // Internal helpers
    //------------------------------------------------------------------------

    bool                 isIndexedTriangles() const;
    RTgeometrybuildflags buildFlags() const;


    //------------------------------------------------------------------------
    // Buffer events
    //------------------------------------------------------------------------
    void bufferFormatDidChange();

    // Buffer and Program management
    // LinkedPtr relationship management
    void detachLinkedChild( const LinkedPtr_Link* link ) override;
    static void detachDefaultAttributeProgramFromParents( Context* context );

  protected:
    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;


    //------------------------------------------------------------------------
    // Allow GeometryTriangle a chance to do something when the intersection program is changed
    //------------------------------------------------------------------------
  protected:
    void intersectionProgramDidChange( Program* program, bool added ) override;

  protected:
    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   writeRecord() const override;

  private:
    void getPrograms( Program*& aabbProg, Program*& intersectProg );
    void     setSpecializedPrograms();
    void     FIXME_MAKE_DEFERRED_SET_SPECIALIZED_PROGRAMS();
    Program* getDefaultAttributeProgram();
    int      getVertexBufferId() const;

  private:
    // not supported by GeometryTriangles, hide methods
    using Geometry::getBoundingBoxProgram;
    using Geometry::getIntersectionProgram;
    using Geometry::setBoundingBoxProgram;
    using Geometry::setFlags;
    using Geometry::setIntersectionProgram;

    friend class NodegraphPrinter;

  private:
    static const unsigned long long MULTI_BUFFER_MOTION_STRIDE = ~(unsigned long long)0;

    VertexBuffers      m_vertexBuffers;
    Buffer             m_multiBufferIds;
    unsigned int       m_numVertices                = 0;
    unsigned long long m_vertexBufferByteOffset     = 0;
    unsigned long long m_vertexByteStride           = 0;
    unsigned long long m_vertexMotionStepByteStride = 0;
    RTformat           m_positionFormat;
    long long          m_vertexBufferByteOffsetInIS = 0;  // adjusted offset to factor out primitive index offset

    TrianglesPtr       m_indexBuffer;
    unsigned long long m_indexBufferByteOffset = 0;
    unsigned long long m_triIndicesByteStride  = 0;
    RTformat           m_triIndicesFormat;
    long long          m_indexBufferByteOffsetInIS = 0;  // adjusted offset to factor out primitive index offset

    TrianglesPtr       m_materialIndexBuffer;
    unsigned int       m_numMaterials                  = 0;
    unsigned long long m_materialIndexBufferByteOffset = 0;
    unsigned long long m_materialIndexByteStride       = 0;
    RTformat           m_materialIndexFormat;

    std::vector<RTgeometryflags> m_geometryFlags;

    std::unique_ptr<Matrix<4, 3>> m_transform;

    RTgeometrybuildflags m_buildFlags;
    bool                 m_isIndexedTriangles = false;

    LinkedPtr<Geometry, Program> m_attributeProgram;

    Program* m_dummyFixedFunctionIntersectionProgram;

    Program* m_motionTrianglesBoundingBoxProgram;
    Program* m_motionIndexedTrianglesBoundingBoxProgram;
    Program* m_motionTrianglesIntersectionProgram;
    Program* m_motionIndexedTrianglesIntersectionProgram;

    Variable* m_varMotionIntervals   = nullptr;
    Variable* m_varMotionRange       = nullptr;
    Variable* m_varMotionBorderModes = nullptr;

    Variable* m_varVertexBufferId               = nullptr;
    Variable* m_varVertexBufferByteOffset       = nullptr;
    Variable* m_varVertexBufferByteStride       = nullptr;
    Variable* m_varVertexBufferMotionByteStride = nullptr;
    Variable* m_varIndexBufferId                = nullptr;
    Variable* m_varIndexBufferByteOffset        = nullptr;
    Variable* m_varIndexBufferByteStride        = nullptr;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GEOMETRY_TRIANGLES};
};

inline bool GeometryTriangles::isA( ManagedObjectType type ) const
{
    return type == m_objectType || Geometry::isA( type );
}

}  // namespace optix
