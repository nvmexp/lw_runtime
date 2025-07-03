#include <srcTests.h>

#include <Context/Context.h>

#include <Objects/Acceleration.h>
#include <Objects/Buffer.h>
#include <Objects/CommandList.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GraphNode.h>
#include <Objects/Group.h>
#include <Objects/LexicalScope.h>
#include <Objects/ManagedObject.h>
#include <Objects/Material.h>
#include <Objects/PostprocessingStage.h>
#include <Objects/PostprocessingStageDenoiser.h>
#include <Objects/PostprocessingStageSSIMPredictor.h>
#include <Objects/PostprocessingStageTonemap.h>
#include <Objects/Program.h>
#include <Objects/Selector.h>
#include <Objects/StreamBuffer.h>
#include <Objects/TextureSampler.h>
#include <Objects/Transform.h>
#include <Util/MakeUnique.h>

using namespace optix;
using namespace testing;

namespace {

//------------------------------------------------------------------------------
class ManagedObjectCastTest : public Test
{
  public:
    static void SetUpTestCase()
    {
        m_context = new Context();
        m_context->setEntryPointCount( 1 );
        m_context->setRayTypeCount( 1 );
    }

    static void TearDownTestCase()
    {
        m_context->tearDown();
        delete m_context;
    }

    static Context* m_context;
};

Context* ManagedObjectCastTest::m_context = nullptr;

//------------------------------------------------------------------------------

template <typename T>
ManagedObjectType staticManagedObjectTypeTester()
{
    return T::m_objectType;
}

// Downcast nullptr
TEST_F( ManagedObjectCastTest, nullptrToAnything )
{
    ManagedObject* mo = nullptr;

    Geometry* geometry = managedObjectCast<Geometry>( mo );

    EXPECT_EQ( geometry, nullptr );
}

TEST_F( ManagedObjectCastTest, nullptrToAnythingConst )
{
    ManagedObject* mo = nullptr;

    const Geometry* geometry = managedObjectCast<const Geometry>( mo );

    EXPECT_EQ( geometry, nullptr );
}

// Abstract Downcasts ----------------------------------------------------------

// ManagedObject to LexicalScope
TEST_F( ManagedObjectCastTest, managedObjectToLexicalScope )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Program>( m_context );

    LexicalScope* ls = managedObjectCast<LexicalScope>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_PROGRAM ) );
    EXPECT_NE( ls, nullptr );
}

// ManagedObject to GraphNode
TEST_F( ManagedObjectCastTest, managedObjectToGraphNode )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Transform>( m_context );

    GraphNode* gn = managedObjectCast<GraphNode>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( gn, nullptr );
}

// ManagedObject to AbstractGroup
TEST_F( ManagedObjectCastTest, managedObjectToAbstractGroup )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Group>( m_context );

    AbstractGroup* ag = managedObjectCast<AbstractGroup>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_ABSTRACT_GROUP ) );
    EXPECT_NE( ag, nullptr );
}

// ManagedObject to PostprocessingStage
TEST_F( ManagedObjectCastTest, managedObjectToPostProcessingStage )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<PostprocessingStageDenoiser>( m_context, nullptr );

    PostprocessingStage* pps = managedObjectCast<PostprocessingStage>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE ) );
    EXPECT_NE( pps, nullptr );
}

// LexicalScopeToGraphNode
TEST_F( ManagedObjectCastTest, lexicalScopeToGraphNode )
{
    std::unique_ptr<LexicalScope> ls = makeUnique<Transform>( m_context );

    GraphNode* gn = managedObjectCast<GraphNode>( ls.get() );

    EXPECT_TRUE( ls->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( gn, nullptr );
}

// LexicalScopeToAbstractGroup
TEST_F( ManagedObjectCastTest, lexicalScopeToAbstractGroup )
{
    std::unique_ptr<LexicalScope> ls = makeUnique<GeometryGroup>( m_context );

    AbstractGroup* ag = managedObjectCast<AbstractGroup>( ls.get() );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( ag, nullptr );
}

// LexicalScopeToPostProcessingStage
TEST_F( ManagedObjectCastTest, lexicalScopeToPostProcessingStage )
{
    std::unique_ptr<LexicalScope> ls = makeUnique<PostprocessingStageDenoiser>( m_context, nullptr );

    PostprocessingStage* pps = managedObjectCast<PostprocessingStage>( ls.get() );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE ) );
    EXPECT_NE( pps, nullptr );
}

// GraphNode to AbstractGroup
TEST_F( ManagedObjectCastTest, GraphNodeToAbstractGroup )
{
    std::unique_ptr<Group> gn = makeUnique<Group>( m_context );

    AbstractGroup* ag = managedObjectCast<AbstractGroup>( static_cast<GraphNode*>( gn.get() ) );

    EXPECT_TRUE( gn->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( ag, nullptr );
}

// Concrete ManagedObject Downcasts ----------------------------------------------------------

// Buffer cast

TEST_F( ManagedObjectCastTest, staticManagedObjectTypeBuffer )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Buffer>(), MO_TYPE_BUFFER );
}

TEST_F( ManagedObjectCastTest, managedObjectToBuffer )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Buffer>( m_context, 0 );

    Buffer* buffer = managedObjectCast<Buffer>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_BUFFER ) );
    EXPECT_NE( buffer, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToBufferConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Buffer>( m_context, 0 );

    const Buffer* buffer = managedObjectCast<const Buffer>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_BUFFER ) );
    EXPECT_NE( buffer, nullptr );
}

// StreamBuffer cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeStreamBuffer )
{
    EXPECT_EQ( staticManagedObjectTypeTester<StreamBuffer>(), MO_TYPE_STREAM_BUFFER );
}

TEST_F( ManagedObjectCastTest, managedObjectToStreamBuffer )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<StreamBuffer>( m_context );

    StreamBuffer* sb = managedObjectCast<StreamBuffer>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_STREAM_BUFFER ) );
    EXPECT_NE( sb, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToStreamBufferConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<StreamBuffer>( m_context );

    const StreamBuffer* sb = managedObjectCast<const StreamBuffer>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_STREAM_BUFFER ) );
    EXPECT_NE( sb, nullptr );
}

// CommandList cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeCommandList )
{
    EXPECT_EQ( staticManagedObjectTypeTester<CommandList>(), MO_TYPE_COMMAND_LIST );
}

TEST_F( ManagedObjectCastTest, managedObjectToCommandList )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<CommandList>( m_context );

    CommandList* cl = managedObjectCast<CommandList>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_COMMAND_LIST ) );
    EXPECT_NE( cl, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToCommandListConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<CommandList>( m_context );

    const CommandList* cl = managedObjectCast<const CommandList>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_COMMAND_LIST ) );
    EXPECT_NE( cl, nullptr );
}

// TextureSampler cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeTextureSampler )
{
    EXPECT_EQ( staticManagedObjectTypeTester<TextureSampler>(), MO_TYPE_TEXTURE_SAMPLER );
}

TEST_F( ManagedObjectCastTest, managedObjectToTextureSampler )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<TextureSampler>( m_context );

    TextureSampler* ts = managedObjectCast<TextureSampler>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_TEXTURE_SAMPLER ) );
    EXPECT_NE( ts, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToTextureSamplerConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<TextureSampler>( m_context );

    const TextureSampler* ts = managedObjectCast<const TextureSampler>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_TEXTURE_SAMPLER ) );
    EXPECT_NE( ts, nullptr );
}

// Material cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeMaterial )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Material>(), MO_TYPE_MATERIAL );
}

TEST_F( ManagedObjectCastTest, managedObjectToMaterial )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Material>( m_context );

    Material* mat = managedObjectCast<Material>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_MATERIAL ) );
    EXPECT_NE( mat, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToMaterialConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Material>( m_context );

    const Material* mat = managedObjectCast<const Material>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_MATERIAL ) );
    EXPECT_NE( mat, nullptr );
}

// Program cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeProgram )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Program>(), MO_TYPE_PROGRAM );
}

TEST_F( ManagedObjectCastTest, managedObjectToProgram )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Program>( m_context );

    Program* program = managedObjectCast<Program>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_PROGRAM ) );
    EXPECT_NE( program, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToProgramConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Program>( m_context );

    const Program* program = managedObjectCast<const Program>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_PROGRAM ) );
    EXPECT_NE( program, nullptr );
}

// Acceleration cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeAcceleration )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Acceleration>(), MO_TYPE_ACCELERATION );
}

TEST_F( ManagedObjectCastTest, managedObjectToAcceleration )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Acceleration>( m_context );

    Acceleration* accel = managedObjectCast<Acceleration>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_ACCELERATION ) );
    EXPECT_NE( accel, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToAccelerationConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Acceleration>( m_context );

    const Acceleration* accel = managedObjectCast<const Acceleration>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_ACCELERATION ) );
    EXPECT_NE( accel, nullptr );
}

// GeometryInstance cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeGeometryInstance )
{
    EXPECT_EQ( staticManagedObjectTypeTester<GeometryInstance>(), MO_TYPE_GEOMETRY_INSTANCE );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometryInstance )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<GeometryInstance>( m_context );

    GeometryInstance* gi = managedObjectCast<GeometryInstance>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_INSTANCE ) );
    EXPECT_NE( gi, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometryInstanceConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<GeometryInstance>( m_context );

    const GeometryInstance* gi = managedObjectCast<const GeometryInstance>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_INSTANCE ) );
    EXPECT_NE( gi, nullptr );
}

// Geometry cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeGeometry )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Geometry>(), MO_TYPE_GEOMETRY );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometry )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Geometry>( m_context, false );

    Geometry* geometry = managedObjectCast<Geometry>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY ) );
    EXPECT_NE( geometry, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometryConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Geometry>( m_context, false );

    const Geometry* geometry = managedObjectCast<const Geometry>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY ) );
    EXPECT_NE( geometry, nullptr );
}

// GeometryTriangles cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeGeometryTriangles )
{
    EXPECT_EQ( staticManagedObjectTypeTester<GeometryTriangles>(), MO_TYPE_GEOMETRY_TRIANGLES );
}

TEST_F( ManagedObjectCastTest, managedObjctToGeometryTriangles )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<GeometryTriangles>( m_context );

    GeometryTriangles* gt = managedObjectCast<GeometryTriangles>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjctToGeometryTrianglesConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<GeometryTriangles>( m_context );

    const GeometryTriangles* gt = managedObjectCast<const GeometryTriangles>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

// PostprocessingStageDenoiser cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypePPSDenoiser )
{
    EXPECT_EQ( staticManagedObjectTypeTester<PostprocessingStageDenoiser>(), MO_TYPE_POSTPROC_STAGE_DENOISER );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSDenoiser )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<PostprocessingStageDenoiser>( m_context, nullptr );

    PostprocessingStageDenoiser* ppsd = managedObjectCast<PostprocessingStageDenoiser>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_DENOISER ) );
    EXPECT_NE( ppsd, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSDenoiserConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<PostprocessingStageDenoiser>( m_context, nullptr );

    const PostprocessingStageDenoiser* ppsd = managedObjectCast<const PostprocessingStageDenoiser>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_DENOISER ) );
    EXPECT_NE( ppsd, nullptr );
}

// PostprocessingStageSSIM cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypePPSSSIM )
{
    EXPECT_EQ( staticManagedObjectTypeTester<PostprocessingStageSSIMPredictor>(), MO_TYPE_POSTPROC_STAGE_SSIM );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSSSIM )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<PostprocessingStageSSIMPredictor>( m_context, nullptr );

    PostprocessingStageSSIMPredictor* pps = managedObjectCast<PostprocessingStageSSIMPredictor>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_SSIM ) );
    EXPECT_NE( pps, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSSSIMConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<PostprocessingStageSSIMPredictor>( m_context, nullptr );

    const PostprocessingStageSSIMPredictor* pps = managedObjectCast<const PostprocessingStageSSIMPredictor>( mo.get() );
    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_SSIM ) );
    EXPECT_NE( pps, nullptr );
}

// PostprocessingStageTonemap cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypePPSToneMap )
{
    EXPECT_EQ( staticManagedObjectTypeTester<PostprocessingStageTonemap>(), MO_TYPE_POSTPROC_STAGE_TONEMAP );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSToneMap )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<PostprocessingStageTonemap>( m_context );

    PostprocessingStageTonemap* pps = managedObjectCast<PostprocessingStageTonemap>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_TONEMAP ) );
    EXPECT_NE( pps, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToPPSToneMapConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<PostprocessingStageTonemap>( m_context );

    const PostprocessingStageTonemap* pps = managedObjectCast<const PostprocessingStageTonemap>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_POSTPROC_STAGE_TONEMAP ) );
    EXPECT_NE( pps, nullptr );
}

// Selector cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeSelector )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Selector>(), MO_TYPE_SELECTOR );
}

TEST_F( ManagedObjectCastTest, managedObjectToSelector )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Selector>( m_context );

    Selector* selector = managedObjectCast<Selector>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_SELECTOR ) );
    EXPECT_NE( selector, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToSelectorConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Selector>( m_context );

    const Selector* selector = managedObjectCast<const Selector>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_SELECTOR ) );
    EXPECT_NE( selector, nullptr );
}

// Transform cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeTransform )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Transform>(), MO_TYPE_TRANSFORM );
}

TEST_F( ManagedObjectCastTest, managedObjectToTransform )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Transform>( m_context );

    Transform* transform = managedObjectCast<Transform>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( transform, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToTransformConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Transform>( m_context );

    const Transform* transform = managedObjectCast<const Transform>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( transform, nullptr );
}

// Group cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeGroup )
{
    EXPECT_EQ( staticManagedObjectTypeTester<Group>(), MO_TYPE_GROUP );
}

TEST_F( ManagedObjectCastTest, managedObjectToGroup )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<Group>( m_context );

    Group* group = managedObjectCast<Group>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToGroupConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<Group>( m_context );

    const Group* group = managedObjectCast<const Group>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

// GeometryGroup cast
TEST_F( ManagedObjectCastTest, staticManagedObjectTypeGeometryGroup )
{
    EXPECT_EQ( staticManagedObjectTypeTester<GeometryGroup>(), MO_TYPE_GEOMETRY_GROUP );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometryGroup )
{
    std::unique_ptr<ManagedObject> mo = makeUnique<GeometryGroup>( m_context );

    GeometryGroup* gg = managedObjectCast<GeometryGroup>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}

TEST_F( ManagedObjectCastTest, managedObjectToGeometryGroupConst )
{
    std::unique_ptr<const ManagedObject> mo = makeUnique<GeometryGroup>( m_context );

    const GeometryGroup* gg = managedObjectCast<const GeometryGroup>( mo.get() );

    EXPECT_TRUE( mo->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}

// Concrete LexicalScope Downcasts ----------------------------------------------------------

TEST_F( ManagedObjectCastTest, lexicalScopeToMaterial )
{
    LexicalScope* ls = new Material( m_context );

    Material* mat = managedObjectCast<Material>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_MATERIAL ) );
    EXPECT_NE( mat, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToMaterialConst )
{
    const LexicalScope* ls = new Material( m_context );

    const Material* mat = managedObjectCast<const Material>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_MATERIAL ) );
    EXPECT_NE( mat, nullptr );
}

// Program cast
TEST_F( ManagedObjectCastTest, lexicalScopeToProgram )
{
    LexicalScope* ls = new Program( m_context );

    Program* program = managedObjectCast<Program>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_PROGRAM ) );
    EXPECT_NE( program, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToProgramConst )
{
    const LexicalScope* ls = new Program( m_context );

    const Program* program = managedObjectCast<const Program>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_PROGRAM ) );
    EXPECT_NE( program, nullptr );
}

// Acceleration cast
TEST_F( ManagedObjectCastTest, lexicalScopeToAcceleration )
{
    LexicalScope* ls = new Acceleration( m_context );

    Acceleration* accel = managedObjectCast<Acceleration>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_ACCELERATION ) );
    EXPECT_NE( accel, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToAccelerationConst )
{
    const LexicalScope* ls = new Acceleration( m_context );

    const Acceleration* accel = managedObjectCast<const Acceleration>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_ACCELERATION ) );
    EXPECT_NE( accel, nullptr );
}

// GeometryInstance cast
TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryInstance )
{
    LexicalScope* ls = new GeometryInstance( m_context );

    GeometryInstance* gi = managedObjectCast<GeometryInstance>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_INSTANCE ) );
    EXPECT_NE( gi, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryInstanceConst )
{
    const LexicalScope* ls = new GeometryInstance( m_context );

    const GeometryInstance* gi = managedObjectCast<const GeometryInstance>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_INSTANCE ) );
    EXPECT_NE( gi, nullptr );
}

// Geometry cast
TEST_F( ManagedObjectCastTest, lexicalScopeToGeometry )
{
    LexicalScope* ls = new Geometry( m_context, false );

    Geometry* geometry = managedObjectCast<Geometry>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY ) );
    EXPECT_NE( geometry, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryConst )
{
    const LexicalScope* ls = new Geometry( m_context, false );

    const Geometry* geometry = managedObjectCast<const Geometry>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY ) );
    EXPECT_NE( geometry, nullptr );
}

// GeometryTriangles cast
TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryTriangles )
{
    LexicalScope* ls = new GeometryTriangles( m_context );

    GeometryTriangles* gt = managedObjectCast<GeometryTriangles>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryTrianglesConst )
{
    const LexicalScope* ls = new GeometryTriangles( m_context );

    const GeometryTriangles* gt = managedObjectCast<const GeometryTriangles>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

// PostprocessingStageDenoiser cast
TEST_F( ManagedObjectCastTest, lexicalScopeToPPSDenoiser )
{
    LexicalScope* ls = new PostprocessingStageDenoiser( m_context, nullptr );

    PostprocessingStageDenoiser* ppsd = managedObjectCast<PostprocessingStageDenoiser>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_DENOISER ) );
    EXPECT_NE( ppsd, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToPPSDenoiserConst )
{
    const LexicalScope* ls = new PostprocessingStageDenoiser( m_context, nullptr );

    const PostprocessingStageDenoiser* ppsd = managedObjectCast<const PostprocessingStageDenoiser>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_DENOISER ) );
    EXPECT_NE( ppsd, nullptr );
}

// PostprocessingStageSSIM cast
TEST_F( ManagedObjectCastTest, lexicalScopeToPPSSSIM )
{
    LexicalScope* ls = new PostprocessingStageSSIMPredictor( m_context, nullptr );

    PostprocessingStageSSIMPredictor* pps = managedObjectCast<PostprocessingStageSSIMPredictor>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_SSIM ) );
    EXPECT_NE( pps, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToPPSSSIMConst )
{
    const LexicalScope* ls = new PostprocessingStageSSIMPredictor( m_context, nullptr );

    const PostprocessingStageSSIMPredictor* pps = managedObjectCast<const PostprocessingStageSSIMPredictor>( ls );
    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_SSIM ) );
    EXPECT_NE( pps, nullptr );
}

// PostprocessingStageTonemap cast
TEST_F( ManagedObjectCastTest, lexicalScopeToPPSToneMap )
{
    LexicalScope* ls = new PostprocessingStageTonemap( m_context );

    PostprocessingStageTonemap* pps = managedObjectCast<PostprocessingStageTonemap>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_TONEMAP ) );
    EXPECT_NE( pps, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToPPSToneMapConst )
{
    const LexicalScope* ls = new PostprocessingStageTonemap( m_context );

    const PostprocessingStageTonemap* pps = managedObjectCast<const PostprocessingStageTonemap>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_POSTPROC_STAGE_TONEMAP ) );
    EXPECT_NE( pps, nullptr );
}

// Selector cast
TEST_F( ManagedObjectCastTest, lexicalScopeToSelector )
{
    LexicalScope* ls = new Selector( m_context );

    Selector* selector = managedObjectCast<Selector>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_SELECTOR ) );
    EXPECT_NE( selector, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToSelectorConst )
{
    const LexicalScope* ls = new Selector( m_context );

    const Selector* selector = managedObjectCast<const Selector>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_SELECTOR ) );
    EXPECT_NE( selector, nullptr );
}

// Transform cast
TEST_F( ManagedObjectCastTest, lexicalScopeToTransform )
{
    LexicalScope* ls = new Transform( m_context );

    Transform* transform = managedObjectCast<Transform>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( transform, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToTransformConst )
{
    const LexicalScope* ls = new Transform( m_context );

    const Transform* transform = managedObjectCast<const Transform>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_TRANSFORM ) );
    EXPECT_NE( transform, nullptr );
}

// Group cast
TEST_F( ManagedObjectCastTest, lexicalScopeToGroup )
{
    LexicalScope* ls = new Group( m_context );

    Group* group = managedObjectCast<Group>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToGroupConst )
{
    const LexicalScope* ls = new Group( m_context );

    const Group* group = managedObjectCast<const Group>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

// GeometryGroup cast
TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryGroup )
{
    LexicalScope* ls = new GeometryGroup( m_context );

    GeometryGroup* gg = managedObjectCast<GeometryGroup>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}

TEST_F( ManagedObjectCastTest, lexicalScopeToGeometryGroupConst )
{
    const LexicalScope* ls = new GeometryGroup( m_context );

    const GeometryGroup* gg = managedObjectCast<const GeometryGroup>( ls );

    EXPECT_TRUE( ls->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}


// Concrete AbstractGroup Downcasts ----------------------------------------------------------

// Group cast
TEST_F( ManagedObjectCastTest, abstractGroupToGroup )
{
    AbstractGroup* ag = new Group( m_context );

    Group* group = managedObjectCast<Group>( ag );

    EXPECT_TRUE( ag->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

TEST_F( ManagedObjectCastTest, abstractGroupToGroupConst )
{
    const AbstractGroup* ag = new Group( m_context );

    const Group* group = managedObjectCast<const Group>( ag );

    EXPECT_TRUE( ag->isA( MO_TYPE_GROUP ) );
    EXPECT_NE( group, nullptr );
}

// GeometryGroup cast
TEST_F( ManagedObjectCastTest, abstractGroupToGeometryGroup )
{
    AbstractGroup* ag = new GeometryGroup( m_context );

    GeometryGroup* gg = managedObjectCast<GeometryGroup>( ag );

    EXPECT_TRUE( ag->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}

TEST_F( ManagedObjectCastTest, abstractGroupToGeometryGroupConst )
{
    const AbstractGroup* ag = new GeometryGroup( m_context );

    const GeometryGroup* gg = managedObjectCast<const GeometryGroup>( ag );

    EXPECT_TRUE( ag->isA( MO_TYPE_GEOMETRY_GROUP ) );
    EXPECT_NE( gg, nullptr );
}

// Concrete Geometry Downcasts ----------------------------------------------------------

TEST_F( ManagedObjectCastTest, geometryToGeometryTriangles )
{
    Geometry* geometry = new GeometryTriangles( m_context );

    GeometryTriangles* gt = managedObjectCast<GeometryTriangles>( geometry );

    EXPECT_TRUE( geometry->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

TEST_F( ManagedObjectCastTest, geometryToGeometryTrianglesConst )
{
    const Geometry* geometry = new GeometryTriangles( m_context );

    const GeometryTriangles* gt = managedObjectCast<const GeometryTriangles>( geometry );

    EXPECT_TRUE( geometry->isA( MO_TYPE_GEOMETRY_TRIANGLES ) );
    EXPECT_NE( gt, nullptr );
}

}  // end namespace
