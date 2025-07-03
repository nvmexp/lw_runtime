// We need to get access to members (such as the direct caller property) without having to
// add accessor functions.  In order to make this compile in MSVS we need to include a
// couple of files first that rely on the original private/protected versions to avoid
// instantiating certain functions that will result in link errors later.  If you see
// these link errors, please just add the appropriate include in this above block.  In
// addition we need the _ALLOW_KEYWORD_MACROS to avoid errors in xkeycheck.h.
#include <srcTests.h>
#include <string>
#define private public
#define protected public

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Control/ErrorManager.h>
#include <Device/DeviceManager.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/Transform.h>
#include <prodlib/exceptions/UnknownError.h>

#include <stdarg.h>

#define RT_CHECK_ERROR( func )                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        RTresult code = func;                                                                                          \
        if( code != RT_SUCCESS )                                                                                       \
        {                                                                                                              \
            throw prodlib::UnknownError( RT_EXCEPTION_INFO, "Test Error" );                                            \
        }                                                                                                              \
    } while( 0 )

using namespace optix;
using namespace testing;

struct PTXModule
{
    const char* description;
    const char* metadata;
    const char* code;
};

#define PTX_MODULE( desc, ... )                                                                                        \
    {                                                                                                                  \
        desc, "", #__VA_ARGS__                                                                                         \
    }


// clang-format off
PTXModule progPtxEmpty = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .global .align 4 .b8 top_object[4];

  .entry prog
  {
    .reg .u32 %r<23>;
    ld.global.u32 	%r6, [top_object+0];
    st.local.u32 [0], %r6;

    ret;
  }  

  .global .align 4 .b8 _ZN21rti_internal_typeinfo10top_objectE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename10top_objectE[9] = {0x72,0x74,0x4f,0x62,0x6a,0x65,0x63,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum10top_objectE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic10top_objectE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation10top_objectE[1] = {0x0};
);

PTXModule progPtxEmptyB = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .global .align 4 .b8 top_object[4];

  .entry prog
  {
    .reg .u32 %r<23>;
    ld.global.u32 	%r6, [top_object+0];
    st.local.u32 [100], %r6;

    ret;
  }  

  .global .align 4 .b8 _ZN21rti_internal_typeinfo10top_objectE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename10top_objectE[9] = {0x72,0x74,0x4f,0x62,0x6a,0x65,0x63,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum10top_objectE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic10top_objectE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation10top_objectE[1] = {0x0};
);


PTXModule progPtxNoTopObj = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .entry prog
  {
    ret;
  }  

);

PTXModule progPtxAlt = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .global .align 4 .b8 top_object[4];

  .global .f32 var = 0f00000000;

  .entry prog
  {
    .reg .u32 %r<23>;
    ld.global.u32 	%r7, [top_object+0];
    st.local.u32 [0], %r7;

    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
 
    ret;
  }  

  .global .align 4 .b8 _ZN21rti_internal_typeinfo10top_objectE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename10top_objectE[9] = {0x72,0x74,0x4f,0x62,0x6a,0x65,0x63,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum10top_objectE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic10top_objectE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation10top_objectE[1] = {0x0};

);

PTXModule progPtxRefFrame = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32

  .global .align 4 .u32 frame;
                                         
  .entry prog
  {
    .reg .u32 %r<23>;
    ld.global.u32 	%r11, [frame];
    st.local.u32 [0], %r11;
    ret;
  }  

  .global .align 4 .b8 _ZN21rti_internal_typeinfo5frameE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
  .global .align 1 .b8 _ZN21rti_internal_typename5frameE[4] = {105, 110, 116, 0};
  .global .align 4 .u32 _ZN21rti_internal_typeenum5frameE = 4919;
  .global .align 1 .b8 _ZN21rti_internal_semantic5frameE[1];
  .global .align 1 .b8 _ZN23rti_internal_annotation5frameE[1];
                                         
);

// clang-format on


// These are used to parameterize ordering for various graph construction events
enum GraphNodeOp
{
    Raygen_A_Attach0,                // sets raygen[0] to rgA
    Raygen_A_Attach1,                // ...         1     rgA
    Raygen_B_Attach0,                // ...         0     rgB
    Raygen_B_Attach1,                // ...         1     rgB
    Raygen_Alt_Attach0,              // ...         0  to rg_alt
    Raygen_Alt_Attach1,              // ...         1  to rg_alt
    Raygen_A_Detach,                 // sets rgA to a program that doesn't refer to "top_object", destroys old rgA
    Raygen_B_Detach,                 //  ... rgB                              ...                              rgB
    GeometryGroup0Create,            // create gg0
    GeometryGroup1Create,            // ...    gg1
    Raygen_A_Declare,                // rgA declares "top_object" variable
    Raygen_B_Declare,                // rgB ...
    Raygen_A_Set0,                   // rgA declares and sets "top_object" to gg0
    Raygen_B_Set0,                   // rgB ...                               gg0
    Raygen_A_Set1,                   // rgA ...                               gg1
    Raygen_B_Set1,                   // rgB ...                               gg1
    Raygen_A_Undeclare,              // rgA removes variable "top_object"
    Raygen_B_Undeclare,              // rgB ...
    Raygen_A_Create_Attach_Destroy,  // creates rgA, attaches rgA, destroys rgA
    Raygen_A_Reset,                  // recreates rgA
    Raygen_A_Destroy,                // destroys rgA
    GlobalScopeSet,                  // context declares and sets "top_object" to gg0
    GlobalScopeUndeclare,            // context removes variable "top_object"
};

struct GraphConstructionOrder
{
    std::vector<GraphNodeOp> ops;
};


class TestGraphConstruction : public ::testing::Test
{
  public:
    RTcontext       context;
    RTprogram       rgA;
    RTprogram       rgB;
    RTprogram       rg_alt;
    RTgeometrygroup gg0;
    RTgeometrygroup gg1;
    RTvariable      ctx_var;
    RTvariable      rgA_var;
    RTvariable      rgB_var;

    TestGraphConstruction() {}

    void SetUp() {}

    void TearDown()
    {
        RTcontext ctx_api = RTcontext( context );
        ASSERT_EQ( RT_SUCCESS, rtContextDestroy( ctx_api ) );
        context = nullptr;
    }

    void CreateContext()
    {
        ASSERT_EQ( RT_SUCCESS, rtContextCreate( &context ) );

        rtContextSetEntryPointCount( context, 2 );
        rtContextSetRayTypeCount( context, 1 );

        rtProgramCreateFromPTXString( context, progPtxEmpty.code, "prog", &rgA );
        rtProgramCreateFromPTXString( context, progPtxEmptyB.code, "prog", &rgB );
        rtProgramCreateFromPTXString( context, progPtxAlt.code, "prog", &rg_alt );
    }

    void CreateGeometryGroup( RTgeometrygroup& gg )
    {
        RTmaterial         material;
        RTgeometryinstance instance;
        RTgeometry         geom;
        RTacceleration     accel;
        RTprogram          prog;

        rtProgramCreateFromPTXString( context, progPtxAlt.code, "prog", &prog );

        rtAccelerationCreate( context, &accel );
        rtGeometryCreate( context, &geom );
        rtGeometryInstanceCreate( context, &instance );
        rtMaterialCreate( context, &material );
        rtGeometryGroupCreate( context, &gg );

        rtGeometrySetBoundingBoxProgram( geom, prog );
        rtGeometrySetIntersectionProgram( geom, prog );
        rtMaterialSetClosestHitProgram( material, 0, prog );
        rtMaterialSetAnyHitProgram( material, 0, prog );

        rtGeometryInstanceSetGeometry( instance, geom );
        rtGeometryInstanceSetMaterialCount( instance, 1 );
        rtGeometryInstanceSetMaterial( instance, 0, material );
        rtGeometryGroupSetChildCount( gg, 1 );
        rtGeometryGroupSetChild( gg, 0, instance );
        rtGeometryGroupSetAcceleration( gg, accel );
    }

    void BuildGraph( GraphNodeOp op )
    {
        switch( op )
        {

            case Raygen_A_Declare:
                rtProgramDeclareVariable( rgA, "top_object", &rgA_var );
                break;

            case Raygen_B_Declare:
                rtProgramDeclareVariable( rgB, "top_object", &rgB_var );
                break;

            case Raygen_A_Create_Attach_Destroy:
            {
                rtProgramCreateFromPTXString( context, progPtxNoTopObj.code, "prog", &rgA );
                rtContextSetRayGenerationProgram( context, 0, rgA );
                rtProgramDestroy( rgA );
            }
            break;

            case Raygen_A_Reset:
                rtProgramCreateFromPTXString( context, progPtxEmpty.code, "prog", &rgA );
                break;

            case Raygen_A_Attach0:
                rtContextSetRayGenerationProgram( context, 0, rgA );
                break;

            case Raygen_A_Attach1:
                rtContextSetRayGenerationProgram( context, 1, rgA );
                break;

            case Raygen_B_Attach0:
                rtContextSetRayGenerationProgram( context, 0, rgB );
                break;

            case Raygen_B_Attach1:
                rtContextSetRayGenerationProgram( context, 1, rgB );
                break;

            case Raygen_Alt_Attach0:
                rtContextSetRayGenerationProgram( context, 0, rg_alt );
                break;

            case Raygen_Alt_Attach1:
                rtContextSetRayGenerationProgram( context, 1, rg_alt );
                break;

            case Raygen_A_Detach:
            {
                RTprogram prog;
                rtProgramCreateFromPTXString( context, progPtxNoTopObj.code, "prog", &prog );
                rtContextSetRayGenerationProgram( context, 0, prog );
                rtProgramDestroy( rgA );
            }
            break;

            case Raygen_B_Detach:
            {
                RTprogram prog;
                rtProgramCreateFromPTXString( context, progPtxNoTopObj.code, "prog", &prog );
                rtContextSetRayGenerationProgram( context, 1, prog );
                rtProgramDestroy( rgB );
            }
            break;

            case Raygen_A_Destroy:
            {
                rtProgramDestroy( rgA );
            }
            break;

            case GeometryGroup0Create:
                CreateGeometryGroup( gg0 );
                break;

            case GeometryGroup1Create:
                CreateGeometryGroup( gg1 );
                break;

            case Raygen_A_Set0:
            {
                RTvariable v;
                rtProgramQueryVariable( rgA, "top_object", &v );
                if( !v )
                    rtProgramDeclareVariable( rgA, "top_object", &rgA_var );
                rtVariableSetObject( rgA_var, gg0 );
            }
            break;

            case Raygen_A_Set1:
            {
                RTvariable v;
                rtProgramQueryVariable( rgA, "top_object", &v );
                if( !v )
                    rtProgramDeclareVariable( rgA, "top_object", &rgA_var );
                rtVariableSetObject( rgA_var, gg1 );
            }
            break;

            case Raygen_B_Set0:
            {
                RTvariable v;
                rtProgramQueryVariable( rgB, "top_object", &v );
                if( !v )
                    rtProgramDeclareVariable( rgB, "top_object", &rgB_var );
                rtVariableSetObject( rgB_var, gg0 );
            }
            break;

            case Raygen_B_Set1:
            {
                RTvariable v;
                rtProgramQueryVariable( rgB, "top_object", &v );
                if( !v )
                    rtProgramDeclareVariable( rgB, "top_object", &rgB_var );
                rtVariableSetObject( rgB_var, gg1 );
            }
            break;

            case Raygen_A_Undeclare:
            {
                RTvariable v;
                rtProgramQueryVariable( rgA, "top_object", &v );
                rtProgramRemoveVariable( rgA, v );
            }
            break;

            case Raygen_B_Undeclare:
            {
                RTvariable v;
                rtProgramQueryVariable( rgB, "top_object", &v );
                rtProgramRemoveVariable( rgB, v );
            }
            break;

            case GlobalScopeSet:
                rtContextDeclareVariable( context, "top_object", &ctx_var );
                rtVariableSetObject( ctx_var, gg0 );
                break;

            case GlobalScopeUndeclare:
            {
                RTvariable v;
                rtContextQueryVariable( context, "top_object", &v );
                rtContextRemoveVariable( context, v );
            }
            break;
        }
    }
};

GraphConstructionOrder createConstructionOrder( int numOps, ... )
{
    va_list                args;
    GraphConstructionOrder retVal;

    va_start( args, numOps );
    for( int i = 0; i < numOps; ++i )
        retVal.ops.push_back( ( GraphNodeOp )( va_arg( args, int ) ) );
    va_end( args );
    return retVal;
}

GraphConstructionOrder directCallerOrders[] = {
    createConstructionOrder( 6, GeometryGroup0Create, Raygen_A_Create_Attach_Destroy, Raygen_A_Create_Attach_Destroy, Raygen_A_Create_Attach_Destroy, Raygen_B_Attach0, Raygen_B_Set0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_B_Attach1, Raygen_A_Set0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_Alt_Attach1, Raygen_A_Detach ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Undeclare ),
    createConstructionOrder( 6, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Undeclare, Raygen_B_Set0, Raygen_B_Attach1 ),
    createConstructionOrder( 6, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_B_Set0, Raygen_B_Attach1, Raygen_A_Undeclare ),
    createConstructionOrder( 6, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Detach, Raygen_B_Attach1 ),
    createConstructionOrder( 6, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Detach, Raygen_A_Reset, Raygen_A_Attach0 ),
    createConstructionOrder( 4, GeometryGroup0Create, Raygen_A_Set0, Raygen_A_Attach0, Raygen_A_Attach0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, GlobalScopeUndeclare, Raygen_A_Attach0, Raygen_A_Set0 ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Attach0, GlobalScopeSet, GlobalScopeUndeclare, Raygen_A_Set0 ),
    createConstructionOrder( 6, Raygen_A_Declare, GeometryGroup0Create, Raygen_A_Set0, Raygen_A_Attach0, GlobalScopeSet, GlobalScopeUndeclare ),
    createConstructionOrder( 6, Raygen_A_Declare, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, GlobalScopeSet, GlobalScopeUndeclare ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Set0, Raygen_A_Attach0, GlobalScopeSet, GlobalScopeUndeclare ),
    createConstructionOrder( 6, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, GlobalScopeUndeclare, Raygen_A_Set0, Raygen_A_Attach1 ),
    createConstructionOrder( 6, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, GlobalScopeUndeclare, Raygen_A_Attach1, Raygen_A_Set0 ),
    createConstructionOrder( 7, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Declare, Raygen_A_Attach0, Raygen_A_Attach1, GlobalScopeUndeclare, Raygen_A_Set0 ),
    createConstructionOrder( 7, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Declare, Raygen_A_Attach0, Raygen_A_Attach1, Raygen_A_Set0, GlobalScopeUndeclare ),
    createConstructionOrder( 7, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Declare, Raygen_A_Attach0, Raygen_B_Attach1, Raygen_A_Set0, GlobalScopeUndeclare ),
    createConstructionOrder( 8, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Declare, Raygen_A_Attach0, Raygen_B_Attach1, Raygen_A_Set0, Raygen_B_Set0, GlobalScopeUndeclare ),
    createConstructionOrder( 8, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Declare, Raygen_A_Attach0, Raygen_B_Attach1, Raygen_A_Set0, GlobalScopeUndeclare, Raygen_B_Set0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Destroy, Raygen_B_Attach0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_B_Attach0, Raygen_A_Destroy ),
    createConstructionOrder( 4, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_B_Attach0 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_B_Set0, Raygen_B_Attach0 ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_B_Set0, Raygen_B_Attach0 ),
};

GraphConstructionOrder nonDirectCallerOrders[] = {
    createConstructionOrder( 3, GeometryGroup0Create, Raygen_A_Attach0, Raygen_B_Attach1 ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Detach ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Detach, Raygen_B_Attach1 ),
    createConstructionOrder( 7, GeometryGroup0Create, GeometryGroup1Create, Raygen_A_Attach0, GlobalScopeSet, Raygen_B_Attach1, Raygen_A_Set1, Raygen_B_Set1 ),
    createConstructionOrder( 4, GeometryGroup0Create, Raygen_A_Attach0, GlobalScopeSet, GlobalScopeUndeclare ),
    createConstructionOrder( 4, GeometryGroup0Create, GlobalScopeSet, GlobalScopeUndeclare, Raygen_A_Attach0 ),
    createConstructionOrder( 6, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, GlobalScopeUndeclare, Raygen_A_Set0, Raygen_A_Undeclare ),
    createConstructionOrder( 4, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_A_Undeclare ),
    createConstructionOrder( 7, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Set0, Raygen_B_Set0, Raygen_B_Attach1, Raygen_A_Undeclare, Raygen_B_Undeclare ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Set0, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Detach ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Attach0, GlobalScopeSet, Raygen_A_Attach1, GlobalScopeUndeclare ),
    createConstructionOrder( 5, GeometryGroup0Create, Raygen_A_Attach0, Raygen_A_Attach1, GlobalScopeSet, GlobalScopeUndeclare ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Attach1, GlobalScopeUndeclare ),
    createConstructionOrder( 5, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, GlobalScopeUndeclare, Raygen_A_Attach1 ),
    createConstructionOrder( 4, GeometryGroup0Create, GlobalScopeSet, Raygen_A_Attach0, Raygen_A_Destroy ),
};

class DirectCalledByAnyEntryPoint : public TestGraphConstruction
{
  public:
    bool hasDirectCaller()
    {
        Context*       ctx             = reinterpret_cast<Context*>( context );
        GeometryGroup* gg_obj          = reinterpret_cast<GeometryGroup*>( gg0 );
        bool           hasDirectCaller = false;
        for( auto& device : ctx->getDeviceManager()->uniqueActiveDevices() )
        {
            std::set<const CanonicalProgram*, CanonicalProgram::IDCompare> rayGen;
            for( unsigned int i = 0; i < ctx->getEntryPointCount(); ++i )
                rayGen.insert( ctx->getGlobalScope()->getRayGenerationProgram( i )->getCanonicalProgram( device ) );
            for( auto candidate : rayGen )
                if( gg_obj->m_directCaller.contains( candidate->getID() ) )
                    hasDirectCaller = true;
        }
        return hasDirectCaller;
    }
};

class DirectCalledByAnyEntryPointTrue : public DirectCalledByAnyEntryPoint, public WithParamInterface<GraphConstructionOrder>
{
};
class DirectCalledByAnyEntryPointFalse : public DirectCalledByAnyEntryPoint, public WithParamInterface<GraphConstructionOrder>
{
};

TEST_P( DirectCalledByAnyEntryPointTrue, TopObjectHasRayGenDirectCaller )
{
    CreateContext();

    for( size_t i = 0; i < GetParam().ops.size(); i++ )
        BuildGraph( GetParam().ops[i] );

    ASSERT_TRUE( hasDirectCaller() );
}

TEST_P( DirectCalledByAnyEntryPointFalse, TopObjectHasNoRayGenDirectCaller )
{
    CreateContext();

    for( size_t i = 0; i < GetParam().ops.size(); i++ )
        BuildGraph( GetParam().ops[i] );

    ASSERT_FALSE( hasDirectCaller() );
}

INSTANTIATE_TEST_SUITE_P( Misc, DirectCalledByAnyEntryPointTrue, ValuesIn( directCallerOrders ) );

INSTANTIATE_TEST_SUITE_P( Misc, DirectCalledByAnyEntryPointFalse, ValuesIn( nonDirectCallerOrders ) );

class TestSwitching : public TestGraphConstruction
{
};

TEST_F( TestSwitching, SwapOutMaterial )
{
    CreateContext();

    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &gg0 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( gg0, 1 ) );

    RTgeometryinstance gi;
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &gi ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( gg0, 0, gi ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( gi, 1 ) );
    RTvariable frame;
    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( gi, "frame", &frame ) );
    RT_CHECK_ERROR( rtVariableSet1ui( frame, 1u ) );

    RTprogram closestHit;
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, progPtxRefFrame.code, "prog", &closestHit ) );

    RTmaterial material;
    RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, 0, closestHit ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( gi, 0, material ) );

    RTmaterial material2;
    RT_CHECK_ERROR( rtMaterialCreate( context, &material2 ) );
    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material2, 0, closestHit ) );
    // This overrides the material with material2
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( gi, 0, material2 ) );
}

TEST_F( TestSwitching, SwapOutGeometry )
{
    CreateContext();

    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &gg0 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( gg0, 1 ) );

    RTgeometryinstance gi;
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &gi ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( gg0, 0, gi ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( gi, 1 ) );
    RTvariable frame;
    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( gi, "frame", &frame ) );
    RT_CHECK_ERROR( rtVariableSet1ui( frame, 1u ) );

    RTprogram intersection;
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, progPtxRefFrame.code, "prog", &intersection ) );

    RTgeometry geometry;
    RT_CHECK_ERROR( rtGeometryCreate( context, &geometry ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( geometry, intersection ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( gi, geometry ) );

    RTgeometry geometry2;
    RT_CHECK_ERROR( rtGeometryCreate( context, &geometry2 ) );
    RT_CHECK_ERROR( rtGeometrySetIntersectionProgram( geometry2, intersection ) );
    // This overrides the geometry with geometry2
    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( gi, geometry2 ) );
}

TEST_F( TestSwitching, SwapBufferOnTextureSampler )
{
    CreateContext();

    // Test when a buffer is changed while the texture sampler is attached

    RTtexturesampler ts;
    RTvariable       tsvar;
    RTbuffer         buf1, buf2;
    RT_CHECK_ERROR( rtTextureSamplerCreate( context, &ts ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "ts", &tsvar ) );
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &buf1 ) );
    RT_CHECK_ERROR( rtTextureSamplerSetBuffer( ts, 0, 0, buf1 ) );
    RT_CHECK_ERROR( rtVariableSetObject( tsvar, ts ) );
    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_INPUT, &buf2 ) );
    RT_CHECK_ERROR( rtTextureSamplerSetBuffer( ts, 0, 0, buf2 ) );
}

class TransformHeight : public TestGraphConstruction
{
  public:
    void CreateGroup( RTgroup& grp, bool attach )
    {
        RTacceleration accel;

        rtGroupCreate( context, &grp );
        rtAccelerationCreate( context, &accel );
        rtGroupSetChildCount( grp, 3 );
        rtGroupSetAcceleration( grp, accel );

        if( attach )
        {
            rtContextDeclareVariable( context, "top_object", &ctx_var );
            rtVariableSetObject( ctx_var, grp );
        }
    }
};

TEST_F( TransformHeight, OnlyOneTransform )
{
    CreateContext();

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
}


TEST_F( TransformHeight, OnlyTwoTransforms )
{
    CreateContext();

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    RTtransform t1;
    rtTransformCreate( context, &t1 );

    rtTransformSetChild( t0, t1 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );
    Transform* tt1 = reinterpret_cast<Transform*>( t1 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt0->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1->getMaxTransformHeight() ) );
}


TEST_F( TransformHeight, GroupUnderTransform )
{
    CreateContext();
    RTgroup group, group2;
    CreateGroup( group, false );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    CreateGroup( group2, false );

    rtGroupSetChild( group2, 0, t0 );
    rtTransformSetChild( t0, group );

    Context*   ctx  = reinterpret_cast<Context*>( context );
    Group*     grp  = reinterpret_cast<Group*>( group );
    Group*     grp2 = reinterpret_cast<Group*>( group2 );
    Transform* tt0  = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp2->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, UnattachedDoesAffectHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, false );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    rtGroupSetChild( group, 0, t0 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, UnattachedDoesAffectHeight2 )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    // We leave the transform dangling
    //rtGroupSetChild(group, 0, t0);

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, DetachChangesHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    rtGroupSetChild( group, 0, t0 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );

    rtGeometryGroupCreate( context, &gg0 );
    rtGroupSetChild( group, 0, gg0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, AttachChangesHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    rtGeometryGroupCreate( context, &gg0 );
    rtGroupSetChild( group, 0, gg0 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );

    rtGroupSetChild( group, 0, t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
}


TEST_F( TransformHeight, CanRestorePreviousHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( group, 0, t0 );

    RTtransform t1;
    rtTransformCreate( context, &t1 );
    rtTransformSetChild( t1, t0 );

    rtGroupSetChild( group, 1, t1 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );
    Transform* tt1 = reinterpret_cast<Transform*>( t1 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );

    rtTransformDestroy( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1->getMaxTransformHeight() ) );
}


TEST_F( TransformHeight, CanRestorePreviousHeight2 )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( group, 0, t0 );

    RTtransform t1;
    RTtransform t1_1;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t1_1 );
    rtGroupSetChild( group, 1, t1 );
    rtTransformSetChild( t1, t1_1 );

    Context*   ctx   = reinterpret_cast<Context*>( context );
    Group*     grp   = reinterpret_cast<Group*>( group );
    Transform* tt0   = reinterpret_cast<Transform*>( t0 );
    Transform* tt1   = reinterpret_cast<Transform*>( t1 );
    Transform* tt1_1 = reinterpret_cast<Transform*>( t1_1 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_1->getMaxTransformHeight() ) );

    rtGeometryGroupCreate( context, &gg0 );
    rtGroupSetChild( group, 1, gg0 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_1->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, CanRestorePreviousHeight3 )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( group, 0, t0 );

    Context*   ctx = reinterpret_cast<Context*>( context );
    Group*     grp = reinterpret_cast<Group*>( group );
    Transform* tt0 = reinterpret_cast<Transform*>( t0 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt0->getMaxTransformHeight() ) );

    rtTransformDestroy( t0 );

    ASSERT_THAT( 0, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, RemoveLeafDoesDecrementHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t1_1;
    RTtransform t1_2;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t1_1 );
    rtTransformCreate( context, &t1_2 );
    rtGroupSetChild( group, 1, t1 );
    rtTransformSetChild( t1, t1_1 );
    rtTransformSetChild( t1_1, t1_2 );

    Context*   ctx   = reinterpret_cast<Context*>( context );
    Group*     grp   = reinterpret_cast<Group*>( group );
    Transform* tt1   = reinterpret_cast<Transform*>( t1 );
    Transform* tt1_1 = reinterpret_cast<Transform*>( t1_1 );
    Transform* tt1_2 = reinterpret_cast<Transform*>( t1_2 );

    ASSERT_THAT( 3, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1_1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_2->getMaxTransformHeight() ) );

    rtTransformDestroy( t1_2 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_1->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, RemoveMiddleDoesDecreaseHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t1_1;
    RTtransform t1_2;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t1_1 );
    rtTransformCreate( context, &t1_2 );
    rtGroupSetChild( group, 1, t1 );
    rtTransformSetChild( t1, t1_1 );
    rtTransformSetChild( t1_1, t1_2 );

    Context*   ctx   = reinterpret_cast<Context*>( context );
    Group*     grp   = reinterpret_cast<Group*>( group );
    Transform* tt1   = reinterpret_cast<Transform*>( t1 );
    Transform* tt1_1 = reinterpret_cast<Transform*>( t1_1 );
    Transform* tt1_2 = reinterpret_cast<Transform*>( t1_2 );

    ASSERT_THAT( 3, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1_1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_2->getMaxTransformHeight() ) );

    rtTransformDestroy( t1_1 );

    ASSERT_THAT( 1, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_2->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, RemoveRootDoesDecreaseHeight )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t1_1;
    RTtransform t1_2;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t1_1 );
    rtTransformCreate( context, &t1_2 );
    rtGroupSetChild( group, 1, t1 );
    rtTransformSetChild( t1, t1_1 );
    rtTransformSetChild( t1_1, t1_2 );

    Context*   ctx   = reinterpret_cast<Context*>( context );
    Group*     grp   = reinterpret_cast<Group*>( group );
    Transform* tt1   = reinterpret_cast<Transform*>( t1 );
    Transform* tt1_1 = reinterpret_cast<Transform*>( t1_1 );
    Transform* tt1_2 = reinterpret_cast<Transform*>( t1_2 );

    ASSERT_THAT( 3, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 3, Eq( tt1->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1_1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_2->getMaxTransformHeight() ) );

    rtTransformDestroy( t1 );

    ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
    ASSERT_THAT( 0, Eq( grp->getMaxTransformHeight() ) );
    ASSERT_THAT( 2, Eq( tt1_1->getMaxTransformHeight() ) );
    ASSERT_THAT( 1, Eq( tt1_2->getMaxTransformHeight() ) );
}

TEST_F( TransformHeight, CanDetectSimpleCycle )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );

    rtTransformSetChild( t1, t2 );

    RTresult shouldFail = rtTransformSetChild( t2, t1 );
    ASSERT_TRUE( RT_ERROR_ILWALID_CONTEXT == shouldFail );

    Context* ctx = reinterpret_cast<Context*>( context );

    ASSERT_THAT( ctx->getErrorManager()->getErrorString( shouldFail ),
                 HasSubstr( std::string( "Cycle detected in node graph" ) ) );
}

TEST_F( TransformHeight, CanDetectSimpleCycle2 )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );

    rtGroupSetChild( group, 0, t1 );
    rtGroupSetChild( group, 1, t2 );
    rtTransformSetChild( t1, t2 );


    RTresult shouldFail = rtTransformSetChild( t2, t1 );
    ASSERT_TRUE( RT_ERROR_ILWALID_CONTEXT == shouldFail );

    Context* ctx = reinterpret_cast<Context*>( context );

    ASSERT_THAT( ctx->getErrorManager()->getErrorString( shouldFail ),
                 HasSubstr( std::string( "Cycle detected in node graph" ) ) );
}

TEST_F( TransformHeight, DiamondIsNotCycle )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    RTtransform t3;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );
    rtTransformCreate( context, &t3 );

    rtGroupSetChild( group, 0, t1 );
    rtGroupSetChild( group, 1, t2 );
    rtTransformSetChild( t1, t3 );

    RTresult shouldNotFail = rtTransformSetChild( t2, t3 );
    ASSERT_TRUE( RT_SUCCESS == shouldNotFail );
}

TEST_F( TransformHeight, DiamondWithChildIsNotCycle )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    RTtransform t3;
    RTtransform t4;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );
    rtTransformCreate( context, &t3 );
    rtTransformCreate( context, &t4 );

    rtGroupSetChild( group, 0, t1 );
    rtGroupSetChild( group, 1, t2 );
    rtTransformSetChild( t1, t3 );
    rtTransformSetChild( t2, t3 );

    RTresult shouldNotFail = rtTransformSetChild( t3, t4 );
    ASSERT_TRUE( RT_SUCCESS == shouldNotFail );
}

TEST_F( TransformHeight, CanDetectComplexCycle )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    RTtransform t3;
    RTtransform t4;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );
    rtTransformCreate( context, &t3 );
    rtTransformCreate( context, &t4 );

    rtGroupSetChild( group, 0, t1 );
    rtGroupSetChild( group, 1, t2 );
    rtTransformSetChild( t1, t3 );
    rtTransformSetChild( t2, t3 );
    rtTransformSetChild( t3, t4 );

    RTresult shouldFail = rtTransformSetChild( t4, t1 );
    ASSERT_TRUE( RT_ERROR_ILWALID_CONTEXT == shouldFail );

    Context* ctx = reinterpret_cast<Context*>( context );

    ASSERT_THAT( ctx->getErrorManager()->getErrorString( shouldFail ),
                 HasSubstr( std::string( "Cycle detected in node graph" ) ) );
}

TEST_F( TransformHeight, CanDetectLongCycle )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    RTtransform t3;
    RTtransform t4;
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );
    rtTransformCreate( context, &t3 );
    rtTransformCreate( context, &t4 );

    rtGroupSetChild( group, 0, t1 );

    rtTransformSetChild( t1, t2 );
    rtTransformSetChild( t2, t3 );
    rtTransformSetChild( t3, t4 );

    RTresult shouldFail = rtTransformSetChild( t4, t1 );
    ASSERT_TRUE( RT_ERROR_ILWALID_CONTEXT == shouldFail );

    Context* ctx = reinterpret_cast<Context*>( context );

    ASSERT_THAT( ctx->getErrorManager()->getErrorString( shouldFail ),
                 HasSubstr( std::string( "Cycle detected in node graph" ) ) );
}

TEST_F( TransformHeight, CanDetectCycleWithNonXform )
{
    CreateContext();
    RTgroup group;
    CreateGroup( group, true );

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTtransform t1;
    RTtransform t2;
    RTgroup     groupCycle;
    CreateGroup( groupCycle, false );
    rtTransformCreate( context, &t1 );
    rtTransformCreate( context, &t2 );

    rtGroupSetChild( group, 0, t1 );
    rtTransformSetChild( t1, t2 );
    rtTransformSetChild( t2, groupCycle );

    RTresult shouldFail = rtGroupSetChild( groupCycle, 0, t1 );
    ASSERT_TRUE( RT_ERROR_ILWALID_CONTEXT == shouldFail );

    Context* ctx = reinterpret_cast<Context*>( context );

    ASSERT_THAT( ctx->getErrorManager()->getErrorString( shouldFail ),
                 HasSubstr( std::string( "Cycle detected in node graph" ) ) );
}

#if 0
// SGP Bigler : figure out how to replace computeMinTransformHeight or delete these tests
TEST_F( TransformHeight, CanRestorePreviousMin )
{
  CreateContext();
  RTgroup group;
  CreateGroup( group, true );

  rtContextSetRayGenerationProgram( context, 0, rgA );

  RTtransform t1;
  RTtransform t2;
  rtTransformCreate( context, &t1 );
  rtTransformCreate( context, &t2 );

  Context* ctx = reinterpret_cast<Context*>( context );

  rtGroupSetChild( group, 0, t1 );
  rtGroupSetChild( group, 1, t2 );

  ASSERT_THAT( 1, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );

  RTgroup group2;
  CreateGroup( group2, false );
  rtGroupSetChild( group, 0, group2 );

  ASSERT_THAT( 0, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );
}

TEST_F( TransformHeight, CanRestorePreviousMin2 )
{
  CreateContext();
  RTgroup group;
  CreateGroup( group, true );

  rtContextSetRayGenerationProgram( context, 0, rgA );

  RTtransform t1;
  RTtransform t2;
  rtTransformCreate( context, &t1 );
  rtTransformCreate( context, &t2 );

  Context* ctx = reinterpret_cast<Context*>( context );

  rtGroupSetChild( group, 0, t1 );
  rtGroupSetChild( group, 1, t2 );

  ASSERT_THAT( 1, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );

  rtTransformDestroy( t1 );
  rtTransformDestroy( t2 );

  ASSERT_THAT( 0, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );
}

TEST_F( TransformHeight, CanComputeCorrectMin )
{
  CreateContext();
  RTgroup group;
  CreateGroup( group, true );

  rtContextSetRayGenerationProgram( context, 0, rgA );

  RTtransform t1;
  RTtransform t2;
  RTtransform t3;
  rtTransformCreate( context, &t1 );
  rtTransformCreate( context, &t2 );
  rtTransformCreate( context, &t3 );

  Context* ctx = reinterpret_cast<Context*>( context );

  rtGroupSetChild( group, 0, t1 );
  rtGroupSetChild( group, 1, t2 );
  rtTransformSetChild( t2, t3 );

  ASSERT_THAT( 1, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );
}

TEST_F( TransformHeight, CanMaintainCorrectMin )
{
  CreateContext();
  RTgroup group;
  CreateGroup( group, true );

  rtContextSetRayGenerationProgram( context, 0, rgA );

  RTtransform t1;
  RTtransform t2;
  RTtransform t3;
  rtTransformCreate( context, &t1 );
  rtTransformCreate( context, &t2 );
  rtTransformCreate( context, &t3 );

  Context* ctx = reinterpret_cast<Context*>( context );

  rtGroupSetChild( group, 0, t1 );
  rtGroupSetChild( group, 1, t2 );
  rtTransformSetChild( t2, t3 );

  // Min is 1, even though max is 2
  ASSERT_THAT( 2, Eq( ctx->getBindingManager()->getMaxTransformHeight() ) );
  ASSERT_THAT( 1, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );

  rtTransformDestroy( t3 );

  ASSERT_THAT( 1, Eq( ctx->getGlobalScope()->computeMinTransformHeight() ) );
}
#endif


class HasMotion : public TestGraphConstruction
{
  public:
    void CreateGroup( RTgroup& grp, int child_count, RTacceleration& accel )
    {
        rtGroupCreate( context, &grp );
        rtAccelerationCreate( context, &accel );
        rtGroupSetChildCount( grp, child_count );
        rtGroupSetAcceleration( grp, accel );
    }

    void SetTopObject( RTgroup& grp )
    {
        RTvariable v;
        rtProgramQueryVariable( rgA, "top_object", &v );
        if( !v )
            rtProgramDeclareVariable( rgA, "top_object", &v );
        rtVariableSetObject( v, grp );
    }

    void CreateGeometryGroup( RTgeometrygroup& grp, RTgeometry& geom, RTacceleration& accel )
    {
        rtGeometryGroupCreate( context, &grp );
        rtAccelerationCreate( context, &accel );
        rtGeometryGroupSetAcceleration( grp, accel );

        rtGeometryCreate( context, &geom );
        RTmaterial mat;
        rtMaterialCreate( context, &mat );
        RTgeometryinstance gi;
        rtGeometryInstanceCreate( context, &gi );
        rtGeometryInstanceSetGeometry( gi, geom );
        rtGeometryInstanceSetMaterialCount( gi, 1 );
        rtGeometryInstanceSetMaterial( gi, 0, mat );

        rtGeometryGroupSetChildCount( grp, 1 );
        rtGeometryGroupSetChild( grp, 0, gi );
    }

    void SetMotionKeysOnTransform( RTtransform& transform )
    {
        const int num_keys = 2;
        float     keys[12 * num_keys];
        for( int i = 0; i < 12 * num_keys; ++i )
        {
            keys[i] = i;  // values don't matter
        }

        rtTransformSetMotionKeys( transform, num_keys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, keys );
    }
};

TEST_F( HasMotion, TransformMotionPropagatesUp )
{
    CreateContext();

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTacceleration accel;
    RTgroup        g0;
    CreateGroup( g0, 2, accel );
    SetTopObject( g0 );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( g0, 0, t0 );

    RTtransform t1;
    rtTransformCreate( context, &t1 );
    rtGroupSetChild( g0, 1, t1 );

    RTtransform t2;
    rtTransformCreate( context, &t2 );
    rtTransformSetChild( t1, t2 );

    Context* ctx = reinterpret_cast<Context*>( context );
    ASSERT_FALSE( ctx->getBindingManager()->hasMotionTransforms() );

    // Set motion on bottom transform of one child of group
    // and it should propagate up to group, and to global "hasMotionTransforms" prop.
    SetMotionKeysOnTransform( t2 );

    Group*        gg0 = reinterpret_cast<Group*>( g0 );
    Transform*    tt0 = reinterpret_cast<Transform*>( t0 );
    Transform*    tt1 = reinterpret_cast<Transform*>( t1 );
    Transform*    tt2 = reinterpret_cast<Transform*>( t2 );
    Acceleration* a   = reinterpret_cast<Acceleration*>( accel );

    ASSERT_FALSE( tt0->hasMotionAabbs() );
    ASSERT_TRUE( tt1->hasMotionAabbs() );
    ASSERT_TRUE( tt2->hasMotionAabbs() );
    ASSERT_TRUE( gg0->hasMotionAabbs() );
    ASSERT_TRUE( a->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_TRUE( ctx->getBindingManager()->hasMotionTransforms() );

    // Now disable motion, by setting static matrix
    optix::Matrix4x4 matrix = optix::Matrix4x4::identity();
    tt2->setMatrix( matrix.getData(), false );
    ASSERT_FALSE( tt1->hasMotionAabbs() );
    ASSERT_FALSE( tt2->hasMotionAabbs() );
    ASSERT_FALSE( gg0->hasMotionAabbs() );
    ASSERT_FALSE( a->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_FALSE( ctx->getBindingManager()->hasMotionTransforms() );

    // Enable it again
    SetMotionKeysOnTransform( t2 );
    ASSERT_TRUE( tt1->hasMotionAabbs() );
    ASSERT_TRUE( tt2->hasMotionAabbs() );
    ASSERT_TRUE( gg0->hasMotionAabbs() );
    ASSERT_TRUE( a->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_TRUE( ctx->getBindingManager()->hasMotionTransforms() );
}

TEST_F( HasMotion, GeometryMotionPropagatesUp )
{
    CreateContext();

    RTacceleration accel0;
    RTgroup        g0;
    CreateGroup( g0, 2, accel0 );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( g0, 0, t0 );

    RTacceleration  accel1;
    RTgeometry      geom;
    RTgeometrygroup geomgroup;
    CreateGeometryGroup( geomgroup, geom, accel1 );

    rtTransformSetChild( t0, geomgroup );

    // Scene: Group -> T -> GG -> GI -> Geometry
    // set motion on geometry and it should propagate up
    rtGeometrySetMotionSteps( geom, 2 );

    Group*        gg0 = reinterpret_cast<Group*>( g0 );
    Transform*    tt0 = reinterpret_cast<Transform*>( t0 );
    Acceleration* a0  = reinterpret_cast<Acceleration*>( accel0 );
    Acceleration* a1  = reinterpret_cast<Acceleration*>( accel1 );

    ASSERT_TRUE( tt0->hasMotionAabbs() );
    ASSERT_TRUE( gg0->hasMotionAabbs() );
    ASSERT_TRUE( a0->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_TRUE( a1->hasMotionAabbs_publicMethodForTesting() );

    // Disable motion
    rtGeometrySetMotionSteps( geom, 1 );  // 1 means static
    ASSERT_FALSE( tt0->hasMotionAabbs() );
    ASSERT_FALSE( gg0->hasMotionAabbs() );
    ASSERT_FALSE( a0->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_FALSE( a1->hasMotionAabbs_publicMethodForTesting() );
}

TEST_F( HasMotion, TransformMotionDetectedOnSetChild )
{
    CreateContext();

    rtContextSetRayGenerationProgram( context, 0, rgA );

    RTacceleration accel;
    RTgroup        g0;
    CreateGroup( g0, 2, accel );

    RTtransform t0;
    rtTransformCreate( context, &t0 );

    // Set motion on transform BEFORE adding to group
    SetMotionKeysOnTransform( t0 );

    // No global transform motion yet
    Context* ctx = reinterpret_cast<Context*>( context );
    ASSERT_FALSE( ctx->getBindingManager()->hasMotionTransforms() );

    rtGroupSetChild( g0, 0, t0 );

    Group*        gg0 = reinterpret_cast<Group*>( g0 );
    Transform*    tt0 = reinterpret_cast<Transform*>( t0 );
    Acceleration* a   = reinterpret_cast<Acceleration*>( accel );

    // Connect group to scene --> enables motion transforms
    SetTopObject( g0 );
    ASSERT_TRUE( ctx->getBindingManager()->hasMotionTransforms() );

    ASSERT_TRUE( tt0->hasMotionAabbs() );
    ASSERT_TRUE( gg0->hasMotionAabbs() );
    ASSERT_TRUE( a->hasMotionAabbs_publicMethodForTesting() );

    // Detach transform from scene
    rtGroupSetChildCount( g0, 0 );
    ASSERT_TRUE( tt0->hasMotionAabbs() );
    ASSERT_FALSE( gg0->hasMotionAabbs() );
    ASSERT_FALSE( a->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_FALSE( ctx->getBindingManager()->hasMotionTransforms() );
}

TEST_F( HasMotion, GeometryMotionDetectedOnSetChild )
{
    CreateContext();

    RTacceleration accel0;
    RTgroup        g0;
    CreateGroup( g0, 2, accel0 );

    RTtransform t0;
    rtTransformCreate( context, &t0 );
    rtGroupSetChild( g0, 0, t0 );

    RTacceleration  accel1;
    RTgeometry      geom;
    RTgeometrygroup geomgroup;
    CreateGeometryGroup( geomgroup, geom, accel1 );

    // set motion on geometry before adding to scene
    rtGeometrySetMotionSteps( geom, 2 );

    rtTransformSetChild( t0, geomgroup );

    // Scene: Group -> T -> GG -> GI -> Geometry

    Group*        gg0 = reinterpret_cast<Group*>( g0 );
    Transform*    tt0 = reinterpret_cast<Transform*>( t0 );
    Acceleration* a0  = reinterpret_cast<Acceleration*>( accel0 );
    Acceleration* a1  = reinterpret_cast<Acceleration*>( accel1 );

    ASSERT_TRUE( tt0->hasMotionAabbs() );
    ASSERT_TRUE( gg0->hasMotionAabbs() );
    ASSERT_TRUE( a0->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_TRUE( a1->hasMotionAabbs_publicMethodForTesting() );

    // Detach geometry from graph; should remove motion from parents
    rtGeometryGroupSetChildCount( geomgroup, 0 );
    ASSERT_FALSE( tt0->hasMotionAabbs() );
    ASSERT_FALSE( gg0->hasMotionAabbs() );
    ASSERT_FALSE( a0->hasMotionAabbs_publicMethodForTesting() );
    ASSERT_FALSE( a1->hasMotionAabbs_publicMethodForTesting() );
}

class AttachedToVariable : public TestGraphConstruction
{
  public:
    void CreateGeometryGroup( RTgeometrygroup& gg )
    {
        RTacceleration accel;
        rtGeometryGroupCreate( context, &gg );
        rtAccelerationCreate( context, &accel );
        rtGeometryGroupSetAcceleration( gg, accel );
    }
};

TEST_F( AttachedToVariable, TwoVariables )
{
    CreateContext();

    RTgeometrygroup geometrygroup;
    CreateGeometryGroup( geometrygroup );

    GeometryGroup* gg0 = reinterpret_cast<GeometryGroup*>( geometrygroup );

    ASSERT_TRUE( gg0->m_attachedToVariable.empty() );

    RTvariable var1;
    rtContextDeclareVariable( context, "top_object", &var1 );
    rtVariableSetObject( var1, geometrygroup );

    ASSERT_THAT( 1, Eq( gg0->m_attachedToVariable.count() ) );

    RTvariable var2;
    rtContextDeclareVariable( context, "top_shadower", &var2 );
    rtVariableSetObject( var2, geometrygroup );

    ASSERT_THAT( 2, Eq( gg0->m_attachedToVariable.count() ) );

    rtContextRemoveVariable( context, var2 );

    ASSERT_THAT( 1, Eq( gg0->m_attachedToVariable.count() ) );

    rtContextRemoveVariable( context, var1 );

    ASSERT_TRUE( gg0->m_attachedToVariable.empty() );
}


// Regression test for bug 25157411, an attachment count underflow arising from cyclic node graphs:
// "The issue arises when a GeometryInstance has a variable that points to its parent GeometryGroup.
// If the instance is detached first, it sends an attachment property change to its variables, which
// is notifies the parent group.  The parent group then sends a redundant attachment property change
// to the instance, causing its attachment count to underflow."
TEST_F( AttachedToVariable, TestCirlwlarAttachments )
{
    RTgroup            top_group;
    RTgeometrygroup    group;
    RTgeometryinstance instance;
    RTgeometry         geom;
    RTmaterial         material;
    RTacceleration     accel;
    RTprogram          prog;

    // top_group -> group -> instance
    // instance["top_shadower"] = group
    // context["top_object"] = top_group

    CreateContext();
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, progPtxAlt.code, "prog", &prog ) );
    RT_CHECK_ERROR( rtAccelerationCreate( context, &accel ) );

    RT_CHECK_ERROR( rtGroupCreate( context, &top_group ) );

    RT_CHECK_ERROR( rtGeometryCreate( context, &geom ) );
    RT_CHECK_ERROR( rtGeometryInstanceCreate( context, &instance ) );
    RT_CHECK_ERROR( rtMaterialCreate( context, &material ) );
    RT_CHECK_ERROR( rtGeometryGroupCreate( context, &group ) );

    RT_CHECK_ERROR( rtMaterialSetClosestHitProgram( material, 0, prog ) );
    RT_CHECK_ERROR( rtMaterialSetAnyHitProgram( material, 0, prog ) );

    RT_CHECK_ERROR( rtGeometryInstanceSetGeometry( instance, geom ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterialCount( instance, 1 ) );
    RT_CHECK_ERROR( rtGeometryInstanceSetMaterial( instance, 0, material ) );

    RT_CHECK_ERROR( rtGeometryGroupSetChildCount( group, 1 ) );
    RT_CHECK_ERROR( rtGeometryGroupSetChild( group, 0, instance ) );

    RTvariable instance_var;
    RTvariable ctx_var;
    RT_CHECK_ERROR( rtGeometryInstanceDeclareVariable( instance, "top_shadower", &instance_var ) );
    RT_CHECK_ERROR( rtVariableSetObject( instance_var, group ) );

    RT_CHECK_ERROR( rtGroupSetChildCount( top_group, 1 ) );
    RT_CHECK_ERROR( rtGroupSetChild( top_group, 0, group ) );

    RT_CHECK_ERROR( rtContextDeclareVariable( context, "top_object", &ctx_var ) );
    RT_CHECK_ERROR( rtVariableSetObject( ctx_var, top_group ) );

    // Test attachment property change for this particular object destruction order.
    RT_CHECK_ERROR( rtGroupDestroy( top_group ) );
    RT_CHECK_ERROR( rtGeometryInstanceDestroy( instance ) );
    RT_CHECK_ERROR( rtGeometryGroupDestroy( group ) );
}
