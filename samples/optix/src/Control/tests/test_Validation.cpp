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

#include <srcTests.h>

#include <c-api/rtapi.h>

#include <stdarg.h>

#include <optixu/optixpp_namespace.h>

#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace testing;


struct PTXModule
{
    const char* description;
    const char* metadata;
    const char* code;
};

// clang-format off
#define PTX_MODULE( desc, ... )\
{ desc, "", #__VA_ARGS__ }


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


PTXModule progPtxVar1f = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 var;
  .entry prog
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }  
  .global .align 4 .b8 _ZN21rti_internal_typeinfo3varE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename3varE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum3varE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic3varE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation3varE[1] = {0x0};
);

PTXModule progPtxStaticGlobalVar1f = PTX_MODULE( "prog_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 var = 0f00000000;
  .entry prog
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }  
); 
PTXModule progPtxStaticGlobalVarCallableProg = PTX_MODULE( "callableprog_ptx",
  .version 4.2
  .target sm_20
  .address_size 64
  .global .align 4 .b8 cp_0[4];
  .global .align 4 .b8 _ZN21rti_internal_typeinfo4cp_0E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
  .global .align 1 .b8 _ZN21rti_internal_typename4cp_0E[40] = {111, 112, 116, 105, 120, 58, 58, 98, 111, 117, 110, 100, 67, 97, 108, 108, 97, 98, 108, 101, 80, 114, 111, 103, 114, 97, 109, 73, 100, 60, 102, 108, 111, 97, 116, 32, 40, 41, 62, 0};
  .global .align 4 .u32 _ZN21rti_internal_typeenum4cp_0E = 4921;
  .global .align 1 .b8 _ZN21rti_internal_semantic4cp_0E[1];
  .global .align 1 .b8 _ZN23rti_internal_annotation4cp_0E[1];
  .visible .func  (.param .b32 func_retval0) _Z7store_0v()
  {
	  .reg .f32 %f<2>;
	  mov.f32 %f1, 0f3DCCCCCD;
	  st.param.f32	[func_retval0+0], %f1;
	  ret;
  }
  .visible .func  (.param .b32 func_retval0) _Z7store_1v(.param .s32 %val)
  {
	  .reg .f32 %f<2>;
	  mov.f32 %f1, 0f3DCCCCCD;
	  st.param.f32	[func_retval0+0], %f1;
	  ret;
  }
  .visible.func( .param.b32 func_retval0 ) call_callable()
  {
    .reg.u32 %r;
    .reg.u64 %rd;
    .reg.f32 %f<2>;
    ldu.global.u32 	%r, [cp_0];
    call( %rd ), _rt_callable_program_from_id_64, (%r);
    {
      .reg.b32 temp_param_reg;
      .param.b32 retval0;
      prototype_0:.callprototype( .param.b32 _ ) _();
      call( retval0 ), %rd, (), prototype_0;
      ld.param.f32	%f1, [retval0 + 0];
    }
    st.param.f32[func_retval0 + 0], %f1;
    ret;
  } 
  .visible.func( .param.b32 func_retval0 ) call_trace()
  {
    .reg.f32 %f<13>;
    .reg.u32 %r<13>;
    .reg.u64 %rd<13>;
    .local.u64 %prd[3];
    mov.u64 %rd11, %prd;
    mov.u32 %r12, 24;
    call _rt_trace_64, (%r1, %f2, %f3, %f4, %f5, %f6, %f7, %r8, %f9, %f10, %rd11, %r12);
    mov.f32 %f1, 0f3DCCCCCD;
    st.param.f32[func_retval0 + 0], %f1;
    ret;
  }
  .visible.func( .param.b32 func_retval0 ) call_potential_intersection()
  {
    .reg.f32 %f<2>;
    .reg.u32 %r<13>;
    .reg .pred  %p<2>;
    call( %r0 ), _rt_potential_intersection, (%f1);
    setp.eq.s32 %p0, %r0, 0;
    @%p0 bra  exit;
      call( %r0 ), _rt_report_intersection, (%r1);
      mov.f32 %f1, 0f3DCCCCCD;
      st.param.f32[func_retval0 + 0], %f1;
    exit:
      ret;
  }
  .entry prog
  {
    .reg .f32 %f;
    .reg .u32 %r;
    .reg .u64 %rd;
    ldu.global.u32 	%r, [cp_0];
    call( %rd ), _rt_callable_program_from_id_64, (%r);
    {
      .reg.b32 temp_param_reg;
      .param.b32 retval0;
      prototype_0:.callprototype( .param.b32 _ ) _();
      call( retval0 ), %rd, (), prototype_0;
      ld.param.f32	%f, [retval0 + 0];
    }
    st.local.f32 [0], %f;
    ret;
  }  
); 


PTXModule geometryPtxEmpty = PTX_MODULE( "geometry_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .entry box(.param .s32 __lwdaparm__Z10box_boundsiPf___T263, .param .u64 __lwdaparm__Z10box_boundsiPf_result)
  {
    ret;
  }	
  .entry intersect(.param .s32 __lwdaparm__Z13box_intersecti___T232)
  {
    ret;
  }
);

PTXModule geometryPtxVar1f = PTX_MODULE( "geometry_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 var;
  .entry box(.param .s32 __lwdaparm__Z10box_boundsiPf___T263, .param .u64 __lwdaparm__Z10box_boundsiPf_result)
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }	
  .entry intersect(.param .s32 __lwdaparm__Z13box_intersecti___T232)
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo3varE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename3varE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum3varE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic3varE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation3varE[1] = {0x0};
);

PTXModule geometryPtxAttr1f = PTX_MODULE( "geometry_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 attr;
  .entry box(.param .s32 __lwdaparm__Z10box_boundsiPf___T263, .param .u64 __lwdaparm__Z10box_boundsiPf_result)
  {
    ret;
  }	
  .entry intersect(.param .s32 __lwdaparm__Z13box_intersecti___T232)
  {
    .reg .f32 %f;
    mov.b32 %f, 0;
    st.global.f32 [attr], %f;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo4attrE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename4attrE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum4attrE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic4attrE[15] = {0x61,0x74,0x74,0x72,0x69,0x62,0x75,0x74,0x65,0x20,0x61,0x74,0x74,0x72,0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation4attrE[1] = {0x0};
);

PTXModule geometryPtxAttr1i = PTX_MODULE( "geometry_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .s32 attr;
  .entry box(.param .s32 __lwdaparm__Z10box_boundsiPf___T263, .param .u64 __lwdaparm__Z10box_boundsiPf_result)
  {
    ret;
  }	
  .entry intersect(.param .s32 __lwdaparm__Z13box_intersecti___T232)
  {
    .reg .s32 %r;
    mov.b32 %r, 0;
    st.global.s32 [attr], %r;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo4attrE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename4attrE[4] = {0x69,0x6e,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum4attrE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic4attrE[15] = {0x61,0x74,0x74,0x72,0x69,0x62,0x75,0x74,0x65,0x20,0x61,0x74,0x74,0x72,0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation4attrE[1] = {0x0};
);

PTXModule materialPtxEmpty = PTX_MODULE( "materiale_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .entry ch
  {
    ret;
  }
  .entry ah
  {
    ret;
  }
);

PTXModule materialPtxVar1f = PTX_MODULE( "materiale_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 var;
  .entry ch
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }
  .entry ah
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo3varE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename3varE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum3varE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic3varE[1] = {0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation3varE[1] = {0x0};
);

PTXModule materialPtxVar1i = PTX_MODULE( "materiale_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 var;
  .entry ch
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }
  .entry ah
  {
    .reg .f32 %f;
    ld.global.f32 %f, [var];
    st.local.f32 [0], %f;
    ret;
  }
.global .align 4 .b8 _ZN21rti_internal_typeinfo3varE[8] = {82,97,121,0,4,0,0,0};
.global .align 1 .b8 _ZN21rti_internal_typename3varE[4] = {0x69,0x6e,0x74,0x0};
.global .u32 _ZN21rti_internal_typeenum3varE = 256;
.global .align 1 .b8 _ZN21rti_internal_semantic3varE[1] = {0x0};
.global .align 1 .b8 _ZN23rti_internal_annotation3varE[1] = {0x0};
);


PTXModule materialPtxAttr1f = PTX_MODULE( "material_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 attr;
  .entry ch
  {
    .reg .f32 %f;
    ld.global.f32 %f, [attr];
    st.local.f32 [0], %f;
    ret;
  }
  .entry ah
  {
    .reg .f32 %f;
    ld.global.f32 %f, [attr];
    st.local.f32 [0], %f;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo4attrE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename4attrE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum4attrE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic4attrE[15] = {0x61,0x74,0x74,0x72,0x69,0x62,0x75,0x74,0x65,0x20,0x61,0x74,0x74,0x72,0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation4attrE[1] = {0x0};
);

PTXModule materialPtxAttrMismatchVariableName1f = PTX_MODULE( "material_ptx",
  .version 1.4
  .target sm_10, map_f64_to_f32
  .global .f32 attrVariable;
  .entry ch
  {
    .reg .f32 %f;
    ld.global.f32 %f, [attrVariable];
    st.local.f32 [0], %f;
    ret;
  } 
  .entry ah
  {
    .reg .f32 %f;
    ld.global.f32 %f, [attrVariable];
    st.local.f32 [0], %f;
    ret;
  }
  .global .align 4 .b8 _ZN21rti_internal_typeinfo12attrVariableE[8] = {82,97,121,0,4,0,0,0};
  .global .align 1 .b8 _ZN21rti_internal_typename12attrVariableE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
  .global .u32 _ZN21rti_internal_typeenum12attrVariableE = 256;
  .global .align 1 .b8 _ZN21rti_internal_semantic12attrVariableE[15] = {0x61,0x74,0x74,0x72,0x69,0x62,0x75,0x74,0x65,0x20,0x61,0x74,0x74,0x72,0x0};
  .global .align 1 .b8 _ZN23rti_internal_annotation12attrVariabelE[1] = {0x0};
);
// clang-format on

class ValidateFixture : public Test
{
  public:
    Context createContext( int RayTypeCount = 1, int EntryPointCount = 1, const PTXModule& ptx = progPtxEmpty );
    Program callableProgram( const PTXModule& prog_ptx, const char* name );
    Acceleration noAcceleration();
    Group group( Transform trans, Acceleration acceleration );
    Group group( Transform trans0, Transform trans1, Acceleration acceleration );
    GeometryGroup geometryGroup( GeometryInstance inst, Acceleration acceleration );
    GeometryGroup geometryGroup( GeometryInstance inst0, GeometryInstance inst1, Acceleration acceleration );
    Transform transform( GeometryGroup group );
    GeometryInstance instance( Geometry geom, Material mat );
    GeometryInstance instance( Geometry geom, Material mat0, Material mat1 );
    Geometry geometry( const PTXModule& geometry_ptx );
    GeometryInstance instance( GeometryTriangles geom_tri, Material mat );
    GeometryTriangles geometryTriangles();
    Material material( const PTXModule& material_ptx );
    Program program( const PTXModule& prog_ptx );

    void testLegalCallsInCallablePrograms( const char* expectedErrorMessage );

    RTresult validate()
    {
        try
        {
            context->validate();
        }
        catch( const Exception& e )
        {
            return e.getErrorCode();
        }
        return RT_SUCCESS;
    }

    template <typename O, typename V>
    RTresult set( O& obj, const char* varname, V& val )
    {
        try
        {
            obj[varname]->set( val );
        }
        catch( const Exception& e )
        {
            return e.getErrorCode();
        }
        return RT_SUCCESS;
    }
    // Sets up the test fixture.
    virtual void SetUp() {}

    // Tears down the test fixture.
    virtual void TearDown() { ASSERT_NO_THROW( context->destroy() ); }

    Context          context;
    Geometry         geom;
    Material         mat, mat0, mat1;
    Material         prog, prog0, prog1;
    GeometryInstance gi, gi0, gi1;
};


Context ValidateFixture::createContext( int RayTypeCount, int EntryPointCount, const PTXModule& ptx )
{
    context = Context::create();
    context->setRayTypeCount( RayTypeCount );
    context->setEntryPointCount( EntryPointCount );
    Program ray_gen_program = program( ptx );
    context->setRayGenerationProgram( 0, ray_gen_program );
    return context;
}

Program ValidateFixture::callableProgram( const PTXModule& prog_ptx, const char* name )
{
    return context->createProgramFromPTXString( prog_ptx.code, name );
}

Geometry ValidateFixture::geometry( const PTXModule& geometry_ptx = geometryPtxEmpty )
{
    Geometry geom = context->createGeometry();
    geom->setPrimitiveCount( 1 );
    Program boxProgram = context->createProgramFromPTXString( geometry_ptx.code, "box" );
    geom->setBoundingBoxProgram( boxProgram );
    Program intersectProgram = context->createProgramFromPTXString( geometry_ptx.code, "intersect" );
    geom->setIntersectionProgram( intersectProgram );
    return geom;
}

Material ValidateFixture::material( const PTXModule& material_ptx )
{
    Material material  = context->createMaterial();
    Program  chProgram = context->createProgramFromPTXString( material_ptx.code, "ch" );
    material->setClosestHitProgram( 0, chProgram );
    Program ahProgram = context->createProgramFromPTXString( material_ptx.code, "ah" );
    material->setAnyHitProgram( 0, ahProgram );
    return material;
}

GeometryInstance ValidateFixture::instance( Geometry geom, Material mat )
{
    GeometryInstance inst = context->createGeometryInstance();
    inst->setGeometry( geom );
    inst->setMaterialCount( 1 );
    inst->setMaterial( 0, mat );
    return inst;
}

GeometryTriangles ValidateFixture::geometryTriangles()
{
    Buffer positions = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, 0 );
    Buffer indices   = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, 0 );

    GeometryTriangles geom_tri = context->createGeometryTriangles();
    geom_tri->setPrimitiveCount( 0 );
    geom_tri->setVertices( 0, positions, RT_FORMAT_FLOAT3 );
    geom_tri->setTriangleIndices( indices, RT_FORMAT_UNSIGNED_INT3 );
    geom_tri->setBuildFlags( RTgeometrybuildflags( 0 ) );
    return geom_tri;
}

GeometryInstance ValidateFixture::instance( GeometryTriangles geom_tri, Material mat )
{
    GeometryInstance inst = context->createGeometryInstance();
    inst->setGeometryTriangles( geom_tri );
    inst->setMaterialCount( 1 );
    inst->setMaterial( 0, mat );
    return inst;
}

GeometryInstance ValidateFixture::instance( Geometry geom, Material mat0, Material mat1 )
{
    GeometryInstance inst = context->createGeometryInstance();
    inst->setGeometry( geom );
    inst->setMaterialCount( 2 );
    inst->setMaterial( 0, mat0 );
    inst->setMaterial( 1, mat1 );
    return inst;
}

GeometryGroup ValidateFixture::geometryGroup( GeometryInstance inst, Acceleration acceleration )
{
    GeometryGroup group = context->createGeometryGroup();
    group->setChildCount( 1 );
    group->setChild( 0, inst );
    group->setAcceleration( acceleration );
    return group;
}

GeometryGroup ValidateFixture::geometryGroup( GeometryInstance inst0, GeometryInstance inst1, Acceleration acceleration )
{
    GeometryGroup group = context->createGeometryGroup();
    group->setChildCount( 2 );
    group->setChild( 0, inst0 );
    group->setChild( 0, inst1 );
    group->setAcceleration( acceleration );
    return group;
}

Acceleration ValidateFixture::noAcceleration()
{
    Acceleration accel = context->createAcceleration( "NoAccel", "NoAccel" );
    return accel;
}

Transform ValidateFixture::transform( GeometryGroup group )
{
    Transform trans = context->createTransform();
    trans->setChild( group );
    float m[16];
    m[0]  = 1.0f;
    m[1]  = 0.0f;
    m[2]  = 0.0f;
    m[3]  = 0.0f;
    m[4]  = 0.0f;
    m[5]  = 1.0f;
    m[6]  = 0.0f;
    m[7]  = 0.0f;
    m[8]  = 0.0f;
    m[9]  = 0.0f;
    m[10] = 1.0f;
    m[11] = 0.0f;
    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;
    trans->setMatrix( 0, m, nullptr );
    return trans;
}

Group ValidateFixture::group( Transform trans, Acceleration acceleration )
{
    Group group = context->createGroup();
    group->setChildCount( 1 );
    group->setChild( 0, trans );
    group->setAcceleration( acceleration );
    return group;
}

Group ValidateFixture::group( Transform trans0, Transform trans1, Acceleration acceleration )
{
    Group group = context->createGroup();
    group->setChildCount( 2 );
    group->setChild( 0, trans0 );
    group->setChild( 1, trans1 );
    group->setAcceleration( acceleration );
    return group;
}


Program ValidateFixture::program( const PTXModule& prog_ptx )
{
    Program prog = context->createProgramFromPTXString( prog_ptx.code, "prog" );
    return prog;
}

void ValidateFixture::testLegalCallsInCallablePrograms( const char* expectedErrorMessage )
{
    createContext( 1, 1, progPtxStaticGlobalVarCallableProg );

    Program call_callable = callableProgram( progPtxStaticGlobalVarCallableProg, "call_callable" );
    Program call_trace    = callableProgram( progPtxStaticGlobalVarCallableProg, "call_trace" );
    Program call_potential_intersection =
        callableProgram( progPtxStaticGlobalVarCallableProg, "call_potential_intersection" );

    context["cp_0"]->set( call_callable );

    ASSERT_THAT( set( call_callable, "cp_0", call_trace ), Eq( RT_SUCCESS ) );

    set( call_callable, "cp_0", call_potential_intersection );
    ASSERT_THAT( validate(), Eq( RT_ERROR_ILWALID_CONTEXT ) );
    ASSERT_THAT( context->getErrorString( RT_ERROR_ILWALID_CONTEXT ), HasSubstr( expectedErrorMessage ) );
}

class Validate : public ValidateFixture
{
};

TEST_F( Validate, RayGelwariableTypeMismatch )
{
    createContext();
    context->setRayGenerationProgram( 0, program( progPtxVar1f ) );

    context["var"]->setInt( 0 );  // var should be float

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, MissVariableTypeMismatch )
{
    createContext();
    context["top_object"]->set(
        geometryGroup( instance( geom = geometry( geometryPtxEmpty ), mat = material( materialPtxEmpty ) ), noAcceleration() ) );

    context->setMissProgram( 0, program( progPtxVar1f ) );

    context["var"]->setInt( 0 );  // var should be float

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, ExceptiolwariableTypeMismatch )
{
#if defined( DEBUG ) || defined( DEVELOP )
    // Disable knob for the trivial exception program since that program does not have the side
    // effect that we expect here.
    ScopedKnobSetter localKnobEX( "context.forceTrivialExceptionProgram", false );
#endif

    createContext();
    context["top_object"]->set(
        geometryGroup( instance( geom = geometry( geometryPtxEmpty ), mat = material( materialPtxEmpty ) ), noAcceleration() ) );
    context->setExceptionProgram( 0, program( progPtxVar1f ) );

    context["var"]->setInt( 0 );  // var should be float

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, RayGelwariableElementsMismatch )
{
    createContext();
    context->setRayGenerationProgram( 0, program( progPtxVar1f ) );

    context["var"]->setFloat( 0, 0 );  // var should be float1

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, GeometryVariableTypeMismatch )
{
    createContext();
    context["top_object"]->set(
        geometryGroup( instance( geom = geometry( geometryPtxVar1f ), mat = material( materialPtxVar1f ) ), noAcceleration() ) );

    geom["var"]->setInt( 0 );  // var should be float1

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, MaterialVariableTypeMismatch )
{
    createContext();
    context["top_object"]->set( geometryGroup( instance( geometry(), mat = material( materialPtxVar1f ) ), noAcceleration() ) );

    mat["var"]->setInt( 0 );  // should be float1

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, MaterialVsMaterialVariableTypeMatch )
{
    createContext();
    context["top_object"]->set( geometryGroup(
        instance( geometry(), mat0 = material( materialPtxVar1f ), mat1 = material( materialPtxVar1i ) ), noAcceleration() ) );

    mat0["var"]->setFloat( 0 );
    mat1["var"]->setInt( 0 );

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialVsMaterialVsContextVariableTypeMatch )
{
    createContext();
    context["top_object"]->set( geometryGroup(
        instance( geometry(), mat0 = material( materialPtxVar1f ), mat1 = material( materialPtxVar1i ) ), noAcceleration() ) );

    mat0["var"]->setFloat( 0 );
    mat1["var"]->setInt( 0 );
    context["var"]->setInt( 0 );  // does nothing, all material variables are set

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialVsMaterialVsContextVariableTypeMatch2 )
{
    createContext();
    context["top_object"]->set( geometryGroup(
        instance( geometry(), mat0 = material( materialPtxVar1f ), mat1 = material( materialPtxVar1i ) ), noAcceleration() ) );

    mat0["var"]->setFloat( 0 );
    context["var"]->setInt( 0 );  // override mat1["var"]

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialVsMaterialVsContextVariableTypeMismatch )
{
    createContext();
    context["top_object"]->set( geometryGroup(
        instance( geometry(), mat0 = material( materialPtxVar1f ), mat1 = material( materialPtxVar1i ) ), noAcceleration() ) );
    mat1["var"]->setInt( 0 );

    context["var"]->setInt( 0 );  // override mat0["var"], should be float1

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, MaterialVsRayGelwsContextVariableTypeMatch )
{
    createContext( 1, 1, progPtxVar1f );
    context["top_object"]->set( geometryGroup( instance( geometry(), mat = material( materialPtxVar1i ) ), noAcceleration() ) );

    context->getRayGenerationProgram( 0 )["var"]->setFloat( 0 );
    context["var"]->setInt( 0 );  // override mat["var"]

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialAttributeMatch )
{
    createContext();
    context["top_object"]->set( geometryGroup( instance( geometry( geometryPtxAttr1f ),
                                                         material( materialPtxAttr1f )  // same attribute
                                                         ),
                                               noAcceleration() ) );

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialAttributeMismatch )
{
    createContext();
    context["top_object"]->set( geometryGroup( instance( geometry( geometryPtxAttr1i ),
                                                         material( materialPtxAttr1f )  // different attribute
                                                         ),
                                               noAcceleration() ) );

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, AttributeVariableNamesAreAllowedToMismatch )
{
    // The name of the attribute is the same for the geometry and material programs.
    // The variable name used by the two programs are different.
    createContext();
    context["top_object"]->set( geometryGroup(
        instance( geometry( geometryPtxAttr1f ), material( materialPtxAttrMismatchVariableName1f ) ), noAcceleration() ) );

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, MaterialAttributeNotFound )
{
    createContext();
    context["top_object"]->set( geometryGroup( instance( geometry(),  // no attribute
                                                         mat = material( materialPtxAttr1f ) ),
                                               noAcceleration() ) );

    ASSERT_THAT( validate(), Eq( RT_ERROR_ILWALID_CONTEXT ) );
}

TEST_F( Validate, GeometryTrianglesMaterialAttributeNotFound )
{
    createContext();
    context["top_object"]->set( geometryGroup( instance( geometryTriangles(),  // no attribute
                                                         mat = material( materialPtxAttr1f ) ),
                                               noAcceleration() ) );

    ASSERT_THAT( validate(), Eq( RT_ERROR_ILWALID_CONTEXT ) );
}

TEST_F( Validate, GIVariableTypeMismatch )
{
    createContext();
    geom = geometry( geometryPtxEmpty );
    mat  = material( materialPtxVar1f );
    context["top_object"]->set( geometryGroup( gi0 = instance( geom, mat ), gi1 = instance( geom, mat ), noAcceleration() ) );
    gi0["var"]->setFloat( 0 );

    gi1["var"]->setInt( 0 );  // should be float

    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}

TEST_F( Validate, StaticGlobalVar )
{
    createContext( 1, 1, progPtxStaticGlobalVar1f );

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, UninitializedObjectVariable )
{
    createContext();

    ASSERT_THAT( validate(), Eq( RT_ERROR_VARIABLE_NOT_FOUND ) );
}

TEST_F( Validate, UninitializedCallableProgram )
{
    createContext( 1, 1, progPtxStaticGlobalVarCallableProg );

    ASSERT_THAT( validate(), Eq( RT_ERROR_VARIABLE_NOT_FOUND ) );

    context["cp_0"]->set( callableProgram( progPtxStaticGlobalVarCallableProg, "_Z7store_0v" ) );

    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );
}

TEST_F( Validate, LegalCallsInCallableProgram_rtx )
{
    testLegalCallsInCallablePrograms(
        "Validation error: rtPotentialIntersection is not allowed from bound "
        "callable program call_potential_intersection" );
}

TEST_F( Validate, CallableProgramParameterMismatch )
{
    createContext( 1, 1, progPtxStaticGlobalVarCallableProg );

    Program callableParamMatched = callableProgram( progPtxStaticGlobalVarCallableProg, "_Z7store_0v" );
    context["cp_0"]->set( callableParamMatched );
    ASSERT_THAT( validate(), Eq( RT_SUCCESS ) );

    Program callableParamMismatched = callableProgram( progPtxStaticGlobalVarCallableProg, "_Z7store_1v" );
    context["cp_0"]->set( callableParamMismatched );
    ASSERT_THAT( validate(), Eq( RT_ERROR_TYPE_MISMATCH ) );
}
