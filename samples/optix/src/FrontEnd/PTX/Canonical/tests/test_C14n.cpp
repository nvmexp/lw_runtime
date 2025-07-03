#include <srcTests.h>

#include <SafeLLVMFixture.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/system/System.h>
#include <prodlib/system/Knobs.h>

#define private public
#include <Context/LLVMManager.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/Mangle.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <FrontEnd/Canonical/VariableSemantic.h>
#include <FrontEnd/PTX/Canonical/C14n.h>
#include <FrontEnd/PTX/DataLayout.h>
#include <FrontEnd/PTX/PTXHeader.h>
#include <FrontEnd/PTX/PTXNamespaceMangle.h>
#include <FrontEnd/PTX/PTXtoLLVM.h>
#include <Objects/VariableType.h>
#include <prodlib/exceptions/CompileError.h>
#undef private

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_os_ostream.h>

#include <fstream>
#include <sstream>

using namespace optix;
using namespace testing;
using namespace corelib;
using namespace prodlib;

#define ASSERT_NO_THROW_WITH_MESSAGE( code )                                                                           \
    ASSERT_NO_THROW( try { code; } catch( const std::exception& e ) {                                                  \
        ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );                                    \
    } );

#define MAKE_STRING( ... ) #__VA_ARGS__


//---------------------------------
struct PTXModule
{
    const char* description;
    const char* type;
    const char* metadata;
    const char* code;
};

std::ostream& operator<<(::std::ostream& os, const PTXModule& cf )
{
    return os << cf.description;
}

// Don't put code with comments in here, since the string will be all on one line and any
// comment delineated with ';' will simply comment out everything following it.
#define PTX_MODULE( desc, ... )                                                                                        \
    {                                                                                                                  \
        desc, "PTX", "", #__VA_ARGS__                                                                                  \
    }

#define PTX_MODULE_EX( desc, meta, ... )                                                                               \
    {                                                                                                                  \
        desc, "PTX", meta, #__VA_ARGS__                                                                                \
    }

#define LLVM_MODULE( desc, ... )                                                                                       \
    {                                                                                                                  \
        desc, "LLVM", "", #__VA_ARGS__                                                                                 \
    }

//---------------------------------
struct PTXFileEntryPoint
{
    std::string filename;
    std::string functionName;
};

std::ostream& operator<<(::std::ostream& os, const PTXFileEntryPoint& e )
{
    return os << e.filename << " : " << e.functionName;
}

enum WhichFunc
{
    GET_FUNC,
    PUT_FUNC
};

static const char* getSemanticFunctionName( VariableSemantic semantic, WhichFunc which )
{
    if( semantic != VS_PAYLOAD )
        RT_ASSERT( which != PUT_FUNC );
    switch( semantic )
    {
        case VS_PAYLOAD:
            throw CompileError( RT_EXCEPTION_INFO, "getSemanticFunctionName called for payload" );
        case VS_ATTRIBUTE:
            throw CompileError( RT_EXCEPTION_INFO, "getSemanticFunctionName called for attribute variable" );
        case VS_LAUNCHINDEX:
            return "optixi_getLaunchIndex";
        case VS_LAUNCHDIM:
            return "optixi_getLaunchDim";
        case VS_LWRRENTRAY:
            return "optixi_getLwrrentRay";
        case VS_LWRRENTTIME:
            return "optixi_getLwrrentTime";
        case VS_INTERSECTIONDISTANCE:
            return "optixi_getLwrrentTmax";
        case VS_SUBFRAMEINDEX:
            return "optixi_getSubframeIndex";
    }
    throw CompileError( RT_EXCEPTION_INFO, "Unknown semantic type used in getSemanticFunctionName" );
}


//
class KnobsElwironment : public ::testing::Environment
{
  public:
    virtual ~KnobsElwironment() {}
    // Override this to define how to set up the environment.
    virtual void SetUp()
    {
        // Init the knobs file until we get an actual optix::Context created

        // dump all properties to the config file if an empty config file exists, or if the right elw var is set
        if( getelw( "OPTIX_PROPS" ) || corelib::fileSize( KnobRegistry::getOptixPropsLocation().c_str() ) == 0 )
        {
            std::ofstream out( KnobRegistry::getOptixPropsLocation().c_str() );
            knobRegistry().printKnobs( out );
        }
        // log all non-default knobs
        lprint << "Non-default knobs:\n";
        knobRegistry().printNonDefaultKnobs( lprint_stream );
    }
    // Override this to define how to tear down the environment.
    virtual void TearDown() {}
};

// Create the test.  We can't guarantee order when adding tests in the initialization
// phase, but this doesn't matter for this test.
extern ::testing::Environment* const g_test_elw;  // Prevents unused variable warning
::testing::Environment* const        g_test_elw = ::testing::AddGlobalTestElwironment( new KnobsElwironment() );

//-----------------------------------------------------------------------------
// Basic fixture used in other tests
class C14nFixture : public SafeLLVMFixture
{
  public:
    C14nFixture()
        : m_inputModule( nullptr )
        , m_canonicalModule( nullptr )
        , m_canonicalProgram( nullptr )
        , m_llvmManager( nullptr )
    {
    }

    ~C14nFixture() {}

    virtual void SafeSetUp() {}

    virtual void SafeTearDown()
    {
        // This delete causes the tests to segfault!
        // This comment is a temporary hack to get the tests to run.
        // TODO: revist the tear down to find a definitive solution.
        //delete m_canonicalProgram;
    }

    // Per-test-case set-up
    // Called before the first test in this test case
    static void SetUpTestCase() {}

    // Per-test-case tear-down
    // Called after the last test in this test case
    static void TearDownTestCase() {}

    void ptxToLLVM( const std::string& ptxStr, const char* fullPath = nullptr )
    {
        PTXtoLLVM   frontend( m_llvmManager.llvmContext(), &m_llvmManager.llvmDataLayout() );
        std::string declarations = createPTXHeaderString( toStringView( ptxStr ) );
        m_inputModule            = nullptr;
        try
        {
            m_inputModule = frontend.translate( "unitTest", declarations, {toStringView( ptxStr )}, /*parseLineNumbers=*/true );
            ASSERT_TRUE( m_inputModule != nullptr );
        }
        catch( const std::exception& e )
        {
            ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );
            throw e;
        }
    }

    void loadLLVM( const std::string& llvmStr, const char* fullPath = nullptr )
    {
        m_inputModule = nullptr;
        try
        {
            std::string errorString;
            m_inputModule = loadModuleFromAsmString( m_llvmManager.llvmContext(), llvmStr, &errorString );
            if( !errorString.empty() )
            {
                std::cerr << "Error loading LLVM Module from string:" << std::endl;
                std::cerr << errorString << std::endl;
            }
            ASSERT_TRUE( m_inputModule != nullptr );
        }
        catch( const std::exception& e )
        {
            ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );
            throw e;
        }
    }

    void canonicalize( const std::string& functionName )
    {
        llvm::Function* function = m_inputModule->getFunction( functionName );
        if( !function )
        {
            // Try a mangled name, but we need to ignore the call signature, so only check the prefix.
            std::string mangled = PTXNamespaceMangle( functionName, true, true, "" );
            for( llvm::Module::iterator I = m_inputModule->begin(), IE = m_inputModule->end(); I != IE; ++I )
            {
                // Compare the mangled prefix name to the function name
                if( I->getName().startswith( mangled ) )
                {
                    function = &*I;
                    break;
                }
            }
        }
        if( !function )
        {
            llvm::errs() << "Function '" << functionName << "' not found in module\n";
            llvm::errs() << "begin list of functions: \n";
            for( llvm::Module::iterator I = m_inputModule->begin(), IE = m_inputModule->end(); I != IE; ++I )
            {
                std::string f = I->getName();
                llvm::errs() << "\t" << f << "\n";
            }
            llvm::errs() << "end list of functions: \n";
        }
        ASSERT_THAT( function, NotNull() );

        Context*       context = nullptr;
        ProgramManager PM( nullptr );
        ObjectManager  OM( nullptr );
        C14n test( function, CanonicalizationType::CT_PTX, lwca::SM( 30 ), lwca::SM( 999 ), 0, context, &m_llvmManager, &PM, &OM );
        m_canonicalProgram = test.run();
        ASSERT_THAT( m_canonicalProgram, NotNull() );
        ASSERT_THAT( m_canonicalProgram->llvmFunction(), NotNull() );
        m_canonicalModule = m_canonicalProgram->llvmFunction()->getParent();
    }

    void canonicalizePTXTestFunction( const std::string& ptxStr )
    {
        ptxToLLVM( ptxStr );
        canonicalize( "testFunction" );
    }

    void canonicalizeLLVMTestFunction( const std::string& llvmStr )
    {
        loadLLVM( llvmStr );
        canonicalize( "testFunction" );
    }

    bool canonicalProgramHasVariable( const char* var ) { return m_canonicalModule->getGlobalVariable( var, true ); }

    llvm::Module*       m_inputModule;
    const llvm::Module* m_canonicalModule;
    CanonicalProgram*   m_canonicalProgram;
    LLVMManager         m_llvmManager;
};

// C14nFixture parameterized by PTXModule
class C14nFixture_PTXModule : public C14nFixture, public WithParamInterface<PTXModule>
{
};


class CanonicalizationThrows : public C14nFixture_PTXModule
{
};
SAFETEST_P( CanonicalizationThrows, Test )
{
    ASSERT_ANY_THROW( canonicalizePTXTestFunction( GetParam().code ) );
}

class CanonicalizationSucceeds : public C14nFixture_PTXModule
{
};
SAFETEST_P( CanonicalizationSucceeds, Test )
{
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizePTXTestFunction( GetParam().code ) );
}


///////////////////////////////////////////////////////////////////////////////
//
// Static methods
//
///////////////////////////////////////////////////////////////////////////////

TEST( C14nParseTypename, RecognizesFloat )
{
    VariableType vtype = parseTypename( "float" );

    ASSERT_THAT( vtype, Eq( VariableType( VariableType::Float, 1 ) ) );
}

TEST( C14nParseTypename, RecognizesFloat4 )
{
    VariableType vtype = parseTypename( "float4" );

    ASSERT_THAT( vtype, Eq( VariableType( VariableType::Float, 4 ) ) );
}


///////////////////////////////////////////////////////////////////////////////
//
// Canonicalize variables
//
///////////////////////////////////////////////////////////////////////////////

//------------------------------
// Instance variables
//------------------------------
class InstanceVariableIsCanonicalized : public C14nFixture_PTXModule
{
};
SAFETEST_P( InstanceVariableIsCanonicalized, Test )
{
    canonicalizePTXTestFunction( GetParam().code );

    ASSERT_TRUE(
        m_canonicalModule->getFunction( "optixi_getVariableValue.testFunction_ptx0x0000000000000000.testVariable.i32" ) != nullptr );
    ASSERT_TRUE( m_canonicalProgram->getVariableReferences().size() == 1 );
}

static PTXModule ptxInput_instanceVariables[] = {
    // clang-format off
    PTX_MODULE( "OptiX 3.6 variable with typeenum",
      .version 1.4
      .target sm_10
      .global .u32 testVariable;
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,4,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[9] = {0x75,0x6e,0x73,0x69,0x67,0x6e,0x65,0x64,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .global .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .entry testFunction
      {
        .reg .u32 %r<10>;
        ld.global.u32   %r0, [testVariable];
        mov.u32         %r1, 100;
        st.global.u32   [%r1], %r0;
      }
    ),

    PTX_MODULE( "OptiX <= 3.5 variable without typeenum",
      .version 1.4
      .target sm_10
      .global .u32 testVariable;
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,4,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[9] = {0x75,0x6e,0x73,0x69,0x67,0x6e,0x65,0x64,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .entry testFunction
      {
        .reg .u32 %r<10>;
        ld.global.u32   %r0, [testVariable];
        mov.u32         %r1, 100;
        st.global.u32   [%r1], %r0;
      }
    ),

    PTX_MODULE( "Variables accessed with an offset",
      .version 1.4
      .target sm_10
      .global .u32 testVariable[2];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,8,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x75,0x69,0x6e,0x74,0x32,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .entry testFunction
      {
        .reg .u32 %r<10>;
        ld.global.u32   %r0, [testVariable+4];
        mov.u32         %r1, 100;
        st.global.u32   [%r1], %r0;
      }
    ),

    PTX_MODULE( "Variables accessed with an add offset",
      .version 4.0
      .target sm_20
      .address_size 64
      .global .u32 testVariable[2];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,8,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x75,0x69,0x6e,0x74,0x32,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .entry testFunction
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        mov.u64 %rd0, testVariable;
        add.s64 %rd1, %rd0, 4;
        ldu.global.u32 %r0, [%rd1];
        mov.u32         %r1, 100;
        st.global.u32   [%r1], %r0;
      }
    ),

    PTX_MODULE( "Variables accessed with and without an offset",
      .version 1.4
      .target sm_10
      .global .u32 testVariable[2];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,8,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x75,0x69,0x6e,0x74,0x32,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .entry testFunction
      {
        .reg .u32 %r<10>;
        ld.global.u32   %r0, [testVariable+4];
        ld.global.u32   %r1, [testVariable];
        add.u32         %r2, %r0, %r1;
        mov.u32         %r3, 100;
        st.global.u32   [%r3], %r2;
      }
    ),

    PTX_MODULE( "Boolean variable (not an array)",
      .version 1.4
      .target sm_10
      .global .s8 testVariable;
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,1,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[5] = {0x62,0x6f,0x6f,0x6c,0x0};
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[1] = {0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
      .entry testFunction
      {
        .reg .u32 %r<10>;
        ld.global.u32   %r0, [testVariable];
        mov.u32         %r1, 100;
        st.global.u32   [%r1], %r0;
      }
    ),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( Misc, InstanceVariableIsCanonicalized, ValuesIn( ptxInput_instanceVariables ) );


//------------------------------
// Loads/Stores with phi
//------------------------------
class AmbiguousLoadStores : public C14nFixture_PTXModule
{
};
SAFETEST_P( AmbiguousLoadStores, Test )
{
    // You need {} around the if's since the macros can do funny things.
    if( std::string( GetParam().type ) == "PTX" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizePTXTestFunction( GetParam().code ) );
    }
    else if( std::string( GetParam().type ) == "LLVM" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( GetParam().code ) );
    }
    else
    {
        FAIL();
    }
}

static PTXModule ptxInput_ambigiousLoadStores[] = {

    // rtDeclareVariable(int, val1,,);
    // rtDeclareVariable(int, val2,,);
    // rtDeclareVariable(int, val3,,);
    // rtDeclareVariable(int, which,,);
    // rtBuffer<int> result;

    // RT_PROGRAM void testFunction()
    // {
    //   int val;
    //   switch(which) {
    //   case 1: val = val1; break;
    //   case 2: val = val2; break;
    //   case 3: val = val3; break;
    //   }
    //   result[0] = val;
    // }

    // clang-format off
    PTX_MODULE( "Three variables calls select",
      .version 3.2
      .target sm_20
      .address_size 64

      .global .align 4 .u32 val1;
      .global .align 4 .u32 val2;
      .global .align 4 .u32 val3;
      .global .align 4 .u32 which;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo4val1E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo4val2E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo4val3E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5whichE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename4val1E[4] = {105, 110, 116, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename4val2E[4] = {105, 110, 116, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename4val3E[4] = {105, 110, 116, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5whichE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum4val1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum4val2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum4val3E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5whichE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic4val1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic4val2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic4val3E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5whichE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation4val1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation4val2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation4val3E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5whichE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<4>;
        .reg .s32 	%r<10>;
        .reg .s64 	%rd<7>;


        ldu.global.u32 	%r1, [which];
        setp.eq.s32	%p1, %r1, 1;
        @%p1 bra 	BB0_5;

        setp.eq.s32	%p2, %r1, 2;
        @%p2 bra 	BB0_4;

        setp.ne.s32	%p3, %r1, 3;
        @%p3 bra 	BB0_6;

        ldu.global.u32 	%r9, [val3];
        bra.uni 	BB0_6;

      BB0_4:
        ldu.global.u32 	%r9, [val2];
        bra.uni 	BB0_6;

      BB0_5:
        ldu.global.u32 	%r9, [val1];

      BB0_6:
        cvta.global.u64 	%rd2, result;
        mov.u32 	%r7, 1;
        mov.u32 	%r8, 4;
        mov.u64 	%rd6, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r7, %r8, %rd6, %rd6, %rd6, %rd6);
        st.u32 	[%rd1], %r9;
        ret;
      }
    ),

    // rtDeclareVariable(float2, A1,,);
    // rtDeclareVariable(float2, A2,,);
    // rtDeclareVariable(int, which,,);
    // rtBuffer<float3> result;

    // RT_PROGRAM void testFunction()
    // {
    //   float2 A;
    //   if (which) {
    //     A = A1;
    //   } else {
    //     A = A2;
    //   }
    //   result[0] = A;
    // }
    PTX_MODULE( "Two variables, load from selp on pointers",
      .version 3.2
      .target sm_20
      .address_size 64

      .global .align 8 .b8 A1[8];
      .global .align 8 .b8 A2[8];
      .global .align 4 .u32 which;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A1E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A2E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5whichE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A1E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A2E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5whichE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5whichE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic2A1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2A2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5whichE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5whichE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<2>;
        .reg .s32 	%r<4>;
        .reg .f32 	%f<5>;
        .reg .s64 	%rd<10>;

        ldu.global.u32 	%r3, [which];
        setp.eq.s32	%p1, %r3, 0;
        mov.u64 	%rd7, A1;
        mov.u64 	%rd8, A2;
        selp.b64	%rd9, %rd8, %rd7, %p1;
        cvta.global.u64 	%rd2, result;
        mov.u32 	%r1, 1;
        mov.u32 	%r2, 8;
        mov.u64 	%rd6, 0;
        ld.global.v2.f32 	{%f1, %f2}, [%rd9];
        call (%rd1), _rt_buffer_get_64, (%rd2, %r1, %r2, %rd6, %rd6, %rd6, %rd6);
        st.v2.f32 	[%rd1], {%f1, %f2};
        ret;
      }
    ),

    LLVM_MODULE( "Two variables, load from select on pointers (llvm)",
      target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
      target triple = "lwptx64-lwpu-lwca"

      @A1 = internal addrspace(1) global [8 x i8] zeroinitializer, align 8
      @A2 = internal addrspace(1) global [8 x i8] zeroinitializer, align 8
      @which = internal addrspace(1) global i32 0, align 4
      @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN21rti_internal_typeinfo2A1E = internal addrspace(1) global [8 x i8] c"Ray\00\08\00\00\00", align 4
      @_ZN21rti_internal_typeinfo2A2E = internal addrspace(1) global [8 x i8] c"Ray\00\08\00\00\00", align 4
      @_ZN21rti_internal_typeinfo5whichE = internal addrspace(1) global [8 x i8] c"Ray\00\04\00\00\00", align 4
      @_ZN21rti_internal_typename2A1E = internal addrspace(1) global [7 x i8] c"float2\00", align 1
      @_ZN21rti_internal_typename2A2E = internal addrspace(1) global [7 x i8] c"float2\00", align 1
      @_ZN21rti_internal_typename5whichE = internal addrspace(1) global [4 x i8] c"int\00", align 1
      @_ZN21rti_internal_typeenum2A1E = internal addrspace(1) global i32 4919, align 4
      @_ZN21rti_internal_typeenum2A2E = internal addrspace(1) global i32 4919, align 4
      @_ZN21rti_internal_typeenum5whichE = internal addrspace(1) global i32 4919, align 4
      @_ZN21rti_internal_semantic2A1E = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN21rti_internal_semantic2A2E = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN21rti_internal_semantic5whichE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN23rti_internal_annotation2A1E = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN23rti_internal_annotation2A2E = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN23rti_internal_annotation5whichE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1

      declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

      define void @testFunction() {
      Start:
        %val.i = load i32, i32 addrspace(1)* @which, align 4
        %pred.i = icmp eq i32 %val.i, 0
        %0 = select i1 %pred.i, [8 x i8] addrspace(1)* @A2, [8 x i8] addrspace(1)* @A1
        %1 = bitcast [8 x i8] addrspace(1)* %0 to <2 x float> addrspace(1)*
        %val.i1 = load <2 x float>, <2 x float> addrspace(1)* %1, align 8
        %2 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @result to i64), i32 1, i32 8, i64 0, i64 0, i64 0, i64 0)
        %3 = inttoptr i64 %2 to <2 x float>*
        store <2 x float> %val.i1, <2 x float>* %3
        ret void
      }

      declare i64 @optix.ptx.selp.b64(i64, i64, i1)

      !lwvm.annotations = !{!0}
      !0 = !{void ()* @testFunction, !"kernel", i32 1}
    ),

    // rtDeclareVariable(float2, A1,,);
    // rtDeclareVariable(float2, A2,,);
    // rtDeclareVariable(float2, A3,,);
    // rtDeclareVariable(int, which,,);
    // rtBuffer<float2> result;

    // RT_PROGRAM void testFunction()
    // {
    //   float2 A;
    //   switch(which) {
    //   case 1: A = A1; break;
    //   case 2: A = A2; break;
    //   case 3: A = A3; break;
    //   }
    //   result[0] = A;
    // }
    PTX_MODULE( "Three variables, 4 way phi on loaded values (one is undef)",
      .version 3.2
      .target sm_20
      .address_size 64
      .global .align 8 .b8 A1[8];
      .global .align 8 .b8 A2[8];
      .global .align 8 .b8 A3[8];
      .global .align 4 .u32 which;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A1E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A2E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A3E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5whichE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A1E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A2E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A3E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5whichE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A3E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5whichE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic2A1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2A2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2A3E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5whichE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A3E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5whichE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<4>;
        .reg .s32 	%r<4>;
        .reg .f32 	%f<21>;
        .reg .s64 	%rd<7>;


        ldu.global.u32 	%r1, [which];
        setp.eq.s32	%p1, %r1, 1;
        @%p1 bra 	BB0_5;

        setp.eq.s32	%p2, %r1, 2;
        @%p2 bra 	BB0_4;

        setp.ne.s32	%p3, %r1, 3;
        mov.f32 	%f20, %f11;
        mov.f32 	%f19, %f12;
        @%p3 bra 	BB0_6;

        ld.global.v2.f32 	{%f13, %f14}, [A3];
        mov.f32 	%f20, %f14;
        mov.f32 	%f19, %f13;
        bra.uni 	BB0_6;

      BB0_4:
        ld.global.v2.f32 	{%f15, %f16}, [A2];
        mov.f32 	%f20, %f16;
        mov.f32 	%f19, %f15;
        bra.uni 	BB0_6;

      BB0_5:
        ld.global.v2.f32 	{%f17, %f18}, [A1];
        mov.f32 	%f20, %f18;
        mov.f32 	%f19, %f17;

      BB0_6:
        cvta.global.u64 	%rd2, result;
        mov.u32 	%r2, 1;
        mov.u32 	%r3, 8;
        mov.u64 	%rd6, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r2, %r3, %rd6, %rd6, %rd6, %rd6);
        st.v2.f32 	[%rd1], {%f19, %f20};
        ret;
      }
    ),


    // rtDeclareVariable(float2, A1,,);
    // rtDeclareVariable(float2, A2,,);
    // rtDeclareVariable(float2, B1,,);
    // rtDeclareVariable(float2, B2,,);
    // rtDeclareVariable(int, which,,);
    // rtBuffer<float2> result;

    // RT_PROGRAM void testFunction()
    // {
    //   float2 A, B;
    //   if (which) {
    //     A = A1;
    //     B = B1;
    //   } else {
    //     A = A2;
    //     B = B2;
    //   }
    //   result[0] = A + B;
    // }
    PTX_MODULE( "Two variables, phi on loaded values",
      .version 3.2
      .target sm_20
      .address_size 64

      .global .align 8 .b8 A1[8];
      .global .align 8 .b8 A2[8];
      .global .align 8 .b8 B1[8];
      .global .align 8 .b8 B2[8];
      .global .align 4 .u32 which;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A1E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A2E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2B1E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2B2E[8] = {82, 97, 121, 0, 8, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5whichE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A1E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A2E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2B1E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2B2E[7] = {102, 108, 111, 97, 116, 50, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5whichE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2B1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2B2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5whichE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic2A1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2A2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2B1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2B2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5whichE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2B1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2B2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5whichE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<2>;
        .reg .s32 	%r<4>;
        .reg .f32 	%f<27>;
        .reg .s64 	%rd<7>;


        ldu.global.u32 	%r1, [which];
        setp.eq.s32	%p1, %r1, 0;
        @%p1 bra 	BB0_2;

        ld.global.v2.f32 	{%f13, %f14}, [B1];
        mov.f32 	%f26, %f14;
        mov.f32 	%f25, %f13;
        ld.global.v2.f32 	{%f15, %f16}, [A1];
        mov.f32 	%f24, %f16;
        mov.f32 	%f23, %f15;
        bra.uni 	BB0_3;

      BB0_2:
        ld.global.v2.f32 	{%f17, %f18}, [B2];
        mov.f32 	%f26, %f18;
        mov.f32 	%f25, %f17;
        ld.global.v2.f32 	{%f19, %f20}, [A2];
        mov.f32 	%f24, %f20;
        mov.f32 	%f23, %f19;

      BB0_3:
        cvta.global.u64 	%rd2, result;
        mov.u32 	%r2, 1;
        mov.u32 	%r3, 8;
        mov.u64 	%rd6, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r2, %r3, %rd6, %rd6, %rd6, %rd6);
        add.ftz.f32 	%f21, %f23, %f25;
        add.ftz.f32 	%f22, %f24, %f26;
        st.v2.f32 	[%rd1], {%f21, %f22};
        ret;
      }
    ),

    // rtDeclareVariable(float3, A1,,);
    // rtDeclareVariable(float3, A2,,);
    // rtDeclareVariable(float3, B1,,);
    // rtDeclareVariable(float3, B2,,);
    // rtDeclareVariable(int, which,,);
    // rtBuffer<float3> result;

    // RT_PROGRAM void testFunction()
    // {
    //   float3 A, B;
    //   if (which) {
    //     A = A1;
    //     B = B1;
    //   } else {
    //     A = A2;
    //     B = B2;
    //   }
    //   result[0] = A + B;
    // }
    PTX_MODULE( "Four variables, load from phi on pointers + phi on loaded values",
      .version 3.2
      .target sm_20
      .address_size 64

      .global .align 8 .b8 A1[12];
      .global .align 8 .b8 A2[12];
      .global .align 8 .b8 B1[12];
      .global .align 8 .b8 B2[12];
      .global .align 4 .u32 which;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A1E[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2A2E[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2B1E[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo2B2E[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5whichE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A1E[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2A2E[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2B1E[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename2B2E[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5whichE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2A2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2B1E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum2B2E = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5whichE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic2A1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2A2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2B1E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic2B2E[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5whichE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2A2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2B1E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation2B2E[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5whichE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<2>;
        .reg .s32 	%r<4>;
        .reg .f32 	%f<36>;
        .reg .s64 	%rd<7>;


        ldu.global.u32 	%r1, [which];
        setp.eq.s32	%p1, %r1, 0;
        @%p1 bra 	BB0_2;

        ld.global.f32 	%f32, [A1+8];
        ld.global.f32 	%f35, [B1+8];
        ld.global.v2.f32 	{%f19, %f20}, [B1];
        mov.f32 	%f34, %f20;
        mov.f32 	%f33, %f19;
        ld.global.v2.f32 	{%f21, %f22}, [A1];
        mov.f32 	%f31, %f22;
        mov.f32 	%f30, %f21;
        bra.uni 	BB0_3;

      BB0_2:
        ld.global.f32 	%f32, [A2+8];
        ld.global.f32 	%f35, [B2+8];
        ld.global.v2.f32 	{%f23, %f24}, [B2];
        mov.f32 	%f34, %f24;
        mov.f32 	%f33, %f23;
        ld.global.v2.f32 	{%f25, %f26}, [A2];
        mov.f32 	%f31, %f26;
        mov.f32 	%f30, %f25;

      BB0_3:
        cvta.global.u64 	%rd2, result;
        mov.u32 	%r2, 1;
        mov.u32 	%r3, 12;
        mov.u64 	%rd6, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r2, %r3, %rd6, %rd6, %rd6, %rd6);
        add.ftz.f32 	%f27, %f30, %f33;
        add.ftz.f32 	%f28, %f31, %f34;
        add.ftz.f32 	%f29, %f32, %f35;
        st.f32 	[%rd1+8], %f29;
        st.f32 	[%rd1+4], %f28;
        st.f32 	[%rd1], %f27;
        ret;
      }
    ),

    // struct S {
    //   float a[1];
    // };

    // rtBuffer<S> A;
    // rtDeclareVariable(int, index,,);
    // rtDeclareVariable(int, count,,);
    // rtBuffer<float> result;

    // RT_PROGRAM void testFunction()
    // {
    //   float sum = 0;
    //   S& s = A[index];
    //   for(int i = 0; i < count; ++i)
    //     sum += s.a[i];
    //   result[0] = sum;
    // }
    PTX_MODULE( "Phi in loop",
      .version 3.2
      .target sm_20
      .address_size 64
      .global .align 1 .b8 A[1];
      .global .align 4 .u32 index;
      .global .align 4 .u32 count;
      .global .align 1 .b8 result[1];
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5indexE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 4 .b8 _ZN21rti_internal_typeinfo5countE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5indexE[4] = {105, 110, 116, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename5countE[4] = {105, 110, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum5indexE = 4919;
      .global .align 4 .u32 _ZN21rti_internal_typeenum5countE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic5indexE[1];
      .global .align 1 .b8 _ZN21rti_internal_semantic5countE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5indexE[1];
      .global .align 1 .b8 _ZN23rti_internal_annotation5countE[1];

      .visible .entry testFunction
      {
        .reg .pred 	%p<3>;
        .reg .s32 	%r<11>;
        .reg .f32 	%f<8>;
        .reg .s64 	%rd<17>;


        cvta.global.u64 	%rd5, A;
        ldu.global.u32 	%r6, [index];
        cvt.s64.s32	%rd6, %r6;
        mov.u32 	%r4, 1;
        mov.u32 	%r5, 4;
        mov.u64 	%rd9, 0;
        call (%rd4), _rt_buffer_get_64, (%rd5, %r4, %r5, %rd6, %rd9, %rd9, %rd9);
        ld.global.u32 	%r1, [count];
        setp.gt.s32	%p1, %r1, 0;
        mov.u64 	%rd16, %rd4;
        mov.f32 	%f7, 0f00000000;
        @%p1 bra 	BB0_1;
        bra.uni 	BB0_3;

      BB0_1:
        mov.u32 	%r10, 0;

      BB0_2:
        ld.f32 	%f6, [%rd16];
        add.ftz.f32 	%f7, %f7, %f6;
        add.s64 	%rd16, %rd16, 4;
        add.s32 	%r10, %r10, 1;
        setp.lt.s32	%p2, %r10, %r1;
        @%p2 bra 	BB0_2;

      BB0_3:
        cvta.global.u64 	%rd11, result;
        call (%rd10), _rt_buffer_get_64, (%rd11, %r4, %r5, %rd9, %rd9, %rd9, %rd9);
        st.f32 	[%rd10], %f7;
        ret;
      }
    ),

    LLVM_MODULE( "Phi in loop (llvm)",
      target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
      target triple = "lwptx64-lwpu-lwca"

      @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @index = internal addrspace(1) global i32 0, align 4
      @count = internal addrspace(1) global i32 0, align 4
      @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN21rti_internal_typeinfo5indexE = internal addrspace(1) global [8 x i8] c"Ray\00\04\00\00\00", align 4
      @_ZN21rti_internal_typeinfo5countE = internal addrspace(1) global [8 x i8] c"Ray\00\04\00\00\00", align 4
      @_ZN21rti_internal_typename5indexE = internal addrspace(1) global [4 x i8] c"int\00", align 1
      @_ZN21rti_internal_typename5countE = internal addrspace(1) global [4 x i8] c"int\00", align 1
      @_ZN21rti_internal_typeenum5indexE = internal addrspace(1) global i32 4919, align 4
      @_ZN21rti_internal_typeenum5countE = internal addrspace(1) global i32 4919, align 4
      @_ZN21rti_internal_semantic5indexE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN21rti_internal_semantic5countE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN23rti_internal_annotation5indexE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
      @_ZN23rti_internal_annotation5countE = internal addrspace(1) global [1 x i8] zeroinitializer, align 1

      declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

      define ptx_kernel void @testFunction() {
      Start:
        %val.i7 = load i32, i32 addrspace(1)* @index, align 4
        %p1.i = sext i32 %val.i7 to i64
        %0 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 %p1.i, i64 0, i64 0, i64 0)
        %val.i6 = load i32, i32 addrspace(1)* @count, align 4
        %pred.i2 = icmp sgt i32 %val.i6, 0
        br i1 %pred.i2, label %BB0_2, label %BB0_3

      BB0_2:
        %"%r10.0" = phi i32 [ %r.i, %BB0_2 ], [ 0, %Start ]
        %"%f7.0" = phi float [ %2, %BB0_2 ], [ 0.000000e+00, %Start ]
        %"%rd16.0" = phi i64 [ %r.i1, %BB0_2 ], [ %0, %Start ]
        %1 = inttoptr i64 %"%rd16.0" to float*
        %val.i = load float, float* %1, align 4
        %2 = tail call float @optix.ptx.add.f32.ftz(float %"%f7.0", float %val.i)
        %r.i1 = add i64 %"%rd16.0", 4
        %r.i = add i32 %"%r10.0", 1
        %pred.i = icmp slt i32 %r.i, %val.i6
        br i1 %pred.i, label %BB0_2, label %BB0_3

      BB0_3:
        %"%f7.1" = phi float [ 0.000000e+00, %Start ], [ %2, %BB0_2 ]
        %3 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @result to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
        %4 = inttoptr i64 %3 to float*
        store float %"%f7.1", float* %4
        ret void
      }

      declare float @optix.ptx.add.f32.ftz(float, float)

      !lwvm.annotations = !{!0}
      !0 = !{void ()* @testFunction, !"kernel", i32 1}
    ),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( Misc, AmbiguousLoadStores, ValuesIn( ptxInput_ambigiousLoadStores ) );


//--------------------------------------
// Semantic variables
//--------------------------------------
class SemanticVariableCanonicalizedCorrectly : public C14nFixture_PTXModule
{
  public:
    bool canonicalProgramHasSemantic( const std::string& name )
    {
        const char* funcName = getSemanticFunctionName( getVariableSemanticFromString( name, "" ), GET_FUNC );
        return m_canonicalModule->getFunction( funcName ) != nullptr;
    }
    bool canonicalProgramHasPayload()
    {
        for( const llvm::Function* F : getFunctions( m_canonicalModule ) )
        {
            if( F->getName().startswith(
                    "optixi_getPayloadValue.prd12b.testFunction_ptx0x0000000000000000.testVariable" )
                || F->getName().startswith(
                       "optixi_setPayloadValue.prd12b.testFunction_ptx0x0000000000000000.testVariable" ) )
                return true;
        }
        return false;
    }
};

SAFETEST_P( SemanticVariableCanonicalizedCorrectly, Test )
{
    std::string semantic = GetParam().metadata;

    canonicalizePTXTestFunction( GetParam().code );

    // TODO: We should probably have a more rigorous test
    if( semantic == "rtPayload" )
    {
        ASSERT_TRUE( canonicalProgramHasPayload() );
    }
    else
    {
        ASSERT_TRUE( canonicalProgramHasSemantic( semantic ) );
    }
}

static PTXModule ptxInput_SemanticVariables[] = {
    // clang-format off
    PTX_MODULE_EX( "uint rtLaunchIndex", "rtLaunchIndex",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .u32 testVariable;
      .visible .global .s32 __testing_allowed_global_dummy;
      .entry testFunction
      {
        .reg .u32 %r<3>;
        ld.global.u32 	%r1, [testVariable];
        st.global.s32 	[__testing_allowed_global_dummy], %r1;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,4,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[5] = {0x75,0x69,0x6e,0x74,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[14] = {0x72,0x74,0x4c,0x61,0x75,0x6e,0x63,0x68,0x49,0x6e,0x64,0x65,0x78,0x0}; // rtLaunchIndex
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "uint3 rtLaunchIndex", "rtLaunchIndex",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 16 .b8 testVariable[12];
      .visible .global .align 16 .b8 __testing_allowed_global_dummy[12];
      .entry testFunction
      {
        .reg .u32 %r<5>;
        ld.global.v4.u32 	{%r1,%r2,%r3,_}, [testVariable+0];
        st.global.v2.u32 	[__testing_allowed_global_dummy+0], {%r1,%r2};
        st.global.u32 	[__testing_allowed_global_dummy+8], %r3;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,12,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x75,0x69,0x6e,0x74,0x33,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[14] = {0x72,0x74,0x4c,0x61,0x75,0x6e,0x63,0x68,0x49,0x6e,0x64,0x65,0x78,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "uint3 rtLaunchDim", "rtLaunchDim",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 16 .b8 testVariable[12];
      .visible .global .align 16 .b8 __testing_allowed_global_dummy[12];
      .entry testFunction
      {
        .reg .u32 %r<5>;
        ld.global.v4.u32 	{%r1,%r2,%r3,_}, [testVariable+0];
        st.global.v2.u32 	[__testing_allowed_global_dummy+0], {%r1,%r2};
        st.global.u32 	[__testing_allowed_global_dummy+8], %r3;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,12,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x75,0x69,0x6e,0x74,0x33,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[12] = {0x72,0x74,0x4c,0x61,0x75,0x6e,0x63,0x68,0x44,0x69,0x6d,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "rtLwrrentRay", "rtLwrrentRay",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 16 .b8 testVariable[36];
      .visible .global .align 16 .b8 __testing_allowed_global_dummy[36];
      .entry testFunction
      {
        .reg .u32 %r<3>;
        .reg .f32 %f<10>;
        ld.global.v4.f32 	{%f1,%f2,%f3,_}, [testVariable+0];
        st.global.v2.f32 	[__testing_allowed_global_dummy+0], {%f1,%f2};
        st.global.f32 	[__testing_allowed_global_dummy+8], %f3;
        ld.global.f32 	%f4, [testVariable+12];
        st.global.f32 	[__testing_allowed_global_dummy+12], %f4;
        ld.global.f32 	%f5, [testVariable+16];
        st.global.f32 	[__testing_allowed_global_dummy+16], %f5;
        ld.global.f32 	%f6, [testVariable+20];
        st.global.f32 	[__testing_allowed_global_dummy+20], %f6;
        ld.global.u32 	%r1, [testVariable+24];
        st.global.u32 	[__testing_allowed_global_dummy+24], %r1;
        ld.global.f32 	%f7, [testVariable+28];
        st.global.f32 	[__testing_allowed_global_dummy+28], %f7;
        ld.global.f32 	%f8, [testVariable+32];
        st.global.f32 	[__testing_allowed_global_dummy+32], %f8;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,36,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[11] = {0x6f,0x70,0x74,0x69,0x78,0x3a,0x3a,0x52,0x61,0x79,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[13] = {0x72,0x74,0x43,0x75,0x72,0x72,0x65,0x6e,0x74,0x52,0x61,0x79,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "rtIntersectionDistance", "rtIntersectionDistance",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .f32 testVariable;
      .visible .global .f32 __testing_allowed_global_dummy;
      .entry testFunction
      {
        .reg .f32 %f<3>;
        ld.global.f32 	%f1, [testVariable];
        st.global.f32 	[__testing_allowed_global_dummy], %f1;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,4,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {0x66,0x6c,0x6f,0x61,0x74,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[23] = {0x72,0x74,0x49,0x6e,0x74,0x65,0x72,0x73,0x65,0x63,0x74,0x69,0x6f,0x6e,0x44,0x69,0x73,0x74,0x61,0x6e,0x63,0x65,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "rtLwrrentTime", "rtLwrrentTime",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .f32 testVariable;
      .visible .global .f32 __testing_allowed_global_dummy;
      .entry testFunction
      {
        .reg .f32 %f<3>;
        ld.global.f32 	%f1, [testVariable];
        st.global.f32 	[__testing_allowed_global_dummy], %f1;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[6] = {102, 108, 111, 97, 116, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4919;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[14] = {114, 116, 67, 117, 114, 114, 101, 110, 116, 84, 105, 109, 101, 0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "rtPayload load", "rtPayload",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 16 .b8 testVariable[12];
      .visible .global .align 16 .b8 __testing_allowed_global_dummy[12];
      .entry testFunction
      {
        .reg .f32 %f<5>;
        ld.global.v4.f32  {%f1,%f2,%f3,_}, [testVariable+0];
        st.global.v2.f32  [__testing_allowed_global_dummy+0], {%f1,%f2};
        st.global.f32     [__testing_allowed_global_dummy+8], %f3;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,12,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[8] = {0x50,0x61,0x79,0x6c,0x6f,0x61,0x64,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[10] = {0x72,0x74,0x50,0x61,0x79,0x6c,0x6f,0x61,0x64,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),

    PTX_MODULE_EX( "rtPayload store", "rtPayload",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .visible .global .align 16 .b8 testVariable[12];
      .entry testFunction
      {
        .reg .u32 %r<5>;
        .reg .f32 %f<5>;
        mov.u32         %r1, 100;
        ld.global.v4.f32 	{%f1,%f2,%f3,_}, [%r1+0];
        st.global.v2.f32 	[testVariable+0], {%f1,%f2};
        st.global.f32 	[testVariable+8], %f3;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo12testVariableE[8] = {82,97,121,0,12,0,0,0};
      .global .align 1 .b8 _ZN21rti_internal_typename12testVariableE[8] = {0x50,0x61,0x79,0x6c,0x6f,0x61,0x64,0x0};
      .global         .u32 _ZN21rti_internal_typeenum12testVariableE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic12testVariableE[10] = {0x72,0x74,0x50,0x61,0x79,0x6c,0x6f,0x61,0x64,0x0};
      .global .align 1 .b8 _ZN23rti_internal_annotation12testVariableE[1] = {0x0};
    ),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( Misc, SemanticVariableCanonicalizedCorrectly, ValuesIn( ptxInput_SemanticVariables ) );

//--------------------------------------
// Attribute variables
//--------------------------------------
class AttributeVariableCanonicalizedCorrectly : public C14nFixture
{
  public:
    char         name[64];
    VariableType vtype;
    unsigned int numElements;

    void parserParams( const char* metadata )
    {
        char type[64]{};
        EXPECT_EQ( sscanf( metadata, "attribute %s %63s", name, type ), 2 );
        vtype = parseTypename( type );
    }
};

SAFETEST_F( AttributeVariableCanonicalizedCorrectly, TestSet )
{
    // clang-format off
 PTXModule inputProgram
    PTX_MODULE_EX( "store y to float3 shading_normal attribute", "attribute shading_normal float3",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 4 .b8 shading_normal[12];
      .entry testFunction
      {
        .reg .s32 %r<2>;
        mov.u32 %r1, 0;
        st.global.u32 [shading_normal+4], %r1;
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo14shading_normalE[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename14shading_normalE[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum14shading_normalE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic14shading_normalE[25] = {97, 116, 116, 114, 105, 98, 117, 116, 101, 32, 115, 104, 97, 100, 105, 110, 103, 95, 110, 111, 114, 109, 97, 108, 0};
      .global .align 1 .b8 _ZN23rti_internal_annotation14shading_normalE[1];
    );
    // clang-format on

    parserParams( inputProgram.metadata );
    canonicalizePTXTestFunction( inputProgram.code );
    const VariableReference* attrRef = m_canonicalProgram->findAttributeReference( std::string( "_attribute_" ) + name );

    ASSERT_THAT( m_canonicalModule->getFunction(
                     std::string( "optixi_setAttributeValue.testFunction_ptx0x0000000000000000._attribute_" ) + name + ".i32" ),
                 NotNull() );
    ASSERT_THAT( attrRef, NotNull() );
    ASSERT_THAT( attrRef->getType(), Eq( vtype ) );
}

SAFETEST_F( AttributeVariableCanonicalizedCorrectly, TestGet )
{
    // clang-format off
  PTXModule inputProgram
    PTX_MODULE_EX( "load y from float3 shading_normal attribute", "attribute shading_normal float3",
      .version 1.4
      .target sm_10, map_f64_to_f32
      .global .align 4 .b8 shading_normal[12];
      .entry testFunction
      {
        .reg .s32 %r<2>;
        mov.u32 %r1, 0;
        ld.global.u32 %r1, [shading_normal+4];
        st.local.u32 [0], %r1;  // to prevent eliding of the ld
        ret;
      }
      .global .align 4 .b8 _ZN21rti_internal_typeinfo14shading_normalE[8] = {82, 97, 121, 0, 12, 0, 0, 0};
      .global .align 1 .b8 _ZN21rti_internal_typename14shading_normalE[7] = {102, 108, 111, 97, 116, 51, 0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum14shading_normalE = 256;
      .global .align 1 .b8 _ZN21rti_internal_semantic14shading_normalE[25] = {97, 116, 116, 114, 105, 98, 117, 116, 101, 32, 115, 104, 97, 100, 105, 110, 103, 95, 110, 111, 114, 109, 97, 108, 0};
      .global .align 1 .b8 _ZN23rti_internal_annotation14shading_normalE[1];
    );
    // clang-format on

    parserParams( inputProgram.metadata );
    canonicalizePTXTestFunction( inputProgram.code );
    const VariableReference* attrRef = m_canonicalProgram->findAttributeReference( std::string( "_attribute_" ) + name );

    ASSERT_THAT( m_canonicalModule->getFunction(
                     std::string( "optixi_getAttributeValue.testFunction_ptx0x0000000000000000._attribute_" ) + name + ".i32" ),
                 NotNull() );
    ASSERT_THAT( attrRef, NotNull() );
    ASSERT_THAT( attrRef->getType(), Eq( vtype ) );
}

//-----------------------------------------------
// Historical rti_internal_register variables
//-----------------------------------------------
static PTXModule ptxInput_registerVariables[] = {
    // clang-format off
    PTX_MODULE( "Ray Index",
    .version 1.4
    .target sm_10
    .global .u32 _ZN21rti_internal_register14reg_rayIndex_xE;
    .global .u32 _ZN21rti_internal_register14reg_rayIndex_yE;
    .global .u32 _ZN21rti_internal_register14reg_rayIndex_zE;
    .entry testFunction
    {
      .reg .u32 %r<10>;
      .reg .u32 %addr;
      mov.u32 %addr, 100;
      ld.volatile.global.u32  %r0, [_ZN21rti_internal_register14reg_rayIndex_xE];
      st.global.u32 [%addr], %r0;
      ld.volatile.global.u32  %r1, [_ZN21rti_internal_register14reg_rayIndex_yE];
      st.global.u32 [%addr], %r1;
      ld.volatile.global.u32  %r2, [_ZN21rti_internal_register14reg_rayIndex_zE];
      st.global.u32 [%addr], %r2;
    }
    ),

    PTX_MODULE( "Exception detail 0-8",
      .version 1.4
      .target sm_10
      .global .u32 _ZN21rti_internal_register21reg_exception_detail0E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail1E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail2E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail3E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail4E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail5E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail6E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail7E;
      .global .u32 _ZN21rti_internal_register21reg_exception_detail8E;
      .entry testFunction
      {
        .reg .u32 %r<10>;
        .reg .u32 %addr;
        mov.u32 %addr, 100;
        ld.volatile.global.u32  %r0, [_ZN21rti_internal_register21reg_exception_detail0E];
        st.global.u32 [%addr], %r0;
        ld.volatile.global.u32  %r1, [_ZN21rti_internal_register21reg_exception_detail1E];
        st.global.u32 [%addr], %r1;
        ld.volatile.global.u32  %r2, [_ZN21rti_internal_register21reg_exception_detail2E];
        st.global.u32 [%addr], %r2;
        ld.volatile.global.u32  %r3, [_ZN21rti_internal_register21reg_exception_detail3E];
        st.global.u32 [%addr], %r3;
        ld.volatile.global.u32  %r4, [_ZN21rti_internal_register21reg_exception_detail4E];
        st.global.u32 [%addr], %r4;
        ld.volatile.global.u32  %r5, [_ZN21rti_internal_register21reg_exception_detail5E];
        st.global.u32 [%addr], %r5;
        ld.volatile.global.u32  %r6, [_ZN21rti_internal_register21reg_exception_detail6E];
        st.global.u32 [%addr], %r6;
        ld.volatile.global.u32  %r7, [_ZN21rti_internal_register21reg_exception_detail7E];
        st.global.u32 [%addr], %r7;
        ld.volatile.global.u32  %r8, [_ZN21rti_internal_register21reg_exception_detail8E];
        st.global.u32 [%addr], %r8;
      }
    ),

    PTX_MODULE( "Exception detail 64-bit 0-6",
      .version 1.4
      .target sm_10
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail0E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail1E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail2E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail3E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail4E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail5E;
      .global .u64 _ZN21rti_internal_register24reg_exception_64_detail6E;
      .entry testFunction
      {
        .reg .u64 %rd<10>;
        .reg .u32 %addr;
        mov.u32 %addr, 100;
        ld.volatile.global.u32  %rd0, [_ZN21rti_internal_register24reg_exception_64_detail0E];
        st.global.u64 [%addr], %rd0;
        ld.volatile.global.u32  %rd1, [_ZN21rti_internal_register24reg_exception_64_detail1E];
        st.global.u64 [%addr], %rd1;
        ld.volatile.global.u32  %rd2, [_ZN21rti_internal_register24reg_exception_64_detail2E];
        st.global.u64 [%addr], %rd2;
        ld.volatile.global.u32  %rd3, [_ZN21rti_internal_register24reg_exception_64_detail3E];
        st.global.u64 [%addr], %rd3;
        ld.volatile.global.u32  %rd4, [_ZN21rti_internal_register24reg_exception_64_detail4E];
        st.global.u64 [%addr], %rd4;
        ld.volatile.global.u32  %rd5, [_ZN21rti_internal_register24reg_exception_64_detail5E];
        st.global.u64 [%addr], %rd5;
        ld.volatile.global.u32  %rd6, [_ZN21rti_internal_register24reg_exception_64_detail6E];
        st.global.u64 [%addr], %rd6;
      }
    ),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( RegisterVariables, CanonicalizationSucceeds, ValuesIn( ptxInput_registerVariables ) );


//------------------------------------------------
// Invalid rti_internal_register variables
//------------------------------------------------

static PTXModule makeInternalRegisterPTXModule( const char* desc, const char* type, const char* reg )
{
    char buffer[1024];
    // clang-format off
  const char* baseCodeFormat = MAKE_STRING
  (
    .version 1.4
    .target sm_10
    .global %s %s;
    .entry testFunction
    {
      .reg %s aReg;
      ld.volatile.global%s  aReg, [%s];
    }
  );
    // clang-format on
    sprintf( buffer, baseCodeFormat, type, reg, type, type, reg );

    static std::vector<std::string> code;
    code.push_back( buffer );

    PTXModule fragment;
    fragment.description = desc;
    fragment.code        = code.back().c_str();
    return fragment;
}


#define MAKE makeInternalRegisterPTXModule
static const PTXModule ptxInput_IlwalidRegisterVariables[] = {
    // clang-format off
    MAKE( "Exception detail 9",          ".u32", "_ZN21rti_internal_register21reg_exception_detail9E" ),
    MAKE( "Exception detail 64-bit 7",   ".u64", "_ZN21rti_internal_register24reg_exception_64_detail7E" ),

    // Legacy internal registers:
    MAKE( "reg_buf_ids",                 ".u64", "_ZN21rti_internal_register11reg_buf_idsE" ),
    MAKE( "reg_ray_tmax",                ".f32", "_ZN21rti_internal_register12reg_ray_tmaxE" ),
    MAKE( "reg_tex_refs",                ".u64", "_ZN21rti_internal_register12reg_tex_refsE" ),
    MAKE( "reg_page_table",              ".u64", "_ZN21rti_internal_register14reg_page_tableE" ),
    MAKE( "reg_return_vpc",              ".u32", "_ZN21rti_internal_register14reg_return_vpcE" ),
    MAKE( "reg_virtual_pc",              ".u32", "_ZN21rti_internal_register14reg_virtual_pcE" ),
    MAKE( "reg_launchDim_x",             ".u32", "_ZN21rti_internal_register15reg_launchDim_xE" ),
    MAKE( "reg_launchDim_y",             ".u32", "_ZN21rti_internal_register15reg_launchDim_yE" ),
    MAKE( "reg_launchDim_z",             ".u32", "_ZN21rti_internal_register15reg_launchDim_zE" ),
    MAKE( "reg_program_ids",             ".u64", "_ZN21rti_internal_register15reg_program_idsE" ),
    MAKE( "reg_trav_bvh_sp",             ".u32", "_ZN21rti_internal_register15reg_trav_bvh_spE" ),
    MAKE( "reg_trav_bvh_lwr",            ".u32", "_ZN21rti_internal_register16reg_trav_bvh_lwrE" ),
    MAKE( "reg_trav_bvh_tmax",           ".f32", "_ZN21rti_internal_register17reg_trav_bvh_tmaxE" ),
    MAKE( "reg_trav_bvh_tmin",           ".f32", "_ZN21rti_internal_register17reg_trav_bvh_tminE" ),
    MAKE( "reg_object_records",          ".u64", "_ZN21rti_internal_register18reg_object_recordsE" ),
    MAKE( "reg_trav_bvh_stack",          ".u32", "_ZN21rti_internal_register18reg_trav_bvh_stackE" ),
    MAKE( "reg_vpc_visitation",          ".u32", "_ZN21rti_internal_register18reg_vpc_visitationE" ),
    MAKE( "reg_lwrrent_program",         ".u32", "_ZN21rti_internal_register19reg_lwrrent_programE" ),
    MAKE( "reg_warp_activation",         ".u32", "_ZN21rti_internal_register19reg_warp_activationE" ),
    MAKE( "reg_page_request_bits",       ".u64", "_ZN21rti_internal_register21reg_page_request_bitsE" ),
    MAKE( "reg_trav_bvh_prim_end",       ".u32", "_ZN21rti_internal_register21reg_trav_bvh_prim_endE" ),
    MAKE( "reg_trav_bvh_prim_begin",     ".u32", "_ZN21rti_internal_register23reg_trav_bvh_prim_beginE" ),
    MAKE( "reg_page_reference_bytes",    ".u64", "_ZN21rti_internal_register24reg_page_reference_bytesE" ),
    MAKE( "reg_kparams_entryPointIndex", ".u32", "_ZN21rti_internal_register27reg_kparams_entryPointIndexE" ),
    // clang-format on
};
#undef MAKE

INSTANTIATE_TEST_SUITE_P( IlwalidRegisterVariables, CanonicalizationThrows, ValuesIn( ptxInput_IlwalidRegisterVariables ) );


//-------------------------
//  Buffer variables
//-------------------------

class BufferVariablesCanonicalizedCorrectly : public C14nFixture_PTXModule
{
  public:
    char     name[64];
    unsigned expectedNumDims;
    unsigned expectedElementSize;

    void parseBufferParams( const char* metadata )
    {
        EXPECT_EQ( sscanf( metadata, "buffer %s dim %u elementSize %u", name, &expectedNumDims, &expectedElementSize ), 3 );
        strncpy( name, canonicalMangleVariableName( name ).c_str(), 64 );
        name[63] = '\0';
    }
};

SAFETEST_P( BufferVariablesCanonicalizedCorrectly, Test )
{
    parseBufferParams( GetParam().metadata );

    canonicalizePTXTestFunction( GetParam().code );
    const VariableReference* bufferVar = m_canonicalProgram->findVariableReference( name );

    ASSERT_THAT( bufferVar, NotNull() );
    ASSERT_THAT( bufferVar->getType(),
                 Eq( VariableType( VariableType( VariableType::Buffer, expectedElementSize, expectedNumDims ) ) ) );
}

static const PTXModule ptxInput_bufferVariables[] = {
    // clang-format off
    PTX_MODULE_EX( "_rt_buffer_get_64 with non-constant offset", "buffer buf0 dim 2 elementSize 4",
      .version 3.2
      .target sm_30
      .global .align 1 .b8 buf0[1];
      .visible .entry testFunction()
      {
        .reg .s32 %r<8>;
        .reg .s64 %rd<16>;
        .reg .f32 %f<10>;
        cvta.global.u64 %rd2, buf0;
        mov.u32 %r1, 2;
        mov.u32 %r4, 4;
        mov.u64 %rd4, 1;
        mov.u64 %rd12, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r1, %r4, %rd12, %rd4, %rd12, %rd12);
        ld.local.s64 %rd14, [0];
        add.s64 %rd15, %rd1, %rd14;
        ld.global.s32 %r4, [%rd15+32];
        st.global.u32 [%rd15+24], %r4;
        ret;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get_64 with dependent load", "buffer buf0 dim 2 elementSize 4",
      .version 3.2
      .target sm_30
      .global .align 1 .b8 buf0[1];
      .global .align 8 .u64 __testing_allowed_global_dummy;
      .visible .entry testFunction()
      {
        .reg .s32 %r<6>;
        .reg .s64 %rd<13>;
        cvta.global.u64 %rd2, buf0;
        mov.u32 %r1, 2;
        mov.u32 %r4, 4;
        mov.u64 %rd4, 1;
        mov.u64 %rd12, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r1, %r4, %rd12, %rd4, %rd12, %rd12);
        ld.u32 %r3, [%rd1];
        st.global.u32 [__testing_allowed_global_dummy], %r3;
        ret;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get_64 with dependent store", "buffer buf0 dim 2 elementSize 4",
      .version 3.2
      .target sm_30
      .global .align 1 .b8 buf0[1];
      .visible .entry testFunction()
      {
        .reg .s32 %r<6>;
        .reg .s64 %rd<13>;
        cvta.global.u64 %rd2, buf0;
        mov.u32 %r1, 2;
        mov.u32 %r4, 4;
        mov.u64 %rd4, 1;
        mov.u64 %rd12, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r1, %r4, %rd12, %rd4, %rd12, %rd12);
        st.u32 [%rd1], %r4;
        ret;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get_size_64", "buffer buf0 dim 2 elementSize 4",
      .version 3.2
      .target sm_30
      .global .align 1 .b8 buf0[1];
      .visible .entry testFunction()
      {
        .reg .s32 %r<6>;
        .reg .s64 %rd<13>;
        cvta.global.u64   %rd5, buf0;
        mov.u32 %r1, 2;
        mov.u32 %r4, 4;
        call (%rd1, %rd2, %rd3, %rd4), _rt_buffer_get_size_64, (%rd5, %r1, %r4);
        ret;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get with three loads", "buffer buf_in dim 1 elementSize 32",
      .version 1.4
      .target sm_10, map_f64_to_f32

      .global .align 1 .b8 buf_in[1];

      .entry testFunction
      {
        .reg .u32 %r<6>;
        .reg .u64 %rda1;
        .reg .u64 %rd<17>;
        .reg .f32 %f<5>;
        mov.u64   %rd1, buf_in;
        mov.u32   %r1, 1;
        mov.u32   %r3, 32;
        mov.u64   %rd3, 0;
        call (%rd11), _rt_buffer_get_64, (%rd1, %r1, %r3, %rd3, %rd3, %rd3, %rd3);
        ld.global.f32   %f1, [%rd11+16];
        ld.global.f32   %f2, [%rd11+20];
        ld.global.f32   %f3, [%rd11+12];
        mov.u64   %rd13, 0xffff0000;
        st.global.f32   [%rd13+0], %f3;
        st.global.f32   [%rd13+4], %f1;
        st.global.f32   [%rd13+8], %f2;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get with three loads + three stores", "buffer buf_in dim 1 elementSize 32",
      .version 1.4
      .target sm_10, map_f64_to_f32

      .global .align 1 .b8 buf_in[1];
      .global .align 1 .b8 buf_out[1];

      .entry testFunction
      {
        .reg .u32 %r<10>;
        .reg .u64 %rda1;
        .reg .u64 %rd<26>;
        .reg .f32 %f<5>;
        mov.u64   %rd1, buf_in;
        mov.u32   %r1, 1;
        mov.u32   %r3, 32;
        mov.u64   %rd3, 0;
        call (%rd11), _rt_buffer_get_64, (%rd1, %r1, %r3, %rd3, %rd3, %rd3, %rd3);
        ld.global.f32   %f1, [%rd11+16];
        ld.global.f32   %f2, [%rd11+20];
        ld.global.f32   %f3, [%rd11+12];
        mov.u64 	%rd13, buf_out;
        mov.u32 	%r5, 1;
        mov.u32 	%r7, 12;
        mov.u64 	%rd15, 0;
        call (%rd23), _rt_buffer_get_64, (%rd13, %r5, %r7, %rd15, %rd15, %rd15, %rd15);
        st.global.f32 	[%rd23+0], %f3;
        st.global.f32 	[%rd23+4], %f1;
        st.global.f32 	[%rd23+8], %f2;
        ret;
      }
    ),

    PTX_MODULE_EX( "_rt_buffer_get_64 with variable in namespace", "buffer buffers::buf0 dim 2 elementSize 4",
      .version 3.2
      .target sm_30
      .global .align 1 .b8 _ZN7buffers4buf0E[1];
      .visible .entry testFunction()
      {
        .reg .s32 %r<8>;
        .reg .s64 %rd<16>;
        .reg .f32 %f<10>;
        cvta.global.u64 %rd2, _ZN7buffers4buf0E;
        mov.u32 %r1, 2;
        mov.u32 %r4, 4;
        mov.u64 %rd4, 1;
        mov.u64 %rd12, 0;
        call (%rd1), _rt_buffer_get_64, (%rd2, %r1, %r4, %rd12, %rd4, %rd12, %rd12);
        ld.local.s64 %rd14, [0];
        add.s64 %rd15, %rd1, %rd14;
        ld.global.s32 %r4, [%rd15+32];
        st.global.u32 [%rd15+24], %r4;
        ret;
      }
    ),
    // clang-format on

};

INSTANTIATE_TEST_SUITE_P( Misc, BufferVariablesCanonicalizedCorrectly, ValuesIn( ptxInput_bufferVariables ) );


//-------------------------
//  Atomics to Buffers
//-------------------------

class AtomicsToBuffersCanonicalizedCorrectly : public C14nFixture_PTXModule
{
  public:
#if 0
    char     name[64];

    void parseBufferParams( const char* metadata )
    {
        EXPECT_EQ( sscanf( metadata, "buffer %s dim %u elementSize %u", name, &expectedNumDims, &expectedElementSize ), 3 );
        strncpy( name, canonicalMangleVariableName( name ).c_str(), 64 );
        name[63] = '\0';
    }
#endif
};

SAFETEST_P( AtomicsToBuffersCanonicalizedCorrectly, Test )
{
    //parseBufferParams( GetParam().metadata );

    canonicalizePTXTestFunction( GetParam().code );
    // const VariableReference* bufferVar = m_canonicalProgram->findVariableReference( name );

    // ASSERT_THAT( bufferVar, NotNull() );
    // ASSERT_THAT( bufferVar->getType(),
    //              Eq( VariableType( VariableType( VariableType::Buffer, expectedElementSize, expectedNumDims ) ) ) );
}

static const PTXModule ptxInput_atomicBuffers[] = {
    // clang-format off
    PTX_MODULE_EX( "atomicAdd to int", "parameters",
      .version 3.2
      .target sm_30
      .global .align 4 .b8 buf0[1];
      .visible .entry testFunction()
      {
          .reg .s32 dims;
          .reg .s32 elem_size;
          .reg .s64 x;
          .reg .s64 zero;
          .reg .s32 output;
          .reg .s64 buf0ptr;
          .reg .u64 optixBufPtr<2>;
          .reg .s64 offsetPtr;
        cvta.global.u64 buf0ptr, buf0;
        mov.u32 dims, 2;
        mov.u32 elem_size, 4;
        mov.u64 x, 1;
        mov.u64 zero, 0;
        call (optixBufPtr0), _rt_buffer_get_64, (buf0ptr, dims, elem_size, x, zero, zero, zero);
        atom.add.u32 	output, [optixBufPtr0], 10;
        ret;
      }
    ),
    PTX_MODULE_EX( "atomicAdd to int2", "parameters",
      .version 3.2
      .target sm_30
      .global .align 4 .b8 buf0[1];
      .visible .entry testFunction()
      {
          .reg .s32 dims;
          .reg .s32 elem_size;
          .reg .s64 x;
          .reg .s64 zero;
          .reg .s32 output;
          .reg .s64 buf0ptr;
          .reg .u64 optixBufPtr<2>;
          .reg .s64 offsetPtr;
        cvta.global.u64 buf0ptr, buf0;
        mov.u32 dims, 2;
        mov.u32 elem_size, 8;
        mov.u64 x, 1;
        mov.u64 zero, 0;
        call (optixBufPtr0), _rt_buffer_get_64, (buf0ptr, dims, elem_size, x, zero, zero, zero);
        atom.add.u32 	output, [optixBufPtr0], 10;
        call (optixBufPtr1), _rt_buffer_get_64, (buf0ptr, dims, elem_size, x, zero, zero, zero);
        add.s64 	offsetPtr, optixBufPtr1, 4;
        atom.add.u32 	output, [offsetPtr], 11;
        ret;
      }
    ),
    // clang-format on

};

INSTANTIATE_TEST_SUITE_P( Misc, AtomicsToBuffersCanonicalizedCorrectly, ValuesIn( ptxInput_atomicBuffers ) );


//---------------------------------
//  Texture variables
//---------------------------------
class TextureVariable : public C14nFixture
{
};
// TODO: We used to loop over instructions looking for calls. We would process the inline
// asms that we are using as a stop-gap for textures. We can either resurrect that
// code when it becomes necessary (see c14n.cpp#36) or wait for LWVM to handle
// textures correctly.
SAFETEST_F( TextureVariable, DISABLED_CanonicalizedCorrectly )
{
    // clang-format off
  const char* code = MAKE_STRING
  (
    .version 3.2
    .target sm_30
    .global .texref tex0;
    .visible .entry testFunction()
    {
      .reg .s32 %r<8>;
      .reg .f32 %f<5>;
      .reg .s64 %rd<8>;
      mov.f32 %f2, 0f00000000;
      tex.2d.v4.u32.f32 {%r1, %r2, %r3, %r4}, [tex0, {%f2, %f2}];
      ret;
    }
  );
    // clang-format on

    canonicalizePTXTestFunction( code );
    const VariableReference* textureVar = m_canonicalProgram->findVariableReference( "tex0" );

    ASSERT_THAT( textureVar, NotNull() );
    ASSERT_THAT( textureVar->getType(), Eq( VariableType( VariableType::TextureSampler, 16, 2 ) ) );
}


//---------------------------------
//  User call variables
//---------------------------------
class UserCallVariable : public C14nFixture
{
};
// TODO: Waiting for the dust to settle on James's changes to callable programs.
// The PTX code may need to be changed.
SAFETEST_F( UserCallVariable, DISABLED_CanonicalizedCorrectly )
{
    // clang-format off
  const char* code = MAKE_STRING
  (
    .version 3.2
    .target sm_30
    .visible .func  (.param .b32 func_retval0) callable_prog (.param .b32 callable_prog_param_0)
    {
      .reg .s32 %r<2>;

      mov.u32 %r1, 0;
      st.param.b32  [func_retval0+0], %r1;
      ret;
    }
    .visible .entry testFunction()
    {
      .reg .s32 %r<4>;
      .reg .s64 %rd<7>;
      mov.u32 %r1, 1;
      {
        .reg .b32 temp_param_reg;
        .param .b32 param0;
        st.param.b32  [param0+0], %r1;
        .param .b32 retval0;
        call.uni (retval0), callable_prog, (param0);
        ld.param.b32  %r3, [retval0+0];
      }
      ret;
    }
  );
    // clang-format on

    canonicalizePTXTestFunction( code );
    const VariableReference* userCallVar = m_canonicalProgram->findVariableReference( "callable_prog" );

    ASSERT_THAT( userCallVar, NotNull() );
}

//---------------------------------
//  Callable Program Callees
//---------------------------------
class CallableProgramCallee : public C14nFixture_PTXModule
{
};
SAFETEST_P( CallableProgramCallee, CanonicalizedWithoutError )
{
    // You need {} around the if's since the macros can do funny things.
    if( std::string( GetParam().type ) == "PTX" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizePTXTestFunction( GetParam().code ) );
    }
    else if( std::string( GetParam().type ) == "LLVM" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( GetParam().code ) );
    }
    else
    {
        FAIL();
    }
}

static PTXModule ptxInput_callableProgramCallee[] = {
    // clang-format off
    PTX_MODULE( "No args",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b64 many_args_buffer;
      .visible .func  () testFunction()
      {
        .reg .f32 	%f<2>;
        .reg .b64 	%rd<2>;
        mov.f32 	%f1, 0f3DCCCCCD;
        ld.b64  %rd1, [many_args_buffer];
        st.f32	[%rd1+0], %f1;
        ret;
      }
    ),
    PTX_MODULE( "float2",
      .version 4.2
      .target sm_20
      .address_size 64
      .visible .func  (.param .align 8 .b8 func_retval0[8]) testFunction(
        .param .align 8 .b8 param_1[8]
      )
      {
        .reg .f32 	%f<8>;
        ld.param.f32 	%f1, [param_1+4];
        ld.param.f32 	%f2, [param_1];
        neg.f32 %f3, %f1;
        neg.f32 %f4, %f2;
        st.param.f32	[func_retval0+0], %f3;
        st.param.f32	[func_retval0+4], %f4;
        ret;
      }
    ),
    PTX_MODULE( "size_t2",
      .version 4.2
      .target sm_20
      .address_size 64
      .visible .func  (.param .align 16 .b8 func_retval0[16]) testFunction(
        .param .align 16 .b8 param_1[16]
      )
      {
        .reg .b64 	%rd<8>;
        ld.param.b64 	%rd1, [param_1+0];
        ld.param.b64 	%rd2, [param_1+8];
        neg.s64 %rd4, %rd1;
        neg.s64 %rd5, %rd2;
        st.param.b64	[func_retval0+0], %rd4;
        st.param.b64	[func_retval0+8], %rd5;
        ret;
      }
    ),
    PTX_MODULE( "short2",
      .version 4.2
      .target sm_20
      .address_size 64
      .visible .func  (.param .align 4 .b8 func_retval0[4]) testFunction(
        .param .align 4 .b8 param_1[4]
      )
      {
        .reg .s16 	%r<8>;
        ld.param.s16 	%r1, [param_1+0];
        ld.param.s16 	%r2, [param_1+2];
        neg.s16 %r4, %r1;
        neg.s16 %r5, %r2;
        st.param.s16	[func_retval0+0], %r4;
        st.param.s16	[func_retval0+2], %r5;
        ret;
      }
    ),
    PTX_MODULE( "pixar_create_ray",
      .version 4.2
      .target sm_20
      .address_size 64
      .visible .func  (.param .align 4 .b8 func_retval0[48]) testFunction(
        .param .align 8 .b8 _Z17thinLensCreateRay6float2S_f_param_0[8],
        .param .align 8 .b8 _Z17thinLensCreateRay6float2S_f_param_1[8],
        .param .b32 _Z17thinLensCreateRay6float2S_f_param_2
      )
      {
        .reg .pred 	%p<5>;
        .reg .f32 	%f<183>;


        ld.param.f32 	%f17, [_Z17thinLensCreateRay6float2S_f_param_0+4];
        ld.param.f32 	%f18, [_Z17thinLensCreateRay6float2S_f_param_0];
        ld.param.f32 	%f19, [_Z17thinLensCreateRay6float2S_f_param_1];
        ld.param.f32 	%f20, [_Z17thinLensCreateRay6float2S_f_param_1+4];
        add.ftz.f32 	%f1, %f18, %f17;
        add.ftz.f32 	%f2, %f19, %f20;
        neg.f32       %f3, %f18;
        neg.f32       %f4, %f17;
        neg.f32       %f5, %f19;
        neg.f32       %f6, %f20;
        neg.f32       %f7, %f1;
        neg.f32       %f8, %f2;
        add.ftz.f32   %f9, %f17, %f19;
        add.ftz.f32   %f10, %f17, %f20;
        add.ftz.f32   %f11, %f18, %f19;
        add.ftz.f32   %f12, %f18, %f20;
        st.param.f32	[func_retval0+0], %f1;
        st.param.f32	[func_retval0+4], %f2;
        st.param.f32	[func_retval0+8], %f3;
        st.param.f32	[func_retval0+12], %f4;
        st.param.f32	[func_retval0+16], %f5;
        st.param.f32	[func_retval0+20], %f6;
        st.param.f32	[func_retval0+24], %f7;
        st.param.f32	[func_retval0+28], %f8;
        st.param.f32	[func_retval0+32], %f9;
        st.param.f32	[func_retval0+36], %f10;
        st.param.f32	[func_retval0+40], %f11;
        st.param.f32	[func_retval0+44], %f12;
        ret;
      }
    ),
    // clang-format on

};

INSTANTIATE_TEST_SUITE_P( CallableProgram, CallableProgramCallee, ValuesIn( ptxInput_callableProgramCallee ) );


//---------------------------------
//  Callable Program Callers
//---------------------------------
class CallableProgramCaller : public C14nFixture_PTXModule
{
  public:
    bool canonicalProgramHasFunction( const std::string& name )
    {
        return m_canonicalModule->getFunction( name ) != nullptr;
    }
};
SAFETEST_P( CallableProgramCaller, FoundCanonicalizedFunction )
{
    // You need {} around the if's since the macros can do funny things.
    if( std::string( GetParam().type ) == "PTX" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizePTXTestFunction( GetParam().code ) );
    }
    else if( std::string( GetParam().type ) == "LLVM" )
    {
        ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( GetParam().code ) );
    }
    else
    {
        FAIL();
    }
    std::string functionName( GetParam().metadata );

    std::stringstream ss;
    ss << functionName << m_canonicalProgram->getFunctionSignature();

    if( !functionName.empty() )
    {
        ASSERT_TRUE( canonicalProgramHasFunction( ss.str() ) );
    }
}

static PTXModule ptxInput_callableProgramCaller[] = {
    // clang-format off
    PTX_MODULE_EX( "Bound", "optixi_callBound.testFunction_ptx0x0000000000000000.testVariable.sig",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    PTX_MODULE_EX( "Bindless", "optixi_callBindless.testFunction_ptx0x0000000000000000.testVariable00000000.sig",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4920;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    PTX_MODULE_EX( "Bindless from variable int", "optixi_callBindless.sig",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4919;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    PTX_MODULE_EX( "Bindless from buffer", "optixi_callBindless.testFunction_ptx0x0000000000000000.functions00000000.sig",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 1 .b8 functions[1];
      .visible .func  () testFunction()
      {
        .reg .u32 	%r<4>;
        .reg .s64 	%rd<19>;
        mov.u64 	%rd9, functions;
        cvta.global.u64 	%rd7, %rd9;
        mov.u32 	%r1, 1;
        mov.u32 	%r2, 4;
        mov.u64 	%rd8, 0;
        call (%rd10), _rt_buffer_get_64, (%rd7, %r1, %r2, %rd8, %rd8, %rd8, %rd8);
        ld.u32 	%r3, [%rd10];
        call (%rd1), _rt_callable_program_from_id_64, (%r3);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    PTX_MODULE( "No use of bound callable program variable",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        ret;
      }
    ),
    PTX_MODULE_EX( "Select on two bindless callable program variables", "optixi_callBindless.sig",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4920;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .global .align 4 .b8 testVariable2[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo13testVariable2E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename13testVariable2E[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum13testVariable2E = 4920;
      .global .align 1 .b8  _ZN21rti_internal_semantic13testVariable2E[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation13testVariable2E[1] = {0};
      .global .align 4 .u32 val1;
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<5>;
        .reg .s64 	%rd<9>;
        .reg .pred  %p;
        ldu.global.u32 	%r1, [testVariable];
        ldu.global.u32 	%r2, [testVariable2];
        ldu.global.u32  %r3, [val1];
        setp.eq.u32 %p, %r3, 0;
        selp.u32 %r4, %r1, %r2, %p;
        call (%rd1), _rt_callable_program_from_id_64, (%r4);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    PTX_MODULE( "2 uses of same bound callable program var in two _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ldu.global.u32 	%r2, [testVariable];
        call (%rd2), _rt_callable_program_from_id_64, (%r2);
        prototype_1 : .callprototype () _ ();
        call (), %rd2, (), prototype_1;
        ret;
      }
    ),
    PTX_MODULE( "2 calls using same function pointer from _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        prototype_1 : .callprototype () _ ();
        call (), %rd1, (), prototype_1;
        ret;
      }
    ),
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( CallableProgram, CallableProgramCaller, ValuesIn( ptxInput_callableProgramCaller ) );

class CallableProgramThrows : public C14nFixture_PTXModule
{
};
SAFETEST_P( CallableProgramThrows, CanonicalizationThrowsError )
{
    // You need {} around the if's since the macros can do funny things.
    try
    {
        if( std::string( GetParam().type ) == "PTX" )
        {
            canonicalizePTXTestFunction( GetParam().code );
        }
        else if( std::string( GetParam().type ) == "LLVM" )
        {
            canonicalizeLLVMTestFunction( GetParam().code );
        }
        else
        {
            FAIL() << "Unknown test code type: " << GetParam().type;
        }
        FAIL() << "No Expected exception thrown";
    }
    catch( const CompileError& e )
    {
        std::cout << "Expected exception thrown: " << e.getDescription() << "\n";
    }
    catch( ... )
    {
        FAIL() << "Wrong type of exception thrown";
    }
}

static PTXModule ptxInput_callableProgramThrows[] = {
    // clang-format off
    PTX_MODULE( "Call + store bound callable program var",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        mov.b64 %rd2, 0;
        st.global.u32 [%rd2], %r1;
        ret;
      }
    ),
    PTX_MODULE( "Store bound callable program var",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        mov.b64 %rd2, 0;
        st.global.u32 [%rd2], %r1;
        ret;
      }
    ),
    PTX_MODULE( "2 uses of same bound callable program var in _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        call (%rd2), _rt_callable_program_from_id_64, (%r1);
        prototype_1 : .callprototype () _ ();
        call (), %rd2, (), prototype_1;
        ret;
      }
    ),
    PTX_MODULE( "No use of result of _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        ret;
      }
    ),
    PTX_MODULE( "Store result of _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        mov.b64 %rd2, 0;
        st.global.u64 [%rd2], %rd1;
        ret;
      }
    ),
    PTX_MODULE( "Call + store result of _rt_callable_program_from_id_64",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        mov.b64 %rd2, 0;
        st.global.u64 [%rd2], %rd1;
        ret;
      }
    ),
    PTX_MODULE( "Use of _rt_callable_program_from_id_64 in call, but not as function pointer",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<4>;
        .reg .s64 	%rd<9>;
        ldu.global.u32 	%r1, [testVariable];
        call (%rd1), _rt_callable_program_from_id_64, (%r1);
        {
          .param .b64 param0;
          st.param.b64 [param0], %rd1;
          prototype_0 : .callprototype () _ (.param .b64 _);
          mov.b64 %rd2, 1337;
          call (), %rd2, (param0), prototype_0;
          }
        ret;
      }
    ),
    PTX_MODULE( "Select on two bound callable program variables",
      .version 4.2
      .target sm_20
      .address_size 64
      .global .align 4 .b8 testVariable[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo12testVariableE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename12testVariableE[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum12testVariableE = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic12testVariableE[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation12testVariableE[1] = {0};
      .global .align 4 .b8 testVariable2[4];
      .global .align 4 .b8  _ZN21rti_internal_typeinfo13testVariable2E[8] = {82, 97, 121, 0, 4, 0, 0, 0};
      .global .align 1 .b8  _ZN21rti_internal_typename13testVariable2E[1] = {0};
      .global .align 4 .u32 _ZN21rti_internal_typeenum13testVariable2E = 4921;
      .global .align 1 .b8  _ZN21rti_internal_semantic13testVariable2E[1] = {0};
      .global .align 1 .b8  _ZN23rti_internal_annotation13testVariable2E[1] = {0};
      .global .align 4 .u32 val1;
      .visible .func  () testFunction()
      {
        .reg .s32 	%r<5>;
        .reg .s64 	%rd<9>;
        .reg .pred  %p;
        ldu.global.u32 	%r1, [testVariable];
        ldu.global.u32 	%r2, [testVariable2];
        ldu.global.u32  %r3, [val1];
        setp.eq.u32 %p, %r3, 0;
        selp.u32 %r4, %r1, %r2, %p;
        call (%rd1), _rt_callable_program_from_id_64, (%r4);
        prototype_0 : .callprototype () _ ();
        call (), %rd1, (), prototype_0;
        ret;
      }
    ),
    // clang-format on

};

INSTANTIATE_TEST_SUITE_P( CallableProgram, CallableProgramThrows, ValuesIn( ptxInput_callableProgramThrows ) );

//---------------------------------
//  Indirect calls to non-callable programs trigger exception.
//---------------------------------

class IndirectCallVariable : public C14nFixture
{
};

SAFETEST_F( IndirectCallVariable, CanonicalizedCorrectly )
{
    // clang-format off
  const char* code = MAKE_STRING
  (
    .version 4.2
    .target sm_35
    .address_size 64

    .visible .func support_function () {}

    .visible .global .u64 support_function_pointer = support_function;

    .visible .entry testFunction()
    {
      .reg .u64   %rd<1>;
      mov.u64  %rd0, support_function_pointer;
      prototype : .callprototype _ ();
      call %rd0, (), prototype;
      ret;
    }
  );
    // clang-format on

    ASSERT_ANY_THROW( canonicalizePTXTestFunction( code ) );
}


///////////////////////////////////////////////////////////////////////////////
//
// Functions
//
///////////////////////////////////////////////////////////////////////////////

//----------------------------------------------------------
// Add state struct as first parameter to each function
//----------------------------------------------------------
class CanonicalizationAddsStateParameter : public C14nFixture_PTXModule
{
  public:
    llvm::Function* findCanonicalFunction( const char* name ) { return m_canonicalModule->getFunction( name ); }

    bool firstArgIsPointerToState( const llvm::Function* func )
    {
        if( !func || func->arg_size() == 0 )
            return false;

        const llvm::Type* argType = func->arg_begin()->getType();
        if( !argType->isPointerTy() )
            return false;

        llvm::Type* stateType = func->getParent()->getTypeByName( "struct.cort::CanonicalState" );
        if( !stateType )
            return false;
        return argType->getPointerElementType() == stateType;
    }
};

SAFETEST_P( CanonicalizationAddsStateParameter, Test )
{
    canonicalizePTXTestFunction( GetParam().code );
    const llvm::Function* testFunctionC   = m_canonicalProgram->llvmFunction();
    llvm::Function*       deviceFunctionC = findCanonicalFunction( "deviceFunction" );

    ASSERT_TRUE( firstArgIsPointerToState( testFunctionC ) );
    if( deviceFunctionC )
    {
        ASSERT_TRUE( firstArgIsPointerToState( deviceFunctionC ) );
    }
}


static const PTXModule ptxInput_functions[] = {
    // clang-format off
    PTX_MODULE( "No parameters",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
      }
    ),

    PTX_MODULE( "AABB-style parameters",
      .version 1.4
      .target sm_10
      .entry testFunction(
          .param .s32 __lwdaparm__Z6boundsiPf_primIdx,
          .param .u64 __lwdaparm__Z6boundsiPf_result)
      {
      }
    ),

  // TODO: Disabling these for now. The test is supposed to show deviceFunction getting the
  // state parameter added to its argument list. The optimizer either removes the device functions
  // or optimizes the state parameter from the list because it is not used and the device
  // function doesn't have external linkage after canonicalization.

    //PTX_MODULE( "device function with no parameters",
    //  .version 1.4
    //  .target sm_10
    //  .visible .func deviceFunction()
    //  {
    //  }

    //  .entry testFunction()
    //  {
    //    call (), deviceFunction, ();
    //  }
    //),

    //PTX_MODULE( "callable program arguments",
    //  .version 3.2
    //  .target sm_30
    //  .extern .func  sink( .reg .b32 val );
    //  .visible .func  (.param .b32 func_retval0) deviceFunction (.param .b32 _Z13callable_progi_param_0)
    //  {
    //    .reg .s32  %r<2>;
    //    mov.u32 %r1, 1234;
    //    st.param.b32  [func_retval0+0], %r1;
    //    call.uni (), sink, (%r1);
    //    ret;
    //  }
    //  .visible .entry testFunction()
    //  {
    //    .reg .s32 %r<4>;
    //    .reg .s64 %rd<7>;
    //    mov.u32 %r1, 1;
    //    {
    //      .reg .b32 temp_param_reg;
    //      .param .b32 param0;
    //      st.param.b32  [param0+0], %r1;
    //      .param .b32 retval0;
    //      call.uni (retval0), deviceFunction, (param0);
    //      ld.param.b32  %r3, [retval0+0];
    //    }
    //    ret;
    //  }
    //),

    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( Functions, CanonicalizationAddsStateParameter, ValuesIn( ptxInput_functions ) );

static const PTXModule ptxInput_relwrsion_functions[] = {
    // clang-format off
    // clang-format off
    PTX_MODULE( "device function with parameters (also relwrsive)",
     .version 3.2
     .target sm_30
     .extern .func sink( .reg .b32 val );
     .visible .func deviceFunction(.reg .u32 %r1)
     {
       .reg .pred %p<3>;
       ld.global.u32 %r1, [%r1];
       setp.eq.s32	%p1, %r1, 0;
       @%p1 bra 	skip;
       call (), deviceFunction, (%r1);
     skip:
       call (), sink, ( %r1 );
       ret;
     }
     .entry testFunction()
     {
       .reg .u32 %r<10>;
       call (), deviceFunction, (%r1);
     }
    ),
    // clang-format on
};

class CanonicalizationAddsStateParameterRelwrsion : public CanonicalizationAddsStateParameter
{
};

SAFETEST_P( CanonicalizationAddsStateParameterRelwrsion, RelwrsionDisallowed )
{
#if defined( DEBUG ) || defined( DEVELOP )
    // Override anyone's knobs
    ScopedKnobSetter enableRelwrsiveFunctions( "c14n.enableRelwrsiveFunctions", false );
#endif
    try
    {
        canonicalizePTXTestFunction( GetParam().code );
    }
    catch( const CompileError& e )
    {
        std::string expected( "Compile Error: Found relwrsive call to deviceFunction" );
        ASSERT_STREQ( expected.c_str(), e.getDescription().substr( 0, expected.size() ).c_str() );
    }
}


SAFETEST_P_DEV( CanonicalizationAddsStateParameterRelwrsion, RelwrsionAllowed )
{
    ScopedKnobSetter enableRelwrsiveFunctions( "c14n.enableRelwrsiveFunctions", true );
    canonicalizePTXTestFunction( GetParam().code );
    const llvm::Function* testFunctionC   = m_canonicalProgram->llvmFunction();
    llvm::Function*       deviceFunctionC = findCanonicalFunction( "deviceFunction" );

    ASSERT_TRUE( firstArgIsPointerToState( testFunctionC ) );
    if( deviceFunctionC )
    {
        ASSERT_TRUE( firstArgIsPointerToState( deviceFunctionC ) );
    }
}

INSTANTIATE_TEST_SUITE_P( Functions, CanonicalizationAddsStateParameterRelwrsion, ValuesIn( ptxInput_relwrsion_functions ) );

class CanonicalizationCallsFunction : public C14nFixture_PTXModule
{
  public:
    bool canonicalProgramCallsFunction( const std::string& name )
    {
        return m_canonicalModule->getFunction( name ) != nullptr;
    }
};

SAFETEST_P( CanonicalizationCallsFunction, Test )
{
    std::string semantic = GetParam().metadata;

    canonicalizePTXTestFunction( GetParam().code );

    ASSERT_TRUE( canonicalProgramCallsFunction( semantic ) );
}


// Built-in functions
static const PTXModule ptxInput_builtinfunctions[] = {
    // clang-format off
    PTX_MODULE_EX( "_rti_compute_geometry_instance_aabb_64", "optixi_computeGeometryInstanceAABB",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        call (), _rti_compute_geometry_instance_aabb_64, (%r1, %r2, %r3, %rd3);
      }
    ),
    PTX_MODULE_EX( "_rti_compute_group_child_aabb_64", "optixi_computeGroupChildAABB",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        call (), _rti_compute_group_child_aabb_64, (%r1, %r2, %rd3);
      }
    ),
    PTX_MODULE_EX( "_rti_gather_motion_aabbs_64", "optixi_gatherMotionAABBs",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        call (), _rti_gather_motion_aabbs_64, (%r1, %rd2);
      }
    ),
    PTX_MODULE_EX( "_rti_get_aabb_request", "optixi_getAabbRequest",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        call (), _rti_get_aabb_request, (%rd1);
      }
    ),
    PTX_MODULE_EX( "_rti_get_status_return", "optixi_getFrameStatus",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u64 %rd<10>;
        call (%rd1), _rti_get_status_return, ();
      }
    ),
    PTX_MODULE_EX( "_rt_trace_64", "optixi_trace.testFunction_ptx0x0000000000000000.prd24b",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<13>;
        .reg .u64 %rd<13>;
        .reg .f32 %f<13>;
        .local .u64 %prd[3];
        mov.u64 %rd11, %prd;
        mov.u32 %r12, 24;
        call _rt_trace_64, ( %r1, %f2, %f3, %f4, %f5, %f6, %f7, %r8, %f9, %f10, %rd11, %r12 );
      }
    ),
    PTX_MODULE_EX( "_rt_trace_with_time_64", "optixi_trace.testFunction_ptx0x0000000000000000.prd24b",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<13>;
        .reg .u64 %rd<13>;
        .reg .f32 %f<13>;
        .local .u64 %prd[3];
        mov.u64 %rd11, %prd;
        mov.u32 %r12, 24;
        call _rt_trace_with_time_64, ( %r1, %f2, %f3, %f4, %f5, %f6, %f7, %r8, %f9, %f10, %f11, %rd11, %r12 );
      }
    ),
    PTX_MODULE_EX( "_rt_ignore_intersection", "optixi_ignoreIntersection",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        call _rt_ignore_intersection, ( );
      }
    ),
    PTX_MODULE_EX( "_rt_transform_tuple", "optixi_transformTuple",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r1;
        .reg .f32 %f<13>;
        mov.s32 %r1, 7937;
        call (%f0, %f1, %f2, %f3), _rt_transform_tuple, ( %r1, %f4, %f5, %f6, %f7 );
      }
    ),
    PTX_MODULE_EX( "_rt_get_primitive_index", "optixi_getPrimitiveIndex",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rt_get_primitive_index, ();
      }
    ),
    PTX_MODULE_EX( "_rti_get_instance_flags", "optixi_getInstanceFlags",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rti_get_instance_flags, ();
      }
    ),
    PTX_MODULE_EX( "_rt_get_ray_flags", "optixi_getRayFlags",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rt_get_ray_flags, ();
      }
    ),
    PTX_MODULE_EX( "_rt_get_ray_mask", "optixi_getRayMask",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rt_get_ray_mask, ();
      }
    ),
    PTX_MODULE_EX( "_rt_is_triangle_hit", "optixi_isTriangleHit",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg.u32 %r<10>;
        call( %r0 ), _rt_is_triangle_hit, ( );
    }
    ),
    PTX_MODULE_EX( "_rt_is_triangle_hit_back_face", "optixi_isTriangleHitBackFace",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg.u32 %r<10>;
        call( %r0 ), _rt_is_triangle_hit_back_face, ( );
    }
    ),
    PTX_MODULE_EX( "_rt_is_triangle_hit_front_face", "optixi_isTriangleHitFrontFace",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg.u32 %r<10>;
        call( %r0 ), _rt_is_triangle_hit_front_face, ( );
    }
    ),
    PTX_MODULE_EX( "_rt_get_triangle_barycentrics", "optixi_getTriangleBarycentrics",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg .f32 	%f<3>;
        call (%f1, %f2), _rt_get_triangle_barycentrics, ();
    }
    ),
    PTX_MODULE_EX( "_rt_get_lowest_group_child_index", "optixi_getLowestGroupChildIndex",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg.u32 %r<10>;
        call (%r0), _rt_get_lowest_group_child_index, ();
    }
    ),
    PTX_MODULE_EX( "_rti_report_full_intersection_ff", "optixi_reportFullIntersection.noUniqueName.float2",
            .version 1.4
            .target sm_10
            .entry testFunction()
    {
        .reg .f32 	%f<3>;
        .reg .u32   %r<3>;
        call (%r0), _rti_report_full_intersection_ff, (%f0, %r1, %r2, %f1, %f2);
    }
    ),
    PTX_MODULE_EX( "_rt_get_exception_code", "optixi_getExceptionCode",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rt_get_exception_code, ();
      }
    ),
    // _rt_potential_intersection and _rt_report_intersection must always be paired.
    PTX_MODULE_EX( "_rt_potential_intersection", "optixi_isPotentialIntersection",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .pred  %p<2>;
        .reg .f32   %f<2>;
        .reg .s32   %r<4>;

        mov.f32   %f1, 0f00000000;
        // inline asm
        call (%r1), _rt_potential_intersection, (%f1);
        // inline asm
        setp.eq.s32 %p1, %r1, 0;
        @%p1 bra  BB1_2;

        mov.u32   %r3, 0;
        // inline asm
        call (%r2), _rt_report_intersection, (%r3);
        // inline asm

      BB1_2:
        ret;
      }
    ),
    PTX_MODULE_EX( "_rt_report_intersection", "optixi_reportIntersection",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .pred  %p<2>;
        .reg .f32   %f<2>;
        .reg .s32   %r<4>;

        mov.f32   %f1, 0f00000000;
        // inline asm
        call (%r1), _rt_potential_intersection, (%f1);
        // inline asm
        setp.eq.s32 %p1, %r1, 0;
        @%p1 bra  BB1_2;

        mov.u32   %r3, 0;
        // inline asm
        call (%r2), _rt_report_intersection, (%r3);
        // inline asm

      BB1_2:
        ret;
      }
    ),
    PTX_MODULE_EX( "_rt_get_entry_point_index", "optixi_getEntryPointIndex",
      .version 1.4
      .target sm_10
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (%r0), _rt_get_entry_point_index, ();
      }
    ),
    /* TODO - get these working or kill
    PTX_MODULE_EX( "_rt_print_start_64", "printf",
      .version 1.4
      .target sm_10
      .const .align 1 .b8 formatString[64] = {0x43,0x61,0x75,0x67,0x68,0x74,0x20,0x52,0x54,0x5f,0x45,0x58,0x43,0x45,0x50,0x54,0x49,0x4f,0x4e,0x5f,0x53,0x54,0x41,0x43,0x4b,0x5f,0x4f,0x56,0x45,0x52,0x46,0x4c,0x4f,0x57,0xa,0x20,0x20,0x6c,0x61,0x75,0x6e,0x63,0x68,0x20,0x69,0x6e,0x64,0x65,0x78,0x20,0x3a,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0xa,0x0};
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        mov.u64 %rd1, formatString;
        mov.u32 %r1, 64;
        call (%r0), _rt_print_start_64, (%rd1, %r2);
      }
    ),
    PTX_MODULE_EX( "_rt_print_write32 - 32bit arg", "printf",
      .version 1.4
      .target sm_10
      .const .align 1 .b8 formatString[64] = {0x43,0x61,0x75,0x67,0x68,0x74,0x20,0x52,0x54,0x5f,0x45,0x58,0x43,0x45,0x50,0x54,0x49,0x4f,0x4e,0x5f,0x53,0x54,0x41,0x43,0x4b,0x5f,0x4f,0x56,0x45,0x52,0x46,0x4c,0x4f,0x57,0xa,0x20,0x20,0x6c,0x61,0x75,0x6e,0x63,0x68,0x20,0x69,0x6e,0x64,0x65,0x78,0x20,0x3a,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0xa,0x0};
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        .reg .pred %p<10>;
        mov.u64 %rd1, formatString;
        mov.u32 %r1, 64;
        call (%r0), _rt_print_start_64, (%rd1, %r2);
        mov.u32         %r3, 0;
        setp.eq.s32     %p1, %r0, %r3;
  @%p1 bra        skip;
        mov.u32 %r4, 0;
        call (), _rt_print_write32, (%r4, %r5);
        mov.u32 %r6, 99;
        call (), _rt_print_write32, (%r6, %r7);
      skip:
      }
    ),

    PTX_MODULE_EX( "_rt_print_write32 - 64bit arg", "printf",
      .version 1.4
      .target sm_10
      .const .align 1 .b8 formatString[64] = {0x43,0x61,0x75,0x67,0x68,0x74,0x20,0x52,0x54,0x5f,0x45,0x58,0x43,0x45,0x50,0x54,0x49,0x4f,0x4e,0x5f,0x53,0x54,0x41,0x43,0x4b,0x5f,0x4f,0x56,0x45,0x52,0x46,0x4c,0x4f,0x57,0xa,0x20,0x20,0x6c,0x61,0x75,0x6e,0x63,0x68,0x20,0x69,0x6e,0x64,0x65,0x78,0x20,0x3a,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0x2c,0x20,0x25,0x64,0xa,0x0};
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        .reg .u64 %rd<10>;
        .reg .pred %p<10>;
        mov.u64 %rd1, formatString;
        mov.u32 %r1, 64;
        call (%r0), _rt_print_start_64, (%rd1, %r2);
        mov.u32         %r3, 0;
        setp.eq.s32     %p1, %r0, %r3;
  @%p1 bra        skip;
        mov.u32 %r4, 1;
        call (), _rt_print_write32, (%r4, %r5);
        mov.u32 %r6, 98;
        call (), _rt_print_write32, (%r6, %r7);
        mov.u32 %r8, 0;
        call (), _rt_print_write32, (%r8, %r9);
      skip:
      }
    ),
    */
    // clang-format on
};

INSTANTIATE_TEST_SUITE_P( BuiltinFunctions, CanonicalizationCallsFunction, ValuesIn( ptxInput_builtinfunctions ) );


///////////////////////////////////////////////////////////////////////////////
//
// Utility functions for catalog tests
//
///////////////////////////////////////////////////////////////////////////////

class CanonicalizationFromFileSucceeds : public C14nFixture, public WithParamInterface<PTXFileEntryPoint>
{
  public:
    void loadLLVMModuleFromPTXFile( const std::string& fullpath )
    {
        std::ifstream file( fullpath.c_str(), std::ios_base::in );
        EXPECT_TRUE( file.good() );
        std::string ptxStr;
        ptxStr.assign( ( std::istreambuf_iterator<char>( file ) ), std::istreambuf_iterator<char>() );

        ptxToLLVM( ptxStr, fullpath.c_str() );
        file.close();
    }
};

SAFETEST_P( CanonicalizationFromFileSucceeds, Test )
{
    std::string fullpath = dataPath() + "/ptxInputs/" + GetParam().filename;
    loadLLVMModuleFromPTXFile( fullpath );

    ASSERT_NO_THROW_WITH_MESSAGE( canonicalize( GetParam().functionName ) );
}

SAFETEST_F( CanonicalizationFromFileSucceeds, DISABLED_TestSingle )
{
    // Use this test to load in single files
    std::string fullpath = "C:/code/2rtsdk/rtmain/build-64-vs14-c80/lib/ptx_sdk/optixHello_generated_draw_color.lw.ptx";
    loadLLVMModuleFromPTXFile( fullpath );

    ASSERT_NO_THROW_WITH_MESSAGE( canonicalize( "draw_solid_color" ) );
}

static std::vector<PTXFileEntryPoint> readCatalogIndex( const std::string& indexfile )
{
    std::vector<PTXFileEntryPoint> list;
    std::string                    fullname = dataPath() + "/ptxInputs/" + indexfile;
    std::ifstream                  indexFile( fullname.c_str() );
    if( !indexFile.good() )
        return list;
    std::string line;
    while( std::getline( indexFile, line ) )
    {
        if( line.substr( 0, 2 ) == "//" )
            continue;

        std::istringstream ss( line );
        PTXFileEntryPoint  entryPoint;
        std::getline( ss, entryPoint.filename, ' ' );
        std::getline( ss, entryPoint.functionName, ' ' );
        list.push_back( entryPoint );
    }
    indexFile.close();
    return list;
}

///////////////////////////////////////////////////////////////////////////////
//
// Canonicalization of a few real life PTX samples
//
///////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P( SmokeTest_Catalog,
                          CanonicalizationFromFileSucceeds,
                          ValuesIn( readCatalogIndex( "smoke_test_index.txt" ) ) );

///////////////////////////////////////////////////////////////////////////////
//
// Canonicalization of callable program test cases
//
///////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P( DISABLED_CallableProgram_Catalog,
                          CanonicalizationFromFileSucceeds,
                          ValuesIn( readCatalogIndex( "callable_program_index.txt" ) ) );

///////////////////////////////////////////////////////////////////////////////
//
// Run canonicalization on a corpus of input PTX
//   - Keep this test last in the file because it runs so long (Ctrl-Break to
//     get out of it)
//
///////////////////////////////////////////////////////////////////////////////

INSTANTIATE_TEST_SUITE_P( SLOW_Catalog,
                          CanonicalizationFromFileSucceeds,
                          ValuesIn( readCatalogIndex( "all_ptx_index.txt" ) ) );

//------------------------------
// Test optimization of get/set calls.
//------------------------------

// Override SetUp/GetUp functions so we can control the get/set opt knobs.
class CanonicalizationOptimization : public C14nFixture
{
  public:
    CanonicalizationOptimization()
        : m_getSetOptKnob( "c14n.enableGetSetOpt", true )
        , m_onlyLocalKnob( "c14n.onlyLocalGetSetOpt", false )
    {
    }

  private:
    ScopedKnobSetter m_getSetOptKnob;
    ScopedKnobSetter m_onlyLocalKnob;
};

static int getNumCalls( const llvm::BasicBlock* block, const char* startsWith )
{
    int numCalls = 0;
    for( llvm::BasicBlock::const_iterator I = block->begin(), IE = block->end(); I != IE; ++I )
    {
        const llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>( I );
        if( !call )
            continue;

        const llvm::Function* callee = call->getCalledFunction();
        if( !callee )
            continue;

        const llvm::StringRef& name = callee->getName();

        if( name.startswith( startsWith ) )
            ++numCalls;
    }
    return numCalls;
}

static int getNumCalls( const llvm::Function* function, const char* startsWith )
{
    int numCalls = 0;
    for( llvm::Function::const_iterator BB = function->begin(), BBE = function->end(); BB != BBE; ++BB )
    {
        numCalls += getNumCalls( &*BB, startsWith );
    }
    return numCalls;
}

SAFETEST_F_DEV( CanonicalizationOptimization, TestNothingToOptimize1 )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      %resptr = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @result to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %resptr2 = inttoptr i64 %resptr to i32*
      store i32 13, i32* %resptr2
      ret void
    }
  );
    // clang-format on

    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 1 );
}

SAFETEST_F_DEV( CanonicalizationOptimization, TestNothingToOptimize2 )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      %0 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 1, i64 0, i64 0, i64 0)
      %ptr = inttoptr i64 %0 to i32*
  )
  MAKE_STRING(
      %ld = load i32, i32* %ptr, align 4
  )
  MAKE_STRING (
      %resptr = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @result to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %resptr2 = inttoptr i64 %resptr to i32*
      store i32 %ld, i32* %resptr2
      ret void
    }
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 1 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 1 );
}

// st x, p <- // dead
// st y, p
SAFETEST_F_DEV( CanonicalizationOptimization, TestDeadStore )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      %buf = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr = inttoptr i64 %buf to i32*
      store i32 13, i32* %ptr
      %buf2 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr2 = inttoptr i64 %buf2 to i32*
      store i32 42, i32* %ptr2
      ret void
    }
  );
    // clang-format on

    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 1 );
}

// Same test without get/set opt results in the getBufferElement not being removed.
SAFETEST_F_DEV( CanonicalizationOptimization, TestDeadStoreOptDisabled )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      %buf = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr = inttoptr i64 %buf to i32*
      store i32 13, i32* %ptr
      %buf2 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr2 = inttoptr i64 %buf2 to i32*
      store i32 42, i32* %ptr2
      ret void
    }
  );
    // clang-format on

    ScopedKnobSetter* tmpKnob = new ScopedKnobSetter( "c14n.enableGetSetOpt", false );
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );
    delete tmpKnob;

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 2 );
}

// x = ld p
// st x, p  <- redundant
SAFETEST_F_DEV( CanonicalizationOptimization, TestRedundantStore )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      %buf = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr = inttoptr i64 %buf to i32*
  )
  MAKE_STRING(
      %ld = load i32, i32* %ptr, align 4
  )
  MAKE_STRING (
      %buf2 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr2 = inttoptr i64 %buf2 to i32*
      store i32 %ld, i32* %ptr2
      ret void
    }
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 0 );
    // If the store is removed correctly, LLVM's optimizations will get rid of all
    // other code, too, which results in an empty block.
    EXPECT_TRUE( function->getEntryBlock().getFirstNonPHI()->isTerminator() );
}

// y0 = ld p           y1 = ld p
//       z = phi(y0,y1)
//       st z, p  <- // redundant
// TODO: This test doesn't really work since most of it gets optimized away
//       before get/set optimization.
SAFETEST_F_DEV( CanonicalizationOptimization, TestRedundantStoreCF )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
  )
  MAKE_STRING(
      %val = load i32, i32 addrspace(1)* @index, align 4
  )
  MAKE_STRING (
      %cond = icmp sgt i32 %val, 0
      br i1 %cond, label %if.then, label %if.else

    if.then:
      %buf = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr = inttoptr i64 %buf to i32*
  )
  MAKE_STRING(
      %ld1 = load i32, i32* %ptr, align 4
  )
  MAKE_STRING (
      br label %if.end

    if.else:
      %buf2 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr2 = inttoptr i64 %buf2 to i32*
  )
  MAKE_STRING(
      %ld2 = load i32, i32* %ptr2, align 4
  )
  MAKE_STRING (
      br label %if.end

    if.end:
      %phi = phi i32 [ %ld1, %if.then ], [ %ld2, %if.else ]
      %buf3 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %ptr3 = inttoptr i64 %buf3 to i32*
      store i32 %phi, i32* %ptr3
      ret void
    }
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 0 );
}

SAFETEST_F_DEV( CanonicalizationOptimization, TestSample6MissProgram )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @bg_color = internal addrspace(1) global [12 x i8] zeroinitializer, align 8
    @prd_radiance = internal addrspace(1) global [20 x i8] zeroinitializer, align 8
    @_ZN21rti_internal_register20reg_bitness_detectorE = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail0E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail1E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail2E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail3E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail4E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail5E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail6E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail7E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail8E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register24reg_exception_64_detail9E = addrspace(1) global i64 0, align 8
    @_ZN21rti_internal_register21reg_exception_detail0E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail1E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail2E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail3E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail4E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail5E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail6E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail7E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail8E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register21reg_exception_detail9E = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register14reg_rayIndex_xE = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register14reg_rayIndex_yE = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_register14reg_rayIndex_zE = addrspace(1) global i32 0, align 4
    @_ZN21rti_internal_typeinfo8bg_colorE = addrspace(1) global [8 x i8] c"Ray\00\0C\00\00\00", align 4
    @_ZN21rti_internal_typeinfo12prd_radianceE = addrspace(1) global [8 x i8] c"Ray\00\14\00\00\00", align 4
    @_ZN21rti_internal_typename8bg_colorE = addrspace(1) global [7 x i8] c"float3\00", align 1
    @_ZN21rti_internal_typename12prd_radianceE = addrspace(1) global [20 x i8] c"PerRayData_radiance\00", align 1
    @_ZN21rti_internal_typeenum8bg_colorE = addrspace(1) global i32 4919, align 4
    @_ZN21rti_internal_typeenum12prd_radianceE = addrspace(1) global i32 4919, align 4
    @_ZN21rti_internal_semantic8bg_colorE = addrspace(1) global [1 x i8] zeroinitializer, align 1
    @_ZN21rti_internal_semantic12prd_radianceE = addrspace(1) global [10 x i8] c"rtPayload\00", align 1
    @_ZN23rti_internal_annotation8bg_colorE = addrspace(1) global [1 x i8] zeroinitializer, align 1
    @_ZN23rti_internal_annotation12prd_radianceE = addrspace(1) global [1 x i8] zeroinitializer, align 1
    @llvm.used = appending global [36 x i8*] [i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail0E to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail6E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail7E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register14reg_rayIndex_yE to i8*), i8* addrspacecast ([1 x i8] addrspace(1)* @_ZN23rti_internal_annotation12prd_radianceE to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail2E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail1E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_typeenum8bg_colorE to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail4E to i8*), i8* addrspacecast ([10 x i8] addrspace(1)* @_ZN21rti_internal_semantic12prd_radianceE to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail5E to i8*), i8* addrspacecast ([8 x i8] addrspace(1)* @_ZN21rti_internal_typeinfo12prd_radianceE to i8*), i8* addrspacecast ([8 x i8] addrspace(1)* @_ZN21rti_internal_typeinfo8bg_colorE to i8*), i8* addrspacecast ([12 x i8] addrspace(1)* @bg_color to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail8E to i8*), i8* addrspacecast ([1 x i8] addrspace(1)* @_ZN23rti_internal_annotation8bg_colorE to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail9E to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail1E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail0E to i8*), i8* addrspacecast ([20 x i8] addrspace(1)* @_ZN21rti_internal_typename12prd_radianceE to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail5E to i8*), i8* addrspacecast ([20 x i8] addrspace(1)* @prd_radiance to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail9E to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail8E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register14reg_rayIndex_zE to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail4E to i8*), i8* addrspacecast ([7 x i8] addrspace(1)* @_ZN21rti_internal_typename8bg_colorE to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail6E to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register20reg_bitness_detectorE to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail7E to i8*), i8* addrspacecast ([1 x i8] addrspace(1)* @_ZN21rti_internal_semantic8bg_colorE to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register14reg_rayIndex_xE to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail3E to i8*), i8* addrspacecast (i64 addrspace(1)* @_ZN21rti_internal_register24reg_exception_64_detail3E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_register21reg_exception_detail2E to i8*), i8* addrspacecast (i32 addrspace(1)* @_ZN21rti_internal_typeenum12prd_radianceE to i8*)], section "llvm.metadata"

    define ptx_kernel void @testFunction() #0 {
    Start:
  )
  MAKE_STRING(
      %val.i = load float, float addrspace(1)* bitcast (i8 addrspace(1)* getelementptr inbounds ([12 x i8], [12 x i8] addrspace(1)* @bg_color, i32 0, i64 8) to float addrspace(1)*), align 4
      %val.i1 = load <2 x float>, <2 x float> addrspace(1)* bitcast ([12 x i8] addrspace(1)* @bg_color to <2 x float> addrspace(1)*), align 8
      store <2 x float> %val.i1, <2 x float> addrspace(1)* bitcast ([20 x i8] addrspace(1)* @prd_radiance to <2 x float> addrspace(1)*)
      store float %val.i, float addrspace(1)* bitcast (i8 addrspace(1)* getelementptr inbounds ([20 x i8], [20 x i8] addrspace(1)* @prd_radiance, i32 0, i64 8) to float addrspace(1)*)
  )
  MAKE_STRING (
      ret void
    }

    attributes #0 = { nounwind }

  )
  MAKE_STRING(
    !lwvm.annotations = !{!0}
    !0 = !{void ()* @testFunction, !"kernel", i32 1}
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );

    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_getVariableValue" ), 2 );
    EXPECT_EQ( getNumCalls( function, "optixi_getPayloadValue" ), 0 );
    EXPECT_EQ( getNumCalls( function, "optixi_setPayloadValue" ), 2 );
}

// TODO: This test doesn't really work since all of it gets optimized away
//       before get/set optimization.
SAFETEST_F_DEV( CanonicalizationOptimization, TestLoop )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-p1:64:64:64-p3:32:32:32-p4:32:32:32-p5:32:32:32-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @A = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @index = internal addrspace(1) global i32 0, align 4
    @count = internal addrspace(1) global i32 0, align 4
    @result = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
      //%0 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @A to i64), i32 1, i32 4, i64 %p1.i, i64 0, i64 0, i64 0)
      //%1 = load i32 addrspace(1)* @count, align 4
  )
  MAKE_STRING(
      %val = load i32, i32 addrspace(1)* @index, align 4
  )
  MAKE_STRING (
      %pred = icmp sgt i32 %val, 0
      br i1 %pred, label %header, label %exit

    header:
      %i = phi i32 [ %inc, %latch ], [ 0, %Start ]
      %pred2 = icmp slt i32 %i, 10
      br i1 %pred2, label %body, label %exit

    body:
      br label %header2

    header2:
      %i2 = phi i32 [ %inc2, %latch2 ], [ 0, %body ]
      %pred3 = icmp slt i32 %i2, 10
      br i1 %pred3, label %body2, label %latch

    body2:
      br label %latch2

    latch2:
      %inc2 = add i32 %i2, 1
      br label %header2

    latch:
      %inc = add i32 %i, 1
      br label %header

    exit:
  )
  MAKE_STRING(
      %valx = load i32, i32 addrspace(1)* @index, align 4
  )
  MAKE_STRING (
      %resptr = tail call i64 @_rt_buffer_get_64(i64 ptrtoint ([1 x i8] addrspace(1)* @result to i64), i32 1, i32 4, i64 0, i64 0, i64 0, i64 0)
      %resptr2 = inttoptr i64 %resptr to i32*
      store i32 %valx, i32* %resptr2
      ret void
    }

    declare float @optix.ptx.add.f32.ftz(float, float)

  )
  MAKE_STRING(
    !lwvm.annotations = !{!0}
    !0 = !{void ()* @testFunction, !"kernel", i32 1}
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );
}

// This test passes when we enable the optimization of redundant insertvalue instructions.
//
// The testFunction function generates a piece of code like the following:
// %from_buffer = @optixi_getBufferElement(...)
// %tmp_value = insertvalue [3 x i32] [i32 123, i32 456, %i32 undef], %i32 from_buffer
// %to_store = insertvalue [3 x i32] %tmp_value, %i32 789
//
// (The first insertvalue instruction is dead because the second one overwrites its result).
// The output of getBufferElement is used by a dead insertvalue instruction and it should be optimized away.
// The optimization the removes the redudant insertvalue instruction has been originally introduced in LLVM in http://reviews.llvm.org/rL208214

SAFETEST_F_DEV( CanonicalizationOptimization, TestRedundantInsertElementEliminationOptimization )
{
    // clang-format off
  const std::string& code = MAKE_STRING
  (
    target datalayout = "e-p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-v16:16:16-v32:32:32-v96:128:128-n16:32:64"
    target triple = "lwptx64-lwpu-lwca"

    @output = internal addrspace(1) global [1 x i8] zeroinitializer, align 1
    @bufferIndex = internal addrspace(1) global i32 0, align 4

    declare i64 @_rt_buffer_get_64(i64, i32, i32, i64, i64, i64, i64)

    define ptx_kernel void @testFunction() {
    Start:
  )
  MAKE_STRING(
      %val.i = load i32, i32 addrspace(1)* @bufferIndex, align 4
  )
  MAKE_STRING (
      %0 = sext i32 %val.i to i64
      %1 = tail call i64 @_rt_buffer_get_64(i64 ptrtoint (i8* addrspacecast ([1 x i8] addrspace(1)* @output to i8*) to i64), i32 1, i32 12, i64 %0, i64 0, i64 0, i64 0)
      %2 = add i64 %1, 4
      %3 = inttoptr i64 %2 to i32*
      store i32 1074580685, i32* %3
      %4 = inttoptr i64 %1 to i32*
      store i32 1066192077, i32* %4
      %5 = add i64 %1, 8
      %6 = inttoptr i64 %5 to i32*
      store i32 1079194419, i32* %6
      ret void
    }

  )
  MAKE_STRING(
    !lwvm.annotations = !{!0}
    !0 = !{void ()* @testFunction, !"kernel", i32 1}
  );
    // clang-format on
    ASSERT_NO_THROW_WITH_MESSAGE( canonicalizeLLVMTestFunction( code ) );
    const llvm::Function* function = m_canonicalProgram->llvmFunction();
    EXPECT_EQ( getNumCalls( function, "optixi_setBufferElement" ), 3 );
    EXPECT_EQ( getNumCalls( function, "optixi_getBufferElement" ), 0 );
}
