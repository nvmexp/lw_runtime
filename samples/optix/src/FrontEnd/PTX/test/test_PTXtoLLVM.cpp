
#include <srcTests.h>

#include <FrontEnd/PTX/DataLayout.h>
#include <FrontEnd/PTX/PTXHeader.h>
#include <FrontEnd/PTX/PTXNamespaceMangle.h>
#include <FrontEnd/PTX/PTXtoLLVM.h>

#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_os_ostream.h>

#include <fstream>
#include <sstream>

using namespace optix;
using namespace testing;


struct PTXFragment
{
    const char* description;
    const char* code;
};

::std::ostream& operator<<(::std::ostream& os, const PTXFragment& cf )
{
    return os << cf.description;
}

#define PTX_MODULE( desc, ... )                                                                                        \
    {                                                                                                                  \
        desc, #__VA_ARGS__                                                                                             \
    }


class TranslateFromString : public TestWithParam<PTXFragment>
{
  protected:
    void SetUp() override
    {
        // Create LLVM state
        llvmContext = new llvm::LLVMContext();
        stateType   = llvm::StructType::create( *llvmContext, "struct.cort::CanonicalState" );
    }
    void TearDown() override
    {
        delete llvmContext;
        llvmContext = nullptr;
        stateType   = nullptr;
    }
    llvm::LLVMContext* llvmContext;
    llvm::StructType*  stateType;
    llvm::Function*    function;
};

TEST_P( TranslateFromString, Test )
{
    // Parse to LLVM
    llvm::DataLayout dataLayout = createDataLayoutForLwrrentProcess();
    PTXtoLLVM        frontend( *llvmContext, &dataLayout );
    std::string      ptxStr( GetParam().code );
    std::string      declarations = createPTXHeaderString( {ptxStr.c_str(), ptxStr.size()} );
    llvm::Module*    llvmModule   = nullptr;
    try
    {
        // Use try/catch instead of EXPECT_NO_THROW so we can see the message
        llvmModule = frontend.translate( "unitTest", declarations, {{ptxStr.c_str(), ptxStr.size()}}, /*parseLineNumbers=*/true  );
    }
    catch( const std::exception& e )
    {
        ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );
    }

    ASSERT_TRUE( llvmModule != nullptr );
    std::string     functionName = "testFunction";
    llvm::Function* function     = llvmModule->getFunction( functionName );
    ASSERT_TRUE( function != nullptr );
}

/*
* Canonicalize variables
*/
static PTXFragment simpleInput[] = {

    // clang-format off
    PTX_MODULE( "device function with no parameters",
      .version 1.4
      .target sm_10
      .visible .func deviceFunction()
      {
      }

      .entry testFunction()
      {
        call (), deviceFunction, ();
      }
    ),

    PTX_MODULE( "callable program arguments",
      .version 3.2
      .target sm_30
      .visible .func  (.param .b32 func_retval0) deviceFunction (.param .b32 _Z13callable_progi_param_0)
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
          call.uni (retval0), deviceFunction, (param0);
          ld.param.b32  %r3, [retval0+0];
        }
        ret;
      }
    ),

    PTX_MODULE( "device function with parameters (also relwrsive)",
      .version 3.2
      .target sm_30
      .visible .func deviceFunction(.reg .u32 %r1)
      {
        st.global.u32 [%r1], %r1;
        call (), deviceFunction, (%r1);
      }
      .entry testFunction()
      {
        .reg .u32 %r<10>;
        call (), deviceFunction, (%r1);
      }
    ),

    PTX_MODULE( "device function with return value",  
      .version 1.4
      .target sm_10
      .visible .func (.reg .u32 retval) deviceFunction()
      {
        mov.u32 retval, 99;
      }

      .entry testFunction()
      {
        .reg .u32 r0;
        call (r0), deviceFunction, ();
      }
    ),

    PTX_MODULE( "device function with ignored return value",  
      .version 1.4
      .target sm_10
      .visible .func (.reg .u32 retval) deviceFunction()
      {
        mov.u32 retval, 99;
      }

      .entry testFunction()
      {
        call (_), deviceFunction, ();
      }
    ),
    // clang-format on

};


INSTANTIATE_TEST_SUITE_P( SimpleInput, TranslateFromString, ValuesIn( simpleInput ) );

struct PTXFragmentVectorWithErrorMessage
{
    std::string              err;  // either empty for no error, or exact error message for exception
    std::vector<PTXFragment> fragments;
};

::std::ostream& operator<<(::std::ostream& os, const PTXFragmentVectorWithErrorMessage& fv )
{
    os << "{";
    os << ( fv.err.empty() ? "(no " : "(" ) << "error expected)";
    for( const PTXFragment& fragment : fv.fragments )
    {
        os << ", " << fragment;
    }
    os << "}";
    return os;
}

class TranslateFromMultipleStrings : public TestWithParam<PTXFragmentVectorWithErrorMessage>
{
  protected:
    void SetUp() override
    {
        // Create LLVM state
        llvmContext = new llvm::LLVMContext();
        stateType   = llvm::StructType::create( *llvmContext, "struct.cort::CanonicalState" );
    }
    void TearDown() override
    {
        delete llvmContext;
        llvmContext = nullptr;
        stateType   = nullptr;
    }
    llvm::LLVMContext* llvmContext;
    llvm::StructType*  stateType;
    llvm::Function*    function;
};

static PTXFragmentVectorWithErrorMessage multipleStringInput[] = {
    // clang-format off

    // callable program in separate string from entry
    {
        std::string(""),
        {
            PTX_MODULE( "entry point and forward decls",
              .version 3.2
              .target sm_30
              .visible .func  (.param .b32 func_retval0) deviceFunction (.param .b32 _Z13callable_progi_param_0);

              .visible .global .align 4 .b8 g_var[12];  // shared global variable
              
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
                  call.uni (retval0), deviceFunction, (param0);
                  ld.param.b32  %r3, [retval0+0];
                }
                ret;
              }
            ),

            PTX_MODULE( "callable program with arguments",
              .version 3.2
              .target sm_30

              .extern .global .align 4 .b8 g_var[12];

              .visible .func  (.param .b32 func_retval0) deviceFunction (.param .b32 _Z13callable_progi_param_0)
              {
                .reg .s32 %r<2>;
                mov.u32 %r1, 0;
                st.param.b32  [func_retval0+0], %r1;
                ret;
              }
            )
        }
    },

    // multiple device functions in separate strings
    {
        std::string(""),
        {
            PTX_MODULE( "device function with parameters (also relwrsive)",
              .version 3.2
              .target sm_30
              .visible .func deviceFunction1(.reg .u32 %r1)
              {
                st.global.u32 [%r1], %r1;
                call (), deviceFunction1, (%r1);
              }
            ),

            PTX_MODULE( "device function with return value",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .u32 retval) deviceFunction2()
              {
                mov.u32 retval, 99;
              }
            ),

            PTX_MODULE( "device function with ignored return value",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .u32 retval) deviceFunction3()
              {
                mov.u32 retval, 99;
              }
            ),

            PTX_MODULE( "entry point and forward decls",  
              .version 1.4
              .target sm_10
              .visible .func deviceFunction1(.reg .u32 %r1);
              .visible .func (.reg .u32 retval) deviceFunction2();
              .visible .func (.reg .u32 retval) deviceFunction3();

              .entry testFunction()
              {
                .reg .u32 %r<10>;
                call (), deviceFunction1, (%r1);
                call (%r2), deviceFunction2, ();
                call (_), deviceFunction3, ();
              }
            )

        }
    },

    // Error cases
    {
        std::string("Undefined symbol: deviceFunction"),
        {
            PTX_MODULE( "device function with return value",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .u32 retval) deviceFunctionXX()
              {
                mov.u32 retval, 99;
              }
            ),
            PTX_MODULE( "entry point and forward decl with no definition",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .f32 retval) deviceFunction();

              .entry testFunction()
              {
                .reg .f32 r0;
                call (r0), deviceFunction, ();
              }
            )
        }
    },
    {
        std::string("multiple declarations that do not have identical return types"),
        {
            PTX_MODULE( "device function with return value",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .u32 retval) deviceFunction()
              {
                mov.u32 retval, 99;
              }
            ),
            PTX_MODULE( "entry point and mismatched forward decl",  
              .version 1.4
              .target sm_10
              .visible .func (.reg .f32 retval) deviceFunction();

              .entry testFunction()
              {
                .reg .f32 r0;
                call (r0), deviceFunction, ();
              }
            )
        }
    },
    {
        std::string("multiple declarations that do not have identical types"),
        {
            PTX_MODULE( "device function with return value",  
              .version 1.4
              .target sm_10

              .extern .global .align 4 .b8 g_var[12];
            ),
            PTX_MODULE( "entry point and forward decl with no definition",  
              .version 1.4
              .target sm_10

              .visible .global .align 4 .b8 g_var[16];
            )
        }
    },

    // clang-format on
};

TEST_P( TranslateFromMultipleStrings, Test )
{
    // Parse to LLVM
    llvm::DataLayout                         dataLayout = createDataLayoutForLwrrentProcess();
    PTXtoLLVM                                frontend( *llvmContext, &dataLayout );
    const PTXFragmentVectorWithErrorMessage& fv( GetParam() );
    std::vector<prodlib::StringView>         ptxStrings;
    for( const PTXFragment& fragment : fv.fragments )
    {
        ptxStrings.push_back( {fragment.code, strlen( fragment.code )} );
    }
    std::string   declarations = createPTXHeaderString( ptxStrings );
    llvm::Module* llvmModule   = nullptr;
    try
    {
        // Use try/catch instead of EXPECT_NO_THROW so we can see the message
        llvmModule = frontend.translate( "unitTest", declarations, ptxStrings, /*parseLineNumbers=*/true );
    }
    catch( const std::exception& e )
    {
        if( fv.err.empty() )
        {
            ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );
        }
        else
        {
            EXPECT_THAT( e.what(), HasSubstr( fv.err ) );
        }
    }

    if( fv.err.empty() )
    {
        ASSERT_TRUE( llvmModule != nullptr );
        std::string     functionName = "testFunction";
        llvm::Function* function     = llvmModule->getFunction( functionName );
        ASSERT_TRUE( function != nullptr );
    }
    else
    {
        ASSERT_TRUE( llvmModule == nullptr );
    }
}


INSTANTIATE_TEST_SUITE_P( MultipleStringInput, TranslateFromMultipleStrings, ValuesIn( multipleStringInput ) );


typedef std::vector<std::pair<std::string, std::vector<std::string>>> vectype;

static vectype readIndex()
{
    vectype      list;
    unsigned int lwr = 0;
    std::map<std::string, unsigned int> unique_names_index;
    std::string   filename = dataPath() + "/ptxInputs/all_ptx_index.txt";
    std::ifstream indexFile( filename.c_str() );
    if( !indexFile.good() )
        return list;
    std::string line;
    std::string lastfile;
    while( std::getline( indexFile, line ) )
    {
        if( line.substr( 0, 2 ) == "//" )
            continue;

        std::istringstream ss( line );
        std::string        ptxfile, ptxfunc;
        std::getline( ss, ptxfile, ' ' );
        std::getline( ss, ptxfunc, ' ' );

        std::map<std::string, unsigned int>::iterator it = unique_names_index.find( ptxfile );
        if( it == unique_names_index.end() )
        {
            list.push_back( std::pair<const std::string, std::vector<std::string>>( ptxfile, std::vector<std::string>( 1, ptxfunc ) ) );
            unique_names_index.insert( std::pair<std::string, unsigned int>( ptxfile, lwr++ ) );
        }
        else
        {
            list[it->second].second.push_back( ptxfunc );
        }
    }
    indexFile.close();
    return list;
}

class TranslateFromFile : public TestWithParam<vectype::value_type>
{
  protected:
    void SetUp() override
    {
        // Create LLVM state
        llvmContext = new llvm::LLVMContext();
        stateType   = llvm::StructType::create( *llvmContext, "struct.cort::CanonicalState" );
    }
    void TearDown() override
    {
        delete llvmContext;
        llvmContext = nullptr;
        stateType   = nullptr;
    }
    llvm::LLVMContext* llvmContext;
    llvm::StructType*  stateType;
    llvm::Function*    function;
};

TEST_P( TranslateFromFile, Test )
{
    // Read PTX
    std::string   filename = dataPath() + "/ptxInputs/" + GetParam().first;
    std::ifstream indexFile( filename.c_str(), std::ios_base::in );
    ASSERT_TRUE( indexFile.good() );
    std::string ptxStr;
    ptxStr.assign( ( std::istreambuf_iterator<char>( indexFile ) ), std::istreambuf_iterator<char>() );

    // Parse to LLVM
    llvm::DataLayout dataLayout = createDataLayoutForLwrrentProcess();
    PTXtoLLVM        frontend( *llvmContext, &dataLayout );
    std::string      declarations = createPTXHeaderString( {ptxStr.c_str(), ptxStr.size()} );
    llvm::Module*    llvmModule   = nullptr;
    try
    {
        // Use try/catch instead of EXPECT_NO_THROW so we can see the message
        llvmModule = frontend.translate( "unitTest", declarations, {{ptxStr.c_str(), ptxStr.size()}}, /*parseLineNumbers=*/true );
    }
    catch( const std::exception& e )
    {
        ADD_FAILURE_WITH_MESSAGE( std::string( "Exception thrown: " ) + e.what() );
    }

    ASSERT_TRUE( llvmModule != nullptr );
    for( const std::string& functionName : GetParam().second )
    {
        llvm::Function* function = llvmModule->getFunction( functionName );
        if( !function )
        {
            // Try a mangled name, but we need to ignore the call signature, so only check the prefix.
            const std::string mangled = PTXNamespaceMangle( functionName, true, true, "" );
            for( auto&& I : *llvmModule )
            {
                // Compare the mangled prefix name to the function name
                if( I.getName().startswith( mangled ) )
                {
                    function = &I;
                    break;
                }
            }
        }
        ASSERT_TRUE( function != nullptr );
    }
}

INSTANTIATE_TEST_SUITE_P( SLOW_PtxInputCatalog, TranslateFromFile, ValuesIn( readIndex() ) );
