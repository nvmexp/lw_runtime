#pragma once

#include <srcTests.h>

#include <llvm/Support/CrashRecoveryContext.h>

#include <iostream>


#ifdef ENABLE_DBGMSG
#define DBGMSG( msg ) std::cerr << msg << std::endl
#else
#define DBGMSG( msg )
#endif


//-----------------------------------------------------------------------------
class CrashHandlerElwironment : public ::testing::Environment
{
  public:
    virtual void SetUp()
    {
        // Disable for now
        // llvm::CrashRecoveryContext::Enable();
    }

    virtual void TearDown() { llvm::CrashRecoveryContext::Disable(); }

    static bool init()
    {
        static ::testing::Environment* theCrashElw = nullptr;
        if( !theCrashElw )
        {
            theCrashElw = ::testing::AddGlobalTestElwironment( new CrashHandlerElwironment );
        }
        return true;
    }

    static bool m_init_dummy;
};

bool CrashHandlerElwironment::m_init_dummy = CrashHandlerElwironment::init();

static bool skipFail = false;

//-----------------------------------------------------------------------------
class SafeLLVMFixture : public ::testing::Test
{
  protected:
    virtual void SafeSetUp() { DBGMSG( "SafeSetup()" ); }
    virtual void SafeTearDown() { DBGMSG( "SafeTearDown()" ); }
    virtual void SafeTestBody() { DBGMSG( "SafeTestBody()" ); }

    virtual void SetUp() { RunSafely( &SafeLLVMFixture::SafeSetUp ); }

    virtual void TearDown() { RunSafely( &SafeLLVMFixture::SafeTearDown ); }

    virtual void TestBody() { RunSafely( &SafeLLVMFixture::SafeTestBody ); }


    template <class T>
    struct MemberFunc
    {
        T* obj;
        void ( T::*func )();

        MemberFunc( T* obj, void ( T::*func )() )
            : obj( obj )
            , func( func )
        {
        }

        void ilwoke() { ( obj->*func )(); }

        static void ilwokeFromVoidPtr( void* userData ) { ( (MemberFunc<T>*)userData )->ilwoke(); }
    };

    void RunSafely( void ( SafeLLVMFixture::*func )() )
    {
        MemberFunc<SafeLLVMFixture> memberFunc( this, func );

        DBGMSG( "\nON=========" );
        llvm::CrashRecoveryContext CRC;
        bool                       success = CRC.RunSafely( memberFunc.ilwokeFromVoidPtr, (void*)&memberFunc );
        DBGMSG( "OFF==========\n" );
        if( !success )
        {
            if( skipFail )
                skipFail = true;
            else
            {
                FAIL();
            }
        }
    }
};

// MakeAndRegisterTestInfo() has now an additional CodeLocation() parameter
#define GTEST_SAFETEST_( test_case_name, test_name, parent_class, parent_id )                                                    \
                                                                                                                                 \
    class GTEST_TEST_CLASS_NAME_( test_case_name, test_name )                                                                    \
        : public parent_class                                                                                                    \
    {                                                                                                                            \
      public:                                                                                                                    \
        GTEST_TEST_CLASS_NAME_( test_case_name, test_name )() {}                                                                 \
      private:                                                                                                                   \
        virtual void                      SafeTestBody();                                                                        \
        static ::testing::TestInfo* const test_info_ GTEST_ATTRIBUTE_UNUSED_;                                                    \
        GTEST_DISALLOW_COPY_AND_ASSIGN_( GTEST_TEST_CLASS_NAME_( test_case_name, test_name ) );                                  \
    };                                                                                                                           \
                                                                                                                                 \
                                                                                                                                 \
    ::testing::TestInfo* const GTEST_TEST_CLASS_NAME_( test_case_name, test_name )::test_info_ =                                 \
        ::testing::internal::MakeAndRegisterTestInfo(                                                                            \
            #test_case_name, #test_name, NULL, NULL, ::testing::internal::CodeLocation(__FILE__, __LINE__),	                     \
			( parent_id ), parent_class::SetUpTestCase, parent_class::TearDownTestCase,                                          \
            new ::testing::internal::TestFactoryImpl<GTEST_TEST_CLASS_NAME_( test_case_name, test_name )> );                     \
                                                                                                                                 \
    void GTEST_TEST_CLASS_NAME_( test_case_name, test_name )::SafeTestBody()


#define SAFETEST( test_case_name, test_name )                                                                          \
    GTEST_SAFETEST_( test_case_name, test_name, SafeLLVMFixture, ::testing::internal::GetTestTypeId() )


#define SAFETEST_F( test_fixture, test_name )                                                                          \
    GTEST_SAFETEST_( test_fixture, test_name, test_fixture, ::testing::internal::GetTypeId<test_fixture>() )


#define SAFETEST_P( test_case_name, test_name )                                                                                      \
    class GTEST_TEST_CLASS_NAME_( test_case_name, test_name )                                                                        \
        : public test_case_name                                                                                                      \
    {                                                                                                                                \
      public:                                                                                                                        \
        GTEST_TEST_CLASS_NAME_( test_case_name, test_name )() {}                                                                     \
        virtual void SafeTestBody();                                                                                                 \
                                                                                                                                     \
      private:                                                                                                                       \
        static int AddToRegistry()                                                                                                   \
        {                                                                                                                            \
            ::testing::UnitTest::GetInstance()                                                                                       \
                ->parameterized_test_registry()                                                                                      \
                .GetTestCasePatternHolder<test_case_name>( #test_case_name, ::testing::internal::CodeLocation{__FILE__, __LINE__} )                                     \
                ->AddTestPattern( #test_case_name, #test_name,                                                                       \
                                  new ::testing::internal::TestMetaFactory<GTEST_TEST_CLASS_NAME_( test_case_name, test_name )>() ); \
            return 0;                                                                                                                \
        }                                                                                                                            \
        static int gtest_registering_dummy_;                                                                                         \
        GTEST_DISALLOW_COPY_AND_ASSIGN_( GTEST_TEST_CLASS_NAME_( test_case_name, test_name ) );                                      \
    };                                                                                                                               \
    int GTEST_TEST_CLASS_NAME_( test_case_name, test_name )::gtest_registering_dummy_ =                                              \
        GTEST_TEST_CLASS_NAME_( test_case_name, test_name )::AddToRegistry();                                                        \
    void GTEST_TEST_CLASS_NAME_( test_case_name, test_name )::SafeTestBody()


#if defined WIN32 && 0  // TODO: disabled for now
SAFETEST( WindowsHackToClearFirstException, ShouldFail )
{
    skipFail = true;
    try
    {
        throw "Some exception";
    }
    catch( ... )
    {
        std::cerr << "This should print, but doesn't because control goes to the CrashRecoveryContext instead.\n";
    }
    skipFail = false;
}
#endif
