#pragma once

#include <gmock/gmock.h>

#include <srcTestConfig.h>

#define FAIL_WITH_MESSAGE( msg )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        FAIL();                                                                                                        \
    } while( 0 )

#define ADD_FAILURE_WITH_MESSAGE( msg )                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        std::cerr << msg << std::endl;                                                                                 \
        ADD_FAILURE();                                                                                                 \
    } while( 0 )

#if defined( DEBUG ) || defined( DEVELOP )
#define TEST_F_DEV( c, n ) TEST_F( c, n )
#define TEST_P_DEV( c, n ) TEST_P( c, n )
#define SAFETEST_F_DEV( c, n ) SAFETEST_F( c, n )
#define SAFETEST_P_DEV( c, n ) SAFETEST_P( c, n )
#else
#define TEST_F_DEV( c, n ) TEST_F( c, DISABLED_##n )
#define TEST_P_DEV( c, n ) TEST_P( c, DISABLED_##n )
#define SAFETEST_F_DEV( c, n ) SAFETEST_F( c, DISABLED_##n )
#define SAFETEST_P_DEV( c, n ) SAFETEST_P( c, DISABLED_##n )
#endif


const std::string ptxPath( const std::string& target, const std::string& base );
const std::string dataPath();


// Colwenience class to dynamically add filter strings to the gtest_filter string. This
// is useful, for example, to remove multi GPU tests on a single GPU machine. We
// use a callback so that the filters can be generated after static initialization
// (mostly so that knobs have been initialized).


class SrcTestFilter
{
  public:
    typedef std::vector<std::string> ( *Callback )();


    SrcTestFilter() {}
    SrcTestFilter( Callback callback ) { add( callback ); }

  protected:
    static void add( Callback callback );
};
