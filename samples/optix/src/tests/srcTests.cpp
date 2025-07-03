#include <srcTests.h>

// clang-format off
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include<windows.h>
#  include<mmsystem.h>
#else /*Apple and Linux both use this */
#    include<sys/time.h>
#    include <dirent.h>
#endif
// clang-format on

#include <prodlib/system/Knobs.h>

static int dirExists( const char* path )
{
#if defined( _WIN32 )
    DWORD attrib = GetFileAttributes( path );
    return ( attrib != ILWALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == nullptr )
        return 0;
    else
    {
        closedir( dir );
        return 1;
    }
#endif
}

const std::string ptxPath( const std::string& target, const std::string& base )
{
    const char*        dir;
    static std::string path;

    // Allow for overrides.
    dir = getelw( "OPTIX_TEST_PTX_DIR" );
    if( dir )
    {
        path = dir;
    }
    else if( dirExists( SRC_TESTS_PTX_DIR ) )
    {
        // Return hardcoded path if it exists.
        path = std::string( SRC_TESTS_PTX_DIR );
    }
    else
    {
        // Last resort.
        path = ".";
    }

    path += "/" + target + "_generated_" + base + ".ptx";
    return path;
}

const std::string dataPath()
{
    // Allow for overrides.
    const char* dir = getelw( "OPTIX_TEST_DATA_DIR" );
    if( dir )
        return dir;

    // Return hardcoded path if it exists.
    if( dirExists( SRC_TESTS_DATA_DIR ) )
        return SRC_TESTS_DATA_DIR;

    // Last resort.
    return ".";
}

// Wrap in a function so we don't have to worry about initialization order
static std::vector<SrcTestFilter::Callback>& getDynamicFilterCallbacks()
{
    static std::vector<SrcTestFilter::Callback> g_dynamicFilterCallbacks;
    return g_dynamicFilterCallbacks;
}

void SrcTestFilter::add( SrcTestFilter::Callback callback )
{
    std::vector<SrcTestFilter::Callback>& dynamicFilterCallbacks = getDynamicFilterCallbacks();
    dynamicFilterCallbacks.push_back( callback );
}


static void appendWithSep( std::string& str, const std::string& toAppend, const std::string& sep )
{
    if( !str.empty() )
        str += sep;
    str += toAppend;
}

static std::string getDynamicFilters()
{
    std::string filterStr;
    for( SrcTestFilter::Callback callback : getDynamicFilterCallbacks() )
    {
        std::vector<std::string> filters = callback();
        for( const std::string& f : filters )
            appendWithSep( filterStr, f, ":" );
    }
    return filterStr;
}

int main( int argc, char** argv )
{
    ::testing::InitGoogleMock( &argc, argv );

    // Add dynamic filters
    std::string dynamicFilters = getDynamicFilters();
    if( !dynamicFilters.empty() )
    {
        std::string filter = ::testing::GTEST_FLAG( filter );
        if( filter.find( "-" ) == filter.npos )     // already has negative filter section?
            dynamicFilters = "-" + dynamicFilters;  // no, start negative filter section
        appendWithSep( filter, dynamicFilters, ":" );
        ::testing::GTEST_FLAG( filter ) = filter;
    }

    // White-box unit tests do not use a wrapper (O6) nor do they query the function table (O7).
    // Therefore, the knobs need to get initialized explicitly.
    knobRegistry().initializeKnobs();

    return RUN_ALL_TESTS();
}
