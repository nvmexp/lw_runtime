#include <srcTests.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <src/Util/RangeVector.h>

using namespace optix;
using namespace corelib;
using namespace prodlib;
using namespace ::testing;

// Throws on error (an exercises from_string() which is used for knob parsing
static RangeVector parse( const std::string& rangeVectorStr )
{
    bool        ok;
    RangeVector rv = corelib::from_string<RangeVector>( rangeVectorStr, &ok );
    if( !ok )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Parse failed" );
    return rv;
}

TEST( RangeVectorParse, WorksWithSingleNumber )
{
    RangeVector rv       = parse( "4" );
    RangeVector expected = {{4, 4 + 1}};

    EXPECT_THAT( rv, Eq( expected ) );
}

TEST( RangeVectorParse, WorksWithSingleRange )
{
    RangeVector rv       = parse( "4-10" );
    RangeVector expected = {{4, 10 + 1}};

    EXPECT_THAT( rv, Eq( expected ) );
}

TEST( RangeVectorParse, WorksWithOpenRanges )
{
    RangeVector rv       = parse( "-5, 10-" );
    RangeVector expected = {{0, 5 + 1}, {10, SIZE_MAX}};

    EXPECT_THAT( rv, Eq( expected ) );
}

TEST( RangeVectorParse, WorksWithMixedRanges )
{
    RangeVector rv       = parse( "-5, 7, 9-12, 15-" );
    RangeVector expected = {{0, 5 + 1}, {7, 7 + 1}, {9, 12 + 1}, {15, SIZE_MAX}};

    EXPECT_THAT( rv, Eq( expected ) );
}

TEST( RangeVectorParse, FailsWithNonNumericInput )
{
    EXPECT_ANY_THROW( parse( "asdf" ) );
}

TEST( RangeVectorParse, FailsWithReversedRangeEndPoints )
{
    EXPECT_ANY_THROW( parse( "6-4" ) );
}

TEST( RangeVectorParse, FailsWithUnorderedRanges )
{
    EXPECT_ANY_THROW( parse( "2,1" ) );
    EXPECT_ANY_THROW( parse( "5-,10" ) );
    EXPECT_ANY_THROW( parse( "3,-10" ) );
}

TEST( RangeVector, StreamOutputWorks )
{
    std::istringstream in( "-5, 7, 9-12, 15-" );
    std::ostringstream out;
    RangeVector        rv;

    in >> rv;
    out << rv;

    EXPECT_THAT( out.str(), Eq( "0-5,7,9-12,15-" ) );
}
