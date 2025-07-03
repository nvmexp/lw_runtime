#include <Util/RangeVector.h>

#include <corelib/misc/String.h>

#include <iostream>

using namespace corelib;

std::ostream& optix::operator<<( std::ostream& out, const RangeVector& ranges )
{
    bool first = true;
    for( const Range& range : ranges )
    {
        if( range.begin == range.end )
            continue;
        if( !first )
            out << ",";
        if( range.end - range.begin == 1 )
            out << range.begin;
        else if( range.end == SIZE_MAX )
            out << range.begin << "-";
        else
            out << range.begin << "-" << range.end - 1;
        first = false;
    }

    return out;
}

std::istream& optix::operator>>( std::istream& in, RangeVector& ranges )
{
    std::string strVal;
    std::getline( in, strVal );
    const std::vector<std::string> rangeStrings = tokenize( strVal, ", " );
    for( const std::string& rangeStr : rangeStrings )
    {
        const std::vector<std::string> valStrings = tokenize( rangeStr, "- " );
        Range                          range;
        if( !valStrings.empty() )
        {
            const size_t val0 = std::stoul( valStrings[0] );
            if( valStrings.size() == 1 )
            {
                if( rangeStr.front() == '-' )
                {
                    range.begin = 0;
                    range.end   = val0 + 1;
                }
                else if( rangeStr.back() == '-' )
                {
                    range.begin = val0;
                    range.end   = SIZE_MAX;
                }
                else
                {
                    range.begin = val0;
                    range.end   = val0 + 1;
                }
            }
            else if( valStrings.size() == 2 )
            {
                const size_t val1 = std::stoul( valStrings[1] );
                range.begin       = val0;
                range.end         = val1 + 1;
            }
            else
            {
                in.setstate( std::ios::failbit );
                continue;
            }

            // error checking
            if( ( range.end <= range.begin ) ||                           // range endpoints out of order
                ( !ranges.empty() && range.begin < ranges.back().end ) )  // ranges must be in increasing order
            {
                in.setstate( std::ios::failbit );
            }

            ranges.push_back( range );
        }
    }
    return in;
}
