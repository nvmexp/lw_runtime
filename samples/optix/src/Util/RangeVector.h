#include <iosfwd>
#include <vector>

namespace optix {

struct Range
{
    size_t begin;
    size_t end;  // exclusive
};

inline bool operator==( const Range& a, const Range& b )
{
    return a.begin == b.begin && a.end == b.end;
}

typedef std::vector<Range> RangeVector;

std::ostream& operator<<( std::ostream& out, const RangeVector& rv );
std::istream& operator>>( std::istream& in, RangeVector& rv );

}  // namespace optix
