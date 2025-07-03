// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/ResampleMotion.h>

#include <optixu/optixu_aabb_namespace.h>
#include <optixu/optixu_math_namespace.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include <prodlib/exceptions/Assert.h>


// Intermediate expressions for round{up|down}_line

float roundup( float x )
{
    return std::nextafter( x, std::numeric_limits<float>::max() );
}

float rounddown( float x )
{
    return std::nextafter( x, -std::numeric_limits<float>::max() );
}

float roundup_expr1( float yA, float xB, float x0 )
{
    float result = 0.0f;
    if( yA >= 0.0f )
    {
        result = roundup( roundup( xB - x0 ) * yA );
    }
    else
    {
        result = roundup( rounddown( xB - x0 ) * yA );
    }
    return result;
}

float rounddown_expr1( float yA, float xB, float x0 )
{
    float result = 0.0f;
    if( yA >= 0.0f )
    {
        result = rounddown( rounddown( xB - x0 ) * yA );
    }
    else
    {
        result = rounddown( roundup( xB - x0 ) * yA );
    }
    return result;
}

float roundup_expr2( float J, float xB, float xA )
{
    float result = 0.0f;
    if( J >= 0 )
    {
        result = roundup( roundup( J ) / rounddown( xB - xA ) );
    }
    else
    {
        result = roundup( roundup( J ) / roundup( xB - xA ) );
    }
    return result;
}

float rounddown_expr2( float J, float xB, float xA )
{
    float result = 0.0f;
    if( J >= 0 )
    {
        result = rounddown( rounddown( J ) / roundup( xB - xA ) );
    }
    else
    {
        result = rounddown( rounddown( J ) / rounddown( xB - xA ) );
    }
    return result;
}

// Project a line defined by two points (xA, yA) and (xB, yB) to a new point x0
// and find the y value y0.
//
// Round UP all intermediate callwlations using float32 math so that y0 is
// guaranteed >= its true value with infinite precision math.
//
float roundup_line( float xA, float xB, float yA, float yB, float x0 )
{
    const float J      = roundup( roundup_expr1( yA, xB, x0 ) + roundup_expr1( yB, x0, xA ) );
    const float result = roundup_expr2( J, xB, xA );
    return result;
}

// Project a line defined by two points (xA, yA) and (xB, yB) to a new point x0
// and find the y value y0.
//
// Round DOWN all intermediate callwlations using float32 math so that y0 is
// guaranteed <= its true value with infinite precision math.
//
float rounddown_line( float xA, float xB, float yA, float yB, float x0 )
{
    const float J      = rounddown( rounddown_expr1( yA, xB, x0 ) + rounddown_expr1( yB, x0, xA ) );
    const float result = rounddown_expr2( J, xB, xA );
    return result;
}

// version with no rounding
float evaluate_line( float xA, float xB, float yA, float yB, float x0 )
{
    const float J      = yA * ( xB - x0 ) + yB * ( x0 - xA );
    const float result = J / ( xB - xA );
    return result;
}

bool compare_points_by_x( const optix::float2& a, const optix::float2& b )
{
    return a.x < b.x;
}


// Given a set of points sorted by x, find two points in the set that form a bounding
// line in y.
// Template args, so we can handle both upper and lower bounds in the same function:
//   compare(A,B)  : returns true if A "bounds" B, e.g., is >= for upper or <= for lower
//   eval_line(...): evaluate a line using up/down rounding.
//
// Returns the indices of the two found points.
//
template <typename C, typename F>
void find_points_on_bounding_line( const std::vector<optix::float2>& points,
                                   C                                 compare,
                                   F                                 eval_line,
                                   //out
                                   int& left_index,
                                   int& right_index )
{
    const int n( points.size() );
    RT_ASSERT( n > 1 );

    // start at interval containing x midpoint.
    {
        const optix::float2 pmid = optix::make_float2( 0.5f * ( points.front().x + points.back().x ), 0.0f );  // y value doesn't matter
        std::vector<optix::float2>::const_iterator it = std::upper_bound( points.begin(), points.end(), pmid, compare_points_by_x );
        RT_ASSERT( it != points.begin() && it != points.end() );

        right_index = it - points.begin();
        left_index  = right_index - 1;
    }

    if( n == 2 )
        return;

    {
        // initial walk right to grow right endpoint

        // vector along line from left endpoint to right endpoint
        optix::float2 v = points[right_index] - points[left_index];

        int i = right_index + 1;
        for( ; i < n; ++i )
        {

            // evaluate the current bounding line
            const float bound = eval_line( points[left_index].x, points[right_index].x, points[left_index].y,
                                           points[right_index].y, points[i].x );
            if( !compare( bound, points[i].y ) )
            {
                // new point becomes the right endpoint
                right_index = i;
                v           = points[right_index] - points[left_index];
            }
        }
    }


    // Now loop until endpoints do not change.
    // Start by forcing an initial left walk, since we went right above.
    bool endpoint_updated = true;
    bool walk_right       = false;

    while( endpoint_updated )
    {

        endpoint_updated = false;

        if( walk_right )
        {
            // walk right
            optix::float2 v = points[right_index] - points[left_index];
            int           i = right_index + 1;

            for( ; i < n; ++i )
            {

                const float bound = eval_line( points[left_index].x, points[right_index].x, points[left_index].y,
                                               points[right_index].y, points[i].x );
                if( !compare( bound, points[i].y ) )
                {
                    // new point is not bounded by current line.  Replace endpoint.
                    right_index      = i;
                    v                = points[right_index] - points[left_index];
                    endpoint_updated = true;
                }
            }
        }
        else
        {
            // walk left
            optix::float2 v = points[left_index] - points[right_index];
            int           i = left_index - 1;

            for( ; i >= 0; --i )
            {

                const float bound = eval_line( points[left_index].x, points[right_index].x, points[left_index].y,
                                               points[right_index].y, points[i].x );
                if( !compare( bound, points[i].y ) )
                {
                    // new point is not bounded by current line.  Replace endpoint.
                    left_index       = i;
                    v                = points[left_index] - points[right_index];
                    endpoint_updated = true;
                }
            }
        }

        walk_right = !walk_right;
    }
}

// holds a single line segment of an envelope
struct Segment
{
    float left  = 0.0f;
    float right = 0.0f;

    Segment( float left, float right )
        : left( left )
        , right( right )
    {
    }
};

// returns a bounding line segment for the points. Templated to handle upper/lower line.
template <typename C, typename F>
Segment find_bounding_segment( const std::vector<optix::float2>& points, C compare, F eval_line )
{
    int left_index = 0, right_index = 0;
    find_points_on_bounding_line( points, compare, eval_line, left_index, right_index );
    float y0 = eval_line( points[left_index].x, points[right_index].x, points[left_index].y, points[right_index].y,
                          points.front().x );
    float y1 = eval_line( points[left_index].x, points[right_index].x, points[left_index].y, points[right_index].y,
                          points.back().x );
    return Segment( y0, y1 );
}

// Evaluate uniform piecewise linear function at x.
// Input points must be sorted by increasing x value.
template <typename F>
float eval_piecewise_linear( const std::vector<optix::float2>& points, float x, F eval )
{
    RT_ASSERT( !points.empty() );

    // Clamp
    if( x <= points.front().x )
        return points.front().y;
    if( x >= points.back().x )
        return points.back().y;

    optix::float2                              p = optix::make_float2( x, 0.0f );
    std::vector<optix::float2>::const_iterator it = std::upper_bound( points.begin(), points.end(), p, compare_points_by_x );

    // Already clamped
    RT_ASSERT( it != points.begin() && it != points.end() );

    const int index1 = it - points.begin();
    const int index0 = index1 - 1;

    return eval( points[index0].x, points[index1].x, points[index0].y, points[index1].y, x );
}

void get_points_between( const std::vector<optix::float2>& points, float xa, float xb, std::vector<optix::float2>& output )
{

    optix::float2 pa = optix::make_float2( xa, 0.0f );

    // Note: std::upper_bound returns the first element greater than 'pa'.
    std::vector<optix::float2>::const_iterator ita = std::upper_bound( points.begin(), points.end(), pa, compare_points_by_x );

    for( std::vector<optix::float2>::const_iterator it = ita; it != points.end() && it->x < xb; ++it )
    {
        output.push_back( *it );
    }
}

// Split a set of sorted (x,y) input points into x ranges.
template <typename F>
void split_points( const std::vector<optix::float2>&        input_points,
                   const std::vector<float>&                split_x,
                   std::vector<std::vector<optix::float2>>& output_points,
                   F                                        eval )

{

    RT_ASSERT( !input_points.empty() );
    RT_ASSERT( split_x.size() > 1 );

    // Get y values at the split points
    std::vector<float> split_y( split_x.size() );
    for( size_t i  = 0; i < split_x.size(); ++i )
        split_y[i] = eval_piecewise_linear( input_points, split_x[i], eval );

    const size_t num_ranges = split_x.size() - 1;
    for( size_t index = 0; index < num_ranges; ++index )
    {

        std::vector<optix::float2> output;

        // first end point of range
        output.push_back( optix::make_float2( split_x[index], split_y[index] ) );

        // points between.
        get_points_between( input_points, split_x[index], split_x[index + 1], output );

        // second end point of range
        output.push_back( optix::make_float2( split_x[index + 1], split_y[index + 1] ) );

        output_points.push_back( output );
    }
}

template <typename C, typename F>
void compute_elwelope( const std::vector<optix::float2>& points,
                       const std::vector<float>&         split_x,
                       C                                 compare,
                       F                                 eval_line,
                       //output: piecewise linear bounding envelope
                       std::vector<float>& envelope )
{
    RT_ASSERT( split_x.size() > 1 );

    // Split input points into ranges

    std::vector<std::vector<optix::float2>> segment_points;
    split_points( points, split_x, segment_points, eval_line );

    RT_ASSERT( segment_points.size() == split_x.size() - 1 );

    std::vector<Segment> segments;
    segments.reserve( segment_points.size() );

    for( std::vector<optix::float2>& segment_point : segment_points )
    {
        Segment segment = find_bounding_segment( segment_point, compare, eval_line );
        segments.push_back( segment );
    }


    // Weld the segments together at the split points.  Without this, each
    // segment would bound the points in its range, but the segments would not be
    // C0 continuous where they meet.

    for( size_t i = 1; i < segment_points.size(); ++i )
    {

        Segment&                    segment_left         = segments[i - 1];
        Segment&                    segment_right        = segments[i];
        std::vector<optix::float2>& segment_points_left  = segment_points[i - 1];
        std::vector<optix::float2>& segment_points_right = segment_points[i];

        // Replace the first and last input points of this segment with the clamped bounds value
        const float y0                 = segment_left.right;
        const float y1                 = segment_right.left;
        const float yclamped           = compare( y0, y1 ) ? y0 : y1;
        segment_points_left.back().y   = yclamped;
        segment_points_right.front().y = yclamped;

        // Re-run the bounds algorithm.  This gives a tighter envelope vs. only clamping.
        segment_left  = find_bounding_segment( segment_points_left, compare, eval_line );
        segment_right = find_bounding_segment( segment_points_right, compare, eval_line );
    }

    // merge segments into piecewise linear envelope, removing shared points

    envelope.clear();
    envelope.reserve( segments.size() + 1 );

    envelope.push_back( segments[0].left );
    for( Segment& segment : segments )
    {
        envelope.push_back( segment.right );
    }
}

void compute_upper_elwelope( const std::vector<optix::float2>& points,
                             const std::vector<float>&         split_x,
                             //piecewise linear bounding line
                             std::vector<float>& envelope )
{
    compute_elwelope( points, split_x, /*functors*/ std::greater_equal<float>(), roundup_line, /*output*/ envelope );
}

void compute_lower_elwelope( const std::vector<optix::float2>& points,
                             const std::vector<float>&         split_x,
                             //piecewise linear bounding line
                             std::vector<float>& envelope )
{
    compute_elwelope( points, split_x, /*functors*/ std::less_equal<float>(), rounddown_line, /*output*/ envelope );
}


// Utils to access float3 with an index

const float& component( const optix::float3& v, int k )
{
    const float* f = reinterpret_cast<const float*>( &v.x );
    return f[k];
}

float& component( optix::float3& v, int k )
{
    float* f = reinterpret_cast<float*>( &v.x );
    return f[k];
}

std::vector<float> expand_time_range( optix::float2 range, unsigned steps )
{
    RT_ASSERT( steps > 1 );

    const float        dt = ( range.y - range.x ) / ( steps - 1 );
    std::vector<float> times( steps );
    for( unsigned int i = 0; i < steps; ++i )
        times[i]        = range.x + i * dt;

    return times;
}


////////////  Public functions

void resample_motion_aabbs( const std::vector<float>& input_times,
                            const optix::Aabb*        input_boxes,
                            const std::vector<float>& output_times,
                            optix::Aabb*              output_boxes )
{
    // Sanity check input
    RT_ASSERT( input_times.size() > 1 && output_times.size() > 1 );
    for( size_t i = 0; i < input_times.size(); ++i )
    {
        RT_ASSERT( input_boxes[i].valid() );
    }

    // Treat the +/- X, Y, Z axes as 6 separate 1d problems.

    // First set up invalid output boxes
    for( size_t i = 0; i < output_times.size(); ++i )
    {
        output_boxes[i].m_min = optix::make_float3( std::numeric_limits<float>::max() );
        output_boxes[i].m_max = -optix::make_float3( std::numeric_limits<float>::max() );
    }

    for( int axis = 0; axis < 3; ++axis )
    {
        // min
        {
            std::vector<optix::float2> points( input_times.size() );
            for( size_t i = 0; i < input_times.size(); ++i )
            {
                points[i] = optix::make_float2( input_times[i], component( input_boxes[i].m_min, axis ) );
            }

            std::vector<float> lower_bounds;
            compute_lower_elwelope( points, output_times, lower_bounds );

            // accumulate into output
            for( size_t i = 0; i < output_times.size(); ++i )
            {
                component( output_boxes[i].m_min, axis ) = std::min( component( output_boxes[i].m_min, axis ), lower_bounds[i] );
            }
        }

        // max
        {
            std::vector<optix::float2> points( input_times.size() );
            for( size_t i = 0; i < input_times.size(); ++i )
            {
                points[i] = optix::make_float2( input_times[i], component( input_boxes[i].m_max, axis ) );
            }

            std::vector<float> upper_bounds;
            compute_upper_elwelope( points, output_times, upper_bounds );

            for( size_t i = 0; i < output_times.size(); ++i )
            {
                component( output_boxes[i].m_max, axis ) = std::max( component( output_boxes[i].m_max, axis ), upper_bounds[i] );
            }
        }
    }

    // Sanity check output
    for( size_t i = 0; i < output_times.size(); ++i )
    {
        RT_ASSERT( output_boxes[i].valid() );
    }
}


// Uniform version
void resample_motion_aabbs( float              input_t0,
                            float              input_t1,
                            unsigned           input_steps,
                            const optix::Aabb* input_boxes,
                            float              output_t0,
                            float              output_t1,
                            unsigned           output_steps,
                            optix::Aabb*       output_boxes )
{
    std::vector<float> input_times  = expand_time_range( optix::make_float2( input_t0, input_t1 ), input_steps );
    std::vector<float> output_times = expand_time_range( optix::make_float2( output_t0, output_t1 ), output_steps );

    resample_motion_aabbs( input_times, input_boxes, output_times, output_boxes );
}
