/*
 * Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <vector_functions.h>
#include <vector_types.h>
#include <exp/accel/motion/motionCommon.hpp>


namespace optix_exp {
namespace motion {

struct ResampleAabbs
{
    const float        inTimeBegin;
    const float        inTimeEnd;
    const Aabb*        inAabbs;
    const unsigned int inMotionSteps;
    const float*       inTimes;
    Aabb*              outAabbs;
    const unsigned int outMotionSteps;
    const float        outTimeBegin;
    const float        outTimeEnd;
};

struct SegmentBorder
{
    float time = 0;
    // index within the current motion aabbs, i.e., [0,maabb.keyCount)
    unsigned int aabbIndex = 0;
    // (potentially) interpolated aabbs at the segment border
    Aabb interpolatedAabb;
};

template <bool timesRegularDistribution = true>
M_DEVICE_HOST inline void resample_aabbs_in_segment( const ResampleAabbs& p,
                                                     const SegmentBorder& segmentBegin,
                                                     const SegmentBorder& segmentEnd,
                                                     Aabb&                outAabb0,
                                                     Aabb&                outAabb1 )
{
    // only used if timesRegularDistribution == true
    const float inStepSize = timesRegularDistribution ? ( p.inTimeEnd - p.inTimeBegin ) / (float)( p.inMotionSteps - 1 ) : 1.f;
    auto        inTime = [&]( int index ) {
        // 'if' will be resolved at compile time
        if( timesRegularDistribution )
            return p.inTimeBegin + index * inStepSize;
        else
            return p.inTimes[index];
    };
    auto inAabb      = [&]( unsigned int index ) -> const Aabb& { return p.inAabbs[index]; };
    auto lower_bound = [&]( unsigned int first, unsigned int last, float value ) -> unsigned int {
        unsigned int it, step;
        unsigned int count = last - first;
        while( count > 0 )
        {
            it   = first;
            step = count / 2;
            it += step;
            if( inTime( it ) < value )
            {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }
        return first;
    };

    // loop over all input aabbs that are within the segment
    const int begin                  = segmentBegin.aabbIndex;
    const int end                    = segmentEnd.aabbIndex;
    const int countIntermediateBoxes = end - begin;
    if( countIntermediateBoxes == 0 )
    {
        outAabb0 = segmentBegin.interpolatedAabb;
        outAabb1 = segmentEnd.interpolatedAabb;
        return;
    }
    // mid is somewhat 'central-ish' in the case that the input aabbs are uniformly distributed since all aabbs [begin,end) are within the current segment
    int mid = begin + ( countIntermediateBoxes - 1 ) / 2;
    // for irregular distribution, we find the index that is close to the center
    if( !timesRegularDistribution )
    {
        mid = lower_bound( begin, end, lerp( segmentBegin.time, segmentEnd.time, 0.5f ) );
        if( mid == end )
            mid = end - 1;
    }
    const float timeMid = inTime( mid );
    // left index for (axis_min/max): x_min, x_max, y_min, ..., z_max
    int3 leftIndex_min = make_int3( mid, mid, mid );
    int3 leftIndex_max = make_int3( mid, mid, mid );
    // same for right index
    int3 rightIndex_min = make_int3( mid + 1, mid + 1, mid + 1 );
    int3 rightIndex_max = make_int3( mid + 1, mid + 1, mid + 1 );
    outAabb0            = inAabb( mid );
    // derivative... increase of min/max from left index to right index
    float3 diff_min, diff_max;
    if( mid + 1 != end )
    {
        // init to right index (which are all the same: mid+1)
        outAabb1 = inAabb( mid + 1 );
        // diffs are per differential time
        diff_min = ( llf( outAabb1 ) - llf( outAabb0 ) ) / ( timesRegularDistribution ? inStepSize : inTime( mid + 1 ) - timeMid );
        diff_max = ( urb( outAabb1 ) - urb( outAabb0 ) ) / ( timesRegularDistribution ? inStepSize : inTime( mid + 1 ) - timeMid );

        auto extrapolateAndUpdate = [&]( int geomIndex, const Aabb& toCheck, const float diffTime ) {
#define PER_COMPONENT( min, x, comp )                                                                                  \
            {                                                                                                          \
                const float extrapolate = outAabb0.min.x + diff_##min.x * diffTime;                                    \
                if( toCheck.min.x comp extrapolate )                                                                   \
                {                                                                                                      \
                    outAabb1.min.x  = toCheck.min.x;                                                                   \
                    rightIndex_##min.x = geomIndex;                                                                    \
                    diff_##min.x       = ( toCheck.min.x - outAabb0.min.x ) / diffTime;                                \
                }                                                                                                      \
            }
            PER_COMPONENT( min, x, < );
            PER_COMPONENT( min, y, < );
            PER_COMPONENT( min, z, < );
            PER_COMPONENT( max, x, > );
            PER_COMPONENT( max, y, > );
            PER_COMPONENT( max, z, > );
#undef PER_COMPONENT
        };

        // walk from mid+2 to end-1
        for( int geomIndex = mid + 2; geomIndex < end; ++geomIndex )
        {
            const Aabb& toCheck = inAabb( geomIndex );
            // extrapolate from known fixed point mid
            const float diffTime = timesRegularDistribution ? ( geomIndex - mid ) * inStepSize : inTime( geomIndex ) - timeMid;
            extrapolateAndUpdate( geomIndex, toCheck, diffTime );
        }

        if( valid( segmentEnd.interpolatedAabb ) )
        {
            // one more box to check... 'end', i.e., segmentEnd.interpolatedAabb
            const Aabb& toCheck  = segmentEnd.interpolatedAabb;
            const float diffTime = segmentEnd.time - timeMid;
            extrapolateAndUpdate( end, toCheck, diffTime );
        }
    }
    else if( !timesRegularDistribution )
    {
        //assert( valid( segmentEnd.interpolatedAabb ) );
        // init to right index (which are all the same: mid+1)
        outAabb1             = segmentEnd.interpolatedAabb;
        const float diffTime = segmentEnd.time - timeMid;
        diff_min             = ( llf( outAabb1 ) - llf( outAabb0 ) ) / diffTime;
        diff_max             = ( urb( outAabb1 ) - urb( outAabb0 ) ) / diffTime;
    }
    else
    {
        // if mid is really the mid index, mid+1==end can only be the case if countIntermediateBoxes==1
        // hence, we must have one geom aabb and one interpolated aabb
        //assert( countIntermediateBoxes == 1 );
        if( valid( segmentBegin.interpolatedAabb ) && valid( segmentEnd.interpolatedAabb ) )
        {
            outAabb1                = segmentEnd.interpolatedAabb;
            const float ilwDiffTime = 1.0f / ( segmentEnd.time - timeMid );
            diff_min                = ( llf( outAabb1 ) - llf( outAabb0 ) ) * ilwDiffTime;
            diff_max                = ( urb( outAabb1 ) - urb( outAabb0 ) ) * ilwDiffTime;
        }
        else if( valid( segmentBegin.interpolatedAabb ) )
        {
            outAabb0 = segmentBegin.interpolatedAabb;
            outAabb1 = inAabb( mid );
            return;
        }
        else
        {
            outAabb1 = segmentEnd.interpolatedAabb;
            return;
        }
    }

    // Now loop until endpoints do not change.
    // Start by forcing an initial left walk, since we went right above.
    bool endpoint_updated = true;
    bool walk_right       = false;

    auto minComponent = []( const int3& v ) { return min( v.x, min( v.y, v.z ) ); };
    auto maxComponent = []( const int3& v ) { return max( v.x, max( v.y, v.z ) ); };

    while( endpoint_updated )
    {
        endpoint_updated = false;
        // This is slightly different from the version atop as
        // a) it has an additional condition that checks if the rightIndex is < that what we lwrrently compare against (geomIndex).
        // b) we don't have a fixed time diff (time of refAabb.m_##min.x is varying now), but a 'reference' time (that at geomIndex, which may be end, i.e., 'out of bounds')
        // c) this can walk left and right using the 'walk_right' variable. Note that the compiler should optimize tests away when inlining the lambda
        auto extrapolateAndUpdate = [&]( int geomIndex, const Aabb& toCheck, const float toCheckTime ) {
            const int3& refIndex_min    = walk_right ? leftIndex_min : rightIndex_min;
            const int3& refIndex_max    = walk_right ? leftIndex_max : rightIndex_max;
            int3&       targetIndex_min = walk_right ? rightIndex_min : leftIndex_min;
            int3&       targetIndex_max = walk_right ? rightIndex_max : leftIndex_max;
            const Aabb& refAabb         = walk_right ? outAabb0 : outAabb1;
            Aabb&       targetAabb      = walk_right ? outAabb1 : outAabb0;
#define PER_COMPONENT( min, x, comp )                                                                                                                                             \
            if( ( walk_right && rightIndex_##min.x < geomIndex ) || ( !walk_right && leftIndex_##min.x > geomIndex ) )                                                            \
            {                                                                                                                                                                     \
                /*if we walk right, check if the left index is < begin, in which case the reference is the beginning of the segment, i.e., we take the time from 'segmentBegin'*/ \
                const float refTime = ( walk_right && refIndex_##min.x < begin ) || ( !walk_right && refIndex_##min.x >= end ) ?                                                  \
                                          ( walk_right ? segmentBegin.time : segmentEnd.time ) :                                                                                  \
                                          inTime( refIndex_##min.x );                                                                                                             \
                const float diffTime    = toCheckTime - refTime;                                                                                                                  \
                const float extrapolate = refAabb.min.x + diff_##min.x * diffTime;                                                                                                \
                if( toCheck.min.x comp extrapolate )                                                                                                                              \
                {                                                                                                                                                                 \
                    targetAabb.min.x = toCheck.min.x;                                                                                                                             \
                    targetIndex_##min.x = geomIndex;                                                                                                                              \
                    diff_##min.x        = ( toCheck.min.x - refAabb.min.x ) / diffTime;                                                                                           \
                    endpoint_updated    = true;                                                                                                                                   \
                }                                                                                                                                                                 \
            }
            PER_COMPONENT( min, x, < );
            PER_COMPONENT( min, y, < );
            PER_COMPONENT( min, z, < );
            PER_COMPONENT( max, x, > );
            PER_COMPONENT( max, y, > );
            PER_COMPONENT( max, z, > );
#undef PER_COMPONENT
        };

        if( walk_right )
        {
            // go right... find min starting point
            for( int geomIndex = min( minComponent( rightIndex_min ), minComponent( rightIndex_max ) ) + 1; geomIndex < end; ++geomIndex )
            {
                const Aabb& toCheck = inAabb( geomIndex );
                extrapolateAndUpdate( geomIndex, toCheck, inTime( geomIndex ) );
            }

            if( valid( segmentEnd.interpolatedAabb ) )
            {
                // one more box to check... 'end', i.e., segmentEnd.interpolatedAabb
                const Aabb& toCheck = segmentEnd.interpolatedAabb;
                // it is important to use an index >=end
                extrapolateAndUpdate( end, toCheck, segmentEnd.time );
            }
        }
        else
        {
            // go left... find max starting point
            int geomIndex = max( maxComponent( leftIndex_min ), maxComponent( leftIndex_max ) ) - 1;
            for( ; geomIndex >= begin; --geomIndex )
            {
                const Aabb& toCheck = inAabb( geomIndex );
                extrapolateAndUpdate( geomIndex, toCheck, inTime( geomIndex ) );
            }

            if( valid( segmentBegin.interpolatedAabb ) )
            {
                // one more box to check... 'begin', i.e., segmentBegin.interpolatedAabb
                const Aabb& toCheck = segmentBegin.interpolatedAabb;
                // it is important to use an index <begin
                extrapolateAndUpdate( begin - 1, toCheck, segmentBegin.time );
            }
        }
        walk_right = !walk_right;
    }

// extrapolate aabb0, aabb1 to the segment bounds
#define PER_COMPONENT( min, x )                                                                                    \
    if( rightIndex_##min.x < end )                                                                                 \
    {                                                                                                              \
        const float diffTime = segmentEnd.time - inTime( rightIndex_##min.x );                                     \
        outAabb1.min.x    = outAabb1.min.x + diff_##min.x * diffTime;                                              \
    }                                                                                                              \
    if( leftIndex_##min.x >= begin )                                                                               \
    {                                                                                                              \
        const float diffTime = segmentBegin.time - inTime( leftIndex_##min.x );                                    \
        outAabb0.min.x    = outAabb0.min.x + diff_##min.x * diffTime;                                              \
    }
    PER_COMPONENT( min, x );
    PER_COMPONENT( min, y );
    PER_COMPONENT( min, z );
    PER_COMPONENT( max, x );
    PER_COMPONENT( max, y );
    PER_COMPONENT( max, z );
#undef PER_COMPONENT
}

template <bool timesRegularDistribution = true>
M_DEVICE_HOST inline void resample_aabbs( const ResampleAabbs& p )
{
    // only used in case of timesRegularDistribution, refactor?
    const float inStepSize = timesRegularDistribution ? ( p.inTimeEnd - p.inTimeBegin ) / (float)( p.inMotionSteps - 1 ) : 1.f;
    const float outStepSize = ( p.outTimeEnd - p.outTimeBegin ) / (float)( p.outMotionSteps - 1 );
    auto        inTime      = [&]( int index ) {
        // 'if' will be resolved at compile time
        if( timesRegularDistribution )
            return p.inTimeBegin + index * inStepSize;
        else
            return p.inTimes[index];
    };
    auto inAabb           = [&]( unsigned int index ) -> const Aabb& { return p.inAabbs[index]; };
    auto interpolateAabbs = [&]( float refTime, unsigned int aabbIndexA, unsigned int aabbIndexB ) -> Aabb {
        float interpolant;
        // 'if' will be resolved at compile time
        if( timesRegularDistribution )
            interpolant = ( refTime - inTime( aabbIndexA ) ) / inStepSize;
        else
        {
            const float timeA = inTime( aabbIndexA );
            const float timeB = inTime( aabbIndexB );
            interpolant       = ( refTime - timeA ) / ( timeB - timeA );
        }
        return lerp( inAabb( aabbIndexA ), inAabb( aabbIndexB ), interpolant );
    };
    auto outAabb = [&]( unsigned int index ) -> Aabb& { return p.outAabbs[index]; };

    // Note that the meaning of segmentBorder.aabbOffset is somewhat different for segmentBegin/segmentEnd as they are considered as standard iterators
    //  When processing a segment, we assume the set of input geom aabbs to be [segmentBegin.aabbOffset, segmentEnd.aabbOffset)
    // Therefore, the handling of inTime(segmentBorder.aabbOffset) == segmentBorder.time is different for segmentBegin/segmentEnd
    // -> for segmentBegin we use segmentBegin.aabbOffset
    // -> for segmentEnd, we set segmentEnd.inInterpolatedAabb = inAabb(segmentEnd.aabbOffset);
    // We could also implement segmentBegin/segmentEnd as pointers and swap the pointers after processing a segment, but that may confuse the compiler when it comes to registers.
    SegmentBorder segmentBegin, segmentEnd;

    // ilwalidate first output aabb, as it may only get extended, all others will be overwritten
    ilwalidate( outAabb( 0 ) );

    // loop over output segments
    // 'segmentIndex' is a bit arbitrary and corresponds here to the aabb index at the end of the segment
    // -> could also be the beginning, but this way seemed more colwinient (fewer +1)
    // find first segment where p.inTimeBegin < segmentEnd.time
    unsigned int segmentIndex = 1u;
    segmentBegin.time         = p.outTimeBegin;
    segmentEnd.time           = ( p.outMotionSteps == 2 ) ? p.outTimeEnd : p.outTimeBegin + outStepSize;
    while( p.inTimeBegin >= segmentEnd.time && segmentIndex < p.outMotionSteps )
    {
        // clamp
        // todo: consider in border mode vanish... outAabb(segmentIndex).ilwalidate()?
        outAabb( segmentIndex ) = inAabb( 0 );
        ++segmentIndex;
        segmentBegin.time = segmentEnd.time;
        segmentEnd.time = ( segmentIndex == p.outMotionSteps - 1 ) ? p.outTimeEnd : p.outTimeBegin + segmentIndex * outStepSize;
    }
    // outAabb(0) has not yet been touched...
    // if we actually 'skipped' some segments, also set it to inAabb(0)
    // alternatively, we could also add 'outAabb(segmentIndex-1) = inAabb(0);' into the loop above
    if( segmentIndex > 1 )
        outAabb( 0 ) = inAabb( 0 );

    //////////////////////////////////////////////////////////////////////////
    // neither of this should be needed/happen since we assume that (as of now):
    // p.outTimeBegin <= p.inTimeBegin && p.outTimeEnd >= p.inTimeEnd
    if( segmentIndex == p.outMotionSteps )
        return;
    while( inTime( segmentBegin.aabbIndex ) < segmentBegin.time && segmentBegin.aabbIndex < p.inMotionSteps )
        ++segmentBegin.aabbIndex;
    // -> inTime(segmentBegin.aabbOffset) >= segmentBegin.time || segmentBegin.aabbOffset == inMotionSteps
    //////////////////////////////////////////////////////////////////////////

    // given: p.inTimeBegin < segmentEnd.time
    if( p.inTimeBegin == segmentBegin.time )
    {
        // if the times align, we want to use in aabb directly (segmentBegin points to the first aabb that will be used for the segment)
        // hence, we ilwalidate the interpolate as we don't want to use it!
        ilwalidate( segmentBegin.interpolatedAabb );
    }
    else if( segmentBegin.aabbIndex == 0 )
    {
        // as above, this must be the only possible else case, if:
        // p.outTimeBegin <= p.inTimeBegin && p.outTimeEnd >= p.inTimeEnd
        segmentBegin.interpolatedAabb = inAabb( segmentBegin.aabbIndex );
    }
    else
    {
        // as above, not reachable if
        // p.outTimeBegin <= p.inTimeBegin && p.outTimeEnd >= p.inTimeEnd
        segmentBegin.interpolatedAabb = interpolateAabbs( segmentBegin.time, segmentBegin.aabbIndex - 1, segmentBegin.aabbIndex );
    }

    segmentEnd.aabbIndex = segmentBegin.aabbIndex;

    for( ; segmentIndex < p.outMotionSteps; ++segmentIndex )
    {
        // hopefully, this test avoids potential (minor) precision issues
        segmentEnd.time = ( segmentIndex == p.outMotionSteps - 1 ) ? p.outTimeEnd : p.outTimeBegin + segmentIndex * outStepSize;
        while( inTime( segmentEnd.aabbIndex ) < segmentEnd.time && segmentEnd.aabbIndex < p.inMotionSteps )
            ++segmentEnd.aabbIndex;
        bool done       = false;
        bool endAligned = false;
        if( segmentEnd.aabbIndex < p.inMotionSteps )
        {
            // inTimeAtIndex >= segmentEnd.time
            if( inTime( segmentEnd.aabbIndex ) == segmentEnd.time )
            {
                // if the times align, we want do NOT use the in directly (the end points to the first aabb that will NOT be used for the segment)
                // hence, we set the interpolate
                // note that we do not want to simply advance segmentEnd.aabbIndex to be able to assign segmentEnd to segmentBegin
                segmentEnd.interpolatedAabb = inAabb( segmentEnd.aabbIndex );
                endAligned                  = true;
            }
            else
            {
                // inTime(segmentEnd.aabbIndex > segmentEnd.time
                segmentEnd.interpolatedAabb = interpolateAabbs( segmentEnd.time, segmentEnd.aabbIndex - 1, segmentEnd.aabbIndex );
            }
        }
        else
        {
            segmentEnd.interpolatedAabb = inAabb( p.inMotionSteps - 1 );
            done                        = true;
        }

        // Note the case of:
        // inTime(segmentBegin.aabbIndex) == segmentEnd.time
        //if( inTime( segmentBegin.aabbIndex ) == segmentEnd.time && segmentBegin.aabbIndex < p.inMotionSteps )
        //    assert( valid( segmentBegin.interpolatedAabb ) && segmentBegin.aabbIndex == segmentEnd.aabbIndex
        //            && valid( segmentEnd.interpolatedAabb ) );

        Aabb outAabb0;
        resample_aabbs_in_segment<timesRegularDistribution>( p, segmentBegin, segmentEnd, outAabb0, outAabb( segmentIndex ) );
        include( outAabb( segmentIndex - 1u ), outAabb0 );

        if( done )
            break;

        // for next segment.. new begin = old end
        segmentBegin = segmentEnd;
        if( endAligned )
            // The handling of inTime(segmentBorder.aabbIndex) == segmentBorder.time is different for segmentBegin/segmentEnd (see above)
            // In case of time alignment (now: inTime(segmentBegin.aabbIndex) == segmentBegin.time)),
            // use the key directly (segmentBegin.aabbIndex), ilwalidate the interpolation
            ilwalidate( segmentBegin.interpolatedAabb );
    }

    while( ++segmentIndex < p.outMotionSteps )
    {
        //assert( segmentEnd.aabbIndex == p.inMotionSteps );
        // clamp
        // todo: consider in border mode vanish... outAabb(segmentIndex).ilwalidate()?
        const float segmentBeginTime = p.outTimeBegin + segmentIndex * outStepSize;
        //assert( ( timesRegularDistribution && p.inTimeEnd < segmentBeginTime )
        //        || ( !timesRegularDistribution && inTime( segmentEnd.aabbIndex - 1 ) < segmentBeginTime ) );
        outAabb( segmentIndex ) = inAabb( p.inMotionSteps - 1 );
        ++segmentIndex;
    }
}

// resampling of regular aabbs
static M_DEVICE_HOST void resampleMotionAabbs( const float        inTimeBegin,
                                               const float        inTimeEnd,
                                               const Aabb*        inAabbs,
                                               const unsigned int inAabbsCount,
                                               const float        outTimeBegin,
                                               const float        outTimeEnd,
                                               Aabb*              outAabbs,
                                               const unsigned int outAabbsCount )
{
    ResampleAabbs p = {inTimeBegin, inTimeEnd, inAabbs, inAabbsCount, 0,
                       outAabbs, outAabbsCount, outTimeBegin, outTimeEnd};
    resample_aabbs<true>( p );

    return;
}

static M_DEVICE_HOST void resampleMotionAabbs( const bool         inputTimesRegularDistribution,
                                               const float        inputTimeBegin,
                                               const float        inputTimeEnd,
                                               const Aabb*        inputAabbs,
                                               const unsigned int inputAabbsCount,
                                               const float*       inputTimes,
                                               Aabb*              buildAabbs,
                                               const unsigned int buildMotionSteps,
                                               const float        buildTimeBegin,
                                               const float        buildTimeEnd )
{
    if( inputAabbsCount == 0 )
    {
        for( size_t k = 0; k < buildMotionSteps; ++k )
            ilwalidate( buildAabbs[k] );
    }
    else if( inputAabbsCount == 1 )
    {
        for( size_t k = 0; k < buildMotionSteps; ++k )
            buildAabbs[k] = inputAabbs[0];
    }
    else
    {
        ResampleAabbs p = {inputTimeBegin, inputTimeEnd, inputAabbs, inputAabbsCount, inputTimes,
            buildAabbs, buildMotionSteps, buildTimeBegin, buildTimeEnd};
        if( inputTimesRegularDistribution )
            resample_aabbs<true>( p );
        else
            resample_aabbs<false>( p );
    }

    return;
}

}  // namespace motion
}  // namespace optix_exp
