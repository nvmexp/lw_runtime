// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/MotionAabb.h>

#include <Util/ContainerAlgorithm.h>

#include <cmath>
#include <functional>

using namespace std;
using namespace optix;


MotionAabb::MotionAabb( float t0, float t1, const std::vector<Aabb>& aabbs )
    : m_irregular_times( {t0, t1} )
    , m_aabbs( aabbs )
{
}

optix::MotionAabb::MotionAabb( float t0, float t1 )
    : m_irregular_times( {t0, t1} )
{
}

optix::MotionAabb::MotionAabb( const std::vector<MotionAabb>& merge )
{
    // TODO: make epsilon dependent on the time ranges of the inputs (merge)
    const float keys_align_epsilon = 0.00001f;

    // loop from front to back time-wise and collect keys from the input
    struct KeyIndex
    {
        float        time;
        unsigned int keyIndex;
        unsigned int maabbIndex;
    };
    // reverse(!!!) ordered list of time to index of input maabbs (to decide which maabb is processed next)
    // each input maabb is only once in that list (unless all its keys are processed)
    // could use a map here, but expected numbers of input maabbs is small (<10)
    std::vector<KeyIndex> nextMaabb;
    // maabbs whose last key is < than the current key.
    // needed for border handling (where aabb for any time after last key may be seen as static aabb - clamp mode)
    std::vector<unsigned int> pastLastKeyMaabbs;
    std::vector<unsigned int> staticMaabbs;

    nextMaabb.reserve( merge.size() );
    pastLastKeyMaabbs.reserve( merge.size() );
    for( size_t i = 0; i < merge.size(); ++i )
    {
        const MotionAabb& maabb = merge[i];
        if( !maabb.isValid() )
            continue;
        if( !maabb.isStatic() )
            nextMaabb.push_back( {maabb.timeFirstKey(), 0u, (unsigned int)i} );
        else
            staticMaabbs.push_back( i );
    }

    // all static or invalid
    if( nextMaabb.empty() )
    {
        if( !staticMaabbs.empty() )
        {
            // merge the static maabbs to a new static maabb
            m_aabbs.push_back( merge[staticMaabbs[0]].aabb( 0 ) );
            for( size_t i = 1; i < staticMaabbs.size(); ++i )
            {
                m_aabbs[0].include( merge[staticMaabbs[i]].aabb( 0 ) );
            }
        }
        // nothing further to do
        return;
    }

    // reverse time sort
    algorithm::sort( nextMaabb, []( const KeyIndex& a, const KeyIndex& b ) { return a.time > b.time; } );

    std::function<bool( unsigned int )> advance = [&]( unsigned int nextMaabbIndex ) {
        KeyIndex&         next  = nextMaabb[nextMaabbIndex];
        const MotionAabb& maabb = merge[next.maabbIndex];

        // update next key to the key after
        const bool isLastKey = maabb.keyCount() - 1 == next.keyIndex;
        if( isLastKey )
        {
            if( maabb.borderModeEnd() != RT_MOTIONBORDERMODE_VANISH )
            {
                RT_ASSERT( maabb.borderModeEnd() == RT_MOTIONBORDERMODE_CLAMP );
                pastLastKeyMaabbs.push_back( next.maabbIndex );
            }
            nextMaabb.erase( nextMaabb.begin() + nextMaabbIndex );
        }
        else
        {
            next.keyIndex++;
            next.time = maabb.keyTime( next.keyIndex );
            // bubble "up" (to front) with new time
            for( size_t i = nextMaabbIndex; i > 0; --i )
            {
                // i-th element is the element in question
                if( nextMaabb[i - 1].time < nextMaabb[i].time )
                    std::swap( nextMaabb[i - 1], nextMaabb[i] );
                else
                    break;
            }
        }
        return !isLastKey;
    };

    // "auto detect" of the output keys are regularly distributed
    // will be set once we have the first interval (i.e., on the second key)
    // all further intervals will be checked against this interval
    // auto detection is not needed if it was previously "decided" (detected) that output is regular
    bool  maabbHasRegularDistribution = true;
    float detectedIntervalTime        = 0.0f;
    while( !nextMaabb.empty() )
    {
        KeyIndex& current = nextMaabb.back();

        //////////////////////////////////////////////////////////////////////////
        // time management
        if( m_irregular_times.size() == 1 )
        {
            detectedIntervalTime = current.time - m_irregular_times[0];
        }
        else if( m_irregular_times.size() > 1 )
        {
            maabbHasRegularDistribution &=
                std::abs( detectedIntervalTime - ( current.time - m_irregular_times.back() ) ) <= keys_align_epsilon;
        }
        m_irregular_times.push_back( current.time );

        //////////////////////////////////////////////////////////////////////////
        // aabb management
        m_aabbs.push_back( merge[current.maabbIndex].aabb( current.keyIndex ) );
        Aabb& lwrrentAabb = m_aabbs.back();

        // include all static aabbs
        for( unsigned int staticMaabb : staticMaabbs )
        {
            lwrrentAabb.include( merge[staticMaabb].aabb( 0 ) );
        }

        // include all clamped aabbs
        for( unsigned int pastLastKeyMaabb : pastLastKeyMaabbs )
        {
            lwrrentAabb.include( merge[pastLastKeyMaabb].aabbs().back() );
        }

        // process all other motion aabbs
        // last one is the current one, so do not touch it
        for( size_t i = 0; i < nextMaabb.size() - 1; )
        {
            KeyIndex& other = nextMaabb[i];
            bool      incrI = true;

            // check if we can merge the other key because it is <= epsilon away
            if( nextMaabb[i].time - current.time <= keys_align_epsilon )
            {
                // merge aabb and advance
                lwrrentAabb.include( merge[other.maabbIndex].aabb( other.keyIndex ) );
                // note, advancing changes nextMaabb, which we are lwrrently looping over
                // element i can
                // a) only bubble "up" (new i will be smaller, hence, we will not process it again in this loop)
                // b) be removed from the nextMaabb list, i.e., could not be advanced
                if( !advance( i ) )
                    // erased element i, do not increment i
                    incrI = false;
            }
            // check for pre-first key aabbs that get clamped
            else if( other.keyIndex == 0 && merge[other.maabbIndex].borderModeBegin() != RT_MOTIONBORDERMODE_VANISH )
            {
                RT_ASSERT( merge[other.maabbIndex].borderModeBegin() == RT_MOTIONBORDERMODE_CLAMP );
                lwrrentAabb.include( merge[other.maabbIndex].aabb( 0 ) );
            }
            else
            {
                // not first key for merge[other.maabbIndex]
                // next key merge[other.maabbIndex] too far away for merging
                // interpolate!
                lwrrentAabb.include(
                    merge[other.maabbIndex].interpolateKeys( other.keyIndex - 1, other.keyIndex, current.time ) );
            }
            if( incrI )
                ++i;
        }

        // advance may 'fail', i.e., 'current' may get removed, but no special handling required
        advance( nextMaabb.size() - 1 );
    }

    RT_ASSERT( isValid() && !isStatic() );
    if( maabbHasRegularDistribution && m_irregular_times.size() > 2 )
    {
        m_irregular_times[1] = m_irregular_times.back();
        m_irregular_times.resize( 2 );
    }
}

bool MotionAabb::isValid() const
{
    if( m_aabbs.empty() )
        return false;
    for( const Aabb& aabb : m_aabbs )
    {
        if( !aabb.valid() )
            return false;
    }
    return true;
}

void MotionAabb::resizeWithRegularDistribution( float t0, float t1, unsigned int steps, const Aabb& resizeWith )
{
    m_irregular_times = {t0, t1};
    m_aabbs.resize( steps, resizeWith );
}

Aabb MotionAabb::interpolateKeys( unsigned int first, unsigned int second, float time ) const
{
    float interpolant;
    if( keysAreRegularlyDistributed() )
    {
        const float interval = keysIntervalTime();
        interpolant          = ( time - ( ( first * interval ) + timeFirstKey() ) ) / ( ( second - first ) * interval );
    }
    else
    {
        interpolant = ( time - m_irregular_times[first] ) / ( m_irregular_times[second] - m_irregular_times[first] );
    }
    return Aabb( optix::lerp( m_aabbs[first].m_min, m_aabbs[second].m_min, interpolant ),
                 optix::lerp( m_aabbs[first].m_max, m_aabbs[second].m_max, interpolant ) );
}

Aabb MotionAabb::aabbUnion() const
{
    Aabb result;
    for( unsigned int i = 0; i < keyCount(); ++i )
    {
        if( aabb( i ).valid() )
            result.include( aabb( i ) );
        else
        {
            result.ilwalidate();
            break;
        }
    }
    return result;
}
