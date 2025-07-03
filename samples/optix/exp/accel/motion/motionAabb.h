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

#pragma once

#include "motionCommon.hpp"
#include <vector_types.h>
#include <vector_functions.h>
#include <cassert>
#include <algorithm>
#include <limits>

namespace optix_exp {
namespace motion {

template<typename T>
M_DEVICE_HOST __inline__ void swap( T& a, T& b )
{
    T c = a;
    a   = b;
    b   = c;
}

class MotionAabb
{
public:

    M_DEVICE_HOST __inline__ MotionAabb( Aabb *aabbs, float* irregular_times )
        : m_irregular_times( irregular_times )
        , m_aabbs( aabbs )
    {}

    M_DEVICE_HOST __inline__ MotionAabb(MotionAabb&& o)
        : m_motion_flag_start_vanish(o.m_motion_flag_start_vanish)
        , m_motion_flag_end_vanish  (o.m_motion_flag_end_vanish)
        , m_has_regular_times       (o.m_has_regular_times)
        , m_t0                      (o.m_t0)
        , m_t1                      (o.m_t1)
        , m_irregular_times         (o.m_irregular_times)
        , m_aabbs                   (o.m_aabbs)
        , m_numKeys                 (o.m_numKeys)
    {
        o.m_irregular_times         = 0;
        o.m_aabbs                   = 0;
    }

    M_DEVICE_HOST __inline__ MotionAabb& operator=(MotionAabb&& o)
    {
        m_motion_flag_start_vanish  = o.m_motion_flag_start_vanish;
        m_motion_flag_end_vanish    = o.m_motion_flag_end_vanish;
        m_has_regular_times         = o.m_has_regular_times;
        m_t0                        = o.m_t0;
        m_t1                        = o.m_t1;

        m_irregular_times           = o.m_irregular_times;
        m_aabbs                     = o.m_aabbs;
        m_numKeys                   = o.m_numKeys;

        o.m_irregular_times         = 0;
        o.m_aabbs                   = 0;

        return *this;
    }

    // valid: we have at least one key, and each key is a valid aabb
    M_DEVICE_HOST __inline__ bool isValid() const
    {
        if( keyCount() == 0 )
            return false;
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( unsigned int i = 0; i < keyCount(); ++i )
        {
            if( !valid( m_aabbs[i] ) )
                return false;
        }
        return true;
    }

    M_DEVICE_HOST __inline__ bool isStatic() const { return keyCount() == 1; }
    M_DEVICE_HOST __inline__ unsigned int keyCount() const { return static_cast<unsigned int>( m_numKeys ); }
    M_DEVICE_HOST __inline__ bool canUseForASBuild( const MotionOptions& options ) const
    {
        return keyCount() == options.numKeys && keysAreRegularlyDistributed() && timeFirstKey() == options.timeBegin
               && timeLastKey() == options.timeEnd;
    }
    // Whether aabbs are uniformly distributed over a time span t0, t1
    M_DEVICE_HOST __inline__ bool keysAreRegularlyDistributed() const { return m_has_regular_times; }
    M_DEVICE_HOST __inline__ bool keysAlign( float otherTimeFirstKey, float otherTimeLastKey, unsigned int otherKeyCount ) const
    {
        return keysAreRegularlyDistributed() && timeFirstKey() == otherTimeFirstKey && timeLastKey() == otherTimeLastKey
               && keyCount() == otherKeyCount;
    }
    M_DEVICE_HOST __inline__ bool keysAlign( const MotionAabb& other ) const
    {
        return other.keysAreRegularlyDistributed() && keysAlign( other.timeFirstKey(), other.timeLastKey(), other.keyCount() );
    }
    M_DEVICE_HOST __inline__ void setIlwalid() { clear(); }
    M_DEVICE_HOST __inline__ void clear() { m_numKeys = 0; }

    M_DEVICE_HOST __inline__ float keysIntervalTime() const { return ( timeLastKey() - timeFirstKey() ) / ( keyCount() - 1 ); }
    M_DEVICE_HOST __inline__ float keyTime( unsigned int i ) const
    {
        assert( ( keysAreRegularlyDistributed() && keyCount() > 1 ) || !keysAreRegularlyDistributed() );
        return !keysAreRegularlyDistributed() ?
                   m_irregular_times[i] :
                   lerp( timeFirstKey(), timeLastKey(), float( i ) / ( keyCount() - 1 ) );
    }
    M_DEVICE_HOST __inline__ float        timeFirstKey() const { return m_t0; }
    M_DEVICE_HOST __inline__ float        timeLastKey() const { return m_t1; }
    M_DEVICE_HOST __inline__ const float* keyTimes() const { return m_irregular_times; }

    M_DEVICE_HOST __inline__ Aabb aabbUnion() const
    {
        Aabb result = aabb( 0 );
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( unsigned int i = 0; i < keyCount(); ++i )
        {
            if( valid( aabb( i ) ) )
                include( result, aabb( i ) );
            else
            {
                ilwalidate( result );
                break;
            }
        }
        return result;
    }

    M_DEVICE_HOST __inline__ Aabb&       aabb( unsigned int key_index ) { return m_aabbs[key_index]; }
    M_DEVICE_HOST __inline__ const Aabb& aabb( unsigned int key_index ) const { return m_aabbs[key_index]; }
    M_DEVICE_HOST __inline__ Aabb*       aabbs() { return m_aabbs; }
    M_DEVICE_HOST __inline__ const Aabb* aabbs() const { return m_aabbs; }


    M_DEVICE_HOST __inline__ Aabb interpolateKeys( unsigned int first, unsigned int second, float time ) const
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
        return lerp( m_aabbs[first], m_aabbs[second], interpolant );
    }

    M_DEVICE_HOST __inline__ void initRegularDistribution( float t0, float t1, unsigned int steps )
    {
        m_t0 = t0; m_t1 = t1;
        m_numKeys = steps;
        m_has_regular_times = true;
    }

    M_DEVICE_HOST __inline__ void initIrregularDistribution( float t0 )
    {
        assert( m_irregular_times != 0 );
        m_t0 = t0; m_t1 = t0;
        m_has_regular_times = false;
        clear();
    }

    M_DEVICE_HOST __inline__ void markRegularDistribution() { m_has_regular_times = true; }
    M_DEVICE_HOST __inline__ bool motionFlagStartVanish() const { return m_motion_flag_start_vanish; }
    M_DEVICE_HOST __inline__ void setMotionFlagStartVanish(bool vanish) { m_motion_flag_start_vanish = vanish; }
    M_DEVICE_HOST __inline__ bool motionFlagEndVanish() const { return m_motion_flag_end_vanish; }
    M_DEVICE_HOST __inline__ void setMotionFlagEndVanish(bool vanish) { m_motion_flag_end_vanish = vanish; }

    M_DEVICE_HOST __inline__ void pushIrregularKey( const float time, const Aabb &aabb )
    {
        assert( !m_has_regular_times );
        m_aabbs[m_numKeys]           = aabb;
        m_irregular_times[m_numKeys] = time;
        m_t1                         = time;
        m_numKeys++;
    }

    M_DEVICE_HOST __inline__ void swap( MotionAabb& o )
    {
        optix_exp::motion::swap( m_motion_flag_start_vanish, o.m_motion_flag_start_vanish );
        optix_exp::motion::swap( m_motion_flag_end_vanish, o.m_motion_flag_end_vanish );
        optix_exp::motion::swap( m_has_regular_times, o.m_has_regular_times );
        optix_exp::motion::swap( m_t0, o.m_t0 );
        optix_exp::motion::swap( m_t1, o.m_t1 );

        optix_exp::motion::swap( m_irregular_times, o.m_irregular_times );
        optix_exp::motion::swap( m_aabbs, o.m_aabbs );
        optix_exp::motion::swap( m_numKeys, o.m_numKeys );
    }

    // swap everything except the aabbs
    M_DEVICE_HOST __inline__ void swapDistribution( MotionAabb& o )
    {
        swap( o );
        optix_exp::motion::swap( m_aabbs, o.m_aabbs );  // swap aabbs back
    }

private:

    M_DEVICE_HOST __inline__ MotionAabb(const MotionAabb& o) = delete;
    M_DEVICE_HOST __inline__ MotionAabb& operator=(const MotionAabb& o) = delete;

    bool m_motion_flag_start_vanish = false;
    bool m_motion_flag_end_vanish   = false;
    bool m_has_regular_times        = false;
    float m_t0, m_t1;

    float*       m_irregular_times = 0;
    Aabb*        m_aabbs           = 0;
    unsigned int m_numKeys         = 0;

};

}
}
