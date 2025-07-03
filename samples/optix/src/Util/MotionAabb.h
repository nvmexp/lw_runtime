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

#pragma once

#include <optixu/optixu_aabb_namespace.h>
#include <prodlib/exceptions/Assert.h>
#include <vector>

namespace optix {

class MotionAabb
{
public:
    MotionAabb() {}
    MotionAabb( float t0, float t1 );
    MotionAabb( float t0, float t1, const std::vector<Aabb>& aabbs );
    MotionAabb( const std::vector<MotionAabb>& merge );

    // valid: we have at least one key, and each key is a valid aabb
    bool isValid() const;

    inline bool         isStatic() const { return keyCount() == 1; }
    inline unsigned int keyCount() const { return m_aabbs.size(); }
    // Whether aabbs are uniformly distributed over a time span t0, t1
    // Note that the static case returns false as there is no distribution in that case
    inline bool keysAreRegularlyDistributed() const { return m_irregular_times.size() == 2; }
    inline bool keysAlign( float otherTimeFirstKey, float otherTimeLastKey, unsigned int otherKeyCount )
    {
        return keysAreRegularlyDistributed() && timeFirstKey() == otherTimeFirstKey && timeLastKey() == otherTimeLastKey
               && keyCount() == otherKeyCount;
    }
    inline bool keysAlign( const MotionAabb& other )
    {
        return other.keysAreRegularlyDistributed() && keysAlign( other.timeFirstKey(), other.timeLastKey(), other.keyCount() );
    }
    inline void setIlwalid() { clear(); }
    inline void clear() { m_aabbs.clear(); }

    inline float keysIntervalTime() const { return ( timeLastKey() - timeFirstKey() ) / ( keyCount() - 1 ); }
    inline float keyTime( unsigned int i ) const
    {
        RT_ASSERT( ( keysAreRegularlyDistributed() && keyCount() > 1 ) || !keysAreRegularlyDistributed() );
        return !keysAreRegularlyDistributed() ?
                   m_irregular_times[i] :
                   optix::lerp( m_irregular_times[0], m_irregular_times[1], float( i ) / ( keyCount() - 1 ) );
    }
    inline float                     timeFirstKey() const { return m_irregular_times[0]; }
    inline float                     timeLastKey() const { return m_irregular_times.back(); }
    inline std::vector<float>&       keyTimes() { return m_irregular_times; }
    inline const std::vector<float>& keyTimes() const { return m_irregular_times; }

    Aabb                            aabbUnion() const;
    inline Aabb&                    aabb( unsigned int key_index ) { return m_aabbs[key_index]; }
    inline const Aabb&              aabb( unsigned int key_index ) const { return m_aabbs[key_index]; }
    inline std::vector<Aabb>&       aabbs() { return m_aabbs; }
    inline const std::vector<Aabb>& aabbs() const { return m_aabbs; }

    Aabb interpolateKeys( unsigned int first, unsigned int second, float time ) const;

    void resizeWithRegularDistribution( float t0, float t1, unsigned int steps, const Aabb& resizeWith );

    inline RTmotionbordermode borderModeBegin() const { return m_border_mode_begin; }
    inline void               setBorderModeBegin( RTmotionbordermode mode ) { m_border_mode_begin = mode; }
    inline RTmotionbordermode borderModeEnd() const { return m_border_mode_end; }
    inline void               setBorderModeEnd( RTmotionbordermode mode ) { m_border_mode_end = mode; }

private:
    // the vector has a size of two if the aabbs are regularly distributed.
    std::vector<float> m_irregular_times;
    std::vector<Aabb>  m_aabbs;
    RTmotionbordermode m_border_mode_begin = RT_MOTIONBORDERMODE_CLAMP;
    RTmotionbordermode m_border_mode_end   = RT_MOTIONBORDERMODE_CLAMP;
};
}
