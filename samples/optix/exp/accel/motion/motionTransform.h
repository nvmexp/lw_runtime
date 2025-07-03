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


#include <rtcore/interface/types.h>
#include "motionCommon.hpp"

namespace optix_exp {
namespace motion {

#ifdef __LWDACC__
namespace lwca {
#endif

template<typename TransformData>
M_DEVICE_HOST void processMotionTransform( const MotionOptions& instanceMO, unsigned int intervalIdx, float intervalBegin, float intervalEnd,
                                           AabbArray*& pInAabbs, AabbArray*& pOutAabbs,
                                           const MotionOptions& transformMO, unsigned int& transformLastKeyIndex, TransformData* transformData );

// General idea:
// We "march" in time from key to key, always considering the next (time-wise) key whether this is an input or a transform key.
// As a result, we generally handle intervals and return the aabb for the beginning of the interval and the aabb at the end of the interval such that
//  the aabbs (outAabb0, outAabb1) can be linearly interpolated and fully contain, i.e., lerp(outAabb0, outAabb1, t) >= transform(key0, key1, t) * lerp(inAabb0, inAabb1, t)
class AabbIntervalComputer
{
public:
    M_DEVICE_HOST AabbIntervalComputer( const MotionOptions& _motionOptions, float _intervalBegin, float _intervalEnd, unsigned _initTransformKeyIdx, const AabbArray& _inAabbs, AabbArray& _outAabbs );

    template<typename TransformData>
    M_DEVICE_HOST void march( const TransformData* data );

    M_DEVICE_HOST inline unsigned int getLwrrentTransformKey() const {
        // nextTransformKey is 0 iff first transform key is after intervalEnd
        return max( 1u, nextTransformKey ) - 1;
    }

private:
    M_DEVICE_HOST float transformKeyTimeInfinity( unsigned int key );
    M_DEVICE_HOST void advanceTransformKey();
    M_DEVICE_HOST void advanceInputKey();

protected:
    const MotionOptions& motionOptions;
    const AabbArray& inAabbs;
    AabbArray& outAabbs;
    float intervalBegin;
    float intervalEnd;

    // Marching state
    const float  transformIntervalTime;
    const float  inputIntervalTime;
    unsigned int nextTransformKey;
    unsigned int nextInputKey;
    float        prevTransformKeyTime;
    float        nextTransformKeyTime;
    float        nextInputKeyTime;
    bool         nextKeyIsTransformKey, nextKeyIsInputKey, prevKeyWasTransformKey = false, prevKeyWasInputKey = false;
};

#ifdef __LWDACC__
} // namespace lwca
#endif

}  // namespace motion
}  // namespace optix_exp

#include "motionTransform.hpp"
