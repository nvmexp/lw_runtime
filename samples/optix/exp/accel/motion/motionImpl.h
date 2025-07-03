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

#include <functional>
#include <optix_types.h>
#include "motionAabb.h"


namespace optix_exp {
namespace motion {

// apply static 3x4 matrix transform to a MotionAabb
M_DEVICE_HOST static void applyMatrixStaticTransform( const float* matrixTransform, MotionAabb& inAabb, MotionAabb& outAabb );
// apply array of N 3x4 matrices to a MotionAabb, N must equal motionOptions::numKeys, N must be >=2
M_DEVICE_HOST static void applyMatrixMotionTransform( const float* matrixTransforms, const MotionOptions& motionOptions, MotionAabb& inAabb, MotionAabb& outAabb );
// apply array of N OptixSRTData to a MotionAabb, N must equal motionOptions::numKeys, N must be >=2
M_DEVICE_HOST static void applySrtMotionTransform( const SRTData* srtTransforms, const MotionOptions& motionOptions, MotionAabb& inAabb, MotionAabb& outAabb );

#ifdef __LWDACC__
namespace lwca {
#endif

// General idea:
// We "march" in time from key to key, always considering the next (time-wise) key whether this is an input or a transform key.
// As a result, we generally handle intervals and return the aabb for the beginning of the interval and the aabb at the end of the interval such that
//  the aabbs (outAabb0, outAabb1) can be linearly interpolated and fully contain, i.e., lerp(outAabb0, outAabb1, t) >= transform(key0, key1, t) * lerp(inAabb0, inAabb1, t)
class AabbIntervalComputerBase
{
public:
    M_DEVICE_HOST AabbIntervalComputerBase( const MotionOptions& _motionOptions, const MotionAabb& _inputMaabb );

    M_DEVICE_HOST void march( MotionAabb &outAabb );

protected:
    // The following functions are ilwoked to handle the special cases during the marching.
    // The functions return the output aabb that satisfies the input / transform interpolation for the end of the interval
    // from the previous key to the next key. I.e., this aabb is pushed back to the list of output aabbs.
    // Typically, the previous Aabb (lastOutput) that marks the beginning of the interval also needs to be adjusted.
    // In case of pre-/post-transform, the transform is constant, hence, it is guaranteed that the previous aabb does not need to be modified.
    //
    // Matrix of possibilities
    // next transform == next key to process is a transform key, i.e., interval goes from previous key (if any) to transform key
    // next input == next key to process is an input key, i.e., interval goes from previous key to input key
    // next merge == next transform key and input key align, i.e., interval goes from previous key to transform/input key
    // pre transform == the interval to process is before the first transform key
    // post transform == the interval to process is after the last transform key
    // pre input == the interval to process is before the first input key
    // post input == the interval to process is after the last input key
    // interpolate == inbetween transform and input keys
    M_DEVICE_HOST virtual Aabb nextTransformPreInput( Aabb* lastOutput ) const    = 0;
    M_DEVICE_HOST virtual Aabb nextTransformPastInput( Aabb* lastOutput ) const   = 0;
    M_DEVICE_HOST virtual Aabb nextTransformInterpolate( Aabb* lastOutput ) const = 0;
    M_DEVICE_HOST virtual Aabb nextInputPreTransform() const                      = 0;
    M_DEVICE_HOST virtual Aabb nextInputPastTransform() const                     = 0;
    M_DEVICE_HOST virtual Aabb nextInputInterpolate( Aabb* lastOutput ) const     = 0;
    M_DEVICE_HOST virtual Aabb nextMergeKeys( Aabb* lastOutput ) const            = 0;

private:
    M_DEVICE_HOST void advanceTransformKey();
    M_DEVICE_HOST void advanceInputKey();

protected:
    // Marching state
    const MotionOptions& motionOptions;
    const MotionAabb&    inputMaabb;
    const float          transformIntervalTime;
    unsigned int         nextTransformKey;
    unsigned int         nextInputKey;
    float                prevTransformKeyTime;
    float                nextTransformKeyTime;
    float                nextInputKeyTime;
    float                prevKeyTime;
    bool                 nextKeyIsTransformKey, prevKeyWasTransformKey, prevKeyWasInputKey;
};

//////////////////////////////////////////////////////////////////////////

class SRTIntervalComputer : public AabbIntervalComputerBase
{
public:
    M_DEVICE_HOST SRTIntervalComputer( const SRTData* _transforms, const MotionOptions& motionOptions, const MotionAabb& inputMaabb );

protected:
    M_DEVICE_HOST Aabb nextTransformPreInput( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextTransformPastInput( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextTransformInterpolate( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextInputPreTransform() const override final;
    M_DEVICE_HOST Aabb nextInputPastTransform() const override final;
    M_DEVICE_HOST Aabb nextInputInterpolate( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextMergeKeys( Aabb* lastOutput ) const override final;

private:
    M_DEVICE_HOST const SRTData& getKeySrt( unsigned int i ) const { return transforms[i]; }
    // helper function for the merge cases
    // inAabb0 is computed if required
    // inAabb1 is provided as parameter and is assumed to be extrapolated to nextTransformKeyTime
    // key0, key1 are always the previous/next transform keys as transformSrtAabb() always requires these inputs
    M_DEVICE_HOST Aabb handleIntervalSRTTransform( const Aabb& inAabb1, Aabb* lastOutput ) const;

private:
    const SRTData* transforms;
};

//////////////////////////////////////////////////////////////////////////

class MatrixAabbIntervalComputer : public AabbIntervalComputerBase
{
public:
    M_DEVICE_HOST MatrixAabbIntervalComputer( const float* _transforms, const MotionOptions& motionOptions, const MotionAabb& inputMaabb );

protected:
    M_DEVICE_HOST Aabb nextTransformPreInput( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextTransformPastInput( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextTransformInterpolate( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextInputPreTransform() const override final;
    M_DEVICE_HOST Aabb nextInputPastTransform() const override final;
    M_DEVICE_HOST Aabb nextInputInterpolate( Aabb* lastOutput ) const override final;
    M_DEVICE_HOST Aabb nextMergeKeys( Aabb* lastOutput ) const override final;

private:
    M_DEVICE_HOST const float* getKeyMatrix( unsigned int keyIndex ) const { return &transforms[keyIndex * 12]; }
    M_DEVICE_HOST void         lerpMatrixTransformKeys( unsigned int index_key0, unsigned int index_key1, float t, Matrix3x4& interpolatedTransform ) const;

    // helper function  for the cases below
    // inAabb0, key0 depend on whether the previous key was a transform or an input key
    // inAabb1, key1 are provided as parameters
    M_DEVICE_HOST Aabb handleIntervalMatrixTransform( const Aabb& inAabb1, const float* key1, Aabb* lastOutput ) const;

private:
    const float* transforms;
};

#ifdef __LWDACC__
} // namespace lwca
#endif

}  // namespace motion
}  // namespace optix_exp

#include "motionImpl.hpp"
