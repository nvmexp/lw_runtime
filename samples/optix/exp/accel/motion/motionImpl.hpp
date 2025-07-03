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
#include <type_traits>
#include <vector_functions.h>
#include <vector_types.h>

#include "matrixTransformImpl.hpp"

//#define DUMMY_SRT_IMPL

#ifndef DUMMY_SRT_IMPL
#include "srtTransformImpl.hpp"
#endif

namespace optix_exp {
namespace motion {

namespace {

//////////////////////////////////////////////////////////////////////////
#ifdef DUMMY_SRT_IMPL
namespace {
M_DEVICE_HOST __inline__ SRTData lerp( const SRTData& key0, const SRTData& key1, float t )
{
    return key0;
}
M_DEVICE_HOST __inline__ Aabb transform( const Aabb& aabb, const SRTData& key )
{
    return aabb;
}

M_DEVICE_HOST __inline__ void transform( const Aabb& aabb, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
}

M_DEVICE_HOST __inline__ void transform( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
}

M_DEVICE_HOST void srtToMatrix( const SRTData& srt, float* m )
{
}
}
#endif
#undef DUMMY_SRT_IMPL
//////////////////////////////////////////////////////////////////////////

M_DEVICE_HOST __inline__ bool keysAlignTrivially( const MotionOptions& motionOptions, const MotionAabb& mAabb )
{
    return motionOptions.numKeys < 2 || mAabb.isStatic()
        || mAabb.keysAlign( motionOptions.timeBegin, motionOptions.timeEnd, motionOptions.numKeys );
}

}

#ifdef __LWDACC__
namespace lwca {
#endif

M_DEVICE_HOST AabbIntervalComputerBase::AabbIntervalComputerBase( const MotionOptions& _motionOptions, const MotionAabb& _inputMaabb )
    : motionOptions( _motionOptions )
    , inputMaabb( _inputMaabb )
    , transformIntervalTime( ( motionOptions.timeEnd - motionOptions.timeBegin ) / ( motionOptions.numKeys - 1 ) )
    , nextTransformKey( 0 )
    , nextInputKey( 0 )
    , prevTransformKeyTime( motionOptions.timeBegin )
    , nextTransformKeyTime( motionOptions.timeBegin )
    , nextInputKeyTime( inputMaabb.timeFirstKey() )
{
}

M_DEVICE_HOST void AabbIntervalComputerBase::advanceTransformKey()
{
    nextTransformKey++;
    prevKeyTime = prevTransformKeyTime = nextTransformKeyTime;
    nextTransformKeyTime               = nextTransformKey < motionOptions.numKeys
                                            ? lerp( motionOptions.timeBegin, motionOptions.timeEnd, float( nextTransformKey ) / ( motionOptions.numKeys - 1u ) )
                                            : FLT_MAX;
}
M_DEVICE_HOST void AabbIntervalComputerBase::advanceInputKey()
{
    nextInputKey++;
    prevKeyTime      = nextInputKeyTime;
    nextInputKeyTime = nextInputKey < inputMaabb.keyCount()
                        ? inputMaabb.keyTime( nextInputKey )
                        : FLT_MAX;
}

M_DEVICE_HOST void AabbIntervalComputerBase::march( MotionAabb &outAabb )
{
    unsigned int numKeys      = motionOptions.numKeys;
    float        timeBegin    = motionOptions.timeBegin;
    float        timeEnd      = motionOptions.timeEnd;
    bool         tStartVanish = motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH;
    bool         tEndVanish   = motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH;

    // while the keys may not align perfectly, they may still create a regular distribution in the end
    // do a simple / partial test for this case here
    const bool outputKeysRegularlyDistributed =
        inputMaabb.keysAreRegularlyDistributed() && timeBegin == inputMaabb.timeFirstKey() && timeEnd == inputMaabb.timeLastKey()
        && ( ( ::max( numKeys, inputMaabb.keyCount() ) - 1 ) % ( ::min( numKeys, inputMaabb.keyCount() ) - 1 ) == 0 );

    // if either the transform or the child has the vanish flag set, nothing is visible in the out-of-borders case
    // hence, we pass this information along
    outAabb.setMotionFlagStartVanish( inputMaabb.motionFlagStartVanish() || tStartVanish );
    outAabb.setMotionFlagEndVanish( inputMaabb.motionFlagEndVanish() || tEndVanish );

    // scale epsilon based on time range of input / transform
    const float keysConsideredAlignedEpsilon =
        0.00001f * ::max( inputMaabb.timeLastKey() - inputMaabb.timeFirstKey(), timeEnd - timeBegin );

    // "auto detect" of the output keys are regularly distributed
    // will be set once we have the first interval (i.e., on the second key)
    // all further intervals will be checked against this interval
    // auto detection is not needed if it was previously "decided" (detected) that output is regular
    bool  mAabbHasRegularDistribution = true;
    float detectedIntervalTime        = 0.0f;

    if( inputMaabb.motionFlagStartVanish() )
    {
#ifdef __LWDACC__
#pragma unroll 1
#endif
        while( ( nextTransformKeyTime + keysConsideredAlignedEpsilon ) < nextInputKeyTime )
        {
            advanceTransformKey();
            prevKeyWasTransformKey = true;
            prevKeyWasInputKey     = false;
        }
    }
    if( tStartVanish )
    {
#ifdef __LWDACC__
#pragma unroll 1
#endif
        while( ( nextInputKeyTime + keysConsideredAlignedEpsilon ) < nextTransformKeyTime )
        {
            advanceInputKey();
            prevKeyWasTransformKey = false;
            prevKeyWasInputKey     = true;
        }
    }

    outAabb.initIrregularDistribution( ::min(nextTransformKeyTime, nextInputKeyTime) );

    auto getLastComputedAabbIfExists = [](MotionAabb& mAabb) -> Aabb*
    {
        return ( mAabb.keyCount() == 0 ) ? nullptr : &mAabb.aabb( mAabb.keyCount() - 1 );
    };

#ifdef __LWDACC__
#pragma unroll 1
#endif
    while( nextInputKey < inputMaabb.keyCount() || nextTransformKey < numKeys )
    {
        assert( nextTransformKeyTime != FLT_MAX
            || nextInputKeyTime != FLT_MAX );
        nextKeyIsTransformKey = nextTransformKeyTime <= nextInputKeyTime;

        const float nextKeyTime = nextKeyIsTransformKey ? nextTransformKeyTime : nextInputKeyTime;
        if( !outputKeysRegularlyDistributed )
        {
            if( outAabb.keyCount() == 1 )
            {
                detectedIntervalTime = nextKeyTime - outAabb.timeFirstKey();
            }
            else if( outAabb.keyCount() > 1 )
            {
                mAabbHasRegularDistribution &=
                    abs( detectedIntervalTime - ( nextKeyTime - prevKeyTime ) ) <= keysConsideredAlignedEpsilon;
            }
        }

        if( nextKeyIsTransformKey )
        {
            assert( nextTransformKey <= numKeys - 1 );
            assert( nextInputKey <= inputMaabb.keyCount() );
            const bool mergeKey = nextInputKeyTime - nextTransformKeyTime <= keysConsideredAlignedEpsilon;
            if( mergeKey )
            {
                // the input key coming after the next TransformKey is less than epsilon time away... let's merge them!
                outAabb.pushIrregularKey( nextKeyTime, nextMergeKeys( getLastComputedAabbIfExists(outAabb) ) );
                advanceInputKey();
            }
            // first input, nothing to interpolate
            else if( nextInputKey == 0 )
            {
                outAabb.pushIrregularKey( nextKeyTime, nextTransformPreInput( getLastComputedAabbIfExists(outAabb) ) );
            }
            // only one left... use that one!
            else if( nextInputKey == inputMaabb.keyCount() )
            {
                outAabb.pushIrregularKey( nextKeyTime, nextTransformPastInput( getLastComputedAabbIfExists(outAabb) ) );
            }
            else
            {
                outAabb.pushIrregularKey( nextKeyTime, nextTransformInterpolate( getLastComputedAabbIfExists(outAabb) ) );
            }
            advanceTransformKey();
            prevKeyWasTransformKey = true;
            prevKeyWasInputKey     = mergeKey;
            // we are done if we vanish is set afterwards!
            if( nextTransformKey == numKeys && tEndVanish )
                break;
        }
        else
        {
            assert( nextInputKey <= inputMaabb.keyCount() - 1 );
            assert( nextTransformKey <= numKeys );
            const bool mergeKey = nextTransformKeyTime - nextInputKeyTime < keysConsideredAlignedEpsilon;
            if( mergeKey )
            {
                // the transform key coming after the next input key is less than epsilon time away... let's merge them!
                outAabb.pushIrregularKey( nextKeyTime, nextMergeKeys( getLastComputedAabbIfExists(outAabb) ) );
                advanceTransformKey();
            }
            // first input, nothing to interpolate
            else if( nextTransformKey == 0 )
            {
                outAabb.pushIrregularKey( nextKeyTime, nextInputPreTransform() );
            }
            else if( nextTransformKey == numKeys )
            {
                outAabb.pushIrregularKey( nextKeyTime, nextInputPastTransform() );
            }
            else
            {
                outAabb.pushIrregularKey( nextKeyTime, nextInputInterpolate( getLastComputedAabbIfExists(outAabb) ) );
            }
            advanceInputKey();
            prevKeyWasInputKey     = true;
            prevKeyWasTransformKey = mergeKey;
            // we are done if we vanish is set afterwards!
            if( nextInputKey == inputMaabb.keyCount() && inputMaabb.motionFlagEndVanish() )
                break;
        }
    }

    if( mAabbHasRegularDistribution )
        outAabb.markRegularDistribution();

    // if we are left with one key only, something went wrong..
    // can only happen, if VANISH clamp mode is used on transform as well as input and they have a single aligning key
    if( outAabb.keyCount() <= 1 )
        outAabb.setIlwalid();
}

//////////////////////////////////////////////////////////////////////////

M_DEVICE_HOST SRTIntervalComputer::SRTIntervalComputer( const SRTData* _transforms, const MotionOptions& motionOptions, const MotionAabb& inputMaabb )
    : AabbIntervalComputerBase( motionOptions, inputMaabb )
    , transforms( _transforms )
{
}

// static input
M_DEVICE_HOST Aabb SRTIntervalComputer::nextTransformPreInput( Aabb* lastOutput ) const
{
    assert( nextInputKey == 0 );
    // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
    // the same is true if we are outputting the first key, i.e., there is no interval to consider
    // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
    if( nextTransformKey == 0 || lastOutput == nullptr )
        // static transform
        return transform( inputMaabb.aabb( nextInputKey ), getKeySrt( nextTransformKey ) );
    else
    {
        Aabb outAabb0, outAabb1;
        transform( inputMaabb.aabb( nextInputKey ), getKeySrt( nextTransformKey - 1 ), getKeySrt( nextTransformKey ),
                   outAabb0, outAabb1 );
        include( *lastOutput, outAabb0 );
        return outAabb1;
    }
}
// static input
M_DEVICE_HOST Aabb SRTIntervalComputer::nextTransformPastInput( Aabb* lastOutput ) const
{
    assert( nextInputKey == inputMaabb.keyCount() );
    // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
    // the same is true if we are outputting the first key, i.e., there is no interval to consider
    // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
    // HOWEVER, this is just a safe guard here as we this function assumes we have process all input keys, i.e.,
    //  we must have created output aabbs when processing them!
    if( nextTransformKey == 0 || lastOutput == nullptr )
        // static transform
        return transform( inputMaabb.aabb( nextInputKey - 1 ), getKeySrt( nextTransformKey ) );
    else
    {
        // prev key may have been an input key, hence, we have an interval from last input key to next srt key
        // if(prevKeyWasTransformKey) srt_interval_t0 = 0.0f;
        const float srt_interval_t0 = ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime;
        Aabb        outAabb0, outAabb1;
        transform( inputMaabb.aabb( nextInputKey - 1 ),
                   lerp( getKeySrt( nextTransformKey - 1 ), getKeySrt( nextTransformKey ), srt_interval_t0), getKeySrt( nextTransformKey ),
                   outAabb0, outAabb1 );
        include( *lastOutput, outAabb0 );
        return outAabb1;
    }
}
// static transform
M_DEVICE_HOST Aabb SRTIntervalComputer::nextInputPreTransform() const
{
    assert( nextTransformKey == 0 );
    return transform( inputMaabb.aabb( nextInputKey ), getKeySrt( nextTransformKey ) );
}
// static transform
M_DEVICE_HOST Aabb SRTIntervalComputer::nextInputPastTransform() const
{
    assert( nextTransformKey == motionOptions.numKeys );
    return transform( inputMaabb.aabb( nextInputKey ), getKeySrt( nextTransformKey - 1 ) );
}

M_DEVICE_HOST Aabb SRTIntervalComputer::nextTransformInterpolate( Aabb* lastOutput ) const
{
    assert( nextInputKey > 0 || nextInputKey < inputMaabb.keyCount() );
    // interpolate input "keys" (aabbs) for t=nextTransformKeyTime
    const Aabb inAabb1 = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, nextTransformKeyTime );
    return handleIntervalSRTTransform( inAabb1, lastOutput );
}
M_DEVICE_HOST Aabb SRTIntervalComputer::nextInputInterpolate( Aabb* lastOutput ) const
{
    // next key is input key
    // since we do not interpolate SRTs, but adjust times, the behavior is the same as for merging!
    return handleIntervalSRTTransform( inputMaabb.aabb( nextInputKey ), lastOutput );
}
M_DEVICE_HOST Aabb SRTIntervalComputer::nextMergeKeys( Aabb* lastOutput ) const
{
    // merge case... next transform/input keys align
    // next key is input key, i.e., pass it as inAabb1
    return handleIntervalSRTTransform( inputMaabb.aabb( nextInputKey ), lastOutput );
}

M_DEVICE_HOST Aabb SRTIntervalComputer::handleIntervalSRTTransform( const Aabb& inAabb1, Aabb* lastOutput ) const
{
    // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
    // the same is true if we are outputting the first key, i.e., there is no interval to consider
    // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
    if( nextTransformKey == 0 || lastOutput == nullptr )
        // static transform
        // key1 == get_key(nextTransformKey)
        return transform( inAabb1, getKeySrt( nextTransformKey ) );
    else if( nextInputKey == 0 )
    {
        // static input
        Aabb outAabb0, outAabb1;
        transform( inAabb1, getKeySrt( nextTransformKey - 1 ), getKeySrt( nextTransformKey ), outAabb0, outAabb1 );
        include( *lastOutput, outAabb0 );
        return outAabb1;
    }
    else
    {


        Aabb inAabb0 = prevKeyWasInputKey ? inputMaabb.aabb( nextInputKey - 1 )
                                          : inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, prevKeyTime );
        SRTData srt0 = prevKeyWasTransformKey ? getKeySrt( nextTransformKey - 1 )
                                                   : lerp( getKeySrt( nextTransformKey - 1 ), getKeySrt( nextTransformKey ),
                                                         ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime );
        SRTData srt1 = nextKeyIsTransformKey ? getKeySrt( nextTransformKey )
                                                  : lerp( getKeySrt( nextTransformKey - 1 ), getKeySrt( nextTransformKey ),
                                                        ( nextInputKeyTime - prevTransformKeyTime ) / transformIntervalTime );

        Aabb outAabb0, outAabb1;
        transform( inAabb0, inAabb1, srt0, srt1, outAabb0, outAabb1 );
        include( *lastOutput, outAabb0 );
        return outAabb1;
    }
}

//////////////////////////////////////////////////////////////////////////

M_DEVICE_HOST MatrixAabbIntervalComputer::MatrixAabbIntervalComputer( const float* _transforms, const MotionOptions& motionOptions, const MotionAabb& inputMaabb )
    : AabbIntervalComputerBase( motionOptions, inputMaabb )
    , transforms( _transforms )
{
}

// if the input OR the transform is static for an interval, we can simply transform at the output key, interpolation from the previous/to the next aabb will be correct
// otherwise, we need to compute bounds for the interval and update the previous aabb
// static input
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextTransformPreInput( Aabb* /*lastOutput*/ ) const
{
    assert( nextInputKey == 0 );
    return transform( inputMaabb.aabb( nextInputKey ), getKeyMatrix( nextTransformKey ) );
}
// static input
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextTransformPastInput( Aabb* /*lastOutput*/ ) const
{
    assert( nextInputKey == inputMaabb.keyCount() );
    return transform( inputMaabb.aabb( nextInputKey - 1 ), getKeyMatrix( nextTransformKey ) );
}

// static transform
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextInputPreTransform() const
{
    assert( nextTransformKey == 0 );
    return transform( inputMaabb.aabb( nextInputKey ), getKeyMatrix( nextTransformKey ) );
}
// static transform
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextInputPastTransform() const
{
    assert( nextTransformKey == motionOptions.numKeys );
    return transform( inputMaabb.aabb( nextInputKey ), getKeyMatrix( nextTransformKey - 1 ) );
}

M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextMergeKeys( Aabb* lastOutput ) const
{
    // merge case... next transform/input keys align
    // next key is input key, i.e., pass it as inAabb1
    // next key is transform key, i.e., pass it as key1
    return handleIntervalMatrixTransform( inputMaabb.aabb( nextInputKey ), getKeyMatrix( nextTransformKey ), lastOutput );
}
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextTransformInterpolate( Aabb* lastOutput ) const
{
    assert( nextInputKey > 0 || nextInputKey < inputMaabb.keyCount() );
    // interpolate input "keys" (aabbs) for t=nextTransformKeyTime
    const Aabb inAabb1 = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, nextTransformKeyTime );
    // next key is transform key, i.e., pass it as key1
    return handleIntervalMatrixTransform( inAabb1, getKeyMatrix( nextTransformKey ), lastOutput );
}
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::nextInputInterpolate( Aabb* lastOutput ) const
{
    assert( nextTransformKey > 0 || nextTransformKey < motionOptions.numKeys );
    // next key is input key, i.e., pass it as inAabb1
    const Aabb& inAabb1 = inputMaabb.aabb( nextInputKey );
    // interpolate key1
    Matrix3x4 key1;
    const float t = ( nextInputKeyTime - prevTransformKeyTime ) / transformIntervalTime;
    lerpMatrixTransformKeys( nextTransformKey - 1, nextTransformKey, t, key1 );
    return handleIntervalMatrixTransform( inAabb1, key1.m, lastOutput );
}

M_DEVICE_HOST void MatrixAabbIntervalComputer::lerpMatrixTransformKeys( unsigned int index_key0, unsigned int index_key1, float t, Matrix3x4& interpolatedTransform ) const
{
    const Matrix3x4& key0 = *(const Matrix3x4*)getKeyMatrix( index_key0 );
    const Matrix3x4& key1 = *(const Matrix3x4*)getKeyMatrix( index_key1 );
    interpolatedTransform = lerp( key0, key1, t );
}

// helper function  for the cases below
// inAabb0, key0 depend on whether the previous key was a transform or an input key
// inAabb1, key1 are provided as parameters
M_DEVICE_HOST Aabb MatrixAabbIntervalComputer::handleIntervalMatrixTransform( const Aabb& inAabb1, const float* key1, Aabb* lastOutput ) const
{
    // if we are processing the first SRT key, we do not need to consider an interval, just transform the key
    // the same is true if we are outputting the first key, i.e., there is no interval to consider
    // note that due to vanish border mode handling, we may process the first SRT key with nextTransformKey != 0
    if( nextTransformKey == 0 || nextInputKey == 0 || lastOutput == nullptr )
        return transform( inAabb1, key1 );
    else
    {
        const float* key0;
        Matrix3x4 interpolatedTransform;
        if( prevKeyWasTransformKey )
        {
            key0 = getKeyMatrix( nextTransformKey - 1 );
        }
        else
        {
            const float t = ( prevKeyTime - prevTransformKeyTime ) / transformIntervalTime;
            lerpMatrixTransformKeys( nextTransformKey - 1, nextTransformKey, t, interpolatedTransform );
            key0 = interpolatedTransform.m;
        }
        const Aabb* pinAabb0;
        Aabb        inAabb0;
        if( prevKeyWasInputKey )
        {
            pinAabb0 = &inputMaabb.aabb( nextInputKey - 1 );
        }
        else
        {
            inAabb0  = inputMaabb.interpolateKeys( nextInputKey - 1, nextInputKey, prevKeyTime );
            pinAabb0 = &inAabb0;
        }
        Aabb outAabb0, outAabb1;
        transform( *pinAabb0, inAabb1, key0, key1, outAabb0, outAabb1 );
        include( *lastOutput, outAabb0 );
        return outAabb1;
    }
}

#ifdef __LWDACC__
} // namespace lwca
#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

#ifdef __LWDACC__
using namespace lwca;
#endif

M_DEVICE_HOST static void applyMatrixStaticTransform( const float* matrixTransform, MotionAabb& inAabb, MotionAabb& outAabb )
{
    outAabb.swapDistribution( inAabb ); // swap everything except the aabbs. we still want to read from the input aabbs and write to the output aabbs.
#ifdef __LWDACC__
#pragma unroll 1
#endif
    for( unsigned int i = 0; i < outAabb.keyCount(); ++i )
        outAabb.aabb( i ) = transform( inAabb.aabb( i ), matrixTransform );
}

M_DEVICE_HOST static void applyMatrixMotionTransform( const float* matrixTransforms, const MotionOptions& motionOptions, MotionAabb& inAabb, MotionAabb& outAabb )
{
    if( !inAabb.isValid() )
    {
        outAabb.setIlwalid();
        // error handling?
        return;
    }
    else if( motionOptions.numKeys < 2 )
    {
        applyMatrixStaticTransform( matrixTransforms, inAabb, outAabb );
    }
    else if( inAabb.isStatic() )
    {
        outAabb.setMotionFlagStartVanish( motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH );
        outAabb.setMotionFlagEndVanish( motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH );
        outAabb.initRegularDistribution( motionOptions.timeBegin, motionOptions.timeEnd, motionOptions.numKeys );
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( unsigned int i = 0; i < motionOptions.numKeys; ++i )
            outAabb.aabb( i ) = transform( inAabb.aabb( 0 ), &matrixTransforms[i * 12] );
    }
    // true if one of the inputs (mAabb or transform) is static or the keys actually align
    else if( keysAlignTrivially( motionOptions, inAabb ) )
    {
        assert( motionOptions.numKeys > 1 && !inAabb.isStatic() );

        outAabb.setMotionFlagStartVanish( inAabb.motionFlagStartVanish() || ( motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH ) );
        outAabb.setMotionFlagEndVanish( inAabb.motionFlagEndVanish() || ( motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH ) );
        outAabb.initRegularDistribution( motionOptions.timeBegin, motionOptions.timeEnd, motionOptions.numKeys );

        Aabb mAabb;
        ilwalidate( mAabb );

        unsigned int i = 1;
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( ; i < motionOptions.numKeys; ++i )
        {
            // interval [i-1,i] with motion input (2 linearly interpolated aabbs), 2 transform keys (linearly interpolated)
            // this operation is nonlinear, i.e., we cannot just apply the first transform on the first aabb
            // and the second transform on the second aabb to get an interpolatable result that bounds M(t)*aabb(t)
            // this function returns valid bounds at i-1, i for the interval
            // these bounds may be larger than previously computed bounds at i-1, hence, we need to update them!
            Aabb mAabb0, mAabb1;
            transform( inAabb.aabb( i - 1 ), inAabb.aabb( i ), &matrixTransforms[( i - 1 ) * 12],
                       &matrixTransforms[i * 12], mAabb0, mAabb1 );
            // update previous output key to include the required mAabb0 for this interval
            include( mAabb, mAabb0 );
            outAabb.aabb( i - 1 ) = mAabb;

            mAabb = mAabb1;
        }

        outAabb.aabb( i - 1 ) = mAabb;
    }
    else
    {
        MatrixAabbIntervalComputer irregularKeysHandler( matrixTransforms, motionOptions, inAabb );
        irregularKeysHandler.march( outAabb );
    }
}

M_DEVICE_HOST static void applySrtMotionTransform( const SRTData* srtTransforms, const MotionOptions& motionOptions, MotionAabb& inAabb, MotionAabb& outAabb )
{
    if( !inAabb.isValid() )
    {
        outAabb.setIlwalid();
        // error handling?
    }
    else if( motionOptions.numKeys < 2 )
    {
        float m[12];
        srtToMatrix( srtTransforms[0], m );
        applyMatrixStaticTransform( m, inAabb, outAabb );
    }
    else if( inAabb.isStatic() )
    {
        outAabb.setMotionFlagStartVanish( motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH );
        outAabb.setMotionFlagEndVanish( motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH );
        outAabb.initRegularDistribution( motionOptions.timeBegin, motionOptions.timeEnd, motionOptions.numKeys );

        Aabb mAabb;
        ilwalidate( mAabb );

        unsigned int i = 1;
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( ; i < motionOptions.numKeys; ++i )
        {
            Aabb mAabb0, mAabb1;
            transform( inAabb.aabb( 0 ), srtTransforms[i - 1], srtTransforms[i], mAabb0, mAabb1 );

            // update previous output key to include the required mAabb0 for this interval
            include( mAabb, mAabb0 );
            outAabb.aabb( i - 1 ) = mAabb;

            mAabb = mAabb1;
        }

        outAabb.aabb( i - 1 ) = mAabb;
    }
    else if( keysAlignTrivially( motionOptions, inAabb ) )
    {
        assert( motionOptions.numKeys > 1 && !inAabb.isStatic() );

        outAabb.setMotionFlagStartVanish( inAabb.motionFlagStartVanish() || ( motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH ) );
        outAabb.setMotionFlagEndVanish( inAabb.motionFlagEndVanish() || ( motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH ) );
        outAabb.initRegularDistribution( motionOptions.timeBegin, motionOptions.timeEnd, motionOptions.numKeys );

        Aabb mAabb;
        ilwalidate( mAabb );

        unsigned int i = 1;
#ifdef __LWDACC__
#pragma unroll 1
#endif
        for( ; i < motionOptions.numKeys; ++i )
        {
            Aabb mAabb0, mAabb1;
            transform( inAabb.aabb( i - 1 ), inAabb.aabb( i ), srtTransforms[i - 1], srtTransforms[i], mAabb0, mAabb1 );
            // update previous output key to include the required mAabb0 for this interval
            include( mAabb, mAabb0 );
            outAabb.aabb( i - 1 ) = mAabb;

            mAabb = mAabb1;
        }

        outAabb.aabb( i - 1 ) = mAabb;
    }
    else
    {
        SRTIntervalComputer irregularKeysHandler( srtTransforms, motionOptions, inAabb   );
        irregularKeysHandler.march( outAabb );
    }
}

}  // namespace motion
}  // namespace optix_exp
