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

M_DEVICE_HOST __inline__ void transformInOut0Out1( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& inOutAabb0, Aabb& outAabb1 )
{
}

M_DEVICE_HOST __inline__ void transformInOut0InOut1( const Aabb& aabb0, const Aabb& aabb1, const SRTData& key0, const SRTData& key1, Aabb& inOutAabb0, Aabb& inOutAabb1 )
{
}
}
#endif
#undef DUMMY_SRT_IMPL
//////////////////////////////////////////////////////////////////////////

#ifdef __LWDACC__
namespace lwca {
#endif

template <typename TransformData>
M_DEVICE_HOST inline void transformInputStatic( const Aabb& aabb, const TransformData& key0, const TransformData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
}

template <>
M_DEVICE_HOST inline void transformInputStatic<Matrix3x4>( const Aabb& aabb, const Matrix3x4& key0, const Matrix3x4& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    outAabb0 = transform( aabb, key0 );
    outAabb1 = transform( aabb, key1 );
}

template <>
M_DEVICE_HOST inline void transformInputStatic<SRTData>( const Aabb& aabb, const SRTData& key0, const SRTData& key1, Aabb& outAabb0, Aabb& outAabb1 )
{
    transform( aabb, key0, key1, outAabb0, outAabb1 );
}


template <typename TransformData>
M_DEVICE_HOST void processMotionTransform( const MotionOptions& instanceMO,
                                           unsigned int         intervalIdx,
                                           float                intervalBegin,
                                           float                intervalEnd,
                                           const unsigned int   aabbArrayMaxSize,
                                           AabbArray*&          pInAabbs,
                                           AabbArray*&          pOutAabbs,
                                           const MotionOptions& transformMO,
                                           unsigned int&        transformLastKeyIndex,
                                           TransformData*       transformData )
{
    AabbArray& inAabbs  = *pInAabbs;
    AabbArray& outAabbs = *pOutAabbs;

    auto getTransform = [&transformData]( unsigned index ) -> TransformData& {
        return transformData[index];
    };

    if( transformMO.numKeys == 1 )
    {
        for( unsigned int k = 0; k < ( inAabbs.size - 1 ); ++k )
        {
            outAabbs.aabbs[k] = transform( inAabbs.aabbs[k], getTransform( 0 ) );
        }
        return;
    }

    auto twoTransformKeys = [&]( TransformData& intervalBeginTransform, TransformData& intervalEndTransform )
    {
        if( inAabbs.size == 1 )
        {
            outAabbs.size = 2;
            transformInputStatic( inAabbs.aabbs[0],
                                  intervalBeginTransform, intervalEndTransform,
                                  outAabbs.aabbs[0], outAabbs.aabbs[1] );
        }
        else if( inAabbs.size == 2 )
        {
            transform( inAabbs.aabbs[0], inAabbs.aabbs[1],
                       intervalBeginTransform, intervalEndTransform,
                       outAabbs.aabbs[0], outAabbs.aabbs[1] );
        }
        else
        {
            TransformData lerpedMat0;
            TransformData lerpedMat1;
            TransformData& inmat0 = intervalBeginTransform;
            TransformData& inmat1 = intervalEndTransform;
            TransformData* mat0 = &inmat0;
            TransformData* mat1 = &lerpedMat0;
            // below we use inout for the result aabb 0, hence, we need to initialize outAabb 0
            outAabbs.aabbs[0] = transform( inAabbs.aabbs[0], inmat0 );
            for( unsigned int k = 0; k < (inAabbs.size - 1); ++k )
            {
                if( k == inAabbs.size - 2 )
                    mat1 = &inmat1;
                else
                    *mat1 = lerp( inmat0, inmat1, (float)( k+1 ) / (inAabbs.size - 1) );

                transformInOut0Out1( inAabbs.aabbs[k], inAabbs.aabbs[k+1], *mat0, *mat1, outAabbs.aabbs[k], outAabbs.aabbs[k+1] );

                if( k == 0 )
                    mat0 = &lerpedMat1;
                swap( mat0, mat1 );
            }
        }
    };

    if( transformMO.triviallyAligns( instanceMO ) )
    {
        twoTransformKeys( getTransform( intervalIdx ), getTransform( intervalIdx + 1 ) );
        // not needed since this transform will always hit this case, but to make this consistent
        transformLastKeyIndex++;
    }
    else if( transformMO.timesAlign( instanceMO ) && ( ( instanceMO.numKeys - 1 ) % ( transformMO.numKeys - 1 ) == 0 ) )
    {
        unsigned int ratio = ( instanceMO.numKeys - 1 ) / ( transformMO.numKeys - 1 );
        TransformData lerpedMat0;
        TransformData lerpedMat1;
        TransformData& inmat0 = getTransform( transformLastKeyIndex );
        TransformData& inmat1 = getTransform( transformLastKeyIndex + 1 );
        TransformData* mat0 = &inmat0;
        TransformData* mat1 = &inmat1;
        if( intervalIdx % ratio != 0 )
        {
            lerpedMat0 = lerp( inmat0, inmat1,
                ( intervalBegin - transformMO.timeAtKey( transformLastKeyIndex ) ) / transformMO.intervalSize() );
            mat0 = &lerpedMat0;
        }
        if( ( intervalIdx+1 ) % ratio != 0 )
        {
            lerpedMat1 = lerp( inmat0, inmat1,
                ( intervalEnd - transformMO.timeAtKey( transformLastKeyIndex + 1 ) ) / transformMO.intervalSize() );
            mat1 = &lerpedMat1;
        }
        // ias build has more keys, i.e., only one or none of the aabbs align with a transform key
        twoTransformKeys( *mat0, *mat1 );

        if( ( intervalIdx+1 ) % ratio == 0 )
            transformLastKeyIndex++;
    }
    else
    {
        // upsample input iff not yet at full resolution

        if( inAabbs.size != aabbArrayMaxSize )
        {
            // upsample to aabbArrayMaxSize
            // we could do that in place, but it is easier to not do so (and we have two AabbArrays around anyways)
            //assert( ( inAabbs.aabbArrayMaxSize - 1 ) % ( inAabbs.size - 1 ) == 0 );
            outAabbs.size = aabbArrayMaxSize;
            if( inAabbs.size == 1 )
            {
                for( unsigned int k = 0; k < aabbArrayMaxSize; ++k )
                {
                    outAabbs.aabbs[k] = inAabbs.aabbs[0];
                }
            }
            else
            {
                unsigned int ratio = ( outAabbs.size - 1 ) / ( inAabbs.size - 1 );
                for( unsigned int k = 0; k < outAabbs.size; ++k )
                {
                    if( k % ratio == 0 )
                        outAabbs.aabbs[k] = inAabbs.aabbs[k / ratio];
                    else
                        outAabbs.aabbs[k] = lerp( inAabbs.aabbs[k / ratio],
                            inAabbs.aabbs[k / ratio + 1], fract( (float)k / ratio ) );
                }
            }

            swap( pInAabbs, pOutAabbs );
        }

        // use *pInAabbs, *pOutAabbs instead of inAabbs/outAabbs due to the upsampling that may switch the two
        AabbIntervalComputer marcher( transformMO, intervalBegin, intervalEnd, transformLastKeyIndex, *pInAabbs, *pOutAabbs );
        marcher.march<TransformData>( &getTransform( 0 ) );
        transformLastKeyIndex = marcher.getLwrrentTransformKey();
    }

    //else if( transformMO.timesAlign( instanceMO ) && ( ( transformMO.numKeys - 1 ) % ( instanceMO.numKeys - 1 ) == 0 ) )
    //{
    //    // transform has more keys, i.e., there are multiple transform keys within the current output interval
    //}
    //else {
    //    //const float transformStepSize = transformMO.intervalSize();
    //    //auto transformKeyTime = [&transformMO, &transformStepSize]( unsigned idx ) -> float {
    //    //    return transformMO.timeBegin + idx * transformStepSize;
    //    //};
    //    //// tibegin must be <= intervalBegin
    //    //float tibegin = transformKeyTime( transformLastKeyIndex );
    //    //float tiend   = transformKeyTime( transformLastKeyIndex + 1 );
    //    //float lerpedMat0[12];
    //    //float lerpedMat1[12];


    //    //float* inmat0 = getTransform( transformLastKeyIndex + 0 );
    //    //float* inmat1 = getTransform( transformLastKeyIndex + 1 );

    //    //// TODO
    //    ////lerp( inmat0, inmat1, (float)( k+1 ) / aabbs.size, mat1 );
    //    //float* mat0 = inmat0;
    //    //float* mat1 = lerpedMat0;
    //    //if( inAabbs.size == 1 )
    //    //{
    //    //    inAabbs.size = 2;
    //    //    inAabbs.aabbs[0] = transform( inAabbs.aabbs[0], getTransform( intervalIdx+0 ) );
    //    //    inAabbs.aabbs[1] = transform( inAabbs.aabbs[1], getTransform( intervalIdx+1 ) );
    //    //}
    //    //else if( inAabbs.size == 2 )
    //    //{
    //    //    transform( inAabbs.aabbs[0], inAabbs.aabbs[1], getTransform( intervalIdx+0 ), getTransform( intervalIdx+1 ), inAabbs.aabbs[0], inAabbs.aabbs[1] );
    //    //}
    //    //else
    //    //{
    //    //    Aabb inoutAabb0, inoutAabb1;
    //    //    for( unsigned int k = 0; k < ( inAabbs.size - 1 ); ++k )
    //    //    {
    //    //        lerp( inmat0, inmat1, (float)( k+1 ) / inAabbs.size, mat1 );
    //    //        transform( inAabbs.aabbs[k], inAabbs.aabbs[k+1], mat0, mat1, inoutAabb0, inoutAabb1 );
    //    //        inAabbs.aabbs[k] = inoutAabb0;
    //    //        inoutAabb0 = inoutAabb1;
    //    //        if( k == 0 )
    //    //        {
    //    //            mat0 = lerpedMat1;
    //    //        }
    //    //        else if( k == inAabbs.size - 1 )
    //    //        {
    //    //            mat0 = inmat1;
    //    //        }
    //    //        swap( mat0, mat1 );
    //    //    }
    //    //    inAabbs.aabbs[inAabbs.size - 1] = inoutAabb1;
    //    //}
    //}
}


M_DEVICE_HOST AabbIntervalComputer::AabbIntervalComputer( const MotionOptions& _motionOptions, float _intervalBegin, float _intervalEnd, unsigned _beginTransformKeyIdx, const AabbArray& _inAabbs, AabbArray& _outAabbs )
    : motionOptions( _motionOptions )
    , intervalBegin( _intervalBegin )
    , intervalEnd( _intervalEnd )
    , inAabbs( _inAabbs )
    , outAabbs( _outAabbs )
    , transformIntervalTime( _motionOptions.intervalSize() )
    , nextTransformKey( _beginTransformKeyIdx )
    , inputIntervalTime( ( intervalEnd - intervalBegin ) / ( inAabbs.size - 1 ) )
    , nextInputKey( 0 )
    , prevTransformKeyTime( motionOptions.timeAtKey( _beginTransformKeyIdx ) )
    , nextTransformKeyTime( transformKeyTimeInfinity( nextTransformKey ) )
    , nextInputKeyTime( intervalBegin )
{
}

M_DEVICE_HOST float AabbIntervalComputer::transformKeyTimeInfinity( unsigned int key )
{
    return key < motionOptions.numKeys ? motionOptions.timeBegin + key * transformIntervalTime : FLT_MAX;
}

M_DEVICE_HOST void AabbIntervalComputer::advanceTransformKey()
{
    nextTransformKey++;
    prevTransformKeyTime = nextTransformKeyTime;
    nextTransformKeyTime = transformKeyTimeInfinity( nextTransformKey );
}
M_DEVICE_HOST void AabbIntervalComputer::advanceInputKey()
{
    nextInputKey++;
    nextInputKeyTime = nextInputKey < inAabbs.size
                        ? intervalBegin + nextInputKey * inputIntervalTime
                        : FLT_MAX;
}

template<typename TransformData>
M_DEVICE_HOST void AabbIntervalComputer::march( const TransformData* data )
{
    unsigned int numKeys      = motionOptions.numKeys;
    float        timeBegin    = motionOptions.timeBegin;
    float        timeEnd      = motionOptions.timeEnd;
    //bool         tStartVanish = motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH;
    //bool         tEndVanish   = motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH;

    // scale epsilon based on time range of input / transform
    // FIXME, magic number!
    const float keysConsideredAlignedEpsilon =
        0.00001f * ::max( intervalEnd - intervalBegin, timeEnd - timeBegin );

    // TODO
//    if( tStartVanish )
//    {
//#ifdef __LWDACC__
//#pragma unroll 1
//#endif
//        while( ( nextInputKeyTime + keysConsideredAlignedEpsilon ) < nextTransformKeyTime )
//        {
//            advanceInputKey();
//            prevKeyWasTransformKey = false;
//            prevKeyWasInputKey     = true;
//        }
//    }

    // init output with the transformed aabbs at the target keys
    outAabbs.size = inAabbs.size;

    auto inAabb = [&]( unsigned int i ) -> const Aabb&
    {
        return inAabbs.aabbs[i];
    };
    auto outAabb = [&]( unsigned int i ) -> Aabb&
    {
        return outAabbs.aabbs[i];
    };

    auto transformData = [&]( unsigned int i ) -> const TransformData&
    {
        return data[i];
    };

    auto interpolateTransform = [&]( unsigned int key, float localt )
    {
        if( localt == 0.0f )
            return data[key];
        if( localt == 1.0f )
            return data[key+1];
        return lerp( data[key], data[key+1], localt );
    };

    // Init the output aabbs with the transformed aabbs at the key times of the regularly sampled output aabbs
    // afterwards, we adjust them such that the interpolation of the output aabbs also covers applying the interpolated transforms on the interpolated input aabbs
    {
        float transformLocalTimeValue;
        unsigned int transformKey;
        unsigned i = 0;
        for( ; i < outAabbs.size - 1; ++i )
        {
            motionOptions.keyAtT( intervalBegin + i * inputIntervalTime, transformKey, transformLocalTimeValue );
            outAabb( i ) = transform( inAabb( i ), interpolateTransform( transformKey, transformLocalTimeValue ) );
        }
        // avoid rounding issues? use intervalEnd directly instead of intervalBegin + N * inputIntervalTime
        motionOptions.keyAtT( intervalEnd, transformKey, transformLocalTimeValue );
        outAabb( i ) = transform( inAabb( i ), interpolateTransform( transformKey, transformLocalTimeValue ) );
    }

    //////////////////////////////////////////////////////////////////////////

    // skip all transform keys that are before intervalBegin
    while( intervalBegin - nextTransformKeyTime > keysConsideredAlignedEpsilon )
    {
        advanceTransformKey();
    }

    // skip all transform keys that are before timeBegin, i.e., these have all constant aabbs (clamp transform or vanish)
    while( timeBegin - nextInputKeyTime > keysConsideredAlignedEpsilon )
    {
        advanceInputKey();
    }

    float localInputTimeAtTransformKey;   // wrt. inputs
    Aabb inputAabbAtLastTransformKey;
    Aabb outputAabbAtLastTransformKey;
    TransformData lastTransformAtInputKey;

    while( nextInputKey < outAabbs.size && ( nextInputKey == 0 || nextTransformKey == 0 ) )
    {
        //assert( nextTransformKeyTime != FLT_MAX || nextInputKeyTime != FLT_MAX );
        nextKeyIsTransformKey = nextTransformKeyTime - nextInputKeyTime <= keysConsideredAlignedEpsilon;
        nextKeyIsInputKey     = nextInputKeyTime - nextTransformKeyTime <= keysConsideredAlignedEpsilon;
        const bool mergeKey   = nextKeyIsTransformKey && nextKeyIsInputKey;

        if( mergeKey )
        {
            advanceTransformKey();
            advanceInputKey();
        }
        else if( nextKeyIsTransformKey )
        {
            // local time, i.e., interpolant between the current input keys at time of next transform key
            // nextTransformKeyTime is 'now'
            localInputTimeAtTransformKey = 1.0f - ( ( nextInputKeyTime - nextTransformKeyTime ) / inputIntervalTime );
            // interpolate input to that time
            inputAabbAtLastTransformKey = lerp( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ), localInputTimeAtTransformKey );
            outputAabbAtLastTransformKey = transform( inputAabbAtLastTransformKey, transformData( nextTransformKey ) );
            advanceTransformKey();
        }
        else //if( nextKeyIsInputKey )
        {
            float localTransformTime = 1.0f - ( ( nextTransformKeyTime - nextInputKeyTime ) / transformIntervalTime );
            lastTransformAtInputKey = interpolateTransform( nextTransformKey - 1, localTransformTime );
            advanceInputKey();
        }

        prevKeyWasInputKey = nextKeyIsInputKey;
        prevKeyWasTransformKey = nextKeyIsTransformKey;
    }

#ifdef __LWDACC__
#pragma unroll 1
#endif
    // We are done when we processed all input/output aabbs or we visited all transform keys.
    // The latter is true since all output aabbs were already pre-initialized (input transformed) at the output key times
    while( nextInputKey < outAabbs.size && nextTransformKey < numKeys )
    {
        //assert( nextTransformKeyTime != FLT_MAX || nextInputKeyTime != FLT_MAX );
        nextKeyIsTransformKey = nextTransformKeyTime - nextInputKeyTime <= keysConsideredAlignedEpsilon;
        nextKeyIsInputKey     = nextInputKeyTime - nextTransformKeyTime <= keysConsideredAlignedEpsilon;
        const bool mergeKey   = nextKeyIsTransformKey && nextKeyIsInputKey;

        if( mergeKey )
        {
            if( prevKeyWasInputKey && prevKeyWasTransformKey )
            {
                // simple segment between two input/output keys!
                // FIXME: must not shrink outAabb( nextInputKey - 1 )
                transformInOut0InOut1( inAabb( nextInputKey-1 ), inAabb( nextInputKey ),
                            transformData( nextTransformKey - 1 ), transformData( nextTransformKey ),
                            outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
                // or
                //adjustOut( inAabb( nextInputKey-1 ), inAabb( nextInputKey ),
                //           transf( nextTransformKey - 1 ), transf( nextTransformKey ),
                //           outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
            }
            else if( prevKeyWasInputKey )
            {
                // simple segment between two input/output keys!
                transformInOut0InOut1( inAabb( nextInputKey-1 ), inAabb( nextInputKey ),
                            lastTransformAtInputKey, transformData( nextTransformKey ),
                            outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
                // or
                //adjustOut( inAabb( nextInputKey-1 ), inAabb( nextInputKey ),
                //           transf( nextTransformKey - 1 ), transf( nextTransformKey ),
                //           outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
            }
            else //if( prevKeyWasTransformKey )
            {
                // we do not have the aabb at the time of the previous transform key
                // outputAabbAtLastTransformKey is assumed to be valid here!
                transformInOut0InOut1( inputAabbAtLastTransformKey, inAabb( nextInputKey ),
                            transformData( nextTransformKey - 1 ), transformData( nextTransformKey ),
                            outputAabbAtLastTransformKey, outAabb( nextInputKey ) );

                // Resample outputAabbAtLastTransformKey to interval [nextInputKey-1,nextInputKey]
                // such that the linear transformation of outputAabbAtLastTransformKey -> outAabb( nextInputKey ) with t range [timePrevTransform, timeNextInput]
                // is included in outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                resampleSimpleUniform( outputAabbAtLastTransformKey, localInputTimeAtTransformKey, outAabb( nextInputKey - 1 ), outAabb( nextInputKey ) );
            }

            advanceInputKey();
            //// do not advance the transform key if we are done and it aligns with the last input key.
            //// we need the same transform key for the next interval of input keys!
            //if( nextInputKey == outAabbs.size )
            //    break;
            advanceTransformKey();
        }
        else if( nextKeyIsTransformKey )
        {
            //assert( nextTransformKey <= numKeys - 1 );
            // we should never have nextInputKey == 0 && nextKeyIsInputKey == false && nextKeyIsTransformKey == true
            //assert( nextInputKey != 0 );

            if( prevKeyWasInputKey && prevKeyWasTransformKey )
            {
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                localInputTimeAtTransformKey = 1.0f - ( ( nextInputKeyTime - nextTransformKeyTime ) / inputIntervalTime );
                // interpolate input to that time
                inputAabbAtLastTransformKey = lerp( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ), localInputTimeAtTransformKey );

                transformInOut0Out1( inAabb( nextInputKey - 1 ), inputAabbAtLastTransformKey,
                    transformData( nextTransformKey - 1 ), transformData( nextTransformKey ),
                    outAabb( nextInputKey - 1 ), outputAabbAtLastTransformKey );

                // Resample outputAabbAtLastTransformKey to interval [nextInputKey-1,nextInputKey]
                // such that the linear transformation of outAabb( nextInputKey - 1 ) -> outputAabbAtLastTransformKey with t range [timePrevInput, timeNextTransform]
                // is included in outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                resampleSimpleUniform( outputAabbAtLastTransformKey, localInputTimeAtTransformKey, outAabb( nextInputKey - 1 ), outAabb( nextInputKey ) );
            }
            else if( prevKeyWasInputKey )
            {
                // using last as 'next' here
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                localInputTimeAtTransformKey = 1.0f - ( ( nextInputKeyTime - nextTransformKeyTime ) / inputIntervalTime );
                // interpolate input to that time
                inputAabbAtLastTransformKey = lerp( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ), localInputTimeAtTransformKey );

                transformInOut0Out1( inAabb( nextInputKey-1 ), inputAabbAtLastTransformKey,
                           lastTransformAtInputKey, transformData( nextTransformKey ),
                           outAabb( nextInputKey-1 ), outputAabbAtLastTransformKey );

                // Resample outputAabbAtLastTransformKey to interval [nextInputKey-1,nextInputKey]
                // such that the linear transformation of outAabb( nextInputKey - 1 ) -> outputAabbAtLastTransformKey with t range [timePrevInput, timeNextTransform]
                // is included in outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                resampleSimpleUniform( outputAabbAtLastTransformKey, localInputTimeAtTransformKey, outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
            }
            else //if( prevKeyWasTransformKey )
            {
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                float localInputTimeAtTransformKeyNow = 1.0f - ( ( nextInputKeyTime - nextTransformKeyTime ) / inputIntervalTime );
                // interpolate input to that time
                Aabb inputAabbAtLastTransformKeyNow = lerp( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ), localInputTimeAtTransformKeyNow );
                Aabb outputAabbAtLastTransformKeyNow;

                // we do not have the aabb at the time of the previous transform key
                transformInOut0Out1( inputAabbAtLastTransformKey, inputAabbAtLastTransformKeyNow,
                           transformData( nextTransformKey - 1 ), transformData( nextTransformKey ),
                           outputAabbAtLastTransformKey, outputAabbAtLastTransformKeyNow );

                // Resample such that the linear transformation of outputAabbAtLastTransformKey -> outputAabbAtLastTransformKeyNow with t range [timePrevTransform, timeNextTransform]
                // is included in outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                // Note that resampleSimpleUniform ensures correctness as it does not change the derivative of outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                // However, the order of the two operations has an impact on the output.
                // We may want to switch to a resampling that can look at both aabbs at the same time.
                resampleSimpleUniform( outputAabbAtLastTransformKey, localInputTimeAtTransformKey, outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );
                resampleSimpleUniform( outputAabbAtLastTransformKeyNow, localInputTimeAtTransformKeyNow, outAabb( nextInputKey-1 ), outAabb( nextInputKey ) );

                outputAabbAtLastTransformKey = outputAabbAtLastTransformKeyNow;

                inputAabbAtLastTransformKey = inputAabbAtLastTransformKeyNow;
                localInputTimeAtTransformKey = localInputTimeAtTransformKeyNow;
            }

            advanceTransformKey();
        }
        else //if( nextKeyIsInputKey )
        {
            //assert( nextTransformKey <= numKeys );
            if( prevKeyWasInputKey && prevKeyWasTransformKey )
            {
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                float localTransformTime = 1.0f - ( ( nextTransformKeyTime - nextInputKeyTime ) / transformIntervalTime );
                lastTransformAtInputKey = interpolateTransform( nextTransformKey - 1, localTransformTime );

                transformInOut0InOut1( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ),
                    transformData( nextTransformKey - 1 ), lastTransformAtInputKey,
                    outAabb( nextInputKey - 1 ), outAabb( nextInputKey ) );
            }
            else if( prevKeyWasInputKey )
            {
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                float localTransformTime = 1.0f - ( ( nextTransformKeyTime - nextInputKeyTime ) / transformIntervalTime );
                TransformData lastTransformAtInputKeyNow = interpolateTransform( nextTransformKey - 1, localTransformTime );

                transformInOut0InOut1( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ),
                    lastTransformAtInputKey, lastTransformAtInputKeyNow,
                    outAabb( nextInputKey - 1 ), outAabb( nextInputKey ) );

                lastTransformAtInputKey = lastTransformAtInputKeyNow;
            }
            else //if( prevKeyWasTransformKey )
            {
                // local time, i.e., interpolant between the current input keys at time of next transform key
                // nextTransformKeyTime is 'now'
                float localTransformTime = 1.0f - ( ( nextTransformKeyTime - nextInputKeyTime ) / transformIntervalTime );
                lastTransformAtInputKey = interpolateTransform( nextTransformKey - 1, localTransformTime );

                // outputAabbAtLastTransformKey is assumed to be valid here!
                transformInOut0InOut1( inAabb( nextInputKey - 1 ), inAabb( nextInputKey ),
                    transformData( nextTransformKey - 1 ), lastTransformAtInputKey,
                    outputAabbAtLastTransformKey, outAabb( nextInputKey ) );

                // Resample outputAabbAtLastTransformKey to interval [nextInputKey-1,nextInputKey]
                // such that the linear transformation of outputAabbAtLastTransformKey -> outAabb( nextInputKey ) with t range [timePrevTransform, timeNextInput]
                // is included in outAabb( nextInputKey - 1 ) -> outAabb( nextInputKey ).
                resampleSimpleUniform( outputAabbAtLastTransformKey, localInputTimeAtTransformKey, outAabb( nextInputKey - 1 ), outAabb( nextInputKey ) );
            }

            advanceInputKey();
        }

        prevKeyWasInputKey = nextKeyIsInputKey;
        prevKeyWasTransformKey = nextKeyIsTransformKey;
    }
}

#ifdef __LWDACC__
} // namespace lwca
#endif

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


}  // namespace motion
}  // namespace optix_exp
