/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <optix_types.h>
#include <exp/accel/InstanceAabbComputer.h>

//////////////////////////////////////////////////////////////////////////
// WARNING
// The extension is disabled by default (i.e., is NOT built) and should only be used for debugging.
// This is because it includes rtcore headers and we do not want such a dependency by default.
//////////////////////////////////////////////////////////////////////////
//#define ENABLE_HOST_VARIANT
#ifdef ENABLE_HOST_VARIANT
#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

// TODO: remove dependency on BvhHeader!
#include <rtcore/bvhtools/include/Types.hpp>

using bvhtools::BvhHeader;

#include <algorithm>
#ifndef __LWDACC__
using std::min;
using std::max;
#endif

#if 0
#define PRINTF( ... ) printf( __VA_ARGS__ )
#define PRINT_AABB_ARGS( aabb ) aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z
#define PRINT_SRT_ARGS( key )                                                                                          \
    key.sx, key.a, key.b, key.pvx, key.sy, key.c, key.pvy, key.sz, key.pvz, key.qx, key.qy, key.qz, key.qw, key.tx,    \
        key.ty, key.tz
#else
#define PRINTF( ... )
#endif

#define LWDA_NO_HALF
#include <cassert>
#include <rtcore/bvhtools/src/bounds/motion/motionTypes.h>
#include <rtcore/bvhtools/src/common/PrimitiveAABB.hpp>
#include <rtcore/bvhtools/src/common/TypesInternal.hpp>
#include <rtcore/bvhtools/src/bounds/motion/motionTransform.h>
#include <rtcore/bvhtools/src/bounds/motion/resampleMotionAabb.hpp>
#include <exp/accel/motion/resampleMotionAabb.hpp>
#include <exp/accel/motion/motionAabb.h>
#include <exp/accel/motion/motionImpl.h>
#endif

//////////////////////////////////////////////////////////////////////////

//#define ENABLE_DEVICE_VARIANT
#ifdef ENABLE_DEVICE_VARIANT
#include <exp/accel/InstanceAabbComputerKernels.h>
#endif

//////////////////////////////////////////////////////////////////////////

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/system/Knobs.h>

#include <memory>
#include <vector>

#include <prodlib/system/Knobs.h>
namespace
{
Knob<bool> k_useFixedMemory( RT_DSTRING( "o7.iasAabbsUseFixedMemory" ), true, RT_DSTRING( "Uses algorithm with fixed memory requirements." ) );
Knob<int>  k_instanceAabbComputerSuperSampling( RT_DSTRING( "o7.instanceAabbComputerSuperSampling" ), 3, RT_DSTRING( "Number of temporary AABBs used per output interval for super sampling motion intervals (to decrease output AABB size). Must be <= AabbArray::capacity" ) );
}

inline void syncCheck( const char* file, unsigned int line )
{
    lwdaDeviceSynchronize();
    const lwdaError_t error = lwdaGetLastError();
    if( error != lwdaSuccess )
    {
        printf("LWCA sync check in %s: line %i failed with code %i: %s", file, line, error, lwdaGetErrorString( error ));
    }
}
#define LWDA_SYNC_CHECK() syncCheck( __FILE__, __LINE__ )

namespace optix_exp {

#ifdef ENABLE_HOST_VARIANT

using namespace bvhtools;
using namespace bvhtools::motion;

namespace {

// platform independent count trailing zeros
inline unsigned int ctz( unsigned int x )
{
#ifdef _WIN32
    unsigned long trailing_zero;
    _BitScanForward( &trailing_zero, (unsigned long)x );
    return trailing_zero;
#else
    return __builtin_ctz( x );
#endif
}

// function to callwlate 
// gcd of two numbers. 
unsigned int gcd(unsigned int u, unsigned int v)
{
    int shift;
    if (u == 0) return v;
    if (v == 0) return u;
    shift = ctz(u | v);
    u >>= ctz(u);
    do {
        v >>= ctz(v);
        if (u > v) {
            unsigned int t = v;
            v = u;
            u = t;
        }  
        v = v - u;
    } while (v != 0);
    return u << shift;
}

// function to callwlate 
// lcm of two numbers. 
int lcm(int a, int b) 
{ 
    return (a * b) / gcd(a, b); 
} 

class BasicMemPool
{
public:
    std::pair<void*, unsigned int> alloc( size_t bytes )
    {
        unsigned int idx = (unsigned int)m_dataOffset.size();
        m_dataOffset.push_back( m_data.size() );
        // dummy increase by one to trigger "smart" memory allocation scheme
        m_data.push_back( '0' );
        m_data.pop_back();
        // actual resize
        m_data.resize( m_data.size() + bytes );
        return std::make_pair( (void*)&m_data[m_dataOffset.back()], idx );
    }

    template <typename T>
    const T* element( size_t idx ) const
    {
        return (const T*)( &m_data[m_dataOffset[idx]] );
    }

private:
    std::vector<size_t> m_dataOffset;
    std::vector<char>   m_data;
};


size_t roundUp8( size_t i )
{
    return ( ( i + 8 - 1 ) / 8 ) * 8;
}

size_t sizeofSrtNKeys( unsigned int numKeys )
{
    return roundUp8( sizeof( OptixSRTMotionTransform ) + sizeof( SRTData ) * ( numKeys - 2 ) );
}
size_t sizeofMatrixNKeys( unsigned int numKeys )
{
    return roundUp8( sizeof( OptixMatrixMotionTransform ) + sizeof( float ) * 12 * ( numKeys - 2 ) );
}

template<typename T>
inline void storeUncachedAlign16( T* prt, T v )
{
    *prt = v;
}

}  // namespace

#endif

OptixResult instanceAabbsComputeMemoryUsage( DeviceContext*            context,
                                             LWstream                  stream,
                                             const OptixMotionOptions* iasMotionOptions,
                                             LWdeviceptr               instances,
                                             unsigned int              numInstances,
                                             LWdeviceptr               tempBufferSizeInBytes,
                                             ErrorDetails&             errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( context, stream );

#ifdef ENABLE_DEVICE_VARIANT
    // we need to emit the motion aabbs from the IAS/GAS into a temp buffer
    // this needs to be revisited when doing batched emitting of motion aabbs
    // or when moving part of the instance aabb computation to the device
    size_t tempMemSize = sizeof( size_t ) + sizeof( InstanceNode ) * numInstances;
    if( const LWresult result = corelib::lwdaDriver().LwMemcpyHtoDAsync( tempBufferSizeInBytes, &tempMemSize, sizeof( size_t ), stream ) )
    {
        return errDetails.logDetails( result, "Copying data to device failed." );
    }

    computeMemoryUsage( stream, reinterpret_cast<const MotionOptions*>( iasMotionOptions ),
                        instances, numInstances, tempBufferSizeInBytes );
#elif defined(ENABLE_HOST_VARIANT)
    size_t tempMemSize = 255 * sizeof( RtcEmittedAccelPropertyAabb );
    if( const LWresult result = corelib::lwdaDriver().LwMemcpyHtoDAsync( tempBufferSizeInBytes, &tempMemSize, sizeof( size_t ), stream ) )
    {
        return errDetails.logDetails( result, "Copying data to device failed." );
    }
#endif // ENABLE_HOST_VARIANT

    return OPTIX_SUCCESS;
}


OptixResult instanceAabbsCompute( DeviceContext*            context,
                                  LWstream                  stream,
                                  const OptixMotionOptions* iasMotionOptions,
                                  LWdeviceptr               instances,
                                  unsigned int              numInstances,
                                  LWdeviceptr               tempBuffer,
                                  size_t                    tempBufferSizeInBytes,
                                  LWdeviceptr               outputAabbBuffer,
                                  size_t                    outputAabbBufferSizeInBytes,
                                  ErrorDetails&             errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( context, stream );
#if defined(ENABLE_DEVICE_VARIANT) || defined(ENABLE_HOST_VARIANT)
    if( outputAabbBufferSizeInBytes < numInstances * iasMotionOptions->numKeys * sizeof( Aabb ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( R"msg(Size of "outputAabbBuffer" needs to be at least "numInstances" * "iasMotionOptions->numKeys" * sizeof( OptixAabb ). "outputAabbBufferSizeInBytes" is only: %zi )msg",
                                                                                   outputAabbBufferSizeInBytes ) );
    }
#endif

    unsigned int iasNumKeys = std::max<unsigned int>( 1u, iasMotionOptions->numKeys );

#ifdef ENABLE_DEVICE_VARIANT
    //printf( "//////////////////////////////////////////////////////////////////////////\n" );
    //printf( "// DEVICE \n" );
    //printf( "//////////////////////////////////////////////////////////////////////////\n" );

    size_t tempMemSize = 0;
    if( const LWresult result = corelib::lwdaDriver().LwMemcpyHtoDAsync( tempBuffer, &tempMemSize, sizeof( size_t ), stream ) )
    {
        return errDetails.logDetails( result, "Copying data to device failed." );
    }

    //////////////////////////////////////////////////////////////////////////
    // Part I: compute maximum key count per transform chain, bottom to top
    //////////////////////////////////////////////////////////////////////////
    computePerInstanceMemory( stream, reinterpret_cast<const motion::MotionOptions*>( iasMotionOptions ),
                              instances, numInstances, tempBuffer );

    //////////////////////////////////////////////////////////////////////////
    // Part II: emit GAS (motion) aabbs
    //////////////////////////////////////////////////////////////////////////

    {
        std::vector<InstanceNode> instNodes( numInstances );

        LWdeviceptr d_instanceNodes = tempBuffer + sizeof( size_t );
        // non-async copy
        if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( (void*)instNodes.data(), d_instanceNodes,
            sizeof( InstanceNode ) * numInstances ) )
        {
            return errDetails.logDetails( result, "Copying data to host failed." );
        }

        ScopedCommandList commandList( context );
        if( const OptixResult result = commandList.init( stream, errDetails ) )
        {
            return result;
        }

        LWdeviceptr d_tempAabbs = d_instanceNodes + sizeof( InstanceNode ) * numInstances;

        for( unsigned int i = 0; i < numInstances; ++i )
        {
            InstanceNode& iasNode = instNodes[i];

            LWdeviceptr emitAabbAddress;
            if( iasNode.usesOutputAsTmp && iasNode.usesCasAabbEmitToOutput )
                emitAabbAddress = ( LWdeviceptr ) & ( ( (motion::Aabb*)outputAabbBuffer )[i * iasNumKeys] );
            else
                emitAabbAddress = d_tempAabbs + instNodes[i].tmpAddress;

            if( iasNode.numCasKeys <= 1 )
            {
                if( const RtcResult rtcResult =
                    context->getRtcore().accelEmitProperties( commandList.get(), &iasNode.cas, 1, RTC_PROPERTY_TYPE_AABB,
                        emitAabbAddress, sizeof( RtcEmittedAccelPropertyAabb ) ) )
                {
                    commandList.destroy( errDetails );
                    return errDetails.logDetails( rtcResult, "Querying aabb of acceleration structure failed." );
                }
            }
            else  // motion
            {
                if( const RtcResult rtcResult = context->getRtcore().accelEmitProperties(
                    commandList.get(), &iasNode.cas, 1, RTC_PROPERTY_TYPE_MOTION_AABBS, emitAabbAddress,
                    sizeof( RtcEmittedAccelPropertyAabb ) * iasNode.numCasKeys ) )
                {
                    commandList.destroy( errDetails );
                    return errDetails.logDetails( rtcResult, "Querying aabbs of acceleration structure failed." );
                }
            }
        }
        if( const OptixResult result = commandList.destroy( errDetails ) )
        {
            return result;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Part IV: compute motion aabbs bottom to top
    //////////////////////////////////////////////////////////////////////////

    computeInstanceAabbs( stream, reinterpret_cast<const motion::MotionOptions*>( iasMotionOptions ), instances, numInstances, tempBuffer, outputAabbBuffer );
#endif // ENABLE_DEVICE_VARIANT

#ifdef ENABLE_HOST_VARIANT

    enum {
        MAX_TRANSFORM_CHAIN = 30, // MAX_TRAVERSABLE_GRAPH_DEPTH - 1 for the instance
        MAX_TRAVERSABLE_GRAPH_DEPTH = 31
    };

    //printf( "//////////////////////////////////////////////////////////////////////////\n" );
    //printf( "// HOST \n" );
    //printf( "//////////////////////////////////////////////////////////////////////////\n" );

    std::vector<Aabb> instanceAabbs( numInstances * iasNumKeys );

    std::unique_ptr<OptixInstance[]> h_instanceData( new OptixInstance[numInstances] );
    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( (void*)h_instanceData.get(), instances,
                                                                    numInstances * sizeof( OptixInstance ) ) )
    {
        return errDetails.logDetails( result, "Copying instances to host failed." );
    }

    BasicMemPool transformPool;

    // Transforms and leaf IAS/GASes are TravNodes. Instances (input) are not explicitly turned into a travNode
    struct TravNode
    {
        bool isTransform() const
        {
            return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
        }
        bool isMotionTransform() const
        {
            return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
        }

        RtcTraversableType type = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;

        bool hasChild() const { return childIdx != ~0u; }
        // trav node idx of child
        unsigned int         childIdx = ~0u;

        LWdeviceptr          thisP = 0;
        RtcTraversableHandle traversable = 0;
        // dataIdx may index transformPool
        unsigned int         dataIdx = ~0u;

        unsigned int         numKeys;
        float                timeBegin;
        float                timeEnd;
        unsigned int         flags;
    };

    struct InstanceNode 
    {
        unsigned int maxKeys           : 8;
        unsigned int numGasKeys        : 8;
        unsigned int usesIrregularKeys : 1;
        unsigned int usesOutputAsTmp   : 1;
        unsigned int usesEmitToOutput  : 1;

        RtcTraversableHandle gas;
        LWdeviceptr          tmpAddress;
    };

    // TODO: should be part of the temporary memory
    std::vector<InstanceNode> instNodes;
    instNodes.resize( numInstances );

    std::vector<TravNode> nodes;
    nodes.resize( numInstances );
    nodes.reserve( 2 * numInstances );  // instances themselves + their children

    struct TraversableV1 {

        bool isTransform() const
        {
            return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
        }
        bool isMotionTransform() const
        {
            return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
        }

        const OptixSRTMotionTransform* getSrtMotionTransform( BasicMemPool &transformPool )
        {
            assert( type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM );
            return transformPool.element<OptixSRTMotionTransform>( dataIdx );
        }

        const OptixMatrixMotionTransform* getMatrixMotionTransform( BasicMemPool &transformPool )
        {
            assert( type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM );
            return transformPool.element<OptixMatrixMotionTransform>( dataIdx );
        }

        const OptixStaticTransform* getStaticTransform( BasicMemPool &transformPool )
        {
            assert( type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM );
            return transformPool.element<OptixStaticTransform>( dataIdx );
        }

        unsigned int numIrregularKeys = 0;
        bool         merged           = false;

        // dataIdx may index transformPool
        unsigned int         dataIdx = ~0u;

        RtcTraversableType   type;
        RtcTraversableHandle traversable;
        unsigned int         numKeys;
        float                timeBegin;
        float                timeEnd;
    };

    auto getTraversableChailw1 = [&](
        unsigned int rootIdx,
        TraversableV1  outTraversables[MAX_TRANSFORM_CHAIN],
        int&         outTraversableCount ) -> void {

        outTraversableCount = 0;
        unsigned int j = rootIdx;
        while( nodes[j].hasChild() )
        {
            j = nodes[j].childIdx;

            outTraversables[outTraversableCount].dataIdx     = nodes[j].dataIdx;
            outTraversables[outTraversableCount].type        = nodes[j].type;
            outTraversables[outTraversableCount].traversable = nodes[j].traversable;
            outTraversables[outTraversableCount].numKeys     = nodes[j].numKeys;
            outTraversables[outTraversableCount].timeBegin   = nodes[j].timeBegin;
            outTraversables[outTraversableCount].timeEnd     = nodes[j].timeEnd;
            outTraversableCount++;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    // Part I: download transform chains to host
    //////////////////////////////////////////////////////////////////////////
    {
        struct MotionTransformBase
        {
            RtcTraversableHandle child;
            Rtlw16               numKeys;
            Rtlw16               flags; /* clamp modes and such */
            RtcF32               timeBegin;
            RtcF32               timeEnd;
        };

        auto getNodeFromHandle = [&]( const RtcTraversableHandle& childHandle, TravNode* lwrrentNode,
                                      ErrorDetails& errDetails ) -> TravNode* {
            if( childHandle == 0 )
                return nullptr;

            // new node
            TravNode           newNode;
            RtcAccelType       accelType;
            RtcTraversableType travType;

            if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
                    context->getRtcDeviceContext(), childHandle, (RtcGpuVA*)&newNode.thisP, &travType, &accelType ) )
            {
                errDetails.logDetails( rtcResult, "Internal error." );
                return nullptr;
            }

            newNode.type = travType;
            newNode.traversable   = childHandle;
            lwrrentNode->childIdx = nodes.size();
            nodes.push_back( newNode );
            return &nodes.back();
        };

        // TODO: remove dependency on BvhHeader!
        // fetch data straight from the bvhheader for now.
        // alternative: we can run an optix kernel and use rtcore intrinsics to read the data from the bvh given the handle!
        bvhtools::BvhHeader bvhH;

        for( unsigned int i = 0; i < numInstances; ++i )
        {
            LWdeviceptr thisP                = ( LWdeviceptr )( (char*)instances + i * sizeof( OptixInstance ) );

            // insert instance as node to be able to store the childIdx of the index
            TravNode* lwrrentNode            = &nodes[i];
            lwrrentNode->thisP               = thisP;
            lwrrentNode->type                = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
            RtcTraversableHandle childHandle = h_instanceData[i].traversableHandle;

            lwrrentNode = getNodeFromHandle( childHandle, lwrrentNode, errDetails );

            while( lwrrentNode && lwrrentNode->isTransform() )
            {
                std::pair<void*, unsigned int> transformPoolElement;
                if( lwrrentNode->isMotionTransform() )
                {
                    MotionTransformBase motionTrans;
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH(
                            (void*)&motionTrans, (LWdeviceptr)lwrrentNode->thisP, sizeof( MotionTransformBase ) ) )
                    {
                        return errDetails.logDetails( result, "Copying transform traversable to host failed." );
                    }

                    size_t s = lwrrentNode->type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM ?
                                   sizeofSrtNKeys( motionTrans.numKeys ) :
                                   sizeofMatrixNKeys( motionTrans.numKeys );
                    transformPoolElement = transformPool.alloc( s );
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( transformPoolElement.first,
                                                                                    (LWdeviceptr)lwrrentNode->thisP, s ) )
                    {
                        return errDetails.logDetails( result, "Copying transform to host failed." );
                    }

                    childHandle = motionTrans.child;

                    lwrrentNode->numKeys   = motionTrans.numKeys;
                    lwrrentNode->timeBegin = motionTrans.timeBegin;
                    lwrrentNode->timeEnd   = motionTrans.timeEnd;
                    lwrrentNode->flags     = motionTrans.flags;
                }
                else  // static transform
                {
                    transformPoolElement = transformPool.alloc( sizeof( RtcTravStaticTransform ) );
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH(
                            transformPoolElement.first, (LWdeviceptr)lwrrentNode->thisP, sizeof( RtcTravStaticTransform ) ) )
                    {
                        return errDetails.logDetails( result, "Copying transform to host failed." );
                    }
                    childHandle = ( (RtcTravStaticTransform*)transformPoolElement.first )->child;

                    lwrrentNode->numKeys   = 1;
                    lwrrentNode->timeBegin = 0;
                    lwrrentNode->timeEnd   = 0;
                    lwrrentNode->flags     = 0;
                }
                lwrrentNode->dataIdx = transformPoolElement.second;

                lwrrentNode = getNodeFromHandle( childHandle, lwrrentNode, errDetails );
            }

            if( lwrrentNode == nullptr )
                continue;

            //must have IAS or GAS as child
            assert( lwrrentNode->type == RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL || lwrrentNode->type == RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL );
            if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( (void*)&bvhH, (LWdeviceptr)lwrrentNode->thisP,
                                                                            sizeof( bvhtools::BvhHeader ) ) )
            {
                return errDetails.logDetails( result, "Copying bvh header data to host failed." );
            }

            lwrrentNode->numKeys   = bvhH.numKeys > 1 ? bvhH.numKeys : 1;
            lwrrentNode->timeBegin = bvhH.timeBegin;
            lwrrentNode->timeEnd   = bvhH.timeEnd;
            lwrrentNode->flags     = bvhH.motionFlags;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Part II: compute maximum key count per transform chain, bottom to top
    //          TODO: also runs during the memory usage computation
    //////////////////////////////////////////////////////////////////////////
    size_t allocatedTmpMemoryInBytes = 0;
    {
        for( unsigned int i=0; i<numInstances; ++i )
        {
            TraversableV1 traversables[MAX_TRANSFORM_CHAIN];
            int traversableCount = 0;

            getTraversableChailw1( i, traversables, traversableCount );

            // collect the number of irregular keys per traversable bottom-up
            bool usesIrregularKeys = false;
            for( int i=traversableCount-1; i>=0; --i )
            {
                TraversableV1& s0 = traversables[i];

                // skip nodes that have already been processed
                if( s0.merged )
                    continue;

                const int   numKeys   = s0.numKeys;
                const float timeBegin = s0.timeBegin;
                const float timeEnd   = s0.timeEnd;

                // skip nodes without motion keys
                if( numKeys <= 1 )
                    continue;

                int numRegularKeys = numKeys;

                s0.numIrregularKeys += numKeys;

                // propagate the current regular keys up and try to merge the keys with other regular keys
                for( int j=i-1; j>=0; --j )
                {
                    TraversableV1& s1 = traversables[j];

                    const int   childNumKeys   = s1.numKeys;
                    const float childTimeBegin = s1.timeBegin;
                    const float childTimeEnd   = s1.timeEnd;

                    // try merging regular motion keys
                    bool mergeToRegular = false;

                    // skip traversables that have already been processed
                    if( !s1.merged )
                    {
                        if( childNumKeys <= 1 )
                        {
                            // static nodes can always be merged
                            mergeToRegular = true;
                        }
                        else if( timeBegin == childTimeBegin && timeEnd == childTimeEnd )
                        {
                            // check if keys trivially match
                            if( childNumKeys == numRegularKeys )
                            {
                                // perfect match
                                mergeToRegular = true;
                            }
                            else
                            {
                                usesIrregularKeys = true; // internally uses irregular keys to construct the combined joined distribution

                                // if we merge the keys into regular keys, we need a common multiple of intervals
                                // if we merge the keys into irregular keys, we just need the sum of keys excluding the boundaries
                                // check which is best
                                int commonMultiple = lcm( numRegularKeys - 1, childNumKeys - 1 ) + 1;
                                if( commonMultiple <= (childNumKeys - 2) + numRegularKeys )
                                {
                                    mergeToRegular = true;
                                    numRegularKeys = commonMultiple;
                                }
                            }
                        }
                        else
                        {
                            usesIrregularKeys = true;
                        }
                    }

                    if( mergeToRegular )
                    {
                        // this entry has been merged with descending traversable keys and can be ignored for merging from now on
                        // we still need to add in non-matching keys from descending traversables
                        s1.numIrregularKeys += numRegularKeys;
                        s1.merged = true; // flag as merged
                    }
                    else
                    {
                        int numUnmatchedKeys = numRegularKeys;

                        // check for matching boundaries
                        if( timeBegin == childTimeBegin )
                            numUnmatchedKeys--;
                        if( timeEnd == childTimeEnd )
                            numUnmatchedKeys--;

                        s1.numIrregularKeys += numUnmatchedKeys;
                    }
                }
            }

            const TraversableV1& gasTraversable = traversables[traversableCount-1];

            unsigned int numGasKeys = gasTraversable.numKeys;
            unsigned int maxKeys    = max( 1u, traversables[0].numIrregularKeys );

            size_t tmpMemoryInBytes = 0;
            tmpMemoryInBytes += maxKeys * sizeof( Aabb );

            // use the output buffer when it's big enough to hold the maximum key count
            if( true || maxKeys > iasMotionOptions->numKeys )
            {
                tmpMemoryInBytes += maxKeys * sizeof( Aabb );
                instNodes[i].usesOutputAsTmp  = false;
                instNodes[i].usesEmitToOutput = false;
            }
            else
            {
                instNodes[i].usesOutputAsTmp  = true;
                instNodes[i].usesEmitToOutput = ( (traversableCount % 2) == 0 );
            }

            // we only need key times for irregular keys
            if( usesIrregularKeys )
                tmpMemoryInBytes += (2*maxKeys) * sizeof(float);

            instNodes[i].maxKeys           = maxKeys;
            instNodes[i].numGasKeys        = numGasKeys;
            instNodes[i].usesIrregularKeys = usesIrregularKeys;
            instNodes[i].tmpAddress        = allocatedTmpMemoryInBytes;
            instNodes[i].gas               = gasTraversable.traversable;

            // TODO: allocate using atomics on device
            allocatedTmpMemoryInBytes += tmpMemoryInBytes;
        }
    }

    std::vector<char> tmpMemory;
    tmpMemory.resize(allocatedTmpMemoryInBytes);

    //////////////////////////////////////////////////////////////////////////
    // Part III: emit GAS (motion) aabbs
    //////////////////////////////////////////////////////////////////////////
    {
        ScopedCommandList commandList( context );
        if( const OptixResult result = commandList.init( stream, errDetails ) )
        {
            return result;
        }

        for( unsigned int i = 0; i < numInstances; ++i )
        {
            InstanceNode &gasNode = instNodes[i];

            RtcAccelType       accelType;
            RtcTraversableType travType;
            RtcGpuVA           gasPtr;

            if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
                    context->getRtcDeviceContext(), gasNode.gas, (RtcGpuVA*)&gasPtr, &travType, &accelType ) )
            {
                commandList.destroy( errDetails );   
                return errDetails.logDetails( rtcResult, "Internal error." );
            }

            LWdeviceptr emitAabbAddress;
            if( gasNode.usesOutputAsTmp && gasNode.usesEmitToOutput )
                emitAabbAddress = ( LWdeviceptr )( &instanceAabbs[i * iasNumKeys] );
            else
                emitAabbAddress = ( LWdeviceptr )tmpMemory.data() + instNodes[i].tmpAddress;

            if( accelType == RTC_ACCEL_TYPE_BVH2 || accelType == RTC_ACCEL_TYPE_TTU )
            {
                if( const RtcResult rtcResult =
                        context->getRtcore().accelEmitProperties( commandList.get(), &gasPtr, 1, RTC_PROPERTY_TYPE_AABB,
                                                                  tempBuffer, sizeof( RtcEmittedAccelPropertyAabb ) ) )
                {
                    commandList.destroy( errDetails );
                    return errDetails.logDetails( rtcResult, "Querying aabb of acceleration structure failed." );
                }
            }
            else // motion
            {
                if( const RtcResult rtcResult = context->getRtcore().accelEmitProperties(
                        commandList.get(), &gasPtr, 1, RTC_PROPERTY_TYPE_MOTION_AABBS, tempBuffer,
                        sizeof( RtcEmittedAccelPropertyAabb ) * gasNode.numGasKeys ) )
                {
                    commandList.destroy( errDetails );
                    return errDetails.logDetails( rtcResult, "Querying aabbs of acceleration structure failed." );
                }
            }
            if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH(
                    (void *)emitAabbAddress, tempBuffer, sizeof( RtcEmittedAccelPropertyAabb ) * gasNode.numGasKeys ) )
            {
                commandList.destroy( errDetails );
                return errDetails.logDetails( result, "Copying AABB data to host failed." );
            }
        }

        if( const OptixResult result = commandList.destroy( errDetails ) )
        {
            return result;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    // Part IV: compute motion aabbs bottom to top
    //////////////////////////////////////////////////////////////////////////
    auto computeIASAabbsDynamicMemory = [&]() -> OptixResult {

        for( unsigned int i = 0; i < numInstances; ++i )
        {
            TraversableV1 traversables[MAX_TRANSFORM_CHAIN];
            int traversableCount = 0;

            getTraversableChailw1( i, traversables, traversableCount );

            motion::Aabb* iasBuildAabbs = (motion::Aabb*)&instanceAabbs[i * iasNumKeys];
            unsigned int maxKeys = instNodes[i].maxKeys;

            size_t tmpAddress = instNodes[i].tmpAddress;

            motion::Aabb* outKeyAabbs = reinterpret_cast<motion::Aabb*>( tmpMemory.data() + tmpAddress );
            tmpAddress += sizeof(Aabb) * (maxKeys);
            motion::Aabb* inKeyAabbs  = reinterpret_cast<motion::Aabb*>( tmpMemory.data() + tmpAddress );

            if( instNodes[i].usesOutputAsTmp )
            {
                // we can use the output buffer as temp memory as the max number of keys never exceeds the output key count.
                inKeyAabbs = iasBuildAabbs;

                // make sure the out buffer is not used as input to the resampling step.
                // this assumes the in and out tmp buffers are swapped at every node in the chain.
                if( instNodes[i].usesEmitToOutput )
                    std::swap( outKeyAabbs, inKeyAabbs );
            }
            else
            {
                tmpAddress += sizeof( Aabb ) * ( maxKeys );
            }

            float* inKeyTimes  = 0;
            float* outKeyTimes = 0;
            if( instNodes[i].usesIrregularKeys )
            {
                inKeyTimes  = reinterpret_cast<float*>( tmpMemory.data() + tmpAddress );
                outKeyTimes = inKeyTimes + maxKeys;
                tmpAddress += sizeof(float) * (2*maxKeys);
            }

            //////////////////////////////////////////////////////////////////////////
            // Part IIIa: compute motion aabbs bottom to top
            //////////////////////////////////////////////////////////////////////////

            motion::MotionAabb inAabb ( inKeyAabbs, inKeyTimes );
            motion::MotionAabb outAabb( outKeyAabbs, outKeyTimes );

            // outAabb contains the emitted gas aabbs.
            TraversableV1& gas = traversables[traversableCount-1];
            outAabb.initRegularDistribution( gas.timeBegin, gas.timeEnd, std::max<unsigned int>( 1u, gas.numKeys ) );

            for( int j = traversableCount - 2; j >= 0; j-- )
            {
                // input becomes output, output becomes input.
                std::swap( outAabb, inAabb );

                TraversableV1& traversable = traversables[j];

                if( traversable.isTransform() )
                {
                    if( traversable.type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM )
                    {
                        const OptixSRTMotionTransform* mt = traversable.getSrtMotionTransform( transformPool );
                        applySrtMotionTransform( (const motion::SRTData*)mt->srtData, reinterpret_cast<const motion::MotionOptions&>( mt->motionOptions ), inAabb, outAabb );
                    }
                    else if( traversable.type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM )
                    {
                        const OptixMatrixMotionTransform* mt = traversable.getMatrixMotionTransform( transformPool );
                        applyMatrixMotionTransform( &mt->transform[0][0], reinterpret_cast<const motion::MotionOptions&>( mt->motionOptions ), inAabb, outAabb );
                    }
                    else if( traversable.type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM )
                    {
                        const OptixStaticTransform* mt = traversable.getStaticTransform( transformPool );
                        applyMatrixStaticTransform( mt->transform, inAabb, outAabb );
                    }
                }
            }

            //////////////////////////////////////////////////////////////////////////
            // Part IIIb: resample motion aabbs at instances
            //////////////////////////////////////////////////////////////////////////

            motion::MotionAabb& motionAabb = outAabb;
            assert( motionAabb.aabbs() != iasBuildAabbs );
            if( iasNumKeys == 1 )
            {
                *iasBuildAabbs = motionAabb.aabbUnion();
            }
            else
            {
                if( motionAabb.isStatic() )
                {
                    motion::Aabb unionAabb = motionAabb.aabbUnion(); // union includes check for 0 aabbs / ilwalidity
                    for( unsigned int i = 0; i < motionAabb.keyCount(); ++i )
                        iasBuildAabbs[i] = unionAabb;
                }
                else if( motionAabb.canUseForASBuild( *reinterpret_cast<const motion::MotionOptions*>( iasMotionOptions ) ) )
                {
                    for( unsigned int i = 0; i < motionAabb.keyCount(); ++i )
                        iasBuildAabbs[i] = motionAabb.aabb(i);
                }
                else
                {
                    motion::resampleMotionAabbs(
                        motionAabb.keysAreRegularlyDistributed(),
                        motionAabb.timeFirstKey(), motionAabb.timeLastKey(),
                        motionAabb.aabbs(), (unsigned int)motionAabb.keyCount(),
                        motionAabb.keyTimes(),
                        iasBuildAabbs, iasMotionOptions->numKeys,
                        iasMotionOptions->timeBegin, iasMotionOptions->timeEnd );
                }
            }
        }

        return OPTIX_SUCCESS;
    };


    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////


    auto computeIASAabbsFixedMemory = [&]() -> OptixResult {


        struct InstanceAabbComputerExecParams
        {
            PrimitiveAABB*   outFirstAabbs;
            AABB*            outMotionAabbs;

            const char*      inInstances;
            int              indirect;
            int              readFatInstance;

            int              numInstances;
            MotionOptions    motionOptions;
            unsigned int     numPerIntervalAabbs; // this must be >= 2 and <= AAabbArray::capacity_. Possible values: 2^N + 1. Higher number increases computational effort, but is expected to produce smaller AABBs
            //unsigned int     motionSteps;                 // Number of motion steps, default = 1
            //float            motionTimeBegin;             // TLAS begin time
            //float            motionTimeEnd;               // TLAS end time
            //unsigned int     motionFlags;                 // Combinations of RtcMotionFlags. Clamp modes and such.
        };

        struct Traversable {

            __device__ __inline__ bool isTransform() const
            {
                return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
            }
            __device__ __inline__ bool isStaticTransform() const
            {
                return type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
            }
            __device__ __inline__ bool isMotionTransform() const
            {
                return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM || type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
            }
            __device__ __inline__ bool isSrtMotionTransform() const
            {
                return type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
            }
            __device__ __inline__ bool isMatrixMotionTransform() const
            {
                return type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
            }
            __device__ __inline__ bool isAccel() const
            {
                return type == RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL || type == RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL || type == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL;
            }
            __device__ __inline__ bool isStaticAccel() const
            {
                return isAccel() && motionOptions.numKeys <= 1;
            }
            __device__ __inline__ bool isMotionAccel() const
            {
                return isAccel() && !isStaticAccel();
            }

            const RtcSRTMotionTransform* getSrtMotionTransform( )
            {
                assert( type == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM );
                return transformPool->element<RtcSRTMotionTransform>( dataIdx );
            }

            const RtcMatrixMotionTransform* getMatrixMotionTransform( )
            {
                assert( type == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM );
                return transformPool->element<RtcMatrixMotionTransform>( dataIdx );
            }

            const RtcStaticTransform* getStaticTransform()
            {
                assert( type == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM );
                return transformPool->element<RtcStaticTransform>( dataIdx );
            }

            template<typename T = void>
            __device__ __inline__ const T* getPointer() const
            {
                return transformPool->element<T>( dataIdx );
            }

            unsigned int numIrregularKeys = 0;
            bool         merged           = false;

            // transform and target output keys perfectly align
            bool keysAlignTrivially = false;
            // all transform keys align with output keys, but there are more output keys than transform keys
            bool keysAlignFully = false;
            // all output keys align with transform keys, but there are more transform keys than output keys
            bool keysOversample = false;

            // dataIdx may index transformPool
            unsigned int         dataIdx = ~0u;

            RtcTraversableType   type;
            RtcTraversableHandle traversable;
            MotionOptions        motionOptions;
            // used for transforms
            unsigned int         lastKeyIndex = 0;

            inline float nextKeyTime() const
            {
                return motionOptions.timeAtKey( lastKeyIndex + 1 );
            }

            BasicMemPool* transformPool;
        };

        struct TransformChainProperties
        {
            Rtlw16 flags;
            RtcF32 minTimeBegin;
            RtcF32 maxTimeEnd;
            // add more info like maxKey count?
        };

        auto getTraversableChain = [&](
            unsigned int rootIdx,
            Traversable  outTraversables[MAX_TRANSFORM_CHAIN],
            int&         outTraversableCount,
            TransformChainProperties& minMaxTimeNolwanish ) -> void {

            outTraversableCount = 0;
            unsigned int j = rootIdx;
            while( nodes[j].hasChild() )
            {
                j = nodes[j].childIdx;

                Traversable& trav = outTraversables[outTraversableCount];

                trav.transformPool           = &transformPool;
                trav.dataIdx                 = nodes[j].dataIdx;
                trav.type                    = nodes[j].type;
                trav.traversable             = nodes[j].traversable;
                trav.motionOptions.numKeys   = nodes[j].numKeys;
                trav.motionOptions.timeBegin = nodes[j].timeBegin;
                trav.motionOptions.timeEnd   = nodes[j].timeEnd;
                trav.motionOptions.flags     = nodes[j].flags;

                if( trav.motionOptions.numKeys > 1 )
                {
                    if( trav.motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH && minMaxTimeNolwanish.flags & OPTIX_MOTION_FLAG_START_VANISH )
                        minMaxTimeNolwanish.minTimeBegin = max( minMaxTimeNolwanish.minTimeBegin, trav.motionOptions.timeBegin );
                    else if( trav.motionOptions.flags & OPTIX_MOTION_FLAG_START_VANISH )
                        minMaxTimeNolwanish.minTimeBegin = trav.motionOptions.timeBegin;
                    else if( minMaxTimeNolwanish.flags & OPTIX_MOTION_FLAG_START_VANISH )
                        ;// minMaxTimeNolwanish.minTimeBegin stays as is. This assumes that OPTIX_MOTION_FLAG_START_VANISH can only be set on minMaxTimeNolwanish if it has a valid begin/end time.
                    else
                        minMaxTimeNolwanish.minTimeBegin = min( minMaxTimeNolwanish.minTimeBegin, trav.motionOptions.timeBegin );

                    if( trav.motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH && minMaxTimeNolwanish.flags & OPTIX_MOTION_FLAG_END_VANISH )
                        minMaxTimeNolwanish.maxTimeEnd = min( minMaxTimeNolwanish.maxTimeEnd, trav.motionOptions.timeEnd );
                    else if( trav.motionOptions.flags & OPTIX_MOTION_FLAG_END_VANISH )
                        minMaxTimeNolwanish.maxTimeEnd = trav.motionOptions.timeEnd;
                    else if( minMaxTimeNolwanish.flags & OPTIX_MOTION_FLAG_END_VANISH )
                        ;// minMaxTimeNolwanish.maxTimeEnd stays as is. This assumes that OPTIX_MOTION_FLAG_END_VANISH can only be set on minMaxTimeNolwanish if it has a valid begin/end time.
                    else
                        minMaxTimeNolwanish.maxTimeEnd = max( minMaxTimeNolwanish.maxTimeEnd, trav.motionOptions.timeEnd );
                }

                outTraversableCount++;
            }
        };

        InstanceAabbComputerExecParams p ={};
        p.indirect = false;
        p.motionOptions.numKeys = iasMotionOptions->numKeys;
        p.motionOptions.timeBegin = iasMotionOptions->timeBegin;
        p.motionOptions.timeEnd = iasMotionOptions->timeEnd;
        p.motionOptions.flags = iasMotionOptions->flags;
        p.readFatInstance = false;
        std::vector<PrimitiveAABB> discardData( numInstances );
        p.outFirstAabbs  = discardData.data();
        p.outMotionAabbs = (AABB*)instanceAabbs.data();
        p.numPerIntervalAabbs = k_instanceAabbComputerSuperSampling.get();

        const unsigned int aabbArrayMaxSize = p.numPerIntervalAabbs;

        for( unsigned int idx = 0; idx < numInstances; ++idx )
        {

            Traversable traversablesChain[MAX_TRANSFORM_CHAIN];
            int traversablesChainLength = 0;

            TransformChainProperties chainProperties ={ 0, FLT_MAX, -FLT_MAX };
            getTraversableChain( idx, traversablesChain, traversablesChainLength, chainProperties );

            // FIXME, need to output an invalid AABB!
            if( traversablesChainLength == 0 )
                continue;

            InstanceNode& instanceInfo = instNodes[idx];
            Aabb* CASAabbs = ( instNodes[idx].usesOutputAsTmp && instNodes[idx].usesEmitToOutput ) ? &instanceAabbs[idx * iasNumKeys] : (Aabb*)(tmpMemory.data()+instNodes[idx].tmpAddress);

            // emits #size aabbs of the CAS starting at the 'offset's one.
            auto emitAabbs = [&]( unsigned int offset, unsigned int size, unsigned int, const BvhHeader*, Aabb* outAabbs )
            {
                for(unsigned int i=0; i<size; ++i)
                {
                    outAabbs[i] = CASAabbs[i + offset];
                }
            };

            // FIXME, need to output an invalid AABB, if last traversable is not an AS!
            if( !traversablesChain[traversablesChainLength-1].isAccel() )
                continue;

            AabbArray aabbsData[2];
            AabbArray* pInAabbs  = &aabbsData[0];
            AabbArray* pOutAabbs = &aabbsData[1];

            float intervalStepSize = p.motionOptions.intervalSize();
            const unsigned int numIntervals = max( 1u, p.motionOptions.numKeys - 1u ); // we need to run the following loops at least once, also for a static tlas (iasNumKeys==1)

            Aabb* intervalAabbs = nullptr;

            auto computeIntervalAabbs = [&]( Traversable* traversablesChain, int traversablesChainLength, const TransformChainProperties& chainProperties,
                const MotionOptions& iasMotionOptions, unsigned int intervalIdx,
                AabbArray aabbArrays[2], const unsigned int aabbArrayMaxSize, Aabb*& outIntervalAabbs,
                unsigned int childIdx = 0 )
            {
                const unsigned int iasNumKeys = iasMotionOptions.numKeys < 1 ? 1 : iasMotionOptions.numKeys;

                // TODO, make use of chainProperties minMax times in case of vanish (i.e., discard aabbs outside range and avoid any useless computations)
                float intervalBegin = iasNumKeys == 1 ? chainProperties.minTimeBegin : iasMotionOptions.timeAtKey( intervalIdx );
                float intervalEnd   = iasNumKeys == 1 ? chainProperties.maxTimeEnd   : intervalBegin + iasMotionOptions.intervalSize();

                // these pointers will not be used directly for the emit, but at later stages for applying the transforms
                // however, they may be swapped already during emit
                AabbArray* pInAabbs  = &aabbArrays[0];
                AabbArray* pOutAabbs = &aabbArrays[1];

                // 1. populate the aabb array with the CAS aabbs corresponding to the current interval
                {
                    AabbArray& outAabbs = aabbArrays[0];

                    // last traversable is the CAS
                    Traversable& cas = traversablesChain[traversablesChainLength-1];
                    unsigned short CASnumKeys = cas.motionOptions.numKeys;

                    const BvhHeader* bvhH = nullptr;

                    int emitOffset = 0;
                    int emitNumKeys = 1;
                    bool batchEmit = false;
                    bool lerpEmittedAabbsToInterval = false;
                    bool resampleEmittedAabbsToInterval = false;

                    if( CASnumKeys == 1 )
                    {
                        //emitOffset = 0;
                        //emitNumKeys = 1;
                    }
                    else if( iasNumKeys == 1 && ( intervalBegin == cas.motionOptions.timeBegin && intervalEnd == cas.motionOptions.timeEnd && ( aabbArrayMaxSize - 1 ) % ( CASnumKeys - 1 ) == 0 ) )
                    {
                        // if( iasNumKeys == 1 ) -> p.motionOptions is considered not valid
                        //emitOffset = 0;
                        emitNumKeys = CASnumKeys;
                    }
                    else if( iasNumKeys != 1 && iasMotionOptions.triviallyAligns( cas.motionOptions ) )
                    {
                        emitOffset = intervalIdx;
                        emitNumKeys = 2;
                    }
                    else if( iasNumKeys != 1 && CASnumKeys == 2 && iasMotionOptions.timesAlign( cas.motionOptions ) )
                    {
                        //emitOffset = 0;
                        emitNumKeys = 2;
                        lerpEmittedAabbsToInterval = true;
                    }
                    else
                    {
                        // the general case, resample whatever input to the maximum of available aabbs within the current output interval [intervalBegin, intervalEnd]
                        // TODO: consider more special cases that avoid resampling or the use of all available aabbs if not needed

                        //// this should be at most one iteration since the lastKeyIndex for the previous interval should be already contain the right index!
                        //while( cas.nextKeyTime() < intervalBegin )
                        //    cas.lastKeyIndex++;
                        // note beginKey is at or before intervalBegin
                        emitOffset = cas.motionOptions.keyAtOrBeforeT( intervalBegin );

                        // for the interval end time it is more efficient to compute the corresponding CAS key
                        // endKey is at or after intervalEnd
                        const unsigned int endKey = cas.motionOptions.keyAtOrAfterT( intervalEnd );
                        emitNumKeys = endKey - emitOffset + 1;
                        // this case will cover the special cases of cas.motionOptions.beginTime > intervalEnd as well as cas.motionOptions.endTime < intervalBegin
                        if( emitNumKeys > 1 && emitNumKeys <= aabbArrayMaxSize )
                        {
                            resampleEmittedAabbsToInterval = true;
                        }
                        if( emitNumKeys > aabbArrayMaxSize )
                        {
                            batchEmit = true;
                        }
                    }

                    if( !batchEmit )
                    {
                        emitAabbs( emitOffset, emitNumKeys, childIdx, bvhH, outAabbs.aabbs );

                        outAabbs.size = emitNumKeys;

                        if( lerpEmittedAabbsToInterval )
                        {
                            //assert( emitNumKeys == 2 );
                            AabbArray& inAabbs = aabbArrays[1];
                            inAabbs.size = 2;
                            // iasNumKeys > 1
                            inAabbs.aabbs[0] = lerp( outAabbs.aabbs[0], outAabbs.aabbs[1], (float)intervalIdx / ( iasNumKeys - 1 ) );
                            inAabbs.aabbs[1] = lerp( outAabbs.aabbs[0], outAabbs.aabbs[1], (float)( intervalIdx+1 ) / ( iasNumKeys - 1 ) );

                            // in this case we actually output to aabbArrays[1] instead of aabbArrays[0], so swap pointers for later stages
                            bvhtools::motion::swap( pInAabbs, pOutAabbs );
                        }
                        else if( resampleEmittedAabbsToInterval )
                        {
                            AabbArray& inAabbs = aabbArrays[1];
                            const unsigned int beginKey = emitOffset;
                            const unsigned int endKey = emitOffset + emitNumKeys - 1;

                            //TODO, special case that should be easy to resample!
                            //resample22( cas.motionOptions.timeAtKey( beginKey ), cas.motionOptions.timeAtKey( endKey ), outAabbs.aabbs,
                            //    intervalBegin, intervalEnd, inAabbs.aabbs );
                            // 'upsample' to aabbArrayMaxSize if we are dealing with more than two keys
                            inAabbs.size = emitNumKeys == 2 ? 2 : aabbArrayMaxSize;
                            resampleMotionAabbs(
                                cas.motionOptions.timeAtKey( beginKey ), cas.motionOptions.timeAtKey( endKey ), outAabbs.aabbs, emitNumKeys,
                                intervalBegin, intervalEnd, inAabbs.aabbs, inAabbs.size );

                            bvhtools::motion::swap( pInAabbs, pOutAabbs );
                        }
                    }
                    else
                    {
                        AabbArray& tempAabbs = aabbArrays[1];

                        const unsigned int endKey = emitOffset + emitNumKeys - 1;
                        {
                            // We need to emit in batches and resample in between

                            // Colwenience motion options to call the MotionOptions functions for the current interval
                            MotionOptions mo ={};
                            mo.timeBegin = intervalBegin;
                            mo.timeEnd = intervalEnd;
                            mo.numKeys = aabbArrayMaxSize;

#if 1
                            // strategy a) repeatedly emit aabbs, resample to output, include previous output
                            {
                                AabbArray tempResampleAabbs;
                                tempResampleAabbs.size = aabbArrayMaxSize;
                                tempAabbs.size         = aabbArrayMaxSize;
                                outAabbs.size          = aabbArrayMaxSize;
                                for( unsigned int k = 0; k < aabbArrayMaxSize; ++k )
                                {
                                    ilwalidate( outAabbs.aabbs[k] );
                                }
                                const unsigned int numBatchRuns = ( emitNumKeys + aabbArrayMaxSize - 1 ) / aabbArrayMaxSize;

                                cas.lastKeyIndex = emitOffset;

                                for( unsigned int batch = 0; batch < numBatchRuns; ++batch )
                                {
                                    unsigned int batchSize = min( endKey - cas.lastKeyIndex + 1, tempAabbs.size );
                                    emitAabbs( cas.lastKeyIndex, batchSize, childIdx, bvhH, tempAabbs.aabbs );

                                    const float timeBeginBatch = cas.motionOptions.timeAtKey( cas.lastKeyIndex );
                                    const float timeEndBatch   = cas.motionOptions.timeAtKey( cas.lastKeyIndex + batchSize - 1 );

                                    // find range of outAabbs into which the current batch of emitted aabbs falls
                                    // we will only resample those
                                    // note that keyAtOrBeforeT/keyAtOrAfterT clamp the output to the valid range [0, mo.numkeys-1]
                                    const unsigned int kBegin = mo.keyAtOrBeforeT( timeBeginBatch );
                                    const unsigned int kEnd   = mo.keyAtOrAfterT( timeEndBatch );

                                    const float timeKBegin = mo.timeAtKey( kBegin );
                                    const float timeKEnd = mo.timeAtKey( kEnd );

                                    // 'clamp' fill aabbs that are before the first emit key, those won't be touched below
                                    if( batch == 0 && kBegin > 0 )
                                    {
                                        for( unsigned int k = 0; k < kBegin; ++k )
                                        {
                                            outAabbs.aabbs[k] = tempAabbs.aabbs[0];
                                        }
                                    }
                                    // 'clamp' fill aabbs that are after the last emit key, those won't be touched below
                                    if( batch == numBatchRuns - 1 && kEnd + 1 < aabbArrayMaxSize )
                                    {
                                        for( unsigned int k = kEnd + 1; k < aabbArrayMaxSize; ++k )
                                        {
                                            outAabbs.aabbs[k] = tempAabbs.aabbs[batchSize - 1];
                                        }
                                    }

                                    if( batchSize == 1 )
                                    {
                                        if( kBegin == kEnd )
                                            include( outAabbs.aabbs[kBegin], tempAabbs.aabbs[0] );
                                        else
                                            resampleLocalOptima( tempAabbs.aabbs[0], ( timeBeginBatch - timeKBegin ) / ( timeKEnd - timeKBegin ), outAabbs.aabbs[kBegin], outAabbs.aabbs[kEnd] );
                                    }
                                    else
                                    {
                                        resampleMotionAabbs(
                                            timeBeginBatch, timeEndBatch, tempAabbs.aabbs, batchSize,
                                            timeKBegin, timeKEnd, tempResampleAabbs.aabbs, kEnd - kBegin + 1 );

                                        for( unsigned int k = 0; k < batchSize; ++k )
                                        {
                                            include( outAabbs.aabbs[k + kBegin], tempResampleAabbs.aabbs[k] );
                                        }
                                    }

                                    cas.lastKeyIndex += batchSize;
                                }
                            }
#else
                            // strategy b) init output aabbs with interpolated aabbs. Do simple resampling for in between inputs
                            {
                                outAabbs.size = aabbArrayMaxSize;
                                Aabb emittedAabbs[2];
                                // init outAabbs.aabbs with interpolated emitted aabbs at the outAabbs.aabbs' corresponding times
                                for( unsigned int k = 0; k < outAabbs.size; ++k )
                                {
                                    float t = intervalBegin + k * ( intervalEnd - intervalBegin ) / ( outAabbs.size - 1 );
                                    float keyf = cas.motionOptions.keyAtTNonclamped( t );
                                    if( keyf < 0 )
                                    {
                                        // extrapolate aabb[1] -> aabb[0] to aabb at t
                                        emitAabbs( 0, 2, childIdx, bvhH, emittedAabbs );
                                        outAabbs.aabbs[k] = lerp( emittedAabbs[0], emittedAabbs[1], keyf );
                                        // outAabbs.aabbs[k] may be invalid
                                        // make it valid by including its 'center'. Center still works as it does not care about the aabb being ilwerted (min > max)
                                        include( outAabbs.aabbs[k], center( outAabbs.aabbs[k] ) );
                                    }
                                    else if( keyf >( float )( cas.motionOptions.numKeys-1 ) )
                                    {
                                        // extrapolate aabb[N-2] -> aabb[N-1] to aabb at t
                                        emitAabbs( cas.motionOptions.numKeys-2, 2, childIdx, bvhH, emittedAabbs );
                                        outAabbs.aabbs[k] = lerp( emittedAabbs[0], emittedAabbs[1], 1.f + keyf - (float)( cas.motionOptions.numKeys-1 ) );
                                        // outAabbs.aabbs[k] may be invalid
                                        // make it valid by including its 'center'. Center still works as it does not care about the aabb being ilwerted (min > max)
                                        include( outAabbs.aabbs[k], center( outAabbs.aabbs[k] ) );
                                    }
                                    else if( (unsigned int)keyf == cas.motionOptions.numKeys-1 )
                                    {
                                        emitAabbs( cas.motionOptions.numKeys-1, 1, childIdx, bvhH, &outAabbs.aabbs[k] );
                                    }
                                    else
                                    {
                                        emitAabbs( (unsigned int)keyf, 2, childIdx, bvhH, emittedAabbs );
                                        outAabbs.aabbs[k] = lerp( emittedAabbs[0], emittedAabbs[1], fract( keyf ) );
                                    }
                                }
                                // loop over emitted aabbs, resample neighboring outAabbs.aabbs to ensure any interpolation between is covered
                                unsigned int outI = mo.keyAtOrBeforeT( cas.motionOptions.timeAtKey( cas.lastKeyIndex ) );
                                for( unsigned int j = 0; j < numEmitKeysInInterval; ++j )
                                {
                                    emitAabbs<TTUSplitInstance>( cas.lastKeyIndex + j, 1, childIdx, bvhH, emittedAabbs );
                                    float t = cas.motionOptions.timeAtKey( cas.lastKeyIndex + j );
                                    float outKeyf = mo.keyAtTNonclamped( t );
                                    if( outKeyf <= 0.0f || (unsigned int)outKeyf >= ( aabbArrayMaxSize - 1 ) )
                                        continue;
                                    resampleLocalOptima( emittedAabbs[0], fract( outKeyf ), outAabbs.aabbs[(unsigned int)outKeyf], outAabbs.aabbs[(unsigned int)outKeyf+1u] );
                                }
                            }
#endif
                        }
                        // need max since endKey may even be 0 if the first emit key comes after intervalEnd
                        cas.lastKeyIndex = max( 1u, endKey ) - 1u;
                    }
                }

                // 2. walk the chain of transforms from the CAS to the instance and apply any transforms
                for( int j = traversablesChainLength - 2; j >= 0; j-- )
                {
                    AabbArray& inAabbs  = *pInAabbs;
                    AabbArray& outAabbs = *pOutAabbs;
                    outAabbs.size = inAabbs.size;

                    PRINTF( "[%i] inAabb.size=%i\n", j, inAabbs.size );
                    for( int i = 0; i < inAabbs.size; ++i )
                        PRINTF( "inAabb[%i]: %f,%f,%f - %f,%f,%f\n", i, PRINT_AABB_ARGS( inAabbs.aabbs[i] ) );

                    Traversable& traversable = traversablesChain[j];

                    if( traversable.isSrtMotionTransform() )
                    {
                        const RtcSRTMotionTransform* mt = traversable.getSrtMotionTransform();
                        processMotionTransform( iasMotionOptions, intervalIdx, intervalBegin, intervalEnd, aabbArrayMaxSize, pInAabbs, pOutAabbs, *(const MotionOptions*)&mt->numKeys, (SRTData*)mt->quaternion );
                    }
                    else if( traversable.isMatrixMotionTransform() )
                    {
                        const RtcMatrixMotionTransform* mt = traversable.getMatrixMotionTransform();
                        processMotionTransform( iasMotionOptions, intervalIdx, intervalBegin, intervalEnd, aabbArrayMaxSize, pInAabbs, pOutAabbs, *(const MotionOptions*)&mt->numKeys, (Matrix3x4*)mt->transform );
                    }
                    else if( traversable.isStaticTransform() )
                    {
                        const RtcStaticTransform* mt = traversable.getStaticTransform();
                        // TODO: try to bake the static transform into the next transform
                        // not sure how to do this easily since the S of SRT is only a triangular matrix.
                        // Also not sure if we can simply apply the static transform to the corners of the aabb before applying an SRT (same for matrix motion transforms)
                        for( unsigned int k = 0; k < inAabbs.size; ++k )
                        {
                            outAabbs.aabbs[k] = transform( inAabbs.aabbs[k], (const Matrix3x4&)mt->transform );
                        }
                    }
                    else
                    {
                        // error
                        printf( "ERROR unknown traversable found, which is not a transform.\n" );
                    }

                    bvhtools::motion::swap( pInAabbs, pOutAabbs );
                }

                if( iasNumKeys == 1 )
                {
                    AabbArray& inAabbs = *pInAabbs;

                    for( unsigned int k = 1; k < inAabbs.size; ++k )
                    {
                        include( inAabbs.aabbs[0], inAabbs.aabbs[k] );
                    }

                    outIntervalAabbs = inAabbs.aabbs;
                }
                else
                {
                    AabbArray& inAabbs = *pInAabbs;

                    PRINTF( "pre-out inAabb.size=%i\n", inAabbs.size );
                    for( int i = 0; i < inAabbs.size; ++i )
                        PRINTF( "inAabb[%i]: %f,%f,%f - %f,%f,%f\n", i, PRINT_AABB_ARGS( inAabbs.aabbs[i] ) );

                    // there is space for at least two aabbs

                    if( inAabbs.size == 1 )
                    {
                        inAabbs.aabbs[1] = inAabbs.aabbs[0];
                        outIntervalAabbs = inAabbs.aabbs;
                    }
                    else
                    {
                        // FIXME, do a more lightweight resampling giving the fixed output of 2 aabbs?!
                        if( inAabbs.size > 2 )
                        {
                            outIntervalAabbs = pOutAabbs->aabbs;
                            resampleMotionAabbs( intervalBegin, intervalEnd, inAabbs.aabbs, inAabbs.size, intervalBegin, intervalEnd, outIntervalAabbs, 2 );
                        }
                        else
                        {
                            outIntervalAabbs = inAabbs.aabbs;
                        }
                    }
                }

                PRINTF(
                    "outIntervalAabbs:\n"
                    "  %f,%f,%f - %f,%f,%f\n"
                    "  %f,%f,%f - %f,%f,%f\n",
                    PRINT_AABB_ARGS( outIntervalAabbs[0] ), PRINT_AABB_ARGS( outIntervalAabbs[1] ) );
            };

            // compute aabbs for the fixed index interval from the CAS all the way up to the instance.
            // Only a fixed amount of memory is available, defined by the aabbs array
            for( unsigned int intervalIdx = 0; intervalIdx < numIntervals; ++intervalIdx )
            {
                computeIntervalAabbs( traversablesChain, traversablesChainLength, chainProperties, p.motionOptions, intervalIdx, aabbsData, aabbArrayMaxSize, intervalAabbs );

                if( iasNumKeys == 1 )
                {
                    PrimitiveAABB unionAabb;
                    *(Aabb*)&unionAabb.lox = intervalAabbs[0];
                    p.outMotionAabbs[idx]  = *(AABB*)&intervalAabbs[0];
                    storeUncachedAlign16( p.outFirstAabbs + idx, unionAabb );
                }
                else
                {
                    // ONLY IN RTCORE:
                    // note that outFirstAabbs stores the first motion key, outMotionAabbs stores aabbs for all but the first motion key
                    if( intervalIdx > 0 )
                        include( intervalAabbs[0], *(Aabb*)&p.outMotionAabbs[idx * (iasNumKeys) + intervalIdx] );

                    p.outMotionAabbs[idx * (iasNumKeys) + intervalIdx]   = *(AABB*)&intervalAabbs[0];
                    p.outMotionAabbs[idx * (iasNumKeys) + intervalIdx + 1] = *(AABB*)&intervalAabbs[1];

                    if( intervalIdx == 0 )
                        *(Aabb*)&p.outFirstAabbs[idx] = intervalAabbs[0];
                }
            }
        }
        return OPTIX_SUCCESS;
    };




    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    if( k_useFixedMemory.get() == true )
    {
        if( const OptixResult result = computeIASAabbsFixedMemory() )
            return result;
    }
    else if( const OptixResult result = computeIASAabbsDynamicMemory() )
    {
        return result;
    }
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // Part IV: upload & cleanup
    //////////////////////////////////////////////////////////////////////////

    if( const LWresult result = corelib::lwdaDriver().LwMemcpyHtoD(
            outputAabbBuffer, instanceAabbs.data(), numInstances * iasMotionOptions->numKeys * sizeof( Aabb ) ) )
    {
        return errDetails.logDetails( result, "Copying data to device failed." );
    }

#endif // ENABLE_HOST_VARIANT

    return OPTIX_SUCCESS;
}

}  // namespace optix_exp
