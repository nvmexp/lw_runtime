/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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

#pragma once

#include <optix_types.h>

#include <exp/accel/ExtendedAccelHeader.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

#include <lwca.h>

#include <vector>

namespace optix_exp {

class DeviceContext;

class RtcAccelBuilder
{
  public:
    RtcAccelBuilder( DeviceContext* deviceContext, const OptixAccelBuildOptions* accelOptions, bool computeMemory, ErrorDetails& errDetails );
    OptixResult init( const OptixBuildInput* buildInputs, unsigned int numBuildInputs );

    OptixResult computeMemoryUsage( OptixAccelBufferSizes* bufferSizes );

    OptixResult build( LWstream                  stream,
                       LWdeviceptr               tempBuffer,
                       size_t                    tempBufferSizeInBytes,
                       LWdeviceptr               outputBuffer,
                       size_t                    outputBufferSizeInBytes,
                       OptixTraversableHandle*   outputHandle,
                       const OptixAccelEmitDesc* emittedProperties,
                       unsigned int              numEmittedProperties );

    //----------------------------------------------------------------
    // Testing methods
    RtcAccelBuilder( DeviceContext*                context,
                     const OptixAccelBuildOptions* accelOptions,
                     bool                          computeMemory,
                     ErrorDetails&                 errDetails,
                     unsigned int                  maxPrimsPerGAS,
                     unsigned int                  maxSbtRecordsPerGAS,
                     unsigned int                  maxInstancesPerIAS,
                     int                           abiVersion,
                     bool                          hasTTU,
                     bool                          hasMotionTTU );

    const RtcAccelOptions&                      getRtcAccelOptions() const { return m_rtcAccelOptions; }
    const std::vector<RtcBuildInput>&           getBuildInputs() const { return m_vecBuildInputs; }
    const std::vector<RtcBuildInputOverrides*>& getBuildInputOverridePtrs() const
    {
        return m_vecBuildInputOverridePtrs;
    }
    const std::vector<RtcBuildInputOverrides>& getBuildInputOverrides() const { return m_vecBuildInputOverrides; }
    const RtcAccelBuffers&                     getRtcAccelBuffers() const { return rtcAccelBuffers; }
    void                                       setRtcAccelBuffers( LWdeviceptr          tempBuffer,
                                                                   size_t               tempBufferSizeInBytes,
                                                                   LWdeviceptr          outputBuffer,
                                                                   size_t               outputBufferSizeInBytes,
                                                                   ExtendedAccelHeader& extendedHeader );
    const std::vector<RtcAccelEmitDesc>&       getRtcEmittedProperties() const { return rtcEmittedProperties; }
    void setRtcEmittedProperties( const OptixAccelEmitDesc* emittedProperties, unsigned int numEmittedProperties );
    void addSizeToRtcEmittedProperties( LWdeviceptr outputBuffer, std::vector<RtcAccelEmitDesc>& rtcEmittedProperties, bool& compactedProperty );
    OptixResult validateTempBuffer( LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes, OptixAccelBufferSizes bufferSizes );
    OptixResult validateOutputBuffer( LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes, OptixAccelBufferSizes bufferSizes );
    //----------------------------------------------------------------

  private:
    OptixResult validateBuildInput( unsigned int i, const OptixBuildInput& bi, bool hasMotion, unsigned short motionNumKeys );
    OptixResult validateTrianglesBuildInput( unsigned int i, const OptixBuildInputTriangleArray& bi, unsigned short motionNumKeys );
    OptixResult validateLwrvesBuildInput( unsigned int i, const OptixBuildInputLwrveArray& bi, unsigned short motionNumKeys );
    OptixResult validateSpheresBuildInput( unsigned int i, const OptixBuildInputSphereArray& bi, unsigned short motionNumKeys );
    OptixResult validateFormatStride( unsigned int i,
                                      unsigned int strideInput,
                                      unsigned int naturalStride,
                                      unsigned int alignment,
                                      bool         vertexNotIndex,
                                      const char*  strideText,
                                      const char*  alignText,
                                      const char*  enumText );
    OptixResult validateLwstomPrimitivesBuildInput( unsigned int i, const OptixBuildInputLwstomPrimitiveArray& bi, unsigned short motionNumKeys );
    OptixResult validateMotionOptions( const OptixMotionOptions& mo );
    OptixResult validateInstancesBuildInput( unsigned int                        i,
                                             OptixBuildInputType                 type,
                                             const OptixBuildInputInstanceArray& bi,
                                             bool                                hasMotion,
                                             unsigned short                      motionNumKeys );
    OptixResult validateBuildInputType( unsigned int i, const OptixBuildInput& bi );
    OptixResult validateBuildOverridesInput( unsigned int i, const OptixBuildInput& bi );
    OptixResult validateLwrvesBuildOverridesInput( unsigned int i, const OptixBuildInput& bi, unsigned short motionNumKeys );

    OptixResult colwertTrianglesBuildInputs( const OptixBuildInput* buildInputs,
                                             unsigned int           numBuildInputs,
                                             bool                   hasMotion,
                                             unsigned short         motionNumKeys );
    OptixResult colwertBuildInput( unsigned int i, const OptixBuildInput& bi, bool hasMotion, unsigned short motionNumKeys );
    OptixResult colwertLwrvesBuildInput( const unsigned int               i,
                                         const OptixBuildInputLwrveArray& bi,
                                         const bool                       hasMotion,
                                         const unsigned short             motionNumKeys );
    OptixResult colwertSpheresBuildInput( const unsigned int                i,
                                          const OptixBuildInputSphereArray& bi,
                                          const bool                        hasMotion,
                                          const unsigned short              motionNumKeys );
    OptixResult colwertLwstomPrimitivesBuildInput( unsigned int i, const OptixBuildInputLwstomPrimitiveArray& bi, bool hasMotion );
    OptixResult colwertInstancesBuildInput( OptixBuildInputType type, const OptixBuildInputInstanceArray& bi ) noexcept;

    void getLwrveSplitFactor( const OptixBuildInput* buildInputs, unsigned int numBuildInputs );

    OptixResult logIlwalidValue( const std::string& description )
    {
        return m_errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, description );
    }

    bool buildInputsAreEmpty() const;
    bool buildInputIsEmpty( const RtcBuildInput& buildInput ) const;
    bool triangleArrayBuildInputIsEmpty( const RtcBuildInputTriangleArray& triangles ) const;
    bool aabbArrayBuildInputIsEmpty( const RtcBuildInputAabbArray& aabbArray ) const;

    // Methods for primitives with builtin intersectors, lwrves and spheres.
    OptixResult splitLwrveAdaptive( LWstream stream, LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes );
    OptixResult getLwrveAabbsAndSegments( LWdeviceptr& aabbBuffer,
                                          LWdeviceptr& segments,
                                          LWdeviceptr& indexMap,
                                          LWdeviceptr& data,
                                          LWdeviceptr& inflectionPoints,
                                          LWdeviceptr  tempBuffer,
                                          size_t       tempBufferSizeInBytes );
    OptixResult computeLwrveAabbs( LWstream                               stream,
                                   LWdeviceptr                            tempBuffer,
                                   size_t                                 tempBufferSizeInBytes,
                                   std::vector<std::vector<LWdeviceptr>>& lwrveAabbs );
    OptixResult computeLwrveLSSs( LWstream stream, LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes );
    OptixResult getSphereAabbs( LWdeviceptr& aabbBuffer, LWdeviceptr& sphereData, LWdeviceptr tempBuffer, size_t tempBufferSizeInBytes );
    OptixResult computeSphereAabbs( LWstream                               stream,
                                    LWdeviceptr                            tempBuffer,
                                    size_t                                 tempBufferSizeInBytes,
                                    std::vector<std::vector<LWdeviceptr>>& sphereAabbs );
    OptixResult initExtendedAccelHeader( LWstream stream, LWdeviceptr outputBuffer, ExtendedAccelHeader* extendedHeader );
    OptixResult appendbuiltinISData( LWstream             stream,
                                     LWdeviceptr          tempBuffer,
                                     size_t               tempBufferSizeInBytes,
                                     LWdeviceptr          outputBuffer,
                                     ExtendedAccelHeader* extendedHeader );

    DeviceContext*                m_deviceContext;
    const OptixAccelBuildOptions* m_accelOptions;
    ErrorDetails&                 m_errDetails;
    unsigned long long            m_numPrimsInGASForValidation      = 0;
    unsigned long long            m_numSbtRecordsInGASForValidation = 0;

    std::vector<RtcBuildInput>           m_vecBuildInputs;
    std::vector<RtcBuildInputOverrides*> m_vecBuildInputOverridePtrs;
    std::vector<RtcBuildInputOverrides>  m_vecBuildInputOverrides;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    std::vector<RtcBuildInputVisibilityMap> m_vecBuildInputVm;
    std::vector<RtcBuildInputDisplacement>  m_vecBuildInputDmm;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    const OptixBuildInput* m_buildInputs;
    unsigned int           m_numBuildInputs = 0;

    bool                          m_isIAS         = false;
    bool                          m_computeMemory = false;
    OptixBuildInputType           m_buildInputType{};
    OptixPrimitiveType            m_builtinPrimitiveType{};
    RtcAccelOptions               m_rtcAccelOptions{};
    RtcAccelBuffers               rtcAccelBuffers{};
    std::vector<unsigned int>     motionAabbsEmitDescIndex;
    std::vector<RtcAccelEmitDesc> rtcEmittedProperties;
    unsigned int                  m_maxPrimsPerGAS;
    unsigned int                  m_maxSbtRecordsPerGAS;
    unsigned int                  m_maxInstancesPerIAS;
    // Making a copy of the ABI version is only needed for the tests which need to work without
    // a valid DeviceContext.
    OptixABI m_abiVersion;
    bool     m_hasTTU;
    bool     m_hasMotionTTU;

    unsigned int m_numRtcLwrvesInGAS      = 0;
    unsigned int m_numRtcSpheresInGAS     = 0;
    float        m_numSplits              = 1.f;
    bool         m_lwrveAdaptiveSplitting = true;
    bool         m_builtinISLowMem        = false;
};

}  // namespace optix_exp
