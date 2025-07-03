/*
 * Copyright 2020 LWPU Corporation. All rights reserved.
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

#include <exp/context/ErrorHandling.h>
#include <exp/context/OpaqueApiObject.h>
#include <exp/functionTable/optix_host_denoiser_v1.h>

#include <exp/denoise/inference.h>

namespace optix_exp {

class DeviceContext;
class DeviceContextLogger;

class Denoiser : public OpaqueApiObject
{
  public:
    Denoiser( DeviceContext* context );

    OptixResult initPixelFormat( const OptixDenoiserOptions* options, ErrorDetails& errDetails );
    OptixResult initModelDimension( const Layerdata & ld, ErrorDetails& errDetails );
    OptixResult init( const OptixDenoiserOptions* options, const void * userData, size_t userDataSizeInBytes, ErrorDetails& errDetails );
    OptixResult init_v1( OptixDenoiserInputKind_v1 inputKind, ErrorDetails& errDetails );

    OptixResult destroy( bool doUnregisterDenoiser, ErrorDetails& errDetails );

    OptixResult destroyWithoutUnregistration( ErrorDetails& errDetails ) { return destroy( false, errDetails ); }

    ~Denoiser() {}

    const char* getTrainingName( OptixDenoiserModelKind modelKind, OptixDenoiserInputKind_v1 inputKind )
    {
        if( !(modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL ||
              modelKind == OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV ) )
        {
            switch( inputKind )
            {
                case OPTIX_DENOISER_INPUT_RGB:
                    return "rgb";
                case OPTIX_DENOISER_INPUT_RGB_ALBEDO:
                    return "rgb-albedo";
                case OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL:
                    return modelKind == OPTIX_DENOISER_MODEL_KIND_UPSCALE2X ? "rgb-albedo-normal3" : "rgb-albedo-normal";
            }
        }
        else
        {
            switch( inputKind )
            {
                case OPTIX_DENOISER_INPUT_RGB:
                    return "rgb-flow";
                case OPTIX_DENOISER_INPUT_RGB_ALBEDO:
                    return "rgb-albedo-flow";
                case OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL:
                    return "rgb-albedo-normal3-flow";
            }
        }
        return 0;  // never reached
    }

    // set from tensor in weights data, return total number of channels for input tensor
    int getInputChannels() const
    {
        return m_modelDimension;
    }

    // return number of layers required for current inference mode
    int getInputLayers() const
    {
        int il = isTemporalModel() ? 1 : 0;
        switch( m_inferenceMode )
        {
            case OPTIX_DENOISER_INPUT_RGB:
                il += 1;
                break;
            case OPTIX_DENOISER_INPUT_RGB_ALBEDO:
                il += 2;
                break;
            case OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL:
                il += 3;
                break;
        }
        return il;
    }

    OptixDenoiserInputKind_v1 getInputKind() const
    {
        return m_inferenceMode;
    }

    bool isKernelPrediction() const
    {
        return m_kernelPrediction;
    }

    bool isTemporalModel() const
    {
        return m_temporalModel;
    }

    bool isHdrModel() const
    {
        return m_hdrModel;
    }

    DeviceContextLogger& getLogger();

    DeviceContext* getDeviceContext() const { return m_context; }

    std::mutex& getMutex() { return m_denoiserMutex; }

    // To be used by DeviceContext only.
    struct DeviceContextIndex_fn
    {
        int& operator()( const Denoiser* denoiser ) { return denoiser->m_deviceContextIndex; }
    };

    OptixResult getTrainingNameForUserModel( const void* tdata, size_t sizeInBytes, std::string& trName, ErrorDetails& errDetails );
    OptixResult getInputKindForUserModel( const void* tdata, size_t sizeInBytes, OptixDenoiserInputKind_v1 & inputKind, ErrorDetails& errDetails );
    OptixResult initBuiltinModel( OptixDenoiserModelKind modelKind, const OptixDenoiserOptions* options, ErrorDetails& errDetails );
    OptixResult loadBuiltinModel( OptixDenoiserModelKind modelKind, ErrorDetails& errDetails );
    OptixResult initUserModel( OptixDenoiserInputKind_v1 inputKind, const void* data, size_t sizeInBytes, ErrorDetails& errDetails );
    OptixResult loadUserModel( const void* data, size_t sizeInBytes, ErrorDetails& errDetails );

    Denoise* m_denoiser;

  private:
    DeviceContext*            m_context;
    OptixDenoiserInputKind_v1 m_inferenceMode;
    unsigned int              m_modelDimension;
    bool                      m_kernelPrediction;               // from weights data file/memory
    bool                      m_temporalModel;                  // from weights data file/memory
    bool                      m_hdrModel;
    std::mutex                m_denoiserMutex;
    mutable int               m_deviceContextIndex = -1;
};

}  // namespace optix_exp
