//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#pragma once

#ifndef IRAY_BUILD
#include <corelib/misc/String.h>
#include <exp/context/ErrorHandling.h>
#endif

#include "layerdata_load.h"
#include "layers.h"

namespace optix_exp {

class DeviceContextLogger;

class Denoise
{
  public:
    Denoise( DeviceContextLogger& logger );
    ~Denoise();

    OptixResult init( ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    // blend denoised result with input image
    // r = f * max(input,1) + (1 - f) * denoised
    void setBlendFactor( float f ) { m_blendFactor = f; }

    // non-KPN: set RGB scale in HDR mode, device memory pointer
    void setHdrIntensityFactor( float* f ) { m_intensityFactor = f; }

    // KPN: set hdr per channel log average: 3 floats for RGB
    void setHdrAverageColor( float* f ) { m_hdrAverageColor = f; }

    void setDenoiseAlpha( OptixDenoiserAlphaMode alphaMode ) { m_denoiseAlpha = alphaMode; }

    void setWeights( Layerdata& ld ) { m_layerData = ld; }

    OptixResult createLayers( int width, int height, int inChannels, ErrorDetails& errDetails );
    OptixResult deleteLayers( ErrorDetails& errDetails );
    OptixResult initializeWeights( void* smem, size_t smemSize, void* wmem, size_t wmemSize, lwdaStream_t stream, ErrorDetails& errDetails );

    OptixResult denoiseTile( const OptixImage2D* inputLayers,
                             unsigned int        numInputLayers,
                             const OptixImage2D* flow,
                             const OptixImage2D* outputLayer,
                             const OptixImage2D* hiddenLayerIn,
                             const OptixImage2D* hiddenLayerOut,
                             unsigned int        outputOffsetX,
                             unsigned int        outputOffsetY,
                             void*               scratch,
                             size_t              scratchSize,
                             lwdaStream_t        stream,
                             ErrorDetails&       errDetails );

    // return time for evaluation of all layers
    float getEvalTime() const
    {
        float t = 0;
#ifdef DENOISE_DEBUG_TIMING
        for( AELayer* l : m_layers )
            t += l->getEvalTime();
        t += m_kpnTime;
#endif
        return t;
    }

    bool halfEnabled() const { return m_deviceInfo.m_device_capability >= 60; }

    OptixResult callwlateMemoryResources( int width, int height, int inChannels, int outChannels, size_t* smem, size_t* wmem, ErrorDetails& errDetails ) const;

    // Overlap of denoise images when tiling is used (max for direct and kernel prediction)
    unsigned int getOverlapSize() const { return 128; }

    // Number of hidden channels in kpn5
    unsigned int getNumHiddenChannels() const { return m_layerData.getNumHiddenChannels(); }

    AELayer&     getLayer( int l ) { return *m_layers[l]; }
    unsigned int getNumLayers() const { return m_layers.size(); }

  private:
    DeviceContextLogger& m_logger;

    float  m_blendFactor;
    float* m_intensityFactor;        // device pointer to float
    size_t m_intensityIndex;         // smem offset to float
    float* m_hdrAverageColor;        // device pointer to 3 float
    size_t m_hdrTransformIndex;      // smem offset to 9 floats
    size_t m_kpnScratchOutIndex;     // KPN pass
    float* m_kpnModelAB;             // device pointer to six floats, model A,B parameters

    OptixDenoiserAlphaMode m_denoiseAlpha;

    std::vector<int> m_colwAlgorithms;
    deviceInfo       m_deviceInfo;

    Layerdata m_layerData;

    std::vector<AELayer*>            m_layers;
    std::vector<AEColwolutionLayer*> m_colwLayers;
    AEInputLayer*                    m_inputLayer;
    AELayer*                         m_outputLayer;
    size_t                           m_sizeSmemRequired;
    size_t                           m_sizeWmemRequired;
#ifdef DENOISE_DEBUG_TIMING
    float                            m_kpnTime;
#endif

  private:
    OptixResult allocateLayers( std::vector<AELayer*>&            layers,
                                std::vector<AEColwolutionLayer*>& colwLayers,
                                int                               width,
                                int                               height,
                                int                               inChannels,
                                int                               outChannels,
                                ErrorDetails&                     errDetails ) const;

    AELayer* findLayer( const std::vector<AELayer*>& layers, const char* name ) const
    {
        for( AELayer* layer : layers )
            if( !strcmp( layer->name(), name ) )
                return layer;
        return nullptr;
    }

    AELayer* findLayer( const char* name ) const
    {
        return findLayer( m_layers, name );
    }

    OptixResult initializeHDR( const OptixImage2D*, void* smem, size_t smemSize, lwdaStream_t stream, ErrorDetails& errDetails );

    OptixResult runAOV( const OptixImage2D* inputLayer,
                        const OptixImage2D* outputLayer,
                        const OptixImage2D* prevDenoised,
                        const OptixImage2D* flow,
                        AEKernelpredictionReformat* refl[5],
                        int                 inOp,
                        int                 outOp,
                        unsigned int        inputOffsetX,
                        unsigned int        inputOffsetY,
                        void*               scratch,
                        size_t              scratchSizeInBytes,
                        lwdaStream_t        stream,
                        ErrorDetails&       errDetails );

    OptixResult run( void* smem, size_t smemSize, lwdaStream_t stream, ErrorDetails& errDetails );
};

};  // namespace optix_exp
