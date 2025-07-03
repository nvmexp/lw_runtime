//
// Copyright 2020 LWPU Corporation. All rights reserved.
//
// Auto-encoder layer helper classes

#pragma once

#include <algorithm>
#include <cmath>

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#ifndef IRAY_BUILD
#include <corelib/misc/String.h>
#include <exp/context/ErrorHandling.h>
#endif

#include "fp16_emu.h"
#include <string>
#include <vector>

namespace optix_exp {

static inline unsigned int roundUp( unsigned int nominator, unsigned int denominator )
{
    return ( nominator + denominator - 1 ) / denominator;
}

struct deviceInfo
{
    int m_lwMaxThreadsPerMultiProcessor;
    int m_lwMaxThreadsPerBlock;
    int m_lwMultiProcessorCount;
    int m_device_capability;
};

enum DataType : unsigned int
{
    DATA_FLOAT = 0,
    DATA_HALF  = 1
};

enum TensorFormat
{
    TENSOR_NHWC = 0,
    TENSOR_NCHW = 1
};

struct AELayer
{
    AELayer( const char* name, const deviceInfo& info );

    virtual ~AELayer();

    OptixResult init( ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    // copy layer output to result, RGB formatting and apply transformations
    void copyOutput( const OptixImage2D* result,
                     const OptixImage2D* input,
                     unsigned int        inputOffsetX,
                     unsigned int        inputOffsetY,
                     int                 srcChannel,
                     int                 numChannels,
                     int                 op,
                     void*               smem,
                     lwdaStream_t        stream,
                     bool                upscale2X,
                     float               blendFactor = 0.f );

    virtual OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails )            = 0;
    virtual OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) = 0;

    const char* name() const { return m_name.c_str(); }

    virtual bool needsZeroInit() const { return false; }

    void setHdrScale( float log_scale )
    {
        m_expmul = logf( log_scale );
        m_logmul = 1.f / m_expmul;
    }

    unsigned int getBufferTypeSize() const { return m_dtype == DATA_HALF ? 2 : 4; }

    void getBlockCount( int& bc, int& tpb, int elem ) const
    {
        int ptc = std::min( m_deviceInfo.m_lwMaxThreadsPerMultiProcessor* m_deviceInfo.m_lwMultiProcessorCount, elem );
        tpb     = std::min( 1024, m_deviceInfo.m_lwMaxThreadsPerBlock );
        bc      = std::min( ( ptc + tpb - 1 ) / tpb, m_deviceInfo.m_lwMultiProcessorCount );
    }

#ifdef DENOISE_DEBUG_TIMING
    float getEvalTime() const { return m_evaltime; }
#else
    float getEvalTime() const { return 0.f; }
#endif

    void   setOutBufferShared( bool shared ) { m_outbufferShared = shared; }
    bool   isOutBufferShared() const { return m_outbufferShared; }
    size_t getOutSize() const { return m_outsize; }
    void   setOutDataIndex( size_t index ) { m_outDataIndex = index; }
    size_t getOutDataIndex() const { return m_outDataIndex; }
    void   setWorkmemIndex( size_t index ) { m_workMemIndex = index; }
    void   setWorkmemSize( size_t sz ) { m_workMemSize = sz; }
    void   setIntensityIndex( size_t index ) { m_intensityIndex = index; }
    void   setFused( bool f ) { m_fused = f; }
    bool   isFused() const { return m_fused; }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Members

    const AELayer* m_input;
    AELayer*       m_nextLayer;

    // Size
    unsigned int m_width;  // unmodified resolution
    unsigned int m_height;
    unsigned int m_outWidth;  // res padded to multiples of 32
    unsigned int m_outHeight;
    unsigned int m_outChannels;

    // byte offset relative to scratch memory, output memory
    size_t   m_outDataIndex;

  protected:
    size_t   m_outsize;
    size_t   m_workMemIndex;
    size_t   m_workMemSize;
    bool     m_outbufferShared;
    float    m_logmul;
    float    m_expmul;
    size_t   m_intensityIndex;
    DataType m_dtype;
    bool     m_tc;

    std::string       m_name;
    TensorFormat      m_tensorFormat;
    const deviceInfo& m_deviceInfo;
    bool              m_fused;

#ifdef DENOISE_DEBUG_TIMING
    // timing per layer
    lwdaEvent_t m_startEvent;
    lwdaEvent_t m_stopEvent;
    float       m_evaltime;
    void        beginTiming() { lwdaEventRecord( m_startEvent ); }
    void        endTiming()
    {
        lwdaEventRecord( m_stopEvent );
        lwdaEventSynchronize( m_stopEvent );
        lwdaEventElapsedTime( &m_evaltime, m_startEvent, m_stopEvent );
    }
#else
    void  beginTiming() {}
    void  endTiming() {}
#endif
};

struct AEInputLayer : public AELayer
{
    AEInputLayer( const char* name, const deviceInfo& info );

    ~AEInputLayer();

    OptixResult init( int width, int height, int channels, int minsize, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override { return OPTIX_SUCCESS; }

    void setChannelRGB( const OptixImage2D* data, int where, int op, void* smem, lwdaStream_t stream );
    void setChannelMotion( const OptixImage2D* prev_denoised, const OptixImage2D* motion, int start, int nChannels, int op, void* smem, lwdaStream_t stream );
    OptixResult setBuffer( void* smem, lwdaStream_t stream, float value, ErrorDetails& errDetails );

    void setHdrTransformIndex( size_t index ) { m_hdrTransformIndex = index; }

    virtual bool needsZeroInit() const override { return bool( m_tensorFormat == TENSOR_NHWC ); }

private:
    size_t   m_hdrTransformIndex;
};

struct Xmma_colw;

struct AEColwolutionLayer : public AELayer
{
    enum Activation : unsigned int
    {
        ACTIVATION_NONE,
        ACTIVATION_RELU,
        ACTIVATION_LEAKY_RELU,
    };

    AEColwolutionLayer( const char* name, const deviceInfo& info );

    ~AEColwolutionLayer();

    OptixResult init( const AELayer* input, unsigned int kernelSize, unsigned int outChannels, Activation activation, float alpha, ErrorDetails& errDetails );

    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult setWeights( void* smem, const std::vector<short>& weights, const unsigned int tdim[4], lwdaStream_t stream, ErrorDetails& errDetails ) const;  // actually __half

    OptixResult setBias( const std::vector<short>& bias, unsigned int dim, lwdaStream_t stream, ErrorDetails& errDetails ) const;  // actually __half

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override;

    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

    void   setFilterWeightsPtr( void* ptr ) { m_filterWeights = ptr; }
    void   setBiasPtr( void* ptr ) { m_bias = ptr; }
    size_t getFilterWeightsSize() const { return m_wsize; }
    size_t getBiasSize() const { return m_bsize; }

    void   setupXMMA( int arch, int M, int N, int K );

  private:
    unsigned int        m_kernelSize;
    Activation          m_activ;
    float               m_alpha;

    void*               m_filterWeights;
    void*               m_bias;
    size_t              m_wsize;
    size_t              m_bsize;

    Xmma_colw*          m_xmmaColw;

    void NCHWtoNHWC( void* dst, const void* src, unsigned int n, unsigned int cout, unsigned int c, unsigned int h, unsigned int w, lwdaStream_t stream ) const;
};

struct AEUpscaleConcat : public AELayer
{
    AEUpscaleConcat( const char* name, const deviceInfo& info );
    ~AEUpscaleConcat();

    OptixResult init( const AELayer* decoder, const AELayer* skip, bool bilinear, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    const AELayer* m_decoder;
    const AELayer* m_skip;
    bool           m_bilinearScale;
};

struct AEUpscale : public AELayer
{
    AEUpscale( const char* name, const deviceInfo& info );
    ~AEUpscale();

    OptixResult init( const AELayer* input, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    const AELayer* m_input;
};

struct AEUpscaleSum : public AELayer
{
    AEUpscaleSum( const char* name, const deviceInfo& info );
    ~AEUpscaleSum();

    OptixResult init( const AELayer* decoder, const AELayer* skip, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    const AELayer* m_decoder;
    const AELayer* m_skip;
};

struct AEPoolingLayer : public AELayer
{
    AEPoolingLayer( const char* name, const deviceInfo& info );
    ~AEPoolingLayer();

    OptixResult init( const AELayer* input, bool avgPooling, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    bool m_avgPooling;
};

struct AESpaceToDepth : public AELayer
{
    AESpaceToDepth( const char* name, const deviceInfo& info );
    ~AESpaceToDepth();

    OptixResult init( const AELayer* input, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

    virtual bool needsZeroInit() const override { return bool( m_tensorFormat == TENSOR_NHWC ); }
};

struct AEDepthToSpace : public AELayer
{
    AEDepthToSpace( const char* name, const deviceInfo& info );
    ~AEDepthToSpace();

    OptixResult init( const AELayer* input, unsigned int outChannels, unsigned int hiddenChannels, ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    // write hidden channels to a separate image instead of regular output tensor (only for NHWC)
    void setHiddenOutput( const OptixImage2D* hiddenOut, unsigned inputOffsetX, unsigned int inputOffsetY )
    {
        m_hiddenOut    = *hiddenOut;
        m_inputOffsetX = inputOffsetX;
        m_inputOffsetY = inputOffsetY;
    }
    // returns true if the implementation supports separate write of hidden channels
    bool writesHiddenSeparate() const { return bool( m_tensorFormat == TENSOR_NHWC ); }

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    OptixImage2D m_hiddenOut;
    unsigned int m_inputOffsetX;
    unsigned int m_inputOffsetY;
};

struct AEKernelpredictionReformat : public AELayer
{
    AEKernelpredictionReformat( const char* name, const deviceInfo& info );
    ~AEKernelpredictionReformat();

    OptixResult init( const AELayer* input,
                      bool clipWeights,
                      ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

    unsigned int m_filterZeroChannels;
    unsigned int m_aChannel;
    bool m_clipWeights;
};

struct AEWeightedFilterForward : public AELayer
{
    AEWeightedFilterForward( const char* name, const deviceInfo& info );
    ~AEWeightedFilterForward();

    OptixResult init( const AELayer* input,
                      const AEKernelpredictionReformat* weights,
                      const AELayer* pDenoised,
                      unsigned int kernelSize,
                      ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails );

  private:
    const AEKernelpredictionReformat* m_wl;
    const AELayer*                    m_pDenoised;
    unsigned int m_kernelSize;
};

struct AEKernelpredictionUpsample : public AELayer
{
    AEKernelpredictionUpsample( const char* name, const deviceInfo& info );
    ~AEKernelpredictionUpsample();

    OptixResult init( const AELayer* input,
                      const AELayer* lwrFiltered,
                      const AEKernelpredictionReformat* alpha,
                      ErrorDetails& errDetails );
    OptixResult destroy( ErrorDetails& errDetails );

    OptixResult getWorkspaceSize( size_t& wss, ErrorDetails& errDetails ) override
    {
        wss = 0;
        return OPTIX_SUCCESS;
    }
    OptixResult fwdEval( void* smem, lwdaStream_t stream, ErrorDetails& errDetails ) override;

  private:
    const AELayer*                    m_lwrFiltered;
    const AEKernelpredictionReformat* m_alpha;
    bool  m_lastLayer;
};

};  // namespace optix_exp
