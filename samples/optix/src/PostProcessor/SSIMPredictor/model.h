/******************************************************************************
 * Copyright 2019 LWPU Corporation. All rights reserved.
 *****************************************************************************/

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunreachable-code"
#endif

#include <lwdnn.h>
#include <string>
#include <vector>

#ifndef LWDA_BLOCK
#define LWDA_BLOCK 32
#endif

#include "common.h"
#include "fp16_dev.h"
#include "i_ssim.h"


#undef LWDNN_DTYPE


namespace LW {
namespace SSIM {

// base shared by half and float
struct Layer_base : public AI_error
{
    Layer_base( const char* name, lwdnnHandle_t lwdnn );

    virtual ~Layer_base();

    // get layer name
    const char* name() const { return m_name.c_str(); }

    // half-based?
    virtual bool is_half() const = 0;

    // GPU weight size
    virtual size_t get_weights_size( Eval_mode mode ) const = 0;

    // colwolution weights, relevant for colwolutions only
    virtual bool set_weights( const std::vector<float>& weights,
                              // optional: no bias if empty
                              const std::vector<float>& bias )
    {
        return false;
    }

    // batchnorm weights, relevant for BN only
    virtual bool set_weights( const std::vector<float>& alpha,
                              const std::vector<float>& beta,
                              const std::vector<float>& mean,
                              const std::vector<float>& var,
                              float                     eps )
    {
        return false;
    }

    // get output layer size, in bytes
    virtual size_t get_out_size() const = 0;

    // colwolution workspace
    virtual size_t get_workspace_size( Eval_mode mode ) = 0;

    // forward evaluation step
    virtual void fwd_eval( void* workspace, size_t workspaceSize ) = 0;


    lwdnnHandle_t m_lwdnn;
    lwdaStream_t  m_stream;

    const Layer_base* m_input;

    lwdnnTensorFormat_t m_tensor_format;  // LWDNN_TENSOR_NCHW or LWDNN_TENSOR_NHWC

    unsigned int m_width;  // unmodified resolution
    unsigned int m_height;
    unsigned int m_outWidth;  // res padded to multiples of 32
    unsigned int m_outHeight;
    unsigned int m_outChannels;
    unsigned int m_outPad;  // pad channels added for performance

    lwdnnTensorDescriptor_t m_outTensorDesc;
    void*                   m_outData;

    std::string m_name;

    bool m_multi_output;  // output used as input to multiple layers

    // timing per layer
    lwdaEvent_t m_start_event;
    lwdaEvent_t m_stop_event;
    float       m_evaltime;
};


template <typename T>
struct Layer : public Layer_base
{
    // lwdnn tensor data type to use
    const lwdnnDataType_t LWDNN_DTYPE = ( sizeof( T ) == 2 ) ? LWDNN_DATA_HALF : LWDNN_DATA_FLOAT;

    Layer( const char* name, lwdnnHandle_t lwdnn );

    virtual ~Layer();

    // detect if layer is half-based
    virtual bool is_half() const override { return LWDNN_DTYPE == LWDNN_DATA_HALF; };

    // return layer output size in bytes
    virtual size_t get_out_size() const
    {
        return sizeof( T ) * this->m_outChannels * this->m_outWidth * this->m_outHeight;
    }

    // copy output from GPU to result buffer (which has resolution width x height),
    // into result line y, number of lines = h, number of channels copied nchannel.
    // If in channel number is not equal to the out channel number, each of the
    // output channels is filled with identical values or the average of the in channels.
    // With only valid pairs (1,3) and (3,1), only one of those ops actually applies.
    template <typename Out>
    void copy_output( Out* result,
                      bool tohost,  // copy to host buffer?
                      Out* copybuf,
                      int  width,
                      int  height,
                      int  inc,        // in-channels, 1 or 3
                      int  outc,       // out-channels, 1 or 3
                      int  out_field,  // 1 for nchw, 3 or more for nhwc
                      int  y,
                      int  h,
                      bool apply_clip );  // clip to [0,1]

    // function for coping and colwersion of weights
    lwdaError_t copy_weight_data( T*           dst,      // device
                                  const float* src,      // host, typically floats
                                  size_t       count );  // number of elements

    // function for coping of weight if they keep float type, like batchnorm
    lwdaError_t copy_float_weight_data( float*       dst,      // device
                                        const float* src,      // host, typically floats
                                        size_t       count );  // number of elements

    // assure copy buffer size if at least this large
    lwdaError assure_buffer_size( size_t size );

    // Members

    // 8-bit support
    unsigned char* m_scratch_buffer;
    unsigned int   m_scratch_size;

    // Static scratch buffer. Note: for optimization reason, not thread-safe
    static void*  s_device_buffer;
    static size_t s_device_buffer_size;
};


template <typename T>
struct Input_layer : public Layer<T>
{
    Input_layer( lwdnnHandle_t lwdnn, int width, int height, int channels, bool pad, bool alloc, ILogger* logger );

    ~Input_layer();

    size_t get_weights_size( Eval_mode mode ) const override { return 0; }

    size_t get_workspace_size( Eval_mode mode ) override { return 0; }

    void fwd_eval( void* workspace, size_t workspaceSize ) override {}

    // copy horizontal rows of the data to the GPU for this layer.
    // data is copied into channel beginning at start, number of channels is nchannel
    // (i.e. 3 for RGB)
    // resolution of the data buffer is width x height
    // the data must be in RRRGGG.. layout
    void set_channel( const Image_buffer& buf,
                      void*               tmp,
                      int                 offset,  // input data offset
                      int                 start,
                      int                 nchannel,
                      int                 width,          // actual (tile) input width
                      int                 height,         // actual (tile) input height
                      int                 data_height );  // full res if tiled

    // same as above, but data has the layout RGBRGB..
    // tmp is the copy buffer on the GPU (see create_layer)
    void set_channel_rgb( const Image_buffer& buf,
                          void*               tmp,
                          int                 offset,  // input data offset
                          int                 start,
                          int                 nchannel,
                          int                 width,     // actual input width
                          int                 height );  // actual input height

  private:
    // copy horizontal rows of the data to the GPU for this layer.
    // data is copied into channel beginning at start, number of channels is nchannel
    // (i.e. 3 for RGB)
    // resolution of the data buffer is width x height
    // the data must be in RRRGGG.. layout
    template <typename In>
    void set_channel_t( const In* data,
                        In*       tmp,
                        int       start,
                        int       nchannel,
                        int       width,          // actual (tile) input width
                        int       height,         // actual (tile) input height
                        int       data_height );  // full res if tiled

    // same as above, but data has the layout RGBRGB..
    // tmp is the copy buffer on the GPU (see create_layer)
    template <typename In>
    void set_channel_rgb_t( const In* data,
                            In*       tmp,
                            int       start,
                            int       nchannel,
                            int       width,     // actual input width
                            int       height );  // actual input height
};

template <typename T>
struct Colwolution_layer : public Layer<T>
{
    static const char* algo_name( lwdnnColwolutionFwdAlgo_t algo );

    Colwolution_layer( const char*       name,
                       int               algorithm,
                       lwdnnHandle_t     lwdnn,
                       const Layer_base* input,
                       unsigned int      kernelSize,
                       unsigned int      padding,
                       unsigned int      stride,
                       bool              depthwise,
                       unsigned int      outChannels,
                       bool              bias,
                       bool              alloc,
                       ILogger*          logger );

    ~Colwolution_layer();

    // colwolution weights, relevant for colwolutions only
    using Layer_base::set_weights;
    bool set_weights( const std::vector<float>& weights,
                      // optional: no bias if empty
                      const std::vector<float>& bias ) override;

    size_t get_weights_size( Eval_mode mode ) const override;

    size_t get_workspace_size( Eval_mode mode ) override;

    void fwd_eval( void* workspace, size_t workspaceSize ) override;


    unsigned int m_kernelSize;
    unsigned int m_padding;
    unsigned int m_stride;
    unsigned int m_groupCount;

    lwdnnFilterDescriptor_t      m_filterDesc;
    lwdnnColwolutionDescriptor_t m_colwDesc;
    lwdnnColwolutionFwdAlgo_t    m_colwFwdAlgo;

    T*   m_filterWeights;
    bool m_has_bias;
    T*   m_bias;
};

template <typename T>
struct Classifier_layer : public Colwolution_layer<T>
{
    // Constructor
    Classifier_layer( const char* name, int algo, lwdnnHandle_t lwdnn, const Layer_base* in, bool alloc, ILogger* logger )
        : Colwolution_layer<T>( name, algo, lwdnn, in, 1, 1, 0, false, 1, true, alloc, logger )
        , m_last_ssim( -1.0f )
    {
    }

    // forward evaluation, with a side effect of keeping the ssim value
    void fwd_eval( void* workspace, size_t workspaceSize ) override;

    //  Last computed ssim value. Keep it that way to keep fwd_eval interface compatible
    float get_ssim() { return m_last_ssim; }
  private:
    float m_last_ssim;
};

template <typename T>
struct Batchnorm_layer : public Layer<T>
{
    Batchnorm_layer( const char* name, lwdnnHandle_t lwdnn, const Layer_base* input, bool alloc, ILogger* logger );

    ~Batchnorm_layer();

    using Layer_base::set_weights;
    virtual bool set_weights( const std::vector<float>& alpha,
                              const std::vector<float>& beta,
                              const std::vector<float>& mean,
                              const std::vector<float>& var,
                              float                     eps ) override;

    size_t get_weights_size( Eval_mode mode ) const override;

    size_t get_workspace_size( Eval_mode mode ) override;

    void fwd_eval( void* workspace, size_t workspaceSize ) override;

    unsigned int m_kernelSize;

    lwdnnTensorDescriptor_t m_bnDesc;

    // Batch norm weights are float, never half according to lwdnnDeriveBNTensorDescriptor docs
    float* m_alpha;
    float* m_beta;
    float* m_mean;
    float* m_var;
    double m_eps;  // batchnorm epsilon
};

// Upscale + concatenation layer
template <typename T>
struct Upscale_concat_layer : public Layer<T>
{
    // Constructor
    Upscale_concat_layer( const char* name, lwdnnHandle_t lwdnn, const Layer_base* input, const Layer_base* skip, bool alloc, ILogger* logger );

    // destructor
    ~Upscale_concat_layer();

    size_t get_weights_size( Eval_mode mode ) const override { return 0; }

    size_t get_workspace_size( Eval_mode mode ) override { return 0; }

    void fwd_eval( void* workspace, size_t workspaceSize ) override;


    const Layer_base* m_input;
    const Layer_base* m_skip;
};

// Pooling layer
template <typename T>
struct Avg_pooling_layer : public Layer<T>
{
    Avg_pooling_layer( const char* name, lwdnnHandle_t lwdnn, const Layer_base* input, int outWidth, int outHeight, bool alloc, LW::SSIM::ILogger* logger );
    ~Avg_pooling_layer();

    size_t get_weights_size( Eval_mode mode ) const override { return 0; }

    size_t get_workspace_size( Eval_mode mode ) override { return 0; }

    void fwd_eval( void* workspace, size_t workspaceSize ) override;

    lwdnnPoolingDescriptor_t m_desc;

  private:
    float* m_input_buf;       // input buffer colwersion
    float* m_input_buf_sums;  // input buffer partial sums
};

// Activation layer
template <typename T>
struct Activation_layer : public Layer<T>
{
    Activation_layer( const char*           name,
                      lwdnnHandle_t         lwdnn,
                      const Layer<T>*       input,
                      lwdnnActivationMode_t activation,
                      bool                  alloc,
                      LW::SSIM::ILogger*    logger );
    ~Activation_layer();

    size_t get_weights_size( Eval_mode mode ) const override { return 0; }

    size_t get_workspace_size( Eval_mode mode ) override { return 0; }

    void fwd_eval( void* workspace, size_t workspaceSize ) override;

    lwdnnActivationMode_t       m_activation_mode;
    lwdnnActivationDescriptor_t m_desc;
};

// NHWC / NCHW Colwert layer, for IO & testing.
template <typename T>
struct Colwert_layer : public Layer<T>
{
    Colwert_layer( const char* name, lwdnnHandle_t lwdnn, const Layer_base* input, bool to_nhwc, bool alloc, LW::SSIM::ILogger* logger );
    ~Colwert_layer();

    size_t get_weights_size( Eval_mode mode ) const override { return 0; }

    size_t get_workspace_size( Eval_mode mode ) override { return 0; }

    void fwd_eval( void* workspace, size_t workspaceSize ) override;

    bool m_to_nhwc;
};


// Fully connected layer, seen as colwolutions on 1x1 images
template <typename T>
struct Fc_layer : public Colwolution_layer<T>
{
    Fc_layer( const char* name, lwdnnHandle_t lwdnn, const Layer<T>* input, int n_out, bool alloc, LW::SSIM::ILogger* logger )
        : Colwolution_layer<T>( name, 0, lwdnn, input, 1, 1, 0, false, n_out, true, alloc, logger )
    {
    }

    ~Fc_layer() {}
};


// auxiliary colwersion template
// this is probably the only place where fp1 emulation is used
template <typename T>
inline float to_float( const T& t )
{
    return t;
}

template <>
inline float to_float( const __half& t )
{
    return cpu_half2float( t );
}

template <typename T>
inline void from_float( T& t, float v )
{
    t = v;
}

template <>
inline void from_float( __half& t, float v )
{
    t = cpu_float2half_rn( v );
}


}  // namespace SSIM
}  // namespace LW
