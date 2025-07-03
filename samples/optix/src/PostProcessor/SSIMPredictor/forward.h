/******************************************************************************
 * Copyright 2019 LWPU Corporation. All rights reserved.
 *****************************************************************************/

#pragma once

#include <algorithm>

#include "common.h"
#include "i_ssim.h"
#include "layerdata.h"
#include "model.h"

namespace LW {
namespace SSIM {

template <typename T>
class Forward : public Forward_base
{

  public:
    // constructor
    Forward( int device, ILogger* logger = 0 );  // specify device (gpu)

    // destructor
    virtual ~Forward();

    // tile forward declaration
    typedef typename Forward_base::Forward_tile Forward_tile;

    // set LWCA stream
    virtual void set_stream( lwdaStream_t cs );

    // create DL layers for image with given resolution. The height is either the full resolution
    // or the tile height when the image is processed using horizontal stripes.
    //
    // inp_channels is the number of input channels, lwrrently onl y3 is used
    // out_channels is the number of output channels in the heatmap mask, lwrrently 1
    // mem is GPU memory limit in bytes, if not 0
    virtual void create_layers( const Layerdata& ld, int width, int height, int inp_channels, int out_channels, size_t mem, Eval_mode mode ) override;

    // delete all layers created
    virtual void delete_layers() override;

    // set all layer weights from the layer filter data
    virtual bool set_weights( const Layerdata& ld ) override;

    // get heatmap image width
    int get_heatmap_width() const { return m_output_layer ? m_output_layer->m_outWidth : 0; }

    // get heatmap image height
    int get_heatmap_height() const { return m_output_layer ? m_output_layer->m_outHeight : 0; }

    // Estimate SSIM of a normalized image in "data". feature buffers can be given in "fdata".
    // The resolution of the input image (and feature buffer) is given in width x height,
    // the number of channels in the input image is given in "image_nchannels", i.e. 3 for RGB.
    // Returns GPU inference time in milliseconds.
    virtual float predict_image( Image_buffer*       heatmap_buffer,
                                 const Image_buffer& buffer,
                                 const Image_buffer* denoised_buffer,  // denoised image, optional
                                 int                 tile_height );

    // Low-Level function for processing the layers directly. Execute forward evaluation
    // of all layers
    void run();

    // Get last predicted SSIM value
    virtual float get_ssim() { return m_classifier->get_ssim(); };

    // return time for evaluation of all layers
    float get_eval_time() const
    {
        float t = 0;
        for( auto layer : m_layers )
            t += layer->m_evaltime;
        return t;
    }

    // set algorithms for colwolution layers. the array must have length 22 for this model
    void set_colw_algorithms( const std::vector<int>& alg );

    // return true if half pipeline is used (not yet implemented)
    bool half_enabled() const;

    Eval_mode           m_mode;           // evaluation mode
    lwdnnTensorFormat_t m_tensor_format;  // LWDNN_TENSOR_NCHW or LWDNN_TENSOR_NHWC

    std::vector<Layer_base*> m_layers;

    Input_layer<T>* m_input_layer;
    Layer<T>*       m_output_layer;

    std::vector<Colwolution_layer<T>*> m_base_colw;
    std::vector<Batchnorm_layer<T>*>   m_base_bn;

    std::vector<Colwolution_layer<T>*> m_top_r_colw;
    std::vector<Batchnorm_layer<T>*>   m_top_r_bn;
    Classifier_layer<T>*               m_classifier;

    std::vector<Layer_base*>         m_top_h_colw;  // may combine half and float
    std::vector<Batchnorm_layer<T>*> m_top_h_bn;

    unsigned int m_inp_nchannels;
    unsigned int m_out_nchannels;


    void* m_copy_buffer;

    std::vector<T*> m_layer_buffers;  // layer buffers

    std::vector<int> m_colw_algorithms;

  private:
    std::vector<int> get_algo_vector() const;

    void fill_layers( const Layerdata& ld, int width, int height, int inp_nchannels, int out_nchannels, bool alloc );

    void clear_layers();
};

// Iteration predictor
template <typename T>
class Iter_forward : public Compute_base
{
  public:
    // requires at least this many iteration powee of 2 inputs
    static const int s_input_min_size = 6;

    // max of input iteration power of 2's
    static const int s_input_max_size = 16;

    // twice the max size
    static const int s_input_buf_size = 32;

    // constructor
    Iter_forward( const Iter_layerdata& ld,
                  int                   device,  // specify device (gpu)
                  ILogger*              logger = 0 );

    // destructor
    virtual ~Iter_forward();

    // set all layer weights from the layer filter data
    bool set_weights( const Iter_layerdata& ld );

    // set LWCA stream
    virtual void set_stream( lwdaStream_t cs );

    // inference time
    float get_eval_time() const;

    // Run iteration predictor.
    // Input is an array of floating SSIM values after 1, 2, 4, .... 2^k iterations, k>=6, k <= 16.
    // Output is log_2 of the iteration expected to achieve 98%.
    float run( float* val, int n_val, float target );


    // fixed target, if not 0.0f
    float m_target;
    float m_power;

    // layers
    Input_layer<T>                   m_input_layer;
    std::vector<Fc_layer<T>>         m_fc;
    std::vector<Activation_layer<T>> m_act;

    // buffers
    void* m_fc_out;   // m_fc output buffer
    void* m_act_out;  // m_act output buffer
};


typedef Forward<__half> Forward16;
typedef Forward<float>  Forward32;

typedef Iter_forward<__half> Iter_forward16;
typedef Iter_forward<float>  Iter_forward32;


}  // namespace SSIM
}  // namespace LW
