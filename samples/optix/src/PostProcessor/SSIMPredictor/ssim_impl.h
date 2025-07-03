//*****************************************************************************
// Copyright 2018 LWPU Corporation. All rights reserved.
//*****************************************************************************
/// \file
/// \brief SSIM prediction API implementation
///
//*****************************************************************************

#pragma once

#include <functional>

#include "forward.h"
#include "i_ssim.h"
#include "layerdata.h"

namespace LW {
namespace SSIM {


// On-demand SSIM predictor class implementation.
class Ssim : public ISsim
{
  private:
    ILogger*      m_logger;        // logger
    Layerdata     m_layer_data;    // deserialized training data
    Forward_base* m_forward;       // forward predictor
    int           m_used_lwda_id;  // LWCA id of device used to initialize predictor,
                                   // -1 for none
    int m_rup_height;              // rounded up height
    int m_rup_width;               // rounded up width
    int m_rup_tile_height;         // rounded up tile height, if not zero

    bool m_denoised;  // use for denoised images

    float m_maxmem;  // maximal GPU memory to use

    Eval_mode m_mode;  // model evaluation mode

    // Iteration prediction part
    Iter_layerdata  m_iter_layer_data;
    Iter_forward32* m_iter_forward;

    // clear predictor (resolution, type, device change)
    void clear_predictor();

    // check is existing LWCA device can be reused
    bool device_match( const int* device_ids, unsigned int num_device_ids ) const;

    // check is existing LWCA device and model can be reused
    bool reuse_match( bool                denoised,
                      const int*          device_ids,
                      unsigned int        num_device_ids,
                      const Image_buffer* input_buffer,
                      unsigned int        tile_height,
                      float               maxmem,
                      Eval_mode           mode ) const;

  public:
    // Constructor
    Ssim();

    // Destructor
    virtual ~Ssim();

    // destructor proxy
    virtual void release() { delete this; }

    //Set the logger which can be used to log errors etc.
    //\param logger The logger
    virtual void set_logger( ILogger* logger );

    // set special training data (binary serialized form)
    virtual void set_training_data( const void* data );

    // Retrieve the required size of the heatmap output buffer for a given input size.
    // The input image may be padded, and the heatmap is smaller than the input image,
    // so using this will help ensure you have a buffer of the correct size.
    virtual void get_heatmap_output_size( int input_width, int input_height, int& output_width, int& output_height );

    // Run SSIM predictor, perform re-initialization if needed
    // (e.g. on first run or resolution change)
    // Predictor will run on the specified devices.
    // Running on the CPU (device -1) is not supported.
    // Inputs and output may be on GPU or CPU (CPU: device id == -1).
    //
    // \param device_ids The ids of the devices on which the prediction may run. If this is a
    //        \c nullptr the predictor will attempt to run on any available GPU.
    // \param input_buffer The input buffer.
    // \param output_buffer Optional output buffer which will receive the heatmap result
    // \return Return SSIM prediction value, in case the  was successful, or
    //         a negative value in case of an error.
    virtual float run( const int*          device_ids,
                       unsigned int        num_device_ids,
                       const Image_buffer* input_buffer,
                       Image_buffer*       output_buffer,
                       unsigned int        tile_height );

    // Iteration prediction part of the interface. EXPERIMENTAL
    // May change in the future, in particular to add other SSIM levels than 0.98

    // set special training data for iteration prediction (binary serialized form)
    virtual void set_iteration_training_data( const void* data );

    // Run predictor for iteration number to reach fixed SSIM value of 0.98,
    // perform re-initialization if needed.
    //
    // Predictor will run on the specified devices.
    // Running on the CPU (device -1) is not supported.
    // Inputs and output may be on GPU or CPU (CPU: device id == -1).
    //
    // \param device_ids The ids of the devices on which the prediction may run. If this is a
    //        \c nullptr the predictor will attempt to run on any available GPU.
    // \param ssim_valies  6 to 16 SSIM prediction values after, 1, 2, 4, ... 2^k iterations
    // \param n_ssim_values number of values passed
    // \return Return iteration prediction value at which SSIM=0.98 is reached
    virtual float run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_valies );

    // Run predictor for iteration number to reach fixed SSIM value,
    // perform re-initialization if needed.
    //
    // Predictor will run on the specified devices.
    // Running on the CPU (device -1) is not supported.
    // Inputs and output may be on GPU or CPU (CPU: device id == -1).
    //
    // \param device_ids The ids of the devices on which the prediction may run. If this is a
    //        \c nullptr the predictor will attempt to run on any available GPU.
    // \param ssim_valies 6 to 16 SSIM prediction values after, 1, (skip 2 and 4), 8, 16, 32,
    // 64, ... 2^k iterations
    // \param n_ssim_values number of values passed
    // \param target target SSIM value
    // \return Return iteration prediction value at which SSIM=target is reached
    virtual float run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_valies, float target );


    // Set custom parameter to the SSIM predictor.
    //
    // \param name Parameter name. Lwrrently, "maxmem" - memory in MiBs is supported.
    // \param value Parameter value.
    virtual void set_parameter( const char* name, float value );

    // Run SSIM predictor for a denoised image. Perform re-initialization if needed
    // (e.g. on first run or resolution change)
    // Predictor will run on the specified devices.
    // Running on the CPU (device -1) is not supported.
    // Inputs and output may be on GPU or CPU (CPU: device id == -1).
    //
    // \param device_ids The ids of the devices on which the prediction may run. If this is a
    //        \c nullptr the predictor will attempt to run on any available GPU.
    // \param input_buffer The input buffer.
    // \param denoised_input_buffer The input buffer.
    // \param output_buffer Optional output buffer which will receive the heatmap result
    // \param split image into the stripes of this height, don't split if 0.
    // \return Return SSIM prediction value, in case the  was successful,
    //         or a negative value in case of an error.
    virtual float run_denoised( const int*          device_ids,
                                unsigned int        num_device_ids,
                                const Image_buffer* input_buffer,
                                const Image_buffer* denoised_input_buffer,
                                Image_buffer*       output_buffer,
                                unsigned int        tile_height );


    // Deprecated experimental method for extracting SSIM gradient map.
    // This functionality is removed in the current version.
    virtual float run_grad( const int*          device_ids,
                            unsigned int        num_device_ids,
                            const Image_buffer* input_buffer,
                            Image_buffer*       output_buffer,
                            unsigned int        tile_height )
    {
        return -2.0;
    }

    // For debugging purposes. Allow for argument binding
    void set_layer_callback( std::function<Layer_callback>* lc );
    std::function<Layer_callback>* m_lc;

  private:
    // private internal invocation method
    float run_int( const int*          device_ids,
                   unsigned int        num_device_ids,
                   const Image_buffer* input_buffer,
                   const Image_buffer* denoised_input_buffer,  // optional, denoised image
                   Image_buffer*       heatmap_buffer,         // optional heatmap buffer
                   unsigned int        tile_height );
};


}  // namespace SSIM
}  // namespace LW
