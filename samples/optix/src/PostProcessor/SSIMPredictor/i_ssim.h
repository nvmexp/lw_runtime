//*****************************************************************************
// Copyright 2018 LWPU Corporation. All rights reserved.
//*****************************************************************************
/// \file
/// \brief SSIM prediction public API
///
//*****************************************************************************

#pragma once

namespace LW {
namespace SSIM {

// Data type for input and output buffers
enum Data_type
{
    DATA_INT8  = 0,
    DATA_HALF  = 1,
    DATA_FLOAT = 2
};

// Structure describing input buffer layout
struct Image_buffer
{
    void* m_data;       // input framebuffer data
    int   m_width;      // framebuffer width
    int   m_height;     // framebuffer height
    int   m_nchannels;  // number of data channels
    int   m_device_id;  // the id of the device on which the buffer resides,
                        // -1 for cpu
    Data_type m_type;   // input data type
    bool      m_nhwc;   // indicates if RGB channels are co-located.

    // constructor
    Image_buffer( void* d, int width, int height, int nc, int id, Data_type type, bool nhwc )
        : m_data( d )
        , m_width( width )
        , m_height( height )
        , m_nchannels( nc )
        , m_device_id( id )
        , m_type( type )
        , m_nhwc( nhwc )
    {
    }
};


// The SSIM prediction logger class.
class ILogger
{
  public:
    enum Severity
    {
        S_ERROR    = 0,
        S_WARNING  = 1,
        S_INFO     = 2,
        S_PROGRESS = 3,
        S_DEBUG    = 4
    };

    virtual void log( Severity severity, const char* msg ) = 0;
};


// On-demand SSIM predictor public class interface.
class ISsim
{
  public:
    // destructor proxy
    virtual void release() = 0;

    //Set the logger which can be used to log errors etc.
    //\param logger The logger
    virtual void set_logger( ILogger* logger ) = 0;


    // set special training data (binary serialized form)
    virtual void set_training_data( const void* data ) = 0;

    // Retrieve the required size of the heatmap output buffer for a given input size.
    // The input image may be padded, and the heatmap is smaller than the input image,
    // so using this will help ensure you have a buffer of the correct size.
    virtual void get_heatmap_output_size( int input_width, int input_height, int& output_width, int& output_height ) = 0;

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
    // \param split image into the stripes of this height, don't split if 0.
    // \return Return SSIM prediction value, in case the  was successful,
    //         or a negative value in case of an error.
    virtual float run( const int*          device_ids,
                       unsigned int        num_device_ids,
                       const Image_buffer* input_buffer,
                       Image_buffer*       output_buffer,
                       unsigned int        tile_height ) = 0;

    // Iteration prediction part of the interface. EXPERIMENTAL
    // May change in the future, in particular to add other SSIM levels than 0.98

    // set special training data for iteration prediction (binary serialized form)
    virtual void set_iteration_training_data( const void* data ) = 0;

    // Run predictor for iteration number to reach fixed SSIM value of 0.98,
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
    // \return Return iteration prediction value at which SSIM=0.98 is reached
    virtual float run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_valies ) = 0;

    // Set custom parameter to the SSIM predictor.
    //
    // \param name Parameter name. Lwrrently, "maxmem" - memory in MiBs is supported.
    // \param value Parameter value.
    virtual void set_parameter( const char* name, float value ) = 0;

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
                                unsigned int        tile_height ) = 0;


    // Deprecated experimental method for extracting SSIM gradient map.
    // This functionality is removed in the current version.
    virtual float run_grad( const int*          device_ids,
                            unsigned int        num_device_ids,
                            const Image_buffer* input_buffer,
                            Image_buffer*       output_buffer,
                            unsigned int        tile_height ) = 0;

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
    virtual float run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_valies, float target ) = 0;


    static const unsigned int heatmap_shrink_factor = 16;
};


}  // namespace SSIM
}  // namespace LW


#ifndef IRAY_BUILD

// Library interface (Optix)

#if defined( _WIN32 ) || defined( _WIN64 )

#ifdef COMPILE_SSIM_PREDICTOR
#define SSIM_PREDICTOR_API __declspec( dllexport )
#else  // COMPILE_SSIM_PREDICTOR
#define SSIM_PREDICTOR_API __declspec( dllimport )
#endif  // COMPILE_SSIM_PREDICTOR

#else

#ifdef COMPILE_SSIM_PREDICTOR
#define SSIM_PREDICTOR_API __attribute__( ( __visibility__( "default" ) ) )
#else  // COMPILE_SSIM_PREDICTOR
#define SSIM_PREDICTOR_API
#endif  // COMPILE_SSIM_PREDICTOR

#endif  // defined(_WIN32) || defined(_WIN64)

#else
#define SSIM_PREDICTOR_API
#endif  // IRAY_BUILD

// creator function to be used across DLL boundaries.
extern "C" SSIM_PREDICTOR_API LW::SSIM::ISsim* lw_ssim_predictor_create();
