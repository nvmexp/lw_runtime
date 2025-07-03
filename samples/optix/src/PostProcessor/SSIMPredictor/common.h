/******************************************************************************
 * Copyright 2018 LWPU Corporation. All rights reserved.
 *****************************************************************************/

//Layer helper classes

#pragma once

#include <cstdarg>
#include <cstdio>
#include <lwdnn.h>
#include <functional>
#include <string>
#include <vector>

#ifdef IRAY_BUILD
// Iray build
#include <base/system/main/i_assert.h>
#else
// OptiX build
#ifndef MI_ASSERT
#define MI_ASSERT( X ) ( (void)0 )
#endif
#endif

#include "i_ssim.h"


namespace LW {
namespace SSIM {


class AI_error
{
  public:
    AI_error()
        : m_status( lwdaSuccess )
        , m_logger( nullptr )
    {
    }

    // return true if there was an error
    inline bool check_error() const { return m_status != lwdaSuccess; }

    inline void reset_error() { m_status = lwdaSuccess; }
    inline void set_error( AI_error& other ) { m_status = other.m_status; }

    inline void checkLWDNN( lwdnnStatus_t status )
    {
        if( status != LWDNN_STATUS_SUCCESS )
        {
            log_error( lwdnnGetErrorString( status ) );
            m_status = lwdaError( status );
        }
    }

    inline void checkLwdaErrors( lwdaError_t status )
    {
        if( status != 0 )
        {
            log_error( lwdaGetErrorString( status ) );
            m_status = status;
        }
    }

    void set_logger( ILogger* logger ) { m_logger = logger; }

    void log_info( const char* fmt, ... )
    {
        if( m_logger )
        {
            va_list args;
            va_start( args, fmt );
            char buf[2048];
            int  count = vsnprintf( buf, 2048, fmt, args );
            if( count >= 2048 )
                buf[2046] = '*';
            m_logger->log( ILogger::S_INFO, buf );
            va_end( args );
        }
    }

    void log_debug( const char* fmt, ... )
    {
        if( m_logger )
        {
            va_list args;
            va_start( args, fmt );
            char buf[2048];
            int  count = vsnprintf( buf, 2048, fmt, args );
            if( count >= 2048 )
                buf[2046] = '*';
            m_logger->log( ILogger::S_DEBUG, buf );
            va_end( args );
        }
    }

    void log_error( const char* fmt, ... )
    {
        if( m_logger )
        {
            va_list args;
            va_start( args, fmt );
            char buf[2048];
            int  count = vsnprintf( buf, 2048, fmt, args );
            if( count >= 2048 )
                buf[2046] = '*';
            m_logger->log( ILogger::S_ERROR, buf );
            va_end( args );
        }
        m_status = lwdaErrorUnknown;
    }

  protected:
    // Destructor
    ~AI_error() {}

    lwdaError_t m_status;
    ILogger*    m_logger;
};

// common base for LWCA computation classes
class Compute_base : public AI_error
{
  public:
    // Constructor
    Compute_base( int      device,              // LWCA device
                  ILogger* logger = nullptr );  // optional logger

    // get LWCA stream
    lwdaStream_t get_stream() const { return m_stream; }

    // set LWCA stream
    // Derived classes may need to overwrite it with additional functionality
    virtual void set_stream( lwdaStream_t cs );

    // return device id (GPU id)
    int get_device() const { return m_device; }

  protected:
    // Destructor
    ~Compute_base();

  public:
    // LWCA/LWDNN staff
    lwdnnHandle_t m_lwdnnHandle;
    int           m_device;
    int           m_device_capability;
    std::string   m_device_name;
    lwdaStream_t  m_stream;

    //  working device memory buffer
    void*  m_device_workMem;
    size_t m_device_workMemSize;

    // Flag is base successfully initialized
    bool m_compute_base_ok;
};

class Layerdata;

// model evaluation mode
enum Eval_mode
{
    EVAL_NONE = 0,
    EVAL_FWD  = 1,
    EVAL_ALL  = ~0
};

// This is for debugging purposes for now, so not in the public API. Expensive.
typedef void( Layer_callback )( const char*  layer_name,
                                int          c,    // channels
                                int          h,    // height
                                int          w,    // width
                                const float* data  // layer data, on host, in CHW form
                                );


class Forward_base : public Compute_base
{
  public:
    // Constructor
    Forward_base( int      device,  // LWCA device
                  ILogger* logger = nullptr )
        : Compute_base( device, logger )
        , m_model_shrink_factor( 16 )
        , m_lc( nullptr )
    {
    }

    // Virtual destructor
    virtual ~Forward_base() {}

    virtual void create_layers( const Layerdata& ld,
                                int              width,
                                int              height,
                                int              inp_channels,
                                int              out_channels,
                                size_t           mem,  // GPU memory limit in bytes, if not 0
                                Eval_mode        mode ) = 0;

    virtual void delete_layers() = 0;

    virtual bool set_weights( const Layerdata& ld ) = 0;

    virtual void run() = 0;

    virtual float get_eval_time() const = 0;

    virtual inline void set_colw_algorithms( const std::vector<int>& alg ) = 0;

    virtual bool half_enabled() const = 0;

    // Estimate SSIM of a normalized image in "data". feature buffers can be given in "fdata".
    // The resolution of the input image (and feature buffer) is given in width x height,
    // the number of channels in the input image is given in "image_nchannels", i.e. 3 for RGB.
    // Returns GPU inference time in milliseconds.
    virtual float predict_image( Image_buffer*       heatmap,  // optional, 1/16 res or fullres
                                 const Image_buffer& buffer,
                                 const Image_buffer* denoised_buffer,  // denoised image, optional
                                 int                 tile_height ) = 0;

    // Get last predicted SSIM value
    virtual float get_ssim() { return 0.0f; };

    // Per-tile data
    struct Forward_tile
    {
        int data_offset;   // offset for tile in input image
        int data_h;        // input data height
        int out_offset_h;  // height offset for tile in output image
        int copy_h;        // output image tile size
        int yoffset;       // offset in colwolution buffer for output
    };

    // Split image into vertical tiles (strips)
    // image resolution is given in width x height, tile height in tile_height
    // resulting tiles are stored in tiles parameter
    static void split_image( int                        width,   // input image
                             int                        height,  // input image
                             int                        shrink_factor,
                             int                        tile_height,  // input tiles
                             std::vector<Forward_tile>& tiles );

    int m_model_shrink_factor;

    // for debugging purposes
    std::function<Layer_callback>* m_lc;


    // Overlap when tiling is used (should be divisible by 32, for heat map)
    static const int m_overlap = 32;
};


}  // namespace SSIM
}  // namespace LW
