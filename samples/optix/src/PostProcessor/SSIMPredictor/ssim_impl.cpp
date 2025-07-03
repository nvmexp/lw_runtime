//*****************************************************************************
// Copyright 2018 LWPU Corporation. All rights reserved.
//*****************************************************************************
/// \file
/// \brief SSIM prediction API implementation
///
//*****************************************************************************

#include <algorithm>

#include "forward.h"
#include "ssim_impl.h"
#include "util.h"

// default training data
#include "generated_iter_training_data.h"
#include "generated_training_data.h"


namespace LW {
namespace SSIM {

// Constructor
Ssim::Ssim()
    : m_logger( nullptr )
    , m_forward( nullptr )
    , m_used_lwda_id( -1 )
    , m_rup_height( 0 )
    , m_rup_width( 0 )
    , m_rup_tile_height( 0 )
    , m_denoised( false )
    , m_maxmem( 0.0f )
    , m_iter_layer_data( true )
    , m_iter_forward( nullptr )
    , m_lc( nullptr )

{
    m_layer_data.deserialize( trainssim );
    m_iter_layer_data.deserialize( trainiter );  // which is in old format
}

// Destructor
Ssim::~Ssim()
{
    clear_predictor();
    if( m_iter_forward )
        delete m_iter_forward;
}

void Ssim::set_logger( ILogger* logger )
{
    m_logger = logger;
    if( m_forward )
        m_forward->set_logger( logger );
}


void Ssim::set_training_data( const void* data )
{
    Layerdata new_ld;
    new_ld.deserialize( data ? data : trainssim );
    // check if actually different
    if( new_ld != m_layer_data )
    {
        bool same_model = ( new_ld.is_model_n() == m_layer_data.is_model_n() );
        m_layer_data    = new_ld;
        // change weights of existing predictor, if any
        if( m_forward )
        {
            if( same_model )
                m_forward->set_weights( m_layer_data );
            else
                clear_predictor();
        }
    }
}


void Ssim::set_parameter( const char* name, float value )
{
    if( !name )
        return;

    if( std::string( "maxmem" ) == name )
    {
        m_maxmem = value;
        return;
    }
}

static int ceilDiv( int x, int y )
{
    return x / y + ( x % y != 0 );
}

void Ssim::set_layer_callback( std::function<Layer_callback>* lc )
{
    this->m_lc = lc;
    if( this->m_forward )
    {
        this->m_forward->m_lc = this->m_lc;
    }
}


void Ssim::get_heatmap_output_size( int input_width, int input_height, int& output_width, int& output_height )
{
    int f = m_forward ? m_forward->m_model_shrink_factor : m_layer_data.is_model_n() ? 1 : 16;

    output_width  = ceilDiv( input_width, f );
    output_height = ceilDiv( input_height, f );
}

float Ssim::run( const int* device_ids, unsigned int num_device_ids, const Image_buffer* input_buffer, Image_buffer* output_buffer, unsigned int tile_height )
{
    return run_int( device_ids, num_device_ids, input_buffer, nullptr, output_buffer, tile_height );
}

float Ssim::run_denoised( const int*          device_ids,
                          unsigned int        num_device_ids,
                          const Image_buffer* input_buffer,
                          const Image_buffer* denoised_input_buffer,
                          Image_buffer*       output_buffer,
                          unsigned int        tile_height )
{
    return run_int( device_ids, num_device_ids, input_buffer, denoised_input_buffer, output_buffer, tile_height );
}


float Ssim::run_int( const int*          device_ids,
                     unsigned int        num_device_ids,
                     const Image_buffer* input_buffer,
                     const Image_buffer* denoised_input_buffer,
                     Image_buffer*       heatmap_buffer,
                     unsigned int        tile_height )
{
    MI_ASSERT( input_buffer );


    Eval_mode mode = EVAL_FWD;

    // check if compatible with previous run
    if( m_forward && !reuse_match( denoised_input_buffer != nullptr, device_ids, num_device_ids, input_buffer, tile_height, m_maxmem, mode ) )
    {
        clear_predictor();
    }

    // training using RGB data, ignore alpha channel if any
    int inp_channel = denoised_input_buffer ? 6 : 3;

    // initialize
    if( !m_forward )
    {
        // new mode - update only if re-creating the model
        m_mode = mode;

        // plain stupid first id listed
        m_used_lwda_id = device_ids ? device_ids[0] : 0;

        // set resolution
        m_rup_width       = round_up( input_buffer->m_width );
        m_rup_height      = round_up( input_buffer->m_height );
        m_rup_tile_height = round_up( tile_height );
        if( m_rup_tile_height >= input_buffer->m_height )
            m_rup_tile_height = 0;

        m_forward       = new Forward32( m_used_lwda_id, m_logger );
        m_forward->m_lc = m_lc;  // debugging callback
        if( !m_forward->m_compute_base_ok )
        {
            clear_predictor();
            return -3.0f;
        }

        // Layerdata is only used for model layout, not for weights here
        if( m_rup_tile_height == 0 )
        {
            m_forward->create_layers( m_layer_data, m_rup_width, m_rup_height, inp_channel, 1,
                                      size_t( m_maxmem * ( 1 << 20 ) ), mode );
        }
        else
        {
            // ignore memory limit if tile height is specified explicitly
            int layer_height = m_rup_tile_height + 2 * m_forward->m_overlap;
            MI_ASSERT( round_up( layer_height ) == layer_height );
            m_forward->create_layers( m_layer_data, m_rup_width, layer_height, inp_channel, 1, 0, mode );
        }

        // check for errors
        if( m_forward->check_error() )
        {
            clear_predictor();
            return -2.0f;
        }


        // set current weights
        m_forward->set_weights( m_layer_data );
    }

    MI_ASSERT( m_forward );
    m_forward->predict_image( heatmap_buffer, *input_buffer, denoised_input_buffer,
                              m_rup_tile_height ? m_rup_tile_height : m_rup_height );

    float val = m_forward->get_ssim();
    return val;
}


void Ssim::clear_predictor()
{
    if( m_forward )
    {
        delete m_forward;
        m_forward         = nullptr;
        m_used_lwda_id    = -1;
        m_rup_height      = 0;
        m_rup_width       = 0;
        m_rup_tile_height = 0;
    }
}

// check is existing LWCA device can be reused
bool Ssim::device_match( const int* device_ids, unsigned int num_device_ids ) const
{
    // check LWCA device
    if( m_used_lwda_id == -1 )
        return false;

    // list is given and not in the list
    if( device_ids && std::find( device_ids, device_ids + num_device_ids, m_used_lwda_id ) == device_ids + num_device_ids )
        return false;

    return true;
}

// check if previous predictor can be reused
bool Ssim::reuse_match( bool                denoised,
                        const int*          device_ids,
                        unsigned int        num_device_ids,
                        const Image_buffer* input_buffer,
                        unsigned int        tile_height,
                        float               maxmem,
                        Eval_mode           mode ) const
{
    if( !m_forward )
        return false;

    if( denoised != m_denoised )
        return false;

    if( !device_match( device_ids, num_device_ids ) )
        return false;

    // new mode should be a subset
    if( ( int( mode ) & ~int( m_mode ) ) != 0 )
        return false;

    // memory limit
    if( maxmem != m_maxmem )
        return false;

    // resolution
    if( round_up( input_buffer->m_width ) != (unsigned int)m_rup_width )
        return false;
    if( round_up( input_buffer->m_height ) != (unsigned int)m_rup_height )
        return false;

    // same tile size?
    if( round_up( tile_height ) >= (unsigned int)input_buffer->m_height )
        tile_height = 0;
    if( round_up( tile_height ) != (unsigned int)m_rup_tile_height )
        return false;

    return true;
}

// Iteration prediction part of the interface. EXPERIMENTAL
// May change in the future, in particular to add other SSIM levels than 0.98

// set special training data for iteration prediction (binary serialized form)
void Ssim::set_iteration_training_data( const void* data )
{
    if( !data )
        data = trainiter;
    // guess version
    int  version    = *static_cast<const int*>( data );
    bool old_format = version != 2;

    Iter_layerdata new_ld( old_format );
    new_ld.deserialize( data );
    // check if actually different
    if( new_ld != m_iter_layer_data )
    {
        m_iter_layer_data = new_ld;
        // change weights of existing predictor, if any
        if( m_iter_forward )
        {
            m_iter_forward->set_weights( m_iter_layer_data );
        }
    }
}

// Run predictor for iteration number to reach fixed SSIM value,
// perform re-initialization if needed.
//
// Predictor will run on the specified devices.
// Running on the CPU (device -1) is not supported.
// Inputs and output may be on GPU or CPU (CPU: device id == -1).
//
// \param device_ids The ids of the devices on which the prediction may run. If this is a
//        \c nullptr the predictor will attempt to run on any available GPU.
// \param ssim_values  6 to 16 SSIM prediction values after, 1, 2, 4, ... 2^k iterations
// \param n_ssim_values number of values passed
// \return Return iteration prediction value at which SSIM=0.98 is reached
float Ssim::run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_values )
{
    return run_iteration( device_ids, num_device_ids, ssim_values, n_ssim_values, 0.98f );
}

// Run predictor for iteration number to reach fixed SSIM value,
// perform re-initialization if needed.
//
// Predictor will run on the specified devices.
// Running on the CPU (device -1) is not supported.
// Inputs and output may be on GPU or CPU (CPU: device id == -1).
//
// \param device_ids The ids of the devices on which the prediction may run. If this is a
//        \c nullptr the predictor will attempt to run on any available GPU.
// \param ssim_values  6 to 16 SSIM prediction values after, 1, 2, 4, ... 2^k iterations
// \param n_ssim_values number of values passed
// \param target target SSIM value
// \return Return iteration prediction value at which SSIM=target is reached
float Ssim::run_iteration( const int* device_ids, unsigned int num_device_ids, float* ssim_values, int n_ssim_values, float target )
{
    if( m_iter_forward && !device_match( device_ids, num_device_ids ) )
    {
        delete m_iter_forward;
        m_iter_forward = nullptr;
        clear_predictor();
    }

    if( !m_iter_forward )
    {
        m_used_lwda_id = device_ids ? device_ids[0] : 0;
        m_iter_forward = new Iter_forward32( m_iter_layer_data, m_used_lwda_id, m_logger );
        m_iter_forward->set_weights( m_iter_layer_data );
    }

    float res = m_iter_forward->run( ssim_values, n_ssim_values, target );

    return res;
}


}  // namespace SSIM
}  // namespace LW

extern "C" SSIM_PREDICTOR_API LW::SSIM::ISsim* lw_ssim_predictor_create()
{
    return new LW::SSIM::Ssim();
}
