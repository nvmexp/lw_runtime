/******************************************************************************
 * Copyright 2018 LWPU Corporation. All rights reserved.
 ******************************************************************************
 * Author:  Juri Abramov
 * Purpose: Parsing of trained weight from Json file.
 *****************************************************************************/


#include <functional>
#include <numeric>
#include <vector>

#include "layerdata.h"

namespace LW {
namespace SSIM {

// serialized size of the layer data structure
size_t Layerdata::serialize_size() const
{
    return serialize_stack_size( m_base_colw, m_base_bn ) + serialize_stack_size( m_top_r_colw, m_top_r_bn )
           + m_classifier.serialize_size() + serialize_stack_size( m_top_h_colw, m_top_h_bn ) + sizeof( m_bn_epsilon );
}

// serialize layer data
void* Layerdata::serialize( void* out ) const
{
    out = serialize_stack( out, m_base_colw, m_base_bn );

    out = serialize_stack( out, m_top_r_colw, m_top_r_bn );
    out = m_classifier.serialize( out );
    out = serialize_stack( out, m_top_h_colw, m_top_h_bn );

    auto fout = static_cast<decltype( m_bn_epsilon )*>( out );
    *fout++   = m_bn_epsilon;
    out       = static_cast<void*>( fout );

    return out;
}

// deserialize layer data
const void* Layerdata::deserialize( const void* in )
{
    in = deserialize_stack( in, m_base_colw, m_base_bn );

    in = deserialize_stack( in, m_top_r_colw, m_top_r_bn );
    in = m_classifier.deserialize( in );
    in = deserialize_stack( in, m_top_h_colw, m_top_h_bn );

    auto tin     = static_cast<const decltype( Layerdata::m_bn_epsilon )*>( in );
    m_bn_epsilon = *tin++;
    in           = static_cast<const void*>( tin );

    return in;
}


bool Layerdata::operator==( const Layerdata& ld ) const
{
    return ld.m_base_colw == m_base_colw && ld.m_base_bn == m_base_bn && ld.m_top_r_colw == m_top_r_colw
           && ld.m_top_r_bn == m_top_r_bn && ld.m_top_h_colw == m_top_h_colw && ld.m_top_h_bn == m_top_h_bn
           && ld.m_classifier == m_classifier && ld.m_bn_epsilon == m_bn_epsilon;
}

// size of sub-stack serialization
size_t Layerdata::serialize_stack_size( const std::vector<Colw_data>&      colw,       // colwolution layers
                                        const std::vector<Batchnorm_data>& bn ) const  // batch norm layers
{
    return sizeof( int ) + std::accumulate( begin( colw ), end( colw ), (size_t)0,
                                            []( size_t a, const Colw_data& b ) { return a + b.serialize_size(); } )
           + sizeof( int ) + std::accumulate( begin( bn ), end( bn ), (size_t)0,
                                              []( size_t a, const Batchnorm_data& b ) { return a + b.serialize_size(); } );
}


// serialize sub-stack to a buffer
void* Layerdata::serialize_stack( void*                              out,        // buffer pointer
                                  const std::vector<Colw_data>&      colw,       // colwolution layers
                                  const std::vector<Batchnorm_data>& bn ) const  // batch norm layers
{
    {
        auto iout = static_cast<int*>( out );
        *iout++   = (int)colw.size();
        out       = static_cast<void*>( iout );
    }

    for( const auto& c : colw )
        out = c.serialize( out );

    {
        auto iout = static_cast<int*>( out );
        *iout++   = (int)bn.size();
        out       = static_cast<void*>( iout );
    }

    for( const auto& b : bn )
        out = b.serialize( out );

    return out;
}


// deserialize (sub)stack
const void* Layerdata::deserialize_stack( const void*                  in,    // buffer pointer
                                          std::vector<Colw_data>&      colw,  // colwolution layers
                                          std::vector<Batchnorm_data>& bn )   // batch norm layers
{
    auto iin    = static_cast<const int*>( in );
    auto n_colw = *iin++;
    in          = static_cast<const void*>( iin );

    colw.clear();
    for( int i = 0; i < n_colw; ++i )
    {
        Colw_data cd;
        in = cd.deserialize( in );
        colw.push_back( cd );
    }

    iin       = static_cast<const int*>( in );
    auto n_bn = *iin++;
    in        = static_cast<const void*>( iin );

    bn.clear();
    for( int i = 0; i < n_bn; ++i )
    {
        Batchnorm_data bnd;
        in = bnd.deserialize( in );
        bn.push_back( bnd );
    }

    return in;
}

// serialized size of the iteration layer data structure
size_t Iter_layerdata::serialize_size() const
{
    size_t size = m_old_format ? 0 : ( sizeof( int ) + 2 * sizeof( float ) + sizeof( int ) );

    for( const auto& l : m_fc )
        size += l.serialize_size();

    return size;
}

// serialize iteration layer data
void* Iter_layerdata::serialize( void* out ) const
{
    if( !m_old_format )
    {
        int* iout = static_cast<int*>( out );
        *iout++   = 2;  // version 2
        out       = static_cast<void*>( iout );

        float* fout = static_cast<float*>( out );
        *fout++     = m_power;
        *fout++     = m_target;
        out         = static_cast<void*>( fout );

        iout    = static_cast<int*>( out );
        *iout++ = static_cast<int>( m_fc.size() );
        out     = static_cast<void*>( iout );
    }

    for( const auto& l : m_fc )
        out = l.serialize( out );

    return out;
}

// deserialize iteration layer data
const void* Iter_layerdata::deserialize( const void* in )
{
    int n    = 2;  // 2 layers for old format
    m_power  = 1.0f;
    m_target = 0.98f;

    if( !m_old_format )
    {
        auto iin0 = static_cast<const int*>( in );
        iin0++;  // version
        in = static_cast<const void*>( iin0 );

        auto fin = static_cast<const float*>( in );
        m_power  = *fin++;
        m_target = *fin++;
        in       = static_cast<const void*>( fin );

        auto iin = static_cast<const int*>( in );
        n        = *iin++;
        in       = static_cast<const void*>( iin );
    }

    m_fc.clear();
    for( int i = 0; i < n; ++i )
    {
        Colw_data_bias colw;
        in = colw.deserialize( in );
        m_fc.push_back( colw );
    }

    return in;
}


bool Iter_layerdata::operator==( const Iter_layerdata& ld ) const
{
    return ld.m_fc == m_fc && ld.m_target == m_target && ld.m_power == m_power && ld.m_old_format == m_old_format;
}


size_t Batchnorm_data::serialize_size() const
{
    return sizeof( decltype( m_dim ) ) + m_mean.size() * sizeof( decltype( m_mean )::value_type )
           + m_var.size() * sizeof( decltype( m_var )::value_type ) + m_alpha.size() * sizeof( decltype( m_alpha )::value_type )
           + m_beta.size() * sizeof( decltype( m_beta )::value_type );
}


void* Batchnorm_data::serialize( void* out ) const
{
    auto iout = static_cast<decltype( m_dim )*>( out );
    *iout++   = m_dim;
    out       = static_cast<void*>( iout );

    {
        auto mout = static_cast<decltype( m_mean )::value_type*>( out );
        for( const auto v : m_mean )
            *mout++ = v;
        out         = static_cast<void*>( mout );
    }

    {
        auto vout = static_cast<decltype( m_var )::value_type*>( out );
        for( const auto v : m_var )
            *vout++ = v;
        out         = static_cast<void*>( vout );
    }


    {
        auto aout = static_cast<decltype( m_alpha )::value_type*>( out );
        for( const auto v : m_alpha )
            *aout++ = v;
        out         = static_cast<void*>( aout );
    }

    {
        auto bout = static_cast<decltype( m_beta )::value_type*>( out );
        for( const auto v : m_beta )
            *bout++ = v;
        out         = static_cast<void*>( bout );
    }

    return out;
}


const void* Batchnorm_data::deserialize( const void* in )
{
    auto iin = static_cast<const decltype( m_dim )*>( in );
    m_dim    = *iin++;
    in       = static_cast<const void*>( iin );

    // tin == typed in
    {
        m_mean.clear();
        auto tin = static_cast<const decltype( m_mean )::value_type*>( in );
        for( int i = 0; i < m_dim; ++i )
            m_mean.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    {
        m_var.clear();
        auto tin = static_cast<const decltype( m_var )::value_type*>( in );
        for( int i = 0; i < m_dim; ++i )
            m_var.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    {
        m_alpha.clear();
        auto tin = static_cast<const decltype( m_alpha )::value_type*>( in );
        for( int i = 0; i < m_dim; ++i )
            m_alpha.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    {
        m_beta.clear();
        auto tin = static_cast<const decltype( m_beta )::value_type*>( in );
        for( int i = 0; i < m_dim; ++i )
            m_beta.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    return in;
}

// equality operator
bool Batchnorm_data::operator==( const Batchnorm_data& bn ) const
{
    return m_dim == bn.m_dim && m_mean == bn.m_mean && m_var == bn.m_var && m_alpha == bn.m_alpha && m_beta == bn.m_beta;
}

// colwolution data serialize size
size_t Colw_data::serialize_size() const
{
    return sizeof( int ) + m_dims.size() * sizeof( decltype( m_dims )::value_type )
           + m_weights.size() * sizeof( decltype( m_weights )::value_type );
}

// serialize colwolution data
void* Colw_data::serialize( void* out ) const
{
    auto iout = static_cast<int*>( out );
    *iout++   = (int)m_dims.size();
    out       = static_cast<void*>( iout );

    {
        auto dout = static_cast<decltype( m_dims )::value_type*>( out );
        for( const auto v : m_dims )
            *dout++ = v;
        out         = static_cast<void*>( dout );
    }

    {
        auto wout = static_cast<decltype( m_weights )::value_type*>( out );
        for( const auto v : m_weights )
            *wout++ = v;
        out         = static_cast<void*>( wout );
    }

    return out;
}

// deserialize colwolution data
const void* Colw_data::deserialize( const void* in )
{
    auto iin     = static_cast<const int*>( in );
    auto dimsize = *iin++;
    in           = static_cast<const void*>( iin );

    {
        m_dims.clear();
        auto tin = static_cast<const decltype( m_dims )::value_type*>( in );
        for( int i = 0; i < dimsize; ++i )
            m_dims.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    auto prod = std::accumulate( begin( m_dims ), end( m_dims ), 1, std::multiplies<int>() );
    {
        m_weights.clear();
        auto tin = static_cast<const decltype( m_weights )::value_type*>( in );
        for( int i = 0; i < prod; ++i )
            m_weights.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    return in;
}

// equality operator
bool Colw_data::operator==( const Colw_data& c ) const
{
    return m_dims == c.m_dims && m_weights == c.m_weights;
}

// classifier serialized size
size_t Classifier::serialize_size() const
{
    return Colw_data::serialize_size() + sizeof( decltype( m_bias ) );
}


// serialize classifier
void* Classifier::serialize( void* out ) const
{
    out = Colw_data::serialize( out );

    auto fout = static_cast<decltype( m_bias )*>( out );
    *fout++   = m_bias;
    out       = static_cast<void*>( fout );

    return out;
}

// deserialize classifier
const void* Classifier::deserialize( const void* in )
{
    in = Colw_data::deserialize( in );

    auto fin = static_cast<const decltype( m_bias )*>( in );
    m_bias   = *fin++;
    in       = static_cast<const void*>( fin );

    return in;
}

// equal operator
bool Classifier::operator==( const Classifier& c ) const
{
    return Colw_data::operator==( static_cast<const Colw_data&>( c ) ) && m_bias == c.m_bias;
}

// colw data with bias serialize size
size_t Colw_data_bias::serialize_size() const
{
    return Colw_data::serialize_size() + m_bias.size() * sizeof( decltype( m_bias )::value_type );
}

// serialize colw data with bias
void* Colw_data_bias::serialize( void* out ) const
{
    out = Colw_data::serialize( out );

    {
        auto bout = static_cast<decltype( m_bias )::value_type*>( out );
        for( const auto v : m_bias )
            *bout++ = v;
        out         = static_cast<void*>( bout );
    }

    return out;
}

// deserialize colwolution data with bias
const void* Colw_data_bias::deserialize( const void* in )
{
    in = Colw_data::deserialize( in );

    {
        auto n_bias = m_dims[0];
        m_bias.clear();
        auto tin = static_cast<const decltype( m_bias )::value_type*>( in );
        for( int i = 0; i < n_bias; ++i )
            m_bias.push_back( *tin++ );
        in = static_cast<const void*>( tin );
    }

    return in;
}

// equality operator
bool Colw_data_bias::operator==( const Colw_data_bias& c ) const
{
    return *static_cast<const Colw_data*>( this ) == *static_cast<const Colw_data*>( &c ) && m_bias == c.m_bias;
}


}  // namespace SSIM
}  // namespace LW
