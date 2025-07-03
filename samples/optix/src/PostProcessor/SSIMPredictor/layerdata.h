/******************************************************************************
* Copyright 2018 LWPU Corporation. All rights reserved.
******************************************************************************
* Author:  Juri Abramov
* Purpose: Parsing of trained weight from Json file.
*****************************************************************************/

#pragma once

#include <vector>

#include "common.h"

namespace LW {
namespace SSIM {

class Data_element : public AI_error
{

    // size of the serialized element
    virtual size_t serialize_size() const = 0;

    // serialize to a buffer, return pointer after serialization
    virtual void* serialize( void* out ) const = 0;

    // deserialize element into self from a buffer
    virtual const void* deserialize( const void* in ) = 0;
};

class Batchnorm_data : public Data_element
{
  public:
    int                m_dim;
    std::vector<float> m_mean;
    std::vector<float> m_var;
    std::vector<float> m_alpha;
    std::vector<float> m_beta;

    Batchnorm_data()
        : m_dim( 0 )
    {
    }

    // size of the serialized element
    virtual size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    virtual void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    virtual const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Batchnorm_data& bn ) const;
    bool operator!=( const Batchnorm_data& bn ) const { return !( *this == bn ); }
};

class Colw_data : public Data_element
{
  public:
    std::vector<int>   m_dims;
    std::vector<float> m_weights;

    Colw_data() {}

    // size of the serialized element
    virtual size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    virtual void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    virtual const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Colw_data& c ) const;
    bool operator!=( const Colw_data& c ) const { return !( *this == c ); }
};

// Colwolution / fully connected layer with bias
class Colw_data_bias : public Colw_data
{
  public:
    std::vector<float> m_bias;

    Colw_data_bias() {}

    // size of the serialized element
    virtual size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    virtual void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    virtual const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Colw_data_bias& c ) const;
    bool operator!=( const Colw_data_bias& c ) const { return !( *this == c ); }
};

// classifier is like colw data + bias in this model
class Classifier : public Colw_data
{
  public:
    float m_bias;

    Classifier() {}

    // size of the serialized element
    virtual size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    virtual void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    virtual const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Classifier& c ) const;
    bool operator!=( const Classifier& c ) const { return !( *this == c ); }
};


class Layerdata : public Data_element
{

  public:
    std::vector<Colw_data>      m_base_colw;
    std::vector<Batchnorm_data> m_base_bn;

    std::vector<Colw_data>      m_top_r_colw;
    std::vector<Batchnorm_data> m_top_r_bn;
    Classifier                  m_classifier;

    std::vector<Colw_data>      m_top_h_colw;
    std::vector<Batchnorm_data> m_top_h_bn;

    float m_bn_epsilon;  //  batchnorm epsilon

    // load DL filter parameters from given json training file.

    Layerdata()
        : m_bn_epsilon( 1e-5f )
    {
    }
    Layerdata( const char* training_file, ILogger* logger = 0 );

    // size of the serialized element
    size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Layerdata& ld ) const;
    bool operator!=( const Layerdata& ld ) const { return !( *this == ld ); }

    // check of new model is used
    bool is_model_n() const { return m_top_h_colw.size() > 6; }

  private:
    // size of stack serialization
    size_t serialize_stack_size( const std::vector<Colw_data>&      colw,  // colwolution layers
                                 const std::vector<Batchnorm_data>& bn     // batch norm layers
                                 ) const;

    // serialize (sub) stack
    void* serialize_stack( void*                              out,   // buffer pointer
                           const std::vector<Colw_data>&      colw,  // colwolution layers
                           const std::vector<Batchnorm_data>& bn     // batch norm layers
                           ) const;

    // deserialize (sub) stack
    const void* deserialize_stack( const void*                  in,    // buffer pointer
                                   std::vector<Colw_data>&      colw,  // colwolution layers
                                   std::vector<Batchnorm_data>& bn     // batch norm layers
                                   );
};

// class representing layer data for SSIM iteration prediction
class Iter_layerdata : public Data_element
{

  public:
    bool                        m_old_format;  // for fixed target
    float                       m_power;       // input power transform, like 2.0
    float                       m_target;      // specific target value (0.98),  0 if any
    std::vector<Colw_data_bias> m_fc;          // last one is a classifier

    // load DL filter parameters from given json training file.

    Iter_layerdata( bool old_format )
        : m_old_format( old_format )
        , m_power( 1.0f )
        , m_target( 0.0f )
    {
    }

    // size of the serialized element
    size_t serialize_size() const;

    // serialize to a buffer, return pointer after serialization
    void* serialize( void* out ) const;

    // deserialize element into self from a buffer
    const void* deserialize( const void* in );

    // equal/unequal operator
    bool operator==( const Iter_layerdata& ld ) const;
    bool operator!=( const Iter_layerdata& ld ) const { return !( *this == ld ); }
};


}  // namespace SSIM
}  // namespace LW