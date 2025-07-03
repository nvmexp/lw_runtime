// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

/******************************************************************************
 * Copyright 1986-2008 by mental images GmbH, Fasanenstr. 81, D-10623 Berlin,
 * Germany. All rights reserved.
 ******************************************************************************
 *
 *      The AtomicCounter class can be used like any other primitive integer
 *      type except for two details: (a) it is non-copyable and (b) all
 *      operations occur atomically. This means that two threads can use the
 *      same atomic counter conlwrrently without locking.
 *
 *      A SharedObject<T> represents a reference-counted T pointer with a
 *      custom deleter attached to it (through a virtual destructor). The fact
 *      that the deleter is embedded into the instance allows SharedObjects to
 *      be cast to other derived types.
 *
 *****************************************************************************/

#pragma once

#include <Util/AtomicCounterImpl.h>
#include <corelib/misc/Concepts.h>

#include <cstddef>  // std::size_t

namespace optix {

class AtomicCounter : private corelib::NonCopyable
{
  public:
    explicit AtomicCounter( unsigned int val = 0u ) { create_counter( m_counter, val ); }
    ~AtomicCounter() { destroy_counter( m_counter ); }
    operator unsigned int() const { return get_counter( m_counter ); }

    unsigned int operator++() { return atomic_inc( m_counter ); }
    unsigned int operator++( int ) { return atomic_post_inc( m_counter ); }

    unsigned int operator--() { return atomic_dec( m_counter ); }
    unsigned int operator--( int ) { return atomic_post_dec( m_counter ); }

    unsigned int operator+=( unsigned int rhs ) { return atomic_add( m_counter, rhs ); }
    unsigned int operator-=( unsigned int rhs ) { return atomic_sub( m_counter, rhs ); }

  private:
    NativeAtomicCounter m_counter;
};

template <class T>
struct SharedObjectBase : private corelib::AbstractInterface
{
    T* const      m_ptr;  // is never NULL
    AtomicCounter m_ref_count;

    SharedObjectBase( T& obj )
        : m_ptr( &obj )
        , m_ref_count( 1 )
    {
    }

    size_t ref_count() { return m_ref_count; }

    template <class U>
    void assign_to( SharedObjectBase<U>*& dst_ptr )  // store down-casted pointer here
    {
        T* const t_ptr( 0 );
        U* const u_ptr( t_ptr );  // Casting from 'T*' to 'U*' must be legal.
        no_unused_variable_warning( u_ptr );
        dst_ptr = reinterpret_cast<SharedObjectBase<U>*>( this );
    }
};

template <class T, class D>
class SharedObject : public SharedObjectBase<T>
{
  public:
    SharedObject( T& obj, D const& deleter = D() )
        : SharedObjectBase<T>( obj )
        , m_deleter( deleter )
    {
    }
    ~SharedObject() { m_deleter( this->m_ptr ); }

  private:
    D m_deleter;
};

}  // end namespace optix
