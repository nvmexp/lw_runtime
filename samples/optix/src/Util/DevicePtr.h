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

#pragma once


#include <lwca.h>

#include <Util/AtomicCounter.h>
#include <corelib/misc/Concepts.h>

#include <algorithm>

/*
 * DevicePtr class is intended for passing managed LWCA device
 * pointers around on the host.  It is essentially a stripped-down
 * version of a shared pointer that will help you avoid dereferencing a
 * pointer on the host.  The deleter semantics can be used to manage
 * allocations of these pointers.  NOTE: if you do not use a deleter
 * then the object will not be freed automatically.
 */

namespace optix {
template <typename T>
class DevicePtr
{
  public:
    // Provide the usual STL-style type names. Note that although
    // 'DevicePtr<>' is a pointer, it is not an iterator.
    typedef T                 value_type;
    typedef value_type*       pointer;
    typedef value_type const* const_pointer;
    typedef value_type&       reference;
    typedef value_type const& const_reference;

    // The class may be instantiated emptily. Use the member function
    // 'reset()' to assign a pointer into an already constructed instance.
    // This constructor uses standard 'operator delete' to destroy the
    // object.
    explicit DevicePtr( LWdeviceptr ptr = 0 );  // The lwca device pointer to assign to the instance

    // On destruction, the custom deleter function will be ilwoked if the
    // reference count drops to zero.
    ~DevicePtr();

    // Construct a 'DevicePtr<>' while providing a custom deleter
    // function.
    template <class D>
    DevicePtr( LWdeviceptr,  // assign 'T*' to the instance
               D const& );   // functor to ilwoke to destroy 'T'

    // Copy-construct a 'DevicePtr<>' from an existing instance.
    DevicePtr( DevicePtr const& );  // 'DevicePtr<>' to copy from.

    // Copy-construct from an existing instance while performing _static_ type
    // colwersion from 'DevicePtr<T>' to 'DevicePtr<U>'.
    template <class U>
    DevicePtr( DevicePtr<U> const& );  // 'DevicePtr<>' to cast and copy.

    // Re-assign an existing 'DevicePtr<>' with another instance. The
    // resource contained in the left hand-side object is freed if
    // appropriate.
    DevicePtr& operator=( DevicePtr const& );  // 'DevicePtr<>' to copy from.

    // Explicit valid test.
    bool is_valid() const;

  private:
    // Pointer to member functions colwert to bool but to nothing else.
    typedef bool ( DevicePtr::*unspecified_bool_type )() const;

  public:
    // Like a normal pointer, a 'DevicePtr<>' can be used like a boolean.
    // 'if (p) ...' will be true when the 'DevicePtr' is not empty; false
    // otherwise.
    operator unspecified_bool_type() const;

    // Re-set the resource 'DevicePtr<>' points to. This operation may or may
    // not 'delete' the original resource, depending on the reference count.
    void reset( LWdeviceptr ptr = 0 );  // The device pointer to
                                        // assign to the instance.

    // Re-set the resource 'DevicePtr<>' points to and attach a custom
    // deleter function to it rather than having 'DevicePtr<>' use 'operator
    // delete'.
    template <class D>
    void reset( pointer,     // assign 'T*' to the instance
                D const& );  // functor to ilwoke to destroy 'T'

    // Swap the contents of two 'DevicePtr<T>' instances. Never throws.
    void swap( DevicePtr& );  // Reference to the other 'DevicePtr<>'.

    // Get a copy of the real 'T*' pointer stored in this instance as
    // device code. Never use that pointer to dereference the pointer
    // on the host or create another smart pointer.
    pointer getTypedPtr() const;

    // Get the device pointer.  Never use this to create another smart
    // pointer.
    LWdeviceptr get() const;

    // Return the number of instances referencing the shared value.
    std::size_t ref_count() const;

  private:
    template <class U>
    friend class DevicePtr;
    SharedObjectBase<T>* m_shared_obj;
};

//////////////////////////////////////////////////////////////////////////////
///// Implementation
//////////////////////////////////////////////////////////////////////////////

template <class T>
inline DevicePtr<T>::DevicePtr( LWdeviceptr ptr )  // initial pointer value
{
    if( ptr )
        m_shared_obj = new SharedObject<T, corelib::NonDeleter<T>>( *reinterpret_cast<T*>( ptr ) );
    else
        m_shared_obj = nullptr;
}

template <class T>
template <class D>
inline DevicePtr<T>::DevicePtr( LWdeviceptr ptr,       // initial pointer value
                                D const&    deleter )  // deleter to ilwoke on destruction
{
    if( ptr )
        m_shared_obj = new SharedObject<T, D>( *reinterpret_cast<T*>( ptr ), deleter );
    else
        m_shared_obj = nullptr;
}

template <class T>
inline DevicePtr<T>::~DevicePtr()
{
    if( m_shared_obj )
        if( --m_shared_obj->m_ref_count == 0u )
            delete m_shared_obj;
}

// This copy constructor appears to be redundant because the next constructor
// resolves to the same signature when the template types 'T' and 'U' are
// identical. However, for reasons not yet completely understood the C++
// compiler won't use the templated constructor for ordinary
// copy-construction; instead it will use generated 'memcopy()' version which
// doesn't work for smart pointers because it doesn't increment the reference
// count.

template <class T>
inline DevicePtr<T>::DevicePtr( DevicePtr const& rhs )  // value to copy-construct from
    : m_shared_obj( rhs.m_shared_obj )
{
    if( m_shared_obj )
        ++m_shared_obj->m_ref_count;
}

template <class T>
template <class U>
inline DevicePtr<T>::DevicePtr( DevicePtr<U> const& rhs )  // derived class to copy-construct from
{
    if( rhs.m_shared_obj )
    {
        rhs.m_shared_obj->assign_to( m_shared_obj );
        ++m_shared_obj->m_ref_count;
    }
    else
        m_shared_obj = 0;
}

template <class T>
inline DevicePtr<T>& DevicePtr<T>::operator=( DevicePtr const& rhs )  // value to assign to this instance
{
    DevicePtr( rhs ).swap( *this );
    return *this;
}

template <class T>
inline bool DevicePtr<T>::is_valid() const
{
    return this->get() != 0;
}

template <class T>
inline DevicePtr<T>::operator typename DevicePtr<T>::unspecified_bool_type() const
{
    return is_valid() ? &DevicePtr<T>::is_valid : 0;
}

template <class T>
inline LWdeviceptr DevicePtr<T>::get() const
{
    return ( m_shared_obj ? reinterpret_cast<LWdeviceptr>( m_shared_obj->m_ptr ) : 0 );
}

template <class T>
inline T* DevicePtr<T>::getTypedPtr() const
{
    return ( m_shared_obj ? m_shared_obj->m_ptr : nullptr );
}

template <class T>
inline std::size_t DevicePtr<T>::ref_count() const
{
    return ( m_shared_obj ? m_shared_obj->ref_count() : 0u );
}

template <class T>
inline void DevicePtr<T>::reset( LWdeviceptr ptr )  // reset pointer value
{
    DevicePtr( ptr ).swap( *this );
}

template <class T>
template <class D>
inline void DevicePtr<T>::reset( T*       ptr,       // reset pointer value
                                 D const& deleter )  // deleter to ilwoke on destruction
{
    DevicePtr( ptr, deleter ).swap( *this );
}

template <class T>
inline void DevicePtr<T>::swap( DevicePtr& rhs )  // swap pointers with other instance
{
    std::swap( rhs.m_shared_obj, m_shared_obj );
}

// Comparison operators. Needed to be defined, since we use
// operator unspecified_bool_type() for bool colwersion, which we do not
// want to be used for pointer comparisons.

#define COMPARISON_OPERATORS( t )                                                                                      \
    template <class T>                                                                                                 \
    inline bool operator<( t a, t b )                                                                                  \
    {                                                                                                                  \
        return a.get() < b.get();                                                                                      \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    inline bool operator>( t a, t b )                                                                                  \
    {                                                                                                                  \
        return a.get() > b.get();                                                                                      \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    inline bool operator<=( t a, t b )                                                                                 \
    {                                                                                                                  \
        return a.get() <= b.get();                                                                                     \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    inline bool operator>=( t a, t b )                                                                                 \
    {                                                                                                                  \
        return a.get() >= b.get();                                                                                     \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    inline bool operator==( t a, t b )                                                                                 \
    {                                                                                                                  \
        return a.get() == b.get();                                                                                     \
    }                                                                                                                  \
    template <class T>                                                                                                 \
    inline bool operator!=( t a, t b )                                                                                 \
    {                                                                                                                  \
        return a.get() != b.get();                                                                                     \
    }

COMPARISON_OPERATORS( DevicePtr<T> const& )

#undef COMPARISON_OPERATORS

}  // end namespace optix

namespace std {
// Overload standard std::swap(a,b) function.
template <class T>
inline void swap( optix::DevicePtr<T>& a,   // left hand-side instance
                  optix::DevicePtr<T>& b )  // right hand-side instance
{
    a.swap( b );
}
}
