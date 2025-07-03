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

#include <prodlib/exceptions/Assert.h>

namespace optix {

class LinkedPtr_Link
{
  public:
    LinkedPtr_Link() = default;
    virtual ~LinkedPtr_Link()
    {
        RT_ASSERT_NOTHROW( m_managedObjectIndex == -1, "Failed to remove link from managed object" );
    }
    LinkedPtr_Link( const LinkedPtr_Link& ) = delete;
    LinkedPtr_Link& operator=( const LinkedPtr_Link& rhs ) = delete;

  private:
    friend class ManagedObject;
    int m_managedObjectIndex = -1;
    struct managedObjectIndex_fn
    {
        int& operator()( LinkedPtr_Link* link ) { return link->m_managedObjectIndex; }
    };
};

template <typename FromType, typename ToType>
class LinkedPtr : public LinkedPtr_Link
{
    /*
   * WARNING: do not add operator= or copy constructors to this
   * class. They are dangerous, especially in conjunction with the C++
   * auto keyword. The move constructor (&&) is acceptable and is
   * required for use in std::vector.
   */

  public:
    LinkedPtr()                   = default;
    LinkedPtr( const LinkedPtr& ) = delete;

    // Move is allowed but copying is not - primarily to prevent
    // accidental bulk copies in iteration or temporary arrays.
    LinkedPtr( LinkedPtr&& src )
    {
        // Establish new link and remove link from old node
        set( src.m_fromPtr, src.m_toPtr );
        src.reset();
    }
    LinkedPtr& operator=( const LinkedPtr& rhs ) = delete;

    LinkedPtr& operator=( LinkedPtr&& other )
    {
        set( other.m_fromPtr, other.m_toPtr );
        other.reset();
        return *this;
    }

    operator bool() const { return m_toPtr != nullptr; }

    ~LinkedPtr() override { remove_link(); }

    ToType* operator->()
    {
        RT_ASSERT( m_toPtr != nullptr );
        return m_toPtr;
    }
    ToType* operator->() const
    {
        RT_ASSERT( m_toPtr != nullptr );
        return m_toPtr;
    }
    ToType& operator*()
    {
        RT_ASSERT( m_toPtr != nullptr );
        return *m_toPtr;
    }
    const ToType& operator*() const
    {
        RT_ASSERT( m_toPtr != nullptr );
        return *m_toPtr;
    }
    ToType*   get() const { return m_toPtr; }
    FromType* getFrom() const { return m_fromPtr; }


    void set( FromType* fromPtr, ToType* toPtr )
    {
        if( m_fromPtr != fromPtr || m_toPtr != toPtr )
        {
            remove_link();
            m_fromPtr = fromPtr;
            m_toPtr   = toPtr;
            add_link();
        }
    }

    void reset()
    {
        remove_link();
        m_fromPtr = nullptr;
        m_toPtr   = nullptr;
    }

    bool operator==( const LinkedPtr& rhs ) const { return m_toPtr == rhs.m_toPtr; }
    bool operator!=( const LinkedPtr& rhs ) const { return m_toPtr != rhs.m_toPtr; }
    bool operator==( const ToType* rhs ) const { return m_toPtr == rhs; }
    bool operator!=( const ToType* rhs ) const { return m_toPtr != rhs; }

  private:
    void add_link()
    {
        if( m_toPtr )
        {
            m_toPtr->addLink( this );
        }
    }
    void remove_link()
    {
        if( m_toPtr )
        {
            m_toPtr->removeLink( this );
        }
    }
    FromType* m_fromPtr = nullptr;
    ToType*   m_toPtr   = nullptr;
};


}  // end namespace optix
