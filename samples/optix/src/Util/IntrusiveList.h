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

#include <iterator>
#include <stddef.h>


namespace optix {

template <typename T>
struct IntrusiveListDefaultTraits
{
    static T*   newSentinel() { return new T; }
    static void deleteNode( T* node ) { delete node; }
    static T* getNext( T* ptr ) { return ptr->getNext(); }
    static const T* getNext( const T* ptr ) { return ptr->getNext(); }
    static void setNext( T* ptr, T* next ) { ptr->setNext( next ); }
    static T* getPrev( T* ptr ) { return ptr->getPrev(); }
    static const T* getPrev( const T* ptr ) { return ptr->getPrev(); }
    static void setPrev( T* ptr, T* prev ) { ptr->setPrev( prev ); }
};

template <typename T>
struct IntrusiveListTraits : public IntrusiveListDefaultTraits<T>
{
};

template <typename T>
struct IntrusiveListTraits<const T> : public IntrusiveListTraits<T>
{
};

template <typename T, typename Traits>
class IntrusiveList;

template <typename T, typename Traits>
class IntrusiveListIterator
{
    // Types used to implement the safe_bool idiom.
    struct _Hidden_type
    {
        _Hidden_type* _M_bool;
    };
    typedef _Hidden_type* _Hidden_type::*_Safe_bool;

  public:
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef T*                              pointer;
    typedef T&                              reference;
    typedef T                               value_type;
    typedef ptrdiff_t                       difference_type;

    explicit IntrusiveListIterator( T* lwr )
        : m_lwr( lwr )
    {
    }
    ~IntrusiveListIterator() {}

    bool operator!=( const IntrusiveListIterator<T, Traits>& rhs ) const { return m_lwr != rhs.m_lwr; }
    bool operator==( const IntrusiveListIterator<T, Traits>& rhs ) const { return m_lwr == rhs.m_lwr; }
    _Safe_bool operator!() { return m_lwr == 0 || Traits::getNext( m_lwr ) == 0 ? &_Hidden_type::_M_bool : 0; }
    IntrusiveListIterator<T, Traits> operator++()
    {
        m_lwr = Traits::getNext( m_lwr );
        RT_ASSERT( m_lwr != 0 );  // Walk off of end
        return *this;
    }
    IntrusiveListIterator<T, Traits> operator--()
    {
        m_lwr = Traits::getPrev( m_lwr );
        RT_ASSERT( m_lwr != 0 );  // Walk off of beginning of list
        return *this;
    }
    IntrusiveListIterator<T, Traits> operator++( int )
    {
        IntrusiveListIterator<T, Traits> save = *this;
        ++*this;
        return save;
    }
    IntrusiveListIterator<T, Traits> operator--( int )
    {
        IntrusiveListIterator<T, Traits> save = *this;
        --*this;
        return save;
    }

  private:
    T* m_lwr;
    friend class IntrusiveList<T, Traits>;
    template <class U, class UTraits>
    friend bool operator==( const U* left, const IntrusiveListIterator<const U, UTraits>& right );
    template <class U, class UTraits>
    friend bool operator!=( const U* left, const IntrusiveListIterator<U, UTraits>& right );
    template <class U, class UTraits>
    friend bool operator==( const IntrusiveListIterator<const U, UTraits>& left, const U* right );
    template <class U, class UTraits>
    friend bool operator!=( const IntrusiveListIterator<U, UTraits>& left, U* right );
    T*   getPointer() const { return m_lwr; }
    void checkEnd() const
    {
        RT_ASSERT( m_lwr != 0 );                     // Invalid iterator
        RT_ASSERT( Traits::getNext( m_lwr ) != 0 );  // Dereferencing end();
    }

  public:
    operator T*() const
    {
        checkEnd();
        return m_lwr;
    }
    T& operator*() const
    {
        checkEnd();
        return *m_lwr;
    }
    T* operator->() const
    {
        checkEnd();
        return m_lwr;
    }
};

template <typename T, typename Traits>
bool operator==( const T* left, const IntrusiveListIterator<const T, Traits>& right )
{
    return left == right.getPointer();
}
template <typename T, typename Traits>
bool operator!=( const T* left, const IntrusiveListIterator<T, Traits>& right )
{
    return left != right.getPointer();
}
template <typename T, typename Traits>
bool operator==( const IntrusiveListIterator<const T, Traits>& left, const T* right )
{
    return left.getPointer() == right;
}
template <typename T, typename Traits>
bool operator!=( const IntrusiveListIterator<T, Traits>& left, T* right )
{
    return left.getPointer() != right;
}

template <typename T, typename Traits = IntrusiveListTraits<T>>
class IntrusiveList
{
  public:
    typedef T*        pointer;
    typedef const T*  const_pointer;
    typedef T&        reference;
    typedef const T&  const_reference;
    typedef T         value_type;
    typedef size_t    size_type;
    typedef ptrdiff_t difference_type;

    typedef IntrusiveListIterator<T, Traits>       iterator;
    typedef IntrusiveListIterator<const T, Traits> const_iterator;
    typedef std::reverse_iterator<iterator>       reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    IntrusiveList()
        : m_head( 0 )
        , m_tail( 0 )
    {
    }
    ~IntrusiveList()
    {
        erase( begin(), end() );
        Traits::deleteNode( m_tail );
    }

    iterator begin()
    {
        addSentinel();
        return iterator( m_head );
    }
    const_iterator begin() const
    {
        addSentinel();
        return const_iterator( m_head );
    }

    iterator end()
    {
        addSentinel();
        return iterator( m_tail );
    }
    const_iterator end() const
    {
        addSentinel();
        return const_iterator( m_tail );
    }

    reverse_iterator       rbegin() { return reverse_iterator( end() ); }
    const_reverse_iterator rbegin() const { return const_reverse_iterator( end() ); }
    reverse_iterator       rend() { return reverse_iterator( begin() ); }
    const_reverse_iterator rend() const { return const_reverse_iterator( begin() ); }

    void push_back( T* value ) { insert( end(), value ); }
    void push_front( T* value ) { insert( begin(), value ); }

    iterator erase( iterator begin, iterator end )
    {
        while( begin != end )
            begin = erase( begin );
        return end;
    }

    iterator erase( iterator begin )
    {
        T* lwr  = begin.getPointer();
        T* prev = Traits::getPrev( lwr );
        T* next = Traits::getNext( lwr );
        RT_ASSERT( next != 0 );  // Cannot erase end();

        if( prev )
            Traits::setNext( prev, next );
        else
            m_head = next;
        Traits::setPrev( next, prev );

        Traits::deleteNode( lwr );
        return iterator( next );
    }

    iterator erase( T* lwr ) { return erase( iterator( lwr ) ); }

    iterator insert( iterator position, T* value )
    {
        T* pos  = position.getPointer();
        T* prev = Traits::getPrev( pos );
        if( prev )
            Traits::setNext( prev, value );
        else
            m_head = value;
        Traits::setPrev( value, prev );

        Traits::setNext( value, pos );
        Traits::setPrev( pos, value );
        return iterator( value );
    }

    void splice( iterator position, IntrusiveList<T>& x, iterator first, iterator last )
    {
        if( first == last )
            return;
        // Remove elements [first, last) from x and put them in our list

        T* splice_begin = first;
        T* splice_end   = Traits::getPrev( last.getPointer() );

        // Remove from x
        T* x_prev = Traits::getPrev( splice_begin );
        T* x_next = Traits::getNext( splice_end );

        if( x_prev )
            Traits::setNext( x_prev, x_next );
        else
            x.m_head = x_next;
        Traits::setPrev( x_next, x_prev );

        // insert into this
        T* pos       = position.getPointer();
        T* this_prev = Traits::getPrev( pos );
        if( this_prev )
            Traits::setNext( this_prev, splice_begin );
        else
            m_head = splice_begin;
        Traits::setPrev( splice_begin, this_prev );

        Traits::setNext( splice_end, pos );
        Traits::setPrev( pos, splice_end );
    }

    T* back()
    {
        RT_ASSERT( !empty() );
        return Traits::getPrev( m_tail );
    }
    const T* back() const
    {
        RT_ASSERT( !empty() );
        return Traits::getPrev( m_tail );
    }
    T* front()
    {
        RT_ASSERT( !empty() );
        return m_head;
    }
    const T* front() const
    {
        RT_ASSERT( !empty() );
        return m_head;
    }

    bool empty() const { return m_head == m_tail; }

  private:
    IntrusiveList( const IntrusiveList<T>& );
    IntrusiveList<T>& operator=( const IntrusiveList<T>& );
    void internal_erase( T* first, T* last )
    {
        T* prev = first->getPrev();
        T* next = last->getNext();
        if( prev )
        {
            prev->setNext( next );
        }
        else
        {
            RT_ASSERT( first == m_head );
            m_head = next;
        }
        if( next )
        {
            next->setPrev( prev );
        }
        else
        {
            RT_ASSERT( last == m_tail );
            m_tail = prev;
        }
        while( first != next )
        {
            T* n = first->getNext();
            first->setPrev( 0 );
            first->setNext( 0 );
            first = n;
        }
    }

    void addSentinel() const
    {
        if( !m_head )
        {
            m_head = m_tail = Traits::newSentinel();
        }
    }

    mutable T* m_head;
    mutable T* m_tail;
};

template <typename T>
class IntrusiveListNode
{
  protected:
    IntrusiveListNode()
        : m_next( 0 )
        , m_prev( 0 )
    {
    }

  private:
    friend struct IntrusiveListDefaultTraits<T>;
    T*   getNext() const { return m_next; }
    T*   getPrev() const { return m_prev; }
    void setNext( T* next ) { m_next = next; }
    void setPrev( T* prev ) { m_prev = prev; }
    T*               m_next;
    T*               m_prev;
};

}  // end namespace optix
