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

#include <cstdlib>
#include <vector>

namespace optix {
//
// Simple list with O(1) insertion and removal that allows an item
// to appear in multiple lists.  Requires a functor to retrieve an
// integer from each element.
//
template <class T, class IdxFn>
class IndexedVector
{
  private:
    typedef std::vector<T> ListType;

  public:
    IndexedVector() {}
    ~IndexedVector() {}

    void addItem( T elt )
    {
        int& idx = IdxFn()( elt );
        if( idx >= 0 )
            return;  // Item already in list

        idx = static_cast<int>( m_list.size() );
        m_list.push_back( elt );
    }

    void removeItem( T elt )
    {
        int idx = IdxFn()( elt );
        if( idx < 0 )
            return;  // Item not in list

        // Move to end of list and remove
        IdxFn()( m_list.back() ) = idx;
        std::swap( m_list[idx], m_list.back() );
        m_list.pop_back();

        // Reset the index
        IdxFn()( elt ) = -1;
    }

    bool itemIsInList( T elt )
    {
        int idx = IdxFn()( elt );
        return idx >= 0;
    }

    const std::vector<T>& getList() const { return m_list; }

    size_t size() const { return m_list.size(); }

    void clear()
    {
        // Reset all indices
        for( typename std::vector<T>::iterator iter = m_list.begin(); iter != m_list.end(); ++iter )
            IdxFn()( *iter )                        = -1;
        m_list.clear();
    }

    std::vector<T> clearAndMove()
    {
        // Reset all indices
        for( typename std::vector<T>::iterator iter = m_list.begin(); iter != m_list.end(); ++iter )
            IdxFn()( *iter )                        = -1;
        // Move the vector to the return value
        return std::move( m_list );
    }

    bool empty() const { return m_list.empty(); }

    typedef typename ListType::iterator iterator;
    iterator                            begin() { return m_list.begin(); }
    iterator                            end() { return m_list.end(); }

    typedef typename ListType::const_iterator const_iterator;
    const_iterator                            begin() const { return m_list.begin(); }
    const_iterator                            end() const { return m_list.end(); }


  private:
    ListType m_list;
};
}
