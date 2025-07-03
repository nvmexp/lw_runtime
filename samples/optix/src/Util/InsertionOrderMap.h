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

#include <map>
#include <vector>


namespace optix {

// Class that maintains the order of insertion (in the vector), along with a
// sub-linear lookup based on the key.  The iterators iterate over the values
// in the order they were first inserted (e.g. subsequent insertions do not
// change the order of the first insertion).  This class also throws asserts
// when attempting to perform insertions with the [] operator.

template <typename Key, typename Val, typename value_typeT>
struct InsertionOrderMapImpl
{
  public:
    typedef value_typeT value_type;

  private:
    // Keep track of the indicies of where the keys are in the list.
    std::map<Key, size_t> indices;
    // Store the key-value pairs as a vector in the order they were inserted.
    std::vector<value_type> elements;

    // These are overloaded functions that know how to get the key from different
    // value_type types.  The class isn't specifically designed to accomodate arbitrary
    // types for value_type, but rather to accomodate using this base class for both map
    // and set constructs.
    const Key& getKey( const std::pair<Key, Val>& val ) { return val.first; }
    const Key& getKey( const Key& key ) { return key; }
  public:
    typedef typename std::vector<value_type>::iterator       iterator;
    typedef typename std::vector<value_type>::const_iterator const_iterator;

    void clear()
    {
        indices.clear();
        elements.clear();
    }

    iterator find( const Key& key )
    {
        typename std::map<Key, size_t>::const_iterator index = indices.find( key );
        if( index != indices.end() )
            return elements.begin() + index->second;
        else
            return elements.end();
    }

    const_iterator find( const Key& key ) const
    {
        typename std::map<Key, size_t>::const_iterator index = indices.find( key );
        if( index != indices.end() )
            return elements.begin() + index->second;
        else
            return elements.end();
    }

    size_t size() const { return elements.size(); }
    bool   empty() const { return elements.empty(); }

    iterator       begin() { return elements.begin(); }
    const_iterator begin() const { return elements.begin(); }

    iterator       end() { return elements.end(); }
    const_iterator end() const { return elements.end(); }

    value_type&       back() { return elements.back(); }
    const value_type& back() const { return elements.back(); }

    // Don't allow adding a key via these accessors, so that we can catch accidental
    // insertions.
    Val& operator[]( const Key& key )
    {
        iterator val = find( key );
        RT_ASSERT( val != end() );
        return val->second;
    }
    const Val& operator[]( const Key& key ) const
    {
        const_iterator val = find( key );
        RT_ASSERT( val != end() );
        return val->second;
    }

    std::pair<iterator, bool> insert( const value_type& val )
    {
        std::pair<typename std::map<Key, size_t>::iterator, bool> inserted;
        inserted = indices.insert( std::make_pair( getKey( val ), elements.size() ) );
        if( inserted.second )
        {
            elements.push_back( val );
        }
        size_t index = ( inserted.first )->second;
        return std::make_pair( elements.begin() + index, inserted.second );
    }

    iterator insert( const value_type& val, bool& newlyInserted )
    {
        std::pair<typename std::map<Key, size_t>::iterator, bool> inserted;
        inserted = indices.insert( std::make_pair( getKey( val ), elements.size() ) );

        newlyInserted = inserted.second;

        if( inserted.second )
        {
            elements.push_back( val );
        }
        size_t index = ( inserted.first )->second;
        return elements.begin() + index;
    }

    template <typename InputIterator>
    void insert( InputIterator first, const InputIterator& second )
    {
        while( first != second )
        {
            insert( *first );
            ++first;
        }
    }

    void erase( iterator iter )
    {
        // Get the index
        typename std::map<Key, size_t>::iterator index = indices.find( getKey( *iter ) );
        // Iterate over the map and update all the entries that have an index greater
        // than the one we are removing.
        for( typename std::map<Key, size_t>::iterator i_iter = indices.begin(); i_iter != indices.end(); ++i_iter )
            if( i_iter->second > index->second )
                i_iter->second--;
        // Now remove it from both the indicies and the elements
        indices.erase( index );
        elements.erase( iter );
    }

    size_t erase( const Key& key )
    {
        iterator iter = find( key );
        if( iter == end() )
            return 0;
        erase( iter );
        return 1;
    }
};

template <typename Key, typename Val>
struct InsertionOrderMap : public InsertionOrderMapImpl<Key, Val, std::pair<Key, Val>>
{
};

template <typename Key>
struct InsertionOrderSet : public InsertionOrderMapImpl<Key, Key, Key>
{
};

}  // end namespace optix
