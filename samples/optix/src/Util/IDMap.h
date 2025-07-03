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

#include <prodlib/exceptions/ValueOutOfRange.h>

#include <map>
#include <vector>


namespace optix {

template <typename T, typename Index = int>
class IDMap
{
  public:
    static const Index ILWALID_INDEX = Index( 0 ) - 1;

    // Inserts the val and returns its corresponding ID.
    Index insert( const T& val );

    // Return the value corresponding to a particular ID.
    const T& get( Index id ) const;

    // Return the ID corresponding to a particular value.
    Index getID( const T& val ) const;

    // Returns the number of allocated size
    size_t size() const { return m_values.size(); }

  private:
    std::vector<T> m_values;
    std::map<T, Index> m_IDs;
};

template <typename T, typename Index>
Index IDMap<T, Index>::insert( const T& val )
{
    Index id = getID( val );
    if( id == ILWALID_INDEX )
    {
        id = Index( m_values.size() );
        if( id == ILWALID_INDEX || id < 0 )
            throw prodlib::ValueOutOfRange( RT_EXCEPTION_INFO, "ID map full" );
        m_values.push_back( val );
        m_IDs[val] = id;
    }
    return id;
}


template <typename T, typename Index>
Index IDMap<T, Index>::getID( const T& val ) const
{
    typename std::map<T, Index>::const_iterator it = m_IDs.find( val );
    if( it != m_IDs.end() )
        return it->second;
    else
        return ILWALID_INDEX;
}


template <typename T, typename Index>
const T& IDMap<T, Index>::get( Index id ) const
{
    return m_values[id];
}


}  // namespace optix
