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

#include <Util/IteratorAdaptors.h>
#include <prodlib/exceptions/Assert.h>

#include <map>
#include <memory>
#include <set>


namespace optix {

typedef int                              ReusableIDValue;
typedef std::shared_ptr<ReusableIDValue> ReusableID;

template <typename T>
class ReusableIDMap
{
  public:
    enum : ReusableIDValue
    {
        NoHint = -1
    };

    // Specify an indexBase > 0 to get N-based IDs.
    ReusableIDMap( int indexBase = 0 )
        : m_nextNewId( indexBase )
        , m_areReservedIdsFinalized( false )
    {
    }

    // Reserve an ID for later generation when it's passed into 'insert' as a hint.
    void reserveIdForHint( ReusableIDValue id )
    {
        RT_ASSERT_MSG( m_idMap.empty(), "Trying to reserve an ID after IDs were already allocated" );
        RT_ASSERT_MSG( !m_areReservedIdsFinalized, "Trying to reserve an ID after call to finalizeReservedIds()" );
#ifdef RT_SANITY_CHECK
        RT_ASSERT_MSG( m_idMap.find( id ) == m_idMap.end(), "Trying to reserve an ID that's already been allocated" );
#endif
        m_reservedForHints.insert( id );
    }

    // This function preprocesses reservedIDs to improve the runtime of insert(). Prepopulates the list of freed ids
    // with any non-reserved IDs whose values are less than the maximum reservedID.
    // Note that this function has to be called following the ID reservations and before any insert() call!
    void finalizeReservedIds()
    {
        RT_ASSERT_MSG( !m_areReservedIdsFinalized, "Trying to call finalizeReservedIds() twice?" );
        // skip all potential holes below m_nextNewId
        // Exploit the fact that the std::set m_reservedForHints returns items in increasing order.
        for( auto rID : m_reservedForHints )
        {
            // fill list of freed ids with (rID - m_nextNewId) increasing indices, starting at m_nextNewId
            // eg, m_nextNewId=2, rID=5 --> values = {2, 3, 4}
            for( ReusableIDValue id = m_nextNewId; id < rID; ++id )
                m_freed.insert( id );

            // now update next new id to point behind the reserved IDs
            m_nextNewId = rID + 1;
        }
        m_areReservedIdsFinalized = true;
    }

    // Insert the value and return a generated ID. If a hint is given, will use that as an ID
    // or fail if the ID has already been taken.
    ReusableID insert( T val, ReusableIDValue hint = NoHint )
    {
        RT_ASSERT_MSG( m_reservedForHints.empty() || m_areReservedIdsFinalized,
                       "Usage Error: Call to finalizeReservedIds() is missing" );

        ReusableIDValue newid = -1;

        // If we've been given a hint, just use that as ID. Make sure it was actually reserved for hints.
        if( hint != NoHint )
        {
#ifdef RT_SANITY_CHECK
            RT_ASSERT_MSG( m_reservedForHints.count( hint ) == 1, "ID allocated with hint, but was never reserved" );
            RT_ASSERT_MSG( m_idMap.find( hint ) == m_idMap.end(), "Hinted ID already taken. Double insertion?" );
#endif
            newid = hint;
        }
        else  // no hint, gotta find an unreserved ID
        {
            // Try recycling the free'd ones first, starting with the lowest to encourage shrinking of the reserved ID range.
            for( auto iter = m_freed.begin(); iter != m_freed.end(); ++iter )
            {
                if( m_reservedForHints.count( *iter ) != 0 )
                    continue;
                newid = *iter;
                m_freed.erase( iter );
                break;
            }

            if( newid == -1 )
                newid = m_nextNewId++;
        }

#ifdef RT_SANITY_CHECK
        // Sanity check. If this fires, we somehow computed an ID that was already taken, i.e. it's bug in the above code.
        RT_ASSERT( m_idMap.find( newid ) == m_idMap.end() );
#endif

        m_idMap.insert( std::make_pair( newid, val ) );
        return ReusableID( new ReusableIDValue( newid ), [this]( ReusableIDValue* id ) { this->deleter( id ); } );
    }

    // Return the number of entries in the map.
    size_t size() const { return m_idMap.size(); }

    // Return whether there are any entries in the map
    size_t empty() const { return m_idMap.empty(); }

    // Return the ID that will be returned next. Lwrrently does not
    // handle the hints and reservations.
    ReusableIDValue nextid() const
    {
        if( !m_freed.empty() )
        {
            return *m_freed.begin();
        }
        else
        {
            return m_idMap.size();
        }
    }

    // Return the size required for a linear array indexed by the reserved IDs.
    size_t linearArraySize() const
    {
        if( m_idMap.empty() )
            return 0;
        return m_idMap.rbegin()->first + 1;  // highest reserved ID plus 1
    }

    // Return the value corresponding to a particular ID. Assert if it isn't found.
    const T& get( ReusableIDValue id ) const
    {
        auto it = m_idMap.find( id );
        RT_ASSERT_MSG( it != m_idMap.end(), "id not found" );
        return it->second;
    }

    // Get a value by ID, return false if it isn't found.
    bool get( ReusableIDValue id, T& value ) const
    {
        auto it = m_idMap.find( id );
        if( it == m_idMap.end() )
            return false;
        value = it->second;
        return true;
    }

    // Iterators that just expose the value (not the id). This also
    // allows the id map to be used with range-based loops.  Iterates over ID order.
    typedef ValueIterator<T, typename std::map<ReusableIDValue, T>::iterator>       iterator;
    typedef ValueIterator<T, typename std::map<ReusableIDValue, T>::const_iterator> const_iterator;
    iterator       begin() { return m_idMap.begin(); }
    iterator       end() { return m_idMap.end(); }
    const_iterator begin() const { return m_idMap.begin(); }
    const_iterator end() const { return m_idMap.end(); }
    const_iterator cbegin() const { return m_idMap.cbegin(); }
    const_iterator cend() const { return m_idMap.cend(); }

    // Iterators for the few cases where the ID is also needed.  Iterates over ID order.
    typedef typename std::map<ReusableIDValue, T>::iterator       mapIterator;
    typedef typename std::map<ReusableIDValue, T>::const_iterator const_mapIterator;
    mapIterator       mapBegin() { return m_idMap.begin(); }
    mapIterator       mapEnd() { return m_idMap.end(); }
    const_mapIterator mapBegin() const { return m_idMap.begin(); }
    const_mapIterator mapEnd() const { return m_idMap.end(); }

  private:
    void deleter( ReusableIDValue* idptr )
    {
        const size_t nerased = m_idMap.erase( *idptr );
        RT_ASSERT_MSG( nerased == 1, "attempting to delete nonexisting ID" );
        m_freed.insert( *idptr );
        delete idptr;
    }

    std::map<ReusableIDValue, T> m_idMap;
    std::set<ReusableIDValue> m_freed;
    std::set<ReusableIDValue> m_reservedForHints;
    int                       m_nextNewId;
    bool                      m_areReservedIdsFinalized;  // for tracking correct usage only
};

}  // namespace optix
