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
#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>
#include <prodlib/exceptions/Assert.h>

#include <functional>
#include <map>


namespace optix {
class PersistentStream;

//
// The Graph property classes are the workhorse of the property
// tracking system in OptiX. They get used to track unresolved
// references, caller/callee relationships, node graph heights,
// semantic types and other properties derived from the structure of
// the graph.
//
// There are four main flavors:
// 1. GraphProperty<P, true>: Holds any number of references to each
//    property P. This is the most common type of property.
//
// 2. GraphProperty<P, false>: Holds only a single reference to each
//    property P. Will throw an exception if a property is added
//    more than once. Equivalent to a set.
//
// 3. GraphPropertySingle: Holds any number of references to a
//    single property. Equivalent to a counter.
//
// 4. GraphPropertyMulti<P, S>: Holds any number references to and
//    property P and subproperty S. Can efficiently return all of
//    the subproperties for a given P.
//
// WARNING: GraphPropertyMulti can be quite expensive. It should be
//    used extremely sparingly! Lwrrently only for ilwerse bindings
//    in BindingManager.
//
// The primary interfaces are as follows:
// CTOR: initialized to empty / zero count.
//
// bool addOrRemoveProperty(P, added): add or remove a single
//      reference. Returns true if the property is new (added=true)
//      or the sole remaining (added=false).
//
// bool empty(): returns true if the set is empty
//
// bool contains(P): returns whether the set contains a specific
//      property
//
// bool count(P): returns the count for a specific property
//      (typically should be used only for debugging)
//
//
// In addition, these methods are defined on normal (type 1/2)
// objects but not on Single/Multi (type 3/4).
//
// size_t size(): returns the size of the set
//
// begin()/end(): iterate over the set
//
// front()/back(): returns the first/last element in the set
//    according to the specified order.
//
// bool intersects(other): returns whether the set shares any
//    elements in common with another graph property set
//
// bool addPropertiesFrom(other): adds the properties from other
//    set, returning true if any new property was added. Note that
//    only one reference is added for each non-zero reference in the
//    other set.
//
// Some variants have a few additional interfaces, dolwmented below.


//----------------------------------------------------------------
// GraphProperty (type 1)
//----------------------------------------------------------------

template <typename P, bool counted = true, class Compare = std::less<P>>
class GraphProperty
{
  private:
    typedef std::map<P, int, Compare> MapType;

  public:
    // CTOR / DTOR
    GraphProperty()  = default;
    ~GraphProperty() = default;

    // Iterator type (count is not exposed)
    typedef KeyIterator<P, typename MapType::const_iterator> const_iterator;

    // Standard interface (see above)
    bool addOrRemoveProperty( const P& property, bool adding );
    bool contains( const P& property ) const { return m_map.count( property ) != 0; }
    int count( const P& property ) const { return contains( property ) ? m_map.at( property ) : 0; }
    size_t              size() const { return m_map.size(); }
    bool                empty() const { return m_map.empty(); }
    const_iterator      begin() const { return m_map.begin(); }
    const_iterator      end() const { return m_map.end(); }
    P                   front() const;
    P                   back() const;
    template <bool      other_counted>
    bool intersects( const GraphProperty<P, other_counted, Compare>& otherset ) const;
    template <bool other_counted>
    bool addPropertiesFrom( const GraphProperty<P, other_counted, Compare>& otherset );

  private:
    MapType m_map;
};


template <typename P, bool counted, class Compare>
bool GraphProperty<P, counted, Compare>::addOrRemoveProperty( const P& property, bool added )
{
    typename MapType::iterator iter = m_map.find( property );
    if( added )
    {
        if( iter == m_map.end() )
        {
            // Insert with count of 1
            m_map.insert( std::make_pair( property, 1 ) );
            return true;
        }
        else
        {
            iter->second++;
            return false;
        }
    }
    else
    {
        RT_ASSERT_MSG( iter != m_map.end(), "Non-existent property removed from graph" );

        if( --iter->second == 0 )
        {
            // Remove from set;
            m_map.erase( iter );
            return true;
        }
        else
        {
            return false;
        }
    }
}

template <typename P, bool counted, class Compare>
P GraphProperty<P, counted, Compare>::front() const
{
    typename MapType::const_iterator iter = m_map.begin();
    RT_ASSERT_MSG( iter != m_map.end(), "GraphProperty back() reference on an empty set" );
    return iter->first;
}

template <typename P, bool counted, class Compare>
P GraphProperty<P, counted, Compare>::back() const
{
    typename MapType::const_iterator iter = m_map.end();
    RT_ASSERT_MSG( iter != m_map.begin(), "GraphProperty back() reference on an empty set" );
    --iter;
    return iter->first;
}

template <typename P, bool counted, class Compare>
template <bool other_counted>
bool GraphProperty<P, counted, Compare>::intersects( const GraphProperty<P, other_counted, Compare>& otherset ) const
{
    // Note: this should be implemented as a joint iteration over both
    // sets. Revisit if necessary.
    for( const auto& prop : m_map )
        if( otherset.contains( prop.first ) )
            return true;
    return false;
}

template <typename P, bool counted, class Compare>
template <bool other_counted>
bool GraphProperty<P, counted, Compare>::addPropertiesFrom( const GraphProperty<P, other_counted, Compare>& otherset )
{
    bool changed = false;
    for( const auto& prop : otherset )
        changed |= addOrRemoveProperty( prop, true );
    return changed;
}


//----------------------------------------------------------------
// GraphProperty (specialization for type 2)
//----------------------------------------------------------------
template <typename P, class Compare>
class GraphProperty<P, false, Compare>
{
  private:
    typedef std::set<P, Compare> SetType;

  public:
    // CTOR / DTOR
    GraphProperty()  = default;
    ~GraphProperty() = default;

    // Iterator type
    typedef typename SetType::const_iterator const_iterator;

    // Standard interface (see above)
    bool addOrRemoveProperty( const P& property, bool adding );
    bool contains( const P& property ) const { return m_set.count( property ) != 0; }
    int count( const P& property ) const { return m_set.count( property ); }
    size_t              size() const { return m_set.size(); }
    bool                empty() const { return m_set.empty(); }
    const_iterator      begin() const { return m_set.begin(); }
    const_iterator      end() const { return m_set.end(); }
    P                   front() const;
    P                   back() const;
    template <bool      other_counted>
    bool intersects( const GraphProperty<P, other_counted, Compare>& otherset ) const;
    template <bool other_counted>
    bool addPropertiesFrom( const GraphProperty<P, other_counted, Compare>& otherset );

  private:
    SetType m_set;
};

template <typename P, class Compare>
bool GraphProperty<P, false, Compare>::addOrRemoveProperty( const P& property, bool added )
{
    typename SetType::iterator iter = m_set.find( property );
    if( added )
    {
        RT_ASSERT_MSG( iter == m_set.end(), "Redundant property added to graph" );
        m_set.insert( property );
        return true;
    }
    else
    {
        RT_ASSERT_MSG( iter != m_set.end(), "Non-existent property removed from graph" );
        // Remove from set;
        m_set.erase( iter );
        return true;
    }
}

template <typename P, class Compare>
P GraphProperty<P, false, Compare>::front() const
{
    typename SetType::const_iterator iter = m_set.begin();
    RT_ASSERT_MSG( iter != m_set.end(), "GraphProperty front() called on an empty set" );
    return *iter;
}

template <typename P, class Compare>
P GraphProperty<P, false, Compare>::back() const
{
    typename SetType::const_iterator iter = m_set.end();
    RT_ASSERT_MSG( iter != m_set.begin(), "GraphProperty back() called on an empty set" );
    --iter;
    return *iter;
}

template <typename P, class Compare>
template <bool other_counted>
bool GraphProperty<P, false, Compare>::intersects( const GraphProperty<P, other_counted, Compare>& otherset ) const
{
    for( const auto& prop : m_set )
        if( otherset.contains( *prop ) )
            return true;
    return false;
}

template <typename P, class Compare>
template <bool other_counted>
bool GraphProperty<P, false, Compare>::addPropertiesFrom( const GraphProperty<P, other_counted, Compare>& otherset )
{
    bool changed = false;
    for( const auto& prop : otherset )
        changed |= m_set.insert( prop ).second;
    return changed;
}


//----------------------------------------------------------------
// GraphPropertySingle (type 3)
//----------------------------------------------------------------
template <typename P = int>
class GraphPropertySingle
{
  private:
    P m_count = 0;

  public:
    // CTOR / DTOR
    GraphPropertySingle()  = default;
    ~GraphPropertySingle() = default;

    // Standard interface (see above)
    bool addOrRemoveProperty( bool adding );
    bool empty() const { return m_count == 0; }
    P    count() const { return m_count; }
};

template <typename P>
bool GraphPropertySingle<P>::addOrRemoveProperty( bool adding )
{
    if( adding )
        return m_count++ == 0;  // was it zero before?

    RT_ASSERT_MSG( !empty(), "Underflow on GraphPropertySingle count" );
    return --m_count == 0;  // is it zero now?
}


//----------------------------------------------------------------
// GraphPropertyMulti (type 4)
//----------------------------------------------------------------

template <typename P, typename S, class CompareP = std::less<P>, class CompareS = std::less<S>>
class GraphPropertyMulti
{
  private:
    typedef GraphProperty<S, true, CompareS>  SubmapType;
    typedef std::map<P, SubmapType, CompareP> MapType;

  public:
    // CTOR / DTOR
    GraphPropertyMulti()  = default;
    ~GraphPropertyMulti() = default;

    // Standard interface( see above)
    bool addOrRemoveProperty( const P& property, const S& subproperty, bool added );
    bool contains( const P& property ) const;
    bool contains( const P& property, const S& subproperty ) const;
    bool empty() const;

    // Return the specific subproperty as a counted GraphProperty
    // set. If the property does not exist, return a reference to an
    // empty set.
    const SubmapType& getSubproperty( const P& property ) const;

  private:
    MapType    m_map;
    SubmapType m_empty;

    // Let NodegraphPrinter see the GraphProperties
    friend class NodegraphPrinter;
};


template <typename P, typename S, class CompareP, class CompareS>
bool GraphPropertyMulti<P, S, CompareP, CompareS>::addOrRemoveProperty( const P& property, const S& subproperty, bool added )
{
    if( added )
    {
        SubmapType& submap = m_map[property];
        return submap.addOrRemoveProperty( subproperty, added );
    }
    else
    {
        typename MapType::iterator iter = m_map.find( property );
        RT_ASSERT_MSG( iter != m_map.end(), "Non-existent property removed from graph" );

        SubmapType& submap  = iter->second;
        bool        changed = submap.addOrRemoveProperty( subproperty, added );
        if( changed && submap.empty() )
            // Cleanup map
            m_map.erase( iter );
        return changed;
    }
}

template <typename P, typename S, class CompareP, class CompareS>
bool GraphPropertyMulti<P, S, CompareP, CompareS>::contains( const P& property ) const
{
    return m_map.count( property ) > 0;
}

template <typename P, typename S, class CompareP, class CompareS>
bool GraphPropertyMulti<P, S, CompareP, CompareS>::contains( const P& property, const S& subproperty ) const
{
    return m_map.count( property ) > 0 && m_map.at( property ).contains( subproperty );
}

template <typename P, typename S, class CompareP, class CompareS>
bool GraphPropertyMulti<P, S, CompareP, CompareS>::empty() const
{
    return m_map.empty();
}

template <typename P, typename S, class CompareP, class CompareS>
const typename GraphPropertyMulti<P, S, CompareP, CompareS>::SubmapType& GraphPropertyMulti<P, S, CompareP, CompareS>::getSubproperty(
    const P& property ) const
{
    typename MapType::const_iterator iter = m_map.find( property );
    if( iter == m_map.end() )
        return m_empty;
    else
        return iter->second;
}

//----------------------------------------------------------------
// Persistent support (as needed)
//----------------------------------------------------------------

template <class T>
void readOrWrite( PersistentStream* stream, GraphProperty<T, false>* props, const char* label )
{
    static const unsigned int* version = getOptixUUID();
    auto                       tmp     = stream->pushObject( label, "GraphProperty" );
    stream->readOrWriteObjectVersion( version );
    if( stream->reading() )
    {
        int n = -1;
        readOrWrite( stream, &n, "size" );
        if( n < 0 )
            stream->setError();
        if( stream->error() )
            return;
        for( int i = 0; i < n; ++i )
        {
            T prop;
            readOrWrite( stream, &prop, "prop" );
            props->addOrRemoveProperty( prop, true );
        }
    }
    else
    {
        int n = props->size();
        readOrWrite( stream, &n, "size" );
        for( auto prop : *props )
            readOrWrite( stream, &prop, "prop" );
    }
}

}  // namespace optix
