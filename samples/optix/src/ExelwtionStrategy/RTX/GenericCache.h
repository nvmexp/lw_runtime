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

#include <map>
#include <memory>
#include <prodlib/exceptions/Assert.h>

namespace optix {

template <typename Key, typename Elt>
class GenericCache
{
  public:
    std::shared_ptr<Elt> getCachedElement( const Key& key ) const;
    void addCachedElement( const Key& key, const std::shared_ptr<Elt>& elt );

  private:
    std::map<Key, std::shared_ptr<Elt>> m_cache;
};

template <typename Key, typename Elt>
std::shared_ptr<Elt> GenericCache<Key, Elt>::getCachedElement( const Key& key ) const
{
    auto iter = m_cache.find( key );
    return iter == m_cache.end() ? std::shared_ptr<Elt>() : iter->second;
}

template <typename Key, typename Elt>
void GenericCache<Key, Elt>::addCachedElement( const Key& key, const std::shared_ptr<Elt>& elt )
{
    bool inserted = m_cache.insert( std::make_pair( key, elt ) ).second;
    RT_ASSERT_MSG( inserted, "Cache element already present" );
}
}
