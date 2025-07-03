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

#include <prodlib/exceptions/IlwalidValue.h>

#include <vector>


namespace optix {

namespace IdPoolTraits {

template <typename T>
struct IntegralTraits;
template <>
struct IntegralTraits<int>
{
    static bool lessThanZero( int val ) { return val < 0; }
};
template <>
struct IntegralTraits<unsigned>
{
    static bool lessThanZero( unsigned val ) { return false; }
};

}  // namespace

template <typename Index = int>
class IdPool
{
  public:
    IdPool()
        : m_maxID( 0 )
    {
    }

    Index get()
    {
        Index ID = m_maxID;
        if( m_freeIDs.size() )
        {
            ID = m_freeIDs.back();
            m_freeIDs.pop_back();
        }
        else
            m_maxID++;

        return ID;
    }


    // No checking for duplicates. Throws IlwalidValue if the ID is out of range of allocated IDs or negative.
    void free( Index ID )
    {
        if( optix::IdPoolTraits::IntegralTraits<Index>::lessThanZero( ID ) || ID >= m_maxID )
            throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "ID out of range" );
        else
            m_freeIDs.push_back( ID );
    }

  private:
    Index              m_maxID;
    std::vector<Index> m_freeIDs;
};

}  // namespace optix
