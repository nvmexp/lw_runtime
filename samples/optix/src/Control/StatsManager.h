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

#include <Objects/Variable.h>

#include <corelib/misc/Cast.h>

#include <iosfwd>
#include <map>


namespace optix {

class Context;
class StatsManager;

namespace StatDataReturnType {
template <typename Return>
struct ReturnTypeClass
{
};
template <>
struct ReturnTypeClass<int>
{
    typedef long long ReturnType;
};
template <>
struct ReturnTypeClass<unsigned int>
{
    typedef unsigned long long ReturnType;
};
template <>
struct ReturnTypeClass<unsigned long long>
{
    typedef unsigned long long ReturnType;
};
template <>
struct ReturnTypeClass<float>
{
    typedef float ReturnType;
};
}

struct Stat
{
    enum Kind
    {
        Register,  // Stat is stored in live register
        Local,     // Stat is stored in local memory, then aclwmulated into global with atomic add
        Global,    // Stat is stored in global memory using atomic add
        Shared,    // Stat is stored in shared memory using atomic add, then aclwmulated into global with atomic add
        Elide      // Stat is deleted from the kernel
    };
    VariableType             m_type;        // Float, int, unsigned int
    unsigned int             m_vectorSize;  // 0 for a scalar
    unsigned int             m_ntypes;      // Ray types or entry points
    std::string              m_name;
    std::string              m_description;
    Kind                     m_kind;
    unsigned int             m_offset;
    unsigned int             m_storageOffset;
    unsigned int             m_functionIndex;
    unsigned int             m_order;
    std::vector<std::string> m_perVectorElemDesc;  // extra description to add per vector element when printing

    bool active() const;

    // Returns the element size of the data
    unsigned int getElemSize() const;

    // Computes how much space the stat will take up.
    unsigned int getSize() const;

    // This is a helper class to get access to the Stat's output.  You will need
    // to create the class based on the type in m_type.  Note that lwrrently the
    // data is copied in the constructor.
    template <typename T>
    struct Data
    {
        Data( const Stat* stat, const std::vector<char*>& in_data )
            : m_ntypes( stat->m_ntypes )
            , m_stat( stat )
        {
            m_numDevices = corelib::range_cast<unsigned int>( in_data.size() );
            // Stat::m_vectorSize can be 0 or a useful number, we just need it for
            // indexing, so make it 1 or the number.  If you want to know if it was
            // zero, look in m_stat again.
            m_vectorSize             = m_stat->m_vectorSize ? m_stat->m_vectorSize : 1;
            size_t byteSize          = m_stat->getSize();
            size_t numElemsPerDevice = byteSize / m_stat->getElemSize();
            m_data.resize( numElemsPerDevice * in_data.size() );
            for( size_t d = 0; d < in_data.size(); ++d )
            {
                T* out = &( m_data[d * numElemsPerDevice] );
                T* in  = reinterpret_cast<T*>( in_data[d] + m_stat->m_storageOffset );
                for( size_t i = 0; i < numElemsPerDevice; ++i )
                    out[i]    = in[i];
            }
        }
        unsigned int   m_numDevices;
        unsigned int   m_vectorSize;  // This differs from Stat::m_vectorSize in that it will be >= 1.
        unsigned int   m_ntypes;
        const Stat*    m_stat;
        std::vector<T> m_data;

        typedef typename StatDataReturnType::ReturnTypeClass<T>::ReturnType ReturnType;

        // Give the range over which you wish to compute a reduction.
        ReturnType getValueRange( unsigned int d_start, unsigned int d_end, unsigned int v_start, unsigned int v_end, unsigned int t_start, unsigned int t_end )
        {
            RT_ASSERT( d_start < m_numDevices );
            RT_ASSERT( d_end <= m_numDevices );
            RT_ASSERT( v_start < m_vectorSize );
            RT_ASSERT( v_end <= m_vectorSize );
            RT_ASSERT( t_start < m_ntypes );
            RT_ASSERT( t_end <= m_ntypes );

            ReturnType   result  = 0;
            unsigned int stride1 = m_ntypes;
            unsigned int stride2 = m_ntypes * m_vectorSize;
            for( unsigned int d = d_start; d < d_end; ++d )
            {
                for( unsigned int v = v_start; v < v_end; ++v )
                    for( unsigned int t = t_start; t < t_end; ++t )
                        result += m_data[d * stride2 + v * stride1 + t];
            }
            return result;
        }

        // Returns the value at a given location.
        ReturnType getValueIndex( unsigned int device, unsigned int vector, unsigned int type )
        {
            RT_ASSERT( device < m_numDevices );
            RT_ASSERT( vector < m_vectorSize );
            RT_ASSERT( type < m_ntypes );

            unsigned int stride1 = m_ntypes;
            unsigned int stride2 = m_ntypes * m_vectorSize;
            return m_data[device * stride2 + vector * stride1 + type];
        }

        // This will return the value for a given vector index and type, but
        // reduces accross devices.
        ReturnType getValueIndexAllDevices( unsigned int v, unsigned int t )
        {
            return getValueRange( 0, m_numDevices, v, v + 1, t, t + 1 );
        }

        // Computes the sum of all values.
        ReturnType getValueAll()
        {
            ReturnType result = 0;
            for( size_t i = 0; i < m_data.size(); ++i )
                result += m_data[i];
            return result;
        }

        // Returns the last vector index across all devices for a given type that is non-zero.
        unsigned int getLastNonZeroByType( unsigned int type )
        {
            RT_ASSERT( type < m_ntypes );

            if( m_stat->m_vectorSize == 0 )
                return 1;
            unsigned int stride1 = m_ntypes;
            unsigned int stride2 = m_ntypes * m_vectorSize;
            for( unsigned int d = 0; d < m_numDevices; ++d )
            {
                for( unsigned int v = m_vectorSize - 1; v > 0; --v )
                    if( m_data[d * stride2 + v * stride1 + type] != 0 )
                        return v + 1;
            }
            return 1;
        }
    };
};

class StatsManager
{
  public:
    StatsManager();
    ~StatsManager();

    void reset();

    const Stat* lookup( const std::string& statname ) const;
    void registerStat( const VariableType& type,
                       const std::string&  statname,
                       const std::string&  desc,
                       unsigned int        ntypes,
                       unsigned int        vectorSize,
                       unsigned int        functionIndex,
                       unsigned int        order );
    void assignSlots( Context* context );
    // The stat needs to be registered, before calling this.
    void addPerVectorElemDesc( const std::string& statname, const std::vector<std::string>& perVectorElemDesc );

    typedef std::map<std::string, Stat> MapType;
    const MapType& getMap() const { return m_lwrrentStats; }

    unsigned int getRegisterCount() const { return m_registerCount; }
    unsigned int getLocalSize() const { return m_localSize; }
    unsigned int getGlobalSize() const { return m_globalSize; }
    unsigned int getSharedSize() const { return m_sharedSize; }

    void print( std::ostream& out, const std::vector<char*>& data_ptrs ) const;

  private:
    MapType      m_lwrrentStats;
    unsigned int m_registerCount;
    unsigned int m_localSize;
    unsigned int m_globalSize;
    unsigned int m_sharedSize;

    enum State
    {
        Open,
        Closed
    };
    State m_state;
};
}
