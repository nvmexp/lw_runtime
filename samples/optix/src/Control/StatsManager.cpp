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


#include <Control/StatsManager.h>

#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <iomanip>
#include <iostream>

using namespace optix;
using namespace prodlib;


unsigned int Stat::getElemSize() const
{
    RT_ASSERT_FAIL_MSG( "Stats: missing implementation" );
#if 0
  unsigned int s = 0;
  switch(m_type){
  case VTYPE_INTERNAL_STATISTIC_INT: s= PTXTypes::lwdaTypeSize<int>(); break;
  case VTYPE_INTERNAL_STATISTIC_UINT: s= PTXTypes::lwdaTypeSize<unsigned int>(); break;
  case VTYPE_INTERNAL_STATISTIC_UINT64: s= PTXTypes::lwdaTypeSize<unsigned long long>(); break;
  case VTYPE_INTERNAL_STATISTIC_FLOAT: s= PTXTypes::lwdaTypeSize<float>(); break;
  default:
    RT_ASSERT(!!!"Unknown stat type");
  }
  return s;
#endif
}

unsigned int Stat::getSize() const
{
    unsigned int s = getElemSize();
    if( m_vectorSize != 0 )
        s *= m_vectorSize;
    s *= m_ntypes;
    return s;
}

bool Stat::active() const
{
    return m_kind != Stat::Elide;
}

StatsManager::StatsManager()
{
    reset();
}

StatsManager::~StatsManager()
{
}

void StatsManager::reset()
{
    m_state         = Open;
    m_registerCount = 0;
    m_localSize     = 0;
    m_globalSize    = 0;
    m_sharedSize    = 0;
    m_lwrrentStats.clear();
}

void StatsManager::assignSlots( Context* context )
{
    RT_ASSERT_FAIL_MSG( "Stats: missing implementation" );
#if 0
  RT_ASSERT(m_state == Open);

  for( MapType::iterator iter = m_lwrrentStats.begin(); iter != m_lwrrentStats.end(); ++iter ) {
    Stat& stat = iter->second;
    std::ostringstream knobname;
    knobname << "dynamic.stats." << stat.m_name;
    DynamicKnob<std::string> knob( knobname.str().c_str(), "" );
    std::string value = knob.get();
    if( value.empty() ) {
      stat.m_kind = Stat::Elide;
    } else {
      if(value[value.size()-1] == '*'){
        value = value.substr(0, value.size()-1);
      } else {
        stat.m_ntypes = 1;
      }
      if( value == "Register" || value == "register" ) {
        RT_ASSERT( stat.m_vectorSize == 0 );
        RT_ASSERT( stat.m_ntypes == 1 );
        stat.m_kind = Stat::Register;
      } else if( value == "Local" || value == "local" ) {
        stat.m_kind = Stat::Local;
      } else if( value == "Global" || value == "global" ) {
        stat.m_kind = Stat::Global;
      } else if( value == "Shared" || value == "shared" ) {
        stat.m_kind = Stat::Shared;
      } else if( value == "Elide" || value == "elide" ) {
        stat.m_kind = Stat::Elide;
      } else {
        throw IlwalidValue( RT_EXCEPTION_INFO, "Unknown statistic property: ", value );
      }
      // If you ever reenable Shared 64 bit stats, you will need to fix up the reduction code in
      // Compile::finalizeStatistics().  That code lwrrently only supports 32 bit numbers.  You will also have to ensure
      // alignment in the shared stats assignment below.
      if (stat.m_type == VTYPE_INTERNAL_STATISTIC_UINT64 && stat.m_kind == Stat::Shared) {
        RT_ASSERT(!!!"unsigned 64 bit stats cannot be Shared");
      }
    }
  }

  // Assign offsets - shared first
  for( MapType::iterator iter = m_lwrrentStats.begin(); iter != m_lwrrentStats.end(); ++iter ) {
    Stat& stat = iter->second;
    unsigned int size = stat.getSize();
    if(stat.m_kind == Stat::Shared){
      stat.m_storageOffset = m_globalSize;
      stat.m_offset = m_sharedSize;
      m_globalSize += size;
      m_sharedSize += size;
    }
  }
  for( MapType::iterator iter = m_lwrrentStats.begin(); iter != m_lwrrentStats.end(); ++iter ) {
    Stat& stat = iter->second;
    unsigned int size = stat.getSize();
    if(stat.m_kind != Stat::Shared && stat.m_kind != Stat::Elide){
      // Make sure that m_globalSize are aligned with the size of the stat.
      unsigned int offset = m_globalSize % stat.getElemSize();
      m_globalSize += offset ? ( stat.getElemSize() - offset) : 0;
      stat.m_storageOffset = m_globalSize;
      m_globalSize += size;
    }
    switch(stat.m_kind){
    case Stat::Register: stat.m_offset = m_registerCount; m_registerCount ++;   break;
    case Stat::Local:
      {
        // Make sure that m_localSize are aligned with the size of the stat.
        unsigned int offset = m_localSize % stat.getElemSize();
        m_localSize += offset ? ( stat.getElemSize() - offset) : 0;
        stat.m_offset = m_localSize;
        m_localSize += size;
      }
      break;
    case Stat::Global:   stat.m_offset = stat.m_storageOffset;                  break;
    case Stat::Shared:   break; // Assigned above
    case Stat::Elide:    stat.m_offset = ~0U;                                   break;
    }
  }
  m_state = Closed;
#endif
}

void StatsManager::addPerVectorElemDesc( const std::string& statname, const std::vector<std::string>& perVectorElemDesc )
{
    RT_ASSERT( m_state == Closed );
    std::map<std::string, Stat>::iterator iter = m_lwrrentStats.find( statname );
    // You shouldn't be doing this to a stat you haven't registered.
    RT_ASSERT( iter != m_lwrrentStats.end() );
    ( &iter->second )->m_perVectorElemDesc = perVectorElemDesc;
}

const Stat* StatsManager::lookup( const std::string& statname ) const
{
    RT_ASSERT( m_state == Closed );
    std::map<std::string, Stat>::const_iterator iter = m_lwrrentStats.find( statname );
    if( iter == m_lwrrentStats.end() )
        return nullptr;
    else
        return &iter->second;
}

void StatsManager::registerStat( const VariableType& type,
                                 const std::string&  statname,
                                 const std::string&  desc,
                                 unsigned int        ntypes,
                                 unsigned int        vectorSize,
                                 unsigned int        functionIndex,
                                 unsigned int        order )
{
    RT_ASSERT( m_state == Open );
    std::map<std::string, Stat>::iterator iter = m_lwrrentStats.find( statname );
    if( iter != m_lwrrentStats.end() )
    {
        Stat& existing = iter->second;
        if( desc != existing.m_description )
        {
            if( existing.m_description.empty() && !desc.empty() )
            {
                existing.m_description = desc;
            }
            else if( !desc.empty() )
            {
                lwarn << "WARNING: Statistic '" << statname << "' declared with different descriptions '" << desc
                      << "' and '" << existing.m_description << '\n';
            }
        }

        if( vectorSize == 0 )
            RT_ASSERT( existing.m_vectorSize == 0 );
        else if( vectorSize > existing.m_vectorSize )
            existing.m_vectorSize = vectorSize;
        RT_ASSERT( existing.m_type == type );
        //RT_ASSERT(existing.m_ntypes == ntypes);
        if( existing.m_ntypes != ntypes )
            existing.m_ntypes = 1;
    }
    else
    {
        Stat stat;
        stat.m_type          = type;
        stat.m_vectorSize    = vectorSize;
        stat.m_ntypes        = ntypes;
        stat.m_name          = statname;
        stat.m_description   = desc;
        stat.m_offset        = ~0U;
        stat.m_storageOffset = ~0U;
        stat.m_functionIndex = functionIndex;
        stat.m_order         = order;
        stat.m_kind          = Stat::Elide;
        m_lwrrentStats.insert( std::make_pair( stat.m_name, stat ) );
    }
}

std::string getName( const Stat& stat, unsigned int type, unsigned int index )
{
    std::ostringstream out;
    out << stat.m_name;
    if( stat.m_ntypes != 1 )
        out << "[" << type << "]";
    if( stat.m_vectorSize )
        out << "[" << index << "]";
    return out.str();
}

namespace {
struct SortOrder
{
    bool operator()( const Stat* lhs, const Stat* rhs ) const
    {
        return lhs->m_functionIndex < rhs->m_functionIndex
               || ( lhs->m_functionIndex == rhs->m_functionIndex && lhs->m_order < rhs->m_order );
    }
};
}

template <typename T>
static std::string getValue( const Stat& stat, const std::vector<char*>& data_ptr )
{
    Stat::Data<T>      data( &stat, data_ptr );
    std::ostringstream out;
    out << data.getValueAll();
    return out.str();
}

template <typename T>
static void printStatValues( std::ostream& out, size_t maxv, size_t maxw, const Stat& stat, const std::vector<char*>& data_ptr )
{
    Stat::Data<T> data( &stat, data_ptr );
    for( unsigned int type = 0; type < stat.m_ntypes; type++ )
    {
        unsigned int n = data.getLastNonZeroByType( type );
        for( unsigned int i = 0; i < n; ++i )
        {
            // Loop over all the vector elements
            std::string perVectorDesc;
            if( i < stat.m_perVectorElemDesc.size() )
                perVectorDesc = stat.m_perVectorElemDesc[i];
            out << std::setw( maxv + 1 ) << std::right << data.getValueIndexAllDevices( i, type ) << " - " << std::setw( maxw )
                << std::left << getName( stat, type, i ) << " - " << stat.m_description << " - " << perVectorDesc << "\n";
        }
        if( stat.m_vectorSize != 0 )
        {
            std::ostringstream star;
            star << stat.m_name;
            if( stat.m_ntypes != 1 )
                star << "[" << type << "]";
            star << "[*]";
            out << std::setw( maxv + 1 ) << std::right << data.getValueRange( 0, data.m_numDevices, 0, n, type, type + 1 )
                << " - " << std::setw( maxw ) << std::left << star.str() << '\n';
        }
    }
    if( stat.m_ntypes != 1 )
    {
        std::ostringstream star;
        star << stat.m_name;
        star << "[*]";
        out << std::setw( maxv + 1 ) << std::right << data.getValueAll() << " - " << std::setw( maxw ) << std::left
            << star.str() << '\n';
    }
}

void StatsManager::print( std::ostream& out, const std::vector<char*>& data_ptr ) const
{
    RT_ASSERT_FAIL_MSG( "Stats: missing implementation" );
#if 0
  RT_ASSERT(m_state == Closed);
  // Print statistics
  std::vector<const Stat*> sorted;
  sorted.reserve( m_lwrrentStats.size() );
  for( StatsManager::MapType::const_iterator iter = m_lwrrentStats.begin(); iter != m_lwrrentStats.end(); ++iter )
    sorted.push_back( &iter->second );
  algorithm::sort( sorted, SortOrder() );

  // Determine the maximum string length of the stat names and values.  This helps make things line up pretty.
  size_t maxv = 0;
  size_t maxw = 3; // for [*]
  for( std::vector<const Stat*>::iterator iter = sorted.begin(); iter != sorted.end(); ++iter ) {
    const Stat& stat = **iter;
    if( stat.m_kind != Stat::Elide ) {
      size_t w = getName( stat, stat.m_ntypes-1, stat.m_vectorSize-1 ).length();
      if( w > maxw )
        maxw = w;
      size_t v;
      switch( stat.m_type ) {
      case VTYPE_INTERNAL_STATISTIC_INT:
        v = getValue<int>(stat, data_ptr).length();
        break;
      case VTYPE_INTERNAL_STATISTIC_UINT:
        v = getValue<unsigned int>(stat, data_ptr).length();
        break;
      case VTYPE_INTERNAL_STATISTIC_UINT64:
        v = getValue<unsigned long long>(stat, data_ptr).length();
        break;
      case VTYPE_INTERNAL_STATISTIC_FLOAT:
        v = getValue<float>(stat, data_ptr).length();
        break;
      default:
        RT_ASSERT(!!!"Unknown stat type");
      }
      if( v > maxv )
        maxv = v;
    }
  }

  for( std::vector<const Stat*>::iterator iter = sorted.begin(); iter != sorted.end(); ++iter ) {
    const Stat& stat = **iter;
    if( stat.m_kind != Stat::Elide ) {
      switch( stat.m_type ) {
      case VTYPE_INTERNAL_STATISTIC_INT:
        printStatValues<int>(out, maxv, maxw, stat, data_ptr);
        break;
      case VTYPE_INTERNAL_STATISTIC_UINT:
        printStatValues<unsigned int>(out, maxv, maxw, stat, data_ptr);
        break;
      case VTYPE_INTERNAL_STATISTIC_UINT64:
        printStatValues<unsigned long long>(out, maxv, maxw, stat, data_ptr);
        break;
      case VTYPE_INTERNAL_STATISTIC_FLOAT:
        printStatValues<float>(out, maxv, maxw, stat, data_ptr);
        break;
      default:
        RT_ASSERT(!!!"Unknown stat type");
      }
    }
  }
#endif
}
