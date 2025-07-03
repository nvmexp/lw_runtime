/*
* Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from LWPU Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#pragma once

#include <Memory/DeviceSpecificTableBase.h>

namespace optix {

class Context;

// A DeviceSpecificTable manages a table that is split between two portions:
//    DeviceIndependentTable contains the portions of the table that are common to all devices.
//    DeviceDependentTable contains the portions of the table that are specific to each device.
//    InterleavedTable represents the combination of both parts.
//
// InterleavedTable should have device independent data first, followed by device dependent data,
//  e.g. be a struct that looks like this:
//  struct InterleavedTable
//  {
//    DeviceIndependentTable di;
//    DeviceDependentTable dd;
//  };
//
// DeviceSpecificTable manages the MBuffers on the host and on the device and synchronizes
// any modified data on the host to the devices during a call to sync().
//
// The template class provides syntactic sugar for the three types representing the three
// views of the table data.
//
template <typename InterleavedTable,
          typename DeviceIndependentTable    = typename InterleavedTable::DeviceIndependent,
          typename DeviceDependentTable      = typename InterleavedTable::DeviceDependent,
          std::size_t Padding                = 0,
          bool        DeviceIndependentFirst = true>
class DeviceSpecificTable : public DeviceSpecificTableBase
{
  public:
    DeviceSpecificTable( Context* context )
        : DeviceSpecificTableBase( context,
                                   sizeof( DeviceIndependentTable ),
                                   DeviceIndependentFirst ? 0U : sizeof( DeviceDependentTable ),
                                   sizeof( DeviceDependentTable ),
                                   DeviceIndependentFirst ? sizeof( DeviceIndependentTable ) + Padding : 0U,
                                   sizeof( InterleavedTable ) )
    {
        static_assert( sizeof( DeviceIndependentTable ) + sizeof( DeviceDependentTable ) + Padding == sizeof( InterleavedTable ),
                       "Interleaved structure doesn't match combination of device independent and device dependent "
                       "structure" );
    }
    DeviceSpecificTable( const DeviceSpecificTable& rhs ) = delete;
    DeviceSpecificTable& operator=( DeviceSpecificTable& rhs ) = delete;

    DeviceIndependentTable* mapDeviceIndependentTable( int dirtyRecordIndexBegin = -1, int dirtyRecordIndexEnd = -1 )
    {
        return reinterpret_cast<DeviceIndependentTable*>( mapDeviceIndependentPtr( dirtyRecordIndexBegin, dirtyRecordIndexEnd ) );
    }
    const DeviceIndependentTable* mapDeviceIndependentTableReadOnly()
    {
        return reinterpret_cast<const DeviceIndependentTable*>( mapDeviceIndependentPtrReadOnly() );
    }
    void unmapDeviceIndependentTable() { unmapDeviceIndependentPtr(); }

    DeviceDependentTable* mapDeviceDependentTable( const unsigned int allDevicesIndex,
                                                   int                dirtyRecordIndexBegin = -1,
                                                   int                dirtyRecordIndexEnd   = -1 )
    {
        return reinterpret_cast<DeviceDependentTable*>(
            mapDeviceDependentPtr( allDevicesIndex, dirtyRecordIndexBegin, dirtyRecordIndexEnd ) );
    }
    const DeviceDependentTable* mapDeviceDependentTableReadOnly( const unsigned int allDevicesIndex )
    {
        return reinterpret_cast<const DeviceDependentTable*>( mapDeviceDependentPtrReadOnly( allDevicesIndex ) );
    }
    void unmapDeviceDependentTable( const unsigned int allDevicesIndex ) { unmapDeviceDependentPtr( allDevicesIndex ); }

    InterleavedTable* getInterleavedTableDevicePtr( const unsigned int allDevicesIndex )
    {
        return reinterpret_cast<InterleavedTable*>( getInterleavedDevicePtr( allDevicesIndex ) );
    }

    // RAII helper class for ensuring that all calls to map match a call to unmap for the device independent data
    class DiTableLock
    {
        using Table = DeviceSpecificTable<InterleavedTable, DeviceIndependentTable, DeviceDependentTable, Padding, DeviceIndependentFirst>;

      public:
        DiTableLock( Table& table )
            : m_table( table )
            , m_data( table.mapDeviceIndependentTable() )
        {
        }
        ~DiTableLock() { m_table.unmapDeviceIndependentTable(); }

        operator DeviceIndependentTable*() { return m_data; }
        DeviceIndependentTable* getData() { return m_data; }

      private:
        Table&                  m_table;
        DeviceIndependentTable* m_data;
    };

    // RAII helper class for ensuring that all calls to map match a call to unmap for the device dependent data
    class DdTableLock
    {
        using Table = DeviceSpecificTable<InterleavedTable, DeviceIndependentTable, DeviceDependentTable, Padding, DeviceIndependentFirst>;

      public:
        DdTableLock( Table& table, unsigned int allDevicesIndex )
            : m_table( table )
            , m_allDevicesIndex( allDevicesIndex )
            , m_data( table.mapDeviceDependentTable( allDevicesIndex ) )
        {
        }
        ~DdTableLock() { m_table.unmapDeviceDependentTable( m_allDevicesIndex ); }

        operator DeviceDependentTable*() { return m_data; }
        DeviceDependentTable* getData() { return m_data; }

      private:
        Table&                m_table;
        unsigned int          m_allDevicesIndex;
        DeviceDependentTable* m_data;
    };

  private:
    // when using the typesafe class, hide these char* pointer functions
    using DeviceSpecificTableBase::getInterleavedDevicePtr;
    using DeviceSpecificTableBase::mapDeviceDependentPtr;
    using DeviceSpecificTableBase::mapDeviceIndependentPtr;
    using DeviceSpecificTableBase::unmapDeviceDependentPtr;
    using DeviceSpecificTableBase::unmapDeviceIndependentPtr;
};

}  // namespace optix
