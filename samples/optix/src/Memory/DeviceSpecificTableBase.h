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

#include <Memory/MBuffer.h>
#include <Memory/MapMode.h>

#include <array>
#include <stddef.h>
#include <vector>

namespace optix {

class Context;
class LWDADevice;

// A DeviceSpecificTable manages a table that is split between two portions:
//    DeviceIndependentTable contains the portions of the table that are common to all devices.
//    DeviceDependentTable contains the portions of the table that are specific to each device.
//    InterleavedTable represents the combination of both parts.
//
// DeviceSpecificTableBase handles all the allocation logic that is independent of the
// types and works only in sizes of bytes.
//
class DeviceSpecificTableBase
{
  public:
    DeviceSpecificTableBase( Context* context, size_t diSize, size_t diOffset, size_t ddSize, size_t ddOffset, size_t interleavedSize );
    DeviceSpecificTableBase( const DeviceSpecificTableBase& rhs ) = delete;
    DeviceSpecificTableBase& operator=( DeviceSpecificTableBase& rhs ) = delete;
    ~DeviceSpecificTableBase();

    bool resize( size_t size );
    void   sync();
    size_t getTableSizeInBytes() const;
    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }

    void activeDeviceRemoved( unsigned int allDevicesIndex );
    void setActiveDevices();
    bool allocatedToDevice( unsigned int allDeviceListIndex ) const;

    char* mapDeviceIndependentPtr( int dirtyRecordIndexBegin = -1, int dirtyRecordIndexEnd = -1 );
    const char* mapDeviceIndependentPtrReadOnly();
    void        unmapDeviceIndependentPtr();

    char* mapDeviceDependentPtr( unsigned int allDevicesIndex, int dirtyRecordIndexBegin = -1, int dirtyRecordIndexEnd = -1 );
    const char* mapDeviceDependentPtrReadOnly( unsigned int allDevicesIndex );
    void unmapDeviceDependentPtr( unsigned int allDevicesIndex );

    char* getInterleavedDevicePtr( unsigned int allDevicesIndex );
    const std::array<std::pair<int, int>, 4>& getDeviceIndependentTableDirtyIndicesForTest();

    void unmapFromHost( unsigned int allDevicesIndex );

  private:
    enum class TableStatus
    {
        NotDirty,
        NeedsPartialCopy,
        NeedsFullCopy
    };

    Context*                   m_context;
    size_t                     m_diSize;
    size_t                     m_diOffset;
    size_t                     m_ddSize;
    size_t                     m_ddOffset;
    size_t                     m_interleavedSize;
    size_t                     m_size     = 0U;  // expressed as the number of elements in the table, not bytes
    size_t                     m_capacity = 0U;
    MBufferHandle              m_deviceIndependentTable;
    char*                      m_deviceIndependentPtr = nullptr;
    std::vector<MBufferHandle> m_deviceDependentTables;
    std::vector<char*>         m_deviceDependentPtrs;
    std::vector<MBufferHandle> m_deviceInterleavedTables;

    // In order to enable partial syncing of the tables, store up to 4 ranges of dirty record
    // indices. These ranges are then copied individually during sync().  If there are more
    // than 4 non-overlapping dirty ranges of indices, the entire table is copied.
    std::vector<TableStatus> m_deviceDependentTablesStatus;
    std::vector<std::array<std::pair<int, int>, 4>> m_deviceDependentTablesDirtyIndices;
    TableStatus m_deviceIndependentTableStatus = TableStatus::NotDirty;
    std::array<std::pair<int, int>, 4> m_deviceIndependentTableDirtyIndices;

    void markRecordsDirty( TableStatus& tableStatus,
                           std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                           int dirtyRecordIndexBegin,
                           int dirtyRecordIndexEnd );
    void syncTableLwdaDevice( TableStatus tableStatus,
                              std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                              const char* src,
                              char* const dst,
                              size_t      srcPitch );
    void syncTableCpuDevice( TableStatus tableStatus,
                             std::array<std::pair<int, int>, 4>& dirtyRecordIndices,
                             const char* src,
                             char* const dst,
                             size_t      recordSize );

    void syncLwdaDevice( LWDADevice* dstDevice, unsigned int allDevicesIndex, const char* diPtr );
    void syncCpuDevice( unsigned int allDevicesIndex, const char* diPtr );
    bool dirty() const;
    void allocateMBuffers();
    void resizeMBuffers();
    void resizeMBuffer( const MBufferHandle& buffer );
};

}  // namespace optix
